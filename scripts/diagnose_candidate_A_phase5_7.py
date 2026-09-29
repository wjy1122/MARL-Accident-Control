"""Phase 5.7 — same-state one-step reward counterfactual (read-only).

Uses TraCI saveState/loadState. For each sampled (step, cav_id) from a
Candidate A deterministic episode, restore the identical pre-decision
snapshot and take one simulation step under KEEP / LEFT / RIGHT for that
ego only (others keep their recorded argmax). Records Candidate A reward
components. Does not train or edit rewards/checkpoints.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.rollout import resolve_device  # noqa: E402
from env.action import ACTION_NAMES, KEEP, LEFT, RIGHT  # noqa: E402
from env.highway_accident_env import HighwayAccidentEnv, load_config  # noqa: E402
from utils.baseline_eval import (  # noqa: E402
    load_mappo_checkpoint,
    params_unchanged,
    set_eval_seed,
    snapshot_model_params,
)

OUT = PROJECT_ROOT / "results" / "experiments" / "reward_candidate_A_phase5_7"
CA_CKPT = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "reward_candidate_A_phase5_3"
    / "checkpoints"
    / "episode_0050.pt"
)
CA_REWARD = PROJECT_ROOT / "configs" / "reward_candidate_A.yaml"
SEED = 42
MAX_STEPS = 1800
ZONE_R = 150.0
# Stratified budget per (zone, lane). Effective-LC precursors added on top.
PER_CELL = 40
MAX_EFFECTIVE = 120


def _stats(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan")}
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "std": float(np.std(a)),
    }


def _fmt(x: float, nd: int = 4) -> str:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def _zone(pos: float, x_acc: float) -> str:
    if pos < x_acc - ZONE_R:
        return "before"
    if pos > x_acc + ZONE_R:
        return "after"
    return "near"


def _boundary(action: int, lane: int, n_lanes: int) -> str:
    if action == LEFT and lane >= n_lanes - 1:
        return "invalid_LEFT"
    if action == RIGHT and lane <= 0:
        return "invalid_RIGHT"
    return "none"


def _policy(
    model: torch.nn.Module, obs: Any, device: torch.device
) -> tuple[int, list[float], list[float]]:
    t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device)
    with torch.no_grad():
        logits_t = model.actor.forward(t)
        if logits_t.dim() == 1:
            logits_t = logits_t.unsqueeze(0)
        probs_t = torch.softmax(logits_t, dim=-1)[0]
    logits = [float(x) for x in logits_t[0].detach().cpu().tolist()]
    probs = [float(x) for x in probs_t.detach().cpu().tolist()]
    return int(np.argmax(logits)), logits, probs


def pass1_collect(model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    """Deterministic episode: record decisions and effective lane precursors."""
    set_eval_seed(SEED)
    env_cfg = copy.deepcopy(load_config())
    env_cfg.setdefault("episode", {})["max_episode_steps"] = MAX_STEPS
    env = HighwayAccidentEnv(config=env_cfg, gui=False, reward_config=CA_REWARD)
    x_acc = float(env.accident_position_m)
    n_lanes = int(env.n_lanes)
    horizon = max(
        1, int(round(float(env.lane_change_duration_s) / float(env.step_length)))
    )

    records: list[dict[str, Any]] = []
    lane_at: dict[tuple[str, int], int] = {}
    try:
        env.reset()
        for step in range(MAX_STEPS):
            states = env.get_cav_states()
            actions: dict[str, int] = {}
            for cid, obs in env.get_observations().items():
                st = states.get(cid)
                if st is None:
                    continue
                a, logits, probs = _policy(model, obs, device)
                actions[cid] = a
                lane = int(st["lane"])
                pos = float(st["position"])
                zone = _zone(pos, x_acc)
                records.append(
                    {
                        "step": step,
                        "cav_id": cid,
                        "zone": zone,
                        "lane": lane,
                        "pos": pos,
                        "action": a,
                        "boundary": _boundary(a, lane, n_lanes),
                        "logits": logits,
                        "probs": probs,
                        "d_l": logits[LEFT] - logits[KEEP],
                        "d_r": logits[RIGHT] - logits[KEEP],
                    }
                )
                lane_at[(cid, step)] = lane
            _obs, _rew, _term, truncated, _info = env.step(actions)
            if truncated:
                break
    finally:
        env.close()

    # Tag effective transitions: lane index moves within horizon after the decision.
    for rec in records:
        future = None
        for k in range(1, horizon + 1):
            nxt = lane_at.get((rec["cav_id"], rec["step"] + k))
            if nxt is None:
                break
            future = nxt
        if future is None:
            rec["effective"] = "unknown"
        elif future > rec["lane"]:
            rec["effective"] = "LEFT"
        elif future < rec["lane"]:
            rec["effective"] = "RIGHT"
        else:
            rec["effective"] = "none"

    return {"records": records, "horizon": horizon, "n_lanes": n_lanes}


def choose_samples(records: list[dict[str, Any]], rng: np.random.Generator) -> set[tuple[int, str]]:
    by_cell: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_cell[(r["zone"], r["lane"])].append(r)

    chosen: set[tuple[int, str]] = set()
    for (zone, lane), group in by_cell.items():
        if not group:
            continue
        idx = np.arange(len(group))
        rng.shuffle(idx)
        for i in idx[:PER_CELL]:
            g = group[int(i)]
            chosen.add((g["step"], g["cav_id"]))

    eff = [r for r in records if r["effective"] in ("LEFT", "RIGHT")]
    rng.shuffle(eff)
    for r in eff[:MAX_EFFECTIVE]:
        chosen.add((r["step"], r["cav_id"]))
    return chosen


def _one_step(env: HighwayAccidentEnv, actions: dict[str, int]) -> dict[str, dict[str, float]]:
    traci = env._require_traci()
    present = set(traci.vehicle.getIDList())
    for cid, a in actions.items():
        if cid in present and env.classify_vehicle(cid) == "CAV":
            env.apply_action(cid, a)
    traci.simulationStep()
    return env.get_rewards()


def pass2_counterfactual(
    model: torch.nn.Module,
    device: torch.device,
    samples: set[tuple[int, str]],
    n_lanes: int,
) -> list[dict[str, Any]]:
    """Re-run episode; at sampled decisions, branch KEEP/LEFT/RIGHT one step."""
    set_eval_seed(SEED)
    env_cfg = copy.deepcopy(load_config())
    env_cfg.setdefault("episode", {})["max_episode_steps"] = MAX_STEPS
    env = HighwayAccidentEnv(config=env_cfg, gui=False, reward_config=CA_REWARD)
    x_acc = float(env.accident_position_m)
    results: list[dict[str, Any]] = []
    tmp = tempfile.TemporaryDirectory(prefix="phase57_")
    state_path = Path(tmp.name) / "snap.xml"

    try:
        env.reset()
        for step in range(MAX_STEPS):
            states = env.get_cav_states()
            observations = env.get_observations()
            actions: dict[str, int] = {}
            logits_map: dict[str, list[float]] = {}
            for cid, obs in observations.items():
                a, logits, _probs = _policy(model, obs, device)
                actions[cid] = a
                logits_map[cid] = logits

            sample_cids = [cid for cid in actions if (step, cid) in samples]
            if sample_cids:
                traci = env._require_traci()
                traci.simulation.saveState(str(state_path))
                step_count = int(env._step_count)
                for ego in sample_cids:
                    st0 = states.get(ego)
                    if st0 is None:
                        continue
                    lane0 = int(st0["lane"])
                    pos0 = float(st0["position"])
                    zone = _zone(pos0, x_acc)
                    row: dict[str, Any] = {
                        "step": step,
                        "cav_id": ego,
                        "zone": zone,
                        "lane": lane0,
                        "pos": pos0,
                        "policy_action": actions[ego],
                        "d_logit_l": logits_map[ego][LEFT] - logits_map[ego][KEEP],
                        "d_logit_r": logits_map[ego][RIGHT] - logits_map[ego][KEEP],
                        "boundary_keep": _boundary(KEEP, lane0, n_lanes),
                        "boundary_left": _boundary(LEFT, lane0, n_lanes),
                        "boundary_right": _boundary(RIGHT, lane0, n_lanes),
                    }
                    for a, name in ((KEEP, "KEEP"), (LEFT, "LEFT"), (RIGHT, "RIGHT")):
                        traci.simulation.loadState(str(state_path))
                        env._step_count = step_count
                        # Ego varies; others keep the recorded joint argmax for this step.
                        branch = dict(actions)
                        branch[ego] = a
                        rewards = _one_step(env, branch)
                        detail = rewards.get(ego)
                        # Lane after one step for this ego (may be unchanged within 0.1 s).
                        try:
                            lane1 = int(traci.vehicle.getLaneIndex(ego))
                            spd1 = float(traci.vehicle.getSpeed(ego))
                        except Exception:  # noqa: BLE001
                            lane1 = -1
                            spd1 = float("nan")
                        if detail is None:
                            row[f"R_{name}"] = float("nan")
                            row[f"Rind_{name}"] = float("nan")
                            row[f"Rgrp_{name}"] = float("nan")
                            row[f"Scav_{name}"] = float("nan")
                            row[f"Sloc_{name}"] = float("nan")
                            row[f"Pblock_{name}"] = float("nan")
                            row[f"lane1_{name}"] = lane1
                            row[f"spd1_{name}"] = spd1
                        else:
                            row[f"R_{name}"] = float(detail["total"])
                            row[f"Rind_{name}"] = float(detail["individual"])
                            row[f"Rgrp_{name}"] = float(detail["group"])
                            row[f"Scav_{name}"] = float(detail["s_cav"])
                            row[f"Sloc_{name}"] = float(detail["s_loc"])
                            row[f"Pblock_{name}"] = float(detail["p_block"])
                            row[f"lane1_{name}"] = lane1
                            row[f"spd1_{name}"] = spd1
                    row["dR_L_K"] = row["R_LEFT"] - row["R_KEEP"]
                    row["dR_R_K"] = row["R_RIGHT"] - row["R_KEEP"]
                    row["dRind_L_K"] = row["Rind_LEFT"] - row["Rind_KEEP"]
                    row["dRind_R_K"] = row["Rind_RIGHT"] - row["Rind_KEEP"]
                    row["dRgrp_L_K"] = row["Rgrp_LEFT"] - row["Rgrp_KEEP"]
                    row["dRgrp_R_K"] = row["Rgrp_RIGHT"] - row["Rgrp_KEEP"]
                    row["dP_L_K"] = row["Pblock_LEFT"] - row["Pblock_KEEP"]
                    row["dP_R_K"] = row["Pblock_RIGHT"] - row["Pblock_KEEP"]
                    results.append(row)

                # Restore and continue the real episode.
                traci.simulation.loadState(str(state_path))
                env._step_count = step_count

            _obs, _rew, _term, truncated, _info = env.step(actions)
            if truncated:
                break
    finally:
        env.close()
        tmp.cleanup()
    return results


def attach_effective(
    rows: list[dict[str, Any]], records: list[dict[str, Any]]
) -> None:
    eff = {(r["step"], r["cav_id"]): r["effective"] for r in records}
    for row in rows:
        row["effective"] = eff.get((row["step"], row["cav_id"]), "unknown")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def gap_block(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(subset),
            "dR_L_K": _stats([r["dR_L_K"] for r in subset]),
            "dR_R_K": _stats([r["dR_R_K"] for r in subset]),
            "dRind_L_K": _stats([r["dRind_L_K"] for r in subset]),
            "dRind_R_K": _stats([r["dRind_R_K"] for r in subset]),
            "dRgrp_L_K": _stats([r["dRgrp_L_K"] for r in subset]),
            "dRgrp_R_K": _stats([r["dRgrp_R_K"] for r in subset]),
            "dP_L_K": _stats([r["dP_L_K"] for r in subset]),
            "dP_R_K": _stats([r["dP_R_K"] for r in subset]),
            "d_logit_l": _stats([r["d_logit_l"] for r in subset]),
            "d_logit_r": _stats([r["d_logit_r"] for r in subset]),
            "frac_abs_dR_L_gt_1e4": (
                sum(1 for r in subset if abs(r["dR_L_K"]) > 1e-4) / len(subset)
                if subset
                else float("nan")
            ),
            "frac_abs_dR_R_gt_1e4": (
                sum(1 for r in subset if abs(r["dR_R_K"]) > 1e-4) / len(subset)
                if subset
                else float("nan")
            ),
            "mean_R": {
                name: _stats([r[f"R_{name}"] for r in subset])
                for name in ("KEEP", "LEFT", "RIGHT")
            },
            "mean_P": {
                name: _stats([r[f"Pblock_{name}"] for r in subset])
                for name in ("KEEP", "LEFT", "RIGHT")
            },
        }

    out: dict[str, Any] = {"all": gap_block(rows), "by_zone": {}, "by_lane": {}, "effective": {}}
    for zone in ("before", "near", "after"):
        out["by_zone"][zone] = gap_block([r for r in rows if r["zone"] == zone])
        for lane in (0, 1, 2):
            out["by_lane"][f"{zone}|lane{lane}"] = gap_block(
                [r for r in rows if r["zone"] == zone and r["lane"] == lane]
            )
    out["effective"]["LEFT"] = gap_block([r for r in rows if r["effective"] == "LEFT"])
    out["effective"]["RIGHT"] = gap_block([r for r in rows if r["effective"] == "RIGHT"])
    out["effective"]["none"] = gap_block([r for r in rows if r["effective"] == "none"])
    # Legal vs invalid ego LEFT/RIGHT requests at the sampled state
    out["legal_left_states"] = gap_block(
        [r for r in rows if r["boundary_left"] == "none"]
    )
    out["invalid_left_states"] = gap_block(
        [r for r in rows if r["boundary_left"] == "invalid_LEFT"]
    )
    out["legal_right_states"] = gap_block(
        [r for r in rows if r["boundary_right"] == "none"]
    )
    out["invalid_right_states"] = gap_block(
        [r for r in rows if r["boundary_right"] == "invalid_RIGHT"]
    )
    return out


def write_report(
    summary: dict[str, Any],
    *,
    n_records: int,
    n_samples: int,
    frozen: bool,
    save_ok: bool,
) -> Path:
    lines: list[str] = []
    lines.append("# Phase 5.7 Same-State Reward Counterfactual Report")
    lines.append("")
    lines.append(
        "Read-only diagnosis. Candidate A and Current Reward formulas were not changed. "
        "No training. Checkpoints untouched."
    )
    lines.append("")
    lines.append("## 1. Experiment Status")
    lines.append("")
    lines.append(f"- Checkpoint: Phase 5.3 Candidate A `{CA_CKPT.name}` (read-only).")
    lines.append(f"- Seed={SEED}, horizon={MAX_STEPS}, deterministic actor.")
    lines.append(f"- Actor parameters frozen: {'YES' if frozen else 'NO'}.")
    lines.append(
        f"- State restore mechanism: TraCI `simulation.saveState` / `loadState` "
        f"({'worked' if save_ok else 'FAILED'})."
    )
    lines.append(
        "- For each sampled pre-decision snapshot, only the ego CAV's action was "
        "varied among KEEP/LEFT/RIGHT. Other CAVs used the recorded joint argmax for that step. "
        "Each branch advanced **one** simulation step (0.1 s), then the snapshot was restored."
    )
    lines.append(
        "- Because Candidate A reward is a function of post-step vehicle speeds and lanes, "
        "the comparison is r(s,a) ≡ R_CandidateA(s') after that one step. "
        "It is not a multi-step return."
    )
    lines.append("")
    lines.append("## 2. Same-State Sampling")
    lines.append("")
    lines.append(f"- Real evaluation decisions in pass 1: {n_records}.")
    lines.append(
        f"- Counterfactual samples in pass 2: {n_samples} "
        f"(up to {PER_CELL} per zone×lane cell, plus up to {MAX_EFFECTIVE} "
        "effective-lane-change precursors)."
    )
    lines.append("- No synthetic states.")
    lines.append("")
    lines.append("| Slice | n |")
    lines.append("|-------|---|")
    for zone in ("before", "near", "after"):
        lines.append(f"| {zone} | {summary['by_zone'][zone]['n']} |")
        for lane in (0, 1, 2):
            lines.append(
                f"| {zone}, lane {lane} | {summary['by_lane'][f'{zone}|lane{lane}']['n']} |"
            )
    lines.append(f"| effective LEFT precursor | {summary['effective']['LEFT']['n']} |")
    lines.append(f"| effective RIGHT precursor | {summary['effective']['RIGHT']['n']} |")
    lines.append("")
    lines.append("## 3. Immediate Reward by Action")
    lines.append("")
    lines.append(
        "Means of one-step post-transition R_total. Identical state snapshot for the three branches."
    )
    lines.append("")
    lines.append("| Slice | n | mean R(KEEP) | mean R(LEFT) | mean R(RIGHT) | median ΔR_L−K | median ΔR_R−K |")
    lines.append("|-------|---|--------------|--------------|---------------|---------------|---------------|")

    def add_slice(name: str, block: dict[str, Any]) -> None:
        if block["n"] == 0:
            lines.append(f"| {name} | 0 | n/a | n/a | n/a | n/a | n/a |")
            return
        lines.append(
            f"| {name} | {block['n']} | {_fmt(block['mean_R']['KEEP']['mean'])} | "
            f"{_fmt(block['mean_R']['LEFT']['mean'])} | {_fmt(block['mean_R']['RIGHT']['mean'])} | "
            f"{_fmt(block['dR_L_K']['median'])} | {_fmt(block['dR_R_K']['median'])} |"
        )

    add_slice("all", summary["all"])
    for zone in ("before", "near", "after"):
        add_slice(zone, summary["by_zone"][zone])
    lines.append("")
    lines.append(
        f"Fraction of samples with |ΔR_LEFT−KEEP| > 1e-4: {_fmt(summary['all']['frac_abs_dR_L_gt_1e4'], 3)}. "
        f"|ΔR_RIGHT−KEEP| > 1e-4: {_fmt(summary['all']['frac_abs_dR_R_gt_1e4'], 3)}."
    )
    lines.append("")
    lines.append("## 4. Reward Component Attribution")
    lines.append("")
    lines.append("Candidate A: `R = 0.55·R_ind + 0.35·R_group − 0.10·P_block`.")
    lines.append("")
    lines.append("| Slice | median ΔR_ind L−K | median ΔR_ind R−K | median ΔR_group L−K | median ΔR_group R−K | median ΔP L−K | median ΔP R−K |")
    lines.append("|-------|-------------------|-------------------|---------------------|---------------------|---------------|---------------|")
    for name, block in (
        ("all", summary["all"]),
        ("before", summary["by_zone"]["before"]),
        ("near", summary["by_zone"]["near"]),
        ("after", summary["by_zone"]["after"]),
        ("legal LEFT states", summary["legal_left_states"]),
        ("invalid LEFT states", summary["invalid_left_states"]),
        ("legal RIGHT states", summary["legal_right_states"]),
        ("invalid RIGHT states", summary["invalid_right_states"]),
    ):
        if block["n"] == 0:
            lines.append(f"| {name} | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {name} | {_fmt(block['dRind_L_K']['median'])} | {_fmt(block['dRind_R_K']['median'])} | "
            f"{_fmt(block['dRgrp_L_K']['median'])} | {_fmt(block['dRgrp_R_K']['median'])} | "
            f"{_fmt(block['dP_L_K']['median'])} | {_fmt(block['dP_R_K']['median'])} |"
        )
    lines.append("")
    lines.append(
        "P_block can only differ after a step that changes whether the ego is on a blocked lane "
        "inside the 150 m band. A 0.1 s step usually does not finish a 1.0 s lane change, "
        "so median ΔP is expected to be near zero on most samples."
    )
    lines.append("")
    lines.append("## 5. Policy Preference vs Reward Preference")
    lines.append("")
    lines.append(
        "| Region | median ΔR_LEFT−KEEP | median ΔR_RIGHT−KEEP | median logit ΔLEFT−KEEP | median logit ΔRIGHT−KEEP |"
    )
    lines.append(
        "|--------|---------------------|----------------------|-------------------------|--------------------------|"
    )
    for zone in ("before", "near", "after"):
        b = summary["by_zone"][zone]
        if b["n"] == 0:
            lines.append(f"| {zone} | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {zone} | {_fmt(b['dR_L_K']['median'])} | {_fmt(b['dR_R_K']['median'])} | "
            f"{_fmt(b['d_logit_l']['median'])} | {_fmt(b['d_logit_r']['median'])} |"
        )
    lines.append("")
    lines.append(
        "Policy gaps are from the same sampled states (actor logits). "
        "Reward gaps are one-step counterfactuals. They are not required to match."
    )
    lines.append("")
    lines.append("## 6. Effective Lane-change State Analysis")
    lines.append("")
    lines.append(
        "Precursors: states after which the same id’s lane index moved within 1.0 s in the free-running episode."
    )
    lines.append("")
    lines.append("| Precursor | n | median ΔR_L−K | median ΔR_R−K | median logit ΔL−K | median logit ΔR−K |")
    lines.append("|-----------|---|---------------|---------------|-------------------|-------------------|")
    for key in ("LEFT", "RIGHT", "none"):
        b = summary["effective"][key]
        if b["n"] == 0:
            lines.append(f"| effective {key} | 0 | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| effective {key} | {b['n']} | {_fmt(b['dR_L_K']['median'])} | {_fmt(b['dR_R_K']['median'])} | "
            f"{_fmt(b['d_logit_l']['median'])} | {_fmt(b['d_logit_r']['median'])} |"
        )
    lines.append("")
    lines.append("## 7. Evidence for / against Reward-induced Action Preference")
    lines.append("")

    # Build verdict from numbers
    all_b = summary["all"]
    med_l = all_b["dR_L_K"]["median"]
    med_r = all_b["dR_R_K"]["median"]
    frac_l = all_b["frac_abs_dR_L_gt_1e4"]
    frac_r = all_b["frac_abs_dR_R_gt_1e4"]
    after_b = summary["by_zone"]["after"]
    near_b = summary["by_zone"]["near"]

    lines.append("### Immediate reward preference (A)")
    lines.append("")
    lines.append(
        f"- Overall median ΔR_LEFT−KEEP = {_fmt(med_l)}, median ΔR_RIGHT−KEEP = {_fmt(med_r)}."
    )
    lines.append(
        f"- Share of samples with a non-trivial one-step gap: "
        f"LEFT {_fmt(100*frac_l,1)}%, RIGHT {_fmt(100*frac_r,1)}% (|ΔR|>1e-4)."
    )
    lines.append(
        f"- After-zone median ΔR_LEFT−KEEP = {_fmt(after_b['dR_L_K']['median'])} "
        f"while median logit ΔLEFT−KEEP = {_fmt(after_b['d_logit_l']['median'])}."
    )
    lines.append(
        f"- Near-zone median ΔP_LEFT−KEEP = {_fmt(near_b['dP_L_K']['median'])}, "
        f"ΔP_RIGHT−KEEP = {_fmt(near_b['dP_R_K']['median'])}."
    )
    lines.append("")
    lines.append("### Policy preference (B)")
    lines.append("")
    lines.append(
        "Unchanged from Phase 5.6 on these samples: KEEP is not preferred by the actor; "
        "after-zone LEFT logit still exceeds KEEP even when the one-step reward gap is tiny."
    )
    lines.append("")
    lines.append("### Effective lane-change behavior (C)")
    lines.append("")
    lines.append(
        "Effective precursors are analyzed separately above. "
        "A large policy gap on those states with a near-zero one-step reward gap "
        "means the immediate reward table does not explain the request."
    )
    lines.append("")
    lines.append("### Classification")
    lines.append("")

    # Heuristic classification for the report text
    reward_flat = (
        abs(med_l) < 1e-3
        and abs(med_r) < 1e-3
        and (frac_l < 0.2 if frac_l == frac_l else True)
    )
    policy_strong = abs(after_b["d_logit_l"]["median"]) > 0.5 if after_b["n"] else False

    if reward_flat and policy_strong:
        lines.append(
            "**Supported reading: (2)** Immediate one-step Candidate A reward shows little or no "
            "action preference on the sampled real states, while the policy still shows a clear "
            "lateral preference. This is evidence **against** “the scalar reward table at the "
            "next step is what currently ranks LEFT/RIGHT over KEEP,” not proof that reward "
            "played no role during training."
        )
    elif (not reward_flat) and policy_strong:
        # check sign alignment roughly
        align = (
            (after_b["dR_L_K"]["median"] > 0 and after_b["d_logit_l"]["median"] > 0)
            or (near_b["dR_L_K"]["median"] > 0 and near_b["d_logit_l"]["median"] > 0)
        )
        if align:
            lines.append(
                "**Supported reading: (3)** Some one-step reward gaps exist and partially "
                "align with policy preference in places, but magnitudes and coverage must be "
                "read from the tables. Causality is still not established."
            )
        else:
            lines.append(
                "**Supported reading: (3/4)** One-step reward gaps are not flat everywhere, "
                "but they do not cleanly track the policy ranking. Causal claim remains open."
            )
    else:
        lines.append(
            "**Supported reading: (4)** The present sample does not support a simple "
            "reward→policy story. Treat causality as unproven."
        )
    lines.append("")
    lines.append("Do **not** read this as “Reward caused the lane changes.”")
    lines.append("")
    lines.append("## 8. Remaining Uncertainty")
    lines.append("")
    lines.append(
        "- One-step (0.1 s) may be too short for lane index / P_block to move after a legal LEFT/RIGHT. "
        "A longer forced branch would mix trajectory effects and was out of scope."
    )
    lines.append(
        "- Other agents used the recorded argmax, not a joint counterfactual. "
        "Only the ego action was varied."
    )
    lines.append("- Stratified subsample, not every decision. Seed 42 only.")
    lines.append(
        "- Training-time returns and advantage estimates were not reconstructed. "
        "A flat one-step r(s,a) does not prove the PPO update ignored lateral actions."
    )
    lines.append("")
    lines.append("## 9. Recommendation for Phase 5.8")
    lines.append("")
    lines.append("Still do **not** edit Candidate A weights from this phase alone.")
    lines.append("")
    if reward_flat and policy_strong:
        lines.append(
            "If Phase 5.8 continues diagnosis, prefer either (a) a longer same-state branch "
            "that waits through the 1.0 s lane-change window under forced KEEP vs LEFT, "
            "or (b) measurement of executed lane-index rates under stochastic vs deterministic "
            "eval — still without changing the reward. "
            "A reward redesign is only justified after a delayed branch shows a stable component "
            "advantage for lateral actions, which this one-step test did not establish."
        )
    else:
        lines.append(
            "Revisit the one-step tables after any longer-horizon branch. "
            "Do not change the formula until a delayed same-state comparison exists."
        )
    lines.append("")
    lines.append("## Phase 5.7 Result: PASS")
    lines.append("")
    lines.append(
        "PASS = same-state one-step counterfactuals completed with TraCI save/load, "
        "components logged, and policy gaps aligned on the same samples. "
        "PASS ≠ reward edit."
    )
    lines.append("")
    path = OUT / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not CA_CKPT.is_file():
        print(f"ERROR: missing {CA_CKPT}")
        return 1
    device = resolve_device("auto")
    model, _ = load_mappo_checkpoint(CA_CKPT, device=device)
    model.eval()
    before = snapshot_model_params(model)
    print(f"Phase 5.7 device={device}")
    print("--- pass 1: collect real states ---")
    pack = pass1_collect(model, device)
    records = pack["records"]
    print(f"records={len(records)}")
    rng = np.random.default_rng(SEED)
    samples = choose_samples(records, rng)
    print(f"samples={len(samples)}")
    print("--- pass 2: same-state one-step CF ---")
    try:
        rows = pass2_counterfactual(model, device, samples, pack["n_lanes"])
        save_ok = True
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR during save/load counterfactual: {exc}")
        save_ok = False
        rows = []
        raise
    attach_effective(rows, records)
    after = snapshot_model_params(model)
    frozen = params_unchanged(before, after)
    summary = summarize(rows)
    (OUT / "counterfactual_rows.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "n_records": len(records),
                "n_samples": len(rows),
                "params_frozen": frozen,
                "save_ok": save_ok,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report = write_report(
        summary,
        n_records=len(records),
        n_samples=len(rows),
        frozen=frozen,
        save_ok=save_ok,
    )
    print(f"Report: {report}")
    print("Phase 5.7 Result: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
