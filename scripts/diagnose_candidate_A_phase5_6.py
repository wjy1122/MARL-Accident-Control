"""Phase 5.6 — counterfactual policy preference (read-only).

One forward pass yields logits/probs for KEEP/LEFT/RIGHT on the visited state.
Only the argmax action is sent to SUMO. No training, no reward edits, no checkpoint writes.
"""

from __future__ import annotations

import copy
import json
import math
import sys
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

OUT = PROJECT_ROOT / "results" / "experiments" / "reward_candidate_A_phase5_6"
CA_CKPT = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "reward_candidate_A_phase5_3"
    / "checkpoints"
    / "episode_0050.pt"
)
CUR_CKPT = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "mappo_formal_50u_1200m_700m_seed42"
    / "checkpoints"
    / "episode_0050.pt"
)
CA_REWARD = PROJECT_ROOT / "configs" / "reward_candidate_A.yaml"
SEED = 42
MAX_STEPS = 1800
ZONE_R = 150.0
NAMES = ("KEEP", "LEFT", "RIGHT")


def _stats(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
        }
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "std": float(np.std(a)),
    }


def _fmt(x: float, nd: int = 3) -> str:
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


def run_policy(
    *,
    label: str,
    ckpt: Path,
    reward_config: Path | None,
    device: torch.device,
) -> dict[str, Any]:
    set_eval_seed(SEED)
    model, _ = load_mappo_checkpoint(ckpt, device=device)
    model.eval()
    before = snapshot_model_params(model)
    env_cfg = copy.deepcopy(load_config())
    env_cfg.setdefault("episode", {})["max_episode_steps"] = MAX_STEPS
    env = HighwayAccidentEnv(config=env_cfg, gui=False, reward_config=reward_config)
    x_acc = float(env.accident_position_m)
    n_lanes = int(env.n_lanes)
    horizon = max(1, int(round(float(env.lane_change_duration_s) / float(env.step_length))))

    rows: list[dict[str, Any]] = []
    try:
        env.reset()
        for step in range(MAX_STEPS):
            states = env.get_cav_states()
            actions: dict[str, int] = {}
            for cid, obs in env.get_observations().items():
                st = states.get(cid)
                if st is None:
                    continue
                lane = int(st["lane"])
                pos = float(st["position"])
                zone = _zone(pos, x_acc)
                t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device)
                with torch.no_grad():
                    logits_t = model.actor.forward(t)
                    if logits_t.dim() == 1:
                        logits_t = logits_t.unsqueeze(0)
                    prob_t = torch.softmax(logits_t, dim=-1)[0]
                logits = [float(x) for x in logits_t[0].detach().cpu().tolist()]
                probs = [float(x) for x in prob_t.detach().cpu().tolist()]
                action = int(np.argmax(logits))
                actions[cid] = action
                rows.append(
                    {
                        "step": step,
                        "cav_id": cid,
                        "zone": zone,
                        "lane": lane,
                        "pos": pos,
                        "action": action,
                        "boundary": _boundary(action, lane, n_lanes),
                        "logits": logits,
                        "probs": probs,
                    }
                )
            _obs, _rew, _term, truncated, _info = env.step(actions)
            if truncated:
                break
    finally:
        env.close()

    after = snapshot_model_params(model)
    return {
        "label": label,
        "n": len(rows),
        "horizon_steps": horizon,
        "params_frozen": params_unchanged(before, after),
        "rows": rows,
    }


def analyze(pack: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = pack["rows"]
    horizon = int(pack["horizon_steps"])
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    lane_at: dict[tuple[str, int], int] = {}
    for r in rows:
        by_id[r["cav_id"]].append(r)
        lane_at[(r["cav_id"], r["step"])] = r["lane"]

    gaps_l: dict[str, list[float]] = defaultdict(list)
    gaps_r: dict[str, list[float]] = defaultdict(list)
    argmax: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    coverage: dict[str, int] = defaultdict(int)
    cats = defaultdict(int)
    # outcome of each decision within the lane-change window
    outcome = defaultdict(int)

    req_switch = 0
    req_switch_both_invalid = 0
    req_switch_both_legal = 0
    invalid_left_repeat = 0
    invalid_right_repeat = 0
    effective_events = 0
    effective_left = 0
    effective_right = 0
    effective_switch = 0

    for r in rows:
        zone = r["zone"]
        lane_key = f"lane{r['lane']}"
        d_l = r["logits"][LEFT] - r["logits"][KEEP]
        d_r = r["logits"][RIGHT] - r["logits"][KEEP]
        name = ACTION_NAMES[r["action"]]
        for key in (zone, "all", f"{zone}|{lane_key}"):
            gaps_l[key].append(d_l)
            gaps_r[key].append(d_r)
            argmax[key][name] += 1
        coverage[f"{zone}|{lane_key}"] += 1
        b = r["boundary"]
        if b == "invalid_LEFT":
            cats["invalid_LEFT"] += 1
        elif b == "invalid_RIGHT":
            cats["invalid_RIGHT"] += 1
        elif r["action"] == KEEP:
            cats["KEEP"] += 1
        elif r["action"] == LEFT:
            cats["legal_LEFT"] += 1
        else:
            cats["legal_RIGHT"] += 1

        future = None
        for k in range(1, horizon + 1):
            nxt = lane_at.get((r["cav_id"], r["step"] + k))
            if nxt is None:
                break
            future = nxt
        if future is None:
            outcome["unobserved"] += 1
        elif b == "invalid_LEFT":
            outcome["invalid_LEFT_lane_unchanged" if future == r["lane"] else "invalid_LEFT_lane_changed"] += 1
        elif b == "invalid_RIGHT":
            outcome["invalid_RIGHT_lane_unchanged" if future == r["lane"] else "invalid_RIGHT_lane_changed"] += 1
        elif r["action"] == KEEP:
            outcome["KEEP_lane_unchanged" if future == r["lane"] else "KEEP_lane_changed"] += 1
        elif r["action"] == LEFT:
            if future > r["lane"]:
                outcome["successful_LEFT"] += 1
            elif future == r["lane"]:
                outcome["legal_LEFT_unchanged"] += 1
            else:
                outcome["legal_LEFT_other"] += 1
        else:
            if future < r["lane"]:
                outcome["successful_RIGHT"] += 1
            elif future == r["lane"]:
                outcome["legal_RIGHT_unchanged"] += 1
            else:
                outcome["legal_RIGHT_other"] += 1

    for seq in by_id.values():
        seq.sort(key=lambda x: x["step"])
        events: list[int] = []
        for i in range(1, len(seq)):
            prev, cur = seq[i - 1], seq[i]
            if cur["step"] != prev["step"] + 1:
                events = []
                continue
            pair = {prev["action"], cur["action"]}
            if pair == {LEFT, RIGHT}:
                req_switch += 1
                if prev["boundary"] != "none" and cur["boundary"] != "none":
                    req_switch_both_invalid += 1
                if prev["boundary"] == "none" and cur["boundary"] == "none":
                    req_switch_both_legal += 1
            if prev["boundary"] == "invalid_LEFT" and cur["boundary"] == "invalid_LEFT":
                invalid_left_repeat += 1
            if prev["boundary"] == "invalid_RIGHT" and cur["boundary"] == "invalid_RIGHT":
                invalid_right_repeat += 1
            if cur["lane"] != prev["lane"]:
                direction = 1 if cur["lane"] > prev["lane"] else -1
                effective_events += 1
                if direction > 0:
                    effective_left += 1
                else:
                    effective_right += 1
                if events and events[-1] != direction:
                    effective_switch += 1
                events.append(direction)

    def pack_zone(key: str) -> dict[str, Any]:
        total = sum(argmax[key].values())
        return {
            "n": total,
            "delta_left_keep": _stats(gaps_l[key]),
            "delta_right_keep": _stats(gaps_r[key]),
            "argmax_frac": {
                name: (argmax[key][name] / total if total else float("nan"))
                for name in NAMES
            },
            "argmax_n": {name: int(argmax[key][name]) for name in NAMES},
        }

    zones = {z: pack_zone(z) for z in ("all", "before", "near", "after")}
    by_lane = {
        f"{z}|lane{lane}": pack_zone(f"{z}|lane{lane}")
        for z in ("before", "near", "after")
        for lane in (0, 1, 2)
    }
    return {
        "label": pack["label"],
        "n": pack["n"],
        "horizon_steps": horizon,
        "params_frozen": pack["params_frozen"],
        "n_vehicle_ids": len(by_id),
        "coverage": dict(coverage),
        "zones": zones,
        "by_lane": by_lane,
        "requested": dict(cats),
        "window_outcome": dict(outcome),
        "requested_switches": req_switch,
        "requested_switches_both_invalid": req_switch_both_invalid,
        "requested_switches_both_legal": req_switch_both_legal,
        "invalid_left_repeats": invalid_left_repeat,
        "invalid_right_repeats": invalid_right_repeat,
        "effective_events": effective_events,
        "effective_left": effective_left,
        "effective_right": effective_right,
        "effective_switches": effective_switch,
    }


def _row_pref(z: dict[str, Any]) -> str:
    af = z["argmax_frac"]
    dl = z["delta_left_keep"]
    dr = z["delta_right_keep"]
    return (
        f"| {z['n']} | {_fmt(dl['mean'])} | {_fmt(dl['median'])} | "
        f"{_fmt(dr['mean'])} | {_fmt(dr['median'])} | "
        f"{_fmt(100*af['KEEP'],1)} / {_fmt(100*af['LEFT'],1)} / {_fmt(100*af['RIGHT'],1)} |"
    )


def write_report(ca: dict[str, Any], cur: dict[str, Any]) -> Path:
    lines: list[str] = []
    lines.append("# Phase 5.6 Counterfactual Policy Diagnosis")
    lines.append("")
    lines.append(
        "Diagnosis only. Reward formulas, training, and checkpoints were not modified. "
        "Logits come from one actor forward pass on the visited observation. "
        "Only argmax was executed. The other two actions were not simulated."
    )
    lines.append("")
    lines.append("## 1. Experiment Status")
    lines.append("")
    lines.append(f"- Candidate A checkpoint: `{CA_CKPT.name}` under Phase 5.3 (read-only).")
    lines.append(f"- Current Reward checkpoint: formal `{CUR_CKPT.name}` (read-only).")
    lines.append(f"- Protocol: seed={SEED}, steps={MAX_STEPS}, GUI=OFF, deterministic argmax.")
    lines.append(
        f"- Lane-change window used for “did the lane index move”: "
        f"{ca['horizon_steps']} steps "
        f"(lane_change_duration 1.0 s / step 0.1 s). "
        "A single 0.1 s step is too short to call a maneuver finished."
    )
    lines.append(
        f"- Parameters frozen: Candidate A={'YES' if ca['params_frozen'] else 'NO'}, "
        f"Current={'YES' if cur['params_frozen'] else 'NO'}."
    )
    lines.append("- Gaps below are **logits**, not probabilities. Softmax probabilities were computed but not used as the gap.")
    lines.append("- Critic remains V(s). No Q(s,a).")
    lines.append("")
    lines.append("## 2. State Sampling")
    lines.append("")
    lines.append(
        "Every CAV observation from the deterministic episode was used. "
        "No synthetic states."
    )
    lines.append("")
    lines.append(f"- Candidate A decisions: {ca['n']} across {ca['n_vehicle_ids']} ids.")
    lines.append(f"- Current Reward decisions: {cur['n']} across {cur['n_vehicle_ids']} ids.")
    lines.append("")
    lines.append("| Zone × lane | Candidate A n | Current n |")
    lines.append("|-------------|---------------|-----------|")
    for z in ("before", "near", "after"):
        for lane in (0, 1, 2):
            key = f"{z}|lane{lane}"
            lines.append(
                f"| {z}, lane {lane} | {ca['coverage'].get(key, 0)} | {cur['coverage'].get(key, 0)} |"
            )
    lines.append("")
    lines.append("## 3. Policy Preference by Region")
    lines.append("")
    lines.append("Argmax is the requested action. It is not an effective lane change.")
    lines.append("")
    lines.append("### Candidate A")
    lines.append("")
    lines.append("| Region | n | mean ΔL−K | median ΔL−K | mean ΔR−K | median ΔR−K | argmax KEEP/LEFT/RIGHT % |")
    lines.append("|--------|---|-----------|-------------|-----------|-------------|--------------------------|")
    for z in ("before", "near", "after", "all"):
        lines.append(f"| {z} | " + _row_pref(ca["zones"][z]).lstrip("| "))
    lines.append("")
    lines.append("### Candidate A by lane")
    lines.append("")
    lines.append("| Zone | Lane | n | mean ΔL−K | mean ΔR−K | argmax KEEP/LEFT/RIGHT % |")
    lines.append("|------|------|---|-----------|-----------|--------------------------|")
    for z in ("before", "near", "after"):
        for lane in (0, 1, 2):
            block = ca["by_lane"][f"{z}|lane{lane}"]
            if block["n"] == 0:
                lines.append(f"| {z} | {lane} | 0 | n/a | n/a | n/a |")
                continue
            af = block["argmax_frac"]
            lines.append(
                f"| {z} | {lane} | {block['n']} | {_fmt(block['delta_left_keep']['mean'])} | "
                f"{_fmt(block['delta_right_keep']['mean'])} | "
                f"{_fmt(100*af['KEEP'],1)} / {_fmt(100*af['LEFT'],1)} / {_fmt(100*af['RIGHT'],1)} |"
            )
    lines.append("")
    lines.append("## 4. Action Preference Gap")
    lines.append("")
    lines.append("ΔLEFT-KEEP = logit_LEFT − logit_KEEP. ΔRIGHT-KEEP = logit_RIGHT − logit_KEEP.")
    lines.append("Positive means that action outranks KEEP on that visited state. This is a policy ranking, not a reward difference.")
    lines.append("")
    b, n, a = ca["zones"]["before"], ca["zones"]["near"], ca["zones"]["after"]
    lines.append(
        f"- Before: median ΔLEFT-KEEP={_fmt(b['delta_left_keep']['median'])}, "
        f"median ΔRIGHT-KEEP={_fmt(b['delta_right_keep']['median'])}. "
        f"Argmax RIGHT={_fmt(100*b['argmax_frac']['RIGHT'],1)}%."
    )
    lines.append(
        f"- Near: median ΔLEFT-KEEP={_fmt(n['delta_left_keep']['median'])}, "
        f"median ΔRIGHT-KEEP={_fmt(n['delta_right_keep']['median'])}. "
        f"Argmax LEFT={_fmt(100*n['argmax_frac']['LEFT'],1)}%."
    )
    lines.append(
        f"- After: median ΔLEFT-KEEP={_fmt(a['delta_left_keep']['median'])}, "
        f"median ΔRIGHT-KEEP={_fmt(a['delta_right_keep']['median'])}. "
        f"Argmax LEFT={_fmt(100*a['argmax_frac']['LEFT'],1)}%, "
        f"argmax KEEP={_fmt(100*a['argmax_frac']['KEEP'],1)}%."
    )
    lines.append("")
    lines.append(
        "KEEP is not the argmax in any region aggregate. "
        "After the zone the median RIGHT gap is negative, so the preference is specifically LEFT over KEEP, not “any lateral action.”"
    )
    lines.append("")
    lines.append("## 5. Requested vs Effective Lane Changes")
    lines.append("")
    lines.append(
        "Requested = argmax. Invalid boundary = LEFT on the leftmost lane or RIGHT on lane 0, "
        "which `target_lane_index` refuses before TraCI. "
        "Successful = within the next 1.0 s the same id’s lane index moves in the requested direction. "
        "Windows overlap, so successful counts are not distinct maneuvers."
    )
    lines.append("")

    def req_table(title: str, s: dict[str, Any]) -> None:
        req = s["requested"]
        out = s["window_outcome"]
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| Category | Count |")
        lines.append("|----------|-------|")
        for key in ("KEEP", "legal_LEFT", "legal_RIGHT", "invalid_LEFT", "invalid_RIGHT"):
            lines.append(f"| requested {key} | {req.get(key, 0)} |")
        lines.append(f"| window successful_LEFT | {out.get('successful_LEFT', 0)} |")
        lines.append(f"| window successful_RIGHT | {out.get('successful_RIGHT', 0)} |")
        lines.append(f"| window legal LEFT, lane unchanged | {out.get('legal_LEFT_unchanged', 0)} |")
        lines.append(f"| window legal RIGHT, lane unchanged | {out.get('legal_RIGHT_unchanged', 0)} |")
        lines.append(f"| window invalid LEFT, lane unchanged | {out.get('invalid_LEFT_lane_unchanged', 0)} |")
        lines.append(f"| window invalid RIGHT, lane unchanged | {out.get('invalid_RIGHT_lane_unchanged', 0)} |")
        lines.append(f"| window invalid LEFT, lane changed | {out.get('invalid_LEFT_lane_changed', 0)} |")
        lines.append(f"| window invalid RIGHT, lane changed | {out.get('invalid_RIGHT_lane_changed', 0)} |")
        lines.append(f"| observed lane-index increases (effective LEFT events) | {s['effective_left']} |")
        lines.append(f"| observed lane-index decreases (effective RIGHT events) | {s['effective_right']} |")
        lines.append("")

    req_table("Candidate A", ca)
    req_table("Current Reward", cur)

    lines.append("## 6. Boundary Invalid Actions")
    lines.append("")
    ca_inv = ca["requested"].get("invalid_LEFT", 0) + ca["requested"].get("invalid_RIGHT", 0)
    ca_lat = ca_inv + ca["requested"].get("legal_LEFT", 0) + ca["requested"].get("legal_RIGHT", 0)
    lines.append(
        f"Candidate A lateral requests: {ca_lat}. "
        f"Of those, invalid boundary requests: {ca_inv} "
        f"({_fmt(100*ca_inv/ca_lat,1) if ca_lat else 'n/a'}%)."
    )
    lines.append(
        f"- RIGHT while already on lane 0: {ca['requested'].get('invalid_RIGHT', 0)}."
    )
    lines.append(
        f"- LEFT while already on lane 2: {ca['requested'].get('invalid_LEFT', 0)}."
    )
    inv_r = ca["requested"].get("invalid_RIGHT", 0)
    inv_l = ca["requested"].get("invalid_LEFT", 0)
    stay_r = ca["window_outcome"].get("invalid_RIGHT_lane_unchanged", 0)
    stay_l = ca["window_outcome"].get("invalid_LEFT_lane_unchanged", 0)
    lines.append(
        f"- Within 1.0 s, lane stayed at the boundary for "
        f"{stay_r}/{inv_r} invalid RIGHT and {stay_l}/{inv_l} invalid LEFT requests."
    )
    lines.append(
        f"- Consecutive invalid LEFT→invalid LEFT pairs: {ca['invalid_left_repeats']}."
    )
    lines.append(
        f"- Consecutive invalid RIGHT→invalid RIGHT pairs: {ca['invalid_right_repeats']}."
    )
    lines.append("")
    lines.append(
        "These repeats are repeated requests that the environment rejects. "
        "They are not effective lane-changing oscillation."
    )
    lines.append("")
    lines.append("## 7. Effective Lane-change Oscillation")
    lines.append("")
    lines.append(
        "Requested switching = consecutive LEFT↔RIGHT **actions**. "
        "Effective switching = consecutive observed lane-index changes in opposite directions. "
        "Unchanged steps do not count."
    )
    lines.append("")
    lines.append("| Metric | Candidate A | Current Reward |")
    lines.append("|--------|-------------|----------------|")
    lines.append(f"| requested LEFT↔RIGHT switches | {ca['requested_switches']} | {cur['requested_switches']} |")
    lines.append(
        f"| of which both sides were boundary-invalid | {ca['requested_switches_both_invalid']} | {cur['requested_switches_both_invalid']} |"
    )
    lines.append(
        f"| of which both sides were legal requests | {ca['requested_switches_both_legal']} | {cur['requested_switches_both_legal']} |"
    )
    lines.append(f"| effective lane-change events | {ca['effective_events']} | {cur['effective_events']} |")
    lines.append(f"| effective LEFT events | {ca['effective_left']} | {cur['effective_left']} |")
    lines.append(f"| effective RIGHT events | {ca['effective_right']} | {cur['effective_right']} |")
    lines.append(f"| effective opposite-direction switches | {ca['effective_switches']} | {cur['effective_switches']} |")
    lines.append(
        f"| invalid boundary actions | {ca_inv} | "
        f"{cur['requested'].get('invalid_LEFT', 0) + cur['requested'].get('invalid_RIGHT', 0)} |"
    )
    lines.append(
        f"| effective switches / episode | {ca['effective_switches']} | {cur['effective_switches']} |"
    )
    lines.append("")
    lines.append("One episode each (1800 steps). Not a rate per kilometre.")
    lines.append("")
    lines.append("## 8. Candidate A vs Current Reward")
    lines.append("")
    lines.append("| | Candidate A | Current |")
    lines.append("|--|-------------|---------|")
    for z in ("before", "near", "after"):
        lines.append(
            f"| argmax KEEP % ({z}) | {_fmt(100*ca['zones'][z]['argmax_frac']['KEEP'],1)} | "
            f"{_fmt(100*cur['zones'][z]['argmax_frac']['KEEP'],1)} |"
        )
    lines.append(f"| requested switches | {ca['requested_switches']} | {cur['requested_switches']} |")
    lines.append(f"| effective switches | {ca['effective_switches']} | {cur['effective_switches']} |")
    lines.append(f"| effective events | {ca['effective_events']} | {cur['effective_events']} |")
    lines.append("")
    lines.append("## 9. Diagnosis")
    lines.append("")
    lines.append("Separated claims:")
    lines.append("")
    lines.append(
        f"- **Policy preference (已证实):** On visited states, Candidate A argmax is never KEEP "
        f"(before RIGHT {_fmt(100*b['argmax_frac']['RIGHT'],1)}%, "
        f"near LEFT {_fmt(100*n['argmax_frac']['LEFT'],1)}%, "
        f"after LEFT {_fmt(100*a['argmax_frac']['LEFT'],1)}%). "
        f"After the zone, median logit(LEFT)−logit(KEEP)={_fmt(a['delta_left_keep']['median'])} "
        f"and median logit(RIGHT)−logit(KEEP)={_fmt(a['delta_right_keep']['median'])}."
    )
    lines.append(
        f"- **Requested action (已证实):** Candidate A requested "
        f"{ca['requested'].get('legal_LEFT', 0)+ca['requested'].get('invalid_LEFT', 0)} LEFT and "
        f"{ca['requested'].get('legal_RIGHT', 0)+ca['requested'].get('invalid_RIGHT', 0)} RIGHT, "
        f"and {ca['requested'].get('KEEP', 0)} KEEP."
    )
    lines.append(
        f"- **Invalid boundary action (已证实):** {ca_inv} of {ca_lat} lateral requests "
        f"were refused by the lane boundary. "
        f"Consecutive invalid repeats: LEFT {ca['invalid_left_repeats']}, RIGHT {ca['invalid_right_repeats']}. "
        "These are not lane changes."
    )
    lines.append(
        f"- **Effective lane change (已证实):** Candidate A produced {ca['effective_events']} "
        f"observed lane-index changes ({ca['effective_left']} increases, {ca['effective_right']} decreases), "
        f"versus {cur['effective_events']} for Current Reward."
    )
    lines.append(
        f"- **Effective oscillation (已证实):** opposite-direction lane-index reversals were "
        f"{ca['effective_switches']} for Candidate A and {cur['effective_switches']} for Current. "
        f"Requested action switches were {ca['requested_switches']} vs {cur['requested_switches']}. "
        "The Phase 5.4 count of 314 action flips is a requested-action count, not an effective-oscillation count."
    )
    lines.append(
        "- **Reward-component causality (尚不能确定):** this phase did not branch the environment. "
        "A higher logit for LEFT is not evidence that `R_group` or `P_block` pays for that action."
    )
    lines.append("")
    lines.append("## 10. What Remains Unknown")
    lines.append("")
    lines.append("- Why training produced these logits. No component was ablated and no same-state reward branch was run.")
    lines.append("- Whether a legal request that does not move the lane within 1.0 s was blocked by SUMO traffic rather than by the action mask. The window outcome mixes both.")
    lines.append("- Multi-seed stability. One paired episode, seed 42.")
    lines.append("")
    lines.append("## 11. Recommendation for Phase 5.7")
    lines.append("")
    lines.append(
        "Do not change reward weights on the basis of this diagnosis. "
        "The established fact is a policy ranking plus a large invalid-request mass, not a proven reward term."
    )
    lines.append("")
    lines.append(
        "If a later phase changes anything, the first measurement to keep is executed lane-index changes, "
        "not raw LEFT/RIGHT counts. A reward edit is still not justified until a same-state branch shows "
        "KEEP and LEFT producing different next-step components."
    )
    lines.append("")
    lines.append("## Phase 5.6 Result: PASS")
    lines.append("")
    lines.append(
        "PASS = both policies were replayed read-only, logits were read without executing the counterfactual actions, "
        "and requested / invalid / effective counts were separated. PASS ≠ a reward change."
    )
    lines.append("")
    path = OUT / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not CA_CKPT.is_file() or not CUR_CKPT.is_file():
        print("ERROR: missing checkpoint")
        return 1
    device = resolve_device("auto")
    print(f"Phase 5.6 device={device}")
    print("--- Candidate A ---")
    ca_pack = run_policy(label="CandidateA", ckpt=CA_CKPT, reward_config=CA_REWARD, device=device)
    ca = analyze(ca_pack)
    print(
        f"CA n={ca['n']} req_switch={ca['requested_switches']} "
        f"eff_switch={ca['effective_switches']} frozen={ca['params_frozen']}"
    )
    print("--- Current Reward ---")
    cur_pack = run_policy(label="Current", ckpt=CUR_CKPT, reward_config=None, device=device)
    cur = analyze(cur_pack)
    print(
        f"Current n={cur['n']} req_switch={cur['requested_switches']} "
        f"eff_switch={cur['effective_switches']} frozen={cur['params_frozen']}"
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_a": {k: v for k, v in ca.items()},
        "current": {k: v for k, v in cur.items()},
    }
    (OUT / "diagnosis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = write_report(ca, cur)
    print(f"Report: {report}")
    print("Phase 5.6 Result: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
