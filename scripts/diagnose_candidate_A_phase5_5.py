"""Phase 5.5 — Candidate A reward attribution (read-only).

Uses Phase 5.4 step traces for component stats.
One deterministic replay collects actor logits only.
Does not train, does not write checkpoints, does not change Current Reward.
"""

from __future__ import annotations

import csv
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
from env.action import KEEP, LEFT, RIGHT, ACTION_NAMES  # noqa: E402
from env.highway_accident_env import HighwayAccidentEnv, load_config  # noqa: E402
from utils.baseline_eval import load_mappo_checkpoint, set_eval_seed  # noqa: E402

OUT = PROJECT_ROOT / "results" / "experiments" / "reward_candidate_A_phase5_5"
TRACE = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "reward_candidate_A_phase5_4"
    / "traces"
    / "candidate_A_steps.csv"
)
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
W_IND, W_GRP, W_B, ALPHA = 0.55, 0.35, 0.10, 0.6
COMPONENTS = ("r_total", "r_ind", "r_group", "s_cav", "s_loc", "p_block", "w_ind", "w_grp", "w_block")


def _stats(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std(ddof=0)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def _fmt(x: float, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:.{nd}f}"


def load_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rec = {
                "step": int(float(r["step"])),
                "cav_id": r["cav_id"],
                "action": int(float(r["action"])),
                "action_name": r["action_name"],
                "pos": float(r["pos"]),
                "lane": int(float(r["lane"])),
                "speed": float(r["speed"]),
                "zone": r["zone"],
                "on_blocked": int(float(r["on_blocked"])),
                "near": int(float(r["near"])),
                "r_total": float(r["r_total"]),
                "r_ind": float(r["r_ind"]),
                "r_group": float(r["r_group"]),
                "s_cav": float(r["s_cav"]),
                "s_loc": float(r["s_loc"]),
                "p_block": float(r["p_block"]),
            }
            rec["w_ind"] = W_IND * rec["r_ind"]
            rec["w_grp"] = W_GRP * rec["r_group"]
            rec["w_block"] = -W_B * rec["p_block"]
            rec["r_unclipped"] = rec["w_ind"] + rec["w_grp"] + rec["w_block"]
            rows.append(rec)
    return rows


def group_stats(rows: list[dict[str, Any]], key_fn) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = key_fn(r)
        for c in COMPONENTS:
            buckets[k][c].append(float(r[c]))
    return {k: {c: _stats(v) for c, v in comps.items()} for k, comps in buckets.items()}


def analyze_oscillation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[r["cav_id"]].append(r)
    flips: list[dict[str, Any]] = []
    boundary = {"RIGHT_from_lane0": 0, "LEFT_from_lane2": 0, "lc_cmds": 0, "paired": 0}
    for seq in by.values():
        seq.sort(key=lambda x: x["step"])
        for i in range(1, len(seq)):
            prev, cur = seq[i - 1], seq[i]
            if cur["step"] != prev["step"] + 1:
                continue
            boundary["paired"] += 1
            if cur["action"] in (LEFT, RIGHT):
                boundary["lc_cmds"] += 1
                if cur["action"] == RIGHT and prev["lane"] == 0:
                    boundary["RIGHT_from_lane0"] += 1
                if cur["action"] == LEFT and prev["lane"] == 2:
                    boundary["LEFT_from_lane2"] += 1
            pair = {prev["action"], cur["action"]}
            if pair != {LEFT, RIGHT}:
                continue
            flips.append(
                {
                    "from": prev["action_name"],
                    "to": cur["action_name"],
                    "zone": cur["zone"],
                    "d_total": cur["r_total"] - prev["r_total"],
                    "d_ind": cur["r_ind"] - prev["r_ind"],
                    "d_group": cur["r_group"] - prev["r_group"],
                    "d_scav": cur["s_cav"] - prev["s_cav"],
                    "d_sloc": cur["s_loc"] - prev["s_loc"],
                    "d_pblock": cur["p_block"] - prev["p_block"],
                    "d_speed": cur["speed"] - prev["speed"],
                    "d_lane": cur["lane"] - prev["lane"],
                    "lane_before": prev["lane"],
                    "lane_after": cur["lane"],
                    "on_blocked_before": prev["on_blocked"],
                    "on_blocked_after": cur["on_blocked"],
                    "near": cur["near"],
                }
            )
    def col(name: str) -> list[float]:
        return [float(f[name]) for f in flips]

    by_dir: dict[str, int] = defaultdict(int)
    by_zone: dict[str, int] = defaultdict(int)
    for f in flips:
        by_dir[f"{f['from']}→{f['to']}"] += 1
        by_zone[f["zone"]] += 1
    n = len(flips)
    speed_up = sum(1 for f in flips if f["d_speed"] > 0.1)
    speed_down = sum(1 for f in flips if f["d_speed"] < -0.1)
    lane_changed = sum(1 for f in flips if f["d_lane"] != 0)
    pblock_down = sum(1 for f in flips if f["d_pblock"] < -0.5)
    r_up = sum(1 for f in flips if f["d_total"] > 0.001)
    r_down = sum(1 for f in flips if f["d_total"] < -0.001)
    return {
        "n_flips": n,
        "by_direction": dict(by_dir),
        "by_zone": dict(by_zone),
        "delta": {k: _stats(col(k)) for k in ("d_total", "d_ind", "d_group", "d_scav", "d_sloc", "d_pblock", "d_speed", "d_lane")},
        "frac_speed_up": speed_up / n if n else float("nan"),
        "frac_speed_down": speed_down / n if n else float("nan"),
        "frac_lane_changed": lane_changed / n if n else float("nan"),
        "frac_pblock_down": pblock_down / n if n else float("nan"),
        "frac_rtotal_up": r_up / n if n else float("nan"),
        "frac_rtotal_down": r_down / n if n else float("nan"),
        "boundary_inferred": boundary,
    }


def analyze_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_action = group_stats(rows, lambda r: r["action_name"])
    by_zone_action = group_stats(rows, lambda r: f"{r['zone']}|{r['action_name']}")
    recon_err = [abs(r["r_total"] - min(1.0, max(0.0, r["r_unclipped"]))) for r in rows]
    return {
        "n_rows": len(rows),
        "by_action": by_action,
        "by_zone_action": by_zone_action,
        "recon_abs_err": _stats(recon_err),
        "oscillation": analyze_oscillation(rows),
        "p_block_nonzero": int(sum(1 for r in rows if r["p_block"] > 0.5)),
    }


def collect_policy(device: torch.device) -> dict[str, Any]:
    import copy

    set_eval_seed(SEED)
    model, _ = load_mappo_checkpoint(CA_CKPT, device=device)
    model.eval()
    before = [p.detach().cpu().clone() for p in model.parameters()]
    env_cfg = copy.deepcopy(load_config())
    env_cfg.setdefault("episode", {})["max_episode_steps"] = MAX_STEPS
    env = HighwayAccidentEnv(config=env_cfg, gui=False, reward_config=CA_REWARD)
    x_acc = float(env.accident_position_m)
    gaps_l = defaultdict(list)
    gaps_r = defaultdict(list)
    probs = defaultdict(lambda: defaultdict(list))
    argmax_n = defaultdict(lambda: defaultdict(int))
    boundary_live = {"RIGHT_lane0": 0, "LEFT_lane2": 0, "RIGHT": 0, "LEFT": 0, "KEEP": 0}
    n = 0
    try:
        env.reset()
        for _ in range(MAX_STEPS):
            states = env.get_cav_states()
            actions: dict[str, int] = {}
            for cid, obs in env.get_observations().items():
                st = states.get(cid)
                if st is None:
                    continue
                pos = float(st["position"])
                lane = int(st["lane"])
                if pos < x_acc - 150:
                    zone = "before"
                elif pos > x_acc + 150:
                    zone = "after"
                else:
                    zone = "near"
                t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    logits = model.actor.forward(t)
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)
                    prob = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
                    logit = logits[0].detach().cpu().numpy()
                a = int(np.argmax(logit))
                actions[cid] = a
                n += 1
                gaps_l[zone].append(float(logit[LEFT] - logit[KEEP]))
                gaps_r[zone].append(float(logit[RIGHT] - logit[KEEP]))
                gaps_l["all"].append(float(logit[LEFT] - logit[KEEP]))
                gaps_r["all"].append(float(logit[RIGHT] - logit[KEEP]))
                for i, name in ACTION_NAMES.items():
                    probs[zone][name].append(float(prob[i]))
                    probs["all"][name].append(float(prob[i]))
                argmax_n[zone][ACTION_NAMES[a]] += 1
                argmax_n["all"][ACTION_NAMES[a]] += 1
                if a == RIGHT:
                    boundary_live["RIGHT"] += 1
                    if lane == 0:
                        boundary_live["RIGHT_lane0"] += 1
                elif a == LEFT:
                    boundary_live["LEFT"] += 1
                    if lane == 2:
                        boundary_live["LEFT_lane2"] += 1
                else:
                    boundary_live["KEEP"] += 1
            _obs, _rew, _term, truncated, _info = env.step(actions)
            if truncated:
                break
    finally:
        env.close()
    after = [p.detach().cpu().clone() for p in model.parameters()]
    frozen = all(torch.equal(a, b) for a, b in zip(before, after))
    out: dict[str, Any] = {"n": n, "params_frozen": frozen, "boundary_live": boundary_live, "zones": {}}
    for zone in ("all", "before", "near", "after"):
        total = sum(argmax_n[zone].values())
        out["zones"][zone] = {
            "n": total,
            "logit_LEFT_minus_KEEP": _stats(gaps_l[zone]),
            "logit_RIGHT_minus_KEEP": _stats(gaps_r[zone]),
            "prob": {name: _stats(probs[zone][name]) for name in ("KEEP", "LEFT", "RIGHT")},
            "argmax_frac": {
                name: (argmax_n[zone][name] / total if total else float("nan"))
                for name in ("KEEP", "LEFT", "RIGHT")
            },
        }
    return out


def _cell(block: dict[str, Any] | None, comp: str) -> str:
    if not block or comp not in block or block[comp]["n"] == 0:
        return "n/a"
    s = block[comp]
    return f"{_fmt(s['mean'])} (n={s['n']})"


def write_report(trace: dict[str, Any], policy: dict[str, Any]) -> Path:
    by_a = trace["by_action"]
    by_za = trace["by_zone_action"]
    osc = trace["oscillation"]
    b = osc["boundary_inferred"]
    live = policy["boundary_live"]
    lines: list[str] = []
    lines.append("# Phase 5.5 Reward Attribution Report")
    lines.append("")
    lines.append("Read-only attribution. No retraining. Current Reward and all checkpoints unchanged.")
    lines.append("")
    lines.append("## 1. Experiment Status")
    lines.append("")
    lines.append(f"- Component statistics: Phase 5.4 trace `{TRACE.name}` ({trace['n_rows']} CAV-steps).")
    lines.append(f"- Policy logits: one new deterministic replay, seed={SEED}, steps={MAX_STEPS}, checkpoint `episode_0050.pt` of Candidate A.")
    lines.append(f"- Actor/critic parameters frozen during logit replay: {'YES' if policy['params_frozen'] else 'NO'}.")
    lines.append("- Critic is V(s), not Q(s,a). Action-values are **not available** and were not invented.")
    lines.append("- Reward rows are **post-step states of the executed action**, not counterfactual rewards of the other two actions in the same state.")
    lines.append(f"- Reconstructed clip(0.55·R_ind+0.35·R_group−0.10·P_block) vs logged R_total, mean abs error: {_fmt(trace['recon_abs_err']['mean'], 6)}.")
    lines.append("")
    lines.append("## 2. Candidate A Reward Structure")
    lines.append("")
    lines.append("From `env/reward.py` `_get_rewards_candidate_a` and `configs/reward_candidate_A.yaml`:")
    lines.append("")
    lines.append("```text")
    lines.append("R_ind   = clip(v_ego / 30, 0, 1)")
    lines.append("S_cav   = mean neighbor-CAV speed ratios (|Δx|≤150), else R_ind")
    lines.append("S_loc   = mean local non-ACCIDENT speed ratios (|Δx|≤150), else R_ind")
    lines.append("R_group = 0.6·S_cav + 0.4·S_loc")
    lines.append("P_block = 1 iff |x−700|≤150 AND lane∈{0}, else 0")
    lines.append("R_total = clip(0.55·R_ind + 0.35·R_group − 0.10·P_block, 0, 1)")
    lines.append("```")
    lines.append("")
    lines.append("There is no lane-change term, no queue term, and no action-dependent bonus. `P_block` is the only term that depends on lane, and only inside the 150 m accident band.")
    lines.append("")
    lines.append("## 3. Reward Attribution by Action")
    lines.append("")
    lines.append("KEEP is absent (n=0). No synthetic KEEP rewards.")
    lines.append("")
    lines.append("| Action | n | R_total | R_ind | R_group | S_cav | S_loc | P_block | 0.55·R_ind | 0.35·R_group | −0.10·P_block |")
    lines.append("|--------|---|---------|-------|---------|-------|-------|---------|------------|--------------|---------------|")
    for name in ("KEEP", "LEFT", "RIGHT"):
        block = by_a.get(name)
        if not block or block["r_total"]["n"] == 0:
            lines.append(f"| {name} | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {name} | {block['r_total']['n']} | {_fmt(block['r_total']['mean'])} | "
            f"{_fmt(block['r_ind']['mean'])} | {_fmt(block['r_group']['mean'])} | "
            f"{_fmt(block['s_cav']['mean'])} | {_fmt(block['s_loc']['mean'])} | "
            f"{_fmt(block['p_block']['mean'])} | {_fmt(block['w_ind']['mean'])} | "
            f"{_fmt(block['w_grp']['mean'])} | {_fmt(block['w_block']['mean'])} |"
        )
    lines.append("")
    lines.append("Spread (LEFT and RIGHT only):")
    lines.append("")
    lines.append("| Component | LEFT median / std / min / max | RIGHT median / std / min / max |")
    lines.append("|-----------|-------------------------------|--------------------------------|")
    for comp, label in (
        ("r_total", "R_total"),
        ("r_ind", "R_ind"),
        ("r_group", "R_group"),
        ("p_block", "P_block"),
    ):
        L, R = by_a["LEFT"][comp], by_a["RIGHT"][comp]
        lines.append(
            f"| {label} | {_fmt(L['median'])} / {_fmt(L['std'])} / {_fmt(L['min'])} / {_fmt(L['max'])} | "
            f"{_fmt(R['median'])} / {_fmt(R['std'])} / {_fmt(R['min'])} / {_fmt(R['max'])} |"
        )
    d_tot = by_a["LEFT"]["r_total"]["mean"] - by_a["RIGHT"]["r_total"]["mean"]
    lines.append("")
    lines.append(
        f"Mean R_total(LEFT) − R_total(RIGHT) = {_fmt(d_tot)}. "
        "RIGHT has the higher mean total in this executed-action sample, driven by a higher R_group, "
        "while RIGHT also carries a higher mean P_block. This is a correlation on visited states, not proof that RIGHT is rewarded more than LEFT at a fixed state."
    )
    lines.append("")
    lines.append("## 4. Reward Attribution by Spatial Region")
    lines.append("")
    lines.append("Zones match Phase 5.4: before x<550, near |x−700|≤150, after x>850.")
    lines.append("")
    lines.append("| Zone | Action | n | R_total | R_ind | R_group | S_cav | S_loc | P_block |")
    lines.append("|------|--------|---|---------|-------|---------|-------|-------|---------|")
    for zone in ("before", "near", "after"):
        for name in ("KEEP", "LEFT", "RIGHT"):
            block = by_za.get(f"{zone}|{name}")
            if not block or block["r_total"]["n"] == 0:
                lines.append(f"| {zone} | {name} | 0 | n/a | n/a | n/a | n/a | n/a | n/a |")
                continue
            lines.append(
                f"| {zone} | {name} | {block['r_total']['n']} | {_fmt(block['r_total']['mean'])} | "
                f"{_fmt(block['r_ind']['mean'])} | {_fmt(block['r_group']['mean'])} | "
                f"{_fmt(block['s_cav']['mean'])} | {_fmt(block['s_loc']['mean'])} | "
                f"{_fmt(block['p_block']['mean'])} |"
            )
    lines.append("")
    after_l = by_za.get("after|LEFT")
    after_r = by_za.get("after|RIGHT")
    lines.append(
        "After the accident zone, P_block is structurally 0 (outside the 150 m gate). "
        f"Logged after-zone P_block mean on LEFT = {_fmt(after_l['p_block']['mean']) if after_l else 'n/a'}. "
        "RIGHT has n=0 after the zone under the deterministic policy, so no LEFT-vs-RIGHT reward contrast exists there. "
        "Continued LEFT after the zone is therefore **not** supported by a measured P_block advantage, and cannot be attributed to a blocked-lane term."
    )
    lines.append("")
    lines.append("## 5. Lane-change Oscillation Analysis")
    lines.append("")
    lines.append(
        f"Consecutive same-id LEFT↔RIGHT flips in the Phase 5.4 trace: **{osc['n_flips']}** "
        f"(directions {osc['by_direction']}; zones {osc['by_zone']})."
    )
    lines.append("")
    lines.append("| Delta (after flip − before) | mean | median | std |")
    lines.append("|-----------------------------|------|--------|-----|")
    for key, label in (
        ("d_total", "R_total"),
        ("d_ind", "R_ind"),
        ("d_group", "R_group"),
        ("d_scav", "S_cav"),
        ("d_sloc", "S_loc"),
        ("d_pblock", "P_block"),
        ("d_speed", "ego speed (m/s)"),
        ("d_lane", "lane index"),
    ):
        s = osc["delta"][key]
        lines.append(f"| {label} | {_fmt(s['mean'])} | {_fmt(s['median'])} | {_fmt(s['std'])} |")
    lines.append("")
    lines.append(
        f"Among flips: speed up {100*osc['frac_speed_up']:.1f}%, speed down {100*osc['frac_speed_down']:.1f}%, "
        f"lane index changed {100*osc['frac_lane_changed']:.1f}%, P_block fell {100*osc['frac_pblock_down']:.1f}%, "
        f"R_total up {100*osc['frac_rtotal_up']:.1f}%, R_total down {100*osc['frac_rtotal_down']:.1f}%."
    )
    lines.append("")
    lines.append(
        "Median reward and speed deltas are near zero. Most flips do not measurably improve ego speed or clear P_block. "
        "That rejects “each oscillation step improves traffic and is therefore paid by the reward” as a description of these  flips. "
        "It does not prove the flips are reward-neutral in a causal sense, because both sides of the flip are executed states, not a paired counterfactual."
    )
    lines.append("")
    lines.append("### Boundary actions (implementation, not a reward term)")
    lines.append("")
    lines.append(
        f"From consecutive trace pairs (pre-lane = previous sample): RIGHT issued from lane 0 = {b['RIGHT_from_lane0']} "
        f"of {b['lc_cmds']} paired LC commands; LEFT issued from lane 2 = {b['LEFT_from_lane2']}."
    )
    lines.append(
        f"Live replay, lane **before** the action: RIGHT on lane 0 = {live['RIGHT_lane0']} / {live['RIGHT']} RIGHT commands "
        f"({100*live['RIGHT_lane0']/live['RIGHT']:.1f}%); LEFT on lane 2 = {live['LEFT_lane2']} / {live['LEFT']} "
        f"({100*live['LEFT_lane2']/live['LEFT']:.1f}%)."
    )
    lines.append(
        "`apply_action` treats RIGHT on lane 0 and LEFT on lane 2 as no-ops (`target_lane_index` returns None). "
        "A large share of RIGHT commands never request a SUMO lane change. Those steps still count as RIGHT in the action distribution."
    )
    lines.append("")
    lines.append("## 6. Policy Preference Analysis")
    lines.append("")
    lines.append("Same observation, actor logits and softmax. Deterministic action = argmax. No Q-values.")
    lines.append("")
    lines.append("| Zone | n | P(KEEP) | P(LEFT) | P(RIGHT) | argmax KEEP/LEFT/RIGHT | logit(L)−logit(K) | logit(R)−logit(K) |")
    lines.append("|------|---|---------|---------|----------|------------------------|-------------------|-------------------|")
    for zone in ("all", "before", "near", "after"):
        z = policy["zones"][zone]
        af = z["argmax_frac"]
        lines.append(
            f"| {zone} | {z['n']} | {_fmt(z['prob']['KEEP']['mean'])} | {_fmt(z['prob']['LEFT']['mean'])} | "
            f"{_fmt(z['prob']['RIGHT']['mean'])} | "
            f"{_fmt(af['KEEP'], 3)} / {_fmt(af['LEFT'], 3)} / {_fmt(af['RIGHT'], 3)} | "
            f"{_fmt(z['logit_LEFT_minus_KEEP']['mean'])} | {_fmt(z['logit_RIGHT_minus_KEEP']['mean'])} |"
        )
    lines.append("")
    after = policy["zones"]["after"]
    lines.append(
        f"After the zone, mean P(KEEP)={_fmt(after['prob']['KEEP']['mean'])}, "
        f"argmax LEFT fraction={_fmt(after['argmax_frac']['LEFT'], 3)}. "
        "The KEEP logit is below both LC logits on the visited after-zone states. "
        "That is a policy ranking, not a reward-component ranking."
    )
    lines.append("")
    lines.append("## 7. Root-cause Evidence")
    lines.append("")
    lines.append("### 已证实")
    lines.append("")
    lines.append("- Candidate A’s scalar reward has exactly three additive pieces: speed (`R_ind`), group speed (`R_group` from `S_cav`/`S_loc`), and blocked-lane occupancy (`P_block`). No lane-change reward exists in code.")
    lines.append("- `P_block` is zero outside the 150 m band. It cannot explain after-zone LEFT≈100%.")
    lines.append("- Deterministic argmax never selects KEEP on this checkpoint, including before and after the accident. Mean logit(KEEP) is below both alternatives in every zone measured here.")
    lines.append("- A substantial fraction of RIGHT commands are issued on lane 0, where the environment suppresses the lane change. Action histograms therefore overstate completed maneuvers.")
    lines.append(f"- The {osc['n_flips']} LEFT↔RIGHT flips have near-zero median ΔR_total and Δspeed; most do not lower P_block.")
    lines.append("")
    lines.append("### 有证据支持")
    lines.append("")
    lines.append("- **B, with a D contribution:** the over-LC pattern is carried by the learned argmax, not by an explicit LC bonus. After-zone and off-block behavior continue to prefer LEFT even though `P_block` is off. Boundary no-ops inflate RIGHT counts, especially before the zone and while still on lane 0.")
    lines.append("- **A is only locally plausible, and only for `P_block`:** inside the near band, being on lane 0 subtracts 0.10. That can favor leaving lane 0. It does not pay for LEFT versus KEEP once the vehicle is already off lane 0, and the data show LEFT still dominates those states (Phase 5.4 clear-near LEFT≈99.3%). The post-step tables do not show a large R_total gap that would, by itself, select LEFT over RIGHT everywhere.")
    lines.append("- Entropy decline during training (Phase 5.4) is consistent with the policy concentrating on non-KEEP actions; this replay shows that concentration survived into deterministic eval.")
    lines.append("")
    lines.append("### 尚不能确定")
    lines.append("")
    lines.append("- Whether any component **causes** the LC policy. Post-step means mix the action with the state it produced. A proper causal test would branch the same state under KEEP/LEFT/RIGHT and compare next rewards. That was not run.")
    lines.append("- Whether group benefit specifically rewards oscillation. Flip-level ΔS_cav and ΔS_loc are small; that is evidence against a large per-flip group bonus, not a proof that group features never shaped the policy during training.")
    lines.append("- Critic action preference. No Q(s,a) is stored.")
    lines.append("")
    lines.append("Classification against the requested labels:")
    lines.append("")
    lines.append("- **A:** not established as the global cause. `P_block` is a bounded local tax, not an LC reward, and it is off after the zone.")
    lines.append("- **B:** supported. Policy logits rank LC above KEEP on the states the deterministic policy actually visits, while immediate component gaps do not track that ranking.")
    lines.append("- **C:** still true for causal credit assignment. Attribution of *which training signal created the logits* is not identified.")
    lines.append("- **D:** supported as an amplifier. Lane-0 RIGHT is often a no-op; eval argmax removes the KEEP mass that training still sampled.")
    lines.append("")
    lines.append("## 8. What Cannot Yet Be Concluded")
    lines.append("")
    lines.append("- Do not conclude that raising or lowering `w_b`, `α`, or `w_grp` would remove after-zone LEFT. Those weights were not ablated here, and after-zone `P_block` is already zero.")
    lines.append("- Do not conclude Candidate A is better or worse on traffic metrics. This phase did not re-score throughput or queue.")
    lines.append("- Do not treat LEFT vs RIGHT mean rewards as the return of choosing that action.")
    lines.append("")
    lines.append("## 9. Recommendation for Phase 5.6")
    lines.append("")
    lines.append("Do not change Candidate A weights yet. The attribution does not identify a weight that is paying for after-zone LEFT.")
    lines.append("")
    lines.append("A useful next measurement, still without a new reward, is a **same-state branch**: at a sample of before / near / after states, force KEEP, LEFT, and RIGHT for one step and record the next `R_ind`, `R_group`, `P_block`, and whether SUMO actually changed lane. That is the missing causal comparison. Separately, report executed lane changes (not raw action ids) so boundary no-ops are not counted as maneuvers.")
    lines.append("")
    lines.append("Only if that branch shows a stable component advantage for LC over KEEP should Phase 5.6 touch the formula.")
    lines.append("")
    lines.append("## Phase 5.5 Result: PASS")
    lines.append("")
    lines.append("PASS = attribution completed from the existing trace plus one frozen-policy replay. PASS ≠ a new reward and ≠ Candidate A superiority.")
    lines.append("")
    path = OUT / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not TRACE.is_file():
        print(f"ERROR: missing trace {TRACE}")
        return 1
    if not CA_CKPT.is_file():
        print(f"ERROR: missing checkpoint {CA_CKPT}")
        return 1
    print("Phase 5.5 attribution")
    rows = load_trace(TRACE)
    trace = analyze_trace(rows)
    print(f"trace rows={trace['n_rows']} flips={trace['oscillation']['n_flips']}")
    device = resolve_device("auto")
    print(f"logit replay on {device}")
    policy = collect_policy(device)
    print(f"policy n={policy['n']} frozen={policy['params_frozen']}")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trace": trace,
        "policy": policy,
    }
    (OUT / "attribution_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = write_report(trace, policy)
    print(f"Report: {report}")
    print("Phase 5.5 Result: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
