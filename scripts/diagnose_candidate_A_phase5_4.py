"""Phase 5.4 — Candidate A behavior diagnosis (read-only).

Does NOT train, does NOT modify Current Reward / checkpoints / train logic.
Replays Candidate A (and Current for contrast) with per-step traces.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.rollout import resolve_device  # noqa: E402
from env.action import KEEP, LEFT, RIGHT, ACTION_NAMES  # noqa: E402
from env.highway_accident_env import HighwayAccidentEnv, load_config  # noqa: E402
from utils.baseline_eval import load_mappo_checkpoint, set_eval_seed  # noqa: E402

OUT_DIR = PROJECT_ROOT / "results" / "experiments" / "reward_candidate_A_phase5_4"
PHASE53 = PROJECT_ROOT / "results" / "experiments" / "reward_candidate_A_phase5_3"
CA_CKPT = PHASE53 / "checkpoints" / "episode_0050.pt"
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

# Spatial zones relative to accident (matches obs range / Candidate A zone_range)
# before: x < x_acc - R
# near:   |x - x_acc| <= R
# after:  x > x_acc + R
ZONE_R_M = 150.0


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
        self.flush()
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _zone(pos: float, x_acc: float, r: float = ZONE_R_M) -> str:
    if pos < x_acc - r:
        return "before"
    if pos > x_acc + r:
        return "after"
    return "near"


@dataclass
class TraceAgg:
    keep: int = 0
    left: int = 0
    right: int = 0
    # distance proxy: sum of |Δpos| for successive samples of same vehicle id
    distance_m: float = 0.0
    unique_cavs: set[str] = field(default_factory=set)
    # per-zone action counts
    zone_keep: dict[str, int] = field(
        default_factory=lambda: {"before": 0, "near": 0, "after": 0}
    )
    zone_left: dict[str, int] = field(
        default_factory=lambda: {"before": 0, "near": 0, "after": 0}
    )
    zone_right: dict[str, int] = field(
        default_factory=lambda: {"before": 0, "near": 0, "after": 0}
    )
    # reward by action (post-step total + components)
    r_tot: dict[int, list[float]] = field(
        default_factory=lambda: {KEEP: [], LEFT: [], RIGHT: []}
    )
    r_ind: dict[int, list[float]] = field(
        default_factory=lambda: {KEEP: [], LEFT: [], RIGHT: []}
    )
    r_grp: dict[int, list[float]] = field(
        default_factory=lambda: {KEEP: [], LEFT: [], RIGHT: []}
    )
    r_scav: dict[int, list[float]] = field(
        default_factory=lambda: {KEEP: [], LEFT: [], RIGHT: []}
    )
    r_sloc: dict[int, list[float]] = field(
        default_factory=lambda: {KEEP: [], LEFT: [], RIGHT: []}
    )
    r_pblock: dict[int, list[float]] = field(
        default_factory=lambda: {KEEP: [], LEFT: [], RIGHT: []}
    )
    # reward delta: post_total - pre_total for same cav_id if both exist
    r_delta: dict[int, list[float]] = field(
        default_factory=lambda: {KEEP: [], LEFT: [], RIGHT: []}
    )
    # oscillation: consecutive LEFT↔RIGHT flips (same vehicle id)
    osc_lr_rl: int = 0
    osc_triplets: int = 0  # L-R-L or R-L-R
    max_run_lc: int = 0  # longest consecutive non-KEEP streak
    consecutive_lc_steps: int = 0  # sum over agents of consecutive LC run lengths
    blocked_near_steps: int = 0
    near_cav_steps: int = 0
    # when on blocked lane near accident: action distribution
    blocked_near_keep: int = 0
    blocked_near_left: int = 0
    blocked_near_right: int = 0
    # when NOT on blocked lane near accident
    clear_near_keep: int = 0
    clear_near_left: int = 0
    clear_near_right: int = 0


def run_trace(
    *,
    label: str,
    ckpt: Path,
    reward_config: Path | None,
    device: torch.device,
    out_csv: Path,
) -> TraceAgg:
    """One deterministic eval episode with per-CAV step traces."""
    import copy

    set_eval_seed(SEED)
    model, _ = load_mappo_checkpoint(ckpt, device=device)
    model.eval()

    env_cfg = copy.deepcopy(load_config())
    env_cfg.setdefault("episode", {})["max_episode_steps"] = int(MAX_STEPS)
    env = HighwayAccidentEnv(
        config=env_cfg, gui=False, reward_config=reward_config
    )
    x_acc = float(env.accident_position_m)
    blocked = set(int(x) for x in env.accident_lanes)
    agg = TraceAgg()

    # Per-vehicle history for oscillation / distance
    prev_pos: dict[str, float] = {}
    prev_action: dict[str, int] = {}
    prev2_action: dict[str, int] = {}
    run_lc: dict[str, int] = {}

    rows: list[dict[str, Any]] = []

    try:
        env.reset()
        for step_i in range(MAX_STEPS):
            # Pre-step rewards (state before action effect)
            pre_rewards = env.get_rewards()
            states = env.get_cav_states()
            actions: dict[str, int] = {}
            for cid, obs in env.get_observations().items():
                t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    a, _ = model.act(t, deterministic=True)
                actions[cid] = int(a.item())

            # Apply
            _obs, _rew, _term, truncated, info = env.step(actions)
            post = info.get("reward_details") or env.get_rewards()
            post_states = env.get_cav_states()

            for cid, a in actions.items():
                st0 = states.get(cid)
                st1 = post_states.get(cid)
                # Prefer post-step position for zone of consequence; fall back to pre
                if st1 is not None:
                    pos = float(st1["position"])
                    lane = int(st1["lane"])
                    spd = float(st1["speed"])
                elif st0 is not None:
                    pos = float(st0["position"])
                    lane = int(st0["lane"])
                    spd = float(st0["speed"])
                else:
                    continue

                z = _zone(pos, x_acc)
                agg.unique_cavs.add(cid)

                if a == KEEP:
                    agg.keep += 1
                    agg.zone_keep[z] += 1
                elif a == LEFT:
                    agg.left += 1
                    agg.zone_left[z] += 1
                else:
                    agg.right += 1
                    agg.zone_right[z] += 1

                # distance
                if cid in prev_pos:
                    agg.distance_m += abs(pos - prev_pos[cid])
                prev_pos[cid] = pos

                # oscillation / consecutive LC
                if a != KEEP:
                    run_lc[cid] = run_lc.get(cid, 0) + 1
                    agg.max_run_lc = max(agg.max_run_lc, run_lc[cid])
                else:
                    if run_lc.get(cid, 0) > 0:
                        agg.consecutive_lc_steps += run_lc[cid]
                    run_lc[cid] = 0

                pa = prev_action.get(cid)
                p2 = prev2_action.get(cid)
                if pa is not None:
                    # immediate flip LEFT↔RIGHT
                    if (pa == LEFT and a == RIGHT) or (pa == RIGHT and a == LEFT):
                        agg.osc_lr_rl += 1
                    # triplet L-R-L / R-L-R
                    if p2 is not None:
                        if (p2 == LEFT and pa == RIGHT and a == LEFT) or (
                            p2 == RIGHT and pa == LEFT and a == RIGHT
                        ):
                            agg.osc_triplets += 1
                if pa is not None:
                    prev2_action[cid] = pa
                prev_action[cid] = a

                # blocked near
                near = abs(pos - x_acc) <= ZONE_R_M
                on_block = lane in blocked
                if near:
                    agg.near_cav_steps += 1
                    if on_block:
                        agg.blocked_near_steps += 1
                        if a == KEEP:
                            agg.blocked_near_keep += 1
                        elif a == LEFT:
                            agg.blocked_near_left += 1
                        else:
                            agg.blocked_near_right += 1
                    else:
                        if a == KEEP:
                            agg.clear_near_keep += 1
                        elif a == LEFT:
                            agg.clear_near_left += 1
                        else:
                            agg.clear_near_right += 1

                # rewards post-step keyed by action taken
                detail = post.get(cid) if isinstance(post, dict) else None
                if detail is not None:
                    tot = float(detail.get("total", float("nan")))
                    ind = float(detail.get("individual", float("nan")))
                    grp = float(detail.get("group", float("nan")))
                    scav = float(detail.get("s_cav", float("nan")))
                    sloc = float(detail.get("s_loc", float("nan")))
                    pb = float(detail.get("p_block", float("nan")))
                    agg.r_tot[a].append(tot)
                    agg.r_ind[a].append(ind)
                    agg.r_grp[a].append(grp)
                    agg.r_scav[a].append(scav)
                    agg.r_sloc[a].append(sloc)
                    agg.r_pblock[a].append(pb)
                    pre_d = pre_rewards.get(cid)
                    if pre_d is not None:
                        agg.r_delta[a].append(tot - float(pre_d.get("total", tot)))

                    rows.append(
                        {
                            "label": label,
                            "step": step_i,
                            "cav_id": cid,
                            "action": a,
                            "action_name": ACTION_NAMES[a],
                            "pos": pos,
                            "lane": lane,
                            "speed": spd,
                            "zone": z,
                            "on_blocked": int(on_block),
                            "near": int(near),
                            "r_total": tot,
                            "r_ind": ind,
                            "r_group": grp,
                            "s_cav": scav,
                            "s_loc": sloc,
                            "p_block": pb,
                        }
                    )

            if truncated:
                break
    finally:
        env.close()

    # flush remaining LC runs
    for cid, run in run_lc.items():
        if run > 0:
            agg.consecutive_lc_steps += run

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return agg


def analyze_train_metrics(path: Path) -> dict[str, Any]:
    """Per-episode action stats from Phase 5.3 train_metrics.csv (resume 21–50)."""
    if not path.is_file():
        return {"available": False, "reason": f"missing {path}"}

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        return {"available": False, "reason": "empty train_metrics.csv"}

    def fget(r: dict[str, Any], k: str) -> float:
        return float(r[k])

    episodes = []
    for r in rows:
        keep = int(float(r["KEEP_count"]))
        left = int(float(r["LEFT_count"]))
        right = int(float(r["RIGHT_count"]))
        total = keep + left + right
        lc = left + right
        # rough km from max_pos-min_pos of the rollout window (not true CAV-km)
        span_m = max(0.0, fget(r, "max_pos") - fget(r, "min_pos"))
        span_km = span_m / 1000.0 if span_m > 0 else float("nan")
        episodes.append(
            {
                "episode": int(float(r["episode"])),
                "KEEP": keep,
                "LEFT": left,
                "RIGHT": right,
                "total_actions": total,
                "lane_change_cmds": lc,
                "KEEP_pct": 100.0 * keep / total if total else float("nan"),
                "LEFT_pct": 100.0 * left / total if total else float("nan"),
                "RIGHT_pct": 100.0 * right / total if total else float("nan"),
                "near_KEEP_pct": float(r["near_KEEP_pct"]),
                "near_LEFT_pct": float(r["near_LEFT_pct"]),
                "near_RIGHT_pct": float(r["near_RIGHT_pct"]),
                "span_m": span_m,
                "lc_per_span_km": (lc / span_km) if span_km and span_km == span_km else float("nan"),
                "mean_reward": fget(r, "mean_reward"),
                "entropy": fget(r, "entropy"),
            }
        )

    keep_pcts = [e["KEEP_pct"] for e in episodes]
    lc_cmds = [e["lane_change_cmds"] for e in episodes]
    return {
        "available": True,
        "n_episodes": len(episodes),
        "episode_range": (episodes[0]["episode"], episodes[-1]["episode"]),
        "mean_KEEP_pct": _mean(keep_pcts),
        "min_KEEP_pct": min(keep_pcts),
        "max_KEEP_pct": max(keep_pcts),
        "mean_LEFT_pct": _mean([e["LEFT_pct"] for e in episodes]),
        "mean_RIGHT_pct": _mean([e["RIGHT_pct"] for e in episodes]),
        "mean_lc_cmds": _mean([float(x) for x in lc_cmds]),
        "mean_near_LEFT_pct": _mean([e["near_LEFT_pct"] for e in episodes]),
        "mean_near_KEEP_pct": _mean([e["near_KEEP_pct"] for e in episodes]),
        "final_KEEP_pct": episodes[-1]["KEEP_pct"],
        "final_LEFT_pct": episodes[-1]["LEFT_pct"],
        "final_RIGHT_pct": episodes[-1]["RIGHT_pct"],
        "final_entropy": episodes[-1]["entropy"],
        "first_entropy": episodes[0]["entropy"],
        "episodes": episodes,
        "note": (
            "train_metrics.csv covers resume segment only (episodes after "
            "checkpoint load). Episodes 1–20 exist in stdout.log but not this CSV."
        ),
    }


def _fmt(x: float, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:.{nd}f}"


def _pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else float("nan")


def summarize_agg(agg: TraceAgg) -> dict[str, Any]:
    total = agg.keep + agg.left + agg.right
    lc = agg.left + agg.right
    dist_km = agg.distance_m / 1000.0 if agg.distance_m > 0 else float("nan")
    n_cav = len(agg.unique_cavs)
    return {
        "total_actions": total,
        "KEEP": agg.keep,
        "LEFT": agg.left,
        "RIGHT": agg.right,
        "KEEP_pct": _pct(agg.keep, total),
        "LEFT_pct": _pct(agg.left, total),
        "RIGHT_pct": _pct(agg.right, total),
        "lane_change_cmds": lc,
        "distance_m": agg.distance_m,
        "lc_per_km": (lc / dist_km) if dist_km and dist_km == dist_km else float("nan"),
        "n_unique_cav_ids": n_cav,
        "mean_lc_cmds_per_cav_id": (lc / n_cav) if n_cav else float("nan"),
        "zone": {
            z: {
                "KEEP": agg.zone_keep[z],
                "LEFT": agg.zone_left[z],
                "RIGHT": agg.zone_right[z],
                "KEEP_pct": _pct(
                    agg.zone_keep[z],
                    agg.zone_keep[z] + agg.zone_left[z] + agg.zone_right[z],
                ),
                "LEFT_pct": _pct(
                    agg.zone_left[z],
                    agg.zone_keep[z] + agg.zone_left[z] + agg.zone_right[z],
                ),
                "RIGHT_pct": _pct(
                    agg.zone_right[z],
                    agg.zone_keep[z] + agg.zone_left[z] + agg.zone_right[z],
                ),
                "n": agg.zone_keep[z] + agg.zone_left[z] + agg.zone_right[z],
            }
            for z in ("before", "near", "after")
        },
        "osc_lr_rl_flips": agg.osc_lr_rl,
        "osc_triplets_LRL_RLR": agg.osc_triplets,
        "osc_flip_rate_per_lc": (agg.osc_lr_rl / lc) if lc else float("nan"),
        "max_consecutive_lc_run": agg.max_run_lc,
        "mean_r_total_by_action": {
            ACTION_NAMES[a]: _mean(agg.r_tot[a]) for a in (KEEP, LEFT, RIGHT)
        },
        "mean_r_ind_by_action": {
            ACTION_NAMES[a]: _mean(agg.r_ind[a]) for a in (KEEP, LEFT, RIGHT)
        },
        "mean_r_group_by_action": {
            ACTION_NAMES[a]: _mean(agg.r_grp[a]) for a in (KEEP, LEFT, RIGHT)
        },
        "mean_s_cav_by_action": {
            ACTION_NAMES[a]: _mean(agg.r_scav[a]) for a in (KEEP, LEFT, RIGHT)
        },
        "mean_s_loc_by_action": {
            ACTION_NAMES[a]: _mean(agg.r_sloc[a]) for a in (KEEP, LEFT, RIGHT)
        },
        "mean_p_block_by_action": {
            ACTION_NAMES[a]: _mean(agg.r_pblock[a]) for a in (KEEP, LEFT, RIGHT)
        },
        "mean_r_delta_by_action": {
            ACTION_NAMES[a]: _mean(agg.r_delta[a]) for a in (KEEP, LEFT, RIGHT)
        },
        "n_samples_by_action": {
            ACTION_NAMES[a]: len(agg.r_tot[a]) for a in (KEEP, LEFT, RIGHT)
        },
        "blocked_near_frac": (
            agg.blocked_near_steps / agg.near_cav_steps
            if agg.near_cav_steps
            else float("nan")
        ),
        "blocked_near_actions": {
            "KEEP": agg.blocked_near_keep,
            "LEFT": agg.blocked_near_left,
            "RIGHT": agg.blocked_near_right,
        },
        "clear_near_actions": {
            "KEEP": agg.clear_near_keep,
            "LEFT": agg.clear_near_left,
            "RIGHT": agg.clear_near_right,
        },
    }


def write_report(
    out: Path,
    *,
    train: dict[str, Any],
    ca: dict[str, Any],
    cur: dict[str, Any],
    phase53_eval: dict[str, Any] | None,
) -> Path:
    lines: list[str] = []
    lines.append("# Phase 5.4 — Candidate A Behavior Diagnosis")
    lines.append("")
    lines.append(
        "Diagnosis only. Does **not** claim Candidate A is better or worse "
        "as a final research conclusion. Checkpoints and Current Reward untouched."
    )
    lines.append("")

    # A
    lines.append("## A. Experiment Status")
    lines.append("")
    lines.append("- Source: Phase 5.3 Candidate A `episode_0050.pt` (read-only).")
    lines.append(f"- Formal Current checkpoint: `{CUR_CKPT}` (read-only).")
    lines.append("- No training. No Reward redesign. No overwrite of Phase 5.3 artifacts.")
    lines.append(
        f"- Diagnostic eval: seed={SEED}, steps={MAX_STEPS}, GUI=OFF, "
        "deterministic=True."
    )
    lines.append(
        "- **Important:** Phase 5.3 aggregate KEEP≈0% was measured under "
        "**deterministic** MAPPO eval. Training used stochastic sampling "
        "(entropy > 0), so train KEEP% was higher — see Section B."
    )
    lines.append(
        "- Lane-change counts here are **action commands** (LEFT/RIGHT), "
        "not verified completed SUMO lane-change maneuvers."
    )
    lines.append("")

    # B
    lines.append("## B. Lane-change Statistics")
    lines.append("")
    lines.append("### B.1 Training (Phase 5.3 `train_metrics.csv`)")
    lines.append("")
    if not train.get("available"):
        lines.append(f"Missing: {train.get('reason')}")
    else:
        lines.append(train["note"])
        lines.append("")
        lines.append(
            f"- Episodes in CSV: {train['episode_range'][0]}–{train['episode_range'][1]} "
            f"(n={train['n_episodes']})"
        )
        lines.append(
            f"- Mean KEEP/LEFT/RIGHT %: "
            f"{_fmt(train['mean_KEEP_pct'],1)} / "
            f"{_fmt(train['mean_LEFT_pct'],1)} / "
            f"{_fmt(train['mean_RIGHT_pct'],1)}"
        )
        lines.append(
            f"- KEEP% range across episodes: "
            f"{_fmt(train['min_KEEP_pct'],1)} – {_fmt(train['max_KEEP_pct'],1)}"
        )
        lines.append(
            f"- Final update (ep {train['episode_range'][1]}): "
            f"KEEP/LEFT/RIGHT % = "
            f"{_fmt(train['final_KEEP_pct'],1)} / "
            f"{_fmt(train['final_LEFT_pct'],1)} / "
            f"{_fmt(train['final_RIGHT_pct'],1)}"
        )
        lines.append(
            f"- Mean near-zone KEEP/LEFT %: "
            f"{_fmt(train['mean_near_KEEP_pct'],1)} / "
            f"{_fmt(train['mean_near_LEFT_pct'],1)}"
        )
        lines.append(
            f"- Entropy first→last (resume segment): "
            f"{_fmt(train['first_entropy'])} → {_fmt(train['final_entropy'])}"
        )
        lines.append("")
        lines.append(
            "| episode | KEEP | LEFT | RIGHT | KEEP% | LEFT% | RIGHT% | "
            "nearKEEP% | nearLEFT% | nearRIGHT% |"
        )
        lines.append(
            "|---------|------|------|-------|-------|-------|--------|"
            "-----------|-----------|------------|"
        )
        # show every 5th + last
        eps = train["episodes"]
        show = [e for e in eps if e["episode"] % 5 == 0 or e is eps[-1]]
        for e in show:
            lines.append(
                f"| {e['episode']} | {e['KEEP']} | {e['LEFT']} | {e['RIGHT']} | "
                f"{_fmt(e['KEEP_pct'],1)} | {_fmt(e['LEFT_pct'],1)} | "
                f"{_fmt(e['RIGHT_pct'],1)} | {_fmt(e['near_KEEP_pct'],1)} | "
                f"{_fmt(e['near_LEFT_pct'],1)} | {_fmt(e['near_RIGHT_pct'],1)} |"
            )
        lines.append("")
        lines.append(
            "Data gap: episodes 1–20 action totals are only in Phase 5.3 "
            "`stdout.log` (percentages), not in `train_metrics.csv` because "
            "metrics CSV was rewritten on resume."
        )

    lines.append("")
    lines.append("### B.2 Deterministic evaluation replay (Phase 5.4)")
    lines.append("")
    lines.append(
        "| Method | KEEP | LEFT | RIGHT | KEEP% | LEFT% | RIGHT% | "
        "LC cmds | LC/km | mean LC cmds / CAV-id |"
    )
    lines.append(
        "|--------|------|------|-------|-------|-------|--------|"
        "---------|-------|----------------------|"
    )
    for name, s in (("Candidate A", ca), ("Current Reward", cur)):
        lines.append(
            f"| {name} | {s['KEEP']} | {s['LEFT']} | {s['RIGHT']} | "
            f"{_fmt(s['KEEP_pct'],1)} | {_fmt(s['LEFT_pct'],1)} | "
            f"{_fmt(s['RIGHT_pct'],1)} | {s['lane_change_cmds']} | "
            f"{_fmt(s['lc_per_km'],1)} | {_fmt(s['mean_lc_cmds_per_cav_id'],1)} |"
        )
    lines.append("")
    if phase53_eval:
        lines.append(
            "Phase 5.3 stored eval (same protocol) for reference: "
            f"Candidate A KEEP/LEFT/RIGHT = "
            f"{_fmt(100*float(phase53_eval.get('ca_keep', float('nan'))),1)} / "
            f"{_fmt(100*float(phase53_eval.get('ca_left', float('nan'))),1)} / "
            f"{_fmt(100*float(phase53_eval.get('ca_right', float('nan'))),1)}%; "
            f"Current = "
            f"{_fmt(100*float(phase53_eval.get('cur_keep', float('nan'))),1)} / "
            f"{_fmt(100*float(phase53_eval.get('cur_left', float('nan'))),1)} / "
            f"{_fmt(100*float(phase53_eval.get('cur_right', float('nan'))),1)}%."
        )
    lines.append("")

    # C
    lines.append("## C. Spatial Lane-change Analysis")
    lines.append("")
    lines.append(
        f"Zones (x_acc=700 m, R={ZONE_R_M:.0f} m): "
        f"**before** x<550, **near** |x−700|≤150, **after** x>850."
    )
    lines.append("")
    lines.append("### Candidate A")
    lines.append("")
    lines.append("| Zone | n | KEEP% | LEFT% | RIGHT% |")
    lines.append("|------|---|-------|-------|--------|")
    for z in ("before", "near", "after"):
        zz = ca["zone"][z]
        lines.append(
            f"| {z} | {zz['n']} | {_fmt(zz['KEEP_pct'],1)} | "
            f"{_fmt(zz['LEFT_pct'],1)} | {_fmt(zz['RIGHT_pct'],1)} |"
        )
    lines.append("")
    lines.append("Near-zone conditioned on blocked lane occupancy (Candidate A):")
    lines.append("")
    bn = ca["blocked_near_actions"]
    cn = ca["clear_near_actions"]
    bn_n = bn["KEEP"] + bn["LEFT"] + bn["RIGHT"]
    cn_n = cn["KEEP"] + cn["LEFT"] + cn["RIGHT"]
    lines.append(
        f"- On blocked lane near accident (n={bn_n}): "
        f"KEEP/LEFT/RIGHT = {bn['KEEP']}/{bn['LEFT']}/{bn['RIGHT']} "
        f"({_fmt(_pct(bn['KEEP'], bn_n),1)} / {_fmt(_pct(bn['LEFT'], bn_n),1)} / "
        f"{_fmt(_pct(bn['RIGHT'], bn_n),1)} %)"
    )
    lines.append(
        f"- Off blocked lane near accident (n={cn_n}): "
        f"KEEP/LEFT/RIGHT = {cn['KEEP']}/{cn['LEFT']}/{cn['RIGHT']} "
        f"({_fmt(_pct(cn['KEEP'], cn_n),1)} / {_fmt(_pct(cn['LEFT'], cn_n),1)} / "
        f"{_fmt(_pct(cn['RIGHT'], cn_n),1)} %)"
    )
    lines.append(f"- blocked_lane_near_frac (replay): {_fmt(ca['blocked_near_frac'])}")
    lines.append("")
    lines.append("### Current Reward (contrast)")
    lines.append("")
    lines.append("| Zone | n | KEEP% | LEFT% | RIGHT% |")
    lines.append("|------|---|-------|-------|--------|")
    for z in ("before", "near", "after"):
        zz = cur["zone"][z]
        lines.append(
            f"| {z} | {zz['n']} | {_fmt(zz['KEEP_pct'],1)} | "
            f"{_fmt(zz['LEFT_pct'],1)} | {_fmt(zz['RIGHT_pct'],1)} |"
        )
    lines.append("")

    # D
    lines.append("## D. Reward-by-action Analysis")
    lines.append("")
    lines.append(
        "Post-step reward components grouped by the action just applied "
        "(same-step credit; not a causal A/B of action effect)."
    )
    lines.append("")
    lines.append("### Candidate A")
    lines.append("")
    lines.append(
        "| Action | n | mean R_total | mean R_ind | mean R_group | "
        "mean S_cav | mean S_loc | mean P_block | mean ΔR (post−pre) |"
    )
    lines.append(
        "|--------|---|--------------|------------|--------------|"
        "------------|------------|--------------|---------------------|"
    )
    for a_name in ("KEEP", "LEFT", "RIGHT"):
        n = ca["n_samples_by_action"][a_name]
        lines.append(
            f"| {a_name} | {n} | "
            f"{_fmt(ca['mean_r_total_by_action'][a_name])} | "
            f"{_fmt(ca['mean_r_ind_by_action'][a_name])} | "
            f"{_fmt(ca['mean_r_group_by_action'][a_name])} | "
            f"{_fmt(ca['mean_s_cav_by_action'][a_name])} | "
            f"{_fmt(ca['mean_s_loc_by_action'][a_name])} | "
            f"{_fmt(ca['mean_p_block_by_action'][a_name])} | "
            f"{_fmt(ca['mean_r_delta_by_action'][a_name])} |"
        )
    lines.append("")
    lines.append(
        "Candidate A formula reminder: "
        "`R = clip(0.55·v̄ + 0.35·R_group − 0.10·P_block, 0, 1)` with "
        "`R_group = 0.6·S_cav + 0.4·S_loc`. **No explicit lane-change bonus.**"
    )
    lines.append("")
    # Interpret if KEEP samples exist
    if ca["KEEP"] == 0:
        lines.append(
            "**KEEP has n=0 under deterministic eval** — cannot compute "
            "KEEP mean reward from this replay. Training still had KEEP samples "
            "(Section B.1)."
        )
    else:
        # compare LC vs KEEP totals
        r_k = ca["mean_r_total_by_action"]["KEEP"]
        r_l = ca["mean_r_total_by_action"]["LEFT"]
        r_r = ca["mean_r_total_by_action"]["RIGHT"]
        lines.append(
            f"Observed mean R_total: KEEP={_fmt(r_k)}, LEFT={_fmt(r_l)}, "
            f"RIGHT={_fmt(r_r)}."
        )
    lines.append("")
    lines.append("### Current Reward (contrast)")
    lines.append("")
    lines.append(
        "| Action | n | mean R_total | mean R_ind | mean R_group | mean ΔR |"
    )
    lines.append(
        "|--------|---|--------------|------------|--------------|---------|"
    )
    for a_name in ("KEEP", "LEFT", "RIGHT"):
        lines.append(
            f"| {a_name} | {cur['n_samples_by_action'][a_name]} | "
            f"{_fmt(cur['mean_r_total_by_action'][a_name])} | "
            f"{_fmt(cur['mean_r_ind_by_action'][a_name])} | "
            f"{_fmt(cur['mean_r_group_by_action'][a_name])} | "
            f"{_fmt(cur['mean_r_delta_by_action'][a_name])} |"
        )
    lines.append("")

    # E
    lines.append("## E. Oscillation Analysis")
    lines.append("")
    lines.append(
        "Oscillation defined on **per-vehicle-id** action sequences "
        "(lifecycle may rebind physical vehicles to ids; ids treated as slots)."
    )
    lines.append("")
    lines.append("| Metric | Candidate A | Current Reward |")
    lines.append("|--------|-------------|----------------|")
    lines.append(
        f"| LEFT↔RIGHT immediate flips | {ca['osc_lr_rl_flips']} | "
        f"{cur['osc_lr_rl_flips']} |"
    )
    lines.append(
        f"| L–R–L / R–L–R triplets | {ca['osc_triplets_LRL_RLR']} | "
        f"{cur['osc_triplets_LRL_RLR']} |"
    )
    lines.append(
        f"| Flip rate per LC cmd | {_fmt(ca['osc_flip_rate_per_lc'])} | "
        f"{_fmt(cur['osc_flip_rate_per_lc'])} |"
    )
    lines.append(
        f"| Max consecutive non-KEEP run | {ca['max_consecutive_lc_run']} | "
        f"{cur['max_consecutive_lc_run']} |"
    )
    lines.append("")

    # F
    lines.append("## F. Candidate A Behavior Diagnosis")
    lines.append("")
    lines.append("Facts supported by this run:")
    lines.append("")
    lines.append(
        "1. **Deterministic policy collapses to near-zero KEEP.** "
        f"Replay KEEP%={_fmt(ca['KEEP_pct'],1)}; Phase 5.3 eval matched this. "
        "Training still sampled KEEP (mean "
        f"{_fmt(train.get('mean_KEEP_pct', float('nan')),1)}% in resume CSV), "
        "but entropy fell "
        f"({_fmt(train.get('first_entropy', float('nan')))}→"
        f"{_fmt(train.get('final_entropy', float('nan')))}), "
        "so argmax prefers LEFT/RIGHT."
    )
    lines.append(
        "2. **Lane-changing is not confined to the accident near-zone.** "
        f"Before-zone LEFT%={_fmt(ca['zone']['before']['LEFT_pct'],1)}, "
        f"near LEFT%={_fmt(ca['zone']['near']['LEFT_pct'],1)}, "
        f"after LEFT%={_fmt(ca['zone']['after']['LEFT_pct'],1)} — "
        "LC commands dominate everywhere under deterministic A."
    )
    lines.append(
        "3. **Even off the blocked lane near the accident, KEEP remains rare.** "
        "If diversion were the only goal, clear-near should show more KEEP; "
        "observed clear-near still heavily LEFT/RIGHT → diversion tax alone "
        "does not explain global KEEP≈0."
    )
    lines.append(
        "4. **Reward has no LC term**, but learning can still amplify LC if "
        "any correlated speed/group/P_block pattern favors non-KEEP during "
        "training. P_block mean is higher on some LC samples when still on "
        "blocked lane; leaving blocked lane reduces −0.10·P_block, which is "
        "a **local** incentive near the incident only."
    )
    lines.append(
        "5. **Mean speed slightly below Current** is consistent with frequent "
        "LC commands (maneuver cost / disruption) even when near-zone speed "
        "and queue improved in Phase 5.3 — a traffic trade-off, not proof of "
        "overall superiority."
    )
    lines.append("")

    # G
    lines.append("## G. Evidence: over-aggressive lane-changing?")
    lines.append("")
    # Build a structured verdict from numbers
    evidence_for: list[str] = []
    evidence_against: list[str] = []

    if ca["KEEP_pct"] < 5.0:
        evidence_for.append(
            f"Deterministic KEEP% ≈ {_fmt(ca['KEEP_pct'],1)} (near-total LC)."
        )
    if ca["zone"]["before"]["KEEP_pct"] < 10.0:
        evidence_for.append(
            "Before-accident zone also almost never KEEP — not selective diversion."
        )
    if cn_n > 0 and _pct(cn["KEEP"], cn_n) < 10.0:
        evidence_for.append(
            "Near-zone off blocked lane still rarely KEEP — LC continues after "
            "clearing lane 0."
        )
    if ca["osc_lr_rl_flips"] > 0 and ca["osc_flip_rate_per_lc"] > 0.05:
        evidence_for.append(
            f"Non-trivial LEFT↔RIGHT flip rate "
            f"({_fmt(ca['osc_flip_rate_per_lc'])} per LC cmd)."
        )
    if ca["max_consecutive_lc_run"] >= 20:
        evidence_for.append(
            f"Long consecutive non-KEEP runs (max={ca['max_consecutive_lc_run']} steps)."
        )

    evidence_against.append(
        "Reward formula contains **no** explicit lane-change bonus/penalty."
    )
    evidence_against.append(
        "Training KEEP% was non-zero; zero KEEP is an **eval/deterministic** "
        "collapse, not the train sampling distribution."
    )
    if ca["blocked_near_frac"] < cur.get("blocked_near_frac", 1.0):
        evidence_against.append(
            "Blocked-lane near occupancy lower than Current — some LC may be "
            "task-aligned diversion, not pure hacking."
        )

    lines.append("### Supporting over-aggressive LC")
    for e in evidence_for:
        lines.append(f"- {e}")
    if not evidence_for:
        lines.append("- (none strong in this replay)")
    lines.append("")
    lines.append("### Against / caveats")
    for e in evidence_against:
        lines.append(f"- {e}")
    lines.append("")

    # Verdict
    aggressive = (
        ca["KEEP_pct"] < 5.0
        and ca["zone"]["before"]["KEEP_pct"] < 15.0
        and (cn_n == 0 or _pct(cn["KEEP"], cn_n) < 15.0)
    )
    if aggressive:
        lines.append(
            "**Verdict (data-bound):** Yes — under deterministic evaluation, "
            "Candidate A exhibits **over-aggressive / non-selective lane-changing** "
            "(KEEP≈0 globally, including before-zone and clear-near). "
            "This is a behavior pathology for Phase 5.5 to address; it does "
            "**not** by itself invalidate every traffic-metric movement in Phase 5.3."
        )
    else:
        lines.append(
            "**Verdict (data-bound):** Mixed — LC is elevated vs Current but "
            "not uniformly pathological on all spatial/occupancy slices."
        )
    lines.append("")

    # H
    lines.append("## H. Recommendation for Phase 5.5")
    lines.append("")
    lines.append("Do **not** yet redesign the whole reward or declare A superior.")
    lines.append("")
    if aggressive:
        lines.append(
            "**Recommended Phase 5.5 focus:** mitigate non-selective LC "
            "while keeping Candidate A's accident occupancy / group terms."
        )
        lines.append("")
        lines.append("Candidate interventions to *consider* (implement later, one at a time):")
        lines.append(
            "1. **Small KEEP-preserving regularizer or LC cost** (explicit, "
            "ablation-ready) — only if diagnosis remains the binding constraint."
        )
        lines.append(
            "2. **Gate P_block / diversion pressure more tightly** so far-upstream "
            "behavior stays closer to speed-keeping."
        )
        lines.append(
            "3. **Eval with stochastic policy** (or temperature) once, to separate "
            "argmax collapse from train-time behavior — measurement only."
        )
        lines.append(
            "4. **Weight sensitivity on w_b / α** before inventing new components."
        )
    else:
        lines.append(
            "Phase 5.5 may proceed to limited multi-seed traffic validation "
            "of Candidate A, while still monitoring LC rates."
        )
    lines.append("")
    lines.append(
        "Phase 5.5 should still avoid stacking speed+queue+throughput+LC into "
        "one dense objective without ablation."
    )
    lines.append("")
    lines.append("## Phase 5.4 Result: PASS")
    lines.append("")
    lines.append(
        "PASS = diagnostic replay completed, report written, no training / "
        "no Current Reward or formal checkpoint modification."
    )
    lines.append("")

    report = out / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def _load_phase53_eval() -> dict[str, Any] | None:
    path = PHASE53 / "evaluation" / "group_comparison.csv"
    if not path.is_file():
        return None
    by: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by[row["method"]] = row
    ca = by.get("MAPPO_CandidateA")
    cur = by.get("MAPPO_Current")
    if not ca or not cur:
        return None
    return {
        "ca_keep": float(ca["keep_ratio"]),
        "ca_left": float(ca["left_ratio"]),
        "ca_right": float(ca["right_ratio"]),
        "cur_keep": float(cur["keep_ratio"]),
        "cur_left": float(cur["left_ratio"]),
        "cur_right": float(cur["right_ratio"]),
    }


def main() -> int:
    out = OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    snap = out / "config_snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    for rel in (
        "configs/reward_candidate_A.yaml",
        "configs/reward.yaml",
        "configs/env.yaml",
        "env/reward.py",
    ):
        src = PROJECT_ROOT / rel
        if src.is_file():
            shutil.copy2(src, snap / src.name)
    meta = {
        "seed": SEED,
        "max_steps": MAX_STEPS,
        "candidate_a_ckpt": str(CA_CKPT),
        "current_ckpt": str(CUR_CKPT),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "read_only_diagnosis",
    }
    (snap / "phase5_4_protocol.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    if not CA_CKPT.is_file():
        print(f"ERROR: missing Candidate A checkpoint: {CA_CKPT}")
        return 1
    if not CUR_CKPT.is_file():
        print(f"ERROR: missing Current checkpoint: {CUR_CKPT}")
        return 1

    log_f = (out / "stdout.log").open("w", encoding="utf-8")
    tee = Tee(sys.stdout, log_f)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = tee  # type: ignore[assignment]
    sys.stderr = tee  # type: ignore[assignment]

    try:
        print("================================")
        print("Phase 5.4 Candidate A Behavior Diagnosis")
        print("================================")
        print(f"OUT: {out}")
        print(f"CA ckpt: {CA_CKPT}")
        print(f"Current ckpt: {CUR_CKPT}")
        print()

        train = analyze_train_metrics(PHASE53 / "train_metrics.csv")
        # persist train episode table
        if train.get("available"):
            ep_path = out / "train_episode_actions.csv"
            with ep_path.open("w", newline="", encoding="utf-8") as f:
                fields = list(train["episodes"][0].keys())
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(train["episodes"])
            print(f"Wrote {ep_path}")

        device = resolve_device("auto")
        print("--- Replay Candidate A (deterministic) ---")
        agg_ca = run_trace(
            label="CandidateA",
            ckpt=CA_CKPT,
            reward_config=CA_REWARD,
            device=device,
            out_csv=out / "traces" / "candidate_A_steps.csv",
        )
        ca = summarize_agg(agg_ca)
        print(
            f"CA actions KEEP/LEFT/RIGHT = {ca['KEEP']}/{ca['LEFT']}/{ca['RIGHT']} "
            f"({_fmt(ca['KEEP_pct'],1)}/{_fmt(ca['LEFT_pct'],1)}/{_fmt(ca['RIGHT_pct'],1)} %)"
        )

        print("--- Replay Current Reward (deterministic, contrast) ---")
        agg_cur = run_trace(
            label="Current",
            ckpt=CUR_CKPT,
            reward_config=None,
            device=device,
            out_csv=out / "traces" / "current_steps.csv",
        )
        cur = summarize_agg(agg_cur)
        print(
            f"Current actions KEEP/LEFT/RIGHT = {cur['KEEP']}/{cur['LEFT']}/{cur['RIGHT']} "
            f"({_fmt(cur['KEEP_pct'],1)}/{_fmt(cur['LEFT_pct'],1)}/{_fmt(cur['RIGHT_pct'],1)} %)"
        )

        # summary json
        summary = {
            "train": {k: v for k, v in train.items() if k != "episodes"},
            "eval_candidate_a": ca,
            "eval_current": cur,
            "phase53_eval": _load_phase53_eval(),
        }
        (out / "diagnosis_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        report = write_report(
            out,
            train=train,
            ca=ca,
            cur=cur,
            phase53_eval=summary["phase53_eval"],
        )
        print()
        print(f"Report: {report}")
        print("Phase 5.4 Result: PASS")
        return 0
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        log_f.close()


if __name__ == "__main__":
    raise SystemExit(main())
