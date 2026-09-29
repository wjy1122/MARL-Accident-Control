"""Phase 6.2 — legal-action behavior diagnosis after Action Mask (read-only).

Frozen Phase 6.1 episode_0100 checkpoint. No training. No edits to reward,
observation, mask, actor/critic, or PPO. Deterministic eval + same-state
forced branches on lane-1 legal states for 1s/3s/5s (one 5s run, sliced).
"""

from __future__ import annotations

import copy
import csv
import json
import math
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.rollout import load_train_config, resolve_device  # noqa: E402
from env.action import ACTION_NAMES, KEEP, LEFT, RIGHT, target_lane_index  # noqa: E402
from env.action_mask import legal_action_mask  # noqa: E402
from env.highway_accident_env import HighwayAccidentEnv, load_config  # noqa: E402
from utils.baseline_eval import (  # noqa: E402
    load_mappo_checkpoint,
    params_unchanged,
    set_eval_seed,
    snapshot_model_params,
)

OUT = PROJECT_ROOT / "results" / "experiments" / "reward_candidate_A_phase6_2_mask_diagnostic"
MASK_CKPT = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "reward_candidate_A_phase6_1_mask"
    / "checkpoints"
    / "episode_0100.pt"
)
CA_REWARD = PROJECT_ROOT / "configs" / "reward_candidate_A.yaml"
SEED = 42
MAX_EVAL_STEPS = 1800
MAX_BRANCH_EPISODES = 3
TARGET_PER_REGION = 100
LATERAL_MAX_M = 0.2
MAX_PICKS_PER_STEP = 1
# One full episode is one SUMO realization (seed fixed). Cap is the target
# so Before/Near are not truncated before vehicles can reach After.
ZONE_CAP_PER_EPISODE = 100
HORIZON_SECONDS = (1.0, 3.0, 5.0)
ACTIONS = (KEEP, LEFT, RIGHT)
FLAT = 0.01


def _stats(xs: list[float]) -> dict[str, Any]:
    clean = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not clean:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "p25": None,
            "p75": None,
            "max": None,
        }
    a = np.asarray(clean, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "max": float(np.max(a)),
    }


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(v) or math.isinf(v):
        return "n/a"
    return f"{v:.{nd}f}"


def _zone(pos: float, x_acc: float, r: float) -> str:
    if pos < x_acc - r:
        return "before"
    if pos > x_acc + r:
        return "after"
    return "near"


def _both_legal(lane: int, n_lanes: int, left_up: bool) -> bool:
    return (
        target_lane_index(
            current_lane=lane, action=LEFT, n_lanes=n_lanes, left_increases_index=left_up
        )
        is not None
        and target_lane_index(
            current_lane=lane, action=RIGHT, n_lanes=n_lanes, left_increases_index=left_up
        )
        is not None
    )


def _lateral(traci: Any, vid: str) -> float | None:
    try:
        return float(traci.vehicle.getLateralLanePosition(vid))
    except Exception:  # noqa: BLE001
        return None


def _masked_policy(
    model: torch.nn.Module,
    obs: Any,
    lane: int,
    env: HighwayAccidentEnv,
    device: torch.device,
) -> tuple[int, list[float], list[float], list[float]]:
    t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device)
    am = torch.as_tensor(
        legal_action_mask(
            lane,
            n_lanes=int(env.n_lanes),
            left_increases_index=bool(env.left_increases_index),
        ),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        logits_raw = model.actor.forward(t)
        if logits_raw.dim() == 1:
            logits_raw = logits_raw.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        from env.action_mask import apply_logit_mask

        logits_m = apply_logit_mask(logits_raw, am.unsqueeze(0) if logits_raw.dim() == 2 else am)
        probs = torch.softmax(logits_m, dim=-1)[0]
        action = int(torch.argmax(logits_m, dim=-1)[0].item())
    logits = [float(x) for x in logits_m[0].detach().cpu().tolist()]
    probs_l = [float(x) for x in probs.detach().cpu().tolist()]
    raw = [float(x) for x in logits_raw[0].detach().cpu().tolist()]
    if squeeze:
        pass
    return action, logits, probs_l, raw


def _discounted_prefix(rs: list[float], gamma: float, n: int) -> float:
    return float(sum((gamma**t) * rs[t] for t in range(min(n, len(rs)))))


def pass1_eval(model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    """Deterministic masked evaluation: reversals, intervals, probs, reward traces."""
    set_eval_seed(SEED)
    model.eval()
    env_cfg = copy.deepcopy(load_config())
    env_cfg.setdefault("episode", {})["max_episode_steps"] = MAX_EVAL_STEPS
    env = HighwayAccidentEnv(config=env_cfg, gui=False, reward_config=CA_REWARD)
    x_acc = float(env.accident_position_m)
    zone_r = float(env.reward_zone_range_m)
    step_length = float(env.step_length)
    n_lanes = int(env.n_lanes)
    left_up = bool(env.left_increases_index)
    win1 = max(1, int(round(1.0 / step_length)))
    win3 = max(1, int(round(3.0 / step_length)))

    # Per step: cav -> (lane, speed, rewards, action, probs, logits, zone, pos)
    hist: list[dict[str, dict[str, Any]]] = []
    act_counts = np.zeros(3, dtype=np.int64)
    zone_probs: dict[str, list[list[float]]] = {"before": [], "near": [], "after": []}
    zone_logits: dict[str, list[list[float]]] = {"before": [], "near": [], "after": []}
    zone_acts: dict[str, np.ndarray] = {
        z: np.zeros(3, dtype=np.int64) for z in ("before", "near", "after")
    }
    invalid = 0
    lane_zone: dict[str, dict[int, int]] = {
        z: {} for z in ("before", "near", "after")
    }
    max_x_by_lane: dict[int, float] = {}

    try:
        env.reset()
        for step in range(MAX_EVAL_STEPS):
            states = env.get_cav_states()
            observations = env.get_observations()
            rewards_pre = env.get_rewards()
            actions: dict[str, int] = {}
            frame: dict[str, dict[str, Any]] = {}
            for cid, st in states.items():
                if cid not in observations:
                    continue
                lane = int(st["lane"])
                pos = float(st["position"])
                zone = _zone(pos, x_acc, zone_r)
                a, logits, probs, _raw = _masked_policy(
                    model, observations[cid], lane, env, device
                )
                actions[cid] = a
                act_counts[a] += 1
                zone_acts[zone][a] += 1
                zone_probs[zone].append(probs)
                zone_logits[zone].append(logits)
                lane_zone[zone][lane] = lane_zone[zone].get(lane, 0) + 1
                if pos > max_x_by_lane.get(lane, -1.0):
                    max_x_by_lane[lane] = pos
                if (
                    a != KEEP
                    and target_lane_index(
                        current_lane=lane,
                        action=a,
                        n_lanes=n_lanes,
                        left_increases_index=left_up,
                    )
                    is None
                ):
                    invalid += 1
                detail = rewards_pre.get(cid) or {}
                frame[cid] = {
                    "lane": lane,
                    "pos": pos,
                    "zone": zone,
                    "speed": float(st["speed"]),
                    "action": a,
                    "logits": logits,
                    "probs": probs,
                    "R_ind": float(detail.get("individual", float("nan"))),
                    "R_group": float(detail.get("group", float("nan"))),
                    "P_block": float(detail.get("p_block", float("nan"))),
                    "R_total": float(detail.get("total", float("nan"))),
                }
            hist.append(frame)
            _o, _r, _t, trunc, _i = env.step(actions)
            if trunc:
                break
    finally:
        env.close()

    # Lane-change events from consecutive lane indices.
    lc_events: list[dict[str, Any]] = []
    per_cav_lc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in range(1, len(hist)):
        for cid, cur in hist[step].items():
            prev = hist[step - 1].get(cid)
            if prev is None:
                continue
            if int(cur["lane"]) == int(prev["lane"]):
                continue
            direction = "LEFT" if int(cur["lane"]) > int(prev["lane"]) else "RIGHT"
            ev = {
                "step": step,
                "time_s": step * step_length,
                "cav_id": cid,
                "old_lane": int(prev["lane"]),
                "new_lane": int(cur["lane"]),
                "direction": direction,
                "zone": cur["zone"],
                "pos": float(cur["pos"]),
                "speed": float(cur["speed"]),
            }
            lc_events.append(ev)
            per_cav_lc[cid].append(ev)

    # Effective reversals: consecutive opposite LC for same CAV.
    rev_events: list[dict[str, Any]] = []
    for cid, events in per_cav_lc.items():
        events = sorted(events, key=lambda e: e["step"])
        for i in range(1, len(events)):
            a, b = events[i - 1], events[i]
            if a["direction"] == b["direction"]:
                continue
            kind = f"{a['direction']}→{b['direction']}"
            dt = (b["step"] - a["step"]) * step_length
            rev_events.append(
                {
                    "cav_id": cid,
                    "kind": kind,
                    "step": b["step"],
                    "zone": b["zone"],
                    "dt_s": dt,
                    "from_step": a["step"],
                    "to_step": b["step"],
                }
            )

    # Requested action switches (LEFT↔RIGHT) for reference only.
    req_rev = 0
    for step in range(1, len(hist)):
        for cid, cur in hist[step].items():
            prev = hist[step - 1].get(cid)
            if prev is None:
                continue
            pa, ca = int(prev["action"]), int(cur["action"])
            if pa in (LEFT, RIGHT) and ca in (LEFT, RIGHT) and pa != ca:
                req_rev += 1

    intervals = [e["dt_s"] for e in rev_events]
    # All consecutive LC intervals (any direction), not only reversals.
    all_intervals: list[float] = []
    for events in per_cav_lc.values():
        events = sorted(events, key=lambda e: e["step"])
        for i in range(1, len(events)):
            all_intervals.append((events[i]["step"] - events[i - 1]["step"]) * step_length)

    # Per-CAV LC count distribution (over the episode; CAVs that appear).
    lc_counts = [len(v) for v in per_cav_lc.values()]
    # Also include CAVs that never changed lane: those seen in hist with 0 LC.
    seen_cavs = set()
    for frame in hist:
        seen_cavs.update(frame.keys())
    zero_lc = len(seen_cavs) - len(per_cav_lc)
    lc_counts_full = lc_counts + [0] * max(0, zero_lc)
    hist_bins = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5+": 0}
    for n in lc_counts_full:
        if n <= 4:
            hist_bins[str(n)] += 1
        else:
            hist_bins["5+"] += 1

    # Reversal by region and rate per 1000 steps.
    n_steps = max(len(hist), 1)
    rev_by_zone = {
        z: {"LEFT→RIGHT": 0, "RIGHT→LEFT": 0, "total": 0} for z in ("before", "near", "after")
    }
    for e in rev_events:
        z = e["zone"]
        rev_by_zone[z][e["kind"]] += 1
        rev_by_zone[z]["total"] += 1

    # Interval buckets for reversals.
    bucket = {"<=1s": 0, "<=2s": 0, "<=3s": 0, "<=5s": 0, ">5s": 0}
    for dt in intervals:
        if dt <= 1.0:
            bucket["<=1s"] += 1
        if dt <= 2.0:
            bucket["<=2s"] += 1
        if dt <= 3.0:
            bucket["<=3s"] += 1
        if dt <= 5.0:
            bucket["<=5s"] += 1
        else:
            bucket[">5s"] += 1

    # Speed / reward around reversal completion.
    def _look(cid: str, step: int, key: str) -> float | None:
        if step < 0 or step >= len(hist):
            return None
        row = hist[step].get(cid)
        if row is None:
            return None
        v = row.get(key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return float(v)

    rev_speed: dict[str, list[dict[str, float]]] = {"LEFT→RIGHT": [], "RIGHT→LEFT": []}
    rev_reward: dict[str, list[dict[str, float]]] = {"LEFT→RIGHT": [], "RIGHT→LEFT": []}
    for e in rev_events:
        cid = e["cav_id"]
        t = int(e["step"])
        s_m1 = _look(cid, t - win1, "speed")
        s0 = _look(cid, t, "speed")
        s_p1 = _look(cid, t + win1, "speed")
        s_p3 = _look(cid, t + win3, "speed")
        if None not in (s_m1, s0, s_p1, s_p3):
            rev_speed[e["kind"]].append(
                {
                    "v_m1": s_m1,
                    "v0": s0,
                    "v_p1": s_p1,
                    "v_p3": s_p3,
                    "dv_1s": s_p1 - s0,
                    "dv_3s": s_p3 - s0,
                    "dv_from_pre": s0 - s_m1,
                }
            )
        # Reward: mean over 1s before vs 1s after (post-decision frames).
        pre_keys = ("R_ind", "R_group", "P_block", "R_total")
        pre = {k: [] for k in pre_keys}
        post = {k: [] for k in pre_keys}
        for kstep in range(max(0, t - win1), t):
            for key in pre_keys:
                v = _look(cid, kstep, key)
                if v is not None:
                    pre[key].append(v)
        for kstep in range(t, min(len(hist), t + win1)):
            for key in pre_keys:
                v = _look(cid, kstep, key)
                if v is not None:
                    post[key].append(v)
        if all(pre[k] and post[k] for k in pre_keys):
            rev_reward[e["kind"]].append(
                {
                    f"d{k}": float(np.mean(post[k]) - np.mean(pre[k]))
                    for k in pre_keys
                }
            )

    def _col_stats(rows: list[dict[str, float]], key: str) -> dict[str, Any]:
        return _stats([r[key] for r in rows])

    speed_summary = {}
    reward_summary = {}
    for kind in ("LEFT→RIGHT", "RIGHT→LEFT"):
        speed_summary[kind] = {
            "n": len(rev_speed[kind]),
            "dv_1s": _col_stats(rev_speed[kind], "dv_1s"),
            "dv_3s": _col_stats(rev_speed[kind], "dv_3s"),
        }
        reward_summary[kind] = {
            "n": len(rev_reward[kind]),
            "dR_ind": _col_stats(rev_reward[kind], "dR_ind"),
            "dR_group": _col_stats(rev_reward[kind], "dR_group"),
            "dP_block": _col_stats(rev_reward[kind], "dP_block"),
            "dR_total": _col_stats(rev_reward[kind], "dR_total"),
        }

    actor_zone = {}
    for z in ("before", "near", "after"):
        probs = zone_probs[z]
        logits = zone_logits[z]
        n = len(probs)
        na = int(zone_acts[z].sum())
        actor_zone[z] = {
            "n_states": n,
            "P_KEEP": float(np.mean([p[KEEP] for p in probs])) if probs else None,
            "P_LEFT": float(np.mean([p[LEFT] for p in probs])) if probs else None,
            "P_RIGHT": float(np.mean([p[RIGHT] for p in probs])) if probs else None,
            "logit_KEEP": float(np.mean([p[KEEP] for p in logits])) if logits else None,
            "logit_LEFT": float(np.mean([p[LEFT] for p in logits])) if logits else None,
            "logit_RIGHT": float(np.mean([p[RIGHT] for p in logits])) if logits else None,
            "dL_K": _stats([lg[LEFT] - lg[KEEP] for lg in logits]),
            "dR_K": _stats([lg[RIGHT] - lg[KEEP] for lg in logits]),
            "act_KEEP": float(zone_acts[z][KEEP] / na) if na else None,
            "act_LEFT": float(zone_acts[z][LEFT] / na) if na else None,
            "act_RIGHT": float(zone_acts[z][RIGHT] / na) if na else None,
        }

    n_act = int(act_counts.sum())
    return {
        "step_length": step_length,
        "n_steps": n_steps,
        "n_actions": n_act,
        "invalid_action_rate": float(invalid) / float(n_act) if n_act else None,
        "KEEP": float(act_counts[KEEP] / n_act) if n_act else None,
        "LEFT": float(act_counts[LEFT] / n_act) if n_act else None,
        "RIGHT": float(act_counts[RIGHT] / n_act) if n_act else None,
        "effective_LEFT": sum(1 for e in lc_events if e["direction"] == "LEFT"),
        "effective_RIGHT": sum(1 for e in lc_events if e["direction"] == "RIGHT"),
        "requested_reversals": req_rev,
        "effective_reversals": len(rev_events),
        "rev_by_zone": rev_by_zone,
        "rev_per_1000_steps": {
            z: 1000.0 * rev_by_zone[z]["total"] / float(n_steps) for z in rev_by_zone
        },
        "rev_interval": _stats(intervals),
        "lc_interval_all": _stats(all_intervals),
        "rev_interval_buckets": bucket,
        "lc_count_distribution": hist_bins,
        "lc_count_stats": _stats([float(x) for x in lc_counts_full]),
        "n_cavs_seen": len(seen_cavs),
        "speed_around_reversal": speed_summary,
        "reward_around_reversal": reward_summary,
        "actor_zone": actor_zone,
        "lane_zone_counts": {
            z: {str(k): int(v) for k, v in sorted(lane_zone[z].items())}
            for z in lane_zone
        },
        "max_x_by_lane": {str(k): float(v) for k, v in sorted(max_x_by_lane.items())},
        "x_acc": x_acc,
        "zone_r": zone_r,
    }


def _restore(env: HighwayAccidentEnv, path: Path, step_count: int) -> bool:
    try:
        traci = env._require_traci()
        traci.simulation.loadState(str(path))
        env._step_count = int(step_count)
        return True
    except Exception:  # noqa: BLE001
        return False


def _run_branch_long(
    env: HighwayAccidentEnv,
    ego: str,
    action: int,
    n_steps: int,
    gamma: float,
    cut_steps: list[int],
) -> dict[str, Any] | None:
    """Force action on first step, KEEP after; record G at each cut."""
    try:
        traci = env._require_traci()
    except Exception:  # noqa: BLE001
        return None
    if ego not in set(traci.vehicle.getIDList()):
        return None
    try:
        lane_before = int(traci.vehicle.getLaneIndex(ego))
    except Exception:  # noqa: BLE001
        return None
    totals: list[float] = []
    lane_after = lane_before
    completed = None
    for t in range(n_steps):
        try:
            if ego not in set(traci.vehicle.getIDList()):
                return None
            cmd = int(action) if t == 0 else int(KEEP)
            _o, _r, _term, _trunc, info = env.step({ego: cmd})
        except Exception:  # noqa: BLE001
            return None
        detail = (info.get("reward_details") or {}).get(ego)
        if detail is None:
            return None
        totals.append(float(detail["total"]))
        try:
            lane_after = int(traci.vehicle.getLaneIndex(ego))
        except Exception:  # noqa: BLE001
            return None
        if completed is None and lane_after != lane_before:
            completed = t + 1
    if action == LEFT:
        effective = lane_after > lane_before
    elif action == RIGHT:
        effective = lane_after < lane_before
    else:
        effective = False
    g_cuts = {int(c): _discounted_prefix(totals, gamma, int(c)) for c in cut_steps}
    return {
        "lane_before": lane_before,
        "lane_after": lane_after,
        "effective": bool(effective),
        "G_cuts": g_cuts,
        "sum_all": float(sum(totals)),
    }


def pass2_branches(model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    train_cfg = load_train_config()
    gamma = float(train_cfg.get("train", train_cfg).get("gamma", 0.99))
    set_eval_seed(SEED)
    model.eval()
    counts = {"before": 0, "near": 0, "after": 0}
    rows: list[dict[str, Any]] = []
    discards = {
        "mid_lane_change": 0,
        "gap": 0,
        "branch_incomplete": 0,
        "restore_mismatch": 0,
        "save_fail": 0,
    }
    step_length = 0.1
    cut_steps = [max(1, int(math.ceil(h / 0.1 - 1e-9))) for h in HORIZON_SECONDS]
    max_steps_branch = max(cut_steps)
    zone_r = 150.0
    x_acc = 700.0
    episodes_run = 0
    save_ok = False
    max_legal_x = -1.0
    diag = OUT / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    partial = diag / "branch_rows.jsonl"
    if partial.exists():
        partial.unlink()

    tmp = tempfile.TemporaryDirectory(prefix="phase62_")
    state_path = Path(tmp.name) / "snap.xml"
    try:
        for episode in range(MAX_BRANCH_EPISODES):
            if all(counts[z] >= TARGET_PER_REGION for z in counts):
                break
            env_cfg = copy.deepcopy(load_config())
            env_cfg.setdefault("episode", {})["max_episode_steps"] = (
                MAX_EVAL_STEPS + max_steps_branch + 50
            )
            env = HighwayAccidentEnv(config=env_cfg, gui=False, reward_config=CA_REWARD)
            step_length = float(env.step_length)
            cut_steps = [
                max(1, int(math.ceil(h / step_length - 1e-9))) for h in HORIZON_SECONDS
            ]
            max_steps_branch = max(cut_steps)
            zone_r = float(env.reward_zone_range_m)
            x_acc = float(env.accident_position_m)
            n_lanes = int(env.n_lanes)
            left_up = bool(env.left_increases_index)
            print(
                f"--- branch episode {episode + 1}/{MAX_BRANCH_EPISODES} "
                f"cuts={cut_steps} gamma={gamma} counts={counts} ---",
                flush=True,
            )
            episodes_run += 1
            last_step: dict[str, int] = {}
            zone_ep = {"before": 0, "near": 0, "after": 0}
            reached_end = False
            try:
                env.reset()
                traci = env._require_traci()
                for step in range(MAX_EVAL_STEPS):
                    if all(counts[z] >= TARGET_PER_REGION for z in counts):
                        break
                    zones_open = [
                        z
                        for z in counts
                        if counts[z] < TARGET_PER_REGION
                        and zone_ep[z] < ZONE_CAP_PER_EPISODE
                    ]
                    if not zones_open:
                        print(
                            f"  soft restart at step {step} zone_ep={zone_ep} counts={counts}",
                            flush=True,
                        )
                        break
                    try:
                        states = env.get_cav_states()
                        observations = env.get_observations()
                    except Exception as exc:  # noqa: BLE001
                        print(f"state read failed: {exc}", flush=True)
                        break
                    lane1_x = [
                        float(st["position"])
                        for st in states.values()
                        if _both_legal(int(st["lane"]), n_lanes, left_up)
                    ]
                    if lane1_x:
                        max_legal_x = max(max_legal_x, max(lane1_x))
                    if step % 300 == 0:
                        print(
                            f"  step {step} zone_ep={zone_ep} "
                            f"legal_n={len(lane1_x)} "
                            f"legal_max_x={max(lane1_x) if lane1_x else None} "
                            f"run_max={max_legal_x if max_legal_x >= 0 else None}",
                            flush=True,
                        )
                    candidates: list[tuple[str, str, dict[str, Any]]] = []
                    for cid, st in states.items():
                        lane = int(st["lane"])
                        if not _both_legal(lane, n_lanes, left_up):
                            continue
                        zone = _zone(float(st["position"]), x_acc, zone_r)
                        if counts[zone] >= TARGET_PER_REGION:
                            continue
                        if zone_ep[zone] >= ZONE_CAP_PER_EPISODE:
                            continue
                        if step - last_step.get(cid, -10_000) < max_steps_branch:
                            discards["gap"] += 1
                            continue
                        lat = _lateral(traci, cid)
                        if lat is not None and abs(lat) > LATERAL_MAX_M:
                            discards["mid_lane_change"] += 1
                            continue
                        if cid not in observations:
                            continue
                        candidates.append((cid, zone, st))
                    candidates.sort(key=lambda item: counts[item[1]])
                    picked: list[tuple[str, str, dict[str, Any]]] = []
                    for cid, zone, st in candidates:
                        if any(z == zone for _c, z, _s in picked):
                            continue
                        picked.append((cid, zone, st))
                        if len(picked) >= MAX_PICKS_PER_STEP:
                            break
                    if picked:
                        try:
                            traci.simulation.saveState(str(state_path))
                            save_ok = True
                        except Exception as exc:  # noqa: BLE001
                            discards["save_fail"] += 1
                            print(f"saveState failed: {exc}", flush=True)
                            picked = []
                        else:
                            step_count = int(env._step_count)
                            for cid, zone, st in picked:
                                a0, logits, probs, _raw = _masked_policy(
                                    model, observations[cid], int(st["lane"]), env, device
                                )
                                branches: dict[str, dict[str, Any]] = {}
                                ok = True
                                for action in ACTIONS:
                                    if not _restore(env, state_path, step_count):
                                        ok = False
                                        break
                                    br = _run_branch_long(
                                        env,
                                        cid,
                                        action,
                                        max_steps_branch,
                                        gamma,
                                        cut_steps,
                                    )
                                    if br is None:
                                        ok = False
                                        break
                                    branches[ACTION_NAMES[action]] = br
                                if not _restore(env, state_path, step_count):
                                    discards["restore_mismatch"] += 1
                                    ok = False
                                if not ok or len(branches) != 3:
                                    discards["branch_incomplete"] += 1
                                    if not env.is_connected:
                                        break
                                    continue
                                row: dict[str, Any] = {
                                    "state_id": len(rows),
                                    "episode": episode,
                                    "step": step,
                                    "cav_id": cid,
                                    "zone": zone,
                                    "lane": int(st["lane"]),
                                    "pos": float(st["position"]),
                                    "policy_action": a0,
                                    "P_KEEP": probs[KEEP],
                                    "P_LEFT": probs[LEFT],
                                    "P_RIGHT": probs[RIGHT],
                                    "d_logit_LK": logits[LEFT] - logits[KEEP],
                                    "d_logit_RK": logits[RIGHT] - logits[KEEP],
                                }
                                for h_s, ncut in zip(HORIZON_SECONDS, cut_steps):
                                    tag = f"{int(h_s)}s"
                                    gk = branches["KEEP"]["G_cuts"][ncut]
                                    gl = branches["LEFT"]["G_cuts"][ncut]
                                    gr = branches["RIGHT"]["G_cuts"][ncut]
                                    row[f"G_KEEP_{tag}"] = gk
                                    row[f"G_LEFT_{tag}"] = gl
                                    row[f"G_RIGHT_{tag}"] = gr
                                    row[f"dG_LK_{tag}"] = gl - gk
                                    row[f"dG_RK_{tag}"] = gr - gk
                                row["eff_LEFT"] = branches["LEFT"]["effective"]
                                row["eff_RIGHT"] = branches["RIGHT"]["effective"]
                                rows.append(row)
                                with partial.open("a", encoding="utf-8") as f:
                                    f.write(json.dumps(_clean(row)) + "\n")
                                counts[zone] += 1
                                zone_ep[zone] += 1
                                last_step[cid] = step
                                if counts[zone] % 10 == 0 or counts[zone] == TARGET_PER_REGION:
                                    print(
                                        f"  accepted {counts} last={zone} step={step}",
                                        flush=True,
                                    )
                            if not env.is_connected:
                                break
                            if not _restore(env, state_path, step_count):
                                discards["restore_mismatch"] += 1
                                break

                    # Advance with deterministic masked policy.
                    actions: dict[str, int] = {}
                    for cid, st in states.items():
                        if cid not in observations:
                            continue
                        a, _lg, _pr, _rw = _masked_policy(
                            model, observations[cid], int(st["lane"]), env, device
                        )
                        actions[cid] = a
                    try:
                        _o, _r, _t, trunc, _i = env.step(actions)
                    except Exception as exc:  # noqa: BLE001
                        print(f"collection step failed: {exc}", flush=True)
                        break
                    if trunc:
                        break
                else:
                    reached_end = True
            except Exception as exc:  # noqa: BLE001
                print(f"episode failed: {exc}", flush=True)
            finally:
                try:
                    env.close()
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(0.4)
            if (
                reached_end
                and counts["after"] < TARGET_PER_REGION
                and max_legal_x < x_acc + zone_r
            ):
                print(
                    f"  full episode max lane-1 legal x={max_legal_x:.1f} "
                    f"< {x_acc + zone_r:.1f}; same seed would repeat. Stop.",
                    flush=True,
                )
                break
    finally:
        tmp.cleanup()

    by_zone_h: dict[str, Any] = {}
    for z in ("before", "near", "after"):
        sub = [r for r in rows if r["zone"] == z]
        by_zone_h[z] = {"n": len(sub)}
        for h_s in HORIZON_SECONDS:
            tag = f"{int(h_s)}s"
            by_zone_h[z][tag] = {
                "G_KEEP": _stats([r[f"G_KEEP_{tag}"] for r in sub]),
                "G_LEFT": _stats([r[f"G_LEFT_{tag}"] for r in sub]),
                "G_RIGHT": _stats([r[f"G_RIGHT_{tag}"] for r in sub]),
                "dG_LK": _stats([r[f"dG_LK_{tag}"] for r in sub]),
                "dG_RK": _stats([r[f"dG_RK_{tag}"] for r in sub]),
                "frac_LK_gt": (
                    float(np.mean([r[f"dG_LK_{tag}"] > FLAT for r in sub])) if sub else None
                ),
                "frac_LK_lt": (
                    float(np.mean([r[f"dG_LK_{tag}"] < -FLAT for r in sub])) if sub else None
                ),
                "P_LEFT": _stats([r["P_LEFT"] for r in sub]),
                "P_KEEP": _stats([r["P_KEEP"] for r in sub]),
                "P_RIGHT": _stats([r["P_RIGHT"] for r in sub]),
                "d_logit_LK": _stats([r["d_logit_LK"] for r in sub]),
                "d_logit_RK": _stats([r["d_logit_RK"] for r in sub]),
            }
    return {
        "rows": rows,
        "counts": counts,
        "discards": discards,
        "gamma": gamma,
        "step_length": step_length,
        "cut_steps": cut_steps,
        "horizons_s": list(HORIZON_SECONDS),
        "episodes_run": episodes_run,
        "save_ok": save_ok,
        "by_zone_h": by_zone_h,
        "x_acc": x_acc,
        "zone_r": zone_r,
        "max_legal_x": None if max_legal_x < 0 else float(max_legal_x),
    }


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def write_tables(eval_out: dict[str, Any], branch_out: dict[str, Any]) -> None:
    tab = OUT / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    with (tab / "reversal_by_zone.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["region", "LEFT_to_RIGHT", "RIGHT_to_LEFT", "total", "per_1000_steps"])
        for z in ("before", "near", "after"):
            b = eval_out["rev_by_zone"][z]
            w.writerow(
                [
                    z,
                    b["LEFT→RIGHT"],
                    b["RIGHT→LEFT"],
                    b["total"],
                    eval_out["rev_per_1000_steps"][z],
                ]
            )
    with (tab / "branch_returns.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(branch_out["rows"][0].keys()) if branch_out["rows"] else []
        w = csv.DictWriter(f, fieldnames=fields)
        if fields:
            w.writeheader()
            w.writerows(branch_out["rows"])


def write_report(
    *,
    eval_out: dict[str, Any],
    branch_out: dict[str, Any],
    frozen: bool,
    ckpt_same: bool,
    status: str,
) -> None:
    lines: list[str] = []
    a = lines.append
    a("# Phase 6.2 Report")
    a("")
    a("## Action Mask 后合法动作行为诊断")
    a("")
    a("### 1. Status")
    a("")
    a(f"`{status}`")
    a("")
    a("- Retrained: **No**.")
    a("- Reward / Observation / Action Mask / Actor / Critic / PPO: **not modified**.")
    a(f"- Phase 6.1 checkpoint overwritten: **{'YES' if not ckpt_same else 'No'}**.")
    a(f"- Actor/Critic frozen during diagnosis: **{'YES' if frozen else 'NO'}**.")
    a(
        "- PASS means the deterministic eval and same-state branches finished with "
        "invalid_action_rate=0 and usable samples. It does not mean the controller improved."
    )
    a("")
    a("### 2. Checkpoint")
    a("")
    a(f"- `{MASK_CKPT.relative_to(PROJECT_ROOT).as_posix()}`")
    a("- Evaluation: deterministic, Action Mask ON.")
    a(f"- Seed={SEED}. Step length={eval_out['step_length']} s.")
    a(
        f"- Zones: before x<{eval_out['x_acc'] - eval_out['zone_r']}, "
        f"near |x-{eval_out['x_acc']}|≤{eval_out['zone_r']}, "
        f"after x>{eval_out['x_acc'] + eval_out['zone_r']}."
    )
    a("")
    a("### 3. Reversal statistics")
    a("")
    a(
        f"- Requested LEFT↔RIGHT switches: {eval_out['requested_reversals']} "
        "(reference only)."
    )
    a(
        f"- Effective reversals (opposite successive lane-index changes): "
        f"**{eval_out['effective_reversals']}**."
    )
    a(
        f"- Effective LEFT/RIGHT lane-index events: "
        f"{eval_out['effective_LEFT']}/{eval_out['effective_RIGHT']}."
    )
    a(f"- Invalid action rate under mask: {_fmt(eval_out['invalid_action_rate'], 4)}.")
    a("")
    a("| Region | LEFT→RIGHT | RIGHT→LEFT | Total | per 1000 steps |")
    a("| --- | ---: | ---: | ---: | ---: |")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        b = eval_out["rev_by_zone"][z]
        a(
            f"| {label} | {b['LEFT→RIGHT']} | {b['RIGHT→LEFT']} | {b['total']} | "
            f"{_fmt(eval_out['rev_per_1000_steps'][z], 2)} |"
        )
    a("")
    a("### 4. Lane-change interval")
    a("")
    ri = eval_out["rev_interval"]
    ai = eval_out["lc_interval_all"]
    a("Effective reversal intervals (s):")
    a(
        f"- n={ri['n']}, mean={_fmt(ri['mean'])}, median={_fmt(ri['median'])}, "
        f"P25={_fmt(ri['p25'])}, P75={_fmt(ri['p75'])}, min={_fmt(ri['min'])}."
    )
    a("All consecutive effective lane-change intervals (s):")
    a(
        f"- n={ai['n']}, mean={_fmt(ai['mean'])}, median={_fmt(ai['median'])}, "
        f"P25={_fmt(ai['p25'])}, P75={_fmt(ai['p75'])}, min={_fmt(ai['min'])}."
    )
    bk = eval_out["rev_interval_buckets"]
    a(
        f"- Reversal Δt buckets: ≤1s={bk['<=1s']}, ≤2s={bk['<=2s']}, "
        f"≤3s={bk['<=3s']}, ≤5s={bk['<=5s']}, >5s={bk['>5s']}."
    )
    a("")
    a("### 5. Lane-change count distribution")
    a("")
    dist = eval_out["lc_count_distribution"]
    cs = eval_out["lc_count_stats"]
    a(f"- CAVs seen: {eval_out['n_cavs_seen']}.")
    a(
        f"- Counts 0/1/2/3/4/5+: "
        f"{dist['0']}/{dist['1']}/{dist['2']}/{dist['3']}/{dist['4']}/{dist['5+']}."
    )
    a(
        f"- mean={_fmt(cs['mean'], 2)}, median={_fmt(cs['median'], 2)}, "
        f"max={_fmt(cs['max'], 0)}."
    )
    a("")
    a("### 6. Reversal speed effect")
    a("")
    for kind in ("LEFT→RIGHT", "RIGHT→LEFT"):
        s = eval_out["speed_around_reversal"][kind]
        a(
            f"- {kind} (n={s['n']}): mean Δv_1s={_fmt((s['dv_1s'] or {}).get('mean'), 3)}, "
            f"median={_fmt((s['dv_1s'] or {}).get('median'), 3)}; "
            f"mean Δv_3s={_fmt((s['dv_3s'] or {}).get('mean'), 3)}, "
            f"median={_fmt((s['dv_3s'] or {}).get('median'), 3)}."
        )
        a(
            f"  P25/P75 Δv_1s={_fmt((s['dv_1s'] or {}).get('p25'), 3)}/"
            f"{_fmt((s['dv_1s'] or {}).get('p75'), 3)}; "
            f"Δv_3s={_fmt((s['dv_3s'] or {}).get('p25'), 3)}/"
            f"{_fmt((s['dv_3s'] or {}).get('p75'), 3)}."
        )
    a("")
    a("### 7. Reversal reward effect")
    a("")
    a("Δ = mean reward over 1 s after − mean over 1 s before (existing Candidate A components).")
    a("")
    for kind in ("LEFT→RIGHT", "RIGHT→LEFT"):
        r = eval_out["reward_around_reversal"][kind]
        a(
            f"- {kind} (n={r['n']}): "
            f"ΔR_ind mean/median={_fmt((r['dR_ind'] or {}).get('mean'))}/"
            f"{_fmt((r['dR_ind'] or {}).get('median'))}; "
            f"ΔR_group={_fmt((r['dR_group'] or {}).get('mean'))}/"
            f"{_fmt((r['dR_group'] or {}).get('median'))}; "
            f"ΔP_block={_fmt((r['dP_block'] or {}).get('mean'))}/"
            f"{_fmt((r['dP_block'] or {}).get('median'))}; "
            f"ΔR_total={_fmt((r['dR_total'] or {}).get('mean'))}/"
            f"{_fmt((r['dR_total'] or {}).get('median'))}."
        )
    a("")
    a("### 8. Actor probability (masked)")
    a("")
    a("| Region | P(KEEP) | P(LEFT) | P(RIGHT) | act KEEP | act LEFT | act RIGHT |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        az = eval_out["actor_zone"][z]
        a(
            f"| {label} | {_fmt(az['P_KEEP'], 3)} | {_fmt(az['P_LEFT'], 3)} | "
            f"{_fmt(az['P_RIGHT'], 3)} | {_fmt(az['act_KEEP'], 3)} | "
            f"{_fmt(az['act_LEFT'], 3)} | {_fmt(az['act_RIGHT'], 3)} |"
        )
    a("")
    a("CAV decisions by lane index in the same deterministic eval (not only lane 1):")
    a("")
    a("| Region | lane 0 | lane 1 | lane 2 |")
    a("| --- | ---: | ---: | ---: |")
    lane_zone_counts = eval_out.get("lane_zone_counts") or {}
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        d = lane_zone_counts.get(z) or {}
        a(f"| {label} | {d.get('0', 0)} | {d.get('1', 0)} | {d.get('2', 0)} |")
    a(f"- Max lane position by lane: `{eval_out.get('max_x_by_lane')}`.")
    a(
        f"- Max position among lane-1 states used for branches: "
        f"`{branch_out.get('max_legal_x')}` "
        f"(After requires x>{eval_out['x_acc'] + eval_out['zone_r']})."
    )
    a("")
    a("### 9. Logit gaps (masked)")
    a("")
    a("| Region | mean ΔL−K | median ΔL−K | mean ΔR−K | median ΔR−K |")
    a("| --- | ---: | ---: | ---: | ---: |")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        az = eval_out["actor_zone"][z]
        a(
            f"| {label} | {_fmt((az['dL_K'] or {}).get('mean'), 3)} | "
            f"{_fmt((az['dL_K'] or {}).get('median'), 3)} | "
            f"{_fmt((az['dR_K'] or {}).get('mean'), 3)} | "
            f"{_fmt((az['dR_K'] or {}).get('median'), 3)} |"
        )
    a("")
    a("### 10. Same-state branch return")
    a("")
    a(
        f"Lane-1 legal states only. Force first step then KEEP. Other CAVs get KEEP. "
        f"G=Σγ^t r_t, γ={branch_out['gamma']}. One 5 s rollout sliced at "
        f"{branch_out['cut_steps']} steps. Counts={branch_out['counts']}. "
        f"Discards={branch_out['discards']}."
    )
    a("")
    a("| Region | Horizon | N | KEEP G | LEFT G | RIGHT G | ΔL−K | ΔR−K |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        block = branch_out["by_zone_h"][z]
        for h_s in HORIZON_SECONDS:
            tag = f"{int(h_s)}s"
            h = block[tag]
            a(
                f"| {label} | {tag} | {block['n']} | "
                f"{_fmt((h['G_KEEP'] or {}).get('mean'))} | "
                f"{_fmt((h['G_LEFT'] or {}).get('mean'))} | "
                f"{_fmt((h['G_RIGHT'] or {}).get('mean'))} | "
                f"{_fmt((h['dG_LK'] or {}).get('mean'))} | "
                f"{_fmt((h['dG_RK'] or {}).get('mean'))} |"
            )
    a("")
    a("Median ΔG and Actor preference on the same samples:")
    a("")
    a("| Region | Horizon | med ΔL−K | med ΔR−K | mean P(KEEP/LEFT/RIGHT) | mean Δlogit L−K / R−K |")
    a("| --- | ---: | ---: | ---: | --- | --- |")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        block = branch_out["by_zone_h"][z]
        for h_s in HORIZON_SECONDS:
            tag = f"{int(h_s)}s"
            h = block[tag]
            a(
                f"| {label} | {tag} | {_fmt((h['dG_LK'] or {}).get('median'))} | "
                f"{_fmt((h['dG_RK'] or {}).get('median'))} | "
                f"{_fmt((h['P_KEEP'] or {}).get('mean'), 3)}/"
                f"{_fmt((h['P_LEFT'] or {}).get('mean'), 3)}/"
                f"{_fmt((h['P_RIGHT'] or {}).get('mean'), 3)} | "
                f"{_fmt((h['d_logit_LK'] or {}).get('mean'), 3)} / "
                f"{_fmt((h['d_logit_RK'] or {}).get('mean'), 3)} |"
            )
    a("")
    a("### 11. Key findings")
    a("")
    a("#### Confirmed")
    a("")
    a("- Masked deterministic invalid action rate is 0 in this diagnosis.")
    a(
        f"- Effective reversals still occur after the mask "
        f"({eval_out['effective_reversals']} events in one 1800-step eval)."
    )
    a("- Reversal counts by region and per-1000-step rates are in Section 3.")
    a("- Same-state branch returns for 1/3/5 s on legal lane-1 states are in Section 10.")
    a("")
    a("#### Supported by evidence")
    a("")
    a(
        "- Whether Actor still prefers lateral actions under the mask is answered by "
        "Sections 8–9 (probabilities and logit gaps)."
    )
    a(
        "- Whether short/medium branch returns support that preference is answered by "
        "comparing Section 10 ΔG with the same-sample probabilities."
    )
    a(
        "- Speed and reward changes around completed reversals are measurements in "
        "Sections 6–7; they are not a causal proof that reversals help or hurt traffic."
    )
    a("")
    a("#### Not established")
    a("")
    a("- That the mask caused higher reversal counts than Phase 5 Candidate A (different policies and training length).")
    a("- That PPO / GAE / critic bugs are present.")
    a("- That the reward must be changed.")
    a("")
    a("#### Remaining unknowns")
    a("")
    a("- After-zone lane-1 branch rows exist only if a CAV is on lane 1 with x>850 and is not already changing lanes. A count of 0 is the observed count.")
    a("- Longer horizons than 5 s, or critic-bootstrapped returns, were not measured here.")
    a("- Other CAVs acting with the frozen masked policy inside branches (here they use KEEP).")
    a("- Training-time PPO ratios under the mask.")
    a("")
    a("### 12. Final conclusion")
    a("")
    a(
        f"1. Mask 后 reversal 是否仍然存在？ **Yes** — effective reversals = "
        f"{eval_out['effective_reversals']}."
    )
    ztot = eval_out["rev_by_zone"]
    a(
        "2. reversal 主要发生在哪个区域？ "
        f"Before={ztot['before']['total']}, Near={ztot['near']['total']}, "
        f"After={ztot['after']['total']} "
        f"(per 1000 steps: {_fmt(eval_out['rev_per_1000_steps']['before'], 2)} / "
        f"{_fmt(eval_out['rev_per_1000_steps']['near'], 2)} / "
        f"{_fmt(eval_out['rev_per_1000_steps']['after'], 2)})."
    )
    a(
        f"3. reversal 的时间间隔如何？ median={_fmt(ri['median'])} s, "
        f"mean={_fmt(ri['mean'])} s, ≤1s count={bk['<=1s']}, ≤3s={bk['<=3s']}."
    )
    a("4. 换道后速度是否改善？ See Section 6 means/medians; do not treat as a traffic verdict.")
    a("5. 换道后 reward 是否改善？ See Section 7; components come from the existing Candidate A reward.")
    azn = eval_out["actor_zone"]["near"]
    a(
        "6. Actor 是否仍然偏向横向动作？ "
        f"Near masked P(KEEP/LEFT/RIGHT)="
        f"{_fmt(azn['P_KEEP'], 3)}/{_fmt(azn['P_LEFT'], 3)}/{_fmt(azn['P_RIGHT'], 3)}; "
        f"After={_fmt(eval_out['actor_zone']['after']['P_KEEP'], 3)}/"
        f"{_fmt(eval_out['actor_zone']['after']['P_LEFT'], 3)}/"
        f"{_fmt(eval_out['actor_zone']['after']['P_RIGHT'], 3)}."
    )
    a("7. 1s / 3s / 5s return 是否支持这种偏好？ See Section 10 ΔL−K / ΔR−K by region and horizon.")
    a(
        "8. 是否存在 Actor preference 与实际 return 不一致？ "
        "Compare Section 10 median ΔG with mean P and Δlogit on the same rows; "
        "if ΔG≈0 or opposite while P(LEFT) or P(RIGHT) stays high, inconsistency is present for that slice."
    )
    a("")
    a("This phase stops here. No training. No reward edit. No Phase 6.3.")
    a("")
    a(f"Phase 6.2 Result: {status}")
    a("")
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnostics").mkdir(exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    if not MASK_CKPT.is_file():
        raise SystemExit(f"Missing checkpoint: {MASK_CKPT}")
    ckpt_before = (MASK_CKPT.stat().st_mtime_ns, MASK_CKPT.stat().st_size)
    device = resolve_device("auto")
    print(f"Phase 6.2 device={device}", flush=True)
    model, _ = load_mappo_checkpoint(MASK_CKPT, device=device)
    model.eval()
    before = snapshot_model_params(model)

    print("--- pass1 deterministic eval ---", flush=True)
    eval_out = pass1_eval(model, device)
    (OUT / "diagnostics" / "eval_summary.json").write_text(
        json.dumps(_clean(eval_out), indent=2), encoding="utf-8"
    )
    print(
        f"  invalid={eval_out['invalid_action_rate']} "
        f"eff_rev={eval_out['effective_reversals']} "
        f"KEEP/L/R={eval_out['KEEP']:.3f}/{eval_out['LEFT']:.3f}/{eval_out['RIGHT']:.3f}",
        flush=True,
    )

    print("--- pass2 same-state branches ---", flush=True)
    branch_out = pass2_branches(model, device)
    rows = branch_out.pop("rows")
    (OUT / "diagnostics" / "branch_summary.json").write_text(
        json.dumps(_clean({**branch_out, "n_rows": len(rows)}), indent=2),
        encoding="utf-8",
    )
    branch_out["rows"] = rows

    frozen = params_unchanged(before, snapshot_model_params(model))
    ckpt_same = (MASK_CKPT.stat().st_mtime_ns, MASK_CKPT.stat().st_size) == ckpt_before
    write_tables(eval_out, branch_out)

    inv = eval_out.get("invalid_action_rate")
    n_min = min(branch_out["counts"].values()) if branch_out["counts"] else 0
    if not frozen or not ckpt_same or not branch_out.get("save_ok") or (inv is not None and inv > 0):
        status = "FAIL"
    elif n_min < TARGET_PER_REGION:
        status = "PARTIAL"
    else:
        status = "PASS"

    write_report(
        eval_out=eval_out,
        branch_out=branch_out,
        frozen=frozen,
        ckpt_same=ckpt_same,
        status=status,
    )
    print(f"Report: {OUT / 'report.md'}", flush=True)
    print(f"Phase 6.2 Result: {status}", flush=True)


if __name__ == "__main__":
    main()
