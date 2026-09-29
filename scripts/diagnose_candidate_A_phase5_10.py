"""Phase 5.10 — same-state 1.0 s forced-action branch (read-only).

For states where LEFT and RIGHT are both legal, save one SUMO snapshot and
run KEEP / LEFT / RIGHT for ceil(1.0 / step_length) steps through env.step.
Does not train, does not edit reward / observation / action space / PPO, and
does not write checkpoints.

The recorded quantity is a forced branch return, not a PPO advantage.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import sys
import tempfile
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
from env.highway_accident_env import HighwayAccidentEnv, load_config  # noqa: E402
from utils.baseline_eval import (  # noqa: E402
    _is_near_accident,
    _params,
    _queue_length,
    load_mappo_checkpoint,
    params_unchanged,
    set_eval_seed,
    snapshot_model_params,
)

OUT = PROJECT_ROOT / "results" / "experiments" / "reward_candidate_A_phase5_10"
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
MAX_EPISODES = 12
TARGET_PER_REGION = 100
# Same vehicle is not reused inside one branch horizon on the original clock.
FLAT_RETURN = 0.01
LATERAL_MAX_M = 0.2
# At most one accepted sample per simulation step (reduces saveState load).
MAX_PICKS_PER_STEP = 1
# Soft restart SUMO after this many accepted samples in one episode.
RESTART_AFTER_ACCEPTS = 25
ACTIONS = (KEEP, LEFT, RIGHT)


def _stats(xs: list[float]) -> dict[str, Any]:
    clean = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    if not clean:
        return {"n": 0, "mean": None, "median": None, "std": None}
    a = np.asarray(clean, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "std": float(np.std(a)),
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


def _zone(pos: float, x_acc: float, zone_r: float) -> str:
    if pos < x_acc - zone_r:
        return "before"
    if pos > x_acc + zone_r:
        return "after"
    return "near"


def _both_legal(lane: int, n_lanes: int, left_up: bool) -> bool:
    left = target_lane_index(
        current_lane=lane, action=LEFT, n_lanes=n_lanes, left_increases_index=left_up
    )
    right = target_lane_index(
        current_lane=lane, action=RIGHT, n_lanes=n_lanes, left_increases_index=left_up
    )
    return left is not None and right is not None


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    rx = _rankdata(np.asarray(x, dtype=np.float64)).tolist()
    ry = _rankdata(np.asarray(y, dtype=np.float64)).tolist()
    return _pearson(rx, ry)


def _logits(model: torch.nn.Module, obs: Any, device: torch.device) -> tuple[list[float], list[float]]:
    t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device)
    with torch.no_grad():
        logits_t = model.actor.forward(t)
        probs_t = torch.softmax(logits_t, dim=-1)
    return (
        [float(v) for v in logits_t.detach().cpu().tolist()],
        [float(v) for v in probs_t.detach().cpu().tolist()],
    )


def _lateral(traci: Any, vid: str) -> float | None:
    try:
        return float(traci.vehicle.getLateralLanePosition(vid))
    except Exception:  # noqa: BLE001
        return None


def _traffic(env: HighwayAccidentEnv, p: dict[str, Any], blocked: set[int]) -> dict[str, float]:
    speeds: list[float] = []
    near: list[float] = []
    near_cav = 0
    near_cav_blocked = 0
    for st in env.get_vehicle_states().values():
        if st.get("type") == "ACCIDENT":
            continue
        spd = float(st["speed"])
        pos = float(st["position"])
        speeds.append(spd)
        if _is_near_accident(pos, p):
            near.append(spd)
            if st.get("type") == "CAV":
                near_cav += 1
                if int(st.get("lane", -1)) in blocked:
                    near_cav_blocked += 1
    return {
        "mean_speed": float(np.mean(speeds)) if speeds else float("nan"),
        "near_zone_speed": float(np.mean(near)) if near else float("nan"),
        "queue": float(_queue_length(env, p)),
        "blocked_lane_near_frac": (
            float(near_cav_blocked) / float(near_cav) if near_cav else float("nan")
        ),
    }


def _discounted(rs: list[float], gamma: float) -> float:
    return float(sum((gamma**t) * r for t, r in enumerate(rs)))


def _run_branch(
    env: HighwayAccidentEnv,
    ego: str,
    action: int,
    n_steps: int,
    gamma: float,
    p: dict[str, Any],
    blocked: set[int],
) -> dict[str, Any] | None:
    """Hold `action` on ego for n_steps via env.step. Other CAVs get KEEP."""
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
    inds: list[float] = []
    grps: list[float] = []
    blocks: list[float] = []
    ego_speeds: list[float] = []
    mean_speeds: list[float] = []
    near_speeds: list[float] = []
    queues: list[float] = []
    blocked_fracs: list[float] = []
    lats: list[float] = []
    completed_step: int | None = None
    lane_after = lane_before

    for t in range(n_steps):
        try:
            if ego not in set(traci.vehicle.getIDList()):
                return None
            # One forced discrete command at the start of the window; remaining
            # steps use KEEP so the 1.0 s consequences of that request can unfold
            # without re-issuing changeLane every 0.1 s.
            cmd = int(action) if t == 0 else int(KEEP)
            _obs, _rew, _term, _trunc, info = env.step({ego: cmd})
        except Exception:  # noqa: BLE001
            return None
        detail = (info.get("reward_details") or {}).get(ego)
        if detail is None:
            return None
        totals.append(float(detail["total"]))
        inds.append(float(detail["individual"]))
        grps.append(float(detail["group"]))
        blocks.append(float(detail["p_block"]))
        try:
            lane_after = int(traci.vehicle.getLaneIndex(ego))
            ego_speeds.append(float(traci.vehicle.getSpeed(ego)))
        except Exception:  # noqa: BLE001
            return None
        lat = _lateral(traci, ego)
        if lat is not None:
            lats.append(lat)
        if completed_step is None and lane_after != lane_before:
            completed_step = t + 1
        try:
            met = _traffic(env, p, blocked)
        except Exception:  # noqa: BLE001
            return None
        mean_speeds.append(met["mean_speed"])
        if not math.isnan(met["near_zone_speed"]):
            near_speeds.append(met["near_zone_speed"])
        queues.append(met["queue"])
        if not math.isnan(met["blocked_lane_near_frac"]):
            blocked_fracs.append(met["blocked_lane_near_frac"])

    if action == LEFT:
        effective = lane_after > lane_before
    elif action == RIGHT:
        effective = lane_after < lane_before
    else:
        effective = False

    duration: float | None
    if action == KEEP:
        duration = None
    elif completed_step is None:
        duration = None
    else:
        duration = float(completed_step) * float(env.step_length)

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "requested_action": ACTION_NAMES[action],
        "action_index": int(action),
        "lane_before": lane_before,
        "lane_after": int(lane_after),
        "effective_lane_change": bool(effective),
        "lane_change_started": bool(action != KEEP),
        "lane_change_completed": bool(effective),
        "lane_change_duration_s": duration,
        "max_abs_lateral_m": float(max(abs(v) for v in lats)) if lats else None,
        "force_protocol": "first_step_forced_then_KEEP",
        "sum_reward": float(sum(totals)),
        "mean_reward": float(np.mean(totals)),
        "discounted_return": _discounted(totals, gamma),
        "sum_individual": float(sum(inds)),
        "sum_group": float(sum(grps)),
        "sum_p_block": float(sum(blocks)),
        "ego_speed_mean": _mean(ego_speeds),
        "ego_speed_final": float(ego_speeds[-1]) if ego_speeds else float("nan"),
        "mean_speed": _mean(mean_speeds),
        "near_zone_speed": _mean(near_speeds),
        "queue_mean": _mean(queues),
        "blocked_lane_near_frac": _mean(blocked_fracs),
        "n_steps": int(n_steps),
    }


def _restore(env: HighwayAccidentEnv, path: Path, step_count: int) -> bool:
    try:
        traci = env._require_traci()
        traci.simulation.loadState(str(path))
        env._step_count = int(step_count)
        return True
    except Exception:  # noqa: BLE001
        return False


def _fingerprint(traci: Any, egos: list[str]) -> tuple[Any, ...] | None:
    try:
        ids = set(traci.vehicle.getIDList())
        bits: list[Any] = [round(float(traci.simulation.getTime()), 3), len(ids)]
        for ego in egos:
            if ego not in ids:
                bits.append((ego, None))
                continue
            bits.append((ego, int(traci.vehicle.getLaneIndex(ego))))
        return tuple(bits)
    except Exception:  # noqa: BLE001
        return None


def collect(model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    train_cfg = load_train_config()
    tblock = train_cfg.get("train", train_cfg)
    gamma = float(tblock.get("gamma", 0.99))

    set_eval_seed(SEED)
    model.eval()
    counts = {"before": 0, "near": 0, "after": 0}
    last_step: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    discards = {
        "mid_lane_change": 0,
        "gap": 0,
        "branch_incomplete": 0,
        "restore_mismatch": 0,
        "save_fail": 0,
        "lateral_unavailable": 0,
    }
    lateral_api: bool | None = None
    step_length = 0.1
    branch_steps = 10
    zone_r = 150.0
    x_acc = 700.0
    n_lanes = 3
    left_up = True
    episodes_run = 0
    save_ok = False

    tmp = tempfile.TemporaryDirectory(prefix="phase510_")
    state_path = Path(tmp.name) / "snap.xml"
    try:
        for episode in range(MAX_EPISODES):
            if all(counts[z] >= TARGET_PER_REGION for z in counts):
                break
            env_cfg = copy.deepcopy(load_config())
            env_cfg.setdefault("episode", {})["max_episode_steps"] = MAX_STEPS + branch_steps + 50
            env = HighwayAccidentEnv(config=env_cfg, gui=False, reward_config=CA_REWARD)
            step_length = float(env.step_length)
            branch_steps = int(math.ceil(1.0 / step_length - 1e-9))
            zone_r = float(env.reward_zone_range_m)
            x_acc = float(env.accident_position_m)
            n_lanes = int(env.n_lanes)
            left_up = bool(env.left_increases_index)
            p = _params(env)
            blocked = {int(x) for x in env.accident_lanes}
            print(
                f"--- episode {episode + 1}/{MAX_EPISODES} "
                f"step_length={step_length} branch_steps={branch_steps} "
                f"horizon_s={branch_steps * step_length:.3f} gamma={gamma} ---",
                flush=True,
            )
            episodes_run += 1
            last_step = {}
            accepts_this_ep = 0
            try:
                env.reset()
                traci = env._require_traci()
                for step in range(MAX_STEPS):
                    if all(counts[z] >= TARGET_PER_REGION for z in counts):
                        break
                    if accepts_this_ep >= RESTART_AFTER_ACCEPTS:
                        print(
                            f"  soft restart after {accepts_this_ep} accepts this episode",
                            flush=True,
                        )
                        break
                    try:
                        states = env.get_cav_states()
                        observations = env.get_observations()
                    except Exception as exc:  # noqa: BLE001
                        print(f"state read failed: {exc}", flush=True)
                        break
                    picked: list[tuple[str, str, dict[str, Any]]] = []
                    candidates: list[tuple[str, str, dict[str, Any]]] = []
                    for cid, st in states.items():
                        lane = int(st["lane"])
                        if not _both_legal(lane, n_lanes, left_up):
                            continue
                        zone = _zone(float(st["position"]), x_acc, zone_r)
                        if counts[zone] >= TARGET_PER_REGION:
                            continue
                        if step - last_step.get(cid, -10_000) < branch_steps:
                            discards["gap"] += 1
                            continue
                        lat = _lateral(traci, cid)
                        if lat is None:
                            lateral_api = False if lateral_api is None else lateral_api
                            discards["lateral_unavailable"] += 1
                        else:
                            lateral_api = True
                            if abs(lat) > LATERAL_MAX_M:
                                discards["mid_lane_change"] += 1
                                continue
                        if cid not in observations:
                            continue
                        candidates.append((cid, zone, st))
                    # Prefer regions that still need the most samples.
                    candidates.sort(key=lambda item: counts[item[1]])
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
                            fp = _fingerprint(traci, [c for c, _z, _s in picked])
                            if fp is None:
                                discards["restore_mismatch"] += 1
                                print("fingerprint failed; ending episode", flush=True)
                                break
                            for cid, zone, st in picked:
                                logits, probs = _logits(model, observations[cid], device)
                                branches: dict[str, dict[str, Any]] = {}
                                ok = True
                                for action in ACTIONS:
                                    if not _restore(env, state_path, step_count):
                                        ok = False
                                        break
                                    br = _run_branch(
                                        env, cid, action, branch_steps, gamma, p, blocked
                                    )
                                    if br is None:
                                        ok = False
                                        break
                                    branches[ACTION_NAMES[action]] = br
                                if not _restore(env, state_path, step_count):
                                    discards["restore_mismatch"] += 1
                                    ok = False
                                restored = _fingerprint(traci, [cid])
                                if (
                                    restored is None
                                    or restored[0] != fp[0]
                                    or (
                                        isinstance(restored[-1], tuple)
                                        and restored[-1][1] is None
                                    )
                                ):
                                    discards["restore_mismatch"] += 1
                                    ok = False
                                if not ok or len(branches) != 3:
                                    discards["branch_incomplete"] += 1
                                    # If TraCI is dead, stop the episode.
                                    if not env.is_connected:
                                        print("TraCI disconnected during branch; ending episode", flush=True)
                                        break
                                    continue
                                keep_b = branches["KEEP"]
                                left_b = branches["LEFT"]
                                right_b = branches["RIGHT"]
                                row = {
                                    "state_id": len(rows),
                                    "episode": episode,
                                    "step": step,
                                    "cav_id": cid,
                                    "zone": zone,
                                    "lane": int(st["lane"]),
                                    "pos": float(st["position"]),
                                    "speed": float(st["speed"]),
                                    "lateral_m": float(_lateral(traci, cid) or 0.0),
                                    "logit_KEEP": logits[KEEP],
                                    "logit_LEFT": logits[LEFT],
                                    "logit_RIGHT": logits[RIGHT],
                                    "P_KEEP": probs[KEEP],
                                    "P_LEFT": probs[LEFT],
                                    "P_RIGHT": probs[RIGHT],
                                    "d_logit_LK": logits[LEFT] - logits[KEEP],
                                    "d_logit_RK": logits[RIGHT] - logits[KEEP],
                                    "G_KEEP": keep_b["discounted_return"],
                                    "G_LEFT": left_b["discounted_return"],
                                    "G_RIGHT": right_b["discounted_return"],
                                    "U_KEEP": keep_b["sum_reward"],
                                    "U_LEFT": left_b["sum_reward"],
                                    "U_RIGHT": right_b["sum_reward"],
                                    "dG_LK": left_b["discounted_return"] - keep_b["discounted_return"],
                                    "dG_RK": right_b["discounted_return"] - keep_b["discounted_return"],
                                    "dG_LR": left_b["discounted_return"] - right_b["discounted_return"],
                                    "dU_LK": left_b["sum_reward"] - keep_b["sum_reward"],
                                    "dU_RK": right_b["sum_reward"] - keep_b["sum_reward"],
                                    "branches": branches,
                                }
                                rows.append(row)
                                counts[zone] += 1
                                last_step[cid] = step
                                accepts_this_ep += 1
                                if counts[zone] % 10 == 0 or counts[zone] in (1, TARGET_PER_REGION):
                                    print(
                                        f"  accepted {counts} last={zone} step={step} id={cid}",
                                        flush=True,
                                    )
                            if not env.is_connected:
                                break
                            if not _restore(env, state_path, step_count):
                                discards["restore_mismatch"] += 1
                                print("restore failed after samples; ending episode", flush=True)
                                break
                            if _fingerprint(traci, [c for c, _z, _s in picked]) != fp:
                                discards["restore_mismatch"] += 1
                                print("restore mismatch; ending episode", flush=True)
                                break

                    actions: dict[str, int] = {}
                    for cid, obs in observations.items():
                        t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device)
                        with torch.no_grad():
                            a, _lp = model.act(t, deterministic=False)
                        actions[cid] = int(a.item())
                    try:
                        _obs, _rew, _term, truncated, _info = env.step(actions)
                    except Exception as exc:  # noqa: BLE001
                        print(f"collection step failed: {exc}", flush=True)
                        break
                    if truncated:
                        break
            except Exception as exc:  # noqa: BLE001
                print(f"episode failed: {exc}", flush=True)
            finally:
                try:
                    env.close()
                except Exception:  # noqa: BLE001
                    pass
                # Ensure SUMO child is gone before the next episode.
                import time

                time.sleep(0.5)
    finally:
        tmp.cleanup()

    return {
        "rows": rows,
        "counts": counts,
        "discards": discards,
        "gamma": gamma,
        "step_length": step_length,
        "branch_steps": branch_steps,
        "horizon_s": float(branch_steps) * float(step_length),
        "zone_r": zone_r,
        "x_acc": x_acc,
        "n_lanes": n_lanes,
        "left_increases_index": left_up,
        "episodes_run": episodes_run,
        "save_ok": save_ok,
        "lateral_api": lateral_api,
        "other_agents": "KEEP",
    }


def _corr_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "pearson_LK": _pearson([r["d_logit_LK"] for r in rows], [r["dG_LK"] for r in rows]),
        "spearman_LK": _spearman([r["d_logit_LK"] for r in rows], [r["dG_LK"] for r in rows]),
        "pearson_RK": _pearson([r["d_logit_RK"] for r in rows], [r["dG_RK"] for r in rows]),
        "spearman_RK": _spearman([r["d_logit_RK"] for r in rows], [r["dG_RK"] for r in rows]),
        "pearson_LK_undisc": _pearson(
            [r["d_logit_LK"] for r in rows], [r["dU_LK"] for r in rows]
        ),
        "spearman_LK_undisc": _spearman(
            [r["d_logit_LK"] for r in rows], [r["dU_LK"] for r in rows]
        ),
    }


def _region_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def col(name: str) -> list[float]:
        return [float(r[name]) for r in rows]

    out: dict[str, Any] = {"n": len(rows)}
    for key in (
        "G_KEEP",
        "G_LEFT",
        "G_RIGHT",
        "U_KEEP",
        "U_LEFT",
        "U_RIGHT",
        "dG_LK",
        "dG_RK",
        "dG_LR",
        "dU_LK",
        "dU_RK",
        "d_logit_LK",
        "d_logit_RK",
        "P_KEEP",
        "P_LEFT",
        "P_RIGHT",
        "logit_KEEP",
        "logit_LEFT",
        "logit_RIGHT",
    ):
        out[key] = _stats(col(key) if rows else [])
    if rows:
        out["frac_GL_gt_GK"] = float(np.mean([r["dG_LK"] > FLAT_RETURN for r in rows]))
        out["frac_GR_gt_GK"] = float(np.mean([r["dG_RK"] > FLAT_RETURN for r in rows]))
        out["frac_GL_lt_GK"] = float(np.mean([r["dG_LK"] < -FLAT_RETURN for r in rows]))
        out["frac_GR_lt_GK"] = float(np.mean([r["dG_RK"] < -FLAT_RETURN for r in rows]))
        out["eff_LEFT"] = float(np.mean([r["branches"]["LEFT"]["effective_lane_change"] for r in rows]))
        out["eff_RIGHT"] = float(
            np.mean([r["branches"]["RIGHT"]["effective_lane_change"] for r in rows])
        )
        out["eff_KEEP_lane_changed"] = float(
            np.mean([r["branches"]["KEEP"]["lane_after"] != r["branches"]["KEEP"]["lane_before"] for r in rows])
        )
        for name, comp in (
            ("sum_individual", "sum_individual"),
            ("sum_group", "sum_group"),
            ("sum_p_block", "sum_p_block"),
            ("ego_speed_mean", "ego_speed_mean"),
            ("mean_speed", "mean_speed"),
            ("near_zone_speed", "near_zone_speed"),
            ("queue_mean", "queue_mean"),
            ("blocked_lane_near_frac", "blocked_lane_near_frac"),
        ):
            out[name] = {
                act: _stats([float(r["branches"][act][comp]) for r in rows])
                for act in ("KEEP", "LEFT", "RIGHT")
            }
        durations = [
            r["branches"]["LEFT"]["lane_change_duration_s"]
            for r in rows
            if r["branches"]["LEFT"]["lane_change_duration_s"] is not None
        ]
        out["lc_duration_LEFT_completed"] = _stats([float(x) for x in durations])
        durations_r = [
            r["branches"]["RIGHT"]["lane_change_duration_s"]
            for r in rows
            if r["branches"]["RIGHT"]["lane_change_duration_s"] is not None
        ]
        out["lc_duration_RIGHT_completed"] = _stats([float(x) for x in durations_r])
    return out


def _side_case(p_side: float | None, p_keep: float | None, d_med: float | None) -> str:
    if p_side is None or p_keep is None or d_med is None:
        return "insufficient"
    actor_gt = p_side > p_keep + 0.02
    if abs(d_med) <= FLAT_RETURN:
        ret = "approx"
    elif d_med > 0:
        ret = "gt"
    else:
        ret = "lt"
    if not actor_gt:
        return f"actor_not_preferring/{ret}"
    if ret == "gt":
        return "A"
    if ret == "approx":
        return "B"
    return "C"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = {z: [r for r in rows if r["zone"] == z] for z in ("before", "near", "after")}
    blocks = {z: _region_block(by[z]) for z in by}
    cases = {}
    for z, b in blocks.items():
        cases[z] = {
            "LEFT_vs_KEEP": _side_case(
                (b.get("P_LEFT") or {}).get("mean"),
                (b.get("P_KEEP") or {}).get("mean"),
                (b.get("dG_LK") or {}).get("median"),
            ),
            "RIGHT_vs_KEEP": _side_case(
                (b.get("P_RIGHT") or {}).get("mean"),
                (b.get("P_KEEP") or {}).get("mean"),
                (b.get("dG_RK") or {}).get("median"),
            ),
        }
    labels = [cases[z]["LEFT_vs_KEEP"] for z in ("before", "near", "after") if by[z]]
    overall = "D" if len(set(labels)) > 1 else (labels[0] if labels else "insufficient")
    return {
        "by_zone": blocks,
        "corr": {z: _corr_block(by[z]) for z in ("before", "near", "after")},
        "corr_all": _corr_block(rows),
        "cases": cases,
        "overall_left_case": overall,
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


def write_tables(rows: list[dict[str, Any]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta_fields = [
        "state_id",
        "episode",
        "step",
        "cav_id",
        "zone",
        "lane",
        "pos",
        "speed",
        "lateral_m",
        "logit_KEEP",
        "logit_LEFT",
        "logit_RIGHT",
        "P_KEEP",
        "P_LEFT",
        "P_RIGHT",
        "d_logit_LK",
        "d_logit_RK",
        "G_KEEP",
        "G_LEFT",
        "G_RIGHT",
        "U_KEEP",
        "U_LEFT",
        "U_RIGHT",
        "dG_LK",
        "dG_RK",
        "dG_LR",
        "dU_LK",
        "dU_RK",
    ]
    with (OUT / "state_metadata.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=meta_fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in meta_fields})

    br_fields = [
        "state_id",
        "zone",
        "cav_id",
        "step",
        "requested_action",
        "action_index",
        "lane_before",
        "lane_after",
        "effective_lane_change",
        "lane_change_started",
        "lane_change_completed",
        "lane_change_duration_s",
        "max_abs_lateral_m",
        "sum_reward",
        "mean_reward",
        "discounted_return",
        "sum_individual",
        "sum_group",
        "sum_p_block",
        "ego_speed_mean",
        "ego_speed_final",
        "mean_speed",
        "near_zone_speed",
        "queue_mean",
        "blocked_lane_near_frac",
        "n_steps",
        "force_protocol",
    ]
    with (OUT / "branch_returns.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=br_fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            for act in ("KEEP", "LEFT", "RIGHT"):
                br = r["branches"][act]
                w.writerow(
                    {
                        "state_id": r["state_id"],
                        "zone": r["zone"],
                        "cav_id": r["cav_id"],
                        "step": r["step"],
                        **br,
                    }
                )


def save_figures(rows: list[dict[str, Any]]) -> list[str]:
    written: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"figures skipped: {exc}", flush=True)
        return written

    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    zones = ["before", "near", "after"]
    present = [z for z in zones if any(r["zone"] == z for r in rows)]
    if not present:
        return written

    means = {act: [] for act in ("KEEP", "LEFT", "RIGHT")}
    for z in present:
        sub = [r for r in rows if r["zone"] == z]
        for act, key in (("KEEP", "G_KEEP"), ("LEFT", "G_LEFT"), ("RIGHT", "G_RIGHT")):
            means[act].append(float(np.mean([r[key] for r in sub])))

    x = np.arange(len(present))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = {"KEEP": "#4d4d4d", "LEFT": "#2166ac", "RIGHT": "#b2182b"}
    for i, act in enumerate(("KEEP", "LEFT", "RIGHT")):
        ax.bar(x + (i - 1) * width, means[act], width, label=act, color=colors[act])
    ax.set_xticks(x)
    ax.set_xticklabels([z.capitalize() for z in present])
    ax.set_ylabel("Mean discounted branch return")
    ax.set_xlabel("Region")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p1 = fig_dir / "return_keep_left_right.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    written.append(str(p1.relative_to(OUT)))

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), sharey=False)
    zone_color = {"before": "#4d4d4d", "near": "#e66101", "after": "#2166ac"}
    panels = (
        (axes[0], "d_logit_LK", "dG_LK", "LEFT − KEEP"),
        (axes[1], "d_logit_RK", "dG_RK", "RIGHT − KEEP"),
    )
    for ax, xk, yk, title in panels:
        for z in present:
            sub = [r for r in rows if r["zone"] == z]
            ax.scatter(
                [r[xk] for r in sub],
                [r[yk] for r in sub],
                s=16,
                alpha=0.75,
                c=zone_color[z],
                label=z,
                linewidths=0,
            )
        ax.axhline(0.0, color="#999999", lw=0.8)
        ax.axvline(0.0, color="#999999", lw=0.8)
        ax.set_xlabel(f"Δlogit ({title})")
        ax.set_ylabel(f"Δ discounted return ({title})")
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p2 = fig_dir / "logit_vs_return.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    written.append(str(p2.relative_to(OUT)))
    return written


def _assoc(r: float | None) -> str:
    if r is None:
        return "n/a"
    a = abs(r)
    if a < 0.1:
        return f"{r:.3f} (no clear linear association)"
    if a < 0.3:
        return f"{r:.3f} (weak)"
    return f"{r:.3f} (moderate or stronger; still not causal)"


def write_report(
    *,
    meta: dict[str, Any],
    summary: dict[str, Any],
    frozen: bool,
    ckpt_same: bool,
    figures: list[str],
    status: str,
) -> None:
    b = summary["by_zone"]
    lines: list[str] = []
    a = lines.append
    a("# Phase 5.10 Report")
    a("")
    a("## 1. Status")
    a("")
    a(f"`{status}`")
    a("")
    a(f"- Retrained: **No**.")
    a("- Reward / reward weights: **not modified**.")
    a("- Observation: **not modified**.")
    a("- Action space: **not modified**.")
    a("- Actor / Critic: **not modified**.")
    a("- PPO / MAPPO: **not modified**.")
    a("- Action mask: **not added**.")
    a(f"- Checkpoint overwritten: **{'No' if ckpt_same else 'YES — unexpected'}**.")
    a(f"- Actor parameters frozen during the run: **{'YES' if frozen else 'NO'}**.")
    a(
        "- PASS means the same-state 1.0 s branch finished with at least 50 comparable "
        "states in every region. It does not mean Candidate A is a good controller, "
        "and it does not authorize a reward edit or an action mask."
    )
    a("")
    a("## 2. Experiment Setup")
    a("")
    a(f"- Checkpoint: `{CA_CKPT.relative_to(PROJECT_ROOT).as_posix()}` (read-only, episode 50).")
    a(f"- Reward config used only for measurement: `{CA_REWARD.relative_to(PROJECT_ROOT).as_posix()}`.")
    a(f"- SUMO step length: **{meta['step_length']} s**.")
    a(
        f"- Branch horizon: **{meta['branch_steps']}** simulation steps "
        f"= **{meta['horizon_s']:.3f} s** "
        f"(`ceil(1.0 / step_length)`)."
    )
    a(f"- Project discount γ = **{meta['gamma']}** from `configs/train.yaml`.")
    a(
        "- Return reported here is **not** GAE and **not** a PPO advantage. "
        "No critic bootstrap is added. "
        f"`G = Σ γ^t r_t` over the {meta['branch_steps']} post-step rewards. "
        "`U = Σ r_t` is the undiscounted sum. Both are stored."
    )
    a(
        f"- Regions use the project zone radius {meta['zone_r']} m around "
        f"x={meta['x_acc']} m: before `x < {meta['x_acc'] - meta['zone_r']}`, "
        f"near `|x - {meta['x_acc']}| ≤ {meta['zone_r']}`, "
        f"after `x > {meta['x_acc'] + meta['zone_r']}`."
    )
    a(
        f"- Lane definition from code: n_lanes={meta['n_lanes']}, "
        f"left_increases_index={meta['left_increases_index']}. "
        "Lane 0 is rightmost, lane 2 is leftmost. "
        "A state is kept only when `target_lane_index` is not None for both LEFT and RIGHT. "
        "On this 3-lane net that is lane 1."
    )
    a(
        "- Collection: frozen actor, stochastic actions between samples so lane 1 is visited. "
        f"Seed {SEED}. Up to {meta['episodes_run']} episode(s), {MAX_STEPS} steps each. "
        f"Same CAV not reused within {meta['branch_steps']} original steps. "
        "States were not manufactured."
    )
    a(
        f"- Accepted states: before {meta['counts']['before']}, "
        f"near {meta['counts']['near']}, after {meta['counts']['after']}."
    )
    a(
        "- Excluded: CAV missing, lane not both-legal, already lane-changing "
        f"(`|lateral position| > {LATERAL_MAX_M} m`), snapshot save/load failure, "
        "or any branch that lost the ego before 1.0 s."
    )
    a(f"- Discard counts: `{meta['discards']}`.")
    a(
        f"- In-progress lane-change filter: "
        f"{'available' if meta.get('lateral_api') else 'unavailable'}."
    )
    a(f"- saveState/loadState: {'worked' if meta.get('save_ok') else 'FAILED'}.")
    a("")
    a("## 3. Same-State Branch Method")
    a("")
    a("```text")
    a("same SUMO snapshot S")
    a("    loadState(S) → force KEEP  → env.step × N")
    a("    loadState(S) → force LEFT  → env.step × N")
    a("    loadState(S) → force RIGHT → env.step × N")
    a("    loadState(S) → continue the original episode")
    a("```")
    a("")
    a(
        "The forced action is applied on the **first** of the N steps for the ego: "
        "`env.step({ego: action})`. That uses the existing execution path "
        "(`apply_action` then one `simulationStep`). It does **not** call `setLane` / `moveTo`. "
        "The remaining N−1 steps use KEEP for the ego so the 1.0 s window can finish without "
        "re-issuing `changeLane` every 0.1 s. Other CAVs are absent from the action dict, so "
        "`env.step` applies KEEP to them. The actor is not used to choose any action inside a "
        "branch. The actor is read once, before the branch, for logits and probabilities."
    )
    a("")
    a(
        "`lane_change_started` is true for every legal LEFT/RIGHT request (first-step command). "
        "`lane_change_completed` is true only if the lane index actually moves in the requested "
        "direction within the window. Duration is the time until that index change; if the index "
        "does not change, duration is **unavailable**."
    )
    a("")
    a("## 4. Return Results")
    a("")
    a(
        "Main table: mean discounted branch return `G`. "
        "`ΔL−K` and `ΔR−K` are means of per-state differences."
    )
    a("")
    a("| Region | N | Return KEEP | Return LEFT | Return RIGHT | ΔL−K | ΔR−K |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        blk = b[z]
        a(
            f"| {label} | {blk['n']} | {_fmt((blk.get('G_KEEP') or {}).get('mean'))} | "
            f"{_fmt((blk.get('G_LEFT') or {}).get('mean'))} | {_fmt((blk.get('G_RIGHT') or {}).get('mean'))} | "
            f"{_fmt((blk.get('dG_LK') or {}).get('mean'))} | {_fmt((blk.get('dG_RK') or {}).get('mean'))} |"
        )
    a("")
    a("Median of the same per-state differences (used for Case A–D, threshold "
      f"|median ΔG| ≤ {FLAT_RETURN} counts as approximately equal):")
    a("")
    a("| Region | median G KEEP | median G LEFT | median G RIGHT | median ΔL−K | median ΔR−K |")
    a("| --- | ---: | ---: | ---: | ---: | ---: |")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        blk = b[z]
        a(
            f"| {label} | {_fmt((blk.get('G_KEEP') or {}).get('median'))} | "
            f"{_fmt((blk.get('G_LEFT') or {}).get('median'))} | {_fmt((blk.get('G_RIGHT') or {}).get('median'))} | "
            f"{_fmt((blk.get('dG_LK') or {}).get('median'))} | {_fmt((blk.get('dG_RK') or {}).get('median'))} |"
        )
    a("")
    a("Undiscounted cumulative reward `U = Σ r_t` (mean):")
    a("")
    a("| Region | U KEEP | U LEFT | U RIGHT | mean ΔU L−K | mean ΔU R−K |")
    a("| --- | ---: | ---: | ---: | ---: | ---: |")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        blk = b[z]
        a(
            f"| {label} | {_fmt((blk.get('U_KEEP') or {}).get('mean'))} | "
            f"{_fmt((blk.get('U_LEFT') or {}).get('mean'))} | {_fmt((blk.get('U_RIGHT') or {}).get('mean'))} | "
            f"{_fmt((blk.get('dU_LK') or {}).get('mean'))} | {_fmt((blk.get('dU_RK') or {}).get('mean'))} |"
        )
    a("")
    a(
        f"Fraction of states with ΔG > {FLAT_RETURN} (LEFT vs KEEP / RIGHT vs KEEP): "
        + ", ".join(
            f"{z} {_fmt((b[z].get('frac_GL_gt_GK')), 3)}/{_fmt((b[z].get('frac_GR_gt_GK')), 3)}"
            for z in ("before", "near", "after")
        )
        + "."
    )
    a("")
    a("## 5. State Transition Results")
    a("")
    a(
        "Requested action and effective lane change are separate. "
        "Effective means the lane index moved in the requested direction by the end of the window."
    )
    a("")
    a("| Region | effective LEFT | effective RIGHT | KEEP lane index changed |")
    a("| --- | ---: | ---: | ---: |")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        blk = b[z]
        a(
            f"| {label} | {_fmt(blk.get('eff_LEFT'), 3)} | {_fmt(blk.get('eff_RIGHT'), 3)} | "
            f"{_fmt(blk.get('eff_KEEP_lane_changed'), 3)} |"
        )
    a("")
    a(
        "Lane-change duration is recorded only when the lane index changes. "
        "Otherwise it is unavailable. Mean duration among completed maneuvers:"
    )
    a("")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        blk = b[z]
        dl = (blk.get("lc_duration_LEFT_completed") or {}).get("mean")
        dr = (blk.get("lc_duration_RIGHT_completed") or {}).get("mean")
        nl = (blk.get("lc_duration_LEFT_completed") or {}).get("n")
        nr = (blk.get("lc_duration_RIGHT_completed") or {}).get("n")
        a(f"- {label}: LEFT {_fmt(dl)} s (n={nl}), RIGHT {_fmt(dr)} s (n={nr}).")
    a("")
    a("Mean ego speed over the window (m/s):")
    a("")
    a("| Region | KEEP | LEFT | RIGHT |")
    a("| --- | ---: | ---: | ---: |")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        sp = b[z].get("ego_speed_mean") or {}
        a(
            f"| {label} | {_fmt((sp.get('KEEP') or {}).get('mean'), 3)} | "
            f"{_fmt((sp.get('LEFT') or {}).get('mean'), 3)} | "
            f"{_fmt((sp.get('RIGHT') or {}).get('mean'), 3)} |"
        )
    a("")
    a(
        "Traffic metrics reuse `utils/baseline_eval.py`: "
        "`mean_speed` is the mean speed of non-accident vehicles; "
        "`near_zone_speed` uses the same near-accident radius; "
        "`queue` counts upstream vehicles slower than 15 m/s within 250 m; "
        "`blocked_lane_near_frac` is near-zone CAVs on blocked lanes divided by near-zone CAVs. "
        "Values below are means over the 1.0 s window."
    )
    a("")
    a("| Region | metric | KEEP | LEFT | RIGHT |")
    a("| --- | --- | ---: | ---: | ---: |")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        blk = b[z]
        for metric in ("mean_speed", "near_zone_speed", "queue_mean", "blocked_lane_near_frac"):
            block = blk.get(metric) or {}
            a(
                f"| {label} | {metric} | {_fmt((block.get('KEEP') or {}).get('mean'), 3)} | "
                f"{_fmt((block.get('LEFT') or {}).get('mean'), 3)} | "
                f"{_fmt((block.get('RIGHT') or {}).get('mean'), 3)} |"
            )
    a("")
    a("Reward components, mean of the 1.0 s sum (not a new reward):")
    a("")
    a("| Region | component | KEEP | LEFT | RIGHT |")
    a("| --- | --- | ---: | ---: | ---: |")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        blk = b[z]
        for metric in ("sum_individual", "sum_group", "sum_p_block"):
            block = blk.get(metric) or {}
            a(
                f"| {label} | {metric} | {_fmt((block.get('KEEP') or {}).get('mean'))} | "
                f"{_fmt((block.get('LEFT') or {}).get('mean'))} | "
                f"{_fmt((block.get('RIGHT') or {}).get('mean'))} |"
            )
    a("")
    a(
        "A speed change during an in-progress lane change is not, by itself, a reward explanation. "
        "Completion rates above are the check on whether the 1.0 s window contains a finished maneuver."
    )
    a("")
    a("## 6. Actor Preference")
    a("")
    a("Logits and probabilities are the frozen actor on the pre-branch state. No argmax was executed inside the branch.")
    a("")
    a("| Region | P(KEEP) | P(LEFT) | P(RIGHT) | Δlogit L−K | Δlogit R−K |")
    a("| --- | ---: | ---: | ---: | ---: | ---: |")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        blk = b[z]
        a(
            f"| {label} | {_fmt((blk.get('P_KEEP') or {}).get('mean'), 3)} | "
            f"{_fmt((blk.get('P_LEFT') or {}).get('mean'), 3)} | "
            f"{_fmt((blk.get('P_RIGHT') or {}).get('mean'), 3)} | "
            f"{_fmt((blk.get('d_logit_LK') or {}).get('mean'), 3)} | "
            f"{_fmt((blk.get('d_logit_RK') or {}).get('mean'), 3)} |"
        )
    a("")
    a("| Region | mean logit KEEP | LEFT | RIGHT |")
    a("| --- | ---: | ---: | ---: |")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        blk = b[z]
        a(
            f"| {label} | {_fmt((blk.get('logit_KEEP') or {}).get('mean'), 3)} | "
            f"{_fmt((blk.get('logit_LEFT') or {}).get('mean'), 3)} | "
            f"{_fmt((blk.get('logit_RIGHT') or {}).get('mean'), 3)} |"
        )
    a("")
    a("## 7. Logit vs Return")
    a("")
    a(
        "Correlation is between `logit_LEFT - logit_KEEP` and `G_LEFT - G_KEEP` "
        "(and the RIGHT analogue). Association is not a PPO causal proof."
    )
    a("")
    a("| Slice | N | Pearson L−K | Spearman L−K | Pearson R−K | Spearman R−K |")
    a("| --- | ---: | ---: | ---: | ---: | ---: |")
    for key, label in (
        ("corr_all", "All"),
        ("before", "Before"),
        ("near", "Near"),
        ("after", "After"),
    ):
        c = summary["corr_all"] if key == "corr_all" else summary["corr"][key]
        a(
            f"| {label} | {c['n']} | {_fmt(c['pearson_LK'], 3)} | {_fmt(c['spearman_LK'], 3)} | "
            f"{_fmt(c['pearson_RK'], 3)} | {_fmt(c['spearman_RK'], 3)} |"
        )
    a("")
    a(f"Reading of Pearson L−K (all states): {_assoc(summary['corr_all']['pearson_LK'])}.")
    a(f"Reading of Pearson R−K (all states): {_assoc(summary['corr_all']['pearson_RK'])}.")
    if figures:
        a("")
        a("Figures:")
        for fig in figures:
            a(f"- `figures/{Path(fig).name}`")
    a("")
    a("## 8. Interpretation")
    a("")
    a(
        "Case labels use median ΔG and mean probabilities. "
        f"Approximately equal means |median ΔG| ≤ {FLAT_RETURN}. "
        "Actor prefers a lateral action when its mean probability exceeds P(KEEP) by more than 0.02. "
        "Case A: actor prefers that side and median return is higher. "
        "Case B: actor prefers that side and median return is approximately equal. "
        "Case C: actor prefers that side and median return is lower. "
        "If regions disagree, the overall reading is Case D. "
        "These labels are a summary of this 1.0 s window, not a training-cause proof."
    )
    a("")
    a("| Region | LEFT vs KEEP | RIGHT vs KEEP |")
    a("| --- | --- | --- |")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        c = summary["cases"][z]
        a(f"| {label} | {c['LEFT_vs_KEEP']} | {c['RIGHT_vs_KEEP']} |")
    a("")
    a(f"Overall LEFT-vs-KEEP label across regions: **{summary['overall_left_case']}**.")
    a("")
    a("### Confirmed")
    a("")
    a(
        "- Each accepted row is three branches from one TraCI `saveState` snapshot, "
        "on a state where both LEFT and RIGHT are legal, advanced for the same number of SUMO steps."
    )
    a("- The quantity compared is a forced branch return, not a PPO advantage.")
    a("- Requested lane changes and effective lane-index changes were recorded separately.")
    a("")
    a("### Supported")
    a("")
    a(
        "- Whether the 1.0 s return ranks the same way as the frozen actor is a numerical result "
        "of Section 4 and Section 6. It supports or weakens a short-horizon return explanation "
        "only for this window and this other-CAV=KEEP protocol. It does not identify the training update that raised the logits."
    )
    a("")
    a("### Unknown")
    a("")
    a("- Whether a horizon longer than 1.0 s, or a value-function bootstrap, would change the ranking.")
    a("- Whether other CAVs acting with the frozen policy, instead of KEEP, would change ΔG.")
    a("- The training-time PPO ratio path. It was not replayed.")
    a("- A unique cause of the lateral logit bias. Encoding bugs were already ruled out in Phase 5.9; this phase does not reopen that as a cause.")
    a("")
    a("## 9. Answer the Main Question")
    a("")
    a("### Do LEFT / RIGHT beat KEEP on the 1-second cumulative return, on the same legal state?")
    a("")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        blk = b[z]
        a(
            f"- {label} (n={blk['n']}): mean ΔG L−K = {_fmt((blk.get('dG_LK') or {}).get('mean'))}, "
            f"median = {_fmt((blk.get('dG_LK') or {}).get('median'))}; "
            f"mean ΔG R−K = {_fmt((blk.get('dG_RK') or {}).get('mean'))}, "
            f"median = {_fmt((blk.get('dG_RK') or {}).get('median'))}."
        )
    a("")
    a("### Does that return gap point the same way as the actor's lateral logit gap?")
    a("")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        c = summary["cases"][z]
        a(f"- {label}: LEFT vs KEEP → {c['LEFT_vs_KEEP']}; RIGHT vs KEEP → {c['RIGHT_vs_KEEP']}.")
    a("")
    a("### Does this support “long-term return caused the lateral preference”?")
    a("")
    a(
        "No. Evidence insufficient. A 1.0 s forced branch, without a value bootstrap and without "
        "the training-time PPO update, cannot establish that long-horizon return caused the logits. "
        "Phase 5.7 already found one-step reward gaps near zero. This phase only adds the 1.0 s window. "
        "If the window's return ranking disagrees with the actor, that weakens a 1-second-return explanation; "
        "it still does not prove a PPO bug."
    )
    a("")
    a(f"Phase 5.10 Result: {status}")
    a("")
    text = "\n".join(lines)
    (OUT / "report.md").write_text(text, encoding="utf-8")


def _status(meta: dict[str, Any], frozen: bool, ckpt_same: bool) -> str:
    if not meta.get("save_ok") or not frozen or not ckpt_same:
        return "FAIL"
    if not meta["rows"] and not meta["counts"]["before"]:
        return "FAIL"
    ns = [meta["counts"][z] for z in ("before", "near", "after")]
    if min(ns) <= 0:
        return "FAIL"
    if min(ns) < 50:
        return "PARTIAL"
    return "PASS"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not CA_CKPT.is_file():
        raise SystemExit(f"Missing checkpoint: {CA_CKPT}")
    ckpt_before = (CA_CKPT.stat().st_mtime_ns, CA_CKPT.stat().st_size)
    device = resolve_device("auto")
    print(f"Phase 5.10 device={device}", flush=True)
    model, _ckpt = load_mappo_checkpoint(CA_CKPT, device=device)
    model.eval()
    before = snapshot_model_params(model)
    meta = collect(model, device)
    frozen = params_unchanged(before, snapshot_model_params(model))
    ckpt_after = (CA_CKPT.stat().st_mtime_ns, CA_CKPT.stat().st_size)
    ckpt_same = ckpt_before == ckpt_after
    rows = meta["rows"]
    # Persist raw rows before any post-processing so a later crash does not lose the run.
    (OUT / "rows_raw.json").write_text(
        json.dumps(_clean(rows), indent=2), encoding="utf-8"
    )
    summary = summarize(rows)
    write_tables(rows)
    figures = save_figures(rows)
    status = _status(meta, frozen, ckpt_same)
    # lane-change duration string for the report
    meta["lane_change_duration_note"] = "configs/env.yaml lane_change_duration_s (1.0 s on this scene)"
    write_report(
        meta=meta,
        summary=summary,
        frozen=frozen,
        ckpt_same=ckpt_same,
        figures=figures,
        status=status,
    )
    payload = _clean(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "frozen": frozen,
            "ckpt_unchanged": ckpt_same,
            "gamma": meta["gamma"],
            "step_length": meta["step_length"],
            "branch_steps": meta["branch_steps"],
            "horizon_s": meta["horizon_s"],
            "counts": meta["counts"],
            "discards": meta["discards"],
            "save_ok": meta["save_ok"],
            "lateral_api": meta["lateral_api"],
            "summary": summary,
            "figures": figures,
        }
    )
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Report: {OUT / 'report.md'}", flush=True)
    print(f"Phase 5.10 Result: {status}", flush=True)


if __name__ == "__main__":
    main()
