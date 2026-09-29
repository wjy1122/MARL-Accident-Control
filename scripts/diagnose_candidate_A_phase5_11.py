"""Phase 5.11 — same-state 1.0 s branch + frozen Critic bootstrap (read-only).

Extends Phase 5.10: after each forced KEEP/LEFT/RIGHT window of N steps,
evaluate the frozen centralized Critic at the terminal joint state and form

    G_boot = Σ_{t=0}^{N-1} γ^t r_t  +  γ^N V(s_N)

This is still not a PPO advantage. It asks whether the critic-backed
1-second return ranks actions the same way as the actor's lateral preference.

Does not train. Does not edit reward / observation / action / PPO / checkpoints.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import sys
import tempfile
import time
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
    load_mappo_checkpoint,
    params_unchanged,
    set_eval_seed,
    snapshot_model_params,
)

OUT = PROJECT_ROOT / "results" / "experiments" / "reward_candidate_A_phase5_11"
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
TARGET_PER_REGION = 50
FLAT_RETURN = 0.01
LATERAL_MAX_M = 0.2
MAX_PICKS_PER_STEP = 1
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
    return _pearson(
        _rankdata(np.asarray(x, dtype=np.float64)).tolist(),
        _rankdata(np.asarray(y, dtype=np.float64)).tolist(),
    )


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


def _discounted(rs: list[float], gamma: float) -> float:
    return float(sum((gamma**t) * r for t, r in enumerate(rs)))


def _central_value(model: torch.nn.Module, env: HighwayAccidentEnv, device: torch.device) -> float | None:
    """Frozen centralized V(s) using the same slotting as rollout (sorted CAVs, pad zeros)."""
    try:
        observations = env.get_observations()
        ids = sorted(env.cav_ids)[: int(model.num_agents)]
        obs_list: list[torch.Tensor] = []
        for i in range(int(model.num_agents)):
            if i < len(ids) and ids[i] in observations:
                obs_list.append(
                    torch.as_tensor(
                        np.asarray(observations[ids[i]], dtype=np.float32),
                        device=device,
                    )
                )
            else:
                obs_list.append(torch.zeros(int(model.obs_dim), device=device))
        central = model.build_central_state(obs_list)
        with torch.no_grad():
            v = model.get_value(central)
        return float(v.reshape(-1)[0].detach().cpu().item())
    except Exception:  # noqa: BLE001
        return None


def _run_branch(
    model: torch.nn.Module,
    env: HighwayAccidentEnv,
    ego: str,
    action: int,
    n_steps: int,
    gamma: float,
    device: torch.device,
) -> dict[str, Any] | None:
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
    completed_step: int | None = None
    lane_after = lane_before

    for t in range(n_steps):
        try:
            if ego not in set(traci.vehicle.getIDList()):
                return None
            cmd = int(action) if t == 0 else int(KEEP)
            _obs, _rew, _term, _trunc, info = env.step({ego: cmd})
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
        if completed_step is None and lane_after != lane_before:
            completed_step = t + 1

    v_end = _central_value(model, env, device)
    if v_end is None:
        return None

    g = _discounted(totals, gamma)
    boot = float((gamma**n_steps) * v_end)
    g_boot = g + boot

    if action == LEFT:
        effective = lane_after > lane_before
    elif action == RIGHT:
        effective = lane_after < lane_before
    else:
        effective = False

    duration = None
    if action != KEEP and completed_step is not None:
        duration = float(completed_step) * float(env.step_length)

    return {
        "requested_action": ACTION_NAMES[action],
        "action_index": int(action),
        "lane_before": lane_before,
        "lane_after": int(lane_after),
        "effective_lane_change": bool(effective),
        "lane_change_completed": bool(effective),
        "lane_change_duration_s": duration,
        "force_protocol": "first_step_forced_then_KEEP",
        "sum_reward": float(sum(totals)),
        "discounted_return": g,
        "V_end": float(v_end),
        "bootstrap_term": boot,
        "G_boot": g_boot,
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
    discards = {
        "mid_lane_change": 0,
        "gap": 0,
        "branch_incomplete": 0,
        "restore_mismatch": 0,
        "save_fail": 0,
        "value_fail": 0,
    }
    rows: list[dict[str, Any]] = []
    step_length = 0.1
    branch_steps = 10
    zone_r = 150.0
    x_acc = 700.0
    n_lanes = 3
    left_up = True
    episodes_run = 0
    save_ok = False
    lateral_api: bool | None = None

    tmp = tempfile.TemporaryDirectory(prefix="phase511_")
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
            print(
                f"--- episode {episode + 1}/{MAX_EPISODES} "
                f"step_length={step_length} branch_steps={branch_steps} "
                f"horizon_s={branch_steps * step_length:.3f} gamma={gamma} ---",
                flush=True,
            )
            episodes_run += 1
            last_step: dict[str, int] = {}
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
                        else:
                            lateral_api = True
                            if abs(lat) > LATERAL_MAX_M:
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
                            fp = _fingerprint(traci, [c for c, _z, _s in picked])
                            if fp is None:
                                discards["restore_mismatch"] += 1
                                break
                            v0 = _central_value(model, env, device)
                            if v0 is None:
                                discards["value_fail"] += 1
                                if not _restore(env, state_path, step_count):
                                    break
                            else:
                                for cid, zone, st in picked:
                                    logits, probs = _logits(model, observations[cid], device)
                                    branches: dict[str, dict[str, Any]] = {}
                                    ok = True
                                    for action in ACTIONS:
                                        if not _restore(env, state_path, step_count):
                                            ok = False
                                            break
                                        br = _run_branch(
                                            model,
                                            env,
                                            cid,
                                            action,
                                            branch_steps,
                                            gamma,
                                            device,
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
                                        if not env.is_connected:
                                            print(
                                                "TraCI disconnected during branch; ending episode",
                                                flush=True,
                                            )
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
                                        "V0": float(v0),
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
                                        "Gb_KEEP": keep_b["G_boot"],
                                        "Gb_LEFT": left_b["G_boot"],
                                        "Gb_RIGHT": right_b["G_boot"],
                                        "Vend_KEEP": keep_b["V_end"],
                                        "Vend_LEFT": left_b["V_end"],
                                        "Vend_RIGHT": right_b["V_end"],
                                        "dG_LK": left_b["discounted_return"]
                                        - keep_b["discounted_return"],
                                        "dG_RK": right_b["discounted_return"]
                                        - keep_b["discounted_return"],
                                        "dGb_LK": left_b["G_boot"] - keep_b["G_boot"],
                                        "dGb_RK": right_b["G_boot"] - keep_b["G_boot"],
                                        "dV_LK": left_b["V_end"] - keep_b["V_end"],
                                        "dV_RK": right_b["V_end"] - keep_b["V_end"],
                                        "branches": branches,
                                    }
                                    rows.append(row)
                                    counts[zone] += 1
                                    last_step[cid] = step
                                    accepts_this_ep += 1
                                    if counts[zone] % 10 == 0 or counts[zone] in (
                                        1,
                                        TARGET_PER_REGION,
                                    ):
                                        print(
                                            f"  accepted {counts} last={zone} step={step} id={cid}",
                                            flush=True,
                                        )
                                if not env.is_connected:
                                    break
                                if not _restore(env, state_path, step_count):
                                    discards["restore_mismatch"] += 1
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
        "num_agents": int(model.num_agents),
        "obs_dim": int(model.obs_dim),
    }


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


def _region_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    keys = [
        "G_KEEP",
        "G_LEFT",
        "G_RIGHT",
        "Gb_KEEP",
        "Gb_LEFT",
        "Gb_RIGHT",
        "dG_LK",
        "dG_RK",
        "dGb_LK",
        "dGb_RK",
        "dV_LK",
        "dV_RK",
        "Vend_KEEP",
        "Vend_LEFT",
        "Vend_RIGHT",
        "V0",
        "d_logit_LK",
        "d_logit_RK",
        "P_KEEP",
        "P_LEFT",
        "P_RIGHT",
    ]
    for key in keys:
        out[key] = _stats([float(r[key]) for r in rows] if rows else [])
    if rows:
        out["eff_LEFT"] = float(
            np.mean([r["branches"]["LEFT"]["effective_lane_change"] for r in rows])
        )
        out["eff_RIGHT"] = float(
            np.mean([r["branches"]["RIGHT"]["effective_lane_change"] for r in rows])
        )
        out["frac_dGb_LK_gt"] = float(np.mean([r["dGb_LK"] > FLAT_RETURN for r in rows]))
        out["frac_dGb_LK_lt"] = float(np.mean([r["dGb_LK"] < -FLAT_RETURN for r in rows]))
        out["frac_dG_LK_gt"] = float(np.mean([r["dG_LK"] > FLAT_RETURN for r in rows]))
        out["frac_dG_LK_lt"] = float(np.mean([r["dG_LK"] < -FLAT_RETURN for r in rows]))
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = {z: [r for r in rows if r["zone"] == z] for z in ("before", "near", "after")}
    blocks = {z: _region_block(by[z]) for z in by}
    cases_g = {}
    cases_gb = {}
    for z, b in blocks.items():
        cases_g[z] = {
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
        cases_gb[z] = {
            "LEFT_vs_KEEP": _side_case(
                (b.get("P_LEFT") or {}).get("mean"),
                (b.get("P_KEEP") or {}).get("mean"),
                (b.get("dGb_LK") or {}).get("median"),
            ),
            "RIGHT_vs_KEEP": _side_case(
                (b.get("P_RIGHT") or {}).get("mean"),
                (b.get("P_KEEP") or {}).get("mean"),
                (b.get("dGb_RK") or {}).get("median"),
            ),
        }
    labels_g = [cases_g[z]["LEFT_vs_KEEP"] for z in ("before", "near", "after") if by[z]]
    labels_gb = [cases_gb[z]["LEFT_vs_KEEP"] for z in ("before", "near", "after") if by[z]]
    overall_g = "D" if len(set(labels_g)) > 1 else (labels_g[0] if labels_g else "insufficient")
    overall_gb = (
        "D" if len(set(labels_gb)) > 1 else (labels_gb[0] if labels_gb else "insufficient")
    )
    corr = {
        "all_G": {
            "n": len(rows),
            "pearson_LK": _pearson([r["d_logit_LK"] for r in rows], [r["dG_LK"] for r in rows]),
            "spearman_LK": _spearman([r["d_logit_LK"] for r in rows], [r["dG_LK"] for r in rows]),
            "pearson_RK": _pearson([r["d_logit_RK"] for r in rows], [r["dG_RK"] for r in rows]),
        },
        "all_Gb": {
            "n": len(rows),
            "pearson_LK": _pearson([r["d_logit_LK"] for r in rows], [r["dGb_LK"] for r in rows]),
            "spearman_LK": _spearman(
                [r["d_logit_LK"] for r in rows], [r["dGb_LK"] for r in rows]
            ),
            "pearson_RK": _pearson([r["d_logit_RK"] for r in rows], [r["dGb_RK"] for r in rows]),
            "spearman_RK": _spearman(
                [r["d_logit_RK"] for r in rows], [r["dGb_RK"] for r in rows]
            ),
        },
    }
    return {
        "by_zone": blocks,
        "cases_G": cases_g,
        "cases_Gb": cases_gb,
        "overall_left_G": overall_g,
        "overall_left_Gb": overall_gb,
        "corr": corr,
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
        "V0",
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
        "Gb_KEEP",
        "Gb_LEFT",
        "Gb_RIGHT",
        "Vend_KEEP",
        "Vend_LEFT",
        "Vend_RIGHT",
        "dG_LK",
        "dG_RK",
        "dGb_LK",
        "dGb_RK",
        "dV_LK",
        "dV_RK",
    ]
    with (OUT / "state_metadata.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=meta_fields, extrasaction="ignore")
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
        "lane_change_completed",
        "lane_change_duration_s",
        "force_protocol",
        "sum_reward",
        "discounted_return",
        "V_end",
        "bootstrap_term",
        "G_boot",
        "n_steps",
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
    present = [z for z in ("before", "near", "after") if any(r["zone"] == z for r in rows)]
    if not present:
        return written

    x = np.arange(len(present))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=False)
    colors = {"KEEP": "#4d4d4d", "LEFT": "#2166ac", "RIGHT": "#b2182b"}
    for ax, prefix, title in (
        (axes[0], "G_", "Bare discounted G"),
        (axes[1], "Gb_", "Bootstrapped G + γ^N V"),
    ):
        for i, act in enumerate(("KEEP", "LEFT", "RIGHT")):
            means = [
                float(np.mean([r[f"{prefix}{act}"] for r in rows if r["zone"] == z]))
                for z in present
            ]
            ax.bar(x + (i - 1) * width, means, width, label=act, color=colors[act])
        ax.set_xticks(x)
        ax.set_xticklabels([z.capitalize() for z in present])
        ax.set_ylabel("Mean return")
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p1 = fig_dir / "return_bare_vs_boot.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    written.append(str(p1.relative_to(OUT)))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    zone_color = {"before": "#4d4d4d", "near": "#e66101", "after": "#2166ac"}
    for z in present:
        sub = [r for r in rows if r["zone"] == z]
        ax.scatter(
            [r["d_logit_LK"] for r in sub],
            [r["dGb_LK"] for r in sub],
            s=18,
            alpha=0.75,
            c=zone_color[z],
            label=z,
            linewidths=0,
        )
    ax.axhline(0.0, color="#999999", lw=0.8)
    ax.axvline(0.0, color="#999999", lw=0.8)
    ax.set_xlabel("Δlogit LEFT−KEEP")
    ax.set_ylabel("ΔG_boot LEFT−KEEP")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p2 = fig_dir / "logit_vs_boot_return.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    written.append(str(p2.relative_to(OUT)))
    return written


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
    a("# Phase 5.11 Report")
    a("")
    a("## 1. Status")
    a("")
    a(f"`{status}`")
    a("")
    a("- Retrained: **No**.")
    a("- Reward / observation / action space / Actor / Critic / PPO: **not modified**.")
    a("- Action mask: **not added**.")
    a(f"- Checkpoint overwritten: **{'No' if ckpt_same else 'YES — unexpected'}**.")
    a(f"- Actor/Critic parameters frozen during the run: **{'YES' if frozen else 'NO'}**.")
    a(
        "- PASS means the bootstrapped same-state branch finished with ≥50 states per region. "
        "It does not authorize a reward edit or an action mask."
    )
    a("")
    a("## 2. Experiment Setup")
    a("")
    a(f"- Checkpoint: `{CA_CKPT.relative_to(PROJECT_ROOT).as_posix()}` (episode 50, read-only).")
    a(f"- SUMO step length: **{meta['step_length']} s**; branch: **{meta['branch_steps']}** steps = **{meta['horizon_s']:.3f} s**.")
    a(f"- γ = **{meta['gamma']}**. Critic input: concat of `{meta['num_agents']}` obs slots (pad zeros), matching rollout.")
    a(
        "- Quantities: `G = Σ γ^t r_t` (ego rewards only, no critic). "
        "`G_boot = G + γ^N V(s_N)` with frozen centralized Critic. "
        "**Neither is a PPO advantage.** Advantage would also subtract V(s_0); "
        "ΔG_boot between actions on the same s_0 cancels V(s_0)."
    )
    a(
        f"- Legal-only lane-1 states. Regions: before/near/after with radius {meta['zone_r']} m "
        f"around x={meta['x_acc']} m."
    )
    a(
        f"- Accepted: before {meta['counts']['before']}, near {meta['counts']['near']}, "
        f"after {meta['counts']['after']}. Episodes run: {meta['episodes_run']}."
    )
    a(f"- Discards: `{meta['discards']}`. saveState/loadState: {'worked' if meta.get('save_ok') else 'FAILED'}.")
    a("- Force protocol: first step = forced action, remaining steps = KEEP (same as Phase 5.10). Other CAVs = KEEP.")
    a("")
    a("## 3. Method")
    a("")
    a("```text")
    a("same snapshot S")
    a("  → force KEEP / LEFT / RIGHT for 1.0 s")
    a("  → read frozen V(s_N)")
    a("  → G_boot = Σ γ^t r_t + γ^N V(s_N)")
    a("```")
    a("")
    a("## 4. Bare G (replication of Phase 5.10 quantity)")
    a("")
    a("| Region | N | G KEEP | G LEFT | G RIGHT | ΔG L−K | ΔG R−K |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        blk = b[z]
        a(
            f"| {label} | {blk['n']} | {_fmt((blk.get('G_KEEP') or {}).get('mean'))} | "
            f"{_fmt((blk.get('G_LEFT') or {}).get('mean'))} | {_fmt((blk.get('G_RIGHT') or {}).get('mean'))} | "
            f"{_fmt((blk.get('dG_LK') or {}).get('mean'))} | {_fmt((blk.get('dG_RK') or {}).get('mean'))} |"
        )
    a("")
    a("## 5. Bootstrapped return G_boot")
    a("")
    a("| Region | N | Gb KEEP | Gb LEFT | Gb RIGHT | ΔGb L−K | ΔGb R−K |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        blk = b[z]
        a(
            f"| {label} | {blk['n']} | {_fmt((blk.get('Gb_KEEP') or {}).get('mean'))} | "
            f"{_fmt((blk.get('Gb_LEFT') or {}).get('mean'))} | {_fmt((blk.get('Gb_RIGHT') or {}).get('mean'))} | "
            f"{_fmt((blk.get('dGb_LK') or {}).get('mean'))} | {_fmt((blk.get('dGb_RK') or {}).get('mean'))} |"
        )
    a("")
    a("Median ΔGb (Case labels use these; |median| ≤ 0.01 ≈ equal):")
    a("")
    a("| Region | median ΔGb L−K | median ΔGb R−K | median ΔG L−K | median ΔG R−K |")
    a("| --- | ---: | ---: | ---: | ---: |")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        blk = b[z]
        a(
            f"| {label} | {_fmt((blk.get('dGb_LK') or {}).get('median'))} | "
            f"{_fmt((blk.get('dGb_RK') or {}).get('median'))} | "
            f"{_fmt((blk.get('dG_LK') or {}).get('median'))} | "
            f"{_fmt((blk.get('dG_RK') or {}).get('median'))} |"
        )
    a("")
    a("## 6. Terminal value V(s_N)")
    a("")
    a("| Region | V0 mean | Vend KEEP | Vend LEFT | Vend RIGHT | ΔV L−K | ΔV R−K |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        blk = b[z]
        a(
            f"| {label} | {_fmt((blk.get('V0') or {}).get('mean'), 3)} | "
            f"{_fmt((blk.get('Vend_KEEP') or {}).get('mean'), 3)} | "
            f"{_fmt((blk.get('Vend_LEFT') or {}).get('mean'), 3)} | "
            f"{_fmt((blk.get('Vend_RIGHT') or {}).get('mean'), 3)} | "
            f"{_fmt((blk.get('dV_LK') or {}).get('mean'), 3)} | "
            f"{_fmt((blk.get('dV_RK') or {}).get('mean'), 3)} |"
        )
    a("")
    a(f"Effective lane-change rates: " + ", ".join(
        f"{z} L={_fmt(b[z].get('eff_LEFT'), 2)} R={_fmt(b[z].get('eff_RIGHT'), 2)}"
        for z in ("before", "near", "after")
    ) + ".")
    a("")
    a("## 7. Actor preference")
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
    a("## 8. Logit vs return correlation")
    a("")
    cg = summary["corr"]["all_G"]
    cb = summary["corr"]["all_Gb"]
    a(
        f"- Bare G: Pearson L−K = {_fmt(cg['pearson_LK'], 3)}, Spearman L−K = {_fmt(cg['spearman_LK'], 3)}, "
        f"Pearson R−K = {_fmt(cg['pearson_RK'], 3)}."
    )
    a(
        f"- Boot G: Pearson L−K = {_fmt(cb['pearson_LK'], 3)}, Spearman L−K = {_fmt(cb['spearman_LK'], 3)}, "
        f"Pearson R−K = {_fmt(cb['pearson_RK'], 3)}, Spearman R−K = {_fmt(cb['spearman_RK'], 3)}."
    )
    if figures:
        a("")
        a("Figures:")
        for fig in figures:
            a(f"- `{fig}`")
    a("")
    a("## 9. Case labels (Actor vs return)")
    a("")
    a("Same Case A/B/C rules as Phase 5.10. Compared for bare G and for G_boot.")
    a("")
    a("| Region | LEFT vs KEEP (G) | LEFT vs KEEP (Gb) | RIGHT vs KEEP (G) | RIGHT vs KEEP (Gb) |")
    a("| --- | --- | --- | --- | --- |")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        cg_ = summary["cases_G"][z]
        cb_ = summary["cases_Gb"][z]
        a(
            f"| {label} | {cg_['LEFT_vs_KEEP']} | {cb_['LEFT_vs_KEEP']} | "
            f"{cg_['RIGHT_vs_KEEP']} | {cb_['RIGHT_vs_KEEP']} |"
        )
    a("")
    a(f"Overall LEFT-vs-KEEP (bare G): **{summary['overall_left_G']}**.")
    a(f"Overall LEFT-vs-KEEP (G_boot): **{summary['overall_left_Gb']}**.")
    a("")
    a("## 10. Interpretation")
    a("")
    a("### Confirmed")
    a("")
    a("- Same-state three-way forced branches on legal lane-1 states, with TraCI save/load.")
    a("- G_boot uses the project’s frozen centralized Critic; parameters were not updated.")
    a("")
    a("### Supported")
    a("")
    a(
        "- Whether adding γ^N V(s_N) flips the Phase 5.10 Case C finding is answered by "
        "Sections 5 and 9. If G_boot still disagrees with the actor’s LEFT preference, "
        "a 1-second critic-backed return also fails to explain the logits."
    )
    a("")
    a("### Unknown")
    a("")
    a("- Horizons longer than 1.0 s.")
    a("- Other CAVs following the frozen policy instead of KEEP.")
    a("- Training-time PPO ratios (still unavailable historically).")
    a("- Unique training cause of the lateral bias.")
    a("")
    a("## 11. Answer the Main Question")
    a("")
    a("### Does critic bootstrap reverse the 1 s return ranking vs KEEP?")
    a("")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        blk = b[z]
        a(
            f"- {label}: mean ΔG L−K={_fmt((blk.get('dG_LK') or {}).get('mean'))}, "
            f"mean ΔGb L−K={_fmt((blk.get('dGb_LK') or {}).get('mean'))}; "
            f"mean ΔG R−K={_fmt((blk.get('dG_RK') or {}).get('mean'))}, "
            f"mean ΔGb R−K={_fmt((blk.get('dGb_RK') or {}).get('mean'))}."
        )
    a("")
    a("### Does G_boot align with the actor’s lateral preference?")
    a("")
    a(f"- Bare-G overall LEFT label: {summary['overall_left_G']}.")
    a(f"- Boot-G overall LEFT label: {summary['overall_left_Gb']}.")
    a("")
    a("### Does this support “long-term / value-backed return caused lateral preference”?")
    a("")
    a(
        "Only if G_boot systematically favors the same lateral action the actor prefers. "
        "If G_boot remains Case C (or approximately equal), evidence is still **insufficient**: "
        "neither bare 1 s reward nor critic-backed 1 s return explains the LEFT logits. "
        "That does not prove a PPO implementation bug."
    )
    a("")
    a(f"Phase 5.11 Result: {status}")
    a("")
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _status(meta: dict[str, Any], frozen: bool, ckpt_same: bool) -> str:
    if not meta.get("save_ok") or not frozen or not ckpt_same:
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
    print(f"Phase 5.11 device={device}", flush=True)
    model, _ckpt = load_mappo_checkpoint(CA_CKPT, device=device)
    model.eval()
    before = snapshot_model_params(model)
    meta = collect(model, device)
    frozen = params_unchanged(before, snapshot_model_params(model))
    ckpt_same = ckpt_before == (CA_CKPT.stat().st_mtime_ns, CA_CKPT.stat().st_size)
    rows = meta["rows"]
    (OUT / "rows_raw.json").write_text(json.dumps(_clean(rows), indent=2), encoding="utf-8")
    summary = summarize(rows)
    write_tables(rows)
    figures = save_figures(rows)
    status = _status(meta, frozen, ckpt_same)
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
            "summary": summary,
            "figures": figures,
        }
    )
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Report: {OUT / 'report.md'}", flush=True)
    print(f"Phase 5.11 Result: {status}", flush=True)


if __name__ == "__main__":
    main()
