"""Phase 6.3 — Critic / GAE / advantage vs same-state branch return (read-only).

Frozen Phase 6.1 episode_0100 (Action Mask ON). No training. No edits to
reward, observation, mask, actor/critic, or PPO.

Compares on the same lane-1 legal snapshots:
  G      = Σ γ^t r_t
  G_boot = G + γ^N V(s_N)     (frozen centralized critic)
  A_boot = G_boot − V(s_0)    (ΔA_boot ≡ ΔG_boot across actions)
  A_gae  = GAE(λ) at t=0 along the forced branch (project compute_gae)

Other CAVs: KEEP (env default). Force protocol: first step forced, then KEEP.
Horizons: 1s / 3s / 5s via one 5s rollout sliced at cut steps.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.gae import compute_gae  # noqa: E402
from algorithms.rollout import load_train_config, resolve_device  # noqa: E402
from env.action import ACTION_NAMES, KEEP, LEFT, RIGHT, target_lane_index  # noqa: E402
from env.action_mask import apply_logit_mask, legal_action_mask  # noqa: E402
from env.highway_accident_env import HighwayAccidentEnv, load_config  # noqa: E402
from utils.baseline_eval import (  # noqa: E402
    load_mappo_checkpoint,
    params_unchanged,
    set_eval_seed,
    snapshot_model_params,
)

OUT = PROJECT_ROOT / "results" / "experiments" / "reward_candidate_A_phase6_3_critic_gae"
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
MAX_BRANCH_EPISODES = 4
TARGET_PER_REGION = 50
LATERAL_MAX_M = 0.2
MAX_PICKS_PER_STEP = 1
ZONE_CAP_PER_EPISODE = 50
HORIZON_SECONDS = (1.0, 3.0, 5.0)
ACTIONS = (KEEP, LEFT, RIGHT)
FLAT = 0.01


def _stats(xs: list[float]) -> dict[str, Any]:
    clean = [
        float(x)
        for x in xs
        if x is not None and not (isinstance(x, float) and math.isnan(x))
    ]
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
) -> tuple[int, list[float], list[float]]:
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
        logits_m = apply_logit_mask(logits_raw, am.unsqueeze(0))
        probs = torch.softmax(logits_m, dim=-1)[0]
        action = int(torch.argmax(logits_m, dim=-1)[0].item())
    logits = [float(x) for x in logits_m[0].detach().cpu().tolist()]
    probs_l = [float(x) for x in probs.detach().cpu().tolist()]
    return action, logits, probs_l


def _central_value(
    model: torch.nn.Module, env: HighwayAccidentEnv, device: torch.device
) -> float | None:
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


def _discounted_prefix(rs: list[float], gamma: float, n: int) -> float:
    return float(sum((gamma**t) * rs[t] for t in range(min(n, len(rs)))))


def _gae_a0(
    rewards: list[float],
    values: list[float],
    v_end: float,
    *,
    gamma: float,
    gae_lambda: float,
    n: int,
) -> float:
    """GAE advantage at t=0 for the first n steps of a forced branch."""
    n = min(n, len(rewards), len(values))
    if n <= 0:
        return float("nan")
    r = np.asarray(rewards[:n], dtype=np.float64).reshape(n, 1)
    v = np.asarray(values[:n], dtype=np.float64).reshape(n, 1)
    dones = np.zeros(n, dtype=np.bool_)
    # Bootstrap with V(s_n) after the nth transition (same as Phase 5.11 G_boot).
    next_v = np.asarray([v_end], dtype=np.float64)
    adv, _ret = compute_gae(
        r, v, dones, gamma=gamma, gae_lambda=gae_lambda, next_values=next_v
    )
    return float(adv[0, 0])


def _restore(env: HighwayAccidentEnv, path: Path, step_count: int) -> bool:
    try:
        traci = env._require_traci()
        traci.simulation.loadState(str(path))
        env._step_count = int(step_count)
        return True
    except Exception:  # noqa: BLE001
        return False


def _run_branch(
    model: torch.nn.Module,
    env: HighwayAccidentEnv,
    ego: str,
    action: int,
    n_steps: int,
    gamma: float,
    gae_lambda: float,
    cut_steps: list[int],
    v0: float,
    device: torch.device,
) -> dict[str, Any] | None:
    """Force first action then KEEP; record G / G_boot / A_boot / A_gae at cuts."""
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
    values: list[float] = [float(v0)]
    lane_after = lane_before
    completed: int | None = None

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
        # V(s_{t+1}) after transition — used as values[t+1] for GAE.
        if t < n_steps - 1:
            vt = _central_value(model, env, device)
            if vt is None:
                return None
            values.append(float(vt))

    v_end = _central_value(model, env, device)
    if v_end is None:
        return None

    if action == LEFT:
        effective = lane_after > lane_before
    elif action == RIGHT:
        effective = lane_after < lane_before
    else:
        effective = False

    cuts: dict[int, dict[str, float]] = {}
    for ncut in cut_steps:
        if ncut > len(totals):
            continue
        # For cut n: rewards[0..n-1], values[0..n-1]=V(s_0)..V(s_{n-1}),
        # bootstrap V(s_n). For n < max, V(s_n) is values[n] if available else v_end.
        if ncut < len(values):
            vend_n = float(values[ncut])
        else:
            vend_n = float(v_end)
        g = _discounted_prefix(totals, gamma, ncut)
        boot = float((gamma**ncut) * vend_n)
        g_boot = g + boot
        a_boot = g_boot - float(v0)
        a_gae = _gae_a0(
            totals,
            values,
            vend_n,
            gamma=gamma,
            gae_lambda=gae_lambda,
            n=ncut,
        )
        cuts[int(ncut)] = {
            "G": g,
            "G_boot": g_boot,
            "A_boot": a_boot,
            "A_gae": a_gae,
            "V_end": vend_n,
            "bootstrap": boot,
        }

    return {
        "lane_before": lane_before,
        "lane_after": lane_after,
        "effective": bool(effective),
        "completed_step": completed,
        "cuts": cuts,
        "V_end_full": float(v_end),
    }


def collect(model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    train_cfg = load_train_config()
    tblock = train_cfg.get("train", train_cfg)
    gamma = float(tblock.get("gamma", 0.99))
    gae_lambda = float(tblock.get("gae_lambda", 0.95))

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
        "value_fail": 0,
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

    tmp = tempfile.TemporaryDirectory(prefix="phase63_")
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
                f"cuts={cut_steps} gamma={gamma} lambda={gae_lambda} counts={counts} ---",
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
                    if step % 300 == 0:
                        lane1_x = [
                            float(st["position"])
                            for st in states.values()
                            if _both_legal(int(st["lane"]), n_lanes, left_up)
                        ]
                        if lane1_x:
                            max_legal_x = max(max_legal_x, max(lane1_x))
                        print(
                            f"  step {step} zone_ep={zone_ep} "
                            f"legal_n={len(lane1_x)} "
                            f"legal_max_x={max(lane1_x) if lane1_x else None}",
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
                            v0 = _central_value(model, env, device)
                            if v0 is None:
                                discards["value_fail"] += 1
                                if not _restore(env, state_path, step_count):
                                    break
                            else:
                                for cid, zone, st in picked:
                                    a0, logits, probs = _masked_policy(
                                        model,
                                        observations[cid],
                                        int(st["lane"]),
                                        env,
                                        device,
                                    )
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
                                            max_steps_branch,
                                            gamma,
                                            gae_lambda,
                                            cut_steps,
                                            float(v0),
                                            device,
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
                                        "speed": float(st["speed"]),
                                        "V0": float(v0),
                                        "policy_action": a0,
                                        "P_KEEP": probs[KEEP],
                                        "P_LEFT": probs[LEFT],
                                        "P_RIGHT": probs[RIGHT],
                                        "d_logit_LK": logits[LEFT] - logits[KEEP],
                                        "d_logit_RK": logits[RIGHT] - logits[KEEP],
                                    }
                                    for h_s, ncut in zip(HORIZON_SECONDS, cut_steps):
                                        tag = f"{int(h_s)}s"
                                        for name in ("KEEP", "LEFT", "RIGHT"):
                                            c = branches[name]["cuts"][ncut]
                                            row[f"G_{name}_{tag}"] = c["G"]
                                            row[f"Gb_{name}_{tag}"] = c["G_boot"]
                                            row[f"Ab_{name}_{tag}"] = c["A_boot"]
                                            row[f"Ag_{name}_{tag}"] = c["A_gae"]
                                        row[f"dG_LK_{tag}"] = (
                                            branches["LEFT"]["cuts"][ncut]["G"]
                                            - branches["KEEP"]["cuts"][ncut]["G"]
                                        )
                                        row[f"dG_RK_{tag}"] = (
                                            branches["RIGHT"]["cuts"][ncut]["G"]
                                            - branches["KEEP"]["cuts"][ncut]["G"]
                                        )
                                        row[f"dGb_LK_{tag}"] = (
                                            branches["LEFT"]["cuts"][ncut]["G_boot"]
                                            - branches["KEEP"]["cuts"][ncut]["G_boot"]
                                        )
                                        row[f"dGb_RK_{tag}"] = (
                                            branches["RIGHT"]["cuts"][ncut]["G_boot"]
                                            - branches["KEEP"]["cuts"][ncut]["G_boot"]
                                        )
                                        row[f"dAg_LK_{tag}"] = (
                                            branches["LEFT"]["cuts"][ncut]["A_gae"]
                                            - branches["KEEP"]["cuts"][ncut]["A_gae"]
                                        )
                                        row[f"dAg_RK_{tag}"] = (
                                            branches["RIGHT"]["cuts"][ncut]["A_gae"]
                                            - branches["KEEP"]["cuts"][ncut]["A_gae"]
                                        )
                                    row["eff_LEFT"] = branches["LEFT"]["effective"]
                                    row["eff_RIGHT"] = branches["RIGHT"]["effective"]
                                    rows.append(row)
                                    with partial.open("a", encoding="utf-8") as f:
                                        f.write(json.dumps(_clean(row)) + "\n")
                                    counts[zone] += 1
                                    zone_ep[zone] += 1
                                    last_step[cid] = step
                                    if (
                                        counts[zone] % 10 == 0
                                        or counts[zone] == TARGET_PER_REGION
                                    ):
                                        print(
                                            f"  accepted {counts} last={zone} step={step}",
                                            flush=True,
                                        )
                                if not env.is_connected:
                                    break
                                if not _restore(env, state_path, step_count):
                                    discards["restore_mismatch"] += 1
                                    break

                    actions: dict[str, int] = {}
                    for cid, st in states.items():
                        if cid not in observations:
                            continue
                        a, _lg, _pr = _masked_policy(
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
                and max_legal_x >= 0
                and max_legal_x < x_acc + zone_r
            ):
                print(
                    f"  full episode max lane-1 legal x={max_legal_x:.1f} "
                    f"< {x_acc + zone_r:.1f}; stop repeating same seed.",
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
            block: dict[str, Any] = {
                "G_KEEP": _stats([r[f"G_KEEP_{tag}"] for r in sub]),
                "G_LEFT": _stats([r[f"G_LEFT_{tag}"] for r in sub]),
                "G_RIGHT": _stats([r[f"G_RIGHT_{tag}"] for r in sub]),
                "dG_LK": _stats([r[f"dG_LK_{tag}"] for r in sub]),
                "dG_RK": _stats([r[f"dG_RK_{tag}"] for r in sub]),
                "Gb_KEEP": _stats([r[f"Gb_KEEP_{tag}"] for r in sub]),
                "Gb_LEFT": _stats([r[f"Gb_LEFT_{tag}"] for r in sub]),
                "Gb_RIGHT": _stats([r[f"Gb_RIGHT_{tag}"] for r in sub]),
                "dGb_LK": _stats([r[f"dGb_LK_{tag}"] for r in sub]),
                "dGb_RK": _stats([r[f"dGb_RK_{tag}"] for r in sub]),
                "dAg_LK": _stats([r[f"dAg_LK_{tag}"] for r in sub]),
                "dAg_RK": _stats([r[f"dAg_RK_{tag}"] for r in sub]),
                "P_KEEP": _stats([r["P_KEEP"] for r in sub]),
                "P_LEFT": _stats([r["P_LEFT"] for r in sub]),
                "P_RIGHT": _stats([r["P_RIGHT"] for r in sub]),
                "d_logit_LK": _stats([r["d_logit_LK"] for r in sub]),
                "d_logit_RK": _stats([r["d_logit_RK"] for r in sub]),
                "V0": _stats([r["V0"] for r in sub]),
            }
            if sub:
                # Sign agreement: actor prefers LEFT over KEEP iff P_LEFT>P_KEEP
                # (or logit gap > 0); return prefers LEFT iff dG > FLAT.
                prefer_L = [r["P_LEFT"] > r["P_KEEP"] for r in sub]
                prefer_R = [r["P_RIGHT"] > r["P_KEEP"] for r in sub]
                g_L = [r[f"dG_LK_{tag}"] > FLAT for r in sub]
                g_R = [r[f"dG_RK_{tag}"] > FLAT for r in sub]
                gb_L = [r[f"dGb_LK_{tag}"] > FLAT for r in sub]
                gb_R = [r[f"dGb_RK_{tag}"] > FLAT for r in sub]
                ag_L = [r[f"dAg_LK_{tag}"] > FLAT for r in sub]
                ag_R = [r[f"dAg_RK_{tag}"] > FLAT for r in sub]
                block["frac_actor_pref_L"] = float(np.mean(prefer_L))
                block["frac_actor_pref_R"] = float(np.mean(prefer_R))
                block["frac_G_favors_L"] = float(np.mean(g_L))
                block["frac_G_favors_R"] = float(np.mean(g_R))
                block["frac_Gb_favors_L"] = float(np.mean(gb_L))
                block["frac_Gb_favors_R"] = float(np.mean(gb_R))
                block["frac_Ag_favors_L"] = float(np.mean(ag_L))
                block["frac_Ag_favors_R"] = float(np.mean(ag_R))
                # Among states where actor prefers LEFT: does G / Gb / Ag agree?
                idx_L = [i for i, p in enumerate(prefer_L) if p]
                idx_R = [i for i, p in enumerate(prefer_R) if p]
                block["agree_G_when_actor_L"] = (
                    float(np.mean([g_L[i] for i in idx_L])) if idx_L else None
                )
                block["agree_Gb_when_actor_L"] = (
                    float(np.mean([gb_L[i] for i in idx_L])) if idx_L else None
                )
                block["agree_Ag_when_actor_L"] = (
                    float(np.mean([ag_L[i] for i in idx_L])) if idx_L else None
                )
                block["agree_G_when_actor_R"] = (
                    float(np.mean([g_R[i] for i in idx_R])) if idx_R else None
                )
                block["agree_Gb_when_actor_R"] = (
                    float(np.mean([gb_R[i] for i in idx_R])) if idx_R else None
                )
                block["agree_Ag_when_actor_R"] = (
                    float(np.mean([ag_R[i] for i in idx_R])) if idx_R else None
                )
            by_zone_h[z][tag] = block

    return {
        "rows": rows,
        "counts": counts,
        "discards": discards,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
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


def write_tables(out: dict[str, Any]) -> None:
    tab = OUT / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    rows = out["rows"]
    if rows:
        with (tab / "branch_critic_gae.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with (tab / "summary_by_zone_horizon.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "region",
                "horizon",
                "n",
                "mean_dG_LK",
                "mean_dG_RK",
                "mean_dGb_LK",
                "mean_dGb_RK",
                "mean_dAg_LK",
                "mean_dAg_RK",
                "mean_P_KEEP",
                "mean_P_LEFT",
                "mean_P_RIGHT",
                "agree_G_L",
                "agree_Gb_L",
                "agree_Ag_L",
                "agree_G_R",
                "agree_Gb_R",
                "agree_Ag_R",
            ]
        )
        for z in ("before", "near", "after"):
            block = out["by_zone_h"][z]
            for h_s in HORIZON_SECONDS:
                tag = f"{int(h_s)}s"
                h = block[tag]
                w.writerow(
                    [
                        z,
                        tag,
                        block["n"],
                        (h["dG_LK"] or {}).get("mean"),
                        (h["dG_RK"] or {}).get("mean"),
                        (h["dGb_LK"] or {}).get("mean"),
                        (h["dGb_RK"] or {}).get("mean"),
                        (h["dAg_LK"] or {}).get("mean"),
                        (h["dAg_RK"] or {}).get("mean"),
                        (h["P_KEEP"] or {}).get("mean"),
                        (h["P_LEFT"] or {}).get("mean"),
                        (h["P_RIGHT"] or {}).get("mean"),
                        h.get("agree_G_when_actor_L"),
                        h.get("agree_Gb_when_actor_L"),
                        h.get("agree_Ag_when_actor_L"),
                        h.get("agree_G_when_actor_R"),
                        h.get("agree_Gb_when_actor_R"),
                        h.get("agree_Ag_when_actor_R"),
                    ]
                )


def write_report(
    *,
    out: dict[str, Any],
    frozen: bool,
    ckpt_same: bool,
    status: str,
) -> None:
    lines: list[str] = []
    a = lines.append
    a("# Phase 6.3 Report")
    a("")
    a("## Critic / GAE vs same-state branch return")
    a("")
    a("### 1. Status")
    a("")
    a(f"`{status}`")
    a("")
    a("- Retrained: **No**.")
    a("- Reward / Observation / Action Mask / Actor / Critic / PPO: **not modified**.")
    a(f"- Phase 6.1 checkpoint overwritten: **{'YES' if not ckpt_same else 'No'}**.")
    a(f"- Actor/Critic frozen: **{'YES' if frozen else 'NO'}**.")
    a(
        "- PASS means usable before/near samples with invalid-free masked policy and "
        "finished branches. It does not mean training should change."
    )
    a("")
    a("### 2. Checkpoint & definitions")
    a("")
    a(f"- `{MASK_CKPT.relative_to(PROJECT_ROOT).as_posix()}`")
    a(f"- γ={out['gamma']}, λ={out['gae_lambda']}, step={out['step_length']} s.")
    a(f"- Cut steps: {out['cut_steps']} for horizons {out['horizons_s']} s.")
    a("- Lane-1 legal states only. Force first step then KEEP. Other CAVs: KEEP.")
    a("- `G = Σ γ^t r_t`")
    a("- `G_boot = G + γ^N V(s_N)` (frozen centralized critic)")
    a("- `A_boot = G_boot − V(s_0)` (ΔA_boot ≡ ΔG_boot across actions on same s_0)")
    a("- `A_gae` = project `compute_gae` at t=0 along the forced branch, bootstrap V(s_N)")
    a(f"- Counts={out['counts']}. Discards={out['discards']}.")
    a(f"- Max lane-1 legal x observed: `{out.get('max_legal_x')}`.")
    a("")
    a("### 3. Bare G (Phase 6.2-style)")
    a("")
    a("| Region | Horizon | N | KEEP G | LEFT G | RIGHT G | ΔL−K | ΔR−K |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        block = out["by_zone_h"][z]
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
    a("### 4. Bootstrapped G_boot")
    a("")
    a("| Region | Horizon | N | KEEP Gb | LEFT Gb | RIGHT Gb | ΔGb L−K | ΔGb R−K |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        block = out["by_zone_h"][z]
        for h_s in HORIZON_SECONDS:
            tag = f"{int(h_s)}s"
            h = block[tag]
            a(
                f"| {label} | {tag} | {block['n']} | "
                f"{_fmt((h['Gb_KEEP'] or {}).get('mean'))} | "
                f"{_fmt((h['Gb_LEFT'] or {}).get('mean'))} | "
                f"{_fmt((h['Gb_RIGHT'] or {}).get('mean'))} | "
                f"{_fmt((h['dGb_LK'] or {}).get('mean'))} | "
                f"{_fmt((h['dGb_RK'] or {}).get('mean'))} |"
            )
    a("")
    a("### 5. Branch GAE ΔA (LEFT−KEEP / RIGHT−KEEP)")
    a("")
    a("| Region | Horizon | N | mean ΔAg L−K | med ΔAg L−K | mean ΔAg R−K | med ΔAg R−K |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        block = out["by_zone_h"][z]
        for h_s in HORIZON_SECONDS:
            tag = f"{int(h_s)}s"
            h = block[tag]
            a(
                f"| {label} | {tag} | {block['n']} | "
                f"{_fmt((h['dAg_LK'] or {}).get('mean'))} | "
                f"{_fmt((h['dAg_LK'] or {}).get('median'))} | "
                f"{_fmt((h['dAg_RK'] or {}).get('mean'))} | "
                f"{_fmt((h['dAg_RK'] or {}).get('median'))} |"
            )
    a("")
    a("### 6. Actor preference on the same samples")
    a("")
    a("| Region | mean P(KEEP/LEFT/RIGHT) | mean Δlogit L−K / R−K | mean V0 |")
    a("| --- | --- | --- | ---: |")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        # Use 1s block for shared actor stats (same for all horizons).
        h = out["by_zone_h"][z]["1s"]
        a(
            f"| {label} | "
            f"{_fmt((h['P_KEEP'] or {}).get('mean'), 3)}/"
            f"{_fmt((h['P_LEFT'] or {}).get('mean'), 3)}/"
            f"{_fmt((h['P_RIGHT'] or {}).get('mean'), 3)} | "
            f"{_fmt((h['d_logit_LK'] or {}).get('mean'), 3)} / "
            f"{_fmt((h['d_logit_RK'] or {}).get('mean'), 3)} | "
            f"{_fmt((h['V0'] or {}).get('mean'))} |"
        )
    a("")
    a("### 7. Agreement: actor prefers X vs return/advantage favors X")
    a("")
    a(
        "Among states where `P(X) > P(KEEP)`, fraction with `Δ > +0.01` "
        "for bare G / G_boot / A_gae."
    )
    a("")
    a("| Region | Horizon | agree G L | agree Gb L | agree Ag L | agree G R | agree Gb R | agree Ag R |")
    a("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for z, label in (("before", "Before"), ("near", "Near"), ("after", "After")):
        block = out["by_zone_h"][z]
        for h_s in HORIZON_SECONDS:
            tag = f"{int(h_s)}s"
            h = block[tag]
            a(
                f"| {label} | {tag} | "
                f"{_fmt(h.get('agree_G_when_actor_L'), 3)} | "
                f"{_fmt(h.get('agree_Gb_when_actor_L'), 3)} | "
                f"{_fmt(h.get('agree_Ag_when_actor_L'), 3)} | "
                f"{_fmt(h.get('agree_G_when_actor_R'), 3)} | "
                f"{_fmt(h.get('agree_Gb_when_actor_R'), 3)} | "
                f"{_fmt(h.get('agree_Ag_when_actor_R'), 3)} |"
            )
    a("")
    a("### 8. Key findings")
    a("")
    a("#### Confirmed")
    a("")
    a("- Diagnosis used the frozen masked episode_0100 checkpoint only.")
    a("- Same-state KEEP/LEFT/RIGHT branches recorded G, G_boot, and branch GAE.")
    a("")
    a("#### Supported by evidence")
    a("")
    a(
        "- Whether critic bootstrap or branch GAE reverses the Phase 6.2 finding "
        "(bare G favors KEEP) is answered by Sections 3–5."
    )
    a(
        "- Whether actor lateral preference agrees with G / G_boot / A_gae is "
        "answered by Section 7."
    )
    a("")
    a("#### Not established")
    a("")
    a("- That PPO minibatch normalization or multi-epoch updates alter this picture.")
    a("- That observation features are insufficient.")
    a("- That the reward must be changed.")
    a("")
    a("#### Remaining unknowns")
    a("")
    a("- After-zone lane-1 sample count may be small under this masked traffic.")
    a("- Training uses next_values=0 at rollout end; this diagnosis bootstraps V(s_N).")
    a("- Other CAVs use KEEP inside branches, not the masked policy.")
    a("")
    a("### 9. Final conclusion")
    a("")
    near5 = out["by_zone_h"]["near"]["5s"]
    a(
        "1. Bare G still favor KEEP over LEFT/RIGHT in Near@5s? "
        f"mean ΔG L−K={_fmt((near5['dG_LK'] or {}).get('mean'))}, "
        f"ΔG R−K={_fmt((near5['dG_RK'] or {}).get('mean'))}."
    )
    a(
        "2. Does G_boot reverse that? "
        f"mean ΔGb L−K={_fmt((near5['dGb_LK'] or {}).get('mean'))}, "
        f"ΔGb R−K={_fmt((near5['dGb_RK'] or {}).get('mean'))}."
    )
    a(
        "3. Does branch GAE reverse that? "
        f"mean ΔAg L−K={_fmt((near5['dAg_LK'] or {}).get('mean'))}, "
        f"ΔAg R−K={_fmt((near5['dAg_RK'] or {}).get('mean'))}."
    )
    a(
        "4. Actor vs advantage agreement (Near@5s, when actor prefers L/R): "
        f"agree Gb L/R={_fmt(near5.get('agree_Gb_when_actor_L'), 3)}/"
        f"{_fmt(near5.get('agree_Gb_when_actor_R'), 3)}; "
        f"agree Ag L/R={_fmt(near5.get('agree_Ag_when_actor_L'), 3)}/"
        f"{_fmt(near5.get('agree_Ag_when_actor_R'), 3)}."
    )
    a(
        "5. If G, G_boot, and A_gae all favor KEEP while P(LEFT/RIGHT) stay high: "
        "Actor preference is inconsistent with critic-bootstrapped and GAE signals "
        "on this slice — next checks are PPO update path / observation, not reward edits."
    )
    a("")
    a("This phase stops here. No training. No reward edit. No Phase 6.4.")
    a("")
    a(f"Phase 6.3 Result: {status}")
    a("")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnostics").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)

    if not MASK_CKPT.is_file():
        raise FileNotFoundError(MASK_CKPT)
    ckpt_before = (MASK_CKPT.stat().st_mtime_ns, MASK_CKPT.stat().st_size)

    device = resolve_device(load_train_config().get("device", "auto"))
    print(f"Phase 6.3 device={device}", flush=True)
    model, _ckpt = load_mappo_checkpoint(MASK_CKPT, device=device)
    model.eval()
    before = snapshot_model_params(model)

    out = collect(model, device)
    rows = out.pop("rows")
    (OUT / "diagnostics" / "summary.json").write_text(
        json.dumps(_clean({**out, "n_rows": len(rows)}), indent=2),
        encoding="utf-8",
    )
    out["rows"] = rows

    frozen = params_unchanged(before, snapshot_model_params(model))
    ckpt_same = (MASK_CKPT.stat().st_mtime_ns, MASK_CKPT.stat().st_size) == ckpt_before
    write_tables(out)

    n_min = min(out["counts"]["before"], out["counts"]["near"])
    if not frozen or not ckpt_same or not out.get("save_ok"):
        status = "FAIL"
    elif n_min < TARGET_PER_REGION:
        status = "PARTIAL"
    else:
        # After may be scarce under masked traffic; before+near at target → PASS.
        status = "PASS" if out["counts"]["after"] >= 10 else "PARTIAL"

    write_report(out=out, frozen=frozen, ckpt_same=ckpt_same, status=status)
    print(f"Report: {OUT / 'report.md'}", flush=True)
    print(f"Phase 6.3 Result: {status}", flush=True)


if __name__ == "__main__":
    main()
