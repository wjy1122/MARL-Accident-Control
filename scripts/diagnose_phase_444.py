"""Phase 4.4.4: near-zone reward / speed / action diagnosis (no training changes).

Does NOT modify Reward, Observation, MAPPO, or SUMO.

Example:
  .\\.venv\\Scripts\\python.exe scripts\\diagnose_phase_444.py
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.rollout import load_train_config, resolve_device  # noqa: E402
from env.action import ACTION_NAMES, KEEP  # noqa: E402
from env.highway_accident_env import HighwayAccidentEnv, load_config  # noqa: E402
from models.mappo_model import MAPPOModel, load_model_config  # noqa: E402
from utils.zone_behavior_diagnostics import ZoneRewardDiag  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _warmup(env: HighwayAccidentEnv, n: int) -> None:
    for _ in range(n):
        if len(env.cav_ids) > 0:
            return
        env.step()


def run_diagnostic_rollout(
    *,
    env: HighwayAccidentEnv,
    model: MAPPOModel,
    device: torch.device,
    rollout_steps: int,
    warmup_steps: int,
    num_agents: int,
    diag: ZoneRewardDiag,
) -> dict[str, float]:
    """One MAPPO→env rollout; accumulate near/far reward/speed/action stats."""
    model.eval()
    env.reset()
    _warmup(env, warmup_steps)

    agent_ids = sorted(env.cav_ids)[:num_agents]
    while len(agent_ids) < num_agents:
        agent_ids.append(None)  # type: ignore[arg-type]

    max_pos = 0.0
    min_pos = float("inf")
    near_steps = 0
    valid_steps = 0

    for _ in range(rollout_steps):
        if len(env.cav_ids) == 0:
            env.reset()
            _warmup(env, warmup_steps)
            agent_ids = sorted(env.cav_ids)[:num_agents]
            while len(agent_ids) < num_agents:
                agent_ids.append(None)  # type: ignore[arg-type]

        # Snapshot lanes BEFORE action (for LC outcome check)
        lane_before: dict[str, int] = {}
        vehicles = env.get_vehicle_states()
        for cid in env.cav_ids:
            if cid in vehicles:
                lane_before[cid] = int(vehicles[cid]["lane"])

        actions_dict: dict[str, int] = {}
        raw_by_id: dict[str, int] = {}
        with torch.no_grad():
            for aid in agent_ids:
                if aid is None or aid not in env.cav_ids:
                    continue
                obs = env.get_observation(aid)
                t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                action, _lp = model.act(t, deterministic=False)
                a = int(action.item())
                actions_dict[aid] = a
                raw_by_id[aid] = a

        _obs, _rew, _term, truncated, info = env.step(actions_dict)
        del _obs, _rew, _term
        reward_details: dict[str, dict[str, float]] = info.get("reward_details", {})

        vehicles_after = env.get_vehicle_states()
        for cid, raw_a in raw_by_id.items():
            if cid not in vehicles_after:
                continue
            st = vehicles_after[cid]
            if st.get("type") != "CAV":
                continue
            pos = float(st["position"])
            spd = float(st["speed"])
            detail = reward_details.get(cid)
            if detail is None:
                continue
            valid_steps += 1
            max_pos = max(max_pos, pos)
            min_pos = min(min_pos, pos)
            if diag.is_near(pos):
                near_steps += 1
            diag.add_sample(
                pos=pos,
                speed=spd,
                individual=float(detail["individual"]),
                group=float(detail["group"]),
                total=float(detail["total"]),
                raw_action=raw_a,
                lane_before=lane_before.get(cid, int(st["lane"])),
                lane_after=int(st["lane"]),
            )

        if truncated:
            env.reset()
            _warmup(env, warmup_steps)
            agent_ids = sorted(env.cav_ids)[:num_agents]
            while len(agent_ids) < num_agents:
                agent_ids.append(None)  # type: ignore[arg-type]

    return {
        "rollout_steps": float(rollout_steps),
        "max_pos": float(max_pos if max_pos > 0 else float("nan")),
        "min_pos": float(min_pos if min_pos < float("inf") else float("nan")),
        "near_frac": float(near_steps / valid_steps) if valid_steps else 0.0,
        "near_steps": float(near_steps),
        "valid_steps": float(valid_steps),
    }


def print_report(coverage: dict[str, float], summary: dict[str, Any]) -> None:
    near_act = summary["near_action_pct"]
    print("==============================")
    print("Phase 4.4.4 Diagnostic")
    print("==============================")
    print()
    print("Coverage")
    print(f"max_pos = {coverage['max_pos']:.1f}")
    print(f"near_frac = {coverage['near_frac']:.4f}")
    print(f"near_steps = {int(coverage['near_steps'])}")
    print()
    print("Reward")
    print(f"global_mean_reward = {summary['global_mean_total']:.6f}")
    print(f"near_mean_reward = {summary['near_mean_total']:.6f}")
    print(f"far_mean_reward = {summary['far_mean_total']:.6f}")
    print(f"near_mean_individual = {summary['near_mean_individual']:.6f}")
    print(f"near_mean_group = {summary['near_mean_group']:.6f}")
    print(f"far_mean_individual = {summary['far_mean_individual']:.6f}")
    print(f"far_mean_group = {summary['far_mean_group']:.6f}")
    print()
    print("Speed")
    print(f"near_mean_speed = {summary['near_mean_speed']:.4f}")
    print(f"near_min_speed = {summary['near_min_speed']:.4f}")
    print(f"near_max_speed = {summary['near_max_speed']:.4f}")
    print(f"near_mean_speed_ratio = {summary['near_mean_speed_ratio']:.6f}")
    print(f"far_mean_speed_ratio = {summary['far_mean_speed_ratio']:.6f}")
    print()
    print("Action distribution in near accident zone")
    print(f"  0=KEEP:  {near_act.get('KEEP', float('nan')):.1f}%")
    print(f"  1=LEFT:  {near_act.get('LEFT', float('nan')):.1f}%")
    print(f"  2=RIGHT: {near_act.get('RIGHT', float('nan')):.1f}%")
    print(f"  n={int(near_act.get('n', 0))}")
    print()
    print("Boundary no-op (only filter in codebase; no Safety Supervisor)")
    print(f"raw LC actions (LEFT/RIGHT) = {summary['lc_raw_n']}")
    print(f"boundary blocked = {summary['lc_boundary_blocked_n']}")
    print(f"filter rate = {summary['boundary_block_rate']}")
    print(f"near LC raw = {summary['near_lc_raw_n']}")
    print(f"near boundary blocked = {summary['near_lc_boundary_blocked_n']}")
    print(f"near LC lane changed after step = {summary['near_lc_lane_changed_n']}")
    print(f"near LC lane unchanged after step = {summary['near_lc_lane_unchanged_n']}")
    print()
    print("Reward components present:", summary["reward_components_present"])
    print("Reward components absent:", summary["reward_components_absent"])
    print("==============================")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4.4.4 near-zone diagnostics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()

    train_cfg = load_train_config()
    tblock = train_cfg.get("train", train_cfg)
    model_cfg = load_model_config().get("model", {})
    r_steps = int(
        args.rollout_steps
        if args.rollout_steps is not None
        else tblock.get("rollout_steps", 512)
    )
    warmup = int(tblock.get("warmup_steps", 50))
    seed = int(args.seed)
    set_seed(seed)
    device = resolve_device(train_cfg.get("device", "auto"))

    print("================================")
    print("Phase 4.4.4 Launcher")
    print("================================")
    print(f"seed = {seed}")
    print(f"rollout_steps = {r_steps}")
    print(f"updates = {args.updates}")
    print(f"device = {device}")
    print()

    env_cfg = copy.deepcopy(load_config())
    # Ensure episode horizon covers a 512-step collect without truncate mid-way.
    env_cfg.setdefault("episode", {})["max_episode_steps"] = max(
        int(env_cfg.get("episode", {}).get("max_episode_steps", 1000)),
        r_steps + warmup + 50,
    )
    env = HighwayAccidentEnv(config=env_cfg, gui=False)

    model = MAPPOModel(
        obs_dim=int(model_cfg.get("observation_dim", 17)),
        action_dim=int(model_cfg.get("action_dim", 3)),
        num_agents=int(model_cfg.get("num_agents", 4)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
    ).to(device)
    if args.checkpoint:
        path = Path(args.checkpoint)
        if not path.is_file():
            path = PROJECT_ROOT / args.checkpoint
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.actor.load_state_dict(ckpt["actor_state_dict"])
        model.critic.load_state_dict(ckpt["critic_state_dict"])
        print(f"Loaded checkpoint: {path}")
    else:
        print("Using freshly initialized MAPPO weights (no checkpoint)")

    diag = ZoneRewardDiag(
        accident_pos_m=float(env.accident_position_m),
        near_m=float(env.cfg.get("observation", {}).get("range_m", 150.0)),
        vmax_mps=float(env.reward_max_speed_mps),
        n_lanes=int(env.n_lanes),
        left_increases_index=bool(env.left_increases_index),
    )

    coverages: list[dict[str, float]] = []
    try:
        for u in range(1, args.updates + 1):
            print(f"--- diagnostic update {u}/{args.updates} ---")
            cov = run_diagnostic_rollout(
                env=env,
                model=model,
                device=device,
                rollout_steps=r_steps,
                warmup_steps=warmup,
                num_agents=int(model.num_agents),
                diag=diag,
            )
            coverages.append(cov)
            print(
                f"  max_pos={cov['max_pos']:.1f} near_frac={cov['near_frac']:.3f} "
                f"near_steps={int(cov['near_steps'])}"
            )
    finally:
        env.close()

    summary = diag.summary()
    # Aggregate coverage across updates (last max / pooled near)
    agg_cov = {
        "max_pos": float(max(c["max_pos"] for c in coverages)),
        "near_frac": float(summary["near_frac"]),
        "near_steps": float(summary["near_n"]),
    }
    print_report(agg_cov, summary)

    out_dir = PROJECT_ROOT / "results" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "phase_444_summary.json"
    payload = {
        "seed": seed,
        "rollout_steps": r_steps,
        "updates": args.updates,
        "coverage_per_update": coverages,
        "summary": summary,
        "action_names": ACTION_NAMES,
        "reward_formula": (
            "R = 0.7 * clip(v/vmax,0,1) + 0.3 * mean_neighbor_ind "
            "(fallback ego_ind if no neighbors)"
        ),
        "accident_zone_defs": {
            "diagnostic": "abs(pos - accident_pos) <= 150",
            "reward": "NONE (reward has no accident-zone condition)",
            "sumo_accident": "static ACCIDENT vehicle at position_m from env.yaml",
        },
    }
    # Make JSON-safe
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj

    json_path.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
    print(f"Saved: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
