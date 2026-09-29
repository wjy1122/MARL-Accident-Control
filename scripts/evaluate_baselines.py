"""Closed-loop evaluation: No Control (KEEP) vs MAPPO.

Does not modify Reward / SUMO / Observation. Logging + comparison only.

Examples:
  .\\.venv\\Scripts\\python.exe scripts\\evaluate_baselines.py --mode no_control
  .\\.venv\\Scripts\\python.exe scripts\\evaluate_baselines.py --mode both --sim-seconds 120
  .\\.venv\\Scripts\\python.exe scripts\\evaluate_baselines.py --mode mappo --checkpoint results/mappo_group_benefit/checkpoints/episode_0010.pt
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.rollout import resolve_device  # noqa: E402
from env.action import KEEP, LEFT, RIGHT  # noqa: E402
from env.highway_accident_env import HighwayAccidentEnv, load_config  # noqa: E402
from models.mappo_model import MAPPOModel, load_model_config  # noqa: E402
from utils.traffic_diagnostics import (  # noqa: E402
    EvalMetrics,
    compute_ttc_front,
    format_action_share_line,
)


def _cfg_floats(env: HighwayAccidentEnv) -> dict[str, float]:
    cfg = env.cfg
    accident = cfg.get("accident", {})
    highway = cfg.get("highway", {})
    obs = cfg.get("observation", {})
    test = cfg.get("test", {})
    return {
        "accident_pos": float(accident.get("position_m", 1200.0)),
        "road_len": float(highway.get("length_m", 2000.0)),
        "max_speed": float(
            obs.get("max_speed_mps", highway.get("speed_limit_mps", 30.0))
        ),
        "range_m": float(obs.get("range_m", 150.0)),
        "upstream_m": float(test.get("congestion_upstream_m", 250.0)),
        "slow_mps": float(test.get("congestion_speed_mps", 15.0)),
        "step_length": float(cfg.get("sumo", {}).get("step_length", 0.1)),
    }


def _queue_length(env: HighwayAccidentEnv, p: dict[str, float]) -> int:
    lo = p["accident_pos"] - p["upstream_m"]
    n = 0
    for st in env.get_vehicle_states().values():
        if st.get("type") == "ACCIDENT":
            continue
        pos = float(st["position"])
        if lo <= pos <= p["accident_pos"] and float(st["speed"]) < p["slow_mps"]:
            n += 1
    return n


def _cav_near_accident(env: HighwayAccidentEnv, p: dict[str, float]) -> int:
    n = 0
    for st in env.get_cav_states().values():
        if abs(float(st["position"]) - p["accident_pos"]) <= p["range_m"]:
            n += 1
    return n


def _leader_on_lane(
    vehicles: dict[str, dict[str, Any]], ego_id: str
) -> tuple[float | None, float | None]:
    ego = vehicles[ego_id]
    ego_pos = float(ego["position"])
    ego_lane = int(ego["lane"])
    best_pos = None
    best_speed = None
    for vid, st in vehicles.items():
        if vid == ego_id or st.get("type") == "ACCIDENT":
            continue
        if int(st["lane"]) != ego_lane:
            continue
        pos = float(st["position"])
        if pos <= ego_pos:
            continue
        if best_pos is None or pos < best_pos:
            best_pos = pos
            best_speed = float(st["speed"])
    return best_pos, best_speed


def _policy_no_control(env: HighwayAccidentEnv, _model: Any) -> dict[str, int]:
    return {cid: KEEP for cid in env.cav_ids}


def _policy_mappo(
    env: HighwayAccidentEnv, model: MAPPOModel, device: torch.device
) -> dict[str, int]:
    obs_dict = env.get_observations()
    out: dict[str, int] = {}
    model.eval()
    with torch.no_grad():
        for cid, obs in obs_dict.items():
            t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            action, _ = model.act(t, deterministic=False)
            out[cid] = int(action.item())
    return out


def run_eval(
    *,
    mode: str,
    sim_seconds: float,
    gui: bool = False,
    model: MAPPOModel | None = None,
    device: torch.device | None = None,
    seed: int = 42,
) -> EvalMetrics:
    """Run one evaluation horizon. mode in {no_control, mappo}."""
    env_cfg = copy.deepcopy(load_config())
    # Long enough horizon for vehicles to reach / pass the accident zone.
    max_steps = int(round(sim_seconds / float(env_cfg["sumo"]["step_length"]))) + 5
    env_cfg.setdefault("episode", {})["max_episode_steps"] = max_steps
    env = HighwayAccidentEnv(config=env_cfg, gui=gui)

    p = _cfg_floats(env)
    steps = max(1, int(round(sim_seconds / p["step_length"])))
    device = device or torch.device("cpu")

    if mode == "no_control":
        policy: Callable[[HighwayAccidentEnv, Any], dict[str, int]] = (
            lambda e, m: _policy_no_control(e, m)
        )
    elif mode == "mappo":
        if model is None:
            raise ValueError("mappo mode requires a model")
        policy = lambda e, m: _policy_mappo(e, m, device)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    speed_all: list[float] = []
    speed_cav: list[float] = []
    speed_hdv: list[float] = []
    speed_before: list[float] = []
    speed_after: list[float] = []
    reward_before: list[float] = []
    reward_after: list[float] = []
    rewards_all: list[float] = []
    queues: list[int] = []
    near_cavs: list[int] = []
    ttc_vals: list[float] = []
    collisions = 0
    arrived = 0

    act_counts = np.zeros(3, dtype=np.int64)
    near_counts = np.zeros(3, dtype=np.int64)

    # Travel time: first-seen time → arrival (leave network)
    first_seen: dict[str, float] = {}
    travel_times: list[float] = []
    prev_ids: set[str] = set()

    try:
        env.reset()
        # Optional short settle (not a full warmup to free-flow only).
        for _ in range(10):
            env.step()

        for _ in range(steps):
            traci = env._require_traci()
            sim_t = float(traci.simulation.getTime())

            # Arrivals / departures bookkeeping
            cur_ids = set(traci.vehicle.getIDList())
            for vid in cur_ids - prev_ids:
                if not str(vid).startswith(
                    str(env.cfg.get("accident", {}).get("vehicle_id_prefix", "ACCIDENT"))
                ):
                    first_seen.setdefault(vid, sim_t)
            for vid in prev_ids - cur_ids:
                if vid in first_seen:
                    travel_times.append(sim_t - first_seen[vid])
                    arrived += 1
                    del first_seen[vid]
            prev_ids = cur_ids

            collisions += int(traci.simulation.getCollidingVehiclesNumber())

            vehicles = env.get_vehicle_states()
            for vid, st in vehicles.items():
                if st["type"] == "ACCIDENT":
                    continue
                spd = float(st["speed"])
                pos = float(st["position"])
                speed_all.append(spd)
                if st["type"] == "CAV":
                    speed_cav.append(spd)
                elif st["type"] == "HDV":
                    speed_hdv.append(spd)
                if pos < p["accident_pos"]:
                    speed_before.append(spd)
                else:
                    speed_after.append(spd)

                if st["type"] in ("CAV", "HDV"):
                    lead_pos, lead_spd = _leader_on_lane(vehicles, vid)
                    ttc = compute_ttc_front(pos, spd, lead_pos, lead_spd)
                    if ttc is not None and ttc < 60.0:
                        ttc_vals.append(float(ttc))

            # CAV rewards by zone
            for cid, detail in env.get_rewards().items():
                st = vehicles.get(cid)
                if st is None:
                    continue
                tot = float(detail["total"])
                rewards_all.append(tot)
                if float(st["position"]) < p["accident_pos"]:
                    reward_before.append(tot)
                else:
                    reward_after.append(tot)

            queues.append(_queue_length(env, p))
            near_cavs.append(_cav_near_accident(env, p))

            # Actions
            actions = policy(env, model)
            for cid, a in actions.items():
                a_i = int(a)
                act_counts[a_i] += 1
                st = vehicles.get(cid)
                if st is not None and abs(float(st["position"]) - p["accident_pos"]) <= p[
                    "range_m"
                ]:
                    near_counts[a_i] += 1

            env.step(actions)
    finally:
        env.close()

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    act_n = int(act_counts.sum())
    near_n = int(near_counts.sum())

    metrics = EvalMetrics(
        mode=mode,
        sim_seconds=float(sim_seconds),
        steps=steps,
        average_speed_mps=_mean(speed_all),
        average_cav_speed_mps=_mean(speed_cav),
        average_hdv_speed_mps=_mean(speed_hdv),
        speed_before_mps=_mean(speed_before),
        speed_after_mps=_mean(speed_after),
        cav_reward_before=_mean(reward_before),
        cav_reward_after=_mean(reward_after),
        queue_length_mean=_mean([float(x) for x in queues]),
        queue_length_max=float(max(queues) if queues else 0),
        cav_near_accident_mean=_mean([float(x) for x in near_cavs]),
        throughput=float(arrived),
        travel_time_mean_s=_mean(travel_times),
        collisions=int(collisions),
        ttc_min_s=float(min(ttc_vals)) if ttc_vals else float("inf"),
        ttc_mean_s=_mean(ttc_vals) if ttc_vals else float("nan"),
        action_keep_pct=100.0 * act_counts[KEEP] / max(act_n, 1),
        action_left_pct=100.0 * act_counts[LEFT] / max(act_n, 1),
        action_right_pct=100.0 * act_counts[RIGHT] / max(act_n, 1),
        near_keep_pct=100.0 * near_counts[KEEP] / max(near_n, 1),
        near_left_pct=100.0 * near_counts[LEFT] / max(near_n, 1),
        near_right_pct=100.0 * near_counts[RIGHT] / max(near_n, 1),
        near_action_n=near_n,
        reward_mean=_mean(rewards_all),
    )
    if mode == "no_control":
        metrics.notes.append("All CAV actions forced to KEEP (SUMO car-following)")
    if metrics.near_action_n == 0:
        metrics.notes.append("No CAV samples within 150m of accident — extend sim_seconds")
    _ = seed  # reserved for future seeding of eval runs
    return metrics


def _print_metrics(m: EvalMetrics) -> None:
    print(f"\n--- {m.mode} ({m.sim_seconds:.0f}s, {m.steps} steps) ---")
    print(f"Average Speed (all): {m.average_speed_mps:.3f} m/s")
    print(f"Average Speed (CAV): {m.average_cav_speed_mps:.3f} m/s")
    print(f"Speed before accident: {m.speed_before_mps:.3f} m/s")
    print(f"Speed after accident:  {m.speed_after_mps:.3f} m/s")
    print(f"CAV reward before: {m.cav_reward_before:.4f}")
    print(f"CAV reward after:  {m.cav_reward_after:.4f}")
    print(f"Queue Length mean/max: {m.queue_length_mean:.2f} / {m.queue_length_max:.0f}")
    print(f"CAV near accident (mean): {m.cav_near_accident_mean:.2f}")
    print(f"Throughput (arrivals): {m.throughput:.0f}")
    print(f"Travel Time mean: {m.travel_time_mean_s:.2f} s")
    print(f"Collisions: {m.collisions}")
    print(f"TTC min/mean: {m.ttc_min_s:.3f} / {m.ttc_mean_s:.3f} s")
    print(f"Reward mean: {m.reward_mean:.4f}")
    print(
        format_action_share_line(
            "Actions:", m.action_keep_pct, m.action_left_pct, m.action_right_pct
        )
    )
    print(
        format_action_share_line(
            "Near accident (<150m):",
            m.near_keep_pct,
            m.near_left_pct,
            m.near_right_pct,
        )
        + f"  n={m.near_action_n}"
    )
    if m.notes:
        print("Notes:", "; ".join(m.notes))


def _print_comparison(nc: EvalMetrics, mp: EvalMetrics) -> None:
    rows = [
        ("Average Speed", f"{nc.average_speed_mps:.3f}", f"{mp.average_speed_mps:.3f}"),
        ("Travel Time", f"{nc.travel_time_mean_s:.2f}", f"{mp.travel_time_mean_s:.2f}"),
        ("Throughput", f"{nc.throughput:.0f}", f"{mp.throughput:.0f}"),
        ("Queue Length", f"{nc.queue_length_mean:.2f}", f"{mp.queue_length_mean:.2f}"),
        ("Collision", f"{nc.collisions}", f"{mp.collisions}"),
        ("TTC (mean)", f"{nc.ttc_mean_s:.3f}", f"{mp.ttc_mean_s:.3f}"),
        ("Reward mean", f"{nc.reward_mean:.4f}", f"{mp.reward_mean:.4f}"),
    ]
    print("\n================================")
    print("No Control vs MAPPO")
    print("================================")
    print(f"{'Metric':<18} {'No Control':>12} {'MAPPO':>12}")
    print("-" * 44)
    for name, a, b in rows:
        print(f"{name:<18} {a:>12} {b:>12}")
    print("================================")


def main() -> int:
    parser = argparse.ArgumentParser(description="No Control vs MAPPO evaluation")
    parser.add_argument(
        "--mode",
        choices=["no_control", "mappo", "both"],
        default="both",
    )
    parser.add_argument("--sim-seconds", type=float, default=120.0)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = resolve_device("auto")
    print("================================")
    print("Baseline Evaluation")
    print("================================")
    print(f"Device: {device}")
    print(f"Sim seconds: {args.sim_seconds}")
    print(f"Mode: {args.mode}")

    model = None
    if args.mode in ("mappo", "both"):
        mcfg = load_model_config().get("model", {})
        model = MAPPOModel(
            obs_dim=int(mcfg.get("observation_dim", 17)),
            action_dim=int(mcfg.get("action_dim", 3)),
            num_agents=int(mcfg.get("num_agents", 4)),
            hidden_dim=int(mcfg.get("hidden_dim", 128)),
        ).to(device)
        if args.checkpoint:
            ckpt_path = Path(args.checkpoint)
            if not ckpt_path.is_file():
                ckpt_path = PROJECT_ROOT / args.checkpoint
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.actor.load_state_dict(ckpt["actor_state_dict"])
            model.critic.load_state_dict(ckpt["critic_state_dict"])
            print(f"Loaded checkpoint: {ckpt_path} (episode={ckpt.get('episode')})")
        else:
            print("MAPPO: random/untrained weights (no --checkpoint)")

    out_dir = PROJECT_ROOT / "results" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[EvalMetrics] = []
    if args.mode in ("no_control", "both"):
        nc = run_eval(
            mode="no_control",
            sim_seconds=args.sim_seconds,
            gui=args.gui,
            seed=args.seed,
        )
        _print_metrics(nc)
        results.append(nc)

    if args.mode in ("mappo", "both"):
        mp = run_eval(
            mode="mappo",
            sim_seconds=args.sim_seconds,
            gui=args.gui,
            model=model,
            device=device,
            seed=args.seed,
        )
        _print_metrics(mp)
        results.append(mp)

    if args.mode == "both" and len(results) == 2:
        _print_comparison(results[0], results[1])

    csv_path = out_dir / f"compare_{args.mode}_{int(args.sim_seconds)}s.csv"
    if results:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].to_dict().keys()))
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_dict())
        print(f"\nSaved: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
