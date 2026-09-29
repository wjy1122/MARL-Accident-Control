"""Phase 4.4.5: Traffic congestion diagnosis around the 1200 m accident.

No MAPPO / Reward / Observation / SUMO scenario changes.
Policy: No Control (all CAV KEEP) — observe SUMO traffic physics only.

Example:
  .\\.venv\\Scripts\\python.exe scripts\\diagnose_traffic_congestion.py
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.action import KEEP  # noqa: E402
from env.highway_accident_env import HighwayAccidentEnv, load_config  # noqa: E402

def make_pos_bins(accident_pos: float, road_len: float) -> list[tuple[str, float, float]]:
    """Relative bins around accident (same offsets as Phase 4.4.5 @1200 on 2000m)."""
    acc = float(accident_pos)
    L = float(road_len)
    edges = [
        0.0,
        max(0.0, acc - 600.0),
        max(0.0, acc - 300.0),
        max(0.0, acc - 150.0),
        acc,
        min(L, acc + 150.0),
        min(L, acc + 300.0),
        L,
    ]
    # Deduplicate while preserving order
    uniq: list[float] = []
    for e in edges:
        if not uniq or abs(e - uniq[-1]) > 1e-6:
            uniq.append(e)
    bins: list[tuple[str, float, float]] = []
    for i in range(len(uniq) - 1):
        lo, hi = uniq[i], uniq[i + 1]
        if hi <= lo:
            continue
        bins.append((f"{int(round(lo))}_{int(round(hi))}", lo, hi))
    return bins


UPSTREAM_WINDOWS_M = (150.0, 300.0, 500.0)


def _empty_agg() -> dict[str, Any]:
    return {
        "n": 0,
        "speed_sum": 0.0,
        "speeds": [],
        "cav": 0,
        "hdv": 0,
        "slow": 0,
        "stopped": 0,
    }


def _accumulate(agg: dict[str, Any], speed: float, vtype: str, *, slow_mps: float) -> None:
    agg["n"] += 1
    agg["speed_sum"] += speed
    agg["speeds"].append(speed)
    if vtype == "CAV":
        agg["cav"] += 1
    elif vtype == "HDV":
        agg["hdv"] += 1
    if speed < slow_mps:
        agg["slow"] += 1
    if speed < 1.0:
        agg["stopped"] += 1


def _finalize(agg: dict[str, Any], *, vmax: float) -> dict[str, float]:
    n = int(agg["n"])
    if n == 0:
        return {
            "vehicle_count": 0.0,
            "mean_speed": float("nan"),
            "min_speed": float("nan"),
            "mean_speed_ratio": float("nan"),
            "CAV_count": 0.0,
            "HDV_count": 0.0,
            "slow_count": 0.0,
            "stopped_count": 0.0,
            "queue_length": 0.0,
        }
    speeds = agg["speeds"]
    return {
        "vehicle_count": float(n),
        "mean_speed": float(agg["speed_sum"] / n),
        "min_speed": float(min(speeds)),
        "mean_speed_ratio": float((agg["speed_sum"] / n) / vmax) if vmax > 0 else float("nan"),
        "CAV_count": float(agg["cav"]),
        "HDV_count": float(agg["hdv"]),
        "slow_count": float(agg["slow"]),
        "stopped_count": float(agg["stopped"]),
        # Instantaneous "queue" proxy used elsewhere: count of slow vehicles
        "queue_length": float(agg["slow"]),
    }


def run_diagnosis(
    *,
    sim_seconds: float,
    seed: int,
    gui: bool = False,
) -> dict[str, Any]:
    del seed  # SUMO scene is deterministic; kept for API symmetry
    env_cfg = copy.deepcopy(load_config())
    step_length = float(env_cfg.get("sumo", {}).get("step_length", 0.1))
    steps = max(1, int(round(sim_seconds / step_length)))
    env_cfg.setdefault("episode", {})["max_episode_steps"] = steps + 50

    accident_pos = float(env_cfg.get("accident", {}).get("position_m", 1200.0))
    road_len = float(env_cfg.get("highway", {}).get("length_m", 2000.0))
    pos_bins = make_pos_bins(accident_pos, road_len)
    vmax = float(
        env_cfg.get("observation", {}).get(
            "max_speed_mps",
            env_cfg.get("highway", {}).get("speed_limit_mps", 30.0),
        )
    )
    slow_mps = float(env_cfg.get("test", {}).get("congestion_speed_mps", 15.0))
    n_lanes = int(env_cfg.get("highway", {}).get("lanes", 3))

    bin_aggs = {name: _empty_agg() for name, _, _ in pos_bins}
    lane_aggs = {i: _empty_agg() for i in range(n_lanes)}
    # Upstream windows relative to accident (pos in [acc-W, acc])
    up_aggs = {w: _empty_agg() for w in UPSTREAM_WINDOWS_M}
    # Lane 0 specifically upstream [acc-250, acc] and all-lane lane0
    lane0_up_agg = _empty_agg()
    lane0_all_agg = _empty_agg()
    lane1_all_agg = _empty_agg()
    lane2_all_agg = _empty_agg()

    # Impact evidence counters (sightings over time)
    evidence = {
        "vehicles_on_lane0_upstream_slow": 0,
        "vehicles_decelerating_upstream_lane0": 0,  # speed < 0.9*vmax upstream on lane0
        "vehicles_on_lane1_or_2_near_accident": 0,  # possible diversion
        "max_queue_upstream_250": 0,
        "max_vehicles_lane0_upstream_250": 0,
        "accident_vehicle_seen": False,
        "accident_lane_index": None,
        "accident_position_obs": None,
    }
    prev_speeds: dict[str, float] = {}

    env = HighwayAccidentEnv(config=env_cfg, gui=gui)
    try:
        env.reset()
        for _ in range(steps):
            vehicles = env.get_vehicle_states()
            # Confirm accident vehicle
            for vid, st in vehicles.items():
                if st.get("type") == "ACCIDENT":
                    evidence["accident_vehicle_seen"] = True
                    evidence["accident_lane_index"] = int(st["lane"])
                    evidence["accident_position_obs"] = float(st["position"])

            # Per-step queue on upstream 250 m
            q250 = 0
            n_l0_up = 0

            for vid, st in vehicles.items():
                if st.get("type") == "ACCIDENT":
                    continue
                pos = float(st["position"])
                spd = float(st["speed"])
                lane = int(st["lane"])
                vtype = str(st.get("type", ""))

                for name, lo, hi in pos_bins:
                    if lo <= pos < hi:
                        _accumulate(bin_aggs[name], spd, vtype, slow_mps=slow_mps)
                        break

                if 0 <= lane < n_lanes:
                    _accumulate(lane_aggs[lane], spd, vtype, slow_mps=slow_mps)

                if lane == 0:
                    _accumulate(lane0_all_agg, spd, vtype, slow_mps=slow_mps)
                    if accident_pos - 250.0 <= pos <= accident_pos:
                        _accumulate(lane0_up_agg, spd, vtype, slow_mps=slow_mps)
                        n_l0_up += 1
                        if spd < slow_mps:
                            q250 += 1
                            evidence["vehicles_on_lane0_upstream_slow"] += 1
                        if spd < 0.9 * vmax:
                            evidence["vehicles_decelerating_upstream_lane0"] += 1
                elif lane == 1:
                    _accumulate(lane1_all_agg, spd, vtype, slow_mps=slow_mps)
                elif lane == 2:
                    _accumulate(lane2_all_agg, spd, vtype, slow_mps=slow_mps)

                if abs(pos - accident_pos) <= 150.0 and lane in (1, 2):
                    evidence["vehicles_on_lane1_or_2_near_accident"] += 1

                for w in UPSTREAM_WINDOWS_M:
                    if accident_pos - w <= pos <= accident_pos:
                        _accumulate(up_aggs[w], spd, vtype, slow_mps=slow_mps)

                prev = prev_speeds.get(vid)
                if (
                    prev is not None
                    and lane == 0
                    and accident_pos - 300.0 <= pos <= accident_pos
                    and spd + 0.5 < prev
                    and prev > 5.0
                ):
                    evidence["vehicles_decelerating_upstream_lane0"] += 0  # already counted via speed
                prev_speeds[vid] = spd

            evidence["max_queue_upstream_250"] = max(
                int(evidence["max_queue_upstream_250"]), q250
            )
            evidence["max_vehicles_lane0_upstream_250"] = max(
                int(evidence["max_vehicles_lane0_upstream_250"]), n_l0_up
            )

            # No Control
            actions = {cid: KEEP for cid in env.cav_ids}
            env.step(actions)
    finally:
        env.close()

    # Convert aggs → means. Counts are totals over all timesteps (sightings).
    # Also report per-step averages by dividing by steps.
    def sighting_to_report(agg: dict[str, Any]) -> dict[str, float]:
        fin = _finalize(agg, vmax=vmax)
        # Average occupancy proxy: mean vehicles per step in this set
        fin["mean_vehicles_per_step"] = float(agg["n"]) / float(steps)
        fin["mean_slow_per_step"] = float(agg["slow"]) / float(steps)
        return fin

    bins_out = {name: sighting_to_report(bin_aggs[name]) for name, _, _ in pos_bins}
    lanes_out = {f"lane_{i}": sighting_to_report(lane_aggs[i]) for i in range(n_lanes)}
    upstream_out = {
        f"upstream_{int(w)}m": sighting_to_report(up_aggs[w]) for w in UPSTREAM_WINDOWS_M
    }

    lane0 = sighting_to_report(lane0_all_agg)
    lane1 = sighting_to_report(lane1_all_agg)
    lane2 = sighting_to_report(lane2_all_agg)
    lane0_up = sighting_to_report(lane0_up_agg)
    up150 = sighting_to_report(up_aggs[150.0])
    up300 = sighting_to_report(up_aggs[300.0])
    up500 = sighting_to_report(up_aggs[500.0])
    # Nearest upstream bin: [acc-150, acc]
    upstream_bin_name = None
    for name, lo, hi in pos_bins:
        if abs(hi - accident_pos) < 1e-6 and lo < hi:
            upstream_bin_name = name
            break
    if upstream_bin_name is None and pos_bins:
        # Fallback: bin whose hi is closest to accident from below
        candidates = [(name, lo, hi) for name, lo, hi in pos_bins if hi <= accident_pos + 1e-6]
        upstream_bin_name = candidates[-1][0] if candidates else pos_bins[0][0]
    upstream_bin = bins_out[upstream_bin_name]

    # Heuristic: "obvious congestion"
    mean_up_speed = upstream_bin["mean_speed"]
    mean_l0_up = lane0_up["mean_speed"]
    max_q = int(evidence["max_queue_upstream_250"])
    obvious = (
        (np.isfinite(mean_up_speed) and mean_up_speed < 0.7 * vmax)
        or (np.isfinite(mean_l0_up) and mean_l0_up < 0.7 * vmax)
        or max_q >= 5
    )
    mild = (
        not obvious
        and (
            (np.isfinite(mean_l0_up) and mean_l0_up < 0.95 * vmax)
            or max_q >= 1
            or int(evidence["vehicles_on_lane0_upstream_slow"]) > 0
        )
    )

    if obvious:
        congestion_label = "YES (obvious)"
    elif mild:
        congestion_label = "WEAK / intermittent (not obvious)"
    else:
        congestion_label = "NO clear congestion"

    # Adequacy for MARL accident-control
    if obvious:
        adequate = "Partially yes — congestion exists; still verify reward sensitivity"
        cause = "n/a"
    elif mild:
        adequate = (
            "Likely insufficient for strong accident-control learning signal: "
            "disturbance is weak, near-zone speeds stay near vmax"
        )
        cause = (
            "Most likely: vehicles divert around blocked lane 0 early / demand not "
            "enough to force queue on remaining lanes; accident blocks only 1 of 3 lanes; "
            "depart_speed=max keeps free-flow until close approach"
        )
    else:
        adequate = "No — scenario does not produce usable congestion for MARL"
        cause = (
            "Traffic demand and/or single-lane blockage insufficient to create upstream queue; "
            "check vehs_per_hour, lane availability, and whether ACCIDENT truly occupies lane 0"
        )

    report = {
        "sim_seconds": sim_seconds,
        "steps": steps,
        "step_length": step_length,
        "accident_pos_m": accident_pos,
        "vmax_mps": vmax,
        "slow_threshold_mps": slow_mps,
        "vehs_per_hour": env_cfg.get("traffic", {}).get("vehs_per_hour"),
        "cav_penetration": env_cfg.get("traffic", {}).get("cav_penetration"),
        "blocked_lane_indices": env_cfg.get("accident", {}).get("blocked_lane_indices"),
        "position_bins": bins_out,
        "lanes": lanes_out,
        "lane0": lane0,
        "lane1": lane1,
        "lane2": lane2,
        "lane0_upstream_250m": lane0_up,
        "upstream_windows": upstream_out,
        "evidence": evidence,
        "congestion_label": congestion_label,
        "adequate_for_marl": adequate,
        "insufficiency_cause": cause,
        "summary": {
            "upstream_bin_name": upstream_bin_name,
            "upstream_bin_mean_speed": upstream_bin["mean_speed"],
            "upstream_bin_mean_speed_ratio": upstream_bin["mean_speed_ratio"],
            # Backward-compatible aliases (same values; name may no longer be 1050_1200)
            "upstream_1050_1200_mean_speed": upstream_bin["mean_speed"],
            "upstream_1050_1200_mean_speed_ratio": upstream_bin["mean_speed_ratio"],
            "lane0_mean_speed": lane0["mean_speed"],
            "lane1_mean_speed": lane1["mean_speed"],
            "lane2_mean_speed": lane2["mean_speed"],
            "lane0_upstream_250_mean_speed": lane0_up["mean_speed"],
            "upstream_150_mean_speed": up150["mean_speed"],
            "upstream_300_mean_speed": up300["mean_speed"],
            "upstream_500_mean_speed": up500["mean_speed"],
            "max_queue_upstream_250": max_q,
            "mean_slow_per_step_upstream_250_lane0": lane0_up["mean_slow_per_step"],
            "total_slow_sightings_all": sum(int(lane_aggs[i]["slow"]) for i in range(n_lanes)),
            "total_stopped_sightings_all": sum(
                int(lane_aggs[i]["stopped"]) for i in range(n_lanes)
            ),
        },
        "pos_bin_names": [name for name, _, _ in pos_bins],
    }
    return report


def _print_report(r: dict[str, Any]) -> None:
    s = r["summary"]
    print("==============================")
    print("Phase 4.4.5 Traffic Diagnosis")
    print("==============================")
    print(f"sim_seconds={r['sim_seconds']} steps={r['steps']} vmax={r['vmax_mps']}")
    print(f"accident_pos={r['accident_pos_m']} blocked_lanes={r['blocked_lane_indices']}")
    print(f"demand={r['vehs_per_hour']} veh/h  cav_pen={r['cav_penetration']}")
    print(f"ACCIDENT seen={r['evidence']['accident_vehicle_seen']} "
          f"lane={r['evidence']['accident_lane_index']} "
          f"pos={r['evidence']['accident_position_obs']}")
    print()
    print("--- Position bins (sighting means) ---")
    for name in r.get("pos_bin_names", list(r["position_bins"].keys())):
        b = r["position_bins"][name]
        print(
            f"  {name:12s}: n_sight={b['vehicle_count']:.0f} "
            f"mean_v={b['mean_speed']:.2f} min_v={b['min_speed']:.2f} "
            f"ratio={b['mean_speed_ratio']:.3f} "
            f"CAV={b['CAV_count']:.0f} HDV={b['HDV_count']:.0f} "
            f"slow/step={b['mean_slow_per_step']:.3f}"
        )
    print()
    print("--- Lanes ---")
    for key in ("lane0", "lane1", "lane2"):
        b = r[key]
        print(
            f"  {key}: mean_v={b['mean_speed']:.2f} ratio={b['mean_speed_ratio']:.3f} "
            f"veh/step={b['mean_vehicles_per_step']:.2f} "
            f"slow/step={b['mean_slow_per_step']:.3f} "
            f"queue_proxy_sightings={b['queue_length']:.0f}"
        )
    print()
    print("--- Upstream windows (all lanes) ---")
    for w in UPSTREAM_WINDOWS_M:
        b = r["upstream_windows"][f"upstream_{int(w)}m"]
        print(
            f"  {int(w)}m: mean_v={b['mean_speed']:.2f} ratio={b['mean_speed_ratio']:.3f} "
            f"slow/step={b['mean_slow_per_step']:.3f}"
        )
    print()
    print("1. 是否形成明显拥堵:", r["congestion_label"])
    up_name = s.get("upstream_bin_name", "upstream")
    print(
        f"2. 事故上游平均速度 ({up_name}m): "
        f"{s['upstream_bin_mean_speed']:.3f} m/s"
    )
    print(f"3. lane 0 平均速度: {s['lane0_mean_speed']:.3f} m/s")
    print(f"4. lane 1 平均速度: {s['lane1_mean_speed']:.3f} m/s")
    print(f"5. lane 2 平均速度: {s['lane2_mean_speed']:.3f} m/s")
    print(f"6. queue length (max slow in lane0 upstream 250m): {s['max_queue_upstream_250']}")
    print(
        f"7. slow/stopped sightings (all lanes): "
        f"slow={s['total_slow_sightings_all']} stopped={s['total_stopped_sightings_all']}"
    )
    print("8. accident impact evidence:")
    print(f"   lane0 upstream 250m mean_v={s['lane0_upstream_250_mean_speed']:.3f}")
    print(f"   upstream 150/300/500m mean_v="
          f"{s['upstream_150_mean_speed']:.3f}/"
          f"{s['upstream_300_mean_speed']:.3f}/"
          f"{s['upstream_500_mean_speed']:.3f}")
    print(f"   lane0 upstream slow sightings={r['evidence']['vehicles_on_lane0_upstream_slow']}")
    print(f"   near-accident on lane1/2 sightings={r['evidence']['vehicles_on_lane1_or_2_near_accident']}")
    print(f"   max vehicles on lane0 upstream250={r['evidence']['max_vehicles_lane0_upstream_250']}")
    print("9. 当前场景是否足以用于 MARL accident-control experiment:")
    print(f"   {r['adequate_for_marl']}")
    print("10. 如果不足，原因:")
    print(f"   {r['insufficiency_cause']}")
    print("==============================")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4.4.5 congestion diagnosis")
    parser.add_argument("--sim-seconds", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    print("Phase 4.4.5: Traffic Congestion Diagnosis")
    print("Policy: No Control (KEEP). No scene/reward/MAPPO changes.")
    print()

    report = run_diagnosis(
        sim_seconds=args.sim_seconds, seed=args.seed, gui=args.gui
    )
    _print_report(report)

    out_dir = PROJECT_ROOT / "results" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "phase_445_traffic.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Flat CSV for bins
    csv_path = out_dir / "phase_445_position_bins.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        rows = []
        for name in report.get("pos_bin_names", list(report["position_bins"].keys())):
            row = {"bin": name, **report["position_bins"][name]}
            rows.append(row)
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
