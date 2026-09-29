"""Phase 4.4.6: Traffic demand strength scan (No Control only).

Sweeps vehs_per_hour while keeping accident / network / CAV penetration fixed.
Regenerates sumo/highway.rou.xml per demand, then restores env.yaml demand.

Does NOT train MAPPO. Does NOT modify Reward / Observation / MAPPO.

Example:
  .\\.venv\\Scripts\\python.exe scripts\\scan_traffic_demand.py
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

# Import builders / diagnostics from scripts/ without package install
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from build_sumo_scenario import build_routes, load_config as load_build_config  # noqa: E402
from diagnose_traffic_congestion import (  # noqa: E402
    POS_BINS,
    _accumulate,
    _empty_agg,
    _finalize,
)

from env.action import KEEP  # noqa: E402
from env.highway_accident_env import HighwayAccidentEnv, load_config  # noqa: E402


def classify_scenario(
    *,
    upstream_ratio: float,
    max_queue: int,
    stopped_sightings: int,
    lane0_ratio: float,
    lane2_ratio: float,
) -> str:
    """Data-driven label — not forced toward a preferred outcome."""
    # SEVERE: strong collapse / queueing
    if (
        (np.isfinite(upstream_ratio) and upstream_ratio < 0.40)
        or max_queue >= 15
        or stopped_sightings >= 500
        or (np.isfinite(lane0_ratio) and lane0_ratio < 0.35)
    ):
        return "SEVERE"

    # WEAK: little impact vs free-flow
    if (
        (np.isfinite(upstream_ratio) and upstream_ratio >= 0.90)
        and max_queue <= 2
        and stopped_sightings <= 20
        and (np.isfinite(lane0_ratio) and lane0_ratio >= 0.85)
    ):
        return "WEAK"

    # Otherwise moderate disturbance with remaining open-lane capacity signal
    return "MODERATE"


def run_no_control_metrics(
    *,
    demand_vph: int,
    sim_seconds: float,
) -> dict[str, Any]:
    env_cfg = copy.deepcopy(load_config())
    step_length = float(env_cfg.get("sumo", {}).get("step_length", 0.1))
    steps = max(1, int(round(sim_seconds / step_length)))
    env_cfg.setdefault("episode", {})["max_episode_steps"] = steps + 50

    accident_pos = float(env_cfg.get("accident", {}).get("position_m", 1200.0))
    vmax = float(
        env_cfg.get("observation", {}).get(
            "max_speed_mps",
            env_cfg.get("highway", {}).get("speed_limit_mps", 30.0),
        )
    )
    slow_mps = float(env_cfg.get("test", {}).get("congestion_speed_mps", 15.0))
    n_lanes = int(env_cfg.get("highway", {}).get("lanes", 3))

    bin_aggs = {name: _empty_agg() for name, _, _ in POS_BINS}
    lane_aggs = {i: _empty_agg() for i in range(n_lanes)}
    global_agg = _empty_agg()
    max_queue_up = 0
    arrived = 0
    prev_ids: set[str] = set()
    prefix = str(env_cfg.get("accident", {}).get("vehicle_id_prefix", "ACCIDENT"))
    runnable = True
    error = ""

    env = HighwayAccidentEnv(config=env_cfg, gui=False)
    try:
        env.reset()
        for _ in range(steps):
            traci = env._require_traci()
            cur_ids = set(traci.vehicle.getIDList())
            for vid in prev_ids - cur_ids:
                if not str(vid).startswith(prefix):
                    arrived += 1
            prev_ids = cur_ids

            vehicles = env.get_vehicle_states()
            q = 0
            for vid, st in vehicles.items():
                if st.get("type") == "ACCIDENT":
                    continue
                pos = float(st["position"])
                spd = float(st["speed"])
                lane = int(st["lane"])
                vtype = str(st.get("type", ""))
                _accumulate(global_agg, spd, vtype, slow_mps=slow_mps)
                for name, lo, hi in POS_BINS:
                    if lo <= pos < hi:
                        _accumulate(bin_aggs[name], spd, vtype, slow_mps=slow_mps)
                        break
                if 0 <= lane < n_lanes:
                    _accumulate(lane_aggs[lane], spd, vtype, slow_mps=slow_mps)
                if lane == 0 and accident_pos - 250.0 <= pos <= accident_pos and spd < slow_mps:
                    q += 1
            max_queue_up = max(max_queue_up, q)

            env.step({cid: KEEP for cid in env.cav_ids})
    except Exception as exc:  # noqa: BLE001
        runnable = False
        error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass

    if not runnable:
        return {
            "demand": demand_vph,
            "runnable": False,
            "error": error,
            "scenario_type": "UNRUNNABLE",
        }

    def fin(agg: dict[str, Any]) -> dict[str, float]:
        out = _finalize(agg, vmax=vmax)
        out["mean_vehicles_per_step"] = float(agg["n"]) / float(steps)
        return out

    up = fin(bin_aggs["1050_1200"])
    zone = fin(bin_aggs["1200_1350"])
    g = fin(global_agg)
    l0 = fin(lane_aggs[0])
    l1 = fin(lane_aggs[1])
    l2 = fin(lane_aggs[2])

    scenario_type = classify_scenario(
        upstream_ratio=float(up["mean_speed_ratio"]),
        max_queue=int(max_queue_up),
        stopped_sightings=int(g["stopped_count"]),
        lane0_ratio=float(l0["mean_speed_ratio"]),
        lane2_ratio=float(l2["mean_speed_ratio"]),
    )

    return {
        "demand": demand_vph,
        "runnable": True,
        "error": "",
        "sim_seconds": sim_seconds,
        "steps": steps,
        "vmax": vmax,
        "slow_mps": slow_mps,
        "mean_speed": g["mean_speed"],
        "mean_speed_ratio": g["mean_speed_ratio"],
        "throughput": float(arrived),
        "vehicle_count_sightings": g["vehicle_count"],
        "mean_vehicles_per_step": g["mean_vehicles_per_step"],
        "lane0_mean_speed": l0["mean_speed"],
        "lane0_mean_speed_ratio": l0["mean_speed_ratio"],
        "lane0_vehicle_count": l0["vehicle_count"],
        "lane0_slow_count": l0["slow_count"],
        "lane0_stopped_count": l0["stopped_count"],
        "lane1_mean_speed": l1["mean_speed"],
        "lane1_mean_speed_ratio": l1["mean_speed_ratio"],
        "lane1_vehicle_count": l1["vehicle_count"],
        "lane1_slow_count": l1["slow_count"],
        "lane1_stopped_count": l1["stopped_count"],
        "lane2_mean_speed": l2["mean_speed"],
        "lane2_mean_speed_ratio": l2["mean_speed_ratio"],
        "lane2_vehicle_count": l2["vehicle_count"],
        "lane2_slow_count": l2["slow_count"],
        "lane2_stopped_count": l2["stopped_count"],
        "upstream_1050_1200_mean_speed": up["mean_speed"],
        "upstream_1050_1200_speed_ratio": up["mean_speed_ratio"],
        "upstream_queue_length_max": float(max_queue_up),
        "upstream_slow_count": up["slow_count"],
        "upstream_stopped_count": up["stopped_count"],
        "accident_zone_1200_1350_mean_speed": zone["mean_speed"],
        "accident_zone_1200_1350_speed_ratio": zone["mean_speed_ratio"],
        "scenario_type": scenario_type,
        "open_lane_speed_gap": float(l2["mean_speed"] - l0["mean_speed"])
        if np.isfinite(l0["mean_speed"]) and np.isfinite(l2["mean_speed"])
        else float("nan"),
    }


def diagnose_depart_speed(cfg: dict[str, Any]) -> dict[str, str]:
    traffic = cfg.get("traffic", {})
    value = str(traffic.get("depart_speed", ""))
    return {
        "config_value": value,
        "where_applied": (
            "Written to sumo/highway.rou.xml as flow attribute departSpeed "
            "for flow_HDV and flow_CAV (build_sumo_scenario.build_routes)."
        ),
        "meaning": (
            "SUMO departSpeed=max sets the vehicle's INITIAL speed at insertion "
            "to the maximum legal speed on the departure lane (≈ speed limit). "
            "It is NOT a continuous free-flow force after insertion."
        ),
        "after_depart": (
            "Once on the network, Krauss car-following / lane-change models "
            "control speed. Upstream of a blockage, vehicles SHOULD decelerate "
            "naturally if density and leader gaps require it."
        ),
        "implication_for_weak_congestion": (
            "depart_speed=max can make early segments look free-flow and delay "
            "queue onset, but it alone does not prevent accident-upstream slowing "
            "if demand is high enough. Weak queues at 1800 veh/h are more about "
            "single-lane blockage + early diversion than departSpeed alone."
        ),
    }


def set_demand_and_rebuild_routes(demand: int) -> tuple[int, int]:
    """Rewrite highway.rou.xml for demand; return (hdv_vph, cav_vph)."""
    cfg = load_build_config(PROJECT_ROOT / "configs" / "env.yaml")
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("traffic", {})["vehs_per_hour"] = int(demand)
    # Keep penetration / accident / network unchanged
    build_routes(cfg)
    total = float(demand)
    cav_p = float(cfg["traffic"]["cav_penetration"])
    cav_vph = int(round(total * cav_p))
    hdv_vph = int(round(total - cav_vph))
    if 0.0 < cav_p < 1.0:
        cav_vph = max(1, cav_vph)
        hdv_vph = max(1, int(round(total - cav_vph)))
    return hdv_vph, cav_vph


def restore_default_routes() -> None:
    cfg = load_build_config(PROJECT_ROOT / "configs" / "env.yaml")
    build_routes(cfg)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4.4.6 demand strength scan")
    parser.add_argument(
        "--demands",
        type=int,
        nargs="+",
        default=[1800, 2400, 3000, 3600],
    )
    parser.add_argument("--sim-seconds", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg0 = load_config()
    depart_info = diagnose_depart_speed(cfg0)

    print("==============================")
    print("Phase 4.4.6 Traffic Scenario Strength Scan")
    print("==============================")
    print("Policy: No Control (KEEP)")
    print(f"Demands: {args.demands}")
    print(f"sim_seconds={args.sim_seconds} seed={args.seed}")
    print(
        f"Fixed: accident@1200m lane0, cav_pen="
        f"{cfg0.get('traffic', {}).get('cav_penetration')}, "
        f"depart_speed={cfg0.get('traffic', {}).get('depart_speed')}"
    )
    print()

    rows: list[dict[str, Any]] = []
    try:
        for demand in args.demands:
            print(f"--- demand={demand} veh/h ---")
            try:
                hdv, cav = set_demand_and_rebuild_routes(demand)
                print(f"  routes rebuilt: HDV={hdv} CAV={cav} veh/h")
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL rebuild routes: {exc}")
                rows.append(
                    {
                        "demand": demand,
                        "runnable": False,
                        "error": str(exc),
                        "scenario_type": "UNRUNNABLE",
                    }
                )
                continue

            row = run_no_control_metrics(
                demand_vph=demand, sim_seconds=args.sim_seconds
            )
            rows.append(row)
            if not row.get("runnable", True):
                print(f"  UNRUNNABLE: {row.get('error')}")
                continue
            print(
                f"  type={row['scenario_type']} "
                f"up_v={row['upstream_1050_1200_mean_speed']:.2f} "
                f"up_ratio={row['upstream_1050_1200_speed_ratio']:.3f} "
                f"queue_max={row['upstream_queue_length_max']:.0f} "
                f"stopped_up={row['upstream_stopped_count']:.0f} "
                f"L0={row['lane0_mean_speed']:.2f} "
                f"L1={row['lane1_mean_speed']:.2f} "
                f"L2={row['lane2_mean_speed']:.2f} "
                f"thr={row['throughput']:.0f}"
            )
    finally:
        restore_default_routes()
        print("\nRestored highway.rou.xml to configs/env.yaml demand.")

    out_dir = PROJECT_ROOT / "results" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "phase_446_demand_scan.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "demands": args.demands,
                "sim_seconds": args.sim_seconds,
                "seed": args.seed,
                "depart_speed_diagnosis": depart_info,
                "rows": rows,
            },
            f,
            indent=2,
        )

    csv_path = out_dir / "phase_446_summary.csv"
    runnable_rows = [r for r in rows if r.get("runnable", True)]
    if runnable_rows:
        fields = [
            "demand",
            "upstream_1050_1200_mean_speed",
            "upstream_queue_length_max",
            "upstream_stopped_count",
            "lane0_mean_speed",
            "lane1_mean_speed",
            "lane2_mean_speed",
            "scenario_type",
            "mean_speed",
            "throughput",
            "accident_zone_1200_1350_mean_speed",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # ---- Report ----
    print()
    print("==============================")
    print("Phase 4.4.6 Report")
    print("==============================")
    print()
    print("1. Demand sweep configuration")
    print(f"   demands={args.demands}")
    print(f"   No Control, {args.sim_seconds}s, same accident@1200 lane0")
    print(f"   cav_penetration={cfg0.get('traffic', {}).get('cav_penetration')}")
    print(f"   routes temporarily rewritten; env.yaml left unchanged; rou restored")
    print()
    print("2. Scenario summary table")
    print(
        f"{'demand':>7} | {'up_spd':>7} | {'queue':>5} | {'stopped':>7} | "
        f"{'L0':>6} | {'L1':>6} | {'L2':>6} | type"
    )
    print("-" * 72)
    for r in rows:
        if not r.get("runnable", True):
            print(f"{r['demand']:>7} | UNRUNNABLE ({r.get('error','')[:40]})")
            continue
        print(
            f"{r['demand']:>7} | "
            f"{r['upstream_1050_1200_mean_speed']:7.2f} | "
            f"{r['upstream_queue_length_max']:5.0f} | "
            f"{r['upstream_stopped_count']:7.0f} | "
            f"{r['lane0_mean_speed']:6.2f} | "
            f"{r['lane1_mean_speed']:6.2f} | "
            f"{r['lane2_mean_speed']:6.2f} | "
            f"{r['scenario_type']}"
        )
    print()
    print("3. Queue formation analysis")
    for r in runnable_rows:
        print(
            f"   {r['demand']}: max_queue_upstream250={r['upstream_queue_length_max']:.0f}, "
            f"upstream_slow_sightings={r['upstream_slow_count']:.0f}, "
            f"upstream_stopped={r['upstream_stopped_count']:.0f}"
        )
    print()
    print("4. Accident upstream speed analysis (1050-1200m)")
    for r in runnable_rows:
        print(
            f"   {r['demand']}: mean_v={r['upstream_1050_1200_mean_speed']:.2f} "
            f"ratio={r['upstream_1050_1200_speed_ratio']:.3f}"
        )
    print()
    print("5. Lane-level comparison")
    for r in runnable_rows:
        print(
            f"   {r['demand']}: L0={r['lane0_mean_speed']:.2f} "
            f"(slow={r['lane0_slow_count']:.0f}, stop={r['lane0_stopped_count']:.0f}) | "
            f"L1={r['lane1_mean_speed']:.2f} | L2={r['lane2_mean_speed']:.2f} | "
            f"L2-L0 gap={r['open_lane_speed_gap']:.2f}"
        )
    print()
    print("6. Demand vs congestion relationship")
    for r in runnable_rows:
        print(f"   {r['demand']} -> {r['scenario_type']}")
    print()
    print("7. depart_speed=max diagnosis")
    for k, v in depart_info.items():
        print(f"   {k}: {v}")
    print()
    print("8. Recommended scenario candidates")
    moderate = [r for r in runnable_rows if r["scenario_type"] == "MODERATE"]
    weak = [r for r in runnable_rows if r["scenario_type"] == "WEAK"]
    severe = [r for r in runnable_rows if r["scenario_type"] == "SEVERE"]
    if moderate:
        for r in moderate:
            print(
                f"   CANDIDATE (MODERATE): {r['demand']} veh/h — "
                f"upstream_v={r['upstream_1050_1200_mean_speed']:.1f}, "
                f"queue_max={r['upstream_queue_length_max']:.0f}, "
                f"L2-L0={r['open_lane_speed_gap']:.1f} "
                f"(open-lane speed advantage remains → control/diversion space)"
            )
    else:
        print("   No MODERATE label in this sweep.")
        if weak and severe:
            print(
                "   Transition exists between WEAK and SEVERE; "
                "consider denser demand grid between them."
            )
        elif weak:
            print(
                "   All runnable cases WEAK — need higher demand or stronger blockage "
                "(not applied in this phase)."
            )
        elif severe:
            print(
                "   Cases tip to SEVERE — may be too harsh for cooperative LC learning."
            )
    print()
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print("==============================")
    print("Phase 4.4.6 complete. No MAPPO training. Core logic unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
