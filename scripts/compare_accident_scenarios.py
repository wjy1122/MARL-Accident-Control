"""Phase 4.4.8: No Accident vs Accident comparison (No Control).

Scenarios:
  A 3000 no accident | B 3000 accident
  C 3600 no accident | D 3600 accident

Does NOT modify Reward / Observation / MAPPO / env.yaml / net.xml.
Temporarily rewrites highway.rou.xml; restores 1800+accident at end.

Example:
  .\\.venv\\Scripts\\python.exe scripts\\compare_accident_scenarios.py
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from build_sumo_scenario import (  # noqa: E402
    SUMO_DIR,
    _vtype_attrs,
    build_routes,
    load_config as load_build_config,
    write_xml,
)
from diagnose_traffic_congestion import (  # noqa: E402
    _accumulate,
    _empty_agg,
    _finalize,
)

from env.action import KEEP  # noqa: E402
from env.highway_accident_env import HighwayAccidentEnv, load_config  # noqa: E402

# Offsets upstream of accident (m). Names are derived from actual accident_pos.
UPSTREAM_OFFSETS_M: tuple[float, ...] = (150.0, 300.0, 500.0)


def make_upstream_windows(accident_pos: float) -> list[tuple[str, float, float]]:
    """Build [acc-delta, acc] windows; names follow lo_hi (not hard-coded 1200)."""
    windows: list[tuple[str, float, float]] = []
    for delta in UPSTREAM_OFFSETS_M:
        lo = max(0.0, float(accident_pos) - float(delta))
        hi = float(accident_pos)
        name = f"{int(round(lo))}_{int(round(hi))}"
        windows.append((name, lo, hi))
    return windows


SCENARIOS: list[dict[str, Any]] = [
    {"id": "A", "demand": 3000, "accident": False},
    {"id": "B", "demand": 3000, "accident": True},
    {"id": "C", "demand": 3600, "accident": False},
    {"id": "D", "demand": 3600, "accident": True},
]


def write_routes_for_scenario(*, demand: int, with_accident: bool) -> Path:
    """Minimal rou.xml rewrite; does not touch env.yaml / net.xml."""
    cfg = copy.deepcopy(load_build_config(PROJECT_ROOT / "configs" / "env.yaml"))
    hw = cfg["highway"]
    traffic = cfg["traffic"]
    accident = cfg["accident"]
    vtypes = cfg["vehicle_types"]

    total = float(demand)
    cav_p = float(traffic["cav_penetration"])
    cav_vph = int(round(total * cav_p))
    hdv_vph = int(round(total - cav_vph))
    if 0.0 < cav_p < 1.0:
        cav_vph = max(1, cav_vph)
        hdv_vph = max(1, int(round(total - cav_vph)))

    edge_id = str(hw["edge_id"])
    root = ET.Element("routes")
    ET.SubElement(root, "vType", **_vtype_attrs("HDV", vtypes["HDV"]))
    ET.SubElement(root, "vType", **_vtype_attrs("CAV", vtypes["CAV"]))
    ET.SubElement(root, "vType", **_vtype_attrs("ACCIDENT", vtypes["ACCIDENT"]))
    ET.SubElement(root, "route", id="main", edges=edge_id)

    flow_common = {
        "route": "main",
        "begin": str(traffic["flow_begin_s"]),
        "end": str(traffic["flow_end_s"]),
        "departLane": str(traffic["depart_lane"]),
        "departSpeed": str(traffic["depart_speed"]),
    }
    ET.SubElement(
        root, "flow", id="flow_HDV", type="HDV", vehsPerHour=str(hdv_vph), **flow_common
    )
    ET.SubElement(
        root, "flow", id="flow_CAV", type="CAV", vehsPerHour=str(cav_vph), **flow_common
    )

    if with_accident:
        n_lanes = int(hw["lanes"])
        blocked = [int(x) for x in accident["blocked_lane_indices"]]
        pos = float(accident["position_m"])
        half = float(accident["stop_half_width_m"])
        stop_start = max(0.0, pos - half)
        stop_end = pos + half
        prefix = str(accident.get("vehicle_id_prefix", "ACCIDENT"))
        for lane_idx in blocked:
            if lane_idx < 0 or lane_idx >= n_lanes:
                raise ValueError(f"blocked lane {lane_idx} out of range")
            veh = ET.SubElement(
                root,
                "vehicle",
                id=f"{prefix}_{lane_idx}",
                type="ACCIDENT",
                route="main",
                depart="0",
                departLane=str(lane_idx),
                departPos=f"{pos:.2f}",
                departSpeed="0",
            )
            ET.SubElement(
                veh,
                "stop",
                lane=f"{edge_id}_{lane_idx}",
                startPos=f"{stop_start:.2f}",
                endPos=f"{stop_end:.2f}",
                duration=str(accident["duration_s"]),
            )

    out = SUMO_DIR / "highway.rou.xml"
    write_xml(out, root)
    return out


def restore_default_routes() -> None:
    build_routes(load_build_config(PROJECT_ROOT / "configs" / "env.yaml"))


def run_scenario(*, scenario_id: str, demand: int, with_accident: bool, sim_seconds: float) -> dict[str, Any]:
    write_routes_for_scenario(demand=demand, with_accident=with_accident)

    env_cfg = copy.deepcopy(load_config())
    step_length = float(env_cfg.get("sumo", {}).get("step_length", 0.1))
    steps = max(1, int(round(sim_seconds / step_length)))
    env_cfg.setdefault("episode", {})["max_episode_steps"] = steps + 50

    accident_pos = float(env_cfg.get("accident", {}).get("position_m", 1200.0))
    near_m = float(env_cfg.get("observation", {}).get("range_m", 150.0))
    upstream_windows = make_upstream_windows(accident_pos)
    vmax = float(
        env_cfg.get("observation", {}).get(
            "max_speed_mps",
            env_cfg.get("highway", {}).get("speed_limit_mps", 30.0),
        )
    )
    slow_mps = float(env_cfg.get("test", {}).get("congestion_speed_mps", 15.0))
    n_lanes = int(env_cfg.get("highway", {}).get("lanes", 3))
    prefix = str(env_cfg.get("accident", {}).get("vehicle_id_prefix", "ACCIDENT"))

    global_agg = _empty_agg()
    lane_aggs = {i: _empty_agg() for i in range(n_lanes)}
    up_aggs = {name: _empty_agg() for name, _, _ in upstream_windows}
    zone_agg = _empty_agg()

    arrived = 0
    departed = 0
    seen: set[str] = set()
    prev_ids: set[str] = set()
    accident_seen = False

    env = HighwayAccidentEnv(config=env_cfg, gui=False)
    try:
        env.reset()
        for _ in range(steps):
            traci = env._require_traci()
            cur_ids = set(traci.vehicle.getIDList())
            for vid in cur_ids - prev_ids:
                if not str(vid).startswith(prefix):
                    departed += 1
                    seen.add(vid)
            for vid in prev_ids - cur_ids:
                if not str(vid).startswith(prefix):
                    arrived += 1
            prev_ids = cur_ids

            vehicles = env.get_vehicle_states()
            for vid, st in vehicles.items():
                if st.get("type") == "ACCIDENT":
                    accident_seen = True
                    continue
                pos = float(st["position"])
                spd = float(st["speed"])
                lane = int(st["lane"])
                vtype = str(st.get("type", ""))
                _accumulate(global_agg, spd, vtype, slow_mps=slow_mps)
                if 0 <= lane < n_lanes:
                    _accumulate(lane_aggs[lane], spd, vtype, slow_mps=slow_mps)
                if abs(pos - accident_pos) <= near_m:
                    _accumulate(zone_agg, spd, vtype, slow_mps=slow_mps)
                for name, lo, hi in upstream_windows:
                    if lo <= pos <= hi:
                        _accumulate(up_aggs[name], spd, vtype, slow_mps=slow_mps)

            env.step({cid: KEEP for cid in env.cav_ids})
    finally:
        env.close()

    def fin(agg: dict[str, Any]) -> dict[str, float]:
        return _finalize(agg, vmax=vmax)

    g = fin(global_agg)
    z = fin(zone_agg)
    row: dict[str, Any] = {
        "scenario_id": scenario_id,
        "demand": demand,
        "accident": bool(with_accident),
        "accident_position_m": accident_pos,
        "accident_vehicle_seen": bool(accident_seen),
        "sim_seconds": sim_seconds,
        "steps": steps,
        "mean_speed_mps": g["mean_speed"],
        "mean_speed_ratio": g["mean_speed_ratio"],
        "accident_zone_mean_speed": z["mean_speed"],
        "throughput": float(arrived),
        "arrived_count": float(arrived),
        "departed_count": float(departed),
        "slow_vehicle_sightings": g["slow_count"],
        "stopped_vehicle_sightings": g["stopped_count"],
        "vehicle_sightings": g["vehicle_count"],
    }
    for i in range(n_lanes):
        li = fin(lane_aggs[i])
        row[f"lane{i}_mean_speed"] = li["mean_speed"]
        row[f"lane{i}_slow_sightings"] = li["slow_count"]
        row[f"lane{i}_stopped_sightings"] = li["stopped_count"]
    for name, _, _ in upstream_windows:
        u = fin(up_aggs[name])
        row[f"upstream_{name}_mean_speed"] = u["mean_speed"]
        row[f"upstream_{name}_slow_sightings"] = u["slow_count"]
        row[f"upstream_{name}_stopped_sightings"] = u["stopped_count"]
    # Convenience alias: nearest upstream window (acc-150 .. acc)
    if upstream_windows:
        nearest = upstream_windows[0][0]
        row["upstream_mean_speed"] = row[f"upstream_{nearest}_mean_speed"]
    return row


def impact_delta(no_acc: dict[str, Any], acc: dict[str, Any]) -> dict[str, float]:
    """NoAccident - Accident for key metrics (positive => accident worsened metric)."""
    keys = [
        "mean_speed_mps",
        "accident_zone_mean_speed",
        "upstream_mean_speed",
        "throughput",
        "arrived_count",
        "slow_vehicle_sightings",
        "stopped_vehicle_sightings",
        "lane0_mean_speed",
        "lane1_mean_speed",
        "lane2_mean_speed",
        "lane0_slow_sightings",
        "lane1_slow_sightings",
        "lane2_slow_sightings",
    ]
    for k in no_acc:
        if k.startswith("upstream_") and (
            k.endswith("_mean_speed") or k.endswith("_slow_sightings")
        ):
            if k not in keys:
                keys.append(k)
    out: dict[str, float] = {"demand": float(no_acc["demand"])}
    for k in keys:
        a = float(no_acc.get(k, float("nan")))
        b = float(acc.get(k, float("nan")))
        # For speed/throughput: drop = no_acc - acc (positive = worse with accident)
        # For slow/stopped: increase = acc - no_acc (positive = more with accident)
        if "slow" in k or "stopped" in k:
            out[f"delta_{k}"] = b - a
        else:
            out[f"delta_{k}"] = a - b
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4.4.8 accident comparison")
    parser.add_argument("--sim-seconds", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("========================================")
    print("Phase 4.4.8 Accident Scenario Comparison")
    print("========================================")
    print("No Control | seed=42 | 180s | cav_pen=0.2")
    print("Scenarios: A/B 3000, C/D 3600 × no/yes accident")
    print()

    rows: list[dict[str, Any]] = []
    try:
        for sc in SCENARIOS:
            print(
                f"--- {sc['id']}: demand={sc['demand']} "
                f"accident={'YES' if sc['accident'] else 'NO'} ---"
            )
            row = run_scenario(
                scenario_id=sc["id"],
                demand=int(sc["demand"]),
                with_accident=bool(sc["accident"]),
                sim_seconds=args.sim_seconds,
            )
            rows.append(row)
            print(
                f"  mean_v={row['mean_speed_mps']:.2f} thr={row['throughput']:.0f} "
                f"dep={row['departed_count']:.0f} arr={row['arrived_count']:.0f} "
                f"slow={row['slow_vehicle_sightings']:.0f} "
                f"L0/L1/L2="
                f"{row['lane0_mean_speed']:.2f}/"
                f"{row['lane1_mean_speed']:.2f}/"
                f"{row['lane2_mean_speed']:.2f} "
                f"acc_seen={row['accident_vehicle_seen']}"
            )
    finally:
        restore_default_routes()
        print("\nRestored highway.rou.xml to env.yaml demand + accident.")

    by_id = {r["scenario_id"]: r for r in rows}
    impacts = {
        "3000": impact_delta(by_id["A"], by_id["B"]),
        "3600": impact_delta(by_id["C"], by_id["D"]),
    }

    out_dir = PROJECT_ROOT / "results" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "phase_448_accident_comparison.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "sim_seconds": args.sim_seconds,
                "scenarios": rows,
                "impacts": impacts,
            },
            f,
            indent=2,
        )

    csv_path = out_dir / "phase_448_accident_comparison.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # ---- Report ----
    a, b, c, d = by_id["A"], by_id["B"], by_id["C"], by_id["D"]
    i3, i6 = impacts["3000"], impacts["3600"]
    up_keys = sorted(
        k for k in c.keys() if k.startswith("upstream_") and k.endswith("_mean_speed")
        and k != "upstream_mean_speed"
    )

    print()
    print("========================================")
    print("Phase 4.4.8 Accident Scenario Comparison")
    print("========================================")
    print()
    print("1. Experiment configuration")
    print("   A: 3000 no accident | B: 3000 accident")
    print("   C: 3600 no accident | D: 3600 accident")
    print(f"   No Control KEEP, seed={args.seed}, sim={args.sim_seconds}s")
    print("   Same net / vTypes / CF / cav_pen=0.2; only demand + accident toggled")
    print()
    print("2. Global comparison")
    print(f"{'ID':>2} {'dem':>5} {'acc':>3} {'mean_v':>7} {'thr':>5} {'slow':>6} {'stop':>5}")
    for r in rows:
        print(
            f"{r['scenario_id']:>2} {r['demand']:>5} "
            f"{'Y' if r['accident'] else 'N':>3} "
            f"{r['mean_speed_mps']:7.2f} {r['throughput']:5.0f} "
            f"{r['slow_vehicle_sightings']:6.0f} {r['stopped_vehicle_sightings']:5.0f}"
        )
    print()
    print("3. Upstream comparison (mean speed)")
    header = f"{'ID':>2}" + "".join(f" {k.replace('upstream_','').replace('_mean_speed',''):>12}" for k in up_keys)
    print(header)
    for r in rows:
        vals = "".join(f" {r[k]:12.2f}" for k in up_keys)
        print(f"{r['scenario_id']:>2}{vals}")
    print()
    print("4. Lane comparison (mean speed)")
    print(f"{'ID':>2} {'L0':>7} {'L1':>7} {'L2':>7}")
    for r in rows:
        print(
            f"{r['scenario_id']:>2} "
            f"{r['lane0_mean_speed']:7.2f} "
            f"{r['lane1_mean_speed']:7.2f} "
            f"{r['lane2_mean_speed']:7.2f}"
        )
    print()
    print("5. Accident impact (NoAccident - Accident for speeds/throughput;")
    print("   Accident - NoAccident for slow/stopped)")
    nearest_up = up_keys[0] if up_keys else "upstream_mean_speed"
    for dem, imp in (("3000", i3), ("3600", i6)):
        print(f"   {dem} veh/h:")
        print(f"     global speed drop:    {imp['delta_mean_speed_mps']:+.3f} m/s")
        print(f"     throughput drop:      {imp['delta_throughput']:+.1f}")
        for uk in up_keys:
            print(f"     {uk}: {imp.get('delta_' + uk, float('nan')):+.3f} m/s")
        print(f"     lane0 speed drop:     {imp['delta_lane0_mean_speed']:+.3f} m/s")
        print(f"     lane1 speed drop:     {imp['delta_lane1_mean_speed']:+.3f} m/s")
        print(f"     lane2 speed drop:     {imp['delta_lane2_mean_speed']:+.3f} m/s")
        print(f"     slow sightings + :    {imp['delta_slow_vehicle_sightings']:+.0f}")
        print(f"     stopped sightings +:  {imp['delta_stopped_vehicle_sightings']:+.0f}")
        print(f"     lane0 slow +:         {imp['delta_lane0_slow_sightings']:+.0f}")
    print()
    print("6. Scenario validity")

    # Answers A-E from data
    clear_3000 = abs(i3["delta_mean_speed_mps"]) >= 0.3 or abs(i3["delta_lane0_mean_speed"]) >= 0.5
    clear_3600 = abs(i6["delta_mean_speed_mps"]) >= 0.3 or abs(i6["delta_lane0_mean_speed"]) >= 0.5
    clear_any = clear_3000 or clear_3600

    print(
        f"   A. 无事故 vs 有事故是否存在明确交通性能差异？ "
        f"{'YES' if clear_any else 'WEAK / unclear'} "
        f"(3000 clear={clear_3000}, 3600 clear={clear_3600})"
    )
    # Which lane most affected: largest speed drop
    lane_drops_3 = {
        0: i3["delta_lane0_mean_speed"],
        1: i3["delta_lane1_mean_speed"],
        2: i3["delta_lane2_mean_speed"],
    }
    lane_drops_6 = {
        0: i6["delta_lane0_mean_speed"],
        1: i6["delta_lane1_mean_speed"],
        2: i6["delta_lane2_mean_speed"],
    }
    main_lane_3 = max(lane_drops_3, key=lane_drops_3.get)
    main_lane_6 = max(lane_drops_6, key=lane_drops_6.get)
    print(
        f"   B. 事故影响主要集中在哪个车道？ "
        f"lane{main_lane_3} @3000 (drop={lane_drops_3[main_lane_3]:+.3f}); "
        f"lane{main_lane_6} @3600 (drop={lane_drops_6[main_lane_6]:+.3f})"
    )
    spill_3 = lane_drops_3[1] > 0.15 or lane_drops_3[2] > 0.15
    spill_6 = lane_drops_6[1] > 0.15 or lane_drops_6[2] > 0.15
    print(
        f"   C. 是否传播到其他车道？ "
        f"3000={'YES' if spill_3 else 'limited'}; "
        f"3600={'YES' if spill_6 else 'limited'} "
        f"(L1/L2 drops: 3000={lane_drops_3[1]:+.3f}/{lane_drops_3[2]:+.3f}, "
        f"3600={lane_drops_6[1]:+.3f}/{lane_drops_6[2]:+.3f})"
    )
    up_drop_key = f"delta_{nearest_up}"
    enough_3000 = (
        i3["delta_lane0_mean_speed"] >= 0.8
        or float(i3.get(up_drop_key, 0.0)) >= 0.8
        or i3["delta_slow_vehicle_sightings"] >= 50
    )
    print(
        f"   D. 3000 veh/h 是否已经产生足够事故扰动？ "
        f"{'BORDERLINE-YES' if enough_3000 else 'MARGINAL'} "
        f"(L0 drop={i3['delta_lane0_mean_speed']:+.3f}, "
        f"up drop={float(i3.get(up_drop_key, float('nan'))):+.3f}, "
        f"slow+={i3['delta_slow_vehicle_sightings']:+.0f})"
    )
    stronger_3600 = (
        i6["delta_lane0_mean_speed"] > i3["delta_lane0_mean_speed"]
        or float(i6.get(up_drop_key, 0.0)) > float(i3.get(up_drop_key, 0.0))
        or i6["delta_slow_vehicle_sightings"] > i3["delta_slow_vehicle_sightings"]
    )
    print(
        f"   E. 3600 是否比 3000 更明显？ "
        f"{'YES' if stronger_3600 else 'NO / similar'} "
        f"(L0 drop 3600={i6['delta_lane0_mean_speed']:+.3f} vs "
        f"3000={i3['delta_lane0_mean_speed']:+.3f}; "
        f"slow+ 3600={i6['delta_slow_vehicle_sightings']:+.0f} vs "
        f"3000={i3['delta_slow_vehicle_sightings']:+.0f})"
    )
    print()
    print("7. Recommendation")
    if stronger_3600 and (enough_3000 or clear_3600):
        print("   Next: adopt 3600 veh/h + accident as primary training scene candidate;")
        print("   keep 3000 as secondary; then freeze scene and proceed to MAPPO training plan.")
    elif enough_3000:
        print("   Next: 3000+accident is usable; compare with 3600 tradeoff then freeze scene.")
    else:
        print("   Next: accident impact still limited — consider stronger scene levers")
        print("   (not applied yet): higher demand grid, or multi-lane blockage experiments.")
    print()
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print("Completed. Stopped — no MAPPO training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
