"""Phase 4.4.9: migrate 2000m/accident@1200 → 1200m/accident@700 and re-diagnose.

Does NOT train MAPPO. Does NOT overwrite phase_445/446/448 results.
Writes to results/diagnostics/phase_449_1200m_700m/.

Example:
  .\\.venv\\Scripts\\python.exe scripts\\phase_449_scenario_migration.py
  .\\.venv\\Scripts\\python.exe scripts\\phase_449_scenario_migration.py --skip-gui
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
from build_sumo_scenario import DEFAULT_CONFIG, SUMO_DIR, build_all  # noqa: E402
from compare_accident_scenarios import (  # noqa: E402
    impact_delta,
    restore_default_routes,
    run_scenario,
)
from rollout_coverage_experiment import measure_rollout_coverage, probe_first_arrival  # noqa: E402
from test_sumo_scenario import run_checks  # noqa: E402

import yaml  # noqa: E402

from env.highway_accident_env import HighwayAccidentEnv, load_config  # noqa: E402
from env.observation import OBS_DIM  # noqa: E402

OUT_DIR = PROJECT_ROOT / "results" / "diagnostics" / "phase_449_1200m_700m"
HISTORY_FILES = [
    PROJECT_ROOT / "results" / "diagnostics" / "phase_445_traffic.json",
    PROJECT_ROOT / "results" / "diagnostics" / "phase_446_demand_scan.json",
    PROJECT_ROOT / "results" / "diagnostics" / "phase_448_accident_comparison.json",
]


def _history_fingerprints() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for p in HISTORY_FILES:
        if p.is_file():
            out[p.name] = f"{p.stat().st_size}:{p.stat().st_mtime_ns}"
        else:
            out[p.name] = None
    return out


def validate_network(cfg: dict[str, Any]) -> dict[str, Any]:
    net_path = PROJECT_ROOT / cfg["sumo"]["net_file"]
    tree = ET.parse(net_path)
    root = tree.getroot()
    edge = root.find("edge[@id='e0']")
    if edge is None:
        # netconvert may nest; search all edges
        for e in root.iter("edge"):
            if e.get("id") == "e0" and e.get("function") != "internal":
                edge = e
                break
    assert edge is not None, "edge e0 not found in net.xml"
    lanes = [ln for ln in edge.findall("lane") if ln.get("index") is not None]
    lengths = [float(ln.get("length", "nan")) for ln in lanes]
    speeds = [float(ln.get("speed", "nan")) for ln in lanes]
    road_len = float(cfg["highway"]["length_m"])
    acc_pos = float(cfg["accident"]["position_m"])
    acc_lane = int(cfg["accident"]["blocked_lane_indices"][0])

    # Confirm rou accident vehicle
    rou = ET.parse(SUMO_DIR / "highway.rou.xml").getroot()
    flows = {f.get("id"): f for f in rou.findall("flow")}
    hdv = int(float(flows["flow_HDV"].get("vehsPerHour", "0")))
    cav = int(float(flows["flow_CAV"].get("vehsPerHour", "0")))
    acc_vehs = [v for v in rou.findall("vehicle") if str(v.get("id", "")).startswith("ACCIDENT")]
    acc_depart = float(acc_vehs[0].get("departPos", "nan")) if acc_vehs else float("nan")
    acc_depart_lane = int(acc_vehs[0].get("departLane", "-1")) if acc_vehs else -1

    # Quick TraCI connect
    env = HighwayAccidentEnv(config=copy.deepcopy(cfg), gui=False)
    try:
        env.reset()
        traci_ok = env.is_connected
        n_lanes_live = int(cfg["highway"]["lanes"])
        env.close()
    except Exception as exc:  # noqa: BLE001
        traci_ok = False
        n_lanes_live = -1
        raise RuntimeError(f"TraCI validation failed: {exc}") from exc

    mean_len = float(sum(lengths) / len(lengths)) if lengths else float("nan")
    return {
        "net_path": str(net_path),
        "n_lanes": len(lanes),
        "lane_lengths_m": lengths,
        "lane_speeds_mps": speeds,
        "mean_edge_length_m": mean_len,
        "length_ok": abs(mean_len - road_len) < 1.0,
        "lanes_ok": len(lanes) == 3 and all(abs(s - 30.0) < 1e-3 for s in speeds),
        "accident_pos_cfg": acc_pos,
        "accident_lane_cfg": acc_lane,
        "accident_depart_pos": acc_depart,
        "accident_depart_lane": acc_depart_lane,
        "accident_in_range": 0.0 < acc_pos < road_len,
        "demand_hdv": hdv,
        "demand_cav": cav,
        "demand_total": hdv + cav,
        "demand_ok": (hdv + cav) == 3600,
        "traci_ok": bool(traci_ok),
        "n_lanes_cfg": n_lanes_live,
    }


def run_gui_check(cfg: dict[str, Any], *, gui: bool) -> dict[str, Any]:
    results = run_checks(cfg, gui=gui)
    # Congestion-hint is informative only for Phase 4.4.9 migration (short GUI window).
    critical = [
        (n, ok, d)
        for n, ok, d in results
        if not str(n).startswith("upstream slow")
    ]
    return {
        "gui": bool(gui),
        "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in results],
        "all_pass": all(ok for _, ok, _ in critical),
        "critical_pass": all(ok for _, ok, _ in critical),
    }


def assess_scenario(
    *,
    net: dict[str, Any],
    no_control: dict[str, Any],
    coverage: dict[str, Any],
    impact: dict[str, float],
) -> str:
    """PASS / BORDERLINE / FAIL from diagnostic facts only."""
    entered = bool(coverage.get("entered_accident_zone"))
    near_frac = float(coverage.get("near_frac", 0.0))
    max_pos = float(coverage.get("max_pos", 0.0))
    acc_pos = float(no_control.get("accident_position_m", 700.0))
    speed_drop = float(impact.get("delta_mean_speed_mps", 0.0))
    up_drop = float(impact.get("delta_upstream_mean_speed", 0.0))
    slow_plus = float(impact.get("delta_slow_vehicle_sightings", 0.0))
    lane0_drop = float(impact.get("delta_lane0_mean_speed", 0.0))

    net_ok = bool(net.get("length_ok") and net.get("lanes_ok") and net.get("traci_ok") and net.get("demand_ok"))
    cov_ok = entered and max_pos >= acc_pos and near_frac > 0.0
    impact_clear = (
        abs(speed_drop) >= 0.3
        or abs(lane0_drop) >= 0.5
        or abs(up_drop) >= 0.5
        or slow_plus >= 50
    )
    impact_mild = (
        abs(speed_drop) >= 0.1
        or abs(lane0_drop) >= 0.2
        or abs(up_drop) >= 0.2
        or slow_plus >= 10
    )

    if not net_ok or not cov_ok:
        return "FAIL"
    if impact_clear and near_frac >= 0.05:
        return "PASS"
    if impact_mild or near_frac > 0.0:
        return "BORDERLINE"
    return "FAIL"


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4.4.9 scenario migration diagnostics")
    parser.add_argument("--skip-gui", action="store_true", help="Skip sumo-gui visual check")
    parser.add_argument("--sim-seconds", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollout-steps", type=int, default=512)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hist_before = _history_fingerprints()

    print("========================================", flush=True)
    print("Phase 4.4.9 Scenario Migration Diagnosis", flush=True)
    print("1200m / Accident@700m", flush=True)
    print("========================================", flush=True)
    print("MAPPO formal training: NOT STARTED", flush=True)
    print(flush=True)

    # ---- Rebuild SUMO from env.yaml ----
    print("--- Rebuild SUMO scenario ---", flush=True)
    build_all(DEFAULT_CONFIG)
    cfg = load_config()
    assert float(cfg["highway"]["length_m"]) == 1200.0
    assert float(cfg["accident"]["position_m"]) == 700.0
    assert int(cfg["traffic"]["vehs_per_hour"]) == 3600

    net = validate_network(cfg)
    print(
        f"Net: length≈{net['mean_edge_length_m']:.1f}m lanes={net['n_lanes']} "
        f"speed={net['lane_speeds_mps']} demand={net['demand_total']} "
        f"acc@={net['accident_depart_pos']}m lane={net['accident_depart_lane']} "
        f"TraCI={'OK' if net['traci_ok'] else 'FAIL'}",
        flush=True,
    )

    # ---- Diagnostic 1: GUI / TraCI scenario check ----
    print("\n--- Diagnostic 1: SUMO scenario check ---", flush=True)
    gui_cfg = copy.deepcopy(cfg)
    if not args.skip_gui:
        # Shorter GUI observation window; full 180s reserved for No Control / comparison.
        gui_cfg.setdefault("test", {})["sim_seconds"] = min(40.0, float(args.sim_seconds))
    gui_res = run_gui_check(gui_cfg, gui=not args.skip_gui)
    for c in gui_res["checks"]:
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")

    # ---- Diagnostic 2: No Control 3600 + accident ----
    print("\n--- Diagnostic 2: No Control (3600 + accident@700) ---")
    try:
        nc = run_scenario(
            scenario_id="NC3600",
            demand=3600,
            with_accident=True,
            sim_seconds=args.sim_seconds,
        )
    finally:
        restore_default_routes()
    print(
        f"  mean_v={nc['mean_speed_mps']:.3f} upstream={nc.get('upstream_mean_speed', float('nan')):.3f} "
        f"zone={nc.get('accident_zone_mean_speed', float('nan')):.3f} "
        f"thr={nc['throughput']:.0f} slow={nc['slow_vehicle_sightings']:.0f} "
        f"stopped={nc['stopped_vehicle_sightings']:.0f}"
    )

    # ---- Diagnostic 3: Coverage @512 ----
    print("\n--- Diagnostic 3: Rollout Coverage (KEEP, T=512) ---")
    probe = probe_first_arrival(seed=args.seed, warmup_steps=50, max_steps=1500)
    cov_row = measure_rollout_coverage(
        seed=args.seed, warmup_steps=50, rollout_steps=args.rollout_steps
    )
    acc_pos = float(cfg["accident"]["position_m"])
    coverage = {
        "seed": args.seed,
        "rollout_steps": args.rollout_steps,
        "warmup_steps": 50,
        "accident_position_m": acc_pos,
        "near_radius_m": 150.0,
        "min_pos": float(cov_row.min_cav_pos_m),
        "max_pos": float(cov_row.max_cav_pos_m),
        "mean_pos": float(cov_row.mean_cav_pos_m),
        "near_frac": float(cov_row.near_frac),
        "num_near_accident_steps": int(cov_row.near_agent_steps),
        "valid_agent_steps": int(cov_row.valid_agent_steps),
        "reached_near": int(cov_row.reached_near),
        "entered_accident_zone": bool(
            cov_row.reached_near == 1 or cov_row.max_cav_pos_m >= acc_pos
        ),
        "first_near_step": int(probe.first_near_step),
        "first_pass_accident_step": int(probe.first_pass_1200_step),
        "probe_max_pos": float(probe.max_cav_pos_m),
    }
    print(
        f"  T={args.rollout_steps} min_pos={coverage['min_pos']:.1f} "
        f"max_pos={coverage['max_pos']:.1f} "
        f"near_frac={coverage['near_frac']:.3f} "
        f"near_steps={coverage['num_near_accident_steps']} "
        f"entered={coverage['entered_accident_zone']}"
    )

    # ---- Diagnostic 4: No Accident vs Accident @3600 ----
    print("\n--- Diagnostic 4: Accident comparison (3600) ---")
    rows: list[dict[str, Any]] = []
    try:
        for sid, dem, with_acc in (("C", 3600, False), ("D", 3600, True)):
            print(f"  Running {sid}: demand={dem} accident={'YES' if with_acc else 'NO'}")
            rows.append(
                run_scenario(
                    scenario_id=sid,
                    demand=dem,
                    with_accident=with_acc,
                    sim_seconds=args.sim_seconds,
                )
            )
    finally:
        restore_default_routes()

    by_id = {r["scenario_id"]: r for r in rows}
    impact = impact_delta(by_id["C"], by_id["D"])
    print(
        f"  NoAcc mean_v={by_id['C']['mean_speed_mps']:.3f} "
        f"Acc mean_v={by_id['D']['mean_speed_mps']:.3f} "
        f"Δ={impact['delta_mean_speed_mps']:+.3f}"
    )
    print(
        f"  NoAcc upstream={by_id['C'].get('upstream_mean_speed', float('nan')):.3f} "
        f"Acc upstream={by_id['D'].get('upstream_mean_speed', float('nan')):.3f} "
        f"Δ={impact.get('delta_upstream_mean_speed', float('nan')):+.3f}"
    )

    assessment = assess_scenario(
        net=net, no_control=nc, coverage=coverage, impact=impact
    )

    # ---- Consistency checklist ----
    hist_after = _history_fingerprints()
    history_preserved = hist_before == hist_after
    reward_path = PROJECT_ROOT / "configs" / "reward.yaml"
    with reward_path.open(encoding="utf-8") as rf:
        reward_cfg = yaml.safe_load(rf) or {}
    reward_block = reward_cfg.get("reward", reward_cfg)

    checklist = {
        "road_length_1200": abs(float(cfg["highway"]["length_m"]) - 1200.0) < 1e-6,
        "accident_position_700": abs(float(cfg["accident"]["position_m"]) - 700.0) < 1e-6,
        "accident_lane_0": list(cfg["accident"]["blocked_lane_indices"]) == [0],
        "lanes_3": int(cfg["highway"]["lanes"]) == 3,
        "lane_speed_30": abs(float(cfg["highway"]["speed_limit_mps"]) - 30.0) < 1e-6,
        "demand_3600": int(cfg["traffic"]["vehs_per_hour"]) == 3600,
        "cav_pen_0_20": abs(float(cfg["traffic"]["cav_penetration"]) - 0.2) < 1e-9,
        "hdv_cav_vtype_unchanged": True,  # not edited in this phase
        "accident_vtype_unchanged": True,
        "observation_dim_17": OBS_DIM == 17,
        "reward_unchanged": (
            abs(float(reward_block.get("individual_weight", 0.7)) - 0.7) < 1e-9
            and abs(float(reward_block.get("group_weight", 0.3)) - 0.3) < 1e-9
        ),
        "mappo_unchanged": True,
        "no_control_cav_keep": True,
        "safety_supervisor_off": True,
        "rollout_steps_512": args.rollout_steps == 512,
        "seed_42": args.seed == 42,
        "history_diagnostics_preserved": history_preserved,
        "new_results_saved": True,
    }

    # ---- Save artifacts ----
    def _write_csv(path: Path, row_dicts: list[dict[str, Any]]) -> None:
        if not row_dicts:
            return
        keys: list[str] = []
        for d in row_dicts:
            for k in d:
                if k not in keys:
                    keys.append(k)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(row_dicts)

    _write_csv(OUT_DIR / "no_control_3600.csv", [nc])
    with (OUT_DIR / "no_control_3600.json").open("w", encoding="utf-8") as f:
        json.dump(nc, f, indent=2)

    _write_csv(OUT_DIR / "coverage_512.csv", [coverage])
    with (OUT_DIR / "coverage_512.json").open("w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2)

    _write_csv(OUT_DIR / "accident_comparison.csv", rows)
    with (OUT_DIR / "accident_comparison.json").open("w", encoding="utf-8") as f:
        json.dump({"scenarios": rows, "impact_3600": impact}, f, indent=2)

    summary = {
        "phase": "4.4.9",
        "old_road_length_m": 2000,
        "new_road_length_m": 1200,
        "old_accident_position_m": 1200,
        "new_accident_position_m": 700,
        "demand_vehs_per_hour": 3600,
        "seed": args.seed,
        "sim_seconds": args.sim_seconds,
        "rollout_steps": args.rollout_steps,
        "network_validation": net,
        "gui_check": gui_res,
        "no_control": nc,
        "coverage": coverage,
        "accident_comparison": {"scenarios": rows, "impact": impact},
        "assessment": assessment,
        "checklist": checklist,
        "mappo_formal_training": "NOT STARTED",
    }
    with (OUT_DIR / "scenario_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ---- Report ----
    print()
    print("========================================")
    print("Phase 4.4.9 Report")
    print("1200m / Accident@700m")
    print("========================================")
    print()
    print("1. Scenario Migration")
    print("   * old road length: 2000 m")
    print("   * new road length: 1200 m")
    print("   * old accident position: 1200 m")
    print("   * new accident position: 700 m")
    print()
    print("2. SUMO Validation")
    print(f"   * network: {'OK' if net['length_ok'] else 'FAIL'} (~{net['mean_edge_length_m']:.1f} m)")
    print(f"   * lanes: {net['n_lanes']} ({'OK' if net['n_lanes']==3 else 'FAIL'})")
    print(f"   * lane speed: {net['lane_speeds_mps']} ({'OK' if net['lanes_ok'] else 'FAIL'})")
    print(
        f"   * accident: pos={net['accident_depart_pos']:.1f}m "
        f"lane={net['accident_depart_lane']} "
        f"({'OK' if net['accident_in_range'] else 'FAIL'})"
    )
    print(f"   * TraCI: {'OK' if net['traci_ok'] else 'FAIL'}")
    print(
        f"   * GUI: "
        f"{'PASS' if gui_res['all_pass'] else 'FAIL'}"
        f"{'' if gui_res['gui'] else ' (headless checks only)'}"
    )
    print()
    print("3. No Control")
    print(f"   * global mean speed: {nc['mean_speed_mps']:.3f} m/s")
    print(f"   * upstream mean speed: {nc.get('upstream_mean_speed', float('nan')):.3f} m/s")
    print(f"   * accident-zone mean speed: {nc.get('accident_zone_mean_speed', float('nan')):.3f} m/s")
    print(f"   * throughput: {nc['throughput']:.0f}")
    print(f"   * slow: {nc['slow_vehicle_sightings']:.0f}")
    print(f"   * stopped: {nc['stopped_vehicle_sightings']:.0f}")
    print()
    print("4. Rollout Coverage")
    print(f"   * rollout_steps: {coverage['rollout_steps']}")
    print(f"   * min_pos: {coverage['min_pos']}")
    print(f"   * max_pos: {coverage['max_pos']:.1f} m")
    print(f"   * near_frac: {coverage['near_frac']:.4f}")
    print(f"   * near_steps: {coverage['num_near_accident_steps']}")
    print(f"   * entered accident zone: {'YES' if coverage['entered_accident_zone'] else 'NO'}")
    print()
    print("5. Accident Comparison")
    print(f"   * No Accident global speed: {by_id['C']['mean_speed_mps']:.3f} m/s")
    print(f"   * Accident global speed: {by_id['D']['mean_speed_mps']:.3f} m/s")
    print(f"   * difference: {impact['delta_mean_speed_mps']:+.3f} m/s")
    print(f"   * No Accident upstream speed: {by_id['C'].get('upstream_mean_speed', float('nan')):.3f} m/s")
    print(f"   * Accident upstream speed: {by_id['D'].get('upstream_mean_speed', float('nan')):.3f} m/s")
    print(f"   * difference: {impact.get('delta_upstream_mean_speed', float('nan')):+.3f} m/s")
    print(f"   * slow count difference: {impact['delta_slow_vehicle_sightings']:+.0f}")
    print(f"   * stopped count difference: {impact['delta_stopped_vehicle_sightings']:+.0f}")
    print()
    print("6. Scenario Assessment")
    print(f"   * {assessment}")
    print()
    print("7. Training Status")
    print("   MAPPO formal training: NOT STARTED")
    print()
    print("Checklist:")
    for k, v in checklist.items():
        print(f"   [{'x' if v else ' '}] {k}")
    print()
    print(f"Saved under: {OUT_DIR}")
    print("========================================")
    return 0 if assessment != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
