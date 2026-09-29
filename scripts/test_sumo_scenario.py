"""Phase 1: verify SUMO highway accident scenario loads and runs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "env.yaml"


def _ensure_sumo_tools() -> None:
    sumo_home = os.environ.get("SUMO_HOME", "")
    if not sumo_home:
        raise RuntimeError("SUMO_HOME is not set")
    tools = os.path.join(sumo_home, "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def start_sumo(sumocfg: Path, step_length: float, gui: bool = False):
    import traci

    binary = "sumo-gui" if gui else "sumo"
    cmd = [
        binary,
        "-c",
        str(sumocfg),
        "--step-length",
        str(step_length),
        "--start",
        "--quit-on-end",
    ]
    traci.start(cmd)
    return traci


def run_checks(cfg: dict, gui: bool) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    sumocfg = PROJECT_ROOT / cfg["sumo"]["config_file"]
    if not sumocfg.is_file():
        results.append(("sumocfg exists", False, f"missing {sumocfg}"))
        return results
    results.append(("sumocfg exists", True, str(sumocfg)))

    net = PROJECT_ROOT / cfg["sumo"]["net_file"]
    results.append(("net.xml exists", net.is_file(), str(net)))

    step_length = float(cfg["sumo"]["step_length"])
    sim_seconds = float(cfg["test"]["sim_seconds"])
    steps = max(1, int(round(sim_seconds / step_length)))
    probe_r = float(cfg["test"]["accident_probe_radius_m"])
    accident_pos = float(cfg["accident"]["position_m"])
    accident_prefix = str(cfg["accident"].get("vehicle_id_prefix", "ACCIDENT"))
    blocked = [int(x) for x in cfg["accident"]["blocked_lane_indices"]]
    edge_id = str(cfg["highway"]["edge_id"])
    cong_upstream = float(cfg["test"].get("congestion_upstream_m", 250.0))
    cong_speed = float(cfg["test"].get("congestion_speed_mps", 15.0))
    min_slow = int(cfg["test"].get("min_slow_upstream", 1))

    _ensure_sumo_tools()
    traci = None
    seen_hdv = 0
    seen_cav = 0
    near_accident = 0
    max_queue_hint = 0
    min_upstream_speed = float("inf")
    accident_present = set()

    try:
        traci = start_sumo(sumocfg, step_length=step_length, gui=gui)
        results.append(("SUMO start", True, "traci.start OK"))

        for _ in range(steps):
            traci.simulationStep()
            ids = list(traci.vehicle.getIDList())
            for lane_idx in blocked:
                aid = f"{accident_prefix}_{lane_idx}"
                if aid in ids:
                    accident_present.add(aid)

            for vid in ids:
                if vid.startswith(accident_prefix):
                    continue
                vtype = traci.vehicle.getTypeID(vid)
                if vtype == "HDV":
                    seen_hdv += 1
                elif vtype == "CAV":
                    seen_cav += 1

                try:
                    lane_pos = float(traci.vehicle.getLanePosition(vid))
                except traci.TraCIException:
                    continue
                if abs(lane_pos - accident_pos) <= probe_r:
                    near_accident += 1

            # Congestion hint: slowed vehicles on the edge upstream of accident.
            slow = 0
            for vid in ids:
                if vid.startswith(accident_prefix):
                    continue
                if traci.vehicle.getRoadID(vid) != edge_id:
                    continue
                lane_pos = float(traci.vehicle.getLanePosition(vid))
                if lane_pos >= accident_pos:
                    continue
                if accident_pos - lane_pos > cong_upstream:
                    continue
                speed = float(traci.vehicle.getSpeed(vid))
                min_upstream_speed = min(min_upstream_speed, speed)
                if speed < cong_speed:
                    slow += 1
            max_queue_hint = max(max_queue_hint, slow)

        results.append(
            (
                "simulation steps",
                True,
                f"ran {steps} steps ({sim_seconds}s)",
            )
        )
        expected_accidents = {f"{accident_prefix}_{i}" for i in blocked}
        results.append(
            (
                "accident vehicle present",
                accident_present == expected_accidents,
                f"seen={sorted(accident_present)} expected={sorted(expected_accidents)}",
            )
        )

        min_hdv = int(cfg["test"]["min_hdv_seen"])
        min_cav = int(cfg["test"]["min_cav_seen"])
        min_near = int(cfg["test"]["min_near_accident"])
        results.append(
            ("HDV detected", seen_hdv >= min_hdv, f"sightings={seen_hdv}")
        )
        results.append(
            ("CAV detected", seen_cav >= min_cav, f"sightings={seen_cav}")
        )
        results.append(
            (
                "near-accident traffic",
                near_accident >= min_near,
                f"sightings={near_accident}, radius={probe_r}m",
            )
        )
        results.append(
            (
                "upstream slow vehicles (congestion hint)",
                max_queue_hint >= min_slow,
                f"max_slow_upstream={max_queue_hint}, "
                f"min_speed_seen={min_upstream_speed if min_upstream_speed < float('inf') else 'n/a'}",
            )
        )

        traci.close()
        results.append(("SUMO close", True, "traci.close OK"))
    except Exception as exc:  # noqa: BLE001
        results.append(("SUMO run", False, str(exc)))
        if traci is not None:
            try:
                traci.close()
            except Exception:  # noqa: BLE001
                pass

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Phase 1 SUMO scenario")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Use sumo-gui (manual visual check)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild scenario from config before testing",
    )
    args = parser.parse_args()

    if args.rebuild:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from build_sumo_scenario import build_all

        build_all(args.config)

    cfg = load_config(args.config)
    if args.gui:
        cfg = dict(cfg)
        cfg["sumo"] = dict(cfg["sumo"])
        cfg["sumo"]["gui"] = True

    print("Python:", sys.executable)
    print("Config:", args.config)
    results = run_checks(cfg, gui=bool(args.gui or cfg["sumo"].get("gui", False)))

    failed = 0
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
        if not ok:
            failed += 1

    if failed:
        print(f"\n{failed} check(s) failed.")
        return 1
    print("\nAll Phase 1 SUMO scenario checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
