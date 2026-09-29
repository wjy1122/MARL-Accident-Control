"""Phase 2.1–2.2: TraCI environment + vehicle state reading tests (headless)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.highway_accident_env import HighwayAccidentEnv  # noqa: E402


def _sumo_process_count() -> int:
    """Count running sumo / sumo-gui processes (Windows-friendly)."""
    try:
        out = subprocess.check_output(
            ["tasklist", "/FO", "CSV", "/NH"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return -1

    count = 0
    for line in out.splitlines():
        lower = line.lower()
        if '"sumo.exe"' in lower or '"sumo-gui.exe"' in lower:
            count += 1
    return count


def main() -> int:
    results: list[tuple[str, bool, str]] = []
    env: HighwayAccidentEnv | None = None
    before = _sumo_process_count()

    try:
        env = HighwayAccidentEnv(gui=False)
        results.append(("env construct", True, "HighwayAccidentEnv OK"))

        state0 = env.reset()
        results.append(("SUMO + TraCI start", env.is_connected, f"time={state0['time']}"))
        results.append(
            (
                "initial snapshot keys",
                all(
                    k in state0
                    for k in (
                        "time",
                        "vehicle_ids",
                        "cav_ids",
                        "hdv_ids",
                        "n_vehicles",
                        "vehicles",
                    )
                ),
                str(sorted(state0.keys())),
            )
        )

        state1 = env.step()
        results.append(
            (
                "simulation step",
                state1["step_count"] == 1 and state1["time"] > state0["time"],
                f"time {state0['time']} -> {state1['time']}",
            )
        )

        last = state1
        max_probe_steps = 2000
        for _ in range(max_probe_steps):
            last = env.step()
            if last["n_cav"] > 0:
                break

        results.append(
            (
                "vehicle list available",
                last["n_vehicles"] > 0,
                f"n_vehicles={last['n_vehicles']} at t={last['time']}",
            )
        )
        results.append(
            (
                "basic vehicle states",
                isinstance(last.get("vehicles"), dict) and len(last["vehicles"]) > 0,
                f"keys={list(last['vehicles'].keys())[:5]}",
            )
        )
        results.append(
            (
                "identify CAV",
                last["n_cav"] > 0,
                f"cav_ids={last['cav_ids'][:5]}",
            )
        )

        # Accident must not be classified as CAV/HDV.
        accident_ok = True
        accident_detail = "none yet"
        for vid, st in last["vehicles"].items():
            if st["type"] == "ACCIDENT":
                accident_ok = (
                    env.classify_vehicle(vid) == "ACCIDENT"
                    and vid not in last["cav_ids"]
                    and vid not in last["hdv_ids"]
                )
                accident_detail = f"{vid} type={st['type']}"
                break
        else:
            # Advance a bit more if accident not spawned yet (should be at t=0).
            for _ in range(50):
                last = env.step()
                for vid, st in last["vehicles"].items():
                    if st["type"] == "ACCIDENT":
                        accident_ok = vid not in last["cav_ids"]
                        accident_detail = f"{vid} type={st['type']}"
                        break
                if "ACCIDENT" in {st["type"] for st in last["vehicles"].values()}:
                    break
        results.append(("ACCIDENT not treated as CAV/HDV", accident_ok, accident_detail))

        # Validate TraCI fields on one CAV.
        cav_id = last["cav_ids"][0]
        cav = env.get_vehicle_state(cav_id)
        required = ("id", "type", "lane", "lane_id", "position", "speed")
        results.append(
            (
                "CAV state fields",
                all(k in cav for k in required) and cav["type"] == "CAV",
                str(cav),
            )
        )
        results.append(
            (
                "lane index readable",
                isinstance(cav["lane"], int) and 0 <= cav["lane"] <= 2,
                f"lane={cav['lane']} lane_id={cav['lane_id']}",
            )
        )
        results.append(
            (
                "position readable",
                isinstance(cav["position"], float) and cav["position"] >= 0.0,
                f"position={cav['position']:.3f}",
            )
        )
        results.append(
            (
                "speed readable",
                isinstance(cav["speed"], float) and cav["speed"] >= 0.0,
                f"speed={cav['speed']:.3f}",
            )
        )

        # Direct API: get_cav_states excludes accident vehicles.
        cav_map = env.get_cav_states()
        results.append(
            (
                "get_cav_states()",
                len(cav_map) == last["n_cav"]
                and all(st["type"] == "CAV" for st in cav_map.values()),
                f"n={len(cav_map)}",
            )
        )

        env.close()
        results.append(("close()", not env.is_connected, "disconnected"))

        after = _sumo_process_count()
        if before >= 0 and after >= 0:
            results.append(
                (
                    "no residual SUMO process increase",
                    after <= before,
                    f"before={before}, after={after}",
                )
            )
        else:
            results.append(
                ("no residual SUMO process increase", True, "tasklist unavailable; skipped")
            )

    except Exception as exc:  # noqa: BLE001
        results.append(("env run", False, str(exc)))
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass

    failed = 0
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
        if not ok:
            failed += 1

    if failed:
        print(f"\n{failed} check(s) failed.")
        return 1
    print("\nAll Phase 2.2 env checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
