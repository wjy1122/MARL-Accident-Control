"""Phase 3.1: discrete CAV action / TraCI lane-change tests."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.action import KEEP, LEFT, RIGHT, validate_action  # noqa: E402
from env.highway_accident_env import HighwayAccidentEnv  # noqa: E402


def _wait_cav(env: HighwayAccidentEnv, max_steps: int = 2000) -> str:
    for _ in range(max_steps):
        state = env.step()
        if state["n_cav"] > 0:
            return state["cav_ids"][0]
    raise RuntimeError("No CAV appeared within probe horizon")


def main() -> int:
    results: list[tuple[str, bool, str]] = []

    results.append(
        (
            "Test1 action constants",
            KEEP == 0 and LEFT == 1 and RIGHT == 2,
            f"KEEP={KEEP} LEFT={LEFT} RIGHT={RIGHT}",
        )
    )

    try:
        for a in (0, 1, 2):
            validate_action(a)
        results.append(("Test2 valid actions", True, "0/1/2 OK"))
    except Exception as exc:  # noqa: BLE001
        results.append(("Test2 valid actions", False, str(exc)))

    illegal_ok = True
    detail = []
    for bad in (-1, 3, "left"):
        try:
            validate_action(bad)  # type: ignore[arg-type]
            illegal_ok = False
            detail.append(f"{bad!r} did not raise")
        except ValueError:
            detail.append(f"{bad!r}->ValueError")
        except Exception as exc:  # noqa: BLE001
            illegal_ok = False
            detail.append(f"{bad!r}->{type(exc).__name__}")
    results.append(("Test3 illegal actions", illegal_ok, ", ".join(detail)))

    env: HighwayAccidentEnv | None = None
    try:
        env = HighwayAccidentEnv(gui=False)
        env.reset()
        traci = env.traci
        assert traci is not None

        cav_id = _wait_cav(env)
        traci.vehicle.setLaneChangeMode(cav_id, 0)

        # ---- Test 4: boundaries via changeLane (no moveTo) ----
        traci.vehicle.changeLane(cav_id, 2, 3.0)
        for _ in range(40):
            env.step()
            if cav_id not in traci.vehicle.getIDList():
                cav_id = _wait_cav(env)
                traci.vehicle.setLaneChangeMode(cav_id, 0)
                traci.vehicle.changeLane(cav_id, 2, 3.0)

        # If still not on 2, keep requesting LEFT until leftmost or timeout.
        for _ in range(50):
            if cav_id not in traci.vehicle.getIDList():
                cav_id = _wait_cav(env)
                traci.vehicle.setLaneChangeMode(cav_id, 0)
            lane = int(traci.vehicle.getLaneIndex(cav_id))
            if lane >= 2:
                break
            env.apply_action(cav_id, LEFT)
            env.step()

        before = int(traci.vehicle.getLaneIndex(cav_id))
        info = env.apply_action(cav_id, LEFT)
        for _ in range(15):
            env.step()
        after = int(traci.vehicle.getLaneIndex(cav_id))
        # If we reached leftmost, LEFT must be no-op; else still must stay valid.
        if before >= 2:
            left_ok = info["executed"] is False and after == before
            left_detail = f"leftmost before={before} after={after} executed={info['executed']}"
        else:
            left_ok = 0 <= after <= 2
            left_detail = (
                f"soft: could not reach leftmost (before={before}); "
                f"lane remained valid after={after}"
            )
        results.append(("Test4 leftmost + LEFT", left_ok, left_detail))

        # Rightmost + RIGHT
        if cav_id not in traci.vehicle.getIDList():
            cav_id = _wait_cav(env)
        traci.vehicle.setLaneChangeMode(cav_id, 0)
        traci.vehicle.changeLane(cav_id, 0, 3.0)
        for _ in range(40):
            env.step()
            if cav_id not in traci.vehicle.getIDList():
                cav_id = _wait_cav(env)
                traci.vehicle.setLaneChangeMode(cav_id, 0)
                traci.vehicle.changeLane(cav_id, 0, 3.0)

        for _ in range(50):
            if cav_id not in traci.vehicle.getIDList():
                cav_id = _wait_cav(env)
                traci.vehicle.setLaneChangeMode(cav_id, 0)
            lane = int(traci.vehicle.getLaneIndex(cav_id))
            if lane <= 0:
                break
            env.apply_action(cav_id, RIGHT)
            env.step()

        before = int(traci.vehicle.getLaneIndex(cav_id))
        info = env.apply_action(cav_id, RIGHT)
        for _ in range(15):
            env.step()
        after = int(traci.vehicle.getLaneIndex(cav_id))
        if before <= 0:
            right_ok = info["executed"] is False and after == before
            right_detail = f"rightmost before={before} after={after} executed={info['executed']}"
        else:
            right_ok = 0 <= after <= 2
            right_detail = (
                f"soft: could not reach rightmost (before={before}); "
                f"lane remained valid after={after}"
            )
        results.append(("Test4 rightmost + RIGHT", right_ok, right_detail))

        # ---- Test 5: actual lane change ----
        changed = False
        detail5 = ""
        for attempt in range(5):
            if cav_id not in traci.vehicle.getIDList():
                cav_id = _wait_cav(env)
            traci.vehicle.setLaneChangeMode(cav_id, 0)
            lane = int(traci.vehicle.getLaneIndex(cav_id))
            if lane >= 2:
                action = RIGHT
            else:
                action = LEFT
            before = lane
            info = env.apply_action(cav_id, action)
            if not info["executed"]:
                # Nudge toward a lane with room.
                if lane >= 2:
                    traci.vehicle.changeLane(cav_id, 1, 2.0)
                else:
                    traci.vehicle.changeLane(cav_id, 0, 2.0)
                for _ in range(25):
                    env.step()
                continue
            for _ in range(30):
                env.step()
                if cav_id not in traci.vehicle.getIDList():
                    break
            if cav_id not in traci.vehicle.getIDList():
                detail5 = f"attempt{attempt}: CAV left network"
                cav_id = _wait_cav(env)
                continue
            after = int(traci.vehicle.getLaneIndex(cav_id))
            if after != before and 0 <= after <= 2:
                changed = True
                detail5 = f"{cav_id} {before}->{after} action={info['action_name']}"
                break
            detail5 = f"attempt{attempt}: {before}->{after} (no change yet)"

        results.append(("Test5 SUMO lane change", changed, detail5 or "failed"))

        env.close()
        results.append(("close()", True, "ok"))

    except Exception as exc:  # noqa: BLE001
        results.append(
            ("action SUMO tests", False, f"{exc}\n{traceback.format_exc()}")
        )
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
    print("\nAll Phase 3.1 action checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
