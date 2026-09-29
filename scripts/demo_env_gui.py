"""GUI demo: Phase 3.2 env.step(actions) closed loop with SUMO-GUI.

Example:
  .\\.venv\\Scripts\\python.exe scripts\\demo_env_gui.py
  .\\.venv\\Scripts\\python.exe scripts\\demo_env_gui.py --steps 500 --delay-ms 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.action import KEEP, LEFT, RIGHT  # noqa: E402
from env.highway_accident_env import HighwayAccidentEnv  # noqa: E402

# Closed-loop decision step -> action for the tracked CAV
ACTION_SCHEDULE: dict[int, int] = {
    1: KEEP,
    2: LEFT,
    3: KEEP,
    4: RIGHT,
    5: KEEP,
}


def _pick_demo_cav(cav_ids: list[str], vehicles: dict) -> str | None:
    if not cav_ids:
        return None
    return max(cav_ids, key=lambda vid: vehicles[vid]["position"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SUMO-GUI demo of Observation+Action+Reward via env.step(actions)"
    )
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--delay-ms", type=float, default=50.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50, help="Legacy steps until CAV appears")
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()

    env = HighwayAccidentEnv(gui=True, delay_ms=args.delay_ms)
    print("[Demo] Starting SUMO-GUI (Phase 3.2 closed-loop step)...")
    print(f"[Demo] Config: {env.sumocfg_path}")
    print(f"[Demo] Accident zone: {env.accident_position_m}m")
    print(f"[Demo] max_episode_steps: {env.max_episode_steps}")

    snap = env.reset()
    print(f"[Demo] reset OK, step_count={env.step_count}, time={snap['time']}")

    # Warmup with legacy step() so traffic appears (does not use RL API).
    for _ in range(args.warmup):
        snap = env.step()
        if snap["n_cav"] > 0:
            break
    print(f"[Demo] Warmup done, n_cav={snap['n_cav']}, sim_step={env.step_count}")

    tracked: str | None = None
    decision = 0
    try:
        while decision < args.steps:
            vehicles = env.get_vehicle_states()
            cav_ids = [vid for vid, st in vehicles.items() if st["type"] == "CAV"]
            if tracked is None or tracked not in cav_ids:
                tracked = _pick_demo_cav(cav_ids, vehicles)

            actions = {cav_id: KEEP for cav_id in cav_ids}
            decision += 1
            if tracked is not None and decision in ACTION_SCHEDULE:
                actions[tracked] = ACTION_SCHEDULE[decision]

            obs, rewards, terminated, truncated, info = env.step(actions)

            if decision % args.log_every == 0 or decision in ACTION_SCHEDULE:
                if tracked is not None and tracked in env.get_vehicle_states():
                    st = env.get_vehicle_state(tracked)
                    action_name = {KEEP: "KEEP", LEFT: "LEFT", RIGHT: "RIGHT"}[
                        actions.get(tracked, KEEP)
                    ]
                    print(f"Step {info['step']}")
                    print(f"CAV {tracked}")
                    print(f"Action: {action_name}")
                    print(f"Lane: {st['lane']}")
                    print(f"Reward: {rewards.get(tracked, float('nan')):.3f}")
                    print(f"Obs dim: {len(obs[tracked]) if tracked in obs else 0}")
                    print(f"terminated={terminated} truncated={truncated}")
                    print()
                else:
                    print(f"Step {info['step']}: no tracked CAV yet")
                    print()

            if truncated or terminated:
                print("[Demo] Episode end flag raised; stopping loop.")
                break

        if args.no_wait:
            print("[Demo] Finished; closing GUI (--no-wait).")
        else:
            print("[Demo] Finished. Press Enter to close SUMO-GUI...")
            try:
                input()
            except EOFError:
                pass
    finally:
        env.close()
        print("[Demo] Closed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
