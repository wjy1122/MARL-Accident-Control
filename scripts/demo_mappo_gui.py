"""Short SUMO-GUI smoke: 3600 + accident + MAPPO (1 update).

Does not change research logic. Restores rou.xml to env.yaml demand after run.

Example:
  .\\.venv\\Scripts\\python.exe scripts\\demo_mappo_gui.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from build_sumo_scenario import build_routes, load_config as load_build_config  # noqa: E402
from compare_accident_scenarios import write_routes_for_scenario  # noqa: E402

from env.highway_accident_env import load_config  # noqa: E402
from scripts.train_mappo import train  # noqa: E402


def main() -> int:
    out = PROJECT_ROOT / "results" / "gui_smoke_3600_accident"
    out.mkdir(parents=True, exist_ok=True)

    print("GUI smoke: demand=3600, accident=ON, updates=1, rollout_steps=512")
    env_cfg = copy.deepcopy(load_config())
    env_cfg.setdefault("traffic", {})["vehs_per_hour"] = 3600
    try:
        write_routes_for_scenario(demand=3600, with_accident=True)
        summary = train(
            num_episodes=1,
            rollout_steps=512,
            gui=True,
            checkpoint_dir=out / "checkpoints",
            metrics_dir=out,
            run_title="MAPPO GUI smoke (3600 + accident, 1 update)",
            env_cfg=env_cfg,
        )
    finally:
        build_routes(load_build_config(PROJECT_ROOT / "configs" / "env.yaml"))
        print("Restored highway.rou.xml to env.yaml demand (1800 + accident).")

    rows = summary.get("log_rows") or []
    if not rows:
        print("GUI smoke FAILED: no rows")
        return 1
    row = rows[0]
    ok = (
        summary.get("actor_changed")
        and int(row.get("rollout_steps", 0)) == 512
        and float(row.get("max_pos", 0)) > 1200.0
    )
    print(
        f"GUI smoke result: max_pos={row.get('max_pos'):.1f} "
        f"near_frac={row.get('near_frac'):.3f} "
        f"actions ran OK, actor_changed={summary.get('actor_changed')}"
    )
    print("GUI smoke PASSED" if ok else "GUI smoke FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
