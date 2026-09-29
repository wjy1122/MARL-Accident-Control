"""Phase 4.4.3: 512-step MAPPO coverage smoke test.

Uses configs/train.yaml rollout_steps=512 (do not hard-code a different value).

Example:
  .\\.venv\\Scripts\\python.exe scripts\\smoke_train_512.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.rollout import load_train_config  # noqa: E402
from scripts.train_mappo import train  # noqa: E402


def main() -> int:
    cfg = load_train_config()
    tblock = cfg.get("train", cfg)
    r_steps = int(tblock.get("rollout_steps", -1))
    print("================================")
    print("Phase 4.4.3 Smoke Launcher")
    print("================================")
    print(f"Config rollout_steps = {r_steps}")
    if r_steps != 512:
        print("FAIL: configs/train.yaml must set rollout_steps: 512")
        return 1

    # Few updates only — coverage verification, not full training.
    summary = train(num_episodes=2, rollout_steps=None, gui=False)
    rows = summary.get("log_rows") or []
    if not rows:
        print("FAIL: no training rows")
        return 1

    # Require every smoke update to cover the accident zone.
    all_cov = True
    for row in rows:
        ok = (
            int(row["rollout_steps"]) == 512
            and float(row["max_pos"]) > 1200.0
            and float(row["near_frac"]) > 0.0
            and int(row["num_near_accident_steps"]) > 0
        )
        print(
            f"update={row['episode']}: max_pos={row['max_pos']:.1f} "
            f"near_frac={row['near_frac']:.3f} "
            f"num_near={row['num_near_accident_steps']} "
            f"mean_reward={row['mean_reward']:.4f} "
            f"[{'PASS' if ok else 'FAIL'}]"
        )
        all_cov = all_cov and ok

    params_ok = summary["actor_changed"] and summary["critic_changed"]
    ok = all_cov and params_ok and bool(summary.get("coverage_ok"))
    print("Phase 4.4.3 Smoke PASSED" if ok else "Phase 4.4.3 Smoke FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
