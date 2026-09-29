"""Phase 4.4.1: smoke test for No Control / MAPPO evaluation helpers."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.baseline_eval import run_baseline_episode  # noqa: E402


def main() -> int:
    # Short horizon for CI-speed smoke (still exercises KEEP path + metrics).
    row = run_baseline_episode(method="NoControl", seed=42, max_steps=100)
    checks = [
        ("method", row.method == "NoControl"),
        ("steps", row.steps == 100),
        ("mean_speed finite", row.mean_speed == row.mean_speed),  # not NaN if traffic
        ("collision >= 0", row.collision >= 0),
        ("keep_ratio ~1", abs(row.keep_ratio - 1.0) < 1e-9),
    ]
    failed = 0
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            failed += 1
    print(
        f"metrics: speed={row.mean_speed:.3f} near={row.mean_speed_near_accident} "
        f"queue={row.queue_length:.2f} thr={row.throughput} tt={row.travel_time} "
        f"col={row.collision}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
