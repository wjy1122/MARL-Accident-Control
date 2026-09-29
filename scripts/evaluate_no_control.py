"""Phase 4.4.1: No Control baseline (all CAV actions = KEEP).

Example:
  .\\.venv\\Scripts\\python.exe scripts\\evaluate_no_control.py --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.baseline_eval import run_baseline_episode, write_csv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="No Control baseline evaluation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--out",
        type=str,
        default="results/baselines/no_control_metrics.csv",
    )
    args = parser.parse_args()

    print("================================")
    print("No Control Baseline")
    print("================================")
    print("Policy: all CAV -> KEEP (0)")
    print(f"Seeds: {args.seeds}")
    print()

    rows = []
    for seed in args.seeds:
        print(f"Running seed={seed} ...")
        row = run_baseline_episode(method="NoControl", seed=seed, gui=args.gui)
        rows.append(row)
        print(
            f"  mean_speed={row.mean_speed:.3f} "
            f"near={row.mean_speed_near_accident:.3f} "
            f"queue={row.queue_length:.2f} "
            f"throughput={row.throughput:.0f} "
            f"travel_time={row.travel_time:.2f} "
            f"collision={row.collision} "
            f"near_actions={row.near_action_n}"
        )

    out = PROJECT_ROOT / args.out
    write_csv(out, rows, comparison_only=True)
    # Also keep a richer dump next to it
    write_csv(out.with_name("no_control_metrics_full.csv"), rows, comparison_only=False)
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
