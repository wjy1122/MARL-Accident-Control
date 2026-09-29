"""Phase 4.4.1: Evaluate a fixed MAPPO checkpoint (no training).

Example:
  .\\.venv\\Scripts\\python.exe scripts\\evaluate_mappo.py \\
    --checkpoint results/mappo_group_benefit/checkpoints/episode_0010.pt \\
    --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.rollout import resolve_device  # noqa: E402
from utils.baseline_eval import (  # noqa: E402
    COMPARISON_FIELDS,
    BaselineRow,
    load_mappo_checkpoint,
    run_baseline_episode,
    write_csv,
)


def _read_comparison_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description="MAPPO checkpoint evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/mappo_group_benefit/checkpoints/episode_0010.pt",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--out",
        type=str,
        default="results/baselines/mappo_metrics.csv",
    )
    parser.add_argument(
        "--no-control-csv",
        type=str,
        default="results/baselines/no_control_metrics.csv",
    )
    parser.add_argument(
        "--comparison-out",
        type=str,
        default="results/comparison/no_control_vs_mappo.csv",
    )
    args = parser.parse_args()

    device = resolve_device("auto")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        ckpt_path = PROJECT_ROOT / args.checkpoint
    if not ckpt_path.is_file():
        print(f"Checkpoint not found: {args.checkpoint}")
        return 1

    model, ckpt = load_mappo_checkpoint(ckpt_path, device=device)
    ep = ckpt.get("episode", "?")

    print("================================")
    print("MAPPO Checkpoint Evaluation")
    print("================================")
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Saved episode: {ep}")
    print("Training: OFF (model.eval + no_grad)")
    print(f"Seeds: {args.seeds}")
    print()

    rows: list[BaselineRow] = []
    for seed in args.seeds:
        print(f"Running seed={seed} ...")
        row = run_baseline_episode(
            method="MAPPO",
            seed=seed,
            model=model,
            device=device,
            gui=args.gui,
        )
        rows.append(row)
        print(
            f"  mean_speed={row.mean_speed:.3f} "
            f"near={row.mean_speed_near_accident:.3f} "
            f"queue={row.queue_length:.2f} "
            f"throughput={row.throughput:.0f} "
            f"travel_time={row.travel_time:.2f} "
            f"collision={row.collision} "
            f"actions K/L/R="
            f"{row.keep_ratio:.2f}/{row.left_ratio:.2f}/{row.right_ratio:.2f} "
            f"near_n={row.near_action_n}"
        )

    out = PROJECT_ROOT / args.out
    write_csv(out, rows, comparison_only=True)
    write_csv(out.with_name("mappo_metrics_full.csv"), rows, comparison_only=False)
    print(f"\nSaved: {out}")

    # Merge with No Control CSV if present
    nc_path = PROJECT_ROOT / args.no_control_csv
    nc_rows = _read_comparison_csv(nc_path)
    cmp_path = PROJECT_ROOT / args.comparison_out
    cmp_path.parent.mkdir(parents=True, exist_ok=True)
    merged: list[dict] = list(nc_rows)
    for r in rows:
        merged.append(r.comparison_dict())
    # Deduplicate by (seed, method) keeping last
    uniq: dict[tuple, dict] = {}
    for d in merged:
        key = (str(d.get("seed")), str(d.get("method")))
        uniq[key] = d
    final = sorted(uniq.values(), key=lambda d: (int(d["seed"]), d["method"]))
    with cmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(final)
    print(f"Comparison: {cmp_path}")
    if not nc_rows:
        print(
            "Note: No Control CSV missing; comparison currently contains MAPPO only. "
            "Run scripts/evaluate_no_control.py first for a full table."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
