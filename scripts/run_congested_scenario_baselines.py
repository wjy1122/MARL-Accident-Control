"""Congested-scenario check: NoControl vs RuleDivert (long horizon).

Uses current configs/env.yaml (2-lane block + higher demand). Goal:
  1) NoControl should show real congestion (queue / slow near-zone).
  2) RuleDivert should beat NoControl on near speed and/or throughput/queue.

Example:
  .\\.venv\\Scripts\\python.exe scripts\\run_congested_scenario_baselines.py
  .\\.venv\\Scripts\\python.exe scripts\\run_congested_scenario_baselines.py --seconds 300 --seeds 42
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.highway_accident_env import load_config  # noqa: E402
from utils.baseline_eval import BaselineRow, run_baseline_episode, write_csv  # noqa: E402

OUT_DIR = PROJECT_ROOT / "results" / "experiments" / "congested_scenario_v1"


def _row_summary(row: BaselineRow) -> dict[str, Any]:
    return {
        "method": row.method,
        "seed": row.seed,
        "mean_speed": row.mean_speed,
        "near_speed": row.mean_speed_near_accident,
        "throughput": row.throughput,
        "queue_mean": row.queue_length,
        "queue_max": row.queue_length_max,
        "blocked_near_frac": row.blocked_lane_near_frac,
        "collision": row.collision,
        "KEEP": row.keep_ratio,
        "LEFT": row.left_ratio,
        "RIGHT": row.right_ratio,
        "sim_seconds": row.sim_seconds,
        "steps": row.steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Congested scenario NC vs RuleDivert")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Eval horizon in simulation seconds (default: configs/env.yaml test.sim_seconds)",
    )
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    step = float(cfg.get("sumo", {}).get("step_length", 0.1))
    seconds = float(
        args.seconds
        if args.seconds is not None
        else cfg.get("test", {}).get("sim_seconds", 600.0)
    )
    max_steps = max(1, int(round(seconds / step)))
    accident = cfg.get("accident", {})
    traffic = cfg.get("traffic", {})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("================================")
    print("Congested Scenario Baselines")
    print("================================")
    print(f"blocked_lanes={accident.get('blocked_lane_indices')}")
    print(f"demand={traffic.get('vehs_per_hour')} cav_pen={traffic.get('cav_penetration')}")
    print(f"horizon={seconds:g}s ({max_steps} steps) seeds={args.seeds}")
    print()

    rows: list[BaselineRow] = []
    for seed in args.seeds:
        for method in ("NoControl", "RuleDivert"):
            print(f"Running {method} seed={seed} ...", flush=True)
            row = run_baseline_episode(
                method=method,  # type: ignore[arg-type]
                seed=seed,
                gui=bool(args.gui),
                max_steps=max_steps,
            )
            rows.append(row)
            print(
                f"  mean={row.mean_speed:.3f} near={row.mean_speed_near_accident:.3f} "
                f"thr={row.throughput:.0f} queue={row.queue_length:.2f} "
                f"(max={row.queue_length_max:.0f}) blocked_near={row.blocked_lane_near_frac:.3f}"
            )

    write_csv(OUT_DIR / "metrics.csv", rows, comparison_only=True)
    write_csv(OUT_DIR / "metrics_full.csv", rows, comparison_only=False)

    # Aggregate by method (mean over seeds)
    by_method: dict[str, list[BaselineRow]] = {}
    for r in rows:
        by_method.setdefault(r.method, []).append(r)

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    agg: dict[str, dict[str, float]] = {}
    for method, rs in by_method.items():
        agg[method] = {
            "mean_speed": _mean([r.mean_speed for r in rs]),
            "near_speed": _mean([r.mean_speed_near_accident for r in rs]),
            "throughput": _mean([r.throughput for r in rs]),
            "queue_mean": _mean([r.queue_length for r in rs]),
            "queue_max": _mean([r.queue_length_max for r in rs]),
            "blocked_near_frac": _mean([r.blocked_lane_near_frac for r in rs]),
        }

    nc = agg.get("NoControl", {})
    rule = agg.get("RuleDivert", {})
    congestion_ok = bool(
        (nc.get("queue_mean", 0.0) >= 0.5)
        or (nc.get("queue_max", 0.0) >= 3.0)
        or (nc.get("near_speed", 30.0) <= 22.0)
    )
    rule_beats = bool(
        rule
        and nc
        and (
            (
                rule["near_speed"] > nc["near_speed"]
                and rule["throughput"] >= nc["throughput"] - 1e-6
            )
            or (
                rule["throughput"] > nc["throughput"]
                and rule["queue_mean"] <= nc["queue_mean"] + 1e-6
            )
            or (
                rule["queue_mean"] + 0.2 < nc["queue_mean"]
                and rule["near_speed"] >= nc["near_speed"] - 0.3
            )
        )
    )

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "blocked_lane_indices": accident.get("blocked_lane_indices"),
            "vehs_per_hour": traffic.get("vehs_per_hour"),
            "cav_penetration": traffic.get("cav_penetration"),
            "accident_position_m": accident.get("position_m"),
            "sim_seconds": seconds,
            "max_steps": max_steps,
            "seeds": args.seeds,
        },
        "aggregate": agg,
        "rows": [_row_summary(r) for r in rows],
        "checks": {
            "congestion_ok": congestion_ok,
            "rule_beats_no_control": rule_beats,
            "legacy_reference_nc_near_speed_3600_1lane": 26.52,
            "note": (
                "congestion_ok if NC queue_mean>=0.5 or queue_max>=3 or near_speed<=22; "
                "rule_beats if Rule improves near/throughput/queue vs NC under simple rules."
            ),
        },
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    md = [
        "# Congested Scenario v1 — Baseline Check",
        "",
        f"- Blocked lanes: `{accident.get('blocked_lane_indices')}`",
        f"- Demand: `{traffic.get('vehs_per_hour')}` veh/h",
        f"- Horizon: `{seconds:g}` s",
        f"- Seeds: `{args.seeds}`",
        "",
        "## Aggregate",
        "",
        "| Method | mean v | near v | thruput | queue | queue_max | blocked_near |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in ("NoControl", "RuleDivert"):
        a = agg.get(method, {})
        md.append(
            f"| {method} | {a.get('mean_speed', float('nan')):.3f} | "
            f"{a.get('near_speed', float('nan')):.3f} | "
            f"{a.get('throughput', float('nan')):.1f} | "
            f"{a.get('queue_mean', float('nan')):.3f} | "
            f"{a.get('queue_max', float('nan')):.1f} | "
            f"{a.get('blocked_near_frac', float('nan')):.3f} |"
        )
    md.extend(
        [
            "",
            "## Checks",
            "",
            f"- congestion_ok: **{congestion_ok}**",
            f"- rule_beats_no_control: **{rule_beats}**",
            "",
            "Legacy reference (1-lane / 3600 / 180s): NC near≈26.52, thruput=136, queue≈0.11.",
            "",
        ]
    )
    (OUT_DIR / "report.md").write_text("\n".join(md), encoding="utf-8")

    print()
    print(f"Saved: {OUT_DIR}")
    print(f"congestion_ok={congestion_ok}  rule_beats_no_control={rule_beats}")
    if not congestion_ok:
        print("WARN: scene still too easy — raise demand or block more before training RL.")
    if congestion_ok and not rule_beats:
        print("WARN: congested but RuleDivert did not beat NC — inspect divert logic / horizon.")
    if congestion_ok and rule_beats:
        print("OK: scene is congested and rule baseline beats NC — ready to retrain MAPPO.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
