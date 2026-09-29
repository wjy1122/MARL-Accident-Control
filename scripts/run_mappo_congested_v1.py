"""Congested v1 formal MAPPO: Candidate A + Action Mask + mid-LC suppress.

Trains from scratch on configs/env.yaml (2-lane block, 4800 veh/h), then
evaluates vs NoControl and RuleDivert (180 s).

Example:
  .\\.venv\\Scripts\\python.exe scripts\\run_mappo_congested_v1.py
  .\\.venv\\Scripts\\python.exe scripts\\run_mappo_congested_v1.py --eval-only
  .\\.venv\\Scripts\\python.exe scripts\\run_mappo_congested_v1.py --updates 100
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.rollout import resolve_device  # noqa: E402
from env.highway_accident_env import load_config  # noqa: E402
from scripts.train_mappo import train as train_mappo  # noqa: E402
from utils.baseline_eval import (  # noqa: E402
    BaselineRow,
    load_mappo_checkpoint,
    run_baseline_episode,
    write_csv,
)

OUT = PROJECT_ROOT / "results" / "experiments" / "mappo_congested_v1"
CA_REWARD = PROJECT_ROOT / "configs" / "reward_candidate_A.yaml"
DEFAULT_UPDATES = 100
ROLLOUT_STEPS = 512
EVAL_SECONDS = 180.0


def _fix_env_cfg() -> dict[str, Any]:
    cfg = copy.deepcopy(load_config())
    cfg.setdefault("action", {})["suppress_mid_lane_change"] = True
    return cfg


def _snapshot(dst: Path) -> None:
    snap = dst / "config_snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    for rel in (
        "configs/env.yaml",
        "configs/train.yaml",
        "configs/reward_candidate_A.yaml",
        "configs/model.yaml",
        "sumo/highway.rou.xml",
        "sumo/highway.sumocfg",
    ):
        src = PROJECT_ROOT / rel
        if src.is_file():
            shutil.copy2(src, snap / src.name)


def _latest_ckpt(folder: Path) -> Path | None:
    files = sorted(folder.glob("episode_*.pt"))
    return files[-1] if files else None


def _row_brief(row: BaselineRow) -> dict[str, Any]:
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
    }


def _beats(challenger: dict[str, float], baseline: dict[str, float]) -> bool:
    """True if challenger improves near speed and does not lose throughput/queue badly."""
    return bool(
        challenger["near_speed"] > baseline["near_speed"]
        and challenger["throughput"] >= baseline["throughput"] - 1e-6
        and challenger["queue_mean"] <= baseline["queue_mean"] + 0.5
    ) or bool(
        challenger["throughput"] > baseline["throughput"]
        and challenger["near_speed"] >= baseline["near_speed"] - 0.3
        and challenger["queue_mean"] <= baseline["queue_mean"] + 0.5
    )


def run_eval(*, ckpt: Path, seeds: list[int], device: str) -> list[BaselineRow]:
    cfg = _fix_env_cfg()
    step = float(cfg.get("sumo", {}).get("step_length", 0.1))
    max_steps = max(1, int(round(EVAL_SECONDS / step)))
    model, _meta = load_mappo_checkpoint(ckpt, device=device)

    rows: list[BaselineRow] = []
    for seed in seeds:
        for method in ("NoControl", "RuleDivert", "MAPPO"):
            print(f"Eval {method} seed={seed} ...", flush=True)
            if method == "MAPPO":
                row = run_baseline_episode(
                    method="MAPPO",
                    seed=seed,
                    model=model,
                    device=device,
                    max_steps=max_steps,
                    deterministic=True,
                    reward_config=CA_REWARD,
                    use_action_mask=True,
                    env_config=cfg,
                )
            else:
                row = run_baseline_episode(
                    method=method,  # type: ignore[arg-type]
                    seed=seed,
                    max_steps=max_steps,
                    reward_config=CA_REWARD,
                    env_config=cfg,
                )
            rows.append(row)
            print(
                f"  mean={row.mean_speed:.3f} near={row.mean_speed_near_accident:.3f} "
                f"thr={row.throughput:.0f} queue={row.queue_length:.2f} "
                f"blocked_near={row.blocked_lane_near_frac:.3f}"
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="MAPPO congested v1 formal train+eval")
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    _snapshot(OUT)

    n_ep = int(args.updates)
    final_ckpt = OUT / "checkpoints" / f"episode_{n_ep:04d}.pt"
    device = str(resolve_device("auto"))
    started = datetime.now(timezone.utc).isoformat()
    train_summary: dict[str, Any] | None = None

    if not args.eval_only:
        if final_ckpt.is_file():
            print(f"Checkpoint exists, skip training: {final_ckpt}", flush=True)
        else:
            resume = _latest_ckpt(OUT / "checkpoints")
            print(
                f"Training congested v1: updates={n_ep}, mask=ON, Candidate A, "
                f"suppress_mid=ON, resume={resume}",
                flush=True,
            )
            train_summary = train_mappo(
                num_episodes=n_ep,
                rollout_steps=ROLLOUT_STEPS,
                gui=False,
                checkpoint_dir=OUT / "checkpoints",
                metrics_dir=OUT / "logs",
                run_title="Congested v1 MAPPO (Candidate A + Mask + mid-LC suppress)",
                env_cfg=_fix_env_cfg(),
                reward_config=CA_REWARD,
                resume_from=resume,
                use_action_mask=True,
            )
            (OUT / "logs" / "train_summary.json").write_text(
                json.dumps(
                    {k: v for k, v in train_summary.items() if k != "log_rows"},
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )

    if not final_ckpt.is_file():
        # Accept latest if target episode not reached yet
        latest = _latest_ckpt(OUT / "checkpoints")
        if latest is None:
            print("FAIL: no checkpoint to evaluate", flush=True)
            return 1
        print(f"WARN: target {final_ckpt.name} missing; using {latest.name}", flush=True)
        final_ckpt = latest

    if args.skip_eval:
        print(f"Skip eval. Checkpoint: {final_ckpt}", flush=True)
        return 0

    rows = run_eval(ckpt=final_ckpt, seeds=args.seeds, device=device)
    write_csv(OUT / "eval_metrics.csv", rows, comparison_only=True)
    write_csv(OUT / "eval_metrics_full.csv", rows, comparison_only=False)

    by: dict[str, list[BaselineRow]] = {}
    for r in rows:
        by.setdefault(r.method, []).append(r)

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    agg: dict[str, dict[str, float]] = {}
    for method, rs in by.items():
        agg[method] = {
            "mean_speed": _mean([r.mean_speed for r in rs]),
            "near_speed": _mean([r.mean_speed_near_accident for r in rs]),
            "throughput": _mean([r.throughput for r in rs]),
            "queue_mean": _mean([r.queue_length for r in rs]),
            "blocked_near_frac": _mean([r.blocked_lane_near_frac for r in rs]),
        }

    nc, rule, mappo = agg.get("NoControl", {}), agg.get("RuleDivert", {}), agg.get("MAPPO", {})
    beats_nc = bool(mappo and nc and _beats(mappo, nc))
    beats_rule = bool(mappo and rule and _beats(mappo, rule))
    status = "PASS" if beats_nc else "PARTIAL"
    if beats_nc and beats_rule:
        status = "PASS_STRONG"

    finished = datetime.now(timezone.utc).isoformat()
    summary = {
        "started_utc": started,
        "finished_utc": finished,
        "checkpoint": str(final_ckpt),
        "updates_target": n_ep,
        "eval_seconds": EVAL_SECONDS,
        "seeds": args.seeds,
        "device": device,
        "aggregate": agg,
        "checks": {
            "mappo_beats_no_control": beats_nc,
            "mappo_beats_rule_divert": beats_rule,
            "status": status,
        },
        "rows": [_row_brief(r) for r in rows],
        "train_summary_keys": list(train_summary.keys()) if train_summary else [],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Congested v1 MAPPO Formal Report",
        "",
        f"- Checkpoint: `{final_ckpt}`",
        f"- Reward: Candidate A + Action Mask + mid-LC suppress",
        f"- Eval horizon: {EVAL_SECONDS:g} s, seeds={args.seeds}",
        f"- Status: **{status}**",
        "",
        "## Aggregate",
        "",
        "| Method | mean v | near v | thruput | queue | blocked_near |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in ("NoControl", "RuleDivert", "MAPPO"):
        a = agg.get(method, {})
        md.append(
            f"| {method} | {a.get('mean_speed', float('nan')):.3f} | "
            f"{a.get('near_speed', float('nan')):.3f} | "
            f"{a.get('throughput', float('nan')):.1f} | "
            f"{a.get('queue_mean', float('nan')):.3f} | "
            f"{a.get('blocked_near_frac', float('nan')):.3f} |"
        )
    md.extend(
        [
            "",
            "## Checks",
            "",
            f"- mappo_beats_no_control: **{beats_nc}**",
            f"- mappo_beats_rule_divert: **{beats_rule}**",
            "",
        ]
    )
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved: {OUT}")
    print(f"status={status} beats_nc={beats_nc} beats_rule={beats_rule}")
    return 0 if beats_nc else 1


if __name__ == "__main__":
    raise SystemExit(main())
