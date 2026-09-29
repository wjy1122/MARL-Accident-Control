"""Formal MAPPO training: 50 updates × 512 steps (headless), Phase 4.4.9 scene.

Does NOT modify MAPPO / Reward / Observation / scene. GUI off.

Example:
  .\\.venv\\Scripts\\python.exe scripts\\run_mappo_formal_50u.py
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_mappo import train  # noqa: E402

OUT_DIR = PROJECT_ROOT / "results" / "experiments" / "mappo_formal_50u_1200m_700m_seed42"
TARGET_UPDATES = 50
ROLLOUT_STEPS = 512


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def _snapshot(dst: Path) -> None:
    snap = dst / "config_snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    for rel in (
        "configs/env.yaml",
        "configs/train.yaml",
        "configs/reward.yaml",
        "configs/model.yaml",
        "sumo/highway.rou.xml",
        "sumo/highway.sumocfg",
        "algorithms/rollout.py",
    ):
        src = PROJECT_ROOT / rel
        if src.is_file():
            shutil.copy2(src, snap / src.name)


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _assess(rows: list[dict[str, Any]], summary: dict[str, Any], *, error: str | None) -> str:
    if error:
        return "FAIL"
    if len(rows) < TARGET_UPDATES:
        return "FAIL"
    if not summary.get("actor_changed") or not summary.get("critic_changed"):
        return "FAIL"
    # Lifecycle: first update should not spam resets beyond start+truncate budget
    resets = [int(r.get("env_reset_count", 99)) for r in rows]
    # Per-update reset counts of 0–2 are OK (start / truncate); frequent high resets = problem
    high_reset_updates = sum(1 for x in resets if x > 3)
    if high_reset_updates > 5:
        return "BORDERLINE"
    return "PASS"


def _write_report(
    *,
    out: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    result: str,
    error: str | None,
    started: str,
    finished: str,
    device: str,
    gpu: str | None,
) -> None:
    n = len(rows)
    idx_mid = n // 2 if n else 0
    r0 = rows[0] if rows else {}
    rm = rows[idx_mid] if rows else {}
    rL = rows[-1] if rows else {}

    def _row_line(label: str, r: dict[str, Any]) -> str:
        return (
            f"- {label} (ep {r.get('episode')}): "
            f"reward={float(r.get('mean_reward', float('nan'))):.6f}, "
            f"actor={float(r.get('actor_loss', float('nan'))):.6f}, "
            f"critic={float(r.get('critic_loss', float('nan'))):.6f}, "
            f"ent={float(r.get('entropy', float('nan'))):.6f}"
        )

    keep = _mean([float(r.get("KEEP_pct", 0)) for r in rows])
    left = _mean([float(r.get("LEFT_pct", 0)) for r in rows])
    right = _mean([float(r.get("RIGHT_pct", 0)) for r in rows])
    sum_starts = sum(int(r.get("sumo_start_count", 0)) for r in rows)
    sum_resets = sum(int(r.get("env_reset_count", 0)) for r in rows)
    sum_dep = sum(int(r.get("departed_cav_count", 0)) for r in rows)
    sum_new = sum(int(r.get("new_cav_count", 0)) for r in rows)
    sum_rebind = sum(int(r.get("rebind_events", 0)) for r in rows)
    peak_active = max((int(r.get("active_cav_peak", 0)) for r in rows), default=0)

    env = yaml.safe_load((PROJECT_ROOT / "configs" / "env.yaml").read_text(encoding="utf-8"))
    collapse = max(keep, left, right) >= 95.0

    md = f"""# MAPPO Formal Training Report (50 updates)

## 1. Configuration

- Road length: {env['highway']['length_m']} m
- Accident: {env['accident']['position_m']} m / lane {env['accident']['blocked_lane_indices']}
- Demand: {env['traffic']['vehs_per_hour']} veh/h
- CAV penetration: {env['traffic']['cav_penetration']}
- Seed: 42
- Rollout steps: {ROLLOUT_STEPS}
- Target updates: {TARGET_UPDATES}
- Device: {device}
- GPU: {gpu or 'n/a'}
- GUI: OFF (sumo headless)
- Reward: 0.7 individual + 0.3 group (unchanged)
- Observation: 17-D (unchanged)
- Lifecycle: fixed-N slot rebind (unchanged)
- Output: `{out.as_posix()}`
- Started (UTC): {started}
- Finished (UTC): {finished}

## 2. Training completion

- target_updates = {TARGET_UPDATES}
- completed_updates = {n}
- total_environment_steps = {n * ROLLOUT_STEPS}
- error = {error or 'none'}

## 3. MAPPO

{_row_line('initial', r0)}
{_row_line('middle', rm)}
{_row_line('final', rL)}

- mean reward (all updates): {_mean([float(r.get('mean_reward', 0)) for r in rows]):.6f}
- mean actor loss: {_mean([float(r.get('actor_loss', 0)) for r in rows]):.6f}
- mean critic loss: {_mean([float(r.get('critic_loss', 0)) for r in rows]):.6f}
- mean entropy: {_mean([float(r.get('entropy', 0)) for r in rows]):.6f}
- Actor params updated: {summary.get('actor_changed')}
- Critic params updated: {summary.get('critic_changed')}

## 4. Action distribution

Mean over all updates:

- KEEP: {keep:.1f}%
- LEFT: {left:.1f}%
- RIGHT: {right:.1f}%
- Collapse (>=95% one action): {'YES' if collapse else 'NO'}

Final update: KEEP={float(rL.get('KEEP_pct', float('nan'))):.1f}% LEFT={float(rL.get('LEFT_pct', float('nan'))):.1f}% RIGHT={float(rL.get('RIGHT_pct', float('nan'))):.1f}%

## 5. Lifecycle

Summed across updates (per-update counters):

- sumo_start_count (sum): {sum_starts}
- env_reset_count (sum): {sum_resets}
- departed_cav_count (sum): {sum_dep}
- new_cav_count (sum): {sum_new}
- rebind_events (sum): {sum_rebind}
- active_cav_peak (max): {peak_active}

Interpretation: resets/starts should reflect initial connect + episode truncate (`max_episode_steps=1000`), **not** CAV departure. departed/new/rebind > 0 indicates continuous vehicle lifecycle.

## 6. Stability

- NaN/Inf: {'YES (aborted)' if error and 'NaN' in error else 'NO (completed)'}
- Unexpected abort: {error or 'none'}
- CUDA / TraCI / shape: see stdout.log if any

## 7. Training curve

Raw series in `train_metrics.csv` (episode, mean_reward, actor_loss, critic_loss, entropy, lifecycle fields). No extra plotting code added for this run.

## 8. Final judgment

**{result}**

Judgment basis: completed {n}/{TARGET_UPDATES} updates, no fatal exception, Actor/Critic updated, lifecycle counters present. **Not** a judgment of final traffic-control performance.
"""
    (out / "report.md").write_text(md, encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    log_path = OUT_DIR / "stdout.log"
    log_f = log_path.open("w", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = Tee(old_out, log_f)
    sys.stderr = Tee(old_err, log_f)

    error: str | None = None
    summary: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    try:
        print("========================================")
        print("MAPPO Formal Training — 50 updates")
        print("1200m / accident@700m / demand=3600")
        print("========================================")
        print(f"Output: {OUT_DIR}")
        print(f"Started UTC: {started}")
        print(f"updates={TARGET_UPDATES} rollout_steps={ROLLOUT_STEPS} GUI=OFF seed=42")
        print()
        _snapshot(OUT_DIR)

        summary = train(
            num_episodes=TARGET_UPDATES,
            rollout_steps=ROLLOUT_STEPS,
            gui=False,
            checkpoint_dir=OUT_DIR / "checkpoints",
            metrics_dir=OUT_DIR,
            run_title="MAPPO Formal 50u (1200m / accident@700m, seed=42, headless)",
        )
        rows = list(summary.get("log_rows") or [])
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        print(f"FATAL: {error}")
        raise
    finally:
        finished = datetime.now(timezone.utc).isoformat()
        # Prefer CSV if train partially wrote metrics before crash
        csv_path = OUT_DIR / "train_metrics.csv"
        if not rows and csv_path.is_file():
            with csv_path.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        result = _assess(rows, summary, error=error)
        device = str(summary.get("device", "unknown"))
        gpu = None
        try:
            import torch

            if str(device).startswith("cuda") and torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            gpu = None
        _write_report(
            out=OUT_DIR,
            rows=rows,
            summary=summary,
            result=result,
            error=error,
            started=started,
            finished=finished,
            device=device,
            gpu=gpu,
        )
        meta = {
            "result": result,
            "error": error,
            "completed_updates": len(rows),
            "target_updates": TARGET_UPDATES,
            "summary_keys": list(summary.keys()),
            "actor_changed": summary.get("actor_changed"),
            "critic_changed": summary.get("critic_changed"),
        }
        (OUT_DIR / "run_meta.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )
        print()
        print(f"Formal 50u Result: {result}")
        print(f"Report: {OUT_DIR / 'report.md'}")
        sys.stdout = old_out
        sys.stderr = old_err
        log_f.close()

    return 0 if result != "FAIL" and error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
