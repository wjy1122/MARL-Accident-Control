"""Phase 5: Formal checkpoint evaluation — MAPPO vs No Control (paired).

Protocol:
  checkpoint = episode_0050.pt
  seeds = 42, 43, 44
  sim = 1800 steps (180 s)
  MAPPO = inference only, deterministic=True
  NoControl = CAV KEEP(0)
  GUI = OFF

Does NOT train / modify Reward / Observation / scene.

Example:
  .\\.venv\\Scripts\\python.exe scripts\\evaluate_formal_checkpoint.py
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.rollout import resolve_device  # noqa: E402
from utils.baseline_eval import (  # noqa: E402
    BaselineRow,
    load_mappo_checkpoint,
    params_unchanged,
    run_baseline_episode,
    snapshot_model_params,
    write_csv,
)

DEFAULT_CKPT = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "mappo_formal_50u_1200m_700m_seed42"
    / "checkpoints"
    / "episode_0050.pt"
)
OUT_DIR = PROJECT_ROOT / "results" / "experiments" / "eval_formal_50u_seed42"
SEEDS = [42, 43, 44]
MAX_STEPS = 1800  # 180 s at step_length=0.1


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


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _snapshot(dst: Path) -> None:
    snap = dst / "config_snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    for rel in (
        "configs/env.yaml",
        "configs/reward.yaml",
        "configs/model.yaml",
        "utils/baseline_eval.py",
    ):
        src = PROJECT_ROOT / rel
        if src.is_file():
            shutil.copy2(src, snap / src.name)
    if DEFAULT_CKPT.is_file():
        meta = {
            "checkpoint": str(DEFAULT_CKPT),
            "seeds": SEEDS,
            "max_steps": MAX_STEPS,
            "deterministic": True,
        }
        (snap / "eval_protocol.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )


def _pair_rows(rows: list[BaselineRow]) -> list[dict[str, Any]]:
    by: dict[tuple[int, str], BaselineRow] = {(r.seed, r.method): r for r in rows}
    out: list[dict[str, Any]] = []
    for seed in SEEDS:
        nc = by.get((seed, "NoControl"))
        mp = by.get((seed, "MAPPO"))
        if nc is None or mp is None:
            continue
        out.append(
            {
                "seed": seed,
                "nc_mean_speed": nc.mean_speed,
                "mappo_mean_speed": mp.mean_speed,
                "delta_mean_speed": mp.mean_speed - nc.mean_speed,
                "nc_throughput": nc.throughput,
                "mappo_throughput": mp.throughput,
                "delta_throughput": mp.throughput - nc.throughput,
                "nc_queue_length": nc.queue_length,
                "mappo_queue_length": mp.queue_length,
                "delta_queue_length": mp.queue_length - nc.queue_length,
                "nc_queue_length_max": nc.queue_length_max,
                "mappo_queue_length_max": mp.queue_length_max,
                "delta_queue_length_max": mp.queue_length_max - nc.queue_length_max,
                "nc_mean_speed_near_accident": nc.mean_speed_near_accident,
                "mappo_mean_speed_near_accident": mp.mean_speed_near_accident,
                "delta_near_speed": mp.mean_speed_near_accident - nc.mean_speed_near_accident,
                "nc_max_pos": nc.max_pos,
                "mappo_max_pos": mp.max_pos,
                "nc_near_accident_samples": nc.near_accident_samples,
                "mappo_near_accident_samples": mp.near_accident_samples,
                "nc_entered_accident_zone": nc.entered_accident_zone,
                "mappo_entered_accident_zone": mp.entered_accident_zone,
                "nc_collision": nc.collision,
                "mappo_collision": mp.collision,
                "mappo_KEEP_pct": 100.0 * mp.keep_ratio if mp.keep_ratio == mp.keep_ratio else float("nan"),
                "mappo_LEFT_pct": 100.0 * mp.left_ratio if mp.left_ratio == mp.left_ratio else float("nan"),
                "mappo_RIGHT_pct": 100.0 * mp.right_ratio if mp.right_ratio == mp.right_ratio else float("nan"),
                "nc_delay_proxy": nc.delay_proxy,
                "mappo_delay_proxy": mp.delay_proxy,
            }
        )
    return out


def _write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for d in rows:
        for k in d:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _assess(
    rows: list[BaselineRow],
    *,
    params_ok: bool,
    error: str | None,
) -> str:
    if error:
        return "FAIL"
    if not params_ok:
        return "FAIL"
    if len(rows) != len(SEEDS) * 2:
        return "FAIL"
    # Coverage: both methods should show accident-zone evidence on at least one seed
    any_enter = any(int(r.entered_accident_zone) == 1 for r in rows)
    any_near = any(int(r.near_accident_samples) > 0 for r in rows)
    any_pos = any(
        float(r.max_pos) >= 550.0 for r in rows if r.max_pos == r.max_pos
    )  # near 700-150
    if not (any_enter and any_near and any_pos):
        return "BORDERLINE"
    return "PASS"


def _write_report(
    *,
    out: Path,
    rows: list[BaselineRow],
    pairs: list[dict[str, Any]],
    result: str,
    ckpt_path: Path,
    device: str,
    gpu: str | None,
    params_ok: bool,
    error: str | None,
    started: str,
    finished: str,
) -> None:
    nc = [r for r in rows if r.method == "NoControl"]
    mp = [r for r in rows if r.method == "MAPPO"]

    def _agg(rs: list[BaselineRow], attr: str) -> float:
        return _mean([float(getattr(r, attr)) for r in rs])

    md = f"""# Phase 5 Formal Checkpoint Evaluation Report

## 1. Protocol

- Checkpoint: `{ckpt_path.as_posix()}`
- Seeds: {SEEDS} (paired)
- Methods: MAPPO | NoControl
- Simulation: {MAX_STEPS} steps (180 s), GUI=OFF
- MAPPO: inference only, **deterministic=True**, no optimizer update
- NoControl: all CAV actions = KEEP(0)
- Scene: 1200 m / accident@700 m lane0 / demand=3600 / cav_pen=0.2
- Device: {device}
- GPU: {gpu or 'n/a'}
- Started UTC: {started}
- Finished UTC: {finished}
- Error: {error or 'none'}
- Model params unchanged after eval: {params_ok}

## 2. Accident-zone coverage (priority)

| seed | method | max_pos | near_samples | entered | mean_ego_pos |
|------|--------|---------|--------------|---------|--------------|
"""
    for r in sorted(rows, key=lambda x: (x.seed, x.method)):
        md += (
            f"| {r.seed} | {r.method} | {r.max_pos:.1f} | "
            f"{r.near_accident_samples} | {r.entered_accident_zone} | "
            f"{r.mean_ego_position:.1f} |\n"
        )

    md += f"""
### Coverage summary

- Any entered accident zone: {any(int(r.entered_accident_zone)==1 for r in rows)}
- MAPPO mean max_pos: {_agg(mp, 'max_pos'):.1f} m
- NoControl mean max_pos: {_agg(nc, 'max_pos'):.1f} m
- MAPPO mean near_accident_samples: {_agg(mp, 'near_accident_samples'):.0f}
- NoControl mean near_accident_samples: {_agg(nc, 'near_accident_samples'):.0f}

## 3. Traffic metrics (mean over seeds)

| metric | NoControl | MAPPO | Δ (MAPPO − NC) |
|--------|-----------|-------|----------------|
| mean_speed | {_agg(nc,'mean_speed'):.3f} | {_agg(mp,'mean_speed'):.3f} | {_agg(mp,'mean_speed')-_agg(nc,'mean_speed'):+.3f} |
| near-accident speed | {_agg(nc,'mean_speed_near_accident'):.3f} | {_agg(mp,'mean_speed_near_accident'):.3f} | {_agg(mp,'mean_speed_near_accident')-_agg(nc,'mean_speed_near_accident'):+.3f} |
| throughput | {_agg(nc,'throughput'):.1f} | {_agg(mp,'throughput'):.1f} | {_agg(mp,'throughput')-_agg(nc,'throughput'):+.1f} |
| queue_length (mean) | {_agg(nc,'queue_length'):.2f} | {_agg(mp,'queue_length'):.2f} | {_agg(mp,'queue_length')-_agg(nc,'queue_length'):+.2f} |
| queue_length_max | {_agg(nc,'queue_length_max'):.2f} | {_agg(mp,'queue_length_max'):.2f} | {_agg(mp,'queue_length_max')-_agg(nc,'queue_length_max'):+.2f} |
| collision | {_agg(nc,'collision'):.1f} | {_agg(mp,'collision'):.1f} | {_agg(mp,'collision')-_agg(nc,'collision'):+.1f} |

## 4. Paired per-seed deltas

See `mappo_vs_no_control.csv`.

"""
    for p in pairs:
        md += (
            f"- seed {p['seed']}: Δspeed={p['delta_mean_speed']:+.3f}, "
            f"Δthr={p['delta_throughput']:+.1f}, "
            f"Δqueue={p['delta_queue_length']:+.2f}, "
            f"Δnear_v={p['delta_near_speed']:+.3f}, "
            f"entered NC/MAPPO={p['nc_entered_accident_zone']}/{p['mappo_entered_accident_zone']}\n"
        )

    md += f"""
## 5. MAPPO action distribution (mean over seeds)

- KEEP: {100.0*_agg(mp,'keep_ratio'):.1f}%
- LEFT: {100.0*_agg(mp,'left_ratio'):.1f}%
- RIGHT: {100.0*_agg(mp,'right_ratio'):.1f}%
- Near-zone KEEP/LEFT/RIGHT: {100.0*_agg(mp,'near_keep_ratio'):.1f}% / {100.0*_agg(mp,'near_left_ratio'):.1f}% / {100.0*_agg(mp,'near_right_ratio'):.1f}%

## 6. Secondary metrics

- delay_proxy (travel_time − trip_m/vmax), mean NC/MAPPO: {_agg(nc,'delay_proxy'):.2f} / {_agg(mp,'delay_proxy'):.2f}
  (secondary only; not used for PASS/FAIL)

## 7. Stability

- Fatal error: {error or 'none'}
- Parameter freeze: {'OK' if params_ok else 'FAIL'}

## 8. Final judgment

**{result}**

PASS/BORDERLINE/FAIL refers to evaluation pipeline fairness, coverage evidence, and completion — **not** whether MAPPO must beat No Control.
"""
    (out / "report.md").write_text(md, encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    log_f = (OUT_DIR / "stdout.log").open("w", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = Tee(old_out, log_f)
    sys.stderr = Tee(old_err, log_f)

    error: str | None = None
    rows: list[BaselineRow] = []
    params_ok = False
    ckpt_path = DEFAULT_CKPT
    device = resolve_device("auto")
    gpu = None
    try:
        print("========================================")
        print("Phase 5 Formal Checkpoint Evaluation")
        print("========================================")
        print(f"Out: {OUT_DIR}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"Seeds: {SEEDS}")
        print(f"max_steps={MAX_STEPS} (180s) GUI=OFF")
        print("MAPPO: deterministic=True, inference only")
        print("NoControl: CAV=KEEP(0)")
        print(f"Device: {device}")
        if device.type == "cuda" and torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            print(f"GPU: {gpu}")
        print()

        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        _snapshot(OUT_DIR)
        model, ckpt = load_mappo_checkpoint(ckpt_path, device=device)
        print(f"Loaded episode={ckpt.get('episode')} meta={ckpt.get('model_meta')}")
        before = snapshot_model_params(model)

        for seed in SEEDS:
            print(f"--- seed={seed} NoControl ---")
            nc = run_baseline_episode(
                method="NoControl",
                seed=seed,
                max_steps=MAX_STEPS,
                gui=False,
            )
            rows.append(nc)
            print(
                f"  speed={nc.mean_speed:.3f} thr={nc.throughput:.0f} "
                f"queue={nc.queue_length:.2f}/{nc.queue_length_max:.0f} "
                f"max_pos={nc.max_pos:.1f} near={nc.near_accident_samples} "
                f"entered={nc.entered_accident_zone} coll={nc.collision}"
            )

            print(f"--- seed={seed} MAPPO (deterministic) ---")
            mp = run_baseline_episode(
                method="MAPPO",
                seed=seed,
                model=model,
                device=device,
                max_steps=MAX_STEPS,
                gui=False,
                deterministic=True,
            )
            rows.append(mp)
            print(
                f"  speed={mp.mean_speed:.3f} thr={mp.throughput:.0f} "
                f"queue={mp.queue_length:.2f}/{mp.queue_length_max:.0f} "
                f"max_pos={mp.max_pos:.1f} near={mp.near_accident_samples} "
                f"entered={mp.entered_accident_zone} coll={mp.collision} "
                f"K/L/R={100*mp.keep_ratio:.1f}/{100*mp.left_ratio:.1f}/{100*mp.right_ratio:.1f}"
            )

        after = snapshot_model_params(model)
        params_ok = params_unchanged(before, after)
        print(f"\nParameter freeze check: {'OK' if params_ok else 'FAIL'}")

        write_csv(OUT_DIR / "evaluation_metrics.csv", rows, comparison_only=False)
        pairs = _pair_rows(rows)
        _write_dict_csv(OUT_DIR / "mappo_vs_no_control.csv", pairs)

        # Optional simple comparison plot if matplotlib available
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
            seeds = [p["seed"] for p in pairs]
            axes[0].bar([s - 0.15 for s in seeds], [p["nc_mean_speed"] for p in pairs], width=0.3, label="NoControl")
            axes[0].bar([s + 0.15 for s in seeds], [p["mappo_mean_speed"] for p in pairs], width=0.3, label="MAPPO")
            axes[0].set_title("Mean speed")
            axes[0].legend(fontsize=8)
            axes[1].bar([s - 0.15 for s in seeds], [p["nc_throughput"] for p in pairs], width=0.3, label="NoControl")
            axes[1].bar([s + 0.15 for s in seeds], [p["mappo_throughput"] for p in pairs], width=0.3, label="MAPPO")
            axes[1].set_title("Throughput")
            axes[2].bar([s - 0.15 for s in seeds], [p["nc_queue_length"] for p in pairs], width=0.3, label="NoControl")
            axes[2].bar([s + 0.15 for s in seeds], [p["mappo_queue_length"] for p in pairs], width=0.3, label="MAPPO")
            axes[2].set_title("Queue length")
            for ax in axes:
                ax.set_xticks(seeds)
            fig.tight_layout()
            fig.savefig(OUT_DIR / "comparison_bars.png", dpi=120)
            plt.close(fig)
            print(f"Saved plot: {OUT_DIR / 'comparison_bars.png'}")
        except Exception as plot_exc:  # noqa: BLE001
            print(f"Plot skipped: {plot_exc}")

    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        print(f"FATAL: {error}")
    finally:
        finished = datetime.now(timezone.utc).isoformat()
        result = _assess(rows, params_ok=params_ok, error=error)
        pairs = _pair_rows(rows)
        _write_report(
            out=OUT_DIR,
            rows=rows,
            pairs=pairs,
            result=result,
            ckpt_path=ckpt_path,
            device=str(device),
            gpu=gpu,
            params_ok=params_ok,
            error=error,
            started=started,
            finished=finished,
        )
        (OUT_DIR / "run_meta.json").write_text(
            json.dumps(
                {
                    "result": result,
                    "error": error,
                    "params_ok": params_ok,
                    "n_rows": len(rows),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print()
        print(f"Phase 5 Result: {result}")
        print(f"Report: {OUT_DIR / 'report.md'}")
        sys.stdout = old_out
        sys.stderr = old_err
        log_f.close()

    return 0 if result != "FAIL" and error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
