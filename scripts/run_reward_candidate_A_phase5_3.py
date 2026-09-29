"""Phase 5.3 — Candidate A Minimal Validation (train + 3-group eval).

Unique experimental variable: Reward Design (Candidate A vs Current).
Does NOT overwrite formal Current Reward checkpoints or Phase 5 eval.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.rollout import resolve_device  # noqa: E402
from scripts.train_mappo import train as train_mappo  # noqa: E402
from utils.baseline_eval import (  # noqa: E402
    BaselineRow,
    load_mappo_checkpoint,
    params_unchanged,
    run_baseline_episode,
    snapshot_model_params,
    write_csv,
)

OUT_DIR = PROJECT_ROOT / "results" / "experiments" / "reward_candidate_A_phase5_3"
CURRENT_CKPT = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "mappo_formal_50u_1200m_700m_seed42"
    / "checkpoints"
    / "episode_0050.pt"
)
CANDIDATE_A_REWARD = PROJECT_ROOT / "configs" / "reward_candidate_A.yaml"
SEED = 42
NUM_EPISODES = 50
ROLLOUT_STEPS = 512
MAX_EVAL_STEPS = 1800


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
        self.flush()
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def _snapshot(out: Path) -> None:
    snap = out / "config_snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    for rel in (
        "configs/env.yaml",
        "configs/reward.yaml",
        "configs/reward_candidate_A.yaml",
        "configs/model.yaml",
        "configs/train.yaml",
        "env/reward.py",
        "utils/baseline_eval.py",
    ):
        src = PROJECT_ROOT / rel
        if src.is_file():
            shutil.copy2(src, snap / src.name)
    meta = {
        "seed": SEED,
        "num_episodes": NUM_EPISODES,
        "rollout_steps": ROLLOUT_STEPS,
        "max_eval_steps": MAX_EVAL_STEPS,
        "current_checkpoint": str(CURRENT_CKPT),
        "candidate_a_reward": str(CANDIDATE_A_REWARD),
        "unique_variable": "reward_design",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (snap / "phase5_3_protocol.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def _row_label(method: str) -> str:
    return {
        "NoControl": "No Control",
        "MAPPO_Current": "Current Reward MAPPO",
        "MAPPO_CandidateA": "Candidate A MAPPO",
    }.get(method, method)


def _fmt(x: float, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:.{nd}f}"


def _delta(a: float, b: float) -> float:
    if any(isinstance(x, float) and (math.isnan(x) or math.isinf(x)) for x in (a, b)):
        return float("nan")
    return float(a) - float(b)


def _write_report(
    out: Path,
    *,
    train_summary: dict[str, Any],
    rows: list[BaselineRow],
    train_ok: bool,
    eval_ok: bool,
    current_params_frozen: bool,
) -> Path:
    by = {r.method: r for r in rows}
    nc = by.get("NoControl")
    cur = by.get("MAPPO_Current")
    ca = by.get("MAPPO_CandidateA")
    assert nc is not None and cur is not None and ca is not None

    # Merit for larger experiment: any clear traffic improvement vs Current
    # without catastrophic regression elsewhere. Descriptive only.
    better_speed = ca.mean_speed > cur.mean_speed
    better_near = ca.mean_speed_near_accident > cur.mean_speed_near_accident
    better_tp = ca.throughput > cur.throughput
    better_q = ca.queue_length < cur.queue_length
    improvements = sum([better_speed, better_near, better_tp, better_q])
    merits = improvements >= 2 and not (
        ca.mean_speed < nc.mean_speed - 2.0
        and ca.throughput < nc.throughput - 20
    )

    phase_pass = bool(train_ok and eval_ok and current_params_frozen)

    lines: list[str] = []
    lines.append("# Phase 5.3 — Candidate A Minimal Validation")
    lines.append("")
    lines.append(
        "PASS/FAIL below means the experiment completed correctly, "
        "**not** that Candidate A is better than Current Reward."
    )
    lines.append("")
    lines.append("## 1. Experiment Objective")
    lines.append("")
    lines.append(
        "Validate whether Candidate A (Phase 5.2) is worth a larger formal "
        "experiment. Candidate A is a research hypothesis — not assumed superior."
    )
    lines.append("")
    lines.append("## 2. Experimental Groups")
    lines.append("")
    lines.append("| Group | Policy | Reward | Train |")
    lines.append("|-------|--------|--------|-------|")
    lines.append("| 1 No Control | KEEP every step | n/a (traffic only) | No |")
    lines.append(
        f"| 2 Current Reward MAPPO | `{CURRENT_CKPT.name}` | "
        "`configs/reward.yaml` | Reuse formal 50u |"
    )
    lines.append(
        "| 3 Candidate A MAPPO | new `episode_0050.pt` under this run | "
        "`configs/reward_candidate_A.yaml` | 50 updates |"
    )
    lines.append("")
    lines.append("## 3. Changed Variable")
    lines.append("")
    lines.append("**Only Reward Design.** Fixed: MAPPO architecture, obs 17-D, "
                 "action KEEP/LEFT/RIGHT, SUMO scene (1200 m / accident@700 m), "
                 "demand, CAV penetration, lifecycle, seed=42, 50 updates × 512, "
                 "eval 1800 steps, GUI=OFF.")
    lines.append("")
    lines.append("Isolation: Candidate A checkpoints live only under "
                 "`results/experiments/reward_candidate_A_phase5_3/`. "
                 "Formal Current Reward `episode_0050.pt` was not overwritten.")
    lines.append("")
    lines.append("## 4. Candidate A Reward Definition")
    lines.append("")
    lines.append("From Phase 5.2 / `configs/reward_candidate_A.yaml`:")
    lines.append("")
    lines.append("```text")
    lines.append("v̄ = clip(v / 30, 0, 1)")
    lines.append("S_cav = mean neighbor-CAV speed ratios (else v̄)")
    lines.append("S_loc = mean local non-ACCIDENT speed ratios (else v̄)")
    lines.append("R_group = 0.6·S_cav + 0.4·S_loc")
    lines.append("P_block = 1 if |x−x_acc|≤150 and lane∈blocked else 0")
    lines.append("R = clip(0.55·v̄ + 0.35·R_group − 0.10·P_block, 0, 1)")
    lines.append("```")
    lines.append("")
    lines.append("No lane-change bonus. No extra components beyond Phase 5.2 A.")
    lines.append("")
    lines.append("## 5. Training Result (Candidate A)")
    lines.append("")
    lines.append(f"- Episodes / updates: {train_summary.get('num_episodes')}")
    resumed = int(train_summary.get("resumed_from_episode") or 0)
    if resumed > 0:
        lines.append(
            f"- Resumed from completed update {resumed} "
            "(actor/critic weights + Adam state). "
            "Python/NumPy/Torch RNG was re-seeded, so the traffic stream "
            "after resume is not bit-identical to an uninterrupted run. "
            "Learned parameters continue; this is not a fresh initialization."
        )
    lines.append(f"- Rollout steps: {train_summary.get('rollout_steps')}")
    lines.append(f"- Device: {train_summary.get('device')}")
    lines.append(f"- First mean reward: {_fmt(float(train_summary.get('first_reward', float('nan'))))}")
    lines.append(f"- Final mean reward: {_fmt(float(train_summary.get('final_reward', float('nan'))))}")
    lines.append(f"- Final actor loss: {_fmt(float(train_summary.get('actor_loss', float('nan'))))}")
    lines.append(f"- Final critic loss: {_fmt(float(train_summary.get('critic_loss', float('nan'))))}")
    lines.append(f"- Final entropy: {_fmt(float(train_summary.get('entropy', float('nan'))))}")
    lines.append(
        f"- Actor params changed: "
        f"{'YES' if train_summary.get('actor_changed') else 'NO'}"
    )
    lines.append(
        f"- Critic params changed: "
        f"{'YES' if train_summary.get('critic_changed') else 'NO'}"
    )
    lines.append(f"- NaN/Inf during train: {'NO' if train_ok else 'YES/ERROR'}")
    lines.append(f"- Checkpoints: `{out / 'checkpoints'}`")
    lines.append(f"- Metrics: `{out / 'train_metrics.csv'}`")
    last = (train_summary.get("log_rows") or [None])[-1]
    if isinstance(last, dict):
        lines.append("")
        lines.append("Last-update lifecycle / actions (train):")
        lines.append(
            f"- reset={last.get('env_reset_count')} sumo_start={last.get('sumo_start_count')} "
            f"departed={last.get('departed_cav_count')} rebind={last.get('rebind_events')} "
            f"active_peak={last.get('active_cav_peak')}"
        )
        lines.append(
            f"- KEEP/LEFT/RIGHT % = "
            f"{_fmt(float(last.get('KEEP_pct', float('nan'))),1)} / "
            f"{_fmt(float(last.get('LEFT_pct', float('nan'))),1)} / "
            f"{_fmt(float(last.get('RIGHT_pct', float('nan'))),1)}"
        )
        lines.append(
            f"- max_pos={_fmt(float(last.get('max_pos', float('nan'))),1)} "
            f"near_frac={_fmt(float(last.get('near_frac', float('nan'))))} "
            f"entered={bool(last.get('reached_accident_zone'))}"
        )
    lines.append("")
    lines.append("## 6. Evaluation Result")
    lines.append("")
    lines.append(f"Protocol: seed={SEED}, steps={MAX_EVAL_STEPS}, GUI=OFF, "
                 "MAPPO deterministic=True.")
    lines.append("")
    lines.append(
        "| Group | mean_speed | near_speed | throughput | queue_mean | "
        "queue_max | max_pos | near_samples | entered | collision |"
    )
    lines.append(
        "|-------|------------|------------|------------|------------|"
        "-----------|---------|--------------|---------|-----------|"
    )
    for key in ("NoControl", "MAPPO_Current", "MAPPO_CandidateA"):
        r = by[key]
        lines.append(
            f"| {_row_label(key)} | {_fmt(r.mean_speed)} | "
            f"{_fmt(r.mean_speed_near_accident)} | {_fmt(r.throughput, 0)} | "
            f"{_fmt(r.queue_length)} | {_fmt(r.queue_length_max, 0)} | "
            f"{_fmt(r.max_pos, 1)} | {r.near_accident_samples} | "
            f"{r.entered_accident_zone} | {r.collision} |"
        )
    lines.append("")
    lines.append(f"Current checkpoint params frozen during eval: "
                 f"{'YES' if current_params_frozen else 'NO'}")
    lines.append("")
    lines.append("## 7. Current Reward MAPPO vs Candidate A MAPPO")
    lines.append("")
    lines.append("| Metric | Current | Candidate A | Δ (A − Current) |")
    lines.append("|--------|---------|-------------|-----------------|")
    pairs = [
        ("mean_speed", cur.mean_speed, ca.mean_speed),
        ("near_zone_speed", cur.mean_speed_near_accident, ca.mean_speed_near_accident),
        ("throughput", cur.throughput, ca.throughput),
        ("queue_mean", cur.queue_length, ca.queue_length),
        ("queue_max", cur.queue_length_max, ca.queue_length_max),
        ("KEEP_ratio", cur.keep_ratio, ca.keep_ratio),
        ("LEFT_ratio", cur.left_ratio, ca.left_ratio),
        ("RIGHT_ratio", cur.right_ratio, ca.right_ratio),
        ("blocked_lane_near_frac", cur.blocked_lane_near_frac, ca.blocked_lane_near_frac),
    ]
    for name, a, b in pairs:
        d = _delta(b, a)
        lines.append(
            f"| {name} | {_fmt(a)} | {_fmt(b)} | {_fmt(d)} |"
        )
    lines.append("")
    lines.append("Directional reading (data only, not a claim of superiority):")
    lines.append(
        f"- mean speed: {'higher' if better_speed else 'lower/equal'} under Candidate A"
    )
    lines.append(
        f"- near-zone speed: {'higher' if better_near else 'lower/equal'} under Candidate A"
    )
    lines.append(
        f"- throughput: {'higher' if better_tp else 'lower/equal'} under Candidate A"
    )
    lines.append(
        f"- queue mean: {'lower' if better_q else 'higher/equal'} under Candidate A"
    )
    lines.append("")
    lines.append("## 8. No Control vs Both MAPPO Results")
    lines.append("")
    lines.append("| Metric | No Control | Current MAPPO | Candidate A MAPPO |")
    lines.append("|--------|------------|---------------|-------------------|")
    for name, attr in (
        ("mean_speed", "mean_speed"),
        ("near_zone_speed", "mean_speed_near_accident"),
        ("throughput", "throughput"),
        ("queue_mean", "queue_length"),
        ("queue_max", "queue_length_max"),
    ):
        lines.append(
            f"| {name} | {_fmt(getattr(nc, attr))} | "
            f"{_fmt(getattr(cur, attr))} | {_fmt(getattr(ca, attr))} |"
        )
    lines.append("")
    lines.append("## 9. Reward Component Analysis")
    lines.append("")
    lines.append(
        "Components recorded during evaluation under each group's reward config "
        "(No Control / Current use current reward fields; Candidate A uses A)."
    )
    lines.append("")
    lines.append(
        "| Group | R_ind | R_group | R_total | S_cav | S_loc | P_block | "
        "blocked_near_frac |"
    )
    lines.append(
        "|-------|-------|---------|---------|-------|-------|---------|-------------------|"
    )
    for key in ("NoControl", "MAPPO_Current", "MAPPO_CandidateA"):
        r = by[key]
        lines.append(
            f"| {_row_label(key)} | {_fmt(r.mean_individual_reward)} | "
            f"{_fmt(r.mean_group_reward)} | {_fmt(r.mean_total_reward)} | "
            f"{_fmt(r.mean_s_cav)} | {_fmt(r.mean_s_loc)} | {_fmt(r.mean_p_block)} | "
            f"{_fmt(r.blocked_lane_near_frac)} |"
        )
    lines.append("")
    lines.append("## 10. Action Distribution")
    lines.append("")
    lines.append("| Group | KEEP% | LEFT% | RIGHT% | near KEEP% | near LEFT% | near RIGHT% |")
    lines.append("|-------|-------|-------|--------|------------|------------|-------------|")
    for key in ("NoControl", "MAPPO_Current", "MAPPO_CandidateA"):
        r = by[key]
        lines.append(
            f"| {_row_label(key)} | {_fmt(100*r.keep_ratio,1)} | "
            f"{_fmt(100*r.left_ratio,1)} | {_fmt(100*r.right_ratio,1)} | "
            f"{_fmt(100*r.near_keep_ratio,1)} | {_fmt(100*r.near_left_ratio,1)} | "
            f"{_fmt(100*r.near_right_ratio,1)} |"
        )
    lines.append("")
    lines.append("## 11. Accident-zone Behavior")
    lines.append("")
    lines.append(
        f"- All groups entered accident zone: "
        f"NC={nc.entered_accident_zone}, Cur={cur.entered_accident_zone}, "
        f"A={ca.entered_accident_zone}"
    )
    lines.append(
        f"- Near-zone samples: NC={nc.near_accident_samples}, "
        f"Cur={cur.near_accident_samples}, A={ca.near_accident_samples}"
    )
    lines.append(
        f"- CAV blocked-lane fraction while near accident: "
        f"Cur={_fmt(cur.blocked_lane_near_frac)}, A={_fmt(ca.blocked_lane_near_frac)} "
        f"(Δ A−Cur={_fmt(_delta(ca.blocked_lane_near_frac, cur.blocked_lane_near_frac))})"
    )
    lines.append(
        f"- Near-zone speed: Cur={_fmt(cur.mean_speed_near_accident)}, "
        f"A={_fmt(ca.mean_speed_near_accident)}"
    )
    lines.append("")
    lines.append("## 12. Limitations")
    lines.append("")
    lines.append("- Single seed (42); Phase 5 showed SUMO seed may not diversify.")
    lines.append("- One 50-update train for Candidate A (same scale as Current, not longer).")
    lines.append("- Candidate A weights are Phase 5.2 starting guesses (not tuned).")
    lines.append("- Traffic metrics are episode-level; no multi-seed CI.")
    lines.append("- Reward magnitude is not comparable across Current vs A as a "
                 "performance score — use traffic metrics for comparison.")
    lines.append("")
    lines.append("## 13. Whether Candidate A Merits a Larger Experiment")
    lines.append("")
    if merits:
        lines.append(
            f"**Tentative YES for a larger / multi-seed experiment.** "
            f"Candidate A improved {improvements}/4 primary traffic metrics "
            "(mean speed, near-zone speed, throughput, queue mean) vs Current MAPPO "
            "in this single-seed run."
        )
    else:
        lines.append(
            f"**Tentative NO / hold for redesign or weight sensitivity.** "
            f"Candidate A improved only {improvements}/4 primary traffic metrics "
            "vs Current MAPPO in this single-seed run. Do not treat this as "
            "proof that Candidate A fails — only that a larger campaign is not "
            "yet clearly justified without further ablation."
        )
    lines.append("")
    lines.append("Do **not** interpret this section as 'Candidate A is better'.")
    lines.append("")
    lines.append("## Phase 5.3 Result: "
                 f"{'PASS' if phase_pass else 'FAIL'}")
    lines.append("")
    lines.append(
        "PASS = train completed without NaN, eval completed for all 3 groups, "
        "Current formal checkpoint untouched. PASS ≠ Candidate A superiority."
    )
    lines.append("")

    report = out / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def _episode_index(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return -1


def _latest_ckpt(ckpt_dir: Path) -> Path | None:
    pts = [p for p in ckpt_dir.glob("episode_*.pt") if _episode_index(p) >= 0]
    if not pts:
        return None
    return max(pts, key=_episode_index)


def main() -> int:
    out = OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "evaluation").mkdir(parents=True, exist_ok=True)
    _snapshot(out)

    if not CURRENT_CKPT.is_file():
        print(f"ERROR: missing Current Reward checkpoint: {CURRENT_CKPT}")
        return 1
    if not CANDIDATE_A_REWARD.is_file():
        print(f"ERROR: missing Candidate A reward config: {CANDIDATE_A_REWARD}")
        return 1

    ckpt_dir = out / "checkpoints"
    final_ckpt = ckpt_dir / "episode_0050.pt"
    resume_ckpt: Path | None = None
    if not final_ckpt.is_file():
        latest = _latest_ckpt(ckpt_dir)
        if latest is not None and _episode_index(latest) < NUM_EPISODES:
            resume_ckpt = latest

    log_path = out / "stdout.log"
    log_f = log_path.open("a" if resume_ckpt is not None else "w", encoding="utf-8")
    tee = Tee(sys.stdout, log_f)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = tee  # type: ignore[assignment]
    sys.stderr = tee  # type: ignore[assignment]

    train_ok = False
    eval_ok = False
    current_params_frozen = False
    train_summary: dict[str, Any] = {}
    rows: list[BaselineRow] = []

    try:
        print("================================")
        print("Phase 5.3 Candidate A Minimal Validation")
        print("================================")
        print(f"OUT: {out}")
        print(f"Current ckpt (read-only): {CURRENT_CKPT}")
        print(f"Candidate A reward: {CANDIDATE_A_REWARD}")
        print()

        # --- Train Candidate A (isolated). Resume if a partial checkpoint exists. ---
        if final_ckpt.is_file():
            print(f"=== TRAIN skipped; using existing {final_ckpt.name} ===")
            train_summary = {
                "num_episodes": NUM_EPISODES,
                "rollout_steps": ROLLOUT_STEPS,
                "device": "existing-checkpoint",
                "first_reward": float("nan"),
                "final_reward": float("nan"),
                "actor_loss": float("nan"),
                "critic_loss": float("nan"),
                "entropy": float("nan"),
                "actor_changed": True,
                "critic_changed": True,
                "checkpoints": [str(final_ckpt)],
                "log_rows": [],
                "resumed_from_episode": 0,
            }
        else:
            print("=== TRAIN Candidate A MAPPO (50 updates) ===")
            if resume_ckpt is not None:
                print(f"Resuming from {resume_ckpt} (not a fresh init)")
            train_summary = train_mappo(
                num_episodes=NUM_EPISODES,
                rollout_steps=ROLLOUT_STEPS,
                gui=False,
                checkpoint_dir=ckpt_dir,
                metrics_dir=out,
                run_title="Phase 5.3 Candidate A MAPPO Training",
                reward_config=CANDIDATE_A_REWARD,
                resume_from=resume_ckpt,
            )
        train_ok = bool(
            train_summary.get("actor_changed")
            and train_summary.get("critic_changed")
            and Path(train_summary["checkpoints"][-1]).is_file()
        )
        ca_ckpt = out / "checkpoints" / "episode_0050.pt"
        if not ca_ckpt.is_file():
            # fall back to last saved
            saved = train_summary.get("checkpoints") or []
            if not saved:
                raise RuntimeError("Candidate A training produced no checkpoints")
            ca_ckpt = Path(saved[-1])
        print(f"Candidate A checkpoint: {ca_ckpt}")
        print()

        # --- Evaluate 3 groups ---
        print("=== EVAL 3 groups (1800 steps, seed=42) ===")
        device = resolve_device("auto")

        # Group 1 No Control
        print("--- Group 1: No Control ---")
        rows.append(
            run_baseline_episode(
                method="NoControl",
                seed=SEED,
                gui=False,
                max_steps=MAX_EVAL_STEPS,
                reward_config=None,
            )
        )
        # Rename method for CSV clarity
        rows[-1].method = "NoControl"

        # Group 2 Current MAPPO
        print("--- Group 2: Current Reward MAPPO ---")
        model_cur, _ = load_mappo_checkpoint(CURRENT_CKPT, device=device)
        before = snapshot_model_params(model_cur)
        r_cur = run_baseline_episode(
            method="MAPPO",
            seed=SEED,
            model=model_cur,
            device=device,
            gui=False,
            max_steps=MAX_EVAL_STEPS,
            deterministic=True,
            reward_config=None,
        )
        after = snapshot_model_params(model_cur)
        current_params_frozen = params_unchanged(before, after)
        r_cur.method = "MAPPO_Current"
        rows.append(r_cur)
        print(f"Current params frozen: {current_params_frozen}")

        # Group 3 Candidate A MAPPO
        print("--- Group 3: Candidate A MAPPO ---")
        model_a, _ = load_mappo_checkpoint(ca_ckpt, device=device)
        before_a = snapshot_model_params(model_a)
        r_a = run_baseline_episode(
            method="MAPPO",
            seed=SEED,
            model=model_a,
            device=device,
            gui=False,
            max_steps=MAX_EVAL_STEPS,
            deterministic=True,
            reward_config=CANDIDATE_A_REWARD,
        )
        after_a = snapshot_model_params(model_a)
        if not params_unchanged(before_a, after_a):
            print("WARNING: Candidate A eval mutated params")
        r_a.method = "MAPPO_CandidateA"
        rows.append(r_a)

        eval_ok = len(rows) == 3 and all(r.steps > 0 for r in rows)

        write_csv(out / "evaluation" / "evaluation_metrics.csv", rows)
        # Compact comparison
        cmp_path = out / "evaluation" / "group_comparison.csv"
        with cmp_path.open("w", newline="", encoding="utf-8") as f:
            fields = [
                "method",
                "mean_speed",
                "mean_speed_near_accident",
                "throughput",
                "queue_length",
                "queue_length_max",
                "max_pos",
                "near_accident_samples",
                "entered_accident_zone",
                "collision",
                "keep_ratio",
                "left_ratio",
                "right_ratio",
                "mean_individual_reward",
                "mean_group_reward",
                "mean_total_reward",
                "mean_s_cav",
                "mean_s_loc",
                "mean_p_block",
                "blocked_lane_near_frac",
                "reward_variant",
            ]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                d = asdict(r)
                w.writerow({k: d.get(k) for k in fields})

        report = _write_report(
            out,
            train_summary=train_summary,
            rows=rows,
            train_ok=train_ok,
            eval_ok=eval_ok,
            current_params_frozen=current_params_frozen,
        )
        print()
        print(f"Report: {report}")
        phase_pass = train_ok and eval_ok and current_params_frozen
        print(f"Phase 5.3 Result: {'PASS' if phase_pass else 'FAIL'}")
        return 0 if phase_pass else 1
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        raise
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        log_f.close()


if __name__ == "__main__":
    raise SystemExit(main())
