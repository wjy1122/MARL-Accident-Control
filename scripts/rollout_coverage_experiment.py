"""Phase 4.4.2: Rollout coverage experiment.

Question only:
  How many training rollout steps are needed to stably cover the
  1200 m accident zone (|pos - 1200| <= 150 m)?

Does NOT train MAPPO. Does NOT modify Reward / Observation / SUMO.

Example:
  .\\.venv\\Scripts\\python.exe scripts\\rollout_coverage_experiment.py
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.action import KEEP  # noqa: E402
from env.highway_accident_env import HighwayAccidentEnv, load_config  # noqa: E402


@dataclass
class ProbeResult:
    seed: int
    warmup_steps: int
    first_near_step: int  # absolute steps after reset; -1 if never
    first_near_sim_s: float
    first_near_pos_m: float
    first_enter_1050_step: int  # pos >= accident-150
    first_pass_1200_step: int
    max_cav_pos_m: float
    notes: str = ""


@dataclass
class CoverageRow:
    seed: int
    warmup_steps: int
    rollout_steps: int
    total_sim_steps: int
    reached_near: int
    near_agent_steps: int
    valid_agent_steps: int
    near_frac: float
    mean_cav_pos_m: float
    max_cav_pos_m: float
    mean_speed_near: float
    n_cav_peak: int
    min_cav_pos_m: float = 0.0


def _cfg_nums(env: HighwayAccidentEnv) -> dict[str, float]:
    cfg = env.cfg
    return {
        "accident_pos": float(cfg.get("accident", {}).get("position_m", 1200.0)),
        "near_m": float(cfg.get("observation", {}).get("range_m", 150.0)),
        "step_length": float(cfg.get("sumo", {}).get("step_length", 0.1)),
    }


def _cav_positions(env: HighwayAccidentEnv) -> list[float]:
    return [float(st["position"]) for st in env.get_cav_states().values()]


def _any_near(positions: list[float], accident_pos: float, near_m: float) -> bool:
    return any(abs(p - accident_pos) <= near_m for p in positions)


def _warmup(env: HighwayAccidentEnv, n: int) -> None:
    for _ in range(n):
        if len(env.cav_ids) > 0:
            # Match RolloutCollector: stop early once a CAV exists.
            return
        env.step()


def probe_first_arrival(
    *,
    seed: int,
    warmup_steps: int,
    max_steps: int,
) -> ProbeResult:
    """After reset+warmup (KEEP), find first step a CAV enters the near zone."""
    env_cfg = copy.deepcopy(load_config())
    env_cfg.setdefault("episode", {})["max_episode_steps"] = max_steps + warmup_steps + 10
    env = HighwayAccidentEnv(config=env_cfg, gui=False)
    p = _cfg_nums(env)
    accident_pos = p["accident_pos"]
    near_m = p["near_m"]
    dt = p["step_length"]

    first_near = -1
    first_near_pos = float("nan")
    first_1050 = -1
    first_1200 = -1
    max_pos = 0.0

    try:
        env.reset()
        _warmup(env, warmup_steps)
        # Continue KEEP control like No Control / collector after warmup.
        for t in range(max_steps):
            abs_step = warmup_steps + t
            positions = _cav_positions(env)
            if positions:
                max_pos = max(max_pos, max(positions))
                if first_1050 < 0 and any(pos >= accident_pos - near_m for pos in positions):
                    first_1050 = abs_step
                if first_1200 < 0 and any(pos >= accident_pos for pos in positions):
                    first_1200 = abs_step
                if first_near < 0 and _any_near(positions, accident_pos, near_m):
                    first_near = abs_step
                    near_pos = [pos for pos in positions if abs(pos - accident_pos) <= near_m]
                    first_near_pos = float(min(near_pos, key=lambda x: abs(x - accident_pos)))

            actions = {cid: KEEP for cid in env.cav_ids}
            env.step(actions)

            if first_near >= 0 and first_1200 >= 0:
                break
    finally:
        env.close()

    note = ""
    if first_near < 0:
        note = f"never reached near-zone within {max_steps} post-warmup steps"

    return ProbeResult(
        seed=seed,
        warmup_steps=warmup_steps,
        first_near_step=first_near,
        first_near_sim_s=float(first_near * dt) if first_near >= 0 else float("nan"),
        first_near_pos_m=first_near_pos,
        first_enter_1050_step=first_1050,
        first_pass_1200_step=first_1200,
        max_cav_pos_m=max_pos,
        notes=note,
    )


def measure_rollout_coverage(
    *,
    seed: int,
    warmup_steps: int,
    rollout_steps: int,
) -> CoverageRow:
    """Mimic training collect: reset → warmup → rollout_steps with KEEP."""
    env_cfg = copy.deepcopy(load_config())
    env_cfg.setdefault("episode", {})[
        "max_episode_steps"
    ] = warmup_steps + rollout_steps + 10
    env = HighwayAccidentEnv(config=env_cfg, gui=False)
    p = _cfg_nums(env)
    accident_pos = p["accident_pos"]
    near_m = p["near_m"]

    near_agent_steps = 0
    valid_agent_steps = 0
    pos_sum = 0.0
    pos_count = 0
    max_pos = 0.0
    min_pos = float("inf")
    speed_near: list[float] = []
    n_cav_peak = 0
    reached = 0

    try:
        env.reset()
        _warmup(env, warmup_steps)
        for _ in range(rollout_steps):
            if len(env.cav_ids) == 0:
                env.reset()
                _warmup(env, warmup_steps)

            n_cav_peak = max(n_cav_peak, len(env.cav_ids))
            for st in env.get_cav_states().values():
                pos = float(st["position"])
                spd = float(st["speed"])
                valid_agent_steps += 1
                pos_sum += pos
                pos_count += 1
                max_pos = max(max_pos, pos)
                min_pos = min(min_pos, pos)
                if abs(pos - accident_pos) <= near_m:
                    near_agent_steps += 1
                    speed_near.append(spd)
                    reached = 1

            actions = {cid: KEEP for cid in env.cav_ids}
            _obs, _r, _term, truncated, _info = env.step(actions)
            del _obs, _r, _term, _info
            if truncated:
                env.reset()
                _warmup(env, warmup_steps)
    finally:
        env.close()

    near_frac = (
        float(near_agent_steps) / float(valid_agent_steps) if valid_agent_steps else 0.0
    )
    return CoverageRow(
        seed=seed,
        warmup_steps=warmup_steps,
        rollout_steps=rollout_steps,
        total_sim_steps=warmup_steps + rollout_steps,
        reached_near=reached,
        near_agent_steps=near_agent_steps,
        valid_agent_steps=valid_agent_steps,
        near_frac=near_frac,
        mean_cav_pos_m=float(pos_sum / pos_count) if pos_count else 0.0,
        max_cav_pos_m=max_pos,
        mean_speed_near=float(np.mean(speed_near)) if speed_near else float("nan"),
        n_cav_peak=n_cav_peak,
        min_cav_pos_m=float(min_pos if pos_count else 0.0),
    )


def recommend_steps(rows: list[CoverageRow], *, min_near_frac: float = 0.05) -> dict:
    """Smallest T such that ALL seeds reach near-zone and mean near_frac >= threshold."""
    by_t: dict[int, list[CoverageRow]] = {}
    for r in rows:
        by_t.setdefault(r.rollout_steps, []).append(r)

    recommendation = None
    summary = []
    for t in sorted(by_t):
        group = by_t[t]
        all_reached = all(r.reached_near == 1 for r in group)
        mean_frac = float(np.mean([r.near_frac for r in group]))
        min_frac = float(np.min([r.near_frac for r in group]))
        mean_max_pos = float(np.mean([r.max_cav_pos_m for r in group]))
        ok = all_reached and min_frac >= min_near_frac
        summary.append(
            {
                "rollout_steps": t,
                "all_reached": all_reached,
                "mean_near_frac": mean_frac,
                "min_near_frac": min_frac,
                "mean_max_pos_m": mean_max_pos,
                "stable": ok,
            }
        )
        if ok and recommendation is None:
            recommendation = t
    return {"recommendation": recommendation, "by_rollout_steps": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4.4.2 rollout coverage")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--probe-max-steps", type=int, default=1500)
    parser.add_argument(
        "--rollout-grid",
        type=int,
        nargs="+",
        default=[128, 256, 320, 384, 400, 450, 512, 640, 768, 1024],
    )
    parser.add_argument("--min-near-frac", type=float, default=0.05)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "results" / "coverage"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("================================")
    print("Phase 4.4.2 Rollout Coverage")
    print("================================")
    _acc = float(load_config().get("accident", {}).get("position_m", 1200.0))
    _near = float(load_config().get("observation", {}).get("range_m", 150.0))
    print(f"Accident: {_acc:g} m | near: |d| <= {_near:g} m")
    print(f"Warmup: {args.warmup_steps} (same early-exit as RolloutCollector)")
    print(f"Seeds: {args.seeds}")
    print(f"Policy: KEEP (coverage physics only; no MAPPO train)")
    print()

    # ---- A) First-arrival probe ----
    print("--- A) First arrival probe ---")
    probes: list[ProbeResult] = []
    for seed in args.seeds:
        pr = probe_first_arrival(
            seed=seed,
            warmup_steps=args.warmup_steps,
            max_steps=args.probe_max_steps,
        )
        probes.append(pr)
        print(
            f"seed={seed}: first_near_step={pr.first_near_step} "
            f"({pr.first_near_sim_s:.1f}s) pos={pr.first_near_pos_m:.1f}m "
            f"enter_1050={pr.first_enter_1050_step} pass_1200={pr.first_pass_1200_step} "
            f"max_pos={pr.max_cav_pos_m:.1f}m"
        )

    probe_csv = out_dir / "first_arrival_probe.csv"
    with probe_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(probes[0]).keys()))
        w.writeheader()
        for pr in probes:
            w.writerow(asdict(pr))

    near_hits = [pr.first_near_step for pr in probes if pr.first_near_step >= 0]
    if near_hits:
        # Steps AFTER reset (includes warmup in the absolute counter).
        # RolloutCollector counts only post-warmup steps in the buffer.
        # first_near_step is absolute from reset; post-warmup need =
        # max(0, first_near - warmup) but warmup may early-exit.
        post_warmup_need = []
        for pr in probes:
            if pr.first_near_step < 0:
                continue
            # Conservative: assume full warmup budget was used.
            post_warmup_need.append(max(0, pr.first_near_step - args.warmup_steps))
        print(
            f"\nProbe summary: first_near absolute steps "
            f"min={min(near_hits)} median={int(np.median(near_hits))} max={max(near_hits)}"
        )
        print(
            f"Post-warmup rollout steps to FIRST near hit "
            f"(assume warmup={args.warmup_steps}): "
            f"min={min(post_warmup_need)} median={int(np.median(post_warmup_need))} "
            f"max={max(post_warmup_need)}"
        )
    else:
        print("\nProbe: NEVER reached near zone in any seed.")
        post_warmup_need = []

    # ---- B) Coverage grid ----
    print("\n--- B) Coverage vs rollout_steps ---")
    rows: list[CoverageRow] = []
    for t in args.rollout_grid:
        for seed in args.seeds:
            row = measure_rollout_coverage(
                seed=seed,
                warmup_steps=args.warmup_steps,
                rollout_steps=t,
            )
            rows.append(row)
            print(
                f"T={t:4d} seed={seed}: reached={row.reached_near} "
                f"near_frac={row.near_frac:.3f} "
                f"near_steps={row.near_agent_steps}/{row.valid_agent_steps} "
                f"mean_pos={row.mean_cav_pos_m:.0f} max_pos={row.max_cav_pos_m:.0f}"
            )

    cov_csv = out_dir / "coverage_by_rollout_steps.csv"
    with cov_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    rec = recommend_steps(rows, min_near_frac=args.min_near_frac)
    print("\n--- Recommendation ---")
    print(f"Stability rule: all seeds reached_near=1 AND min near_frac>={args.min_near_frac}")
    for s in rec["by_rollout_steps"]:
        flag = "OK" if s["stable"] else "NO"
        print(
            f"  T={s['rollout_steps']:4d}: reached_all={s['all_reached']} "
            f"mean_frac={s['mean_near_frac']:.3f} min_frac={s['min_near_frac']:.3f} "
            f"mean_max_pos={s['mean_max_pos_m']:.0f}m  [{flag}]"
        )

    # Practical recommendation: first OK T, else max probed need + margin
    if rec["recommendation"] is not None:
        rec_t = int(rec["recommendation"])
        reason = f"smallest T with stable near coverage (frac>={args.min_near_frac})"
    elif post_warmup_need:
        rec_t = int(max(post_warmup_need) + 100)
        reason = "probe max first-hit + 100 margin (no grid cell met frac threshold)"
    else:
        rec_t = -1
        reason = "insufficient coverage in tested range"

    print()
    print("================================")
    print("ANSWER")
    print("================================")
    print(f"Recommended rollout_steps: {rec_t}")
    print(f"Reason: {reason}")
    if post_warmup_need:
        print(
            f"First near-zone hit (post-warmup, approx): "
            f"{int(np.median(post_warmup_need))} steps "
            f"(~{np.median(post_warmup_need) * 0.1:.1f}s)"
        )
    print(f"Current train.yaml rollout_steps=128 is INSUFFICIENT if probe/grid say so.")
    print(f"CSVs: {probe_csv}")
    print(f"      {cov_csv}")
    print("================================")

    # Write short answer file
    ans_path = out_dir / "recommendation.txt"
    ans_path.write_text(
        f"recommended_rollout_steps={rec_t}\n"
        f"min_near_frac={args.min_near_frac}\n"
        f"warmup_steps={args.warmup_steps}\n"
        f"reason={reason}\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
