"""Phase 6.5 — Candidate A2 + osc suppress (cooldown / mid-mask) to beat No Control.

Changes vs Phase 6.4:
  * Reward Candidate A2: higher individual, LC/rev penalties, progress bonus
  * Learning mask includes mid-LC + opposite-direction cooldown (2.0 s)
  * Env-side mid-LC suppress kept

Resume Phase 6.4 episode_0100 → train to episode_0150.
Does not overwrite Phase 5.3 / 6.1 / 6.4 checkpoints.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.rollout import load_train_config, resolve_device  # noqa: E402
from env.action import KEEP, LEFT, RIGHT, target_lane_index  # noqa: E402
from env.highway_accident_env import HighwayAccidentEnv, load_config  # noqa: E402
from env.reward import compute_candidate_a_total  # noqa: E402
from scripts.train_mappo import train as train_mappo  # noqa: E402
from utils.baseline_eval import (  # noqa: E402
    _is_near_accident,
    _params,
    _queue_length,
    load_mappo_checkpoint,
    set_eval_seed,
)

OUT = PROJECT_ROOT / "results" / "experiments" / "reward_candidate_A_phase6_5_osc_a2"
FIX_CKPT = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "reward_candidate_A_phase6_4_mask_fix"
    / "checkpoints"
    / "episode_0100.pt"
)
A2_REWARD = PROJECT_ROOT / "configs" / "reward_candidate_A2.yaml"
NUM_EPISODES = 150
ROLLOUT_STEPS = 512
MAX_EVAL_STEPS = 1800
# Target: clearly beat No Control on these (Phase 6.4 NC numbers).
NC_REF = {"mean_speed": 27.17, "near_speed": 26.52, "throughput": 136}


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(v) or math.isinf(v):
        return "n/a"
    return f"{v:.{nd}f}"


def _latest_ckpt(folder: Path) -> Path | None:
    files = sorted(folder.glob("episode_*.pt"))
    return files[-1] if files else None


def _osc_env_cfg() -> dict[str, Any]:
    cfg = copy.deepcopy(load_config())
    action = cfg.setdefault("action", {})
    action["suppress_mid_lane_change"] = True
    action["mid_lane_change_lateral_m"] = 0.2
    action["mask_mid_lane_change"] = True
    action["lc_cooldown_s"] = 2.0
    return cfg


def _fix_only_env_cfg() -> dict[str, Any]:
    """Phase 6.4 env: mid-LC env suppress only (no cooldown / mask-mid)."""
    cfg = copy.deepcopy(load_config())
    action = cfg.setdefault("action", {})
    action["suppress_mid_lane_change"] = True
    action["mid_lane_change_lateral_m"] = 0.2
    action["mask_mid_lane_change"] = False
    action["lc_cooldown_s"] = 0.0
    return cfg


def _run_unit_tests() -> dict[str, bool]:
    out: dict[str, bool] = {}
    for name, script in (
        ("action_mask", PROJECT_ROOT / "tests" / "test_action_mask.py"),
        ("gae", PROJECT_ROOT / "tests" / "test_gae.py"),
    ):
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(PROJECT_ROOT),
            check=False,
        )
        out[name] = proc.returncode == 0
    # Formula sanity for A2 extras.
    base = compute_candidate_a_total(
        individual_reward=0.8,
        group_reward=0.7,
        p_block=0.0,
        individual_weight=0.70,
        group_weight=0.18,
        block_weight=0.12,
    )
    with_lc = compute_candidate_a_total(
        individual_reward=0.8,
        group_reward=0.7,
        p_block=0.0,
        individual_weight=0.70,
        group_weight=0.18,
        block_weight=0.12,
        lc_event=1.0,
        lc_weight=0.08,
    )
    out["a2_formula"] = bool(abs(base - with_lc - 0.08) < 1e-9 and 0.0 <= with_lc <= 1.0)
    return out


def run_eval(
    model: torch.nn.Module,
    device: torch.device,
    *,
    use_mask: bool,
    env_cfg: dict[str, Any],
    deterministic: bool,
    seed: int,
    reward_config: Path,
) -> dict[str, Any]:
    set_eval_seed(seed)
    model.eval()
    cfg = copy.deepcopy(env_cfg)
    cfg.setdefault("episode", {})["max_episode_steps"] = MAX_EVAL_STEPS
    env = HighwayAccidentEnv(config=cfg, gui=False, reward_config=reward_config)
    p = _params(env)
    blocked = {int(x) for x in env.accident_lanes}
    prefix = str(p["accident_prefix"])
    n_lanes = int(env.n_lanes)
    left_up = bool(env.left_increases_index)

    speeds: list[float] = []
    speeds_near: list[float] = []
    queues: list[float] = []
    near_cav = 0
    near_cav_blocked = 0
    arrived = 0
    prev_ids: set[str] = set()
    lane_hist: dict[str, list[tuple[int, int]]] = {}
    act_n = np.zeros(3, dtype=np.int64)
    invalid = 0
    mid_suppressed = 0

    try:
        env.reset()
        traci = env._require_traci()
        for step in range(MAX_EVAL_STEPS):
            ids = set(traci.vehicle.getIDList())
            for vid in prev_ids - ids:
                if not str(vid).startswith(prefix):
                    arrived += 1
            prev_ids = ids

            vehicles = env.get_vehicle_states()
            for st in vehicles.values():
                if st.get("type") == "ACCIDENT":
                    continue
                spd = float(st["speed"])
                pos = float(st["position"])
                speeds.append(spd)
                if _is_near_accident(pos, p):
                    speeds_near.append(spd)
                if st.get("type") == "CAV" and _is_near_accident(pos, p):
                    near_cav += 1
                    if int(st.get("lane", -1)) in blocked:
                        near_cav_blocked += 1
            queues.append(float(_queue_length(env, p)))

            states = {
                vid: st for vid, st in vehicles.items() if st.get("type") == "CAV"
            }
            observations = env.get_observations()
            actions: dict[str, int] = {}
            for cid, st in states.items():
                if cid not in observations:
                    continue
                lane = int(st["lane"])
                obs_t = torch.as_tensor(
                    np.asarray(observations[cid], dtype=np.float32), device=device
                )
                am_t = None
                if use_mask:
                    am_np = env.get_action_mask(cid)
                    am_t = torch.as_tensor(
                        am_np, dtype=torch.float32, device=device
                    )
                with torch.no_grad():
                    action_t, _lp = model.act(
                        obs_t, deterministic=deterministic, action_mask=am_t
                    )
                action = int(action_t.item())
                actions[cid] = action
                act_n[action] += 1
                if (
                    action != KEEP
                    and target_lane_index(
                        current_lane=lane,
                        action=action,
                        n_lanes=n_lanes,
                        left_increases_index=left_up,
                    )
                    is None
                ):
                    invalid += 1
                lane_hist.setdefault(cid, []).append((step, lane))

            _o, _r, _t, trunc, info = env.step(actions)
            details = info.get("action_details") or {}
            for _cid, det in details.items():
                if isinstance(det, dict) and det.get("suppressed_mid_lane_change"):
                    mid_suppressed += 1
            if trunc:
                break
    finally:
        env.close()

    eff_left = 0
    eff_right = 0
    rev = 0
    for events in lane_hist.values():
        dirs: list[str] = []
        for i in range(1, len(events)):
            if events[i][1] == events[i - 1][1]:
                continue
            d = "LEFT" if events[i][1] > events[i - 1][1] else "RIGHT"
            dirs.append(d)
            if d == "LEFT":
                eff_left += 1
            else:
                eff_right += 1
        for i in range(1, len(dirs)):
            if dirs[i] != dirs[i - 1]:
                rev += 1

    n_act = int(act_n.sum())
    return {
        "mean_speed": float(np.mean(speeds)) if speeds else None,
        "near_speed": float(np.mean(speeds_near)) if speeds_near else None,
        "throughput": int(arrived),
        "queue": float(np.mean(queues)) if queues else None,
        "blocked_lane_frac": (
            float(near_cav_blocked) / float(near_cav) if near_cav else None
        ),
        "KEEP": float(act_n[KEEP] / n_act) if n_act else None,
        "LEFT": float(act_n[LEFT] / n_act) if n_act else None,
        "RIGHT": float(act_n[RIGHT] / n_act) if n_act else None,
        "invalid_rate": float(invalid) / float(n_act) if n_act else None,
        "effective_LEFT": eff_left,
        "effective_RIGHT": eff_right,
        "effective_reversals": rev,
        "mid_suppressed": mid_suppressed,
        "n_actions": n_act,
        "use_mask": use_mask,
        "deterministic": deterministic,
    }


def run_eval_no_control(*, seed: int) -> dict[str, Any]:
    set_eval_seed(seed)
    env_cfg = copy.deepcopy(load_config())
    env_cfg.setdefault("episode", {})["max_episode_steps"] = MAX_EVAL_STEPS
    env = HighwayAccidentEnv(config=env_cfg, gui=False, reward_config=A2_REWARD)
    p = _params(env)
    blocked = {int(x) for x in env.accident_lanes}
    prefix = str(p["accident_prefix"])

    speeds: list[float] = []
    speeds_near: list[float] = []
    queues: list[float] = []
    near_cav = 0
    near_cav_blocked = 0
    arrived = 0
    prev_ids: set[str] = set()
    lane_hist: dict[str, list[tuple[int, int]]] = {}
    act_n = np.zeros(3, dtype=np.int64)

    try:
        env.reset()
        traci = env._require_traci()
        for step in range(MAX_EVAL_STEPS):
            ids = set(traci.vehicle.getIDList())
            for vid in prev_ids - ids:
                if not str(vid).startswith(prefix):
                    arrived += 1
            prev_ids = ids

            vehicles = env.get_vehicle_states()
            for st in vehicles.values():
                if st.get("type") == "ACCIDENT":
                    continue
                spd = float(st["speed"])
                pos = float(st["position"])
                speeds.append(spd)
                if _is_near_accident(pos, p):
                    speeds_near.append(spd)
                if st.get("type") == "CAV" and _is_near_accident(pos, p):
                    near_cav += 1
                    if int(st.get("lane", -1)) in blocked:
                        near_cav_blocked += 1
            queues.append(float(_queue_length(env, p)))

            states = {
                vid: st for vid, st in vehicles.items() if st.get("type") == "CAV"
            }
            observations = env.get_observations()
            actions: dict[str, int] = {}
            for cid, st in states.items():
                if cid not in observations:
                    continue
                actions[cid] = KEEP
                act_n[KEEP] += 1
                lane_hist.setdefault(cid, []).append((step, int(st["lane"])))

            _o, _r, _t, trunc, _info = env.step(actions)
            if trunc:
                break
    finally:
        env.close()

    eff_left = 0
    eff_right = 0
    rev = 0
    for events in lane_hist.values():
        dirs: list[str] = []
        for i in range(1, len(events)):
            if events[i][1] == events[i - 1][1]:
                continue
            d = "LEFT" if events[i][1] > events[i - 1][1] else "RIGHT"
            dirs.append(d)
            if d == "LEFT":
                eff_left += 1
            else:
                eff_right += 1
        for i in range(1, len(dirs)):
            if dirs[i] != dirs[i - 1]:
                rev += 1

    n_act = int(act_n.sum())
    return {
        "mean_speed": float(np.mean(speeds)) if speeds else None,
        "near_speed": float(np.mean(speeds_near)) if speeds_near else None,
        "throughput": int(arrived),
        "queue": float(np.mean(queues)) if queues else None,
        "blocked_lane_frac": (
            float(near_cav_blocked) / float(near_cav) if near_cav else None
        ),
        "KEEP": float(act_n[KEEP] / n_act) if n_act else None,
        "LEFT": 0.0,
        "RIGHT": 0.0,
        "invalid_rate": 0.0,
        "effective_LEFT": eff_left,
        "effective_RIGHT": eff_right,
        "effective_reversals": rev,
        "mid_suppressed": 0,
        "n_actions": n_act,
        "policy": "no_control_KEEP",
    }


def _beats_nc(e: dict[str, Any]) -> dict[str, bool]:
    return {
        "mean_speed": float(e.get("mean_speed") or 0) > NC_REF["mean_speed"],
        "near_speed": float(e.get("near_speed") or 0) > NC_REF["near_speed"],
        "throughput": int(e.get("throughput") or 0) > NC_REF["throughput"],
    }


def _write_report(
    *,
    status: str,
    unit_ok: dict[str, bool],
    train_summary: dict[str, Any] | None,
    evals: dict[str, dict[str, Any]],
    ckpt_same_fix: bool,
    final_ckpt: Path | None,
) -> None:
    lines: list[str] = []
    a = lines.append
    a("# Phase 6.5 Report")
    a("")
    a("## Candidate A2 + osc suppress (cooldown / mid-mask)")
    a("")
    a("### 1. Status")
    a("")
    a(f"`{status}`")
    a("")
    a("- Reward: **Candidate A2** (`configs/reward_candidate_A2.yaml`).")
    a("- Osc suppress: mid-LC in learning mask + 2.0 s opposite-direction cooldown.")
    a(
        f"- Phase 6.4 checkpoint overwritten: "
        f"**{'YES' if not ckpt_same_fix else 'No'}**."
    )
    a(
        f"- Unit tests: action_mask={'PASS' if unit_ok.get('action_mask') else 'FAIL'}, "
        f"gae={'PASS' if unit_ok.get('gae') else 'FAIL'}, "
        f"a2_formula={'PASS' if unit_ok.get('a2_formula') else 'FAIL'}."
    )
    a("")
    a("### 2. Checkpoints")
    a("")
    a(f"- A+Mask+Fix (resume): `{FIX_CKPT.relative_to(PROJECT_ROOT).as_posix()}`")
    if final_ckpt is not None:
        a(f"- A2+Osc: `{final_ckpt.relative_to(PROJECT_ROOT).as_posix()}`")
    a("")
    a("### 3. Deterministic evaluation (seed 42, 1800 steps)")
    a("")
    a(
        "| Run | mean v | near v | thruput | queue | blocked | "
        "KEEP/L/R | invalid | eff L/R | rev | mid_sup |"
    )
    a("| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |")
    order = ("NC_det", "Fix_det", "A2_det")
    labels = {
        "NC_det": "No Control",
        "Fix_det": "A+Mask+Fix",
        "A2_det": "A2+Osc",
    }
    for key in order:
        e = evals.get(key)
        if not e:
            continue
        a(
            f"| {labels[key]} | {_fmt(e.get('mean_speed'))} | {_fmt(e.get('near_speed'))} | "
            f"{e.get('throughput')} | {_fmt(e.get('queue'))} | "
            f"{_fmt(e.get('blocked_lane_frac'), 3)} | "
            f"{_fmt(e.get('KEEP'), 3)}/{_fmt(e.get('LEFT'), 3)}/{_fmt(e.get('RIGHT'), 3)} | "
            f"{_fmt(e.get('invalid_rate'), 4)} | "
            f"{e.get('effective_LEFT')}/{e.get('effective_RIGHT')} | "
            f"{e.get('effective_reversals')} | {e.get('mid_suppressed')} |"
        )
    a("")
    a2 = evals.get("A2_det") or {}
    beats = _beats_nc(a2) if a2 else {}
    a("### 4. vs No Control")
    a("")
    if beats:
        a(
            f"- mean_speed > NC: **{beats['mean_speed']}** "
            f"({_fmt(a2.get('mean_speed'))} vs {NC_REF['mean_speed']})"
        )
        a(
            f"- near_speed > NC: **{beats['near_speed']}** "
            f"({_fmt(a2.get('near_speed'))} vs {NC_REF['near_speed']})"
        )
        a(
            f"- throughput > NC: **{beats['throughput']}** "
            f"({a2.get('throughput')} vs {NC_REF['throughput']})"
        )
        a(f"- effective_reversals: {a2.get('effective_reversals')} (Fix was 2664)")
    else:
        a("- No A2 eval.")
    a("")
    a("### 5. Final conclusion")
    a("")
    if beats and all(beats.values()):
        a("A2+Osc beats No Control on mean/near speed and throughput.")
    elif beats and any(beats.values()):
        a("A2+Osc partially beats No Control; continue tuning if full win required.")
    else:
        a("A2+Osc does not yet beat No Control on the primary metrics.")
    a("")
    a(f"Phase 6.5 Result: {status}")
    a("")
    a(f"_Generated {datetime.now(timezone.utc).isoformat()}_")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(evals: dict[str, dict[str, Any]]) -> None:
    tab = OUT / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    with (tab / "ablation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run",
                "mean_speed",
                "near_speed",
                "throughput",
                "queue",
                "blocked",
                "KEEP",
                "LEFT",
                "RIGHT",
                "invalid",
                "eff_LEFT",
                "eff_RIGHT",
                "reversals",
                "mid_suppressed",
            ]
        )
        for key, label in (
            ("NC_det", "No Control"),
            ("Fix_det", "A+Mask+Fix"),
            ("A2_det", "A2+Osc"),
        ):
            e = evals.get(key)
            if not e:
                continue
            w.writerow(
                [
                    label,
                    e.get("mean_speed"),
                    e.get("near_speed"),
                    e.get("throughput"),
                    e.get("queue"),
                    e.get("blocked_lane_frac"),
                    e.get("KEEP"),
                    e.get("LEFT"),
                    e.get("RIGHT"),
                    e.get("invalid_rate"),
                    e.get("effective_LEFT"),
                    e.get("effective_RIGHT"),
                    e.get("effective_reversals"),
                    e.get("mid_suppressed"),
                ]
            )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)

    if not FIX_CKPT.is_file():
        raise FileNotFoundError(FIX_CKPT)
    if not A2_REWARD.is_file():
        raise FileNotFoundError(A2_REWARD)
    ckpt_before = (FIX_CKPT.stat().st_mtime_ns, FIX_CKPT.stat().st_size)

    print("Phase 6.5 unit tests", flush=True)
    unit_ok = _run_unit_tests()
    if not all(unit_ok.values()):
        _write_report(
            status="FAIL",
            unit_ok=unit_ok,
            train_summary=None,
            evals={},
            ckpt_same_fix=True,
            final_ckpt=None,
        )
        print("Phase 6.5 Result: FAIL", flush=True)
        return 1

    final_ckpt = OUT / "checkpoints" / f"episode_{NUM_EPISODES:04d}.pt"
    train_summary: dict[str, Any] | None = None
    if final_ckpt.is_file():
        print(f"Checkpoint exists, skip training: {final_ckpt}", flush=True)
    else:
        resume = _latest_ckpt(OUT / "checkpoints") or FIX_CKPT
        print(f"Training from {resume} → episode {NUM_EPISODES}", flush=True)
        train_summary = train_mappo(
            num_episodes=NUM_EPISODES,
            rollout_steps=ROLLOUT_STEPS,
            gui=False,
            checkpoint_dir=OUT / "checkpoints",
            metrics_dir=OUT / "logs",
            run_title="Phase 6.5 Candidate A2 + Osc Suppress",
            env_cfg=_osc_env_cfg(),
            reward_config=A2_REWARD,
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

    ckpt_same_fix = (
        FIX_CKPT.stat().st_mtime_ns,
        FIX_CKPT.stat().st_size,
    ) == ckpt_before
    if not final_ckpt.is_file():
        _write_report(
            status="FAIL",
            unit_ok=unit_ok,
            train_summary=train_summary,
            evals={},
            ckpt_same_fix=ckpt_same_fix,
            final_ckpt=None,
        )
        print("Phase 6.5 Result: FAIL", flush=True)
        return 1

    device = resolve_device("auto")
    seed = int(load_train_config().get("seed", 42))
    ca_reward = PROJECT_ROOT / "configs" / "reward_candidate_A.yaml"
    print(f"Eval device={device}", flush=True)
    evals: dict[str, dict[str, Any]] = {}

    print("--- eval NC_det ---", flush=True)
    evals["NC_det"] = run_eval_no_control(seed=seed)
    print(
        f"  v={evals['NC_det']['mean_speed']} near={evals['NC_det']['near_speed']} "
        f"thruput={evals['NC_det']['throughput']}",
        flush=True,
    )

    print("--- eval Fix_det (6.4 ckpt, A reward, mid suppress only) ---", flush=True)
    model_fix, _ = load_mappo_checkpoint(FIX_CKPT, device=device)
    evals["Fix_det"] = run_eval(
        model_fix,
        device,
        use_mask=True,
        env_cfg=_fix_only_env_cfg(),
        deterministic=True,
        seed=seed,
        reward_config=ca_reward,
    )
    print(
        f"  v={evals['Fix_det']['mean_speed']} near={evals['Fix_det']['near_speed']} "
        f"rev={evals['Fix_det']['effective_reversals']}",
        flush=True,
    )

    print("--- eval A2_det ---", flush=True)
    model_a2, _ = load_mappo_checkpoint(final_ckpt, device=device)
    evals["A2_det"] = run_eval(
        model_a2,
        device,
        use_mask=True,
        env_cfg=_osc_env_cfg(),
        deterministic=True,
        seed=seed,
        reward_config=A2_REWARD,
    )
    print(
        f"  v={evals['A2_det']['mean_speed']} near={evals['A2_det']['near_speed']} "
        f"thruput={evals['A2_det']['throughput']} "
        f"rev={evals['A2_det']['effective_reversals']}",
        flush=True,
    )

    (OUT / "logs" / "eval_summary.json").write_text(
        json.dumps(evals, indent=2), encoding="utf-8"
    )
    _write_csv(evals)

    beats = _beats_nc(evals["A2_det"])
    status = "PASS" if ckpt_same_fix and all(beats.values()) else "PARTIAL"
    if not ckpt_same_fix or evals["A2_det"].get("invalid_rate", 1) not in (0, 0.0):
        status = "FAIL" if not ckpt_same_fix else "PARTIAL"

    _write_report(
        status=status,
        unit_ok=unit_ok,
        train_summary=train_summary,
        evals=evals,
        ckpt_same_fix=ckpt_same_fix,
        final_ckpt=final_ckpt,
    )
    print(f"Report: {OUT / 'report.md'}", flush=True)
    print(f"Phase 6.5 Result: {status}", flush=True)
    print(f"Beat NC: {beats}", flush=True)
    return 0 if status != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
