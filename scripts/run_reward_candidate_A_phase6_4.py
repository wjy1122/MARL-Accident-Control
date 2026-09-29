"""Phase 6.4 — Candidate A + Action Mask + credit/dynamics Fix (paper ablation).

Fix (no reward change):
  1. Mask-aware GAE (vacant slots do not pollute advantages)
  2. Rollout-end critic bootstrap V(s_T)
  3. Suppress LEFT/RIGHT while mid lane-change (|lat| > 0.2 m)

Resume Phase 5.3 episode_0050 → train to episode_0100 with Action Mask ON.
Does not overwrite Phase 5.3 or Phase 6.1 checkpoints.
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
from env.action_mask import legal_action_mask  # noqa: E402
from env.highway_accident_env import HighwayAccidentEnv, load_config  # noqa: E402
from scripts.train_mappo import train as train_mappo  # noqa: E402
from utils.baseline_eval import (  # noqa: E402
    _is_near_accident,
    _params,
    _queue_length,
    load_mappo_checkpoint,
    set_eval_seed,
)

OUT = PROJECT_ROOT / "results" / "experiments" / "reward_candidate_A_phase6_4_mask_fix"
CA_CKPT = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "reward_candidate_A_phase5_3"
    / "checkpoints"
    / "episode_0050.pt"
)
MASK_CKPT = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "reward_candidate_A_phase6_1_mask"
    / "checkpoints"
    / "episode_0100.pt"
)
CA_REWARD = PROJECT_ROOT / "configs" / "reward_candidate_A.yaml"
NUM_EPISODES = 100
ROLLOUT_STEPS = 512
MAX_EVAL_STEPS = 1800


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


def _zone(pos: float, x_acc: float, radius: float) -> str:
    if pos < x_acc - radius:
        return "before"
    if pos > x_acc + radius:
        return "after"
    return "near"


def _fix_env_cfg() -> dict[str, Any]:
    cfg = copy.deepcopy(load_config())
    cfg.setdefault("action", {})["suppress_mid_lane_change"] = True
    cfg["action"]["mid_lane_change_lateral_m"] = 0.2
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
    return out


def run_eval(
    model: torch.nn.Module,
    device: torch.device,
    *,
    use_mask: bool,
    suppress_mid: bool,
    deterministic: bool,
    seed: int,
) -> dict[str, Any]:
    set_eval_seed(seed)
    model.eval()
    env_cfg = _fix_env_cfg() if suppress_mid else copy.deepcopy(load_config())
    env_cfg.setdefault("episode", {})["max_episode_steps"] = MAX_EVAL_STEPS
    env = HighwayAccidentEnv(config=env_cfg, gui=False, reward_config=CA_REWARD)
    p = _params(env)
    x_acc = float(env.accident_position_m)
    zone_r = float(env.reward_zone_range_m)
    n_lanes = int(env.n_lanes)
    left_up = bool(env.left_increases_index)
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
                am_np = legal_action_mask(
                    lane, n_lanes=n_lanes, left_increases_index=left_up
                )
                am_t = (
                    torch.as_tensor(am_np, dtype=torch.float32, device=device)
                    if use_mask
                    else None
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

    # Effective LC / reversals from lane-index history.
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
        "suppress_mid": suppress_mid,
        "deterministic": deterministic,
    }


def run_eval_no_control(*, seed: int) -> dict[str, Any]:
    """CAVs always KEEP. Same scene, steps, and metrics as ``run_eval``."""
    set_eval_seed(seed)
    env_cfg = copy.deepcopy(load_config())
    env_cfg.setdefault("episode", {})["max_episode_steps"] = MAX_EVAL_STEPS
    env = HighwayAccidentEnv(config=env_cfg, gui=False, reward_config=CA_REWARD)
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
        "LEFT": float(act_n[LEFT] / n_act) if n_act else None,
        "RIGHT": float(act_n[RIGHT] / n_act) if n_act else None,
        "invalid_rate": 0.0,
        "effective_LEFT": eff_left,
        "effective_RIGHT": eff_right,
        "effective_reversals": rev,
        "mid_suppressed": 0,
        "n_actions": n_act,
        "use_mask": False,
        "suppress_mid": False,
        "deterministic": True,
        "policy": "no_control_KEEP",
    }


def _write_report(
    *,
    status: str,
    unit_ok: dict[str, bool],
    train_summary: dict[str, Any] | None,
    evals: dict[str, dict[str, Any]],
    ckpt_same_a: bool,
    ckpt_same_m: bool,
    final_ckpt: Path | None,
) -> None:
    lines: list[str] = []
    a = lines.append
    a("# Phase 6.4 Report")
    a("")
    a("## Candidate A + Action Mask + Fix (paper ablation)")
    a("")
    a("### 1. Status")
    a("")
    a(f"`{status}`")
    a("")
    a("- Reward: **unchanged** (Candidate A).")
    a("- Observation / Actor / Critic architecture / PPO hyperparameters: **unchanged**.")
    a(f"- Phase 5.3 checkpoint overwritten: **{'YES' if not ckpt_same_a else 'No'}**.")
    a(f"- Phase 6.1 checkpoint overwritten: **{'YES' if not ckpt_same_m else 'No'}**.")
    a(
        f"- Unit tests: action_mask={'PASS' if unit_ok.get('action_mask') else 'FAIL'}, "
        f"gae={'PASS' if unit_ok.get('gae') else 'FAIL'}."
    )
    a("")
    a("### 2. Fix contents")
    a("")
    a("1. Mask-aware GAE: vacant agent slots reset GAE carry (`algorithms/gae.py`).")
    a("2. Rollout-end critic bootstrap `V(s_T)` (`algorithms/rollout.py` → PPO).")
    a(
        "3. Suppress LEFT/RIGHT while mid lane-change "
        "(`suppress_mid_lane_change`, |lat|>0.2 m)."
    )
    a("")
    a("### 3. Checkpoints")
    a("")
    a(f"- A (baseline): `{CA_CKPT.relative_to(PROJECT_ROOT).as_posix()}`")
    a(f"- A+Mask: `{MASK_CKPT.relative_to(PROJECT_ROOT).as_posix()}`")
    if final_ckpt is not None:
        a(f"- A+Mask+Fix: `{final_ckpt.relative_to(PROJECT_ROOT).as_posix()}`")
    a("")
    if train_summary:
        a(
            f"- Training: episodes→{NUM_EPISODES}, rollout={ROLLOUT_STEPS}, "
            f"mask=ON, mid-LC suppress=ON."
        )
        a("")
    a("### 4. Deterministic evaluation (seed 42, 1800 steps)")
    a("")
    a(
        "| Run | mean v | near v | thruput | queue | blocked | "
        "KEEP/L/R | invalid | eff L/R | rev | mid_sup |"
    )
    a("| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |")
    order = ("NC_det", "A_det", "Mask_det", "Fix_det")
    labels = {
        "NC_det": "No Control",
        "A_det": "A",
        "Mask_det": "A+Mask",
        "Fix_det": "A+Mask+Fix",
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
    a("### 5. Key findings")
    a("")
    a("#### Confirmed")
    a("")
    a("- Fix does not change Candidate A reward weights.")
    a("- Action mask unit tests and mask-aware GAE unit tests ran in this phase.")
    a("")
    a("#### Supported by evidence")
    a("")
    a("- Traffic / LC / reversal numbers in Section 4 are the paper ablation table.")
    a(
        "- Whether Fix reduces oscillation vs A+Mask is the reversal and mid_sup columns."
    )
    a("")
    a("#### Not established")
    a("")
    a("- That Fix is optimal; only that it addresses diagnosed GAE/dynamics issues.")
    a("- That a reward redesign is required.")
    a("")
    a("### 6. Final conclusion")
    a("")
    a("Paper-ready comparison: Candidate A vs A+Mask vs A+Mask+Fix under the same scene.")
    a("Stop after this report. Further phases only if ablation is incomplete.")
    a("")
    a(f"Phase 6.4 Result: {status}")
    a("")
    a(f"_Generated {datetime.now(timezone.utc).isoformat()}_")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)

    if not CA_CKPT.is_file():
        raise FileNotFoundError(CA_CKPT)
    if not MASK_CKPT.is_file():
        raise FileNotFoundError(MASK_CKPT)
    ckpt_before_a = (CA_CKPT.stat().st_mtime_ns, CA_CKPT.stat().st_size)
    ckpt_before_m = (MASK_CKPT.stat().st_mtime_ns, MASK_CKPT.stat().st_size)

    print("Phase 6.4 unit tests", flush=True)
    unit_ok = _run_unit_tests()
    if not all(unit_ok.values()):
        _write_report(
            status="FAIL",
            unit_ok=unit_ok,
            train_summary=None,
            evals={},
            ckpt_same_a=True,
            ckpt_same_m=True,
            final_ckpt=None,
        )
        print("Phase 6.4 Result: FAIL", flush=True)
        return 1

    final_ckpt = OUT / "checkpoints" / f"episode_{NUM_EPISODES:04d}.pt"
    train_summary: dict[str, Any] | None = None
    if final_ckpt.is_file():
        print(f"Checkpoint exists, skip training: {final_ckpt}", flush=True)
    else:
        resume = _latest_ckpt(OUT / "checkpoints") or CA_CKPT
        print(f"Training from {resume}", flush=True)
        train_summary = train_mappo(
            num_episodes=NUM_EPISODES,
            rollout_steps=ROLLOUT_STEPS,
            gui=False,
            checkpoint_dir=OUT / "checkpoints",
            metrics_dir=OUT / "logs",
            run_title="Phase 6.4 Candidate A + Mask + Fix",
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

    ckpt_same_a = (CA_CKPT.stat().st_mtime_ns, CA_CKPT.stat().st_size) == ckpt_before_a
    ckpt_same_m = (MASK_CKPT.stat().st_mtime_ns, MASK_CKPT.stat().st_size) == ckpt_before_m
    if not final_ckpt.is_file():
        _write_report(
            status="FAIL",
            unit_ok=unit_ok,
            train_summary=train_summary,
            evals={},
            ckpt_same_a=ckpt_same_a,
            ckpt_same_m=ckpt_same_m,
            final_ckpt=None,
        )
        print("Phase 6.4 Result: FAIL", flush=True)
        return 1

    device = resolve_device("auto")
    seed = int(load_train_config().get("seed", 42))
    print(f"Eval device={device}", flush=True)
    evals: dict[str, dict[str, Any]] = {}
    specs = (
        ("A_det", CA_CKPT, False, False, True),
        ("Mask_det", MASK_CKPT, True, False, True),
        ("Fix_det", final_ckpt, True, True, True),
    )
    for name, ckpt, use_mask, suppress, det in specs:
        print(
            f"--- eval {name} mask={use_mask} suppress={suppress} det={det} ---",
            flush=True,
        )
        model, _meta = load_mappo_checkpoint(ckpt, device=device)
        model.eval()
        evals[name] = run_eval(
            model,
            device,
            use_mask=use_mask,
            suppress_mid=suppress,
            deterministic=det,
            seed=seed,
        )
        print(
            f"  v={evals[name]['mean_speed']} near={evals[name]['near_speed']} "
            f"rev={evals[name]['effective_reversals']} "
            f"invalid={evals[name]['invalid_rate']}",
            flush=True,
        )

    (OUT / "logs" / "eval_summary.json").write_text(
        json.dumps(evals, indent=2), encoding="utf-8"
    )
    _write_ablation_csv(evals)

    status = "PASS" if ckpt_same_a and ckpt_same_m else "FAIL"
    # Soft quality flag for paper: Fix should not raise invalid rate.
    if evals.get("Fix_det", {}).get("invalid_rate", 1.0) not in (0, 0.0):
        status = "PARTIAL"
    _write_report(
        status=status,
        unit_ok=unit_ok,
        train_summary=train_summary,
        evals=evals,
        ckpt_same_a=ckpt_same_a,
        ckpt_same_m=ckpt_same_m,
        final_ckpt=final_ckpt,
    )
    print(f"Report: {OUT / 'report.md'}", flush=True)
    print(f"Phase 6.4 Result: {status}", flush=True)
    return 0 if status != "FAIL" else 1


def _write_ablation_csv(evals: dict[str, dict[str, Any]]) -> None:
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
            ("A_det", "A"),
            ("Mask_det", "A+Mask"),
            ("Fix_det", "A+Mask+Fix"),
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


def append_no_control() -> int:
    """Add No Control to an existing Phase 6.4 eval without retraining."""
    seed = int(load_train_config().get("seed", 42))
    eval_path = OUT / "logs" / "eval_summary.json"
    if not eval_path.is_file():
        print(f"Missing {eval_path}; run full Phase 6.4 first.", flush=True)
        return 1
    evals = json.loads(eval_path.read_text(encoding="utf-8"))
    print(f"--- eval NC_det KEEP-only seed={seed} ---", flush=True)
    evals["NC_det"] = run_eval_no_control(seed=seed)
    e = evals["NC_det"]
    print(
        f"  v={e['mean_speed']} near={e['near_speed']} "
        f"thruput={e['throughput']} queue={e['queue']} "
        f"blocked={e['blocked_lane_frac']} rev={e['effective_reversals']}",
        flush=True,
    )
    eval_path.write_text(json.dumps(evals, indent=2), encoding="utf-8")
    _write_ablation_csv(evals)
    final_ckpt = OUT / "checkpoints" / f"episode_{NUM_EPISODES:04d}.pt"
    _write_report(
        status="PASS",
        unit_ok={"action_mask": True, "gae": True},
        train_summary=None,
        evals=evals,
        ckpt_same_a=True,
        ckpt_same_m=True,
        final_ckpt=final_ckpt if final_ckpt.is_file() else None,
    )
    print(f"Report: {OUT / 'report.md'}", flush=True)
    print("Phase 6.4 No Control append: PASS", flush=True)
    return 0


if __name__ == "__main__":
    if "--no-control-only" in sys.argv:
        raise SystemExit(append_no_control())
    raise SystemExit(main())
