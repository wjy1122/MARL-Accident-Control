"""Phase 6.1 — Candidate A reward, plus an action mask. Nothing else changes.

Continues from the Phase 5.3 episode-50 checkpoint and trains updates 51–100.
Does not overwrite that checkpoint. Does not edit the reward, observation,
actor/critic architecture, or PPO hyperparameters.
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

OUT = PROJECT_ROOT / "results" / "experiments" / "reward_candidate_A_phase6_1_mask"
CA_CKPT = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "reward_candidate_A_phase5_3"
    / "checkpoints"
    / "episode_0050.pt"
)
CA_REWARD = PROJECT_ROOT / "configs" / "reward_candidate_A.yaml"
NUM_EPISODES = 100  # resume at 50 → train 51..100
ROLLOUT_STEPS = 512
MAX_EVAL_STEPS = 1800
LC_HORIZON = 10


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


def _run_unit_tests() -> bool:
    script = PROJECT_ROOT / "tests" / "test_action_mask.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    return proc.returncode == 0


def run_eval(
    model: torch.nn.Module,
    device: torch.device,
    *,
    use_mask: bool,
    deterministic: bool,
    seed: int,
) -> dict[str, Any]:
    """One closed-loop episode. Mask is applied only when ``use_mask`` is True."""
    set_eval_seed(seed)
    model.eval()
    env_cfg = copy.deepcopy(load_config())
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
    records: list[dict[str, Any]] = []
    lane_at: dict[tuple[str, int], int] = {}
    act_n = np.zeros(3, dtype=np.int64)
    invalid_left = 0
    invalid_right = 0

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
                pos = float(st["position"])
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
                illegal = (
                    action != KEEP
                    and target_lane_index(
                        current_lane=lane,
                        action=action,
                        n_lanes=n_lanes,
                        left_increases_index=left_up,
                    )
                    is None
                )
                if illegal and action == LEFT:
                    invalid_left += 1
                if illegal and action == RIGHT:
                    invalid_right += 1
                records.append(
                    {
                        "step": step,
                        "cav_id": cid,
                        "lane": lane,
                        "pos": pos,
                        "zone": _zone(pos, x_acc, zone_r),
                        "action": action,
                        "illegal": illegal,
                    }
                )
                lane_at[(cid, step)] = lane

            _obs, _rew, _term, truncated, _info = env.step(actions)
            if truncated:
                break
    finally:
        env.close()

    for rec in records:
        future = None
        for k in range(1, LC_HORIZON + 1):
            nxt = lane_at.get((rec["cav_id"], rec["step"] + k))
            if nxt is None:
                break
            future = nxt
        if future is None:
            rec["effective"] = "unknown"
        elif future > rec["lane"]:
            rec["effective"] = "LEFT"
        elif future < rec["lane"]:
            rec["effective"] = "RIGHT"
        else:
            rec["effective"] = "none"

    req_rev = 0
    eff_rev = 0
    by_cav: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_cav.setdefault(rec["cav_id"], []).append(rec)
    for seq in by_cav.values():
        seq.sort(key=lambda r: r["step"])
        prev_a = None
        prev_eff = None
        for rec in seq:
            a = int(rec["action"])
            if prev_a in (LEFT, RIGHT) and a in (LEFT, RIGHT) and a != prev_a:
                req_rev += 1
            if a in (LEFT, RIGHT):
                prev_a = a
            elif a == KEEP:
                prev_a = None
            eff = rec["effective"]
            if eff in ("LEFT", "RIGHT"):
                if prev_eff in ("LEFT", "RIGHT") and eff != prev_eff:
                    eff_rev += 1
                prev_eff = eff

    n_act = int(act_n.sum())
    zones = {}
    for name in ("before", "near", "after"):
        sub = [r for r in records if r["zone"] == name]
        zn = len(sub)
        zones[name] = {
            "n": zn,
            "KEEP": sum(1 for r in sub if r["action"] == KEEP) / zn if zn else None,
            "LEFT": sum(1 for r in sub if r["action"] == LEFT) / zn if zn else None,
            "RIGHT": sum(1 for r in sub if r["action"] == RIGHT) / zn if zn else None,
            "invalid": sum(1 for r in sub if r["illegal"]) / zn if zn else None,
        }

    known = [r for r in records if r["effective"] != "unknown"]
    return {
        "deterministic": bool(deterministic),
        "use_mask": bool(use_mask),
        "n_actions": n_act,
        "mean_speed": float(np.mean(speeds)) if speeds else None,
        "mean_speed_near": float(np.mean(speeds_near)) if speeds_near else None,
        "throughput": int(arrived),
        "queue": float(np.mean(queues)) if queues else None,
        "blocked_lane_fraction": (
            float(near_cav_blocked) / float(near_cav) if near_cav else None
        ),
        "KEEP": float(act_n[KEEP] / n_act) if n_act else None,
        "LEFT": float(act_n[LEFT] / n_act) if n_act else None,
        "RIGHT": float(act_n[RIGHT] / n_act) if n_act else None,
        "invalid_left": int(invalid_left),
        "invalid_right": int(invalid_right),
        "invalid_action_rate": (
            float(invalid_left + invalid_right) / float(n_act) if n_act else None
        ),
        "effective_LEFT": int(sum(1 for r in known if r["effective"] == "LEFT")),
        "effective_RIGHT": int(sum(1 for r in known if r["effective"] == "RIGHT")),
        "requested_reversals": int(req_rev),
        "effective_reversals": int(eff_rev),
        "zones": zones,
    }


def _plot_training(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"figures skipped: {exc}", flush=True)
        return []
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ep = [int(r["episode"]) for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0))
    axes[0, 0].plot(ep, [float(r["mean_reward"]) for r in rows], color="#2166ac")
    axes[0, 0].set_title("Mean reward")
    axes[0, 1].plot(ep, [float(r.get("entropy", 0)) for r in rows], color="#4d4d4d")
    axes[0, 1].set_title("Entropy")
    axes[1, 0].plot(ep, [float(r.get("KEEP_ratio", 0)) for r in rows], label="KEEP")
    axes[1, 0].plot(ep, [float(r.get("LEFT_ratio", 0)) for r in rows], label="LEFT")
    axes[1, 0].plot(ep, [float(r.get("RIGHT_ratio", 0)) for r in rows], label="RIGHT")
    axes[1, 0].set_title("Action share")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].plot(
        ep, [float(r.get("invalid_action_rate", 0)) for r in rows], color="#b2182b"
    )
    axes[1, 1].set_title("Invalid action rate")
    for ax in axes.ravel():
        ax.set_xlabel("Update")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = fig_dir / "training_curves.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return [str(path.relative_to(OUT))]


def _write_report(
    *,
    status: str,
    unit_ok: bool,
    train_summary: dict[str, Any] | None,
    evals: dict[str, dict[str, Any]],
    train_cfg: dict[str, Any],
    ckpt_same: bool,
    figures: list[str],
    final_ckpt: Path | None,
) -> None:
    block = train_cfg.get("train", train_cfg)
    lines: list[str] = []
    a = lines.append
    a("# Phase 6.1 Report")
    a("")
    a("## Candidate A + Action Mask")
    a("")
    a("### 1. Status")
    a("")
    a(f"`{status}`")
    a("")
    a(
        "PASS means the masked run finished, the unit tests passed, and evaluation "
        "illegal-action rate is 0. It does not mean the controller is better, and it "
        "does not authorize a reward edit."
    )
    a("")
    a("### 2. Experimental control")
    a("")
    a("- Reward: **unchanged** (`configs/reward_candidate_A.yaml`).")
    a("- Observation: **unchanged** (still 17-D; no availability bit).")
    a("- Action space: **unchanged** (0 KEEP, 1 LEFT, 2 RIGHT).")
    a("- Actor architecture: **unchanged**. Mask is applied to logits only.")
    a("- Critic: **unchanged**.")
    a("- PPO hyperparameters: **unchanged**.")
    a("- Action mask: **added**.")
    a(f"- Candidate A episode-50 checkpoint overwritten: **{'YES' if not ckpt_same else 'No'}**.")
    a("")
    a("### 3. Action Mask implementation")
    a("")
    a("```text")
    a("raw logits → masked_fill(illegal, -1e9) → Categorical → sample / argmax")
    a("```")
    a("")
    a(
        "- Mask location: `Actor.get_action` and `Actor.evaluate_actions`, before `Categorical`."
    )
    a(
        "- Mask generation: `env/action_mask.py` via `target_lane_index`. "
        "Lane 0 (rightmost): `[1,1,0]`. Lane 1: `[1,1,1]`. Lane 2 (leftmost): `[1,0,1]`. "
        "`left_increases_index=True`."
    )
    a("- Illegal actions are not rewritten to KEEP.")
    a("- Rollout stores `action_mask[t, slot]` with the obs, action, and old log_prob of that decision.")
    a(
        "- PPO recomputes `new_log_prob` with that stored mask, not by reading the live SUMO lane. "
        "A slot rebind writes a new row; it does not reuse the departed vehicle's mask."
    )
    a("- Deterministic eval uses `argmax` on masked logits. Stochastic eval samples the masked categorical.")
    a("- Candidate A comparison eval does **not** apply the mask, because that policy was trained without one.")
    a("")
    a("### 4. Unit Test")
    a("")
    a(f"- `tests/test_action_mask.py`: **{'PASS' if unit_ok else 'FAIL'}**.")
    a("- lane 0: RIGHT probability 0, RIGHT samples 0 / 4000, deterministic argmax is not RIGHT.")
    a("- lane 2: LEFT probability 0, LEFT samples 0 / 4000.")
    a("- lane 1: all three actions sampled.")
    a("- Same mask → log-prob ratio = 1 and finite. Illegal log_prob is finite (`-1e9`), not NaN.")
    a("- Stored mask follows the slot across a simulated rebind (lane 0 then lane 2).")
    a("- One PPO update on a masked batch stayed finite.")
    a("")
    a("### 5. Training setup")
    a("")
    a("- Phase 6.1 continues from Candidate A episode 50 checkpoint with Action Mask enabled.")
    a("- This is not a from-scratch run.")
    a(f"- Initialization: `{CA_CKPT.relative_to(PROJECT_ROOT).as_posix()}` (read-only).")
    a(f"- Continued updates: episodes 51–{NUM_EPISODES} (50 updates, same budget as Candidate A).")
    a(f"- Seed: {train_cfg.get('seed', 42)} (re-seeded by `train()` on resume; RNG is not bit-identical to the original run).")
    a(f"- Rollout length: {ROLLOUT_STEPS}.")
    a(f"- gamma: {block.get('gamma', 0.99)}; gae_lambda: {block.get('gae_lambda', 0.95)}.")
    a(
        f"- clip_eps: {block.get('clip_eps', block.get('clip_epsilon', 0.2))}; "
        f"entropy_coef: {block.get('entropy_coef', 0.01)}; "
        f"value_coef: {block.get('value_coef', 0.5)}."
    )
    a(
        f"- actor_lr: {block.get('actor_lr', block.get('lr', 3e-4))}; "
        f"critic_lr: {block.get('critic_lr', 1e-3)}; "
        f"minibatch: {block.get('minibatch_size', 64)}; "
        f"ppo_epochs: {block.get('ppo_epochs', 4)}."
    )
    if train_summary:
        a(f"- Final mean reward (last update): {_fmt(train_summary.get('final_reward'), 4)}.")
        a(f"- Actor changed: {train_summary.get('actor_changed')}; Critic changed: {train_summary.get('critic_changed')}.")
        a(f"- Metrics: `{train_summary.get('metrics_csv')}`.")
    if final_ckpt is not None:
        a(f"- Final checkpoint: `{final_ckpt.relative_to(PROJECT_ROOT).as_posix()}`.")
    a("")
    a("### 6. Training curves")
    a("")
    if figures:
        for fig in figures:
            a(f"- `{fig}`")
    else:
        a("- No figure written.")
    a("- Per-update invalid action rate is logged in `logs/train_metrics.csv`. The trainer aborts if it is above 0.")
    a("")
    a("### 7. Evaluation results")
    a("")
    a("Same scene, seed 42, 1800 steps, Candidate A reward. Candidate A column is the frozen episode-50 policy **without** a mask.")
    a("")
    a("| Metric | Candidate A det | Mask det | Candidate A stoch | Mask stoch |")
    a("| --- | ---: | ---: | ---: | ---: |")

    def _cell(key: str, name: str, nd: int = 3) -> str:
        row = evals.get(name) or {}
        return _fmt(row.get(key), nd)

    for key, label, nd in (
        ("mean_speed", "mean speed", 3),
        ("mean_speed_near", "near-zone speed", 3),
        ("throughput", "throughput", 0),
        ("queue", "queue", 3),
        ("blocked_lane_fraction", "blocked-lane fraction", 3),
        ("invalid_action_rate", "invalid action rate", 4),
        ("effective_LEFT", "effective LEFT", 0),
        ("effective_RIGHT", "effective RIGHT", 0),
        ("requested_reversals", "requested reversals", 0),
        ("effective_reversals", "effective reversals", 0),
        ("KEEP", "KEEP rate", 3),
        ("LEFT", "LEFT rate", 3),
        ("RIGHT", "RIGHT rate", 3),
    ):
        a(
            f"| {label} | {_cell(key, 'A_det', nd)} | {_cell(key, 'M_det', nd)} | "
            f"{_cell(key, 'A_stoch', nd)} | {_cell(key, 'M_stoch', nd)} |"
        )
    a("")
    a("### 8. Action behavior")
    a("")
    for name, title in (
        ("A_det", "Candidate A deterministic"),
        ("M_det", "Mask deterministic"),
        ("A_stoch", "Candidate A stochastic"),
        ("M_stoch", "Mask stochastic"),
    ):
        row = evals.get(name)
        if not row:
            a(f"- {title}: not run.")
            continue
        a(
            f"- {title}: KEEP/LEFT/RIGHT = "
            f"{_fmt(row.get('KEEP'))}/{_fmt(row.get('LEFT'))}/{_fmt(row.get('RIGHT'))}; "
            f"invalid LEFT/RIGHT counts = {row.get('invalid_left')}/{row.get('invalid_right')}; "
            f"effective LEFT/RIGHT = {row.get('effective_LEFT')}/{row.get('effective_RIGHT')}; "
            f"requested reversals = {row.get('requested_reversals')}; "
            f"effective reversals = {row.get('effective_reversals')}."
        )
    a("")
    a("### 9. Region analysis")
    a("")
    a("Shares are of sampled actions in that region. Zones: before `x<550`, near `|x-700|≤150`, after `x>850`.")
    a("")
    a("| Run | Region | N | KEEP | LEFT | RIGHT | invalid |")
    a("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for name, title in (
        ("A_det", "A det"),
        ("M_det", "Mask det"),
        ("A_stoch", "A stoch"),
        ("M_stoch", "Mask stoch"),
    ):
        zones = (evals.get(name) or {}).get("zones") or {}
        for region in ("before", "near", "after"):
            z = zones.get(region) or {}
            a(
                f"| {title} | {region} | {z.get('n', 0)} | {_fmt(z.get('KEEP'))} | "
                f"{_fmt(z.get('LEFT'))} | {_fmt(z.get('RIGHT'))} | {_fmt(z.get('invalid'), 4)} |"
            )
    a("")
    a("### 10. Main findings")
    a("")
    mdet = evals.get("M_det") or {}
    adet = evals.get("A_det") or {}
    a("#### Confirmed")
    a("")
    a("- The mask is applied before the categorical, and the unit tests passed." if unit_ok else "- Unit tests did not pass.")
    a("- Reward, observation dimension, and network sizes were not edited for this run.")
    if mdet:
        a(
            f"- Deterministic masked eval illegal-action rate = {_fmt(mdet.get('invalid_action_rate'), 4)} "
            f"(Candidate A without mask: {_fmt(adet.get('invalid_action_rate'), 4)})."
        )
    a("")
    a("#### Supported")
    a("")
    a(
        "- Whether masking changes speed, queue, throughput, effective lane changes, or reversals "
        "is the table in Section 7. A zero illegal rate only shows that illegal samples were removed."
    )
    a("")
    a("#### Unknown")
    a("")
    a("- A from-scratch masked run was not trained.")
    a("- Whether a longer horizon or a reward change would be needed if legal lane changes remain excessive.")
    a("")
    a("### 11. Final conclusion")
    a("")
    a("1. Action mask implementation: " + ("yes, unit tests passed." if unit_ok else "no."))
    inv = mdet.get("invalid_action_rate")
    a(
        "2. Illegal action rate under the mask: "
        + (_fmt(inv, 4) if inv is not None else "not evaluated")
        + "."
    )
    a(
        "3. Effective lane changes (deterministic): "
        f"Candidate A LEFT/RIGHT {adet.get('effective_LEFT')}/{adet.get('effective_RIGHT')}; "
        f"Mask {mdet.get('effective_LEFT')}/{mdet.get('effective_RIGHT')}."
    )
    a(
        "4. Direction reversals (deterministic, requested / effective): "
        f"Candidate A {adet.get('requested_reversals')}/{adet.get('effective_reversals')}; "
        f"Mask {mdet.get('requested_reversals')}/{mdet.get('effective_reversals')}."
    )
    a(
        "5. Deterministic KEEP/LEFT/RIGHT: "
        f"Candidate A {_fmt(adet.get('KEEP'))}/{_fmt(adet.get('LEFT'))}/{_fmt(adet.get('RIGHT'))}; "
        f"Mask {_fmt(mdet.get('KEEP'))}/{_fmt(mdet.get('LEFT'))}/{_fmt(mdet.get('RIGHT'))}."
    )
    a("6–7. Region shares and traffic metrics are in Sections 7 and 9. They are measurements, not a claim that the controller improved.")
    a(
        "8. Worth a next experiment only as a controlled comparison, not as a reward change. "
        "This report does not start Phase 6.2."
    )
    a("")
    a(f"Phase 6.1 Result: {status}")
    a("")
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    (OUT / "eval").mkdir(exist_ok=True)
    if not CA_CKPT.is_file():
        raise SystemExit(f"Missing Candidate A checkpoint: {CA_CKPT}")
    ckpt_before = (CA_CKPT.stat().st_mtime_ns, CA_CKPT.stat().st_size)

    print("Phase 6.1 unit tests", flush=True)
    unit_ok = _run_unit_tests()
    if not unit_ok:
        _write_report(
            status="FAIL",
            unit_ok=False,
            train_summary=None,
            evals={},
            train_cfg=load_train_config(),
            ckpt_same=True,
            figures=[],
            final_ckpt=None,
        )
        print("Phase 6.1 Result: FAIL", flush=True)
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
            run_title="Phase 6.1 Candidate A + Action Mask",
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

    ckpt_same = (CA_CKPT.stat().st_mtime_ns, CA_CKPT.stat().st_size) == ckpt_before
    if not final_ckpt.is_file():
        _write_report(
            status="FAIL",
            unit_ok=True,
            train_summary=train_summary,
            evals={},
            train_cfg=load_train_config(),
            ckpt_same=ckpt_same,
            figures=[],
            final_ckpt=None,
        )
        print("Phase 6.1 Result: FAIL", flush=True)
        return 1

    rows: list[dict[str, Any]] = []
    metrics_csv = OUT / "logs" / "train_metrics.csv"
    if metrics_csv.is_file():
        with metrics_csv.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    figures = _plot_training(rows)

    device = resolve_device("auto")
    seed = int(load_train_config().get("seed", 42))
    print(f"Eval device={device}", flush=True)
    evals: dict[str, dict[str, Any]] = {}
    specs = (
        ("A_det", CA_CKPT, False, True),
        ("M_det", final_ckpt, True, True),
        ("A_stoch", CA_CKPT, False, False),
        ("M_stoch", final_ckpt, True, False),
    )
    for name, ckpt, use_mask, deterministic in specs:
        print(f"--- eval {name} mask={use_mask} det={deterministic} ---", flush=True)
        model, _meta = load_mappo_checkpoint(ckpt, device=device)
        model.eval()
        result = run_eval(
            model,
            device,
            use_mask=use_mask,
            deterministic=deterministic,
            seed=seed,
        )
        evals[name] = result
        (OUT / "eval" / f"{name}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(
            f"  invalid={result['invalid_action_rate']} "
            f"KEEP/L/R={result['KEEP']:.3f}/{result['LEFT']:.3f}/{result['RIGHT']:.3f}",
            flush=True,
        )

    mask_inv = float((evals.get("M_det") or {}).get("invalid_action_rate") or 0.0)
    mask_inv_s = float((evals.get("M_stoch") or {}).get("invalid_action_rate") or 0.0)
    if not ckpt_same or mask_inv > 0.0 or mask_inv_s > 0.0:
        status = "FAIL"
    else:
        status = "PASS"
    _write_report(
        status=status,
        unit_ok=True,
        train_summary=train_summary,
        evals=evals,
        train_cfg=load_train_config(),
        ckpt_same=ckpt_same,
        figures=figures,
        final_ckpt=final_ckpt,
    )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "unit_tests": unit_ok,
        "ckpt_unchanged": ckpt_same,
        "final_checkpoint": str(final_ckpt),
        "evals": evals,
    }
    (OUT / "eval" / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Report: {OUT / 'report.md'}", flush=True)
    print(f"Phase 6.1 Result: {status}", flush=True)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
