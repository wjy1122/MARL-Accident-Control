"""Phase 4.4: first closed-loop MAPPO training (Group Benefit, no Attention).

Example (smoke, 10 episodes):
  .\\.venv\\Scripts\\python.exe scripts\\train_mappo.py

Optional overrides:
  .\\.venv\\Scripts\\python.exe scripts\\train_mappo.py --num-episodes 10
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.mappo_trainer import MAPPOTrainer  # noqa: E402
from algorithms.rollout import RolloutCollector, load_train_config, resolve_device  # noqa: E402
from algorithms.rollout_buffer import RolloutBatch  # noqa: E402
from env.action_mask import illegal_action_rate  # noqa: E402
from env.highway_accident_env import HighwayAccidentEnv, load_config  # noqa: E402
from models.mappo_model import MAPPOModel, load_model_config  # noqa: E402
from utils.traffic_diagnostics import (  # noqa: E402
    diagnose_rollout_batch,
    format_action_share_line,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clone_params(module: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().cpu().clone() for p in module.parameters()]


def _params_changed(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    return any(not torch.equal(x, y) for x, y in zip(a, b))


def _masked_reward_stats(batch: RolloutBatch) -> tuple[float, float, int]:
    """Per-agent rewards stay [T,N]; logging uses mask only."""
    rewards = np.asarray(batch.rewards, dtype=np.float64)
    mask = np.asarray(batch.agent_mask, dtype=np.float64)
    valid = rewards[mask > 0.5]
    if valid.size == 0:
        return 0.0, 0.0, int(batch.rollout_steps)
    return float(valid.mean()), float(valid.sum()), int(batch.rollout_steps)


def _mean_cav_speed(env: HighwayAccidentEnv) -> float:
    states = env.get_cav_states()
    if not states:
        return 0.0
    return float(np.mean([float(st["speed"]) for st in states.values()]))


def _approx_queue_length(env: HighwayAccidentEnv) -> float:
    """Logging-only upstream slow-vehicle count (not used in reward/training)."""
    accident_cfg = env.cfg.get("accident", {})
    test_cfg = env.cfg.get("test", {})
    accident_pos = float(accident_cfg.get("position_m", 1200.0))
    upstream_m = float(test_cfg.get("congestion_upstream_m", 250.0))
    slow_mps = float(test_cfg.get("congestion_speed_mps", 15.0))
    lo = accident_pos - upstream_m
    count = 0
    for st in env.get_vehicle_states().values():
        if st.get("type") == "ACCIDENT":
            continue
        pos = float(st["position"])
        if lo <= pos <= accident_pos and float(st["speed"]) < slow_mps:
            count += 1
    return float(count)


def _diag_cfg(env: HighwayAccidentEnv) -> dict[str, float]:
    cfg = env.cfg
    return {
        "road_length_m": float(cfg.get("highway", {}).get("length_m", 2000.0)),
        "max_speed_mps": float(
            cfg.get("observation", {}).get(
                "max_speed_mps",
                cfg.get("highway", {}).get("speed_limit_mps", 30.0),
            )
        ),
        "accident_position_m": float(cfg.get("accident", {}).get("position_m", 1200.0)),
        "obs_range_m": float(cfg.get("observation", {}).get("range_m", 150.0)),
        "near_accident_m": float(cfg.get("observation", {}).get("range_m", 150.0)),
        "individual_weight": float(getattr(env, "reward_individual_weight", 0.7)),
        "group_weight": float(getattr(env, "reward_group_weight", 0.3)),
    }


def train(
    *,
    num_episodes: int | None = None,
    rollout_steps: int | None = None,
    gui: bool = False,
    checkpoint_dir: str | Path | None = None,
    metrics_dir: str | Path | None = None,
    run_title: str | None = None,
    env_cfg: dict[str, Any] | None = None,
    reward_config: dict[str, Any] | Path | str | None = None,
    resume_from: str | Path | None = None,
    use_action_mask: bool = False,
) -> dict[str, Any]:
    train_cfg = load_train_config()
    model_cfg = load_model_config()
    mblock = model_cfg.get("model", model_cfg)
    tblock = train_cfg.get("train", train_cfg)

    seed = int(train_cfg.get("seed", tblock.get("seed", 42)))
    set_seed(seed)

    device = resolve_device(train_cfg.get("device", "auto"))
    n_ep = int(
        num_episodes
        if num_episodes is not None
        else tblock.get("num_episodes", tblock.get("episodes", 10))
    )
    r_steps = int(
        rollout_steps
        if rollout_steps is not None
        else tblock.get("rollout_steps", 128)
    )
    warmup = int(tblock.get("warmup_steps", 50))
    save_interval = int(tblock.get("save_interval", 5))
    log_interval = int(tblock.get("log_interval", 1))
    if checkpoint_dir is not None:
        ckpt_dir = Path(checkpoint_dir)
        if not ckpt_dir.is_absolute():
            ckpt_dir = PROJECT_ROOT / ckpt_dir
    else:
        ckpt_rel = tblock.get(
            "checkpoint_dir", "results/mappo_group_benefit/checkpoints"
        )
        ckpt_dir = PROJECT_ROOT / Path(ckpt_rel)

    num_agents = int(mblock.get("num_agents", 4))
    obs_dim = int(mblock.get("observation_dim", 17))
    action_dim = int(mblock.get("action_dim", 3))
    hidden = int(mblock.get("hidden_dim", 128))

    title = run_title or "Phase 4.4.3 MAPPO Training (512-step coverage smoke)"
    print("================================")
    print(title)
    print("================================")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Seed: {seed}")
    print(f"Episodes / updates: {n_ep}")
    print(f"rollout_steps = {r_steps}")
    if r_steps != 512:
        print(
            f"WARNING: expected rollout_steps=512 for Phase 4.4.3, got {r_steps}"
        )
    print(f"Num agents: {num_agents}")
    print(f"Checkpoint dir: {ckpt_dir}")
    print(f"Action mask: {'ON' if use_action_mask else 'OFF'}")
    print()

    env_cfg = copy.deepcopy(env_cfg if env_cfg is not None else load_config())
    # Keep env horizon; RolloutCollector may reset mid-rollout on truncate.
    # GUI: optional delay only when gui=True (from sumo.gui_delay_ms / gui.delay_ms).
    # reward_config=None → configs/reward.yaml (current). Pass Candidate A YAML for Phase 5.3.
    env = HighwayAccidentEnv(
        config=env_cfg, gui=gui, reward_config=reward_config
    )
    print(f"Reward variant: {getattr(env, 'reward_variant', 'current')}")
    if getattr(env, "reward_config_path", None) is not None:
        print(f"Reward config: {env.reward_config_path}")
    model = MAPPOModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_agents=num_agents,
        hidden_dim=hidden,
    ).to(device)
    collector = RolloutCollector(
        num_agents=num_agents,
        obs_dim=obs_dim,
        rollout_steps=r_steps,
        warmup_steps=warmup,
        device=device,
        use_action_mask=bool(use_action_mask),
    )
    trainer = MAPPOTrainer(model, train_cfg, device=device)

    start_ep = 1
    resumed_from_episode = 0
    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.is_absolute():
            resume_path = PROJECT_ROOT / resume_path
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        resumed_from_episode = int(trainer.load_checkpoint(resume_path))
        start_ep = resumed_from_episode + 1
        print(
            f"Resume: loaded {resume_path.name} "
            f"(completed episode {resumed_from_episode}); "
            f"continuing {start_ep}..{n_ep}"
        )
        if start_ep > n_ep:
            print(f"Resume: checkpoint already at/after episode {n_ep}; no further updates")

    # Snapshot after optional load so "changed" means this run's updates, not the load itself.
    actor_before = _clone_params(model.actor)
    critic_before = _clone_params(model.critic)

    log_rows: list[dict[str, Any]] = []
    first_reward: float | None = None
    last_reward = 0.0
    last_stats = None
    saved_paths: list[Path] = []

    try:
        for ep in range(start_ep, n_ep + 1):
            batch = collector.collect(env, model, rollout_steps=r_steps)
            stats = trainer.update(batch)
            last_stats = stats

            rew_mean, rew_sum, ep_len = _masked_reward_stats(batch)
            mean_speed = _mean_cav_speed(env)
            queue_len = _approx_queue_length(env)
            diag = diagnose_rollout_batch(
                batch.observations,
                batch.actions,
                batch.rewards,
                batch.agent_mask,
                **_diag_cfg(env),
            )

            if first_reward is None:
                first_reward = rew_mean
            last_reward = rew_mean

            finite = all(
                math.isfinite(x)
                for x in (
                    stats.actor_loss,
                    stats.critic_loss,
                    stats.entropy,
                    stats.total_loss,
                    rew_mean,
                    rew_sum,
                )
            )
            if not finite:
                raise RuntimeError(
                    f"NaN/Inf detected at episode {ep}: "
                    f"actor={stats.actor_loss} critic={stats.critic_loss} "
                    f"ent={stats.entropy} reward_mean={rew_mean}"
                )

            life = dict(getattr(collector, "last_lifecycle", {}) or {})
            inv_rate = 0.0
            if batch.action_mask is not None:
                inv_rate = illegal_action_rate(
                    batch.actions, batch.action_mask, batch.agent_mask
                )
                if inv_rate > 0.0:
                    raise RuntimeError(
                        f"Action mask failed at episode {ep}: "
                        f"invalid_action_rate={inv_rate:.6f}"
                    )
            row = {
                "episode": ep,
                "rollout_steps": int(batch.rollout_steps),
                "env_reset_count": int(life.get("env_reset_count", -1)),
                "sumo_start_count": int(life.get("sumo_start_count", -1)),
                "departed_cav_count": int(life.get("departed_cav_count", -1)),
                "new_cav_count": int(life.get("new_cav_count", -1)),
                "active_cav_peak": int(life.get("active_cav_peak", -1)),
                "active_slot_peak": int(life.get("active_slot_peak", -1)),
                "rebind_events": int(life.get("rebind_events", -1)),
                "mean_reward": rew_mean,
                "mean_episode_reward": rew_mean,
                "episode_reward_mean": rew_mean,
                "episode_reward_sum": rew_sum,
                "reward_std": diag.reward_std,
                "mean_individual_reward": diag.mean_individual_reward,
                "mean_group_reward": diag.mean_group_reward,
                "mean_total_reward": diag.mean_total_reward,
                "min_pos": diag.min_pos_m,
                "max_pos": diag.max_pos_m,
                "near_frac": diag.near_accident_frac,
                "num_near_accident_steps": diag.num_near_accident_steps,
                "reached_accident_zone": int(diag.reached_accident_zone),
                "mean_speed": mean_speed,
                "mean_speed_mps": diag.mean_speed_mps,
                "mean_speed_near_accident": diag.mean_speed_near_accident,
                "speed_before_mps": diag.speed_before_mps,
                "speed_after_mps": diag.speed_after_mps,
                "reward_before": diag.reward_before,
                "reward_after": diag.reward_after,
                "mean_queue_length": queue_len,
                "mean_ego_position_m": diag.mean_ego_position_m,
                "mean_accident_dist_m": diag.mean_accident_dist_m,
                "near_accident_frac": diag.near_accident_frac,
                "near_samples": diag.near_samples,
                "KEEP_count": diag.action_keep_n,
                "LEFT_count": diag.action_left_n,
                "RIGHT_count": diag.action_right_n,
                "KEEP_ratio": diag.action_keep_ratio,
                "LEFT_ratio": diag.action_left_ratio,
                "RIGHT_ratio": diag.action_right_ratio,
                "KEEP_pct": diag.action_keep_pct,
                "LEFT_pct": diag.action_left_pct,
                "RIGHT_pct": diag.action_right_pct,
                "near_KEEP_count": diag.near_keep_n,
                "near_LEFT_count": diag.near_left_n,
                "near_RIGHT_count": diag.near_right_n,
                "near_KEEP_pct": diag.near_keep_pct,
                "near_LEFT_pct": diag.near_left_pct,
                "near_RIGHT_pct": diag.near_right_pct,
                "episode_length": ep_len,
                "actor_loss": stats.actor_loss,
                "critic_loss": stats.critic_loss,
                "entropy": stats.entropy,
                "invalid_action_rate": float(inv_rate),
                "action_mask": int(bool(use_action_mask)),
            }
            log_rows.append(row)

            if ep % log_interval == 0 or ep == 1 or ep == n_ep:
                print(
                    f"[ep {ep:04d}/{n_ep}] "
                    f"rollout_steps={batch.rollout_steps} "
                    f"mean_reward={rew_mean:.4f} "
                    f"min_pos={diag.min_pos_m:.1f} max_pos={diag.max_pos_m:.1f} "
                    f"near_frac={diag.near_accident_frac:.3f} "
                    f"num_near={diag.num_near_accident_steps} "
                    f"entered={'YES' if diag.reached_accident_zone else 'NO'}"
                )
                print(
                    f"  R_ind={diag.mean_individual_reward:.4f} "
                    f"R_grp={diag.mean_group_reward:.4f} "
                    f"actor={stats.actor_loss:.4f} "
                    f"critic={stats.critic_loss:.4f} "
                    f"ent={stats.entropy:.4f} "
                    f"invalid={inv_rate:.4f}"
                )
                print(
                    f"  Actions: KEEP={diag.action_keep_n}({diag.action_keep_pct:.1f}%) "
                    f"LEFT={diag.action_left_n}({diag.action_left_pct:.1f}%) "
                    f"RIGHT={diag.action_right_n}({diag.action_right_pct:.1f}%)"
                )
                print(
                    f"  Lifecycle: reset={life.get('env_reset_count', '?')} "
                    f"sumo_start={life.get('sumo_start_count', '?')} "
                    f"departed={life.get('departed_cav_count', '?')} "
                    f"new={life.get('new_cav_count', '?')} "
                    f"active_peak={life.get('active_cav_peak', '?')} "
                    f"rebind={life.get('rebind_events', '?')}"
                )

            if save_interval > 0 and (ep % save_interval == 0 or ep == n_ep):
                path = ckpt_dir / f"episode_{ep:04d}.pt"
                trainer.save_checkpoint(path, episode=ep, extra={"metrics": row})
                saved_paths.append(path)
                print(f"  checkpoint saved: {path}")
    finally:
        env.close()

    actor_after = _clone_params(model.actor)
    critic_after = _clone_params(model.critic)
    actor_changed = _params_changed(actor_before, actor_after)
    critic_changed = _params_changed(critic_before, critic_after)

    # CSV metrics
    if metrics_dir is not None:
        metrics_path = Path(metrics_dir)
        if not metrics_path.is_absolute():
            metrics_path = PROJECT_ROOT / metrics_path
    else:
        metrics_path = PROJECT_ROOT / "results" / "mappo_group_benefit"
    metrics_path.mkdir(parents=True, exist_ok=True)
    metrics_csv = metrics_path / "train_metrics.csv"
    if log_rows:
        with metrics_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)

    print()
    print("Parameter Check:")
    print(f"  Actor parameters changed: {'YES' if actor_changed else 'NO'}")
    print(f"  Critic parameters changed: {'YES' if critic_changed else 'NO'}")
    print(f"Metrics CSV: {metrics_csv}")
    if log_rows:
        last = log_rows[-1]
        print()
        print("Coverage Check (last update):")
        print(f"  rollout_steps = {last.get('rollout_steps')}")
        print(f"  max_pos = {last.get('max_pos'):.1f} m")
        print(f"  near_frac = {last.get('near_frac'):.4f}")
        print(f"  num_near_accident_steps = {last.get('num_near_accident_steps')}")
        entered = bool(last.get("reached_accident_zone"))
        print(f"  entered accident zone = {'YES' if entered else 'NO'}")
        cov_ok = (
            int(last.get("rollout_steps", 0)) == r_steps
            and float(last.get("max_pos", 0)) > 1200.0
            and float(last.get("near_frac", 0)) > 0.0
            and int(last.get("num_near_accident_steps", 0)) > 0
        )
        print(f"  Phase 4.4.3 coverage criteria: {'PASS' if cov_ok else 'FAIL'}")
    print("================================")

    cov_ok = False
    if log_rows:
        last = log_rows[-1]
        cov_ok = (
            int(last.get("rollout_steps", 0)) == r_steps
            and float(last.get("max_pos", 0)) > 1200.0
            and float(last.get("near_frac", 0)) > 0.0
            and int(last.get("num_near_accident_steps", 0)) > 0
        )

    return {
        "num_episodes": n_ep,
        "start_episode": start_ep,
        "resumed_from_episode": resumed_from_episode,
        "rollout_steps": r_steps,
        "device": str(device),
        "first_reward": float(first_reward or 0.0),
        "final_reward": float(last_reward),
        "actor_loss": float(last_stats.actor_loss) if last_stats else 0.0,
        "critic_loss": float(last_stats.critic_loss) if last_stats else 0.0,
        "entropy": float(last_stats.entropy) if last_stats else 0.0,
        "actor_changed": actor_changed,
        "critic_changed": critic_changed,
        "coverage_ok": cov_ok,
        "checkpoints": [str(p) for p in saved_paths],
        "metrics_csv": str(metrics_csv),
        "log_rows": log_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4.4 MAPPO training smoke run")
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    summary = train(
        num_episodes=args.num_episodes,
        rollout_steps=args.rollout_steps,
        gui=args.gui,
    )
    ok = (
        summary["actor_changed"]
        and summary["critic_changed"]
        and bool(summary.get("coverage_ok", False))
    )
    print("Phase 4.4.3 PASSED" if ok else "Phase 4.4.3 FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
