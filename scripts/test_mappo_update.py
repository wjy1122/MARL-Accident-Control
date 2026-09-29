"""Phase 4.3 demo script: rollout → GAE → PPO update, print param change.

Example:
  .\\.venv\\Scripts\\python.exe scripts\\test_mappo_update.py
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.gae import compute_gae  # noqa: E402
from algorithms.mappo_trainer import MAPPOTrainer  # noqa: E402
from algorithms.ppo_update import normalize_advantages  # noqa: E402
from algorithms.rollout import RolloutCollector, load_train_config, resolve_device  # noqa: E402
from env.highway_accident_env import HighwayAccidentEnv, load_config  # noqa: E402
from models.mappo_model import MAPPOModel, load_model_config  # noqa: E402


def _clone_params(module: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().cpu().clone() for p in module.parameters()]


def _changed(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    return any(not torch.equal(x, y) for x, y in zip(a, b))


def main() -> int:
    train_cfg = load_train_config()
    model_cfg = load_model_config()
    mblock = model_cfg.get("model", model_cfg)
    tblock = train_cfg.get("train", train_cfg)

    device = resolve_device(train_cfg.get("device", "auto"))
    num_agents = int(mblock.get("num_agents", 4))
    obs_dim = int(mblock.get("observation_dim", 17))
    action_dim = int(mblock.get("action_dim", 3))
    hidden = int(mblock.get("hidden_dim", 128))

    print("================================")
    print("Phase 4.3 MAPPO Update Test")
    print("================================")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    env_cfg = copy.deepcopy(load_config())
    env_cfg.setdefault("episode", {})["max_episode_steps"] = 10_000
    env = HighwayAccidentEnv(config=env_cfg, gui=False)
    model = MAPPOModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_agents=num_agents,
        hidden_dim=hidden,
    ).to(device)

    collector = RolloutCollector(
        num_agents=num_agents,
        obs_dim=obs_dim,
        rollout_steps=8,
        warmup_steps=int(tblock.get("warmup_steps", 50)),
        device=device,
    )

    train_local = copy.deepcopy(train_cfg)
    train_local.setdefault("train", {})
    train_local["train"]["ppo_epochs"] = 4
    train_local["train"]["minibatch_size"] = 8

    actor_before = _clone_params(model.actor)
    critic_before = _clone_params(model.critic)

    try:
        batch = collector.collect(env, model, rollout_steps=8)
    finally:
        env.close()

    print("Rollout:")
    print(f"  observations: {batch.observations.shape}")
    print(f"  actions:      {batch.actions.shape}")
    print(f"  rewards:      {batch.rewards.shape}")
    print(f"  values:       {batch.values.shape}")
    print()

    adv, ret = compute_gae(
        batch.rewards,
        batch.values,
        batch.dones,
        gamma=float(tblock.get("gamma", 0.99)),
        gae_lambda=float(tblock.get("gae_lambda", 0.95)),
        agent_mask=batch.agent_mask,
    )
    adv_n = normalize_advantages(adv, batch.agent_mask)
    print("GAE:")
    print(f"  advantages:   {adv_n.shape}")
    print(f"  returns:      {ret.shape}")
    print()

    trainer = MAPPOTrainer(model, train_local, device=device)
    stats = trainer.update(batch)

    actor_after = _clone_params(model.actor)
    critic_after = _clone_params(model.critic)
    actor_chg = _changed(actor_before, actor_after)
    critic_chg = _changed(critic_before, critic_after)

    vals = [stats.actor_loss, stats.critic_loss, stats.entropy, stats.total_loss]
    has_nan = any(math.isnan(v) for v in vals)
    has_inf = any(math.isinf(v) for v in vals)

    print("PPO:")
    print(f"  actor_loss:   {stats.actor_loss:.6f}")
    print(f"  critic_loss:  {stats.critic_loss:.6f}")
    print(f"  entropy:      {stats.entropy:.6f}")
    print()
    print("Parameter Check:")
    print(f"  Actor parameters changed: {'YES' if actor_chg else 'NO'}")
    print(f"  Critic parameters changed: {'YES' if critic_chg else 'NO'}")
    print()
    print(f"NaN detected: {'YES' if has_nan else 'NO'}")
    print(f"Inf detected: {'YES' if has_inf else 'NO'}")
    print()

    ok = actor_chg and critic_chg and not has_nan and not has_inf
    print("Phase 4.3 PASSED" if ok else "Phase 4.3 FAILED")
    print("================================")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
