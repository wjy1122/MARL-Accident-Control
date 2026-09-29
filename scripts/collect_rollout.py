"""Phase 4.2 demo: collect one MAPPO rollout (no training).

Example:
  .\\.venv\\Scripts\\python.exe scripts\\collect_rollout.py
  .\\.venv\\Scripts\\python.exe scripts\\collect_rollout.py --rollout-steps 64
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.rollout import RolloutCollector, load_train_config, resolve_device  # noqa: E402
from env.action import ACTION_NAMES  # noqa: E402
from env.highway_accident_env import HighwayAccidentEnv, load_config  # noqa: E402
from models.mappo_model import MAPPOModel, load_model_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect one MAPPO rollout (no train)")
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    train_cfg = load_train_config()
    model_cfg = load_model_config()
    mblock = model_cfg.get("model", model_cfg)
    tblock = train_cfg.get("train", train_cfg)

    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    device_name = args.device or train_cfg.get("device", "auto")
    device = resolve_device(device_name)
    rollout_steps = int(
        args.rollout_steps
        if args.rollout_steps is not None
        else tblock.get("rollout_steps", 128)
    )
    warmup_steps = int(
        args.warmup_steps
        if args.warmup_steps is not None
        else tblock.get("warmup_steps", 50)
    )

    num_agents = int(mblock.get("num_agents", 4))
    obs_dim = int(mblock.get("observation_dim", 17))
    action_dim = int(mblock.get("action_dim", 3))
    hidden = int(mblock.get("hidden_dim", 128))

    print("================================")
    print("MAPPO Rollout Demo")
    print("================================")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Seed: {seed}")
    print(f"Rollout steps: {rollout_steps}")
    print(f"num_agents: {num_agents}  obs_dim: {obs_dim}")
    print()

    env_cfg = copy.deepcopy(load_config())
    env_cfg.setdefault("episode", {})["max_episode_steps"] = max(
        10_000, rollout_steps * 10
    )
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
        rollout_steps=rollout_steps,
        warmup_steps=warmup_steps,
        device=device,
    )

    before = [p.detach().cpu().clone() for p in model.parameters()]
    try:
        batch = collector.collect(env, model, rollout_steps=rollout_steps)
    finally:
        env.close()
    after = [p.detach().cpu().clone() for p in model.parameters()]
    updated = not all(torch.equal(a, b) for a, b in zip(before, after))

    print("Agents (first lock / slot order):")
    for i, aid in enumerate(batch.agent_ids):
        print(f"  slot {i}: {aid}")
    print()
    print(f"Observation shape: {batch.observations.shape}")
    print(f"Action shape:      {batch.actions.shape}")
    print(f"Reward shape:      {batch.rewards.shape}")
    print(f"LogProb shape:     {batch.log_probs.shape}")
    print(f"Value shape:       {batch.values.shape}")
    print(f"Dones shape:       {batch.dones.shape}")
    print()
    print(f"Mean reward: {float(batch.rewards.mean()):.3f}")
    print(f"Mean value:  {float(batch.values.mean()):.3f}")
    print()

    flat = batch.actions.ravel()
    total = max(len(flat), 1)
    print("Action distribution:")
    for a in range(action_dim):
        pct = 100.0 * float(np.mean(flat == a))
        print(f"{ACTION_NAMES.get(a, str(a)):5s}: {pct:.1f}%")
    print()
    print(f"Model parameters updated: {'YES' if updated else 'NO'}")
    print("================================")
    return 0 if not updated else 1


if __name__ == "__main__":
    raise SystemExit(main())
