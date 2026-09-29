"""Phase 4.3–4.4: MAPPO trainer (GAE + PPO update + checkpoint)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from algorithms.ppo_update import PPOUpdateStats, PPOUpdater, build_updater_from_config
from algorithms.rollout import resolve_device
from algorithms.rollout_buffer import RolloutBatch
from models.mappo_model import MAPPOModel


class MAPPOTrainer:
    """Apply GAE + PPO update; optional checkpoint I/O (no full train loop here)."""

    def __init__(
        self,
        model: MAPPOModel,
        train_cfg: dict[str, Any] | None = None,
        *,
        device: str | torch.device | None = "auto",
    ) -> None:
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.train_cfg = train_cfg or {}
        self.updater = build_updater_from_config(self.model, self.train_cfg, self.device)

    def update(self, batch: RolloutBatch) -> PPOUpdateStats:
        return self.updater.update(batch)

    def save_checkpoint(
        self,
        path: Path | str,
        *,
        episode: int,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Save actor/critic + optimizers + episode + config."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "episode": int(episode),
            "actor_state_dict": self.model.actor.state_dict(),
            "critic_state_dict": self.model.critic.state_dict(),
            "optimizer_state_dict": {
                "actor": self.updater.actor_opt.state_dict(),
                "critic": self.updater.critic_opt.state_dict(),
            },
            "config": self.train_cfg,
            "model_meta": {
                "obs_dim": self.model.obs_dim,
                "action_dim": self.model.action_dim,
                "num_agents": self.model.num_agents,
                "hidden_dim": self.model.hidden_dim,
            },
        }
        if extra:
            payload["extra"] = extra
        torch.save(payload, out)
        return out

    def load_checkpoint(self, path: Path | str, *, load_optimizers: bool = True) -> int:
        """Load weights (and optionally optimizers). Returns saved episode."""
        ckpt = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.model.actor.load_state_dict(ckpt["actor_state_dict"])
        self.model.critic.load_state_dict(ckpt["critic_state_dict"])
        if load_optimizers and "optimizer_state_dict" in ckpt:
            opt = ckpt["optimizer_state_dict"]
            if "actor" in opt:
                self.updater.actor_opt.load_state_dict(opt["actor"])
            if "critic" in opt:
                self.updater.critic_opt.load_state_dict(opt["critic"])
        return int(ckpt.get("episode", 0))
