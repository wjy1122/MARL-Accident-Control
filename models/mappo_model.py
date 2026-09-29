"""Phase 4.1: MAPPO model wrapper (shared Actor + centralized Critic).

No Attention, no PPO loss, no training loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from models.actor import Actor
from models.critic import Critic

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs" / "model.yaml"


def load_model_config(path: Path | str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path is not None else DEFAULT_MODEL_CONFIG
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class MAPPOModel(nn.Module):
    """Shared-parameter Actor for all CAVs + one centralized Critic.

              ┌── CAV_0 obs ──┐
    Shared    ├── CAV_1 obs ──┼──→ actions / log_probs
    Actor     └── CAV_N obs ──┘

    Critic(concat(obs_0, ..., obs_{num_agents-1})) → V(s)
    """

    def __init__(
        self,
        obs_dim: int = 17,
        action_dim: int = 3,
        num_agents: int = 4,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if num_agents <= 0:
            raise ValueError(f"num_agents must be positive, got {num_agents}")

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.num_agents = int(num_agents)
        self.hidden_dim = int(hidden_dim)
        self.critic_input_dim = self.obs_dim * self.num_agents

        self.actor = Actor(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
        )
        self.critic = Critic(
            input_dim=self.critic_input_dim,
            hidden_dim=self.hidden_dim,
        )

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None = None) -> MAPPOModel:
        if cfg is None:
            cfg = load_model_config()
        block = cfg.get("model", cfg)
        hidden = int(
            block.get(
                "hidden_dim",
                cfg.get("actor", {}).get("hidden_dim", 128),
            )
        )
        return cls(
            obs_dim=int(block.get("observation_dim", 17)),
            action_dim=int(block.get("action_dim", 3)),
            num_agents=int(block.get("num_agents", 4)),
            hidden_dim=hidden,
        )

    def act(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = False,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decentralized execution: one local obs → action, log_prob.

        ``action_mask`` is optional and is applied inside the actor before
        the categorical distribution. Omit it to keep the historical unmasked policy.
        """
        return self.actor.get_action(
            obs, deterministic=deterministic, action_mask=action_mask
        )

    def get_value(self, global_state: torch.Tensor) -> torch.Tensor:
        """Centralized value: concatenated obs → V(s)."""
        return self.critic(global_state)

    def build_central_state(
        self, obs_list: list[torch.Tensor] | torch.Tensor
    ) -> torch.Tensor:
        """Concatenate per-agent obs into critic input; pad/truncate to num_agents.

        Args:
            obs_list: list of [obs_dim] tensors, or tensor [n_agents, obs_dim]

        Returns:
            state: [obs_dim * num_agents]
        """
        if isinstance(obs_list, torch.Tensor):
            if obs_list.dim() != 2 or obs_list.size(-1) != self.obs_dim:
                raise ValueError(
                    f"Expected [n, {self.obs_dim}] tensor, got {tuple(obs_list.shape)}"
                )
            rows = [obs_list[i] for i in range(obs_list.size(0))]
        else:
            rows = list(obs_list)

        device = rows[0].device if rows else torch.device("cpu")
        dtype = rows[0].dtype if rows else torch.float32
        padded: list[torch.Tensor] = []
        for i in range(self.num_agents):
            if i < len(rows):
                obs_i = rows[i]
                if obs_i.numel() != self.obs_dim:
                    raise ValueError(
                        f"obs[{i}] has {obs_i.numel()} elems, expected {self.obs_dim}"
                    )
                padded.append(obs_i.reshape(self.obs_dim))
            else:
                padded.append(torch.zeros(self.obs_dim, device=device, dtype=dtype))
        return torch.cat(padded, dim=0)
