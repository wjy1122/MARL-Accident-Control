"""Phase 4.1: shared MAPPO Actor (decentralized execution)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from env.action_mask import apply_logit_mask


class Actor(nn.Module):
    """MLP policy: obs -> categorical logits over discrete actions.

    All CAV agents share one Actor instance / one parameter set.
    """

    def __init__(
        self,
        obs_dim: int = 17,
        action_dim: int = 3,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if obs_dim <= 0 or action_dim <= 0 or hidden_dim <= 0:
            raise ValueError("obs_dim, action_dim, hidden_dim must be positive")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return action logits.

        Args:
            obs: [batch_size, obs_dim] or [obs_dim]

        Returns:
            logits: [batch_size, action_dim] or [action_dim]
        """
        squeezed = False
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeezed = True
        if obs.dim() != 2 or obs.size(-1) != self.obs_dim:
            raise ValueError(
                f"Actor expects obs shape [B, {self.obs_dim}], got {tuple(obs.shape)}"
            )
        logits = self.net(obs)
        if squeezed:
            logits = logits.squeeze(0)
        return logits

    def get_action(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = False,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample (or greedily pick) an action and its log-probability.

        ``action_mask`` is optional. 1/True = legal. When given, logits are
        masked before Categorical / argmax. Illegal actions are not rewritten
        to KEEP; they are removed from the distribution.

        Returns:
            action: int tensor with values in {0,1,2}, shape [B] or scalar
            log_prob: matching log π(a|s) under the same (masked) distribution
        """
        logits = self.forward(obs)
        single = logits.dim() == 1
        if single:
            logits = logits.unsqueeze(0)
        logits = apply_logit_mask(logits, action_mask)

        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)

        if single:
            return action.squeeze(0), log_prob.squeeze(0)
        return action, log_prob

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log π(a|s) and entropy for given actions (PPO).

        Args:
            obs: [batch, obs_dim]
            actions: [batch] int64
            action_mask: optional [batch, action_dim], 1/True = legal.
                Must be the mask that was used when ``actions`` were sampled.

        Returns:
            log_probs: [batch]
            entropy: [batch]
        """
        if obs.dim() != 2 or obs.size(-1) != self.obs_dim:
            raise ValueError(
                f"evaluate_actions expects obs [B, {self.obs_dim}], got {tuple(obs.shape)}"
            )
        actions = actions.long().view(-1)
        logits = apply_logit_mask(self.forward(obs), action_mask)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()
