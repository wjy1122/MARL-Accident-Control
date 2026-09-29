"""Phase 4.1: centralized MAPPO Critic (CTDE training)."""

from __future__ import annotations

import torch
import torch.nn as nn


class Critic(nn.Module):
    """MLP value network: centralized state -> V(s)."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return state value.

        Args:
            state: [batch_size, input_dim] or [input_dim]
                   typically concat of all agent observations.

        Returns:
            value: [batch_size, 1] or scalar tensor [1] for single input
        """
        squeezed = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeezed = True
        if state.dim() != 2 or state.size(-1) != self.input_dim:
            raise ValueError(
                f"Critic expects state shape [B, {self.input_dim}], got {tuple(state.shape)}"
            )
        value = self.net(state)
        if squeezed:
            value = value.squeeze(0)
        return value
