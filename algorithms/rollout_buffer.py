"""Phase 4.2: simple rollout trajectory buffer (no GAE / no PPO)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RolloutBatch:
    """Stored trajectory arrays for one collect() call."""

    observations: np.ndarray  # [T, N, O]
    actions: np.ndarray  # [T, N]
    rewards: np.ndarray  # [T, N]
    log_probs: np.ndarray  # [T, N]
    values: np.ndarray  # [T, N]  (central V(s) broadcast per agent)
    dones: np.ndarray  # [T]
    agent_mask: np.ndarray  # [T, N] 1=real agent, 0=padding
    agent_ids: list[str | None]
    obs_dim: int
    num_agents: int
    action_mask: np.ndarray | None = None  # [T, N, A] 1=legal; None = unmasked policy
    next_values: np.ndarray | None = None  # [N] critic V(s_T) for GAE bootstrap

    @property
    def rollout_steps(self) -> int:
        return int(self.observations.shape[0])


class RolloutBuffer:
    """Fixed-capacity buffer: T timesteps × N agents."""

    def __init__(self, rollout_steps: int, num_agents: int, obs_dim: int) -> None:
        if rollout_steps <= 0 or num_agents <= 0 or obs_dim <= 0:
            raise ValueError("rollout_steps, num_agents, obs_dim must be positive")
        self.rollout_steps = int(rollout_steps)
        self.num_agents = int(num_agents)
        self.obs_dim = int(obs_dim)
        self.agent_ids: list[str | None] = [None] * self.num_agents
        self._t = 0
        self._alloc()

    def _alloc(self) -> None:
        t, n, o = self.rollout_steps, self.num_agents, self.obs_dim
        self.observations = np.zeros((t, n, o), dtype=np.float32)
        self.actions = np.zeros((t, n), dtype=np.int64)
        self.rewards = np.zeros((t, n), dtype=np.float32)
        self.log_probs = np.zeros((t, n), dtype=np.float32)
        self.values = np.zeros((t, n), dtype=np.float32)
        self.dones = np.zeros((t,), dtype=np.bool_)
        self.agent_mask = np.zeros((t, n), dtype=np.float32)
        self.action_masks = np.ones((t, n, 3), dtype=np.float32)
        self._store_action_mask = False
        self.next_values: np.ndarray | None = None
        self._t = 0

    def reset(self) -> None:
        self._alloc()

    def set_next_values(self, next_values: np.ndarray) -> None:
        nv = np.asarray(next_values, dtype=np.float32).reshape(self.num_agents)
        self.next_values = nv

    def set_agent_ids(self, agent_ids: list[str | None]) -> None:
        if len(agent_ids) != self.num_agents:
            raise ValueError(
                f"agent_ids length {len(agent_ids)} != num_agents {self.num_agents}"
            )
        self.agent_ids = list(agent_ids)

    @property
    def size(self) -> int:
        return int(self._t)

    def add(
        self,
        *,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        done: bool,
        agent_mask: np.ndarray | None = None,
        action_mask: np.ndarray | None = None,
    ) -> None:
        if self._t >= self.rollout_steps:
            raise RuntimeError("RolloutBuffer is full")
        t = self._t
        obs = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        rewards = np.asarray(rewards, dtype=np.float32)
        log_probs = np.asarray(log_probs, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)

        if obs.shape != (self.num_agents, self.obs_dim):
            raise ValueError(f"obs shape {obs.shape} != {(self.num_agents, self.obs_dim)}")
        if actions.shape != (self.num_agents,):
            raise ValueError(f"actions shape {actions.shape} != {(self.num_agents,)}")
        if rewards.shape != (self.num_agents,):
            raise ValueError(f"rewards shape {rewards.shape} != {(self.num_agents,)}")
        if log_probs.shape != (self.num_agents,):
            raise ValueError(f"log_probs shape {log_probs.shape} != {(self.num_agents,)}")
        if values.shape != (self.num_agents,):
            raise ValueError(f"values shape {values.shape} != {(self.num_agents,)}")

        if agent_mask is None:
            mask = np.ones(self.num_agents, dtype=np.float32)
        else:
            mask = np.asarray(agent_mask, dtype=np.float32).reshape(self.num_agents)

        self.observations[t] = obs
        self.actions[t] = actions
        self.rewards[t] = rewards
        self.log_probs[t] = log_probs
        self.values[t] = values
        self.dones[t] = bool(done)
        self.agent_mask[t] = mask
        if action_mask is not None:
            am = np.asarray(action_mask, dtype=np.float32)
            if am.shape != (self.num_agents, 3):
                raise ValueError(
                    f"action_mask shape {am.shape} != {(self.num_agents, 3)}"
                )
            self.action_masks[t] = am
            self._store_action_mask = True
        self._t += 1

    def get(self) -> RolloutBatch:
        if self._t != self.rollout_steps:
            raise RuntimeError(
                f"Buffer incomplete: {self._t}/{self.rollout_steps} steps stored"
            )
        return RolloutBatch(
            observations=self.observations.copy(),
            actions=self.actions.copy(),
            rewards=self.rewards.copy(),
            log_probs=self.log_probs.copy(),
            values=self.values.copy(),
            dones=self.dones.copy(),
            agent_mask=self.agent_mask.copy(),
            agent_ids=list(self.agent_ids),
            obs_dim=self.obs_dim,
            num_agents=self.num_agents,
            action_mask=self.action_masks.copy() if self._store_action_mask else None,
            next_values=None if self.next_values is None else self.next_values.copy(),
        )

    def as_dict(self) -> dict[str, Any]:
        batch = self.get()
        return {
            "observations": batch.observations,
            "actions": batch.actions,
            "rewards": batch.rewards,
            "log_probs": batch.log_probs,
            "values": batch.values,
            "dones": batch.dones,
            "agent_mask": batch.agent_mask,
            "agent_ids": batch.agent_ids,
        }
