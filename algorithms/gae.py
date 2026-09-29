"""Phase 4.3: Generalized Advantage Estimation (per-agent)."""

from __future__ import annotations

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    next_values: np.ndarray | None = None,
    agent_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-agent GAE advantages and returns.

    Args:
        rewards: [T, N]
        values: [T, N]
        dones: [T]  — True if episode ended after timestep t
        gamma, gae_lambda: GAE hyperparameters
        next_values: [N] bootstrap V(s_T); default zeros
        agent_mask: optional [T, N] (1=valid). Vacant slots get advantage 0 and
                    reset the GAE carry so they do not pollute active steps.

    Returns:
        advantages: [T, N]
        returns: [T, N]  (= advantages + values)
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.bool_)

    if rewards.ndim != 2 or values.ndim != 2:
        raise ValueError("rewards and values must be [T, N]")
    if rewards.shape != values.shape:
        raise ValueError(f"shape mismatch rewards {rewards.shape} vs values {values.shape}")
    t_steps, n_agents = rewards.shape
    if dones.shape != (t_steps,):
        raise ValueError(f"dones must be [T]={t_steps}, got {dones.shape}")

    if next_values is None:
        next_values = np.zeros(n_agents, dtype=np.float64)
    else:
        next_values = np.asarray(next_values, dtype=np.float64).reshape(n_agents)

    if agent_mask is None:
        mask = np.ones((t_steps, n_agents), dtype=np.float64)
    else:
        mask = np.asarray(agent_mask, dtype=np.float64)
        if mask.shape != (t_steps, n_agents):
            raise ValueError(f"agent_mask must be [T,N]={rewards.shape}, got {mask.shape}")

    advantages = np.zeros_like(rewards, dtype=np.float64)
    last_gae = np.zeros(n_agents, dtype=np.float64)

    for t in reversed(range(t_steps)):
        if t == t_steps - 1:
            next_nonterminal = 1.0 - float(dones[t])
            next_v = next_values
            next_m = np.ones(n_agents, dtype=np.float64)
        else:
            next_nonterminal = 1.0 - float(dones[t])
            next_v = values[t + 1]
            next_m = mask[t + 1]

        m_t = mask[t]
        # Vacant slots: zero advantage and reset carry (no credit through padding).
        vacant = m_t < 0.5
        if np.any(vacant):
            last_gae = np.where(vacant, 0.0, last_gae)

        # If the next slot step is vacant (or episode done), cut bootstrap for that slot.
        slot_continue = next_nonterminal * (next_m >= 0.5).astype(np.float64)

        delta = rewards[t] + gamma * next_v * slot_continue - values[t]
        last_gae = delta + gamma * gae_lambda * slot_continue * last_gae
        last_gae = np.where(vacant, 0.0, last_gae)
        advantages[t] = last_gae

    returns = advantages + values

    advantages_f = advantages.astype(np.float32)
    returns_f = returns.astype(np.float32)

    return advantages_f, returns_f
