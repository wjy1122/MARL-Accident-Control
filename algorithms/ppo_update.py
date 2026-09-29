"""Phase 4.3: PPO update for shared Actor + centralized Critic (MAPPO)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from algorithms.gae import compute_gae
from algorithms.rollout_buffer import RolloutBatch
from models.mappo_model import MAPPOModel


@dataclass
class PPOUpdateStats:
    actor_loss: float
    critic_loss: float
    entropy: float
    total_loss: float
    approx_kl: float
    clip_fraction: float


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.reshape_as(x)
    denom = mask.sum().clamp_min(1.0)
    return (x * mask).sum() / denom


def normalize_advantages(
    advantages: np.ndarray, agent_mask: np.ndarray | None = None
) -> np.ndarray:
    adv = np.asarray(advantages, dtype=np.float32)
    if agent_mask is None:
        mean = float(adv.mean())
        std = float(adv.std())
    else:
        mask = np.asarray(agent_mask, dtype=np.float32)
        valid = adv[mask > 0.5]
        if valid.size == 0:
            return adv
        mean = float(valid.mean())
        std = float(valid.std())
    return (adv - mean) / (std + 1e-8)


class PPOUpdater:
    """One MAPPO/PPO update pass over a RolloutBatch (no full training loop)."""

    def __init__(
        self,
        model: MAPPOModel,
        *,
        clip_eps: float = 0.2,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

        self.clip_eps = float(clip_eps)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.ppo_epochs = int(ppo_epochs)
        self.minibatch_size = int(minibatch_size)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)

        self.actor_opt = torch.optim.Adam(self.model.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.model.critic.parameters(), lr=critic_lr)

    def update(self, batch: RolloutBatch) -> PPOUpdateStats:
        obs = batch.observations
        actions = batch.actions
        rewards = batch.rewards
        old_log_probs = batch.log_probs
        values = batch.values
        dones = batch.dones
        mask = getattr(batch, "agent_mask", None)
        if mask is None:
            mask = np.ones(actions.shape, dtype=np.float32)

        advantages, returns = compute_gae(
            rewards,
            values,
            dones,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            next_values=getattr(batch, "next_values", None),
            agent_mask=mask,
        )
        advantages = normalize_advantages(advantages, mask)

        t_steps, n_agents, obs_dim = obs.shape
        # Flatten agent samples; keep slot order (t major, then agent index).
        obs_flat = obs.reshape(t_steps * n_agents, obs_dim)
        act_flat = actions.reshape(t_steps * n_agents)
        old_lp_flat = old_log_probs.reshape(t_steps * n_agents)
        adv_flat = advantages.reshape(t_steps * n_agents)
        ret_flat = returns.reshape(t_steps * n_agents)
        mask_flat = mask.reshape(t_steps * n_agents)
        am = getattr(batch, "action_mask", None)
        am_flat = None
        if am is not None:
            am_flat = np.asarray(am, dtype=np.float32).reshape(t_steps * n_agents, -1)

        # Central state per timestep for critic: [T, obs_dim * N]
        central = np.zeros(
            (t_steps, self.model.critic_input_dim), dtype=np.float32
        )
        for t in range(t_steps):
            central[t] = (
                self.model.build_central_state(
                    torch.as_tensor(obs[t], dtype=torch.float32)
                )
                .detach()
                .cpu()
                .numpy()
            )

        # Only valid agent indices for actor minibatches
        valid_idx = np.flatnonzero(mask_flat > 0.5)
        if valid_idx.size == 0:
            raise RuntimeError("No valid agent samples in rollout batch")

        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []
        total_losses: list[float] = []
        approx_kls: list[float] = []
        clip_fracs: list[float] = []

        self.model.train()
        for _ in range(self.ppo_epochs):
            np.random.shuffle(valid_idx)
            for start in range(0, valid_idx.size, self.minibatch_size):
                mb_idx = valid_idx[start : start + self.minibatch_size]
                if mb_idx.size == 0:
                    continue

                mb_obs = torch.as_tensor(
                    obs_flat[mb_idx], dtype=torch.float32, device=self.device
                )
                mb_act = torch.as_tensor(
                    act_flat[mb_idx], dtype=torch.int64, device=self.device
                )
                mb_old_lp = torch.as_tensor(
                    old_lp_flat[mb_idx], dtype=torch.float32, device=self.device
                )
                mb_adv = torch.as_tensor(
                    adv_flat[mb_idx], dtype=torch.float32, device=self.device
                )
                mb_mask = torch.as_tensor(
                    mask_flat[mb_idx], dtype=torch.float32, device=self.device
                )
                mb_am = None
                if am_flat is not None:
                    mb_am = torch.as_tensor(
                        am_flat[mb_idx], dtype=torch.float32, device=self.device
                    )

                new_lp, entropy = self.model.actor.evaluate_actions(
                    mb_obs, mb_act, action_mask=mb_am
                )
                ratio = torch.exp(new_lp - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    * mb_adv
                )
                actor_obj = torch.min(surr1, surr2)
                actor_loss = -_masked_mean(actor_obj, mb_mask)
                entropy_mean = _masked_mean(entropy, mb_mask)
                policy_loss = actor_loss - self.entropy_coef * entropy_mean

                self.actor_opt.zero_grad(set_to_none=True)
                policy_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.actor.parameters(), self.max_grad_norm
                )
                self.actor_opt.step()

                with torch.no_grad():
                    approx_kl = _masked_mean(mb_old_lp - new_lp, mb_mask).abs()
                    clip_frac = _masked_mean(
                        (torch.abs(ratio - 1.0) > self.clip_eps).float(), mb_mask
                    )

                # Critic minibatch over corresponding timesteps of these samples
                t_idx = (mb_idx // n_agents).astype(np.int64)
                # unique timesteps in this minibatch
                uniq_t = np.unique(t_idx)
                mb_central = torch.as_tensor(
                    central[uniq_t], dtype=torch.float32, device=self.device
                )
                # Map each uniq_t -> returns target: masked mean over agents
                targets = []
                for t in uniq_t:
                    m = mask[t] > 0.5
                    if np.any(m):
                        targets.append(float(returns[t, m].mean()))
                    else:
                        targets.append(float(returns[t].mean()))
                mb_ret = torch.as_tensor(
                    targets, dtype=torch.float32, device=self.device
                ).view(-1, 1)

                value_pred = self.model.critic(mb_central)
                if value_pred.dim() == 1:
                    value_pred = value_pred.view(-1, 1)
                critic_loss = torch.mean((value_pred - mb_ret) ** 2)

                self.critic_opt.zero_grad(set_to_none=True)
                (self.value_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(
                    self.model.critic.parameters(), self.max_grad_norm
                )
                self.critic_opt.step()

                total = (
                    actor_loss.detach()
                    + self.value_coef * critic_loss.detach()
                    - self.entropy_coef * entropy_mean.detach()
                )
                actor_losses.append(float(actor_loss.detach().cpu()))
                critic_losses.append(float(critic_loss.detach().cpu()))
                entropies.append(float(entropy_mean.detach().cpu()))
                total_losses.append(float(total.cpu()))
                approx_kls.append(float(approx_kl.cpu()))
                clip_fracs.append(float(clip_frac.cpu()))

        def _mean(xs: list[float]) -> float:
            return float(sum(xs) / max(len(xs), 1))

        return PPOUpdateStats(
            actor_loss=_mean(actor_losses),
            critic_loss=_mean(critic_losses),
            entropy=_mean(entropies),
            total_loss=_mean(total_losses),
            approx_kl=_mean(approx_kls),
            clip_fraction=_mean(clip_fracs),
        )


def build_updater_from_config(
    model: MAPPOModel, train_cfg: dict[str, Any], device: torch.device | str
) -> PPOUpdater:
    block = train_cfg.get("train", train_cfg)
    clip = float(block.get("clip_eps", block.get("clip_epsilon", 0.2)))
    return PPOUpdater(
        model,
        clip_eps=clip,
        actor_lr=float(block.get("actor_lr", block.get("lr", 3e-4))),
        critic_lr=float(block.get("critic_lr", 1e-3)),
        entropy_coef=float(block.get("entropy_coef", 0.01)),
        value_coef=float(block.get("value_coef", 0.5)),
        max_grad_norm=float(block.get("max_grad_norm", 0.5)),
        ppo_epochs=int(block.get("ppo_epochs", 4)),
        minibatch_size=int(block.get("minibatch_size", 64)),
        gamma=float(block.get("gamma", 0.99)),
        gae_lambda=float(block.get("gae_lambda", 0.95)),
        device=device,
    )
