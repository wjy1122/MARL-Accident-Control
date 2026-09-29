"""Phase 4.2: MAPPO rollout collector (no parameter updates)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from algorithms.rollout_buffer import RolloutBatch, RolloutBuffer
from env.action import KEEP
from env.action_mask import legal_action_mask
from env.highway_accident_env import HighwayAccidentEnv
from models.mappo_model import MAPPOModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_train_config(path: Path | str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path is not None else PROJECT_ROOT / "configs" / "train.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _lock_agent_ids(env: HighwayAccidentEnv, num_agents: int) -> list[str | None]:
    """Stable sorted CAV IDs, padded with None to num_agents."""
    ids = sorted(env.cav_ids)[:num_agents]
    while len(ids) < num_agents:
        ids.append(None)
    return ids


def _rebind_agent_slots(
    env: HighwayAccidentEnv,
    agent_ids: list[str | None],
    num_agents: int,
) -> tuple[list[str | None], int, int]:
    """Fixed-N slot lifecycle: drop departed CAVs, fill vacancies with new CAVs.

    Does **not** restart SUMO. Tensor layout stays [N] with None → masked zeros.
    Returns (new_ids, n_departed_slots, n_newly_bound).
    """
    present = set(env.cav_ids)
    result: list[str | None] = list(agent_ids)
    if len(result) != num_agents:
        result = (result + [None] * num_agents)[:num_agents]

    departed = 0
    for i, aid in enumerate(result):
        if aid is not None and aid not in present:
            result[i] = None
            departed += 1

    occupied = {a for a in result if a is not None}
    available = [c for c in sorted(present) if c not in occupied]
    newly_bound = 0
    for i in range(num_agents):
        if result[i] is None and available:
            result[i] = available.pop(0)
            newly_bound += 1
    return result, departed, newly_bound


def _obs_matrix(
    env: HighwayAccidentEnv,
    agent_ids: list[str | None],
    obs_dim: int,
) -> np.ndarray:
    """Build [N, O] obs in fixed agent order; missing agents → zeros."""
    obs_dict = env.get_observations()
    rows: list[np.ndarray] = []
    for aid in agent_ids:
        if aid is not None and aid in obs_dict:
            rows.append(np.asarray(obs_dict[aid], dtype=np.float32).reshape(obs_dim))
        else:
            rows.append(np.zeros(obs_dim, dtype=np.float32))
    return np.stack(rows, axis=0)


class RolloutCollector:
    """Collect trajectories with shared Actor + centralized Critic (no training)."""

    def __init__(
        self,
        *,
        num_agents: int,
        obs_dim: int,
        rollout_steps: int = 128,
        warmup_steps: int = 50,
        device: str | torch.device | None = "auto",
        use_action_mask: bool = False,
    ) -> None:
        self.num_agents = int(num_agents)
        self.obs_dim = int(obs_dim)
        self.rollout_steps = int(rollout_steps)
        self.warmup_steps = int(warmup_steps)
        self.device = resolve_device(device)
        self.use_action_mask = bool(use_action_mask)
        # Filled by the latest collect() for diagnostics.
        self.last_lifecycle: dict[str, Any] = {}

    def _warmup(self, env: HighwayAccidentEnv) -> None:
        """Advance legacy steps until at least one CAV exists (or warmup budget)."""
        for _ in range(self.warmup_steps):
            if len(env.cav_ids) > 0:
                return
            env.step()  # legacy snapshot step

    def _snapshot_params(self, model: MAPPOModel) -> list[torch.Tensor]:
        return [p.detach().cpu().clone() for p in model.parameters()]

    @staticmethod
    def params_equal(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
        if len(a) != len(b):
            return False
        return all(torch.equal(x, y) for x, y in zip(a, b))

    def _ensure_sumo(self, env: HighwayAccidentEnv, stats: dict[str, int]) -> None:
        """Start SUMO only if not already connected (agent death must not call this)."""
        if not env.is_connected:
            env.reset()
            stats["env_reset_count"] += 1
            stats["sumo_start_count"] += 1

    @torch.no_grad()
    def collect(
        self,
        env: HighwayAccidentEnv,
        model: MAPPOModel,
        *,
        rollout_steps: int | None = None,
    ) -> RolloutBatch:
        """Run env ↔ model for ``rollout_steps`` and return a filled buffer batch.

        Does **not** update model parameters.

        Agent lifecycle (fixed N slots):
          - CAV leave road → clear slot (mask=0), no env.reset
          - New CAV enter → bind to vacant slot
          - env.reset only if SUMO not connected, or episode truncated/terminated
        """
        steps = int(self.rollout_steps if rollout_steps is None else rollout_steps)
        if model.num_agents != self.num_agents or model.obs_dim != self.obs_dim:
            raise ValueError(
                "Model dims mismatch collector: "
                f"model(N={model.num_agents},O={model.obs_dim}) vs "
                f"collector(N={self.num_agents},O={self.obs_dim})"
            )

        model = model.to(self.device)
        model.eval()

        buffer = RolloutBuffer(steps, self.num_agents, self.obs_dim)
        life: dict[str, int] = {
            "env_reset_count": 0,
            "sumo_start_count": 0,
            "departed_cav_count": 0,
            "new_cav_count": 0,
            "active_cav_peak": 0,
            "active_slot_peak": 0,
            "rebind_events": 0,
        }

        self._ensure_sumo(env, life)
        self._warmup(env)
        agent_ids = _lock_agent_ids(env, self.num_agents)
        life["new_cav_count"] += sum(1 for a in agent_ids if a is not None)
        buffer.set_agent_ids(list(agent_ids))

        for _ in range(steps):
            # Lifecycle: drop departed / fill vacant slots — never reset for this.
            agent_ids, n_dep, n_new = _rebind_agent_slots(
                env, agent_ids, self.num_agents
            )
            if n_dep or n_new:
                life["departed_cav_count"] += n_dep
                life["new_cav_count"] += n_new
                life["rebind_events"] += 1

            n_active_road = len(env.cav_ids)
            n_active_slots = sum(1 for a in agent_ids if a is not None)
            life["active_cav_peak"] = max(life["active_cav_peak"], n_active_road)
            life["active_slot_peak"] = max(life["active_slot_peak"], n_active_slots)

            # No CAVs yet: advance SUMO without restarting (wait for inflow).
            if n_active_road == 0:
                _obs_e, _rew_e, terminated_e, truncated_e, _info_e = env.step({})
                del _obs_e, _rew_e, _info_e
                obs_np = np.zeros((self.num_agents, self.obs_dim), dtype=np.float32)
                actions_np = np.zeros(self.num_agents, dtype=np.int64)
                logp_np = np.zeros(self.num_agents, dtype=np.float32)
                values_np = np.zeros(self.num_agents, dtype=np.float32)
                rewards_np = np.zeros(self.num_agents, dtype=np.float32)
                mask_np = np.zeros(self.num_agents, dtype=np.float32)
                done = bool(terminated_e or truncated_e)
                add_kwargs: dict[str, Any] = {}
                if self.use_action_mask:
                    add_kwargs["action_mask"] = np.ones(
                        (self.num_agents, 3), dtype=np.float32
                    )
                buffer.add(
                    obs=obs_np,
                    actions=actions_np,
                    rewards=rewards_np,
                    log_probs=logp_np,
                    values=values_np,
                    done=done,
                    agent_mask=mask_np,
                    **add_kwargs,
                )
                if done:
                    env.reset()
                    life["env_reset_count"] += 1
                    life["sumo_start_count"] += 1
                    self._warmup(env)
                    agent_ids = _lock_agent_ids(env, self.num_agents)
                    life["new_cav_count"] += sum(
                        1 for a in agent_ids if a is not None
                    )
                continue

            obs_np = _obs_matrix(env, agent_ids, self.obs_dim)
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)

            actions_np = np.zeros(self.num_agents, dtype=np.int64)
            logp_np = np.zeros(self.num_agents, dtype=np.float32)
            action_mask_np = np.ones((self.num_agents, 3), dtype=np.float32)
            actions_dict: dict[str, int] = {}
            lane_by_id: dict[str, int] = {}
            if self.use_action_mask:
                lane_by_id = {
                    cid: int(st["lane"]) for cid, st in env.get_cav_states().items()
                }

            for i, aid in enumerate(agent_ids):
                if aid is None or aid not in env.cav_ids:
                    actions_np[i] = KEEP
                    logp_np[i] = 0.0
                    continue
                am_t = None
                if self.use_action_mask:
                    if hasattr(env, "get_action_mask"):
                        action_mask_np[i] = env.get_action_mask(aid)
                    else:
                        lane = int(lane_by_id.get(aid, 1))
                        action_mask_np[i] = legal_action_mask(
                            lane,
                            n_lanes=int(env.n_lanes),
                            left_increases_index=bool(env.left_increases_index),
                        )
                    am_t = torch.as_tensor(
                        action_mask_np[i], dtype=torch.float32, device=self.device
                    )
                action_i, logp_i = model.act(obs_t[i], action_mask=am_t)
                a = int(action_i.item())
                actions_np[i] = a
                logp_np[i] = float(logp_i.item())
                actions_dict[aid] = a

            central = model.build_central_state(obs_t)
            value = model.get_value(central)
            v_scalar = float(value.reshape(-1)[0].item())
            values_np = np.full(self.num_agents, v_scalar, dtype=np.float32)

            _obs_next, rewards_dict, terminated, truncated, _info = env.step(
                actions_dict
            )
            del _obs_next

            rewards_np = np.zeros(self.num_agents, dtype=np.float32)
            mask_np = np.zeros(self.num_agents, dtype=np.float32)
            for i, aid in enumerate(agent_ids):
                if aid is not None and aid in rewards_dict:
                    rewards_np[i] = float(rewards_dict[aid])
                if aid is not None and aid in env.cav_ids:
                    mask_np[i] = 1.0

            done = bool(terminated or truncated)
            add_kwargs = {}
            if self.use_action_mask:
                add_kwargs["action_mask"] = action_mask_np
            buffer.add(
                obs=obs_np,
                actions=actions_np,
                rewards=rewards_np,
                log_probs=logp_np,
                values=values_np,
                done=done,
                agent_mask=mask_np,
                **add_kwargs,
            )

            # Episode truncate/terminate → real env reset (not agent departure).
            if done:
                env.reset()
                life["env_reset_count"] += 1
                life["sumo_start_count"] += 1
                self._warmup(env)
                agent_ids = _lock_agent_ids(env, self.num_agents)
                life["new_cav_count"] += sum(1 for a in agent_ids if a is not None)

        # Critic bootstrap at s_T (after last transition), not zeros.
        try:
            agent_ids, _d, _n = _rebind_agent_slots(env, agent_ids, self.num_agents)
            obs_end = _obs_matrix(env, agent_ids, self.obs_dim)
            obs_end_t = torch.as_tensor(obs_end, dtype=torch.float32, device=self.device)
            central_end = model.build_central_state(obs_end_t)
            v_end = float(model.get_value(central_end).reshape(-1)[0].item())
            # If last stored step was episode-terminal, bootstrap must be 0.
            if bool(buffer.dones[buffer.size - 1]):
                buffer.set_next_values(np.zeros(self.num_agents, dtype=np.float32))
            else:
                buffer.set_next_values(
                    np.full(self.num_agents, v_end, dtype=np.float32)
                )
        except Exception:  # noqa: BLE001
            buffer.set_next_values(np.zeros(self.num_agents, dtype=np.float32))

        self.last_lifecycle = dict(life)
        return buffer.get()
