"""Traffic / training diagnostics (logging only — does not change reward)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from env.action import KEEP, LEFT, RIGHT
from env.observation import OBS_KEYS

IDX_EGO_SPEED = OBS_KEYS.index("ego_speed")
IDX_EGO_POS = OBS_KEYS.index("ego_position")
IDX_ACC_DIST = OBS_KEYS.index("accident_distance")


@dataclass
class ActionShare:
    keep: float = 0.0
    left: float = 0.0
    right: float = 0.0
    n: int = 0


@dataclass
class RolloutDiagnostics:
    """Diagnostics derived from one RolloutBatch (no reward change)."""

    reward_mean: float = 0.0
    reward_sum: float = 0.0
    reward_std: float = 0.0
    reward_min: float = 0.0
    reward_max: float = 0.0

    # Reconstructed from obs ego_speed + stored total (logging only).
    mean_individual_reward: float = 0.0
    mean_group_reward: float = 0.0
    mean_total_reward: float = 0.0

    mean_speed_mps: float = 0.0
    mean_speed_near_accident: float = 0.0
    speed_before_mps: float = 0.0
    speed_after_mps: float = 0.0
    reward_before: float = 0.0
    reward_after: float = 0.0
    n_samples_before: int = 0
    n_samples_after: int = 0

    action_keep_n: int = 0
    action_left_n: int = 0
    action_right_n: int = 0
    action_keep_pct: float = 0.0
    action_left_pct: float = 0.0
    action_right_pct: float = 0.0
    action_keep_ratio: float = 0.0
    action_left_ratio: float = 0.0
    action_right_ratio: float = 0.0

    near_keep_n: int = 0
    near_left_n: int = 0
    near_right_n: int = 0
    near_keep_pct: float = 0.0
    near_left_pct: float = 0.0
    near_right_pct: float = 0.0
    near_samples: int = 0

    near_accident_frac: float = 0.0
    num_near_accident_steps: int = 0  # alias of near_samples (valid agent-steps)
    mean_ego_position_m: float = 0.0
    min_pos_m: float = 0.0
    max_pos_m: float = 0.0
    mean_accident_dist_m: float = float("nan")
    reached_accident_zone: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _action_share(counts: np.ndarray) -> ActionShare:
    n = int(counts.sum())
    if n <= 0:
        return ActionShare()
    return ActionShare(
        keep=float(counts[KEEP]) / n,
        left=float(counts[LEFT]) / n,
        right=float(counts[RIGHT]) / n,
        n=n,
    )


def diagnose_rollout_batch(
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    agent_mask: np.ndarray | None,
    *,
    road_length_m: float = 2000.0,
    max_speed_mps: float = 30.0,
    accident_position_m: float = 1200.0,
    obs_range_m: float = 150.0,
    near_accident_m: float = 150.0,
    individual_weight: float = 0.7,
    group_weight: float = 0.3,
) -> RolloutDiagnostics:
    """Compute training diagnostics from [T,N,*] rollout arrays.

    Uses observation features only. Does **not** modify reward.

    Individual reward is reconstructed as ego_speed (obs already = v/v_max).
    Group reward is reconstructed from:
      total = w_i * ind + w_g * group
    """
    obs = np.asarray(observations, dtype=np.float64)
    act = np.asarray(actions, dtype=np.int64)
    rew = np.asarray(rewards, dtype=np.float64)
    if agent_mask is None:
        mask = np.ones(act.shape, dtype=np.float64)
    else:
        mask = np.asarray(agent_mask, dtype=np.float64)

    valid = mask > 0.5
    diag = RolloutDiagnostics()
    if not np.any(valid):
        return diag

    v_rew = rew[valid]
    diag.reward_mean = float(v_rew.mean())
    diag.reward_sum = float(v_rew.sum())
    diag.reward_std = float(v_rew.std())
    diag.reward_min = float(v_rew.min())
    diag.reward_max = float(v_rew.max())
    diag.mean_total_reward = diag.reward_mean

    # obs ego_speed is already normalized v/v_max == individual reward definition
    ind = obs[..., IDX_EGO_SPEED]
    ind_v = ind[valid]
    diag.mean_individual_reward = float(ind_v.mean())
    # Reconstruct group: total = wi*ind + wg*grp  => grp = (total - wi*ind)/wg
    wg = float(group_weight) if float(group_weight) != 0 else 0.3
    wi = float(individual_weight)
    grp = (rew - wi * ind) / wg
    diag.mean_group_reward = float(grp[valid].mean())

    ego_speed = ind * max_speed_mps
    ego_pos = obs[..., IDX_EGO_POS] * road_length_m
    acc_feat = obs[..., IDX_ACC_DIST]
    raw_dist = acc_feat * obs_range_m
    # Coverage definition (Phase 4.4.2/4.4.3): |pos - accident| <= near_m
    near = valid & (np.abs(ego_pos - accident_position_m) <= near_accident_m)

    diag.mean_speed_mps = float(ego_speed[valid].mean())
    diag.min_pos_m = float(ego_pos[valid].min())
    diag.max_pos_m = float(ego_pos[valid].max())
    if np.any(near):
        diag.mean_speed_near_accident = float(ego_speed[near].mean())

    before = valid & (ego_pos < accident_position_m)
    after = valid & (ego_pos >= accident_position_m)
    diag.n_samples_before = int(before.sum())
    diag.n_samples_after = int(after.sum())
    if diag.n_samples_before:
        diag.speed_before_mps = float(ego_speed[before].mean())
        diag.reward_before = float(rew[before].mean())
    if diag.n_samples_after:
        diag.speed_after_mps = float(ego_speed[after].mean())
        diag.reward_after = float(rew[after].mean())

    v_act = act[valid]
    counts = np.bincount(v_act.clip(0, 2), minlength=3)
    share = _action_share(counts)
    diag.action_keep_n = int(counts[KEEP])
    diag.action_left_n = int(counts[LEFT])
    diag.action_right_n = int(counts[RIGHT])
    diag.action_keep_ratio = share.keep
    diag.action_left_ratio = share.left
    diag.action_right_ratio = share.right
    diag.action_keep_pct = 100.0 * share.keep
    diag.action_left_pct = 100.0 * share.left
    diag.action_right_pct = 100.0 * share.right

    diag.near_samples = int(near.sum())
    diag.num_near_accident_steps = diag.near_samples
    diag.near_accident_frac = float(diag.near_samples) / float(valid.sum())
    diag.reached_accident_zone = diag.near_samples > 0
    if diag.near_samples:
        n_counts = np.bincount(act[near].clip(0, 2), minlength=3)
        n_share = _action_share(n_counts)
        diag.near_keep_n = int(n_counts[KEEP])
        diag.near_left_n = int(n_counts[LEFT])
        diag.near_right_n = int(n_counts[RIGHT])
        diag.near_keep_pct = 100.0 * n_share.keep
        diag.near_left_pct = 100.0 * n_share.left
        diag.near_right_pct = 100.0 * n_share.right

    diag.mean_ego_position_m = float(ego_pos[valid].mean())
    within_sensor = np.abs(acc_feat) < 1.0 - 1e-6
    if np.any(valid & within_sensor):
        diag.mean_accident_dist_m = float(raw_dist[valid & within_sensor].mean())
    else:
        diag.mean_accident_dist_m = float(
            np.sign(acc_feat[valid].mean() + 1e-9) * obs_range_m
        )

    return diag


@dataclass
class EvalMetrics:
    """Horizon traffic metrics for No Control vs MAPPO comparison."""

    mode: str = "unknown"
    sim_seconds: float = 0.0
    steps: int = 0

    average_speed_mps: float = 0.0
    average_cav_speed_mps: float = 0.0
    average_hdv_speed_mps: float = 0.0

    speed_before_mps: float = 0.0
    speed_after_mps: float = 0.0
    cav_reward_before: float = 0.0
    cav_reward_after: float = 0.0

    queue_length_mean: float = 0.0
    queue_length_max: float = 0.0
    cav_near_accident_mean: float = 0.0

    throughput: float = 0.0
    travel_time_mean_s: float = 0.0
    collisions: int = 0
    ttc_min_s: float = float("inf")
    ttc_mean_s: float = float("nan")

    action_keep_pct: float = 100.0
    action_left_pct: float = 0.0
    action_right_pct: float = 0.0
    near_keep_pct: float = 100.0
    near_left_pct: float = 0.0
    near_right_pct: float = 0.0
    near_action_n: int = 0

    reward_mean: float = 0.0

    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["notes"] = "; ".join(self.notes)
        return d


def format_action_share_line(prefix: str, keep: float, left: float, right: float) -> str:
    return f"{prefix} KEEP={keep:.1f}% LEFT={left:.1f}% RIGHT={right:.1f}%"


def compute_ttc_front(
    ego_pos: float,
    ego_speed: float,
    leader_pos: float | None,
    leader_speed: float | None,
    *,
    min_gap: float = 1.0,
) -> float | None:
    """Simple time-to-collision with front vehicle (seconds), or None."""
    if leader_pos is None or leader_speed is None:
        return None
    gap = float(leader_pos) - float(ego_pos) - float(min_gap)
    if gap <= 0:
        return 0.0
    closing = float(ego_speed) - float(leader_speed)
    if closing <= 1e-3:
        return None
    return gap / closing
