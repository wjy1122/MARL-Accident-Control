"""Reward: current Individual+Group (Phase 2.4) and Candidate A (Phase 5.3).

Current:
  R_i = w_ind * R_i^ind + w_group * R_i^group
  R_i^ind = clip(v_i / v_max, 0, 1)
  R_i^group = mean_{j in N_i} R_j^ind   (or R_i^ind if N_i empty)

Candidate A (Phase 5.2 / 5.3):
  R_i^ind   = v̄_i
  S_cav_i   = mean neighbor CAV speed ratios (fallback v̄_i)
  S_loc_i   = mean local (non-ACCIDENT) vehicle speed ratios (fallback v̄_i)
  R_i^group = α · S_cav + (1−α) · S_loc
  P_block   = z_i · b_i  (near accident AND on blocked lane)
  R_i       = clip(w_ind · R_ind + w_grp · R_group − w_b · P_block, 0, 1)
"""

from __future__ import annotations

from typing import Mapping, Sequence


def compute_individual_reward(speed: float, max_speed: float) -> float:
    """R^ind = clip(v / v_max, 0, 1)."""
    if max_speed <= 0:
        raise ValueError(f"max_speed must be > 0, got {max_speed}")
    value = float(speed) / float(max_speed)
    return float(min(1.0, max(0.0, value)))


def compute_group_reward(
    individual_rewards: Mapping[str, float],
    neighbor_ids: Sequence[str],
    ego_id: str,
) -> float:
    """Mean of neighbors' individual rewards; fallback to ego if no neighbors.

    ``neighbor_ids`` should already exclude non-CAV / out-of-range agents.
    Ego is ignored if present in ``neighbor_ids``.
    """
    if ego_id not in individual_rewards:
        raise KeyError(f"ego_id missing from individual_rewards: {ego_id}")

    ego_ind = float(individual_rewards[ego_id])
    neighbors = [
        nid
        for nid in neighbor_ids
        if nid != ego_id and nid in individual_rewards
    ]
    if not neighbors:
        return ego_ind
    return float(sum(float(individual_rewards[j]) for j in neighbors) / len(neighbors))


def compute_total_reward(
    individual_reward: float,
    group_reward: float,
    *,
    individual_weight: float = 0.7,
    group_weight: float = 0.3,
) -> float:
    """R = w_ind * R_ind + w_group * R_group."""
    total = float(individual_weight) * float(individual_reward) + float(
        group_weight
    ) * float(group_reward)
    return float(min(1.0, max(0.0, total)))


def find_cav_neighbors(
    *,
    ego_id: str,
    cav_positions: Mapping[str, float],
    neighbor_range_m: float,
) -> list[str]:
    """Other CAVs within longitudinal |Δpos| <= neighbor_range_m (any lane)."""
    if ego_id not in cav_positions:
        return []
    ego_pos = float(cav_positions[ego_id])
    out: list[str] = []
    for nid, pos in cav_positions.items():
        if nid == ego_id:
            continue
        if abs(float(pos) - ego_pos) <= float(neighbor_range_m):
            out.append(nid)
    return out


def mean_speed_ratio(
    speeds: Sequence[float],
    max_speed: float,
    *,
    fallback: float,
) -> float:
    """Mean of clip(v / v_max, 0, 1); empty → fallback."""
    if not speeds:
        return float(fallback)
    if max_speed <= 0:
        raise ValueError(f"max_speed must be > 0, got {max_speed}")
    vals = [
        float(min(1.0, max(0.0, float(v) / float(max_speed)))) for v in speeds
    ]
    return float(sum(vals) / len(vals))


def find_local_vehicle_ids(
    *,
    ego_id: str,
    positions: Mapping[str, float],
    types: Mapping[str, str],
    neighbor_range_m: float,
) -> list[str]:
    """Non-ACCIDENT vehicles within |Δpos| <= range (excludes ego)."""
    if ego_id not in positions:
        return []
    ego_pos = float(positions[ego_id])
    out: list[str] = []
    for vid, pos in positions.items():
        if vid == ego_id:
            continue
        if str(types.get(vid, "")).upper() == "ACCIDENT":
            continue
        if abs(float(pos) - ego_pos) <= float(neighbor_range_m):
            out.append(vid)
    return out


def compute_block_penalty(
    *,
    ego_position: float,
    ego_lane: int,
    accident_position_m: float,
    blocked_lanes: Sequence[int],
    zone_range_m: float,
) -> float:
    """P_block = z · b ∈ {0, 1}: near accident AND on a blocked lane."""
    z = 1.0 if abs(float(ego_position) - float(accident_position_m)) <= float(
        zone_range_m
    ) else 0.0
    b = 1.0 if int(ego_lane) in {int(x) for x in blocked_lanes} else 0.0
    return float(z * b)


def compute_candidate_a_group(
    *,
    s_cav: float,
    s_loc: float,
    alpha: float,
) -> float:
    """R^group = α · S_cav + (1−α) · S_loc."""
    a = float(alpha)
    return float(a * float(s_cav) + (1.0 - a) * float(s_loc))


def compute_candidate_a_total(
    *,
    individual_reward: float,
    group_reward: float,
    p_block: float,
    individual_weight: float,
    group_weight: float,
    block_weight: float,
    lc_event: float = 0.0,
    rev_event: float = 0.0,
    progress: float = 0.0,
    lc_weight: float = 0.0,
    rev_weight: float = 0.0,
    progress_weight: float = 0.0,
) -> float:
    """R = clip(w_ind·R_ind + w_grp·R_group − w_b·P_block
               − w_lc·I_lc − w_rev·I_rev + w_prog·progress, 0, 1).

    ``lc_event`` / ``rev_event`` are 0/1 indicators. ``progress`` is a clipped
    normalized longitudinal advance in [0, 1]. Extra terms default to 0 so
    Candidate A (Phase 5.3) is unchanged when weights are zero.
    """
    total = (
        float(individual_weight) * float(individual_reward)
        + float(group_weight) * float(group_reward)
        - float(block_weight) * float(p_block)
        - float(lc_weight) * float(lc_event)
        - float(rev_weight) * float(rev_event)
        + float(progress_weight) * float(progress)
    )
    return float(min(1.0, max(0.0, total)))
