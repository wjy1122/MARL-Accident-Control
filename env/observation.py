"""Phase 2.3: local CAV observation builder (fixed-dim, normalized)."""

from __future__ import annotations

from typing import Any

import numpy as np

# Fixed layout for all CAV agents (Phase 2.3 first version).
OBS_KEYS: list[str] = [
    "ego_speed",
    "ego_lane",
    "ego_position",
    "front_distance",
    "front_relative_speed",
    "rear_distance",
    "rear_relative_speed",
    "left_front_distance",
    "left_front_relative_speed",
    "left_rear_distance",
    "left_rear_relative_speed",
    "right_front_distance",
    "right_front_relative_speed",
    "right_rear_distance",
    "right_rear_relative_speed",
    "accident_distance",
    "accident_lane",
]
OBS_DIM: int = len(OBS_KEYS)  # 17


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _clip11(x: float) -> float:
    return float(np.clip(x, -1.0, 1.0))


def _observation_params(cfg: dict[str, Any]) -> dict[str, float | int]:
    hw = cfg.get("highway", {})
    obs = cfg.get("observation", {})
    max_speed = float(
        obs.get(
            "max_speed_mps",
            hw.get("speed_limit_mps", 30.0),
        )
    )
    return {
        "range_m": float(obs.get("range_m", 150.0)),
        "max_speed_mps": max_speed,
        "rel_speed_scale_mps": float(obs.get("rel_speed_scale_mps", max_speed)),
        "road_length_m": float(hw.get("length_m", 2000.0)),
        "n_lanes": int(hw.get("lanes", 3)),
        "accident_position_m": float(cfg.get("accident", {}).get("position_m", 1200.0)),
        # Bitmask of blocked lanes (bit i => lane i blocked), for multi-lane blocks.
        "blocked_lane_mask": int(
            sum(1 << int(i) for i in cfg.get("accident", {}).get("blocked_lane_indices", [0]))
        ),
    }


def _nearest_on_lane(
    vehicles: dict[str, dict[str, Any]],
    *,
    ego_id: str,
    ego_pos: float,
    lane: int,
    ahead: bool,
    range_m: float,
) -> tuple[float, float] | None:
    """Return (distance_m, speed) of nearest vehicle on ``lane`` ahead/behind ego."""
    best: tuple[float, float] | None = None  # (distance, speed)
    for vid, st in vehicles.items():
        if vid == ego_id:
            continue
        if int(st["lane"]) != lane:
            continue
        pos = float(st["position"])
        if ahead:
            gap = pos - ego_pos
            if gap <= 1e-6 or gap > range_m:
                continue
        else:
            gap = ego_pos - pos
            if gap <= 1e-6 or gap > range_m:
                continue
        speed = float(st["speed"])
        if best is None or gap < best[0]:
            best = (gap, speed)
    return best


def build_observation_dict(
    *,
    cav_id: str,
    vehicles: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, float]:
    """Build a named, normalized local observation for one CAV.

    Missing neighbors / missing side lanes use:
      distance -> 1.0  (observation_range after normalization)
      relative_speed -> 0.0
    """
    if cav_id not in vehicles:
        raise KeyError(f"CAV id not in vehicle map: {cav_id}")

    ego = vehicles[cav_id]
    if ego.get("type") != "CAV":
        raise ValueError(f"Observation is only defined for CAV, got type={ego.get('type')}")

    p = _observation_params(cfg)
    range_m = float(p["range_m"])
    max_speed = float(p["max_speed_mps"])
    rel_scale = float(p["rel_speed_scale_mps"])
    road_len = float(p["road_length_m"])
    n_lanes = int(p["n_lanes"])
    max_lane = max(n_lanes - 1, 1)

    ego_speed = float(ego["speed"])
    ego_lane = int(ego["lane"])
    ego_pos = float(ego["position"])

    def neighbor_feats(lane: int | None, ahead: bool) -> tuple[float, float]:
        if lane is None or lane < 0 or lane >= n_lanes:
            return 1.0, 0.0
        hit = _nearest_on_lane(
            vehicles,
            ego_id=cav_id,
            ego_pos=ego_pos,
            lane=lane,
            ahead=ahead,
            range_m=range_m,
        )
        if hit is None:
            return 1.0, 0.0
        dist_m, other_speed = hit
        rel = (other_speed - ego_speed) / rel_scale
        return _clip01(dist_m / range_m), _clip11(rel)

    # SUMO: lane 0 = rightmost → left = +1, right = -1
    left_lane = ego_lane + 1 if ego_lane + 1 < n_lanes else None
    right_lane = ego_lane - 1 if ego_lane - 1 >= 0 else None

    front_d, front_rs = neighbor_feats(ego_lane, ahead=True)
    rear_d, rear_rs = neighbor_feats(ego_lane, ahead=False)
    lf_d, lf_rs = neighbor_feats(left_lane, ahead=True)
    lr_d, lr_rs = neighbor_feats(left_lane, ahead=False)
    rf_d, rf_rs = neighbor_feats(right_lane, ahead=True)
    rr_d, rr_rs = neighbor_feats(right_lane, ahead=False)

    # Signed distance to accident (positive = accident still ahead), clipped to range.
    acc_raw = float(p["accident_position_m"]) - ego_pos
    acc_dist = _clip11(acc_raw / range_m)
    # Keep key name ``accident_lane`` (OBS_DIM unchanged); value is blocked-lane bitmask
    # normalized by (2^{n_lanes}-1), e.g. lanes {0,1} => mask 3 / 7.
    mask_max = max((1 << n_lanes) - 1, 1)
    acc_lane = _clip01(float(p["blocked_lane_mask"]) / float(mask_max))

    return {
        "ego_speed": _clip01(ego_speed / max_speed),
        "ego_lane": _clip01(ego_lane / max_lane),
        "ego_position": _clip01(ego_pos / road_len),
        "front_distance": front_d,
        "front_relative_speed": front_rs,
        "rear_distance": rear_d,
        "rear_relative_speed": rear_rs,
        "left_front_distance": lf_d,
        "left_front_relative_speed": lf_rs,
        "left_rear_distance": lr_d,
        "left_rear_relative_speed": lr_rs,
        "right_front_distance": rf_d,
        "right_front_relative_speed": rf_rs,
        "right_rear_distance": rr_d,
        "right_rear_relative_speed": rr_rs,
        "accident_distance": acc_dist,
        "accident_lane": acc_lane,
    }


def build_observation(
    *,
    cav_id: str,
    vehicles: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
) -> np.ndarray:
    """Return fixed-dim float32 observation vector for ``cav_id``."""
    named = build_observation_dict(cav_id=cav_id, vehicles=vehicles, cfg=cfg)
    vec = np.array([named[k] for k in OBS_KEYS], dtype=np.float32)
    if vec.shape != (OBS_DIM,):
        raise RuntimeError(f"Unexpected observation shape {vec.shape}, expected ({OBS_DIM},)")
    return vec


def observations_for_cavs(
    *,
    cav_ids: list[str],
    vehicles: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Build observations for every CAV id currently available."""
    out: dict[str, np.ndarray] = {}
    for cav_id in cav_ids:
        if cav_id in vehicles:
            out[cav_id] = build_observation(
                cav_id=cav_id, vehicles=vehicles, cfg=cfg
            )
    return out
