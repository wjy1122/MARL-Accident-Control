"""Unit tests for rollout diagnostics (no SUMO required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.traffic_diagnostics import diagnose_rollout_batch  # noqa: E402


def main() -> int:
    t, n, o = 4, 2, 17
    obs = np.zeros((t, n, o), dtype=np.float32)
    # ego_speed=1.0 → 30 m/s; ego_pos far upstream (100/2000)
    obs[..., 0] = 1.0
    obs[..., 2] = 100.0 / 2000.0
    # accident_distance clipped to +1 (far ahead)
    obs[..., 15] = 1.0

    actions = np.array([[0, 1], [0, 2], [1, 0], [2, 1]], dtype=np.int64)
    rewards = np.ones((t, n), dtype=np.float32)
    mask = np.ones((t, n), dtype=np.float32)

    diag = diagnose_rollout_batch(
        obs,
        actions,
        rewards,
        mask,
        road_length_m=2000.0,
        max_speed_mps=30.0,
        accident_position_m=1200.0,
        obs_range_m=150.0,
        near_accident_m=150.0,
    )

    results: list[tuple[str, bool, str]] = []
    results.append(
        (
            "reward ~1",
            abs(diag.reward_mean - 1.0) < 1e-6,
            f"mean={diag.reward_mean}",
        )
    )
    results.append(
        (
            "not near accident",
            diag.reached_accident_zone is False and diag.near_samples == 0,
            f"near={diag.near_samples}",
        )
    )
    results.append(
        (
            "action pct sum ~100",
            abs(diag.action_keep_pct + diag.action_left_pct + diag.action_right_pct - 100.0)
            < 1e-3,
            f"K={diag.action_keep_pct:.1f} L={diag.action_left_pct:.1f} R={diag.action_right_pct:.1f}",
        )
    )

    # Place agents near accident: pos=1100, acc_dist=(1200-1100)/150=0.666
    obs2 = obs.copy()
    obs2[..., 2] = 1100.0 / 2000.0
    obs2[..., 15] = (1200.0 - 1100.0) / 150.0
    diag2 = diagnose_rollout_batch(
        obs2, actions, rewards, mask, accident_position_m=1200.0, obs_range_m=150.0
    )
    results.append(
        (
            "near accident detected",
            diag2.reached_accident_zone and diag2.near_samples == t * n,
            f"near={diag2.near_samples}",
        )
    )

    failed = 0
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
