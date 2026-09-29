"""Phase 4.4.4: near-accident reward / speed / action diagnostics (logging only)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from env.action import ACTION_NAMES, KEEP, LEFT, RIGHT, target_lane_index


@dataclass
class ZoneRewardDiag:
    """Accumulated near/far reward+behavior stats (no reward formula changes)."""

    accident_pos_m: float = 1200.0
    near_m: float = 150.0
    vmax_mps: float = 30.0
    n_lanes: int = 3
    left_increases_index: bool = True

    # Counts
    global_n: int = 0
    near_n: int = 0
    far_n: int = 0

    # Reward components actually present in env: individual, group, total
    sum_total: float = 0.0
    sum_ind: float = 0.0
    sum_grp: float = 0.0
    near_sum_total: float = 0.0
    near_sum_ind: float = 0.0
    near_sum_grp: float = 0.0
    far_sum_total: float = 0.0
    far_sum_ind: float = 0.0
    far_sum_grp: float = 0.0

    # Speed
    near_speeds: list[float] = field(default_factory=list)
    far_speeds: list[float] = field(default_factory=list)

    # Actions (raw policy) in near zone
    near_action_counts: list[int] = field(default_factory=lambda: [0, 0, 0])
    far_action_counts: list[int] = field(default_factory=lambda: [0, 0, 0])

    # Boundary no-op (only "filter" in this codebase — not a safety supervisor)
    lc_raw_n: int = 0  # LEFT/RIGHT issued
    lc_boundary_blocked_n: int = 0  # target_lane_index -> None
    lc_commanded_n: int = 0  # TraCI changeLane would be called
    near_lc_raw_n: int = 0
    near_lc_boundary_blocked_n: int = 0
    near_lc_lane_changed_n: int = 0  # lane index changed after step
    near_lc_lane_unchanged_n: int = 0

    def is_near(self, pos: float) -> bool:
        return abs(float(pos) - self.accident_pos_m) <= self.near_m

    def add_sample(
        self,
        *,
        pos: float,
        speed: float,
        individual: float,
        group: float,
        total: float,
        raw_action: int,
        lane_before: int,
        lane_after: int | None = None,
    ) -> None:
        near = self.is_near(pos)
        self.global_n += 1
        self.sum_total += float(total)
        self.sum_ind += float(individual)
        self.sum_grp += float(group)

        a = int(raw_action)
        if near:
            self.near_n += 1
            self.near_sum_total += float(total)
            self.near_sum_ind += float(individual)
            self.near_sum_grp += float(group)
            self.near_speeds.append(float(speed))
            if 0 <= a <= 2:
                self.near_action_counts[a] += 1
        else:
            self.far_n += 1
            self.far_sum_total += float(total)
            self.far_sum_ind += float(individual)
            self.far_sum_grp += float(group)
            self.far_speeds.append(float(speed))
            if 0 <= a <= 2:
                self.far_action_counts[a] += 1

        # Boundary suppression (same rule as env.apply_action; no extra safety stack)
        if a in (LEFT, RIGHT):
            self.lc_raw_n += 1
            if near:
                self.near_lc_raw_n += 1
            target = target_lane_index(
                current_lane=int(lane_before),
                action=a,
                n_lanes=self.n_lanes,
                left_increases_index=self.left_increases_index,
            )
            if target is None:
                self.lc_boundary_blocked_n += 1
                if near:
                    self.near_lc_boundary_blocked_n += 1
            else:
                self.lc_commanded_n += 1
                if near and lane_after is not None:
                    if int(lane_after) != int(lane_before):
                        self.near_lc_lane_changed_n += 1
                    else:
                        self.near_lc_lane_unchanged_n += 1

    @staticmethod
    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    @staticmethod
    def _pct(counts: list[int]) -> dict[str, float]:
        n = int(sum(counts))
        if n <= 0:
            return {ACTION_NAMES[i]: float("nan") for i in range(3)} | {"n": 0.0}
        return {
            ACTION_NAMES[i]: 100.0 * counts[i] / n for i in range(3)
        } | {"n": float(n)}

    def summary(self) -> dict[str, Any]:
        def avg(s: float, n: int) -> float:
            return float(s / n) if n > 0 else float("nan")

        near_spd = self.near_speeds
        far_spd = self.far_speeds
        return {
            "accident_pos_m": self.accident_pos_m,
            "near_m": self.near_m,
            "global_n": self.global_n,
            "near_n": self.near_n,
            "far_n": self.far_n,
            "near_frac": float(self.near_n / self.global_n) if self.global_n else 0.0,
            "global_mean_total": avg(self.sum_total, self.global_n),
            "global_mean_individual": avg(self.sum_ind, self.global_n),
            "global_mean_group": avg(self.sum_grp, self.global_n),
            "near_mean_total": avg(self.near_sum_total, self.near_n),
            "near_mean_individual": avg(self.near_sum_ind, self.near_n),
            "near_mean_group": avg(self.near_sum_grp, self.near_n),
            "far_mean_total": avg(self.far_sum_total, self.far_n),
            "far_mean_individual": avg(self.far_sum_ind, self.far_n),
            "far_mean_group": avg(self.far_sum_grp, self.far_n),
            "near_mean_speed": self._mean(near_spd),
            "near_min_speed": float(np.min(near_spd)) if near_spd else float("nan"),
            "near_max_speed": float(np.max(near_spd)) if near_spd else float("nan"),
            "near_mean_speed_ratio": (
                self._mean(near_spd) / self.vmax_mps if near_spd else float("nan")
            ),
            "far_mean_speed": self._mean(far_spd),
            "far_mean_speed_ratio": (
                self._mean(far_spd) / self.vmax_mps if far_spd else float("nan")
            ),
            "near_action_pct": self._pct(self.near_action_counts),
            "far_action_pct": self._pct(self.far_action_counts),
            "lc_raw_n": self.lc_raw_n,
            "lc_boundary_blocked_n": self.lc_boundary_blocked_n,
            "lc_commanded_n": self.lc_commanded_n,
            "boundary_block_rate": (
                float(self.lc_boundary_blocked_n / self.lc_raw_n)
                if self.lc_raw_n
                else float("nan")
            ),
            "near_lc_raw_n": self.near_lc_raw_n,
            "near_lc_boundary_blocked_n": self.near_lc_boundary_blocked_n,
            "near_boundary_block_rate": (
                float(self.near_lc_boundary_blocked_n / self.near_lc_raw_n)
                if self.near_lc_raw_n
                else float("nan")
            ),
            "near_lc_lane_changed_n": self.near_lc_lane_changed_n,
            "near_lc_lane_unchanged_n": self.near_lc_lane_unchanged_n,
            "reward_components_present": ["individual", "group", "total"],
            "reward_components_absent": [
                "safety",
                "collision",
                "ttc",
                "lane_change_penalty",
                "accident_penalty",
                "efficiency_extra",
            ],
        }
