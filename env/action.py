"""Phase 3.1: discrete CAV lane-change actions (no safety filtering)."""

from __future__ import annotations

from typing import Any

# Discrete action space
KEEP = 0
LEFT = 1
RIGHT = 2

ACTION_NAMES: dict[int, str] = {
    KEEP: "KEEP",
    LEFT: "LEFT",
    RIGHT: "RIGHT",
}

VALID_ACTIONS: frozenset[int] = frozenset({KEEP, LEFT, RIGHT})


def validate_action(action: Any) -> int:
    """Accept only integer {0,1,2}. Raise ValueError otherwise."""
    if isinstance(action, bool) or not isinstance(action, int):
        raise ValueError(
            f"Invalid action {action!r}: expected int in {{0, 1, 2}} "
            f"(KEEP/LEFT/RIGHT)"
        )
    if action not in VALID_ACTIONS:
        raise ValueError(
            f"Invalid action {action}: expected one of {{0, 1, 2}} "
            f"(KEEP/LEFT/RIGHT)"
        )
    return int(action)


def target_lane_index(
    *,
    current_lane: int,
    action: int,
    n_lanes: int,
    left_increases_index: bool,
) -> int | None:
    """Compute target lane for action, or None if KEEP / blocked by boundary.

    For this project's net (travel +x, right-hand):
      lane 0 = rightmost (y most negative)
      lane 2 = leftmost
    so LEFT increases index, RIGHT decreases index when ``left_increases_index``.
    """
    action = validate_action(action)
    if action == KEEP:
        return None

    if left_increases_index:
        delta = 1 if action == LEFT else -1
    else:
        delta = -1 if action == LEFT else 1

    target = int(current_lane) + delta
    if target < 0 or target >= int(n_lanes):
        return None
    return target
