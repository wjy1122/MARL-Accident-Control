"""Legal-action mask for KEEP / LEFT / RIGHT.

True / 1 = legal. The mask is applied to logits before Categorical.
It does not rewrite an illegal sample into KEEP.

Lane convention (this project): lane 0 = rightmost, lane 2 = leftmost,
LEFT increases the lane index when ``left_increases_index`` is True.
Legality is computed with ``target_lane_index``, not a hard-coded guess.
"""

from __future__ import annotations

import numpy as np
import torch

from env.action import KEEP, LEFT, RIGHT, target_lane_index

# Finite large negative value. -inf can produce NaN inside Categorical.
MASK_LOGIT = -1.0e9


def legal_action_mask(
    lane: int,
    *,
    n_lanes: int = 3,
    left_increases_index: bool = True,
    mid_lane_change: bool = False,
    forbid_left: bool = False,
    forbid_right: bool = False,
) -> np.ndarray:
    """Return float mask shape (3,). 1 = legal, 0 = illegal. KEEP is always legal.

    Extra constraints (Phase 6.5 oscillation suppress):
      * ``mid_lane_change``: forbid LEFT/RIGHT while a LC is in progress.
      * ``forbid_left`` / ``forbid_right``: opposite-direction cooldown after an
        effective lane-index change.
    """
    mask = np.zeros(3, dtype=np.float32)
    lane_i = int(lane)
    n = int(n_lanes)
    for action in (KEEP, LEFT, RIGHT):
        if action == KEEP:
            mask[action] = 1.0
            continue
        target = target_lane_index(
            current_lane=lane_i,
            action=action,
            n_lanes=n,
            left_increases_index=bool(left_increases_index),
        )
        mask[action] = 0.0 if target is None else 1.0
    if mid_lane_change:
        mask[LEFT] = 0.0
        mask[RIGHT] = 0.0
    if forbid_left:
        mask[LEFT] = 0.0
    if forbid_right:
        mask[RIGHT] = 0.0
    return mask


def apply_logit_mask(
    logits: torch.Tensor, action_mask: torch.Tensor | np.ndarray | None
) -> torch.Tensor:
    """Set illegal-action logits to MASK_LOGIT. ``action_mask`` 1/True = legal."""
    if action_mask is None:
        return logits
    legal = torch.as_tensor(action_mask, device=logits.device)
    if legal.dtype != torch.bool:
        legal = legal > 0.5
    if legal.shape[-1] != logits.shape[-1]:
        raise ValueError(
            f"action_mask last dim {legal.shape[-1]} != logits {logits.shape[-1]}"
        )
    if legal.shape != logits.shape:
        if legal.dim() == 1 and logits.dim() == 2:
            legal = legal.unsqueeze(0).expand_as(logits)
        elif legal.dim() == logits.dim() and legal.shape[0] == 1:
            legal = legal.expand_as(logits)
        else:
            raise ValueError(
                f"action_mask shape {tuple(legal.shape)} incompatible with "
                f"logits {tuple(logits.shape)}"
            )
    return logits.masked_fill(~legal, MASK_LOGIT)


def illegal_action_rate(
    actions: np.ndarray,
    action_mask: np.ndarray,
    agent_mask: np.ndarray | None = None,
) -> float:
    """Fraction of live samples whose chosen action is illegal under the stored mask."""
    act = np.asarray(actions)
    am = np.asarray(action_mask)
    if agent_mask is None:
        live = np.ones(act.shape, dtype=bool)
    else:
        live = np.asarray(agent_mask) > 0.5
    if int(live.sum()) == 0:
        return 0.0
    chosen = np.take_along_axis(am, act[..., None].astype(np.int64), axis=-1)[..., 0]
    illegal = live & (chosen < 0.5)
    return float(illegal.sum()) / float(live.sum())
