"""Phase 6.1 action-mask unit tests. No SUMO, no training."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.ppo_update import PPOUpdater  # noqa: E402
from algorithms.rollout_buffer import RolloutBatch, RolloutBuffer  # noqa: E402
from env.action import KEEP, LEFT, RIGHT  # noqa: E402
from env.action_mask import (  # noqa: E402
    MASK_LOGIT,
    apply_logit_mask,
    illegal_action_rate,
    legal_action_mask,
)
from models.actor import Actor  # noqa: E402
from models.mappo_model import MAPPOModel  # noqa: E402


def _probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def main() -> int:
    results: list[tuple[str, bool, str]] = []

    m0 = legal_action_mask(0)
    m1 = legal_action_mask(1)
    m2 = legal_action_mask(2)
    results.append(
        (
            "lane masks",
            bool(
                np.array_equal(m0, [1, 1, 0])
                and np.array_equal(m1, [1, 1, 1])
                and np.array_equal(m2, [1, 0, 1])
            ),
            f"lane0={m0.tolist()} lane1={m1.tolist()} lane2={m2.tolist()}",
        )
    )

    mid = legal_action_mask(1, mid_lane_change=True)
    results.append(
        (
            "mid-LC masks L/R",
            bool(np.array_equal(mid, [1, 0, 0])),
            f"mid={mid.tolist()}",
        )
    )
    cool_r = legal_action_mask(1, forbid_right=True)
    cool_l = legal_action_mask(1, forbid_left=True)
    results.append(
        (
            "cooldown forbids opposite",
            bool(
                np.array_equal(cool_r, [1, 1, 0])
                and np.array_equal(cool_l, [1, 0, 1])
            ),
            f"forbid_R={cool_r.tolist()} forbid_L={cool_l.tolist()}",
        )
    )

    raw = torch.tensor([0.2, 0.4, 5.0])
    p0 = _probs(apply_logit_mask(raw, m0))
    p2 = _probs(apply_logit_mask(raw, m2))
    p1 = _probs(apply_logit_mask(raw, m1))
    results.append(
        (
            "lane0 RIGHT prob = 0",
            float(p0[RIGHT]) < 1e-6 and float(p0.sum()) > 0.999,
            f"P={p0.tolist()}",
        )
    )
    results.append(
        (
            "lane2 LEFT prob = 0",
            float(p2[LEFT]) < 1e-6,
            f"P={p2.tolist()}",
        )
    )
    results.append(
        (
            "lane1 all legal",
            bool(torch.all(p1 > 1e-6)),
            f"P={p1.tolist()}",
        )
    )
    masked0 = apply_logit_mask(raw, m0)
    results.append(
        (
            "illegal logit is finite",
            bool(torch.isfinite(masked0).all()) and float(masked0[RIGHT]) == MASK_LOGIT,
            f"masked={masked0.tolist()}",
        )
    )

    torch.manual_seed(0)
    actor = Actor()

    def _force(obs: torch.Tensor) -> torch.Tensor:
        out = torch.tensor([0.1, 0.2, 4.0], device=obs.device)
        if obs.dim() == 1:
            return out
        return out.unsqueeze(0).expand(obs.shape[0], -1).contiguous()

    actor.forward = _force  # type: ignore[method-assign]

    n_sample = 4000
    counts = {0: np.zeros(3, dtype=np.int64), 1: np.zeros(3, dtype=np.int64), 2: np.zeros(3, dtype=np.int64)}
    masks = {0: m0, 1: m1, 2: m2}
    obs = torch.zeros(17)
    for lane, mask in masks.items():
        am = torch.as_tensor(mask)
        for _ in range(n_sample):
            a, lp = actor.get_action(obs, action_mask=am)
            counts[lane][int(a.item())] += 1
            if not math.isfinite(float(lp)):
                counts[lane][0] = -1
                break
    results.append(
        (
            "lane0 RIGHT sampled = 0",
            int(counts[0][RIGHT]) == 0 and int(counts[0].sum()) == n_sample,
            f"counts={counts[0].tolist()}",
        )
    )
    results.append(
        (
            "lane2 LEFT sampled = 0",
            int(counts[2][LEFT]) == 0 and int(counts[2].sum()) == n_sample,
            f"counts={counts[2].tolist()}",
        )
    )
    results.append(
        (
            "lane1 all actions sampled",
            bool(np.all(counts[1] > 0)),
            f"counts={counts[1].tolist()}",
        )
    )

    det, _lp = actor.get_action(
        obs, deterministic=True, action_mask=torch.as_tensor(m0)
    )
    results.append(
        (
            "deterministic argmax respects mask",
            int(det.item()) == LEFT,
            f"argmax={int(det.item())} (raw prefers RIGHT)",
        )
    )

    # Same mask on old and new log_prob → ratio finite and 1 before any step.
    batch_obs = torch.zeros(3, 17)
    batch_act = torch.tensor([RIGHT, KEEP, LEFT])
    batch_am = torch.stack(
        [torch.as_tensor(m0), torch.as_tensor(m1), torch.as_tensor(m2)]
    )
    # lane0 cannot have RIGHT; use legal actions for the ratio test
    batch_act = torch.tensor([LEFT, KEEP, RIGHT])
    lp1, ent1 = actor.evaluate_actions(batch_obs, batch_act, action_mask=batch_am)
    lp2, ent2 = actor.evaluate_actions(batch_obs, batch_act, action_mask=batch_am)
    ratio = torch.exp(lp2 - lp1)
    results.append(
        (
            "masked log_prob ratio finite",
            bool(torch.isfinite(ratio).all() and torch.isfinite(ent1).all())
            and float((ratio - 1).abs().max()) < 1e-5,
            f"ratio={ratio.tolist()} ent={ent1.tolist()}",
        )
    )
    # Illegal action log_prob is large negative but finite, not NaN.
    bad_lp, bad_ent = actor.evaluate_actions(
        batch_obs[:1], torch.tensor([RIGHT]), action_mask=batch_am[:1]
    )
    results.append(
        (
            "illegal log_prob finite and tiny",
            bool(torch.isfinite(bad_lp).all() and torch.isfinite(bad_ent).all())
            and float(bad_lp) < -10.0,
            f"lp={float(bad_lp):.3f}",
        )
    )

    # Slot rebind: mask is stored on the same (t, slot) as the action.
    buf = RolloutBuffer(2, 1, 17)
    obs0 = np.zeros((1, 17), dtype=np.float32)
    for t, lane in enumerate((0, 2)):
        buf.add(
            obs=obs0,
            actions=np.array([KEEP], dtype=np.int64),
            rewards=np.ones(1, dtype=np.float32),
            log_probs=np.zeros(1, dtype=np.float32),
            values=np.zeros(1, dtype=np.float32),
            done=False,
            agent_mask=np.ones(1, dtype=np.float32),
            action_mask=legal_action_mask(lane)[None, :],
        )
    batch = buf.get()
    aligned = (
        batch.action_mask is not None
        and batch.action_mask.shape == (2, 1, 3)
        and batch.action_mask[0, 0, RIGHT] == 0
        and batch.action_mask[1, 0, LEFT] == 0
        and batch.action_mask[0, 0, LEFT] == 1
        and batch.action_mask[1, 0, RIGHT] == 1
    )
    results.append(
        (
            "mask follows slot rebind per timestep",
            bool(aligned),
            f"m0={None if batch.action_mask is None else batch.action_mask[0, 0].tolist()} "
            f"m1={None if batch.action_mask is None else batch.action_mask[1, 0].tolist()}",
        )
    )
    results.append(
        (
            "illegal rate on stored masks",
            illegal_action_rate(
                np.array([[RIGHT], [LEFT]]),
                batch.action_mask,
                np.ones((2, 1), dtype=np.float32),
            )
            == 1.0
            and illegal_action_rate(
                np.array([[KEEP], [KEEP]]),
                batch.action_mask,
                np.ones((2, 1), dtype=np.float32),
            )
            == 0.0,
            "RIGHT on lane0 and LEFT on lane2 are illegal",
        )
    )

    # PPO update stays finite when the stored mask is reused.
    torch.manual_seed(1)
    model = MAPPOModel(obs_dim=17, action_dim=3, num_agents=2, hidden_dim=32)
    updater = PPOUpdater(
        model,
        ppo_epochs=1,
        minibatch_size=8,
        device="cpu",
    )
    t, n, o = 4, 2, 17
    rng = np.random.default_rng(1)
    obs_np = rng.normal(size=(t, n, o)).astype(np.float32)
    am_np = np.ones((t, n, 3), dtype=np.float32)
    am_np[:, 0, RIGHT] = 0.0  # slot 0 is lane 0
    am_np[:, 1, LEFT] = 0.0  # slot 1 is lane 2
    act_np = np.zeros((t, n), dtype=np.int64)
    act_np[:, 1] = RIGHT  # legal on lane 2
    obs_t = torch.as_tensor(obs_np.reshape(t * n, o))
    act_t = torch.as_tensor(act_np.reshape(t * n))
    am_t = torch.as_tensor(am_np.reshape(t * n, 3))
    with torch.no_grad():
        old_lp, _ent = model.actor.evaluate_actions(obs_t, act_t, action_mask=am_t)
    ppo_batch = RolloutBatch(
        observations=obs_np,
        actions=act_np,
        rewards=rng.uniform(0, 1, size=(t, n)).astype(np.float32),
        log_probs=old_lp.detach().cpu().numpy().reshape(t, n),
        values=np.zeros((t, n), dtype=np.float32),
        dones=np.zeros(t, dtype=np.bool_),
        agent_mask=np.ones((t, n), dtype=np.float32),
        agent_ids=["cavA", "cavB"],
        obs_dim=o,
        num_agents=n,
        action_mask=am_np,
    )
    stats = updater.update(ppo_batch)
    finite = all(
        math.isfinite(x)
        for x in (
            stats.actor_loss,
            stats.critic_loss,
            stats.entropy,
            stats.approx_kl,
            stats.clip_fraction,
        )
    )
    results.append(
        (
            "PPO update with stored mask is finite",
            finite,
            f"actor={stats.actor_loss:.4f} ent={stats.entropy:.4f} kl={stats.approx_kl:.4f}",
        )
    )

    failed = [name for name, ok, _ in results if not ok]
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print("Action mask unit tests:", "PASS" if not failed else "FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
