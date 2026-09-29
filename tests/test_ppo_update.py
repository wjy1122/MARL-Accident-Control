"""Phase 4.3: PPO update unit tests on synthetic rollout data."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.ppo_update import PPOUpdater  # noqa: E402
from algorithms.rollout_buffer import RolloutBatch  # noqa: E402
from models.mappo_model import MAPPOModel  # noqa: E402


def _fake_batch(t: int = 8, n: int = 2, o: int = 17) -> RolloutBatch:
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(t, n, o)).astype(np.float32)
    actions = rng.integers(0, 3, size=(t, n), dtype=np.int64)
    rewards = rng.uniform(0.0, 1.0, size=(t, n)).astype(np.float32)
    log_probs = rng.normal(size=(t, n)).astype(np.float32) * 0.1 - 1.0
    values = rng.normal(size=(t, n)).astype(np.float32) * 0.1
    dones = np.zeros(t, dtype=np.bool_)
    dones[-1] = True
    mask = np.ones((t, n), dtype=np.float32)
    return RolloutBatch(
        observations=obs,
        actions=actions,
        rewards=rewards,
        log_probs=log_probs,
        values=values,
        dones=dones,
        agent_mask=mask,
        agent_ids=[f"a{i}" for i in range(n)],
        obs_dim=o,
        num_agents=n,
    )


def main() -> int:
    results: list[tuple[str, bool, str]] = []
    torch.manual_seed(0)
    model = MAPPOModel(obs_dim=17, action_dim=3, num_agents=2, hidden_dim=64)
    updater = PPOUpdater(
        model,
        clip_eps=0.2,
        actor_lr=3e-4,
        critic_lr=1e-3,
        ppo_epochs=2,
        minibatch_size=8,
        device="cpu",
    )
    batch = _fake_batch()

    # Evaluate new log probs API
    obs = torch.as_tensor(batch.observations.reshape(-1, 17), dtype=torch.float32)
    act = torch.as_tensor(batch.actions.reshape(-1), dtype=torch.int64)
    new_lp, ent = model.actor.evaluate_actions(obs, act)
    results.append(
        (
            "new log_prob / entropy shapes",
            tuple(new_lp.shape) == (16,) and tuple(ent.shape) == (16,),
            f"lp={tuple(new_lp.shape)} ent={tuple(ent.shape)}",
        )
    )

    old_lp = torch.as_tensor(batch.log_probs.reshape(-1), dtype=torch.float32)
    ratio = torch.exp(new_lp.detach() - old_lp)
    results.append(
        (
            "ratio finite",
            bool(torch.isfinite(ratio).all()),
            f"ratio mean={float(ratio.mean()):.3f}",
        )
    )

    stats = updater.update(batch)
    finite = all(
        math.isfinite(x)
        for x in (
            stats.actor_loss,
            stats.critic_loss,
            stats.entropy,
            stats.total_loss,
        )
    )
    results.append(
        (
            "losses finite",
            finite,
            f"actor={stats.actor_loss:.4f} critic={stats.critic_loss:.4f} "
            f"ent={stats.entropy:.4f}",
        )
    )
    results.append(("no NaN", finite, "ok"))
    results.append(("no Inf", finite, "ok"))

    failed = 0
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
        if not ok:
            failed += 1
    if failed:
        print(f"\n{failed} check(s) failed.")
        return 1
    print("\nAll PPO update checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
