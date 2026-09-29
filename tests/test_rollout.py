"""Phase 4.2: rollout buffer / collector tests (no training)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.rollout import RolloutCollector  # noqa: E402
from algorithms.rollout_buffer import RolloutBuffer  # noqa: E402
from env.highway_accident_env import HighwayAccidentEnv, load_config  # noqa: E402
from models.mappo_model import MAPPOModel, load_model_config  # noqa: E402


def main() -> int:
    results: list[tuple[str, bool, str]] = []

    # ---- Test 1: buffer shapes ----
    buf = RolloutBuffer(rollout_steps=4, num_agents=3, obs_dim=17)
    buf.set_agent_ids(["a", "b", "c"])
    for _ in range(4):
        buf.add(
            obs=np.random.randn(3, 17).astype(np.float32),
            actions=np.array([0, 1, 2], dtype=np.int64),
            rewards=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            log_probs=np.array([-0.1, -0.2, -0.3], dtype=np.float32),
            values=np.array([0.5, 0.5, 0.5], dtype=np.float32),
            done=False,
        )
    batch = buf.get()
    results.append(
        (
            "Test1 buffer shapes",
            batch.observations.shape == (4, 3, 17)
            and batch.actions.shape == (4, 3)
            and batch.rewards.shape == (4, 3)
            and batch.log_probs.shape == (4, 3)
            and batch.values.shape == (4, 3)
            and batch.dones.shape == (4,),
            f"obs={batch.observations.shape} act={batch.actions.shape} "
            f"rew={batch.rewards.shape} lp={batch.log_probs.shape}",
        )
    )

    # ---- Test 2: agent order fixed (metadata) ----
    results.append(
        (
            "Test2 agent_ids fixed",
            batch.agent_ids == ["a", "b", "c"],
            str(batch.agent_ids),
        )
    )

    # ---- Tests 3–6: real SUMO rollout ----
    env: HighwayAccidentEnv | None = None
    try:
        model_cfg = load_model_config()
        mblock = model_cfg.get("model", model_cfg)
        num_agents = int(mblock.get("num_agents", 4))
        obs_dim = int(mblock.get("observation_dim", 17))
        action_dim = int(mblock.get("action_dim", 3))
        hidden = int(mblock.get("hidden_dim", 128))

        # Long episode so mid-rollout truncate is unlikely for short collect.
        env_cfg = copy.deepcopy(load_config())
        env_cfg.setdefault("episode", {})["max_episode_steps"] = 10_000

        env = HighwayAccidentEnv(config=env_cfg, gui=False)
        model = MAPPOModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_agents=num_agents,
            hidden_dim=hidden,
        )
        collector = RolloutCollector(
            num_agents=num_agents,
            obs_dim=obs_dim,
            rollout_steps=5,
            warmup_steps=80,
            device="cpu",
        )

        before = [p.detach().cpu().clone() for p in model.parameters()]
        out = collector.collect(env, model, rollout_steps=5)
        after = [p.detach().cpu().clone() for p in model.parameters()]

        results.append(
            (
                "Test3 real SUMO rollout shapes",
                out.observations.shape == (5, num_agents, obs_dim)
                and out.actions.shape == (5, num_agents)
                and out.rewards.shape == (5, num_agents)
                and out.log_probs.shape == (5, num_agents),
                f"obs={out.observations.shape} act={out.actions.shape} "
                f"agents={out.agent_ids}",
            )
        )
        results.append(
            (
                "Test4 actions in {0,1,2}",
                bool(np.all((out.actions >= 0) & (out.actions <= 2))),
                f"unique={sorted(set(out.actions.ravel().tolist()))}",
            )
        )
        results.append(
            (
                "Test5 rewards in [0,1]",
                bool(np.all((out.rewards >= 0.0) & (out.rewards <= 1.0))),
                f"min={out.rewards.min():.3f} max={out.rewards.max():.3f}",
            )
        )
        params_ok = all(torch.equal(a, b) for a, b in zip(before, after))
        results.append(
            (
                "Test6 no training (params unchanged)",
                params_ok,
                f"n_params={len(before)}",
            )
        )

        # Agent-id list length / slot count stable
        results.append(
            (
                "agent slot count == num_agents",
                len(out.agent_ids) == num_agents,
                str(out.agent_ids),
            )
        )

        env.close()
    except Exception as exc:  # noqa: BLE001
        import traceback

        results.append(("SUMO rollout", False, f"{exc}\n{traceback.format_exc()}"))
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass

    failed = 0
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
        if not ok:
            failed += 1

    if failed:
        print(f"\n{failed} check(s) failed.")
        return 1
    print("\nAll Phase 4.2 rollout checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
