"""Phase 4.3: end-to-end MAPPO update — parameters must change."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.mappo_trainer import MAPPOTrainer  # noqa: E402
from algorithms.rollout import RolloutCollector, load_train_config  # noqa: E402
from env.highway_accident_env import HighwayAccidentEnv, load_config  # noqa: E402
from models.mappo_model import MAPPOModel, load_model_config  # noqa: E402


def _params(model: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().cpu().clone() for p in model.parameters()]


def _changed(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    return any(not torch.equal(x, y) for x, y in zip(a, b))


def main() -> int:
    results: list[tuple[str, bool, str]] = []
    env: HighwayAccidentEnv | None = None
    try:
        train_cfg = load_train_config()
        model_cfg = load_model_config()
        mblock = model_cfg.get("model", model_cfg)
        tblock = train_cfg.get("train", train_cfg)

        num_agents = int(mblock.get("num_agents", 4))
        obs_dim = int(mblock.get("observation_dim", 17))
        action_dim = int(mblock.get("action_dim", 3))
        hidden = int(mblock.get("hidden_dim", 128))

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
            rollout_steps=8,
            warmup_steps=80,
            device="cpu",
        )
        # Smaller minibatches for tiny rollout
        train_cfg = copy.deepcopy(train_cfg)
        train_cfg.setdefault("train", {})
        train_cfg["train"]["ppo_epochs"] = 4
        train_cfg["train"]["minibatch_size"] = 8

        actor_before = _params(model.actor)
        critic_before = _params(model.critic)

        batch = collector.collect(env, model, rollout_steps=8)
        trainer = MAPPOTrainer(model, train_cfg, device="cpu")
        stats = trainer.update(batch)

        actor_after = _params(model.actor)
        critic_after = _params(model.critic)

        actor_chg = _changed(actor_before, actor_after)
        critic_chg = _changed(critic_before, critic_after)
        finite = all(
            abs(x) < 1e6
            for x in (stats.actor_loss, stats.critic_loss, stats.entropy, stats.total_loss)
        ) and all(
            __import__("math").isfinite(x)
            for x in (stats.actor_loss, stats.critic_loss, stats.entropy, stats.total_loss)
        )

        results.append(
            (
                "rollout shapes",
                batch.observations.shape[0] == 8
                and batch.observations.shape[-1] == obs_dim,
                f"obs={batch.observations.shape}",
            )
        )
        results.append(("actor params changed", actor_chg, str(actor_chg)))
        results.append(("critic params changed", critic_chg, str(critic_chg)))
        results.append(
            (
                "losses finite",
                finite,
                f"actor={stats.actor_loss:.4f} critic={stats.critic_loss:.4f} "
                f"ent={stats.entropy:.4f}",
            )
        )

        env.close()
    except Exception as exc:  # noqa: BLE001
        import traceback

        results.append(("mappo update", False, f"{exc}\n{traceback.format_exc()}"))
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
    print("\nAll MAPPO update checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
