"""Phase 4.1: MAPPO Actor / Critic forward-pass unit tests (no training)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.observation import OBS_DIM  # noqa: E402
from models.actor import Actor  # noqa: E402
from models.critic import Critic  # noqa: E402
from models.mappo_model import MAPPOModel  # noqa: E402


def main() -> int:
    results: list[tuple[str, bool, str]] = []
    torch.manual_seed(0)

    obs_dim = OBS_DIM  # 17
    action_dim = 3
    hidden_dim = 128
    num_agents = 4

    actor = Actor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    critic = Critic(input_dim=obs_dim * num_agents, hidden_dim=hidden_dim)
    model = MAPPOModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_agents=num_agents,
        hidden_dim=hidden_dim,
    )

    # Actor batch forward
    batch = torch.randn(8, obs_dim)
    logits = actor(batch)
    results.append(
        (
            "Actor logits shape",
            tuple(logits.shape) == (8, action_dim),
            f"shape={tuple(logits.shape)}",
        )
    )
    results.append(
        ("Actor logits finite", bool(torch.isfinite(logits).all()), str(logits[0]))
    )

    # No softmax in network output extremes check: logits can be any real
    action, log_prob = actor.get_action(batch)
    results.append(
        (
            "get_action batch shapes",
            tuple(action.shape) == (8,) and tuple(log_prob.shape) == (8,),
            f"a={tuple(action.shape)} lp={tuple(log_prob.shape)}",
        )
    )
    results.append(
        (
            "actions in {0,1,2}",
            bool(((action >= 0) & (action < action_dim)).all()),
            f"unique={sorted(set(action.tolist()))}",
        )
    )

    single_obs = torch.randn(obs_dim)
    a1, lp1 = actor.get_action(single_obs)
    results.append(
        (
            "get_action single",
            a1.ndim == 0 and lp1.ndim == 0 and int(a1.item()) in (0, 1, 2),
            f"action={int(a1.item())} log_prob={float(lp1.item()):.4f}",
        )
    )

    # Critic
    state = torch.randn(8, obs_dim * num_agents)
    values = critic(state)
    results.append(
        (
            "Critic value shape",
            tuple(values.shape) == (8, 1),
            f"shape={tuple(values.shape)}",
        )
    )
    results.append(
        ("Critic values finite", bool(torch.isfinite(values).all()), str(values[0]))
    )

    # Shared actor: same module object used for every agent forward
    obs0 = torch.randn(obs_dim)
    obs1 = torch.randn(obs_dim)
    logits0 = model.actor(obs0)
    logits1 = model.actor(obs1)
    results.append(
        (
            "shared Actor module object",
            isinstance(model.actor, Actor),
            f"type={type(model.actor).__name__} id={id(model.actor)}",
        )
    )
    w_before = model.actor.net[0].weight.data.clone()
    _ = model.act(obs0)
    _ = model.act(obs1)
    w_after = model.actor.net[0].weight.data
    results.append(
        (
            "Actor weights unchanged by forward",
            bool(torch.equal(w_before, w_after)),
            "no training / no in-place update",
        )
    )
    results.append(
        (
            "multi-agent forward via shared actor",
            tuple(logits0.shape) == (action_dim,)
            and tuple(logits1.shape) == (action_dim,),
            f"l0={tuple(logits0.shape)} l1={tuple(logits1.shape)}",
        )
    )

    # Critic input dim not hard-coded 51
    results.append(
        (
            "critic_input_dim = obs_dim * num_agents",
            model.critic_input_dim == obs_dim * num_agents
            and model.critic.input_dim == obs_dim * num_agents,
            f"dim={model.critic_input_dim}",
        )
    )

    central = model.build_central_state([torch.randn(obs_dim) for _ in range(3)])
    results.append(
        (
            "build_central_state pad to num_agents",
            tuple(central.shape) == (obs_dim * num_agents,),
            f"shape={tuple(central.shape)} (3 agents padded to {num_agents})",
        )
    )
    v = model.get_value(central)
    results.append(
        (
            "MAPPOModel.get_value",
            tuple(v.shape) == (1,) and bool(torch.isfinite(v).all()),
            f"V={float(v.item()):.4f}",
        )
    )

    # from_config
    try:
        m2 = MAPPOModel.from_config()
        results.append(
            (
                "from_config",
                m2.obs_dim == 17 and m2.action_dim == 3 and m2.num_agents >= 1,
                f"obs={m2.obs_dim} act={m2.action_dim} n={m2.num_agents}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        results.append(("from_config", False, str(exc)))

    # No Attention modules present
    module_names = [type(m).__name__ for m in model.modules()]
    results.append(
        (
            "no Attention modules",
            all("Attention" not in name for name in module_names),
            str(sorted(set(module_names))),
        )
    )

    failed = 0
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
        if not ok:
            failed += 1

    if failed:
        print(f"\n{failed} check(s) failed.")
        return 1
    print("\nAll Phase 4.1 MAPPO network checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
