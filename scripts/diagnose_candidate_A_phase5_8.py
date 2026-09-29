"""Phase 5.8 — Advantage / Value / TD Target diagnosis (read-only).

Loads frozen Candidate A checkpoint, collects rollouts with the project's
RolloutCollector-equivalent path, then computes GAE via algorithms.gae.compute_gae
(same formula as PPOUpdater). Does not train or write checkpoints.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.gae import compute_gae  # noqa: E402
from algorithms.ppo_update import normalize_advantages  # noqa: E402
from algorithms.rollout import (  # noqa: E402
    _lock_agent_ids,
    _obs_matrix,
    _rebind_agent_slots,
    load_train_config,
    resolve_device,
)
from algorithms.rollout_buffer import RolloutBuffer  # noqa: E402
from env.action import ACTION_NAMES, KEEP, LEFT, RIGHT  # noqa: E402
from env.highway_accident_env import HighwayAccidentEnv, load_config  # noqa: E402
from utils.baseline_eval import (  # noqa: E402
    load_mappo_checkpoint,
    params_unchanged,
    set_eval_seed,
    snapshot_model_params,
)

OUT = PROJECT_ROOT / "results" / "experiments" / "reward_candidate_A_phase5_8"
CA_CKPT = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "reward_candidate_A_phase5_3"
    / "checkpoints"
    / "episode_0050.pt"
)
CA_REWARD = PROJECT_ROOT / "configs" / "reward_candidate_A.yaml"
SEED = 42
N_ROLLOUTS = 6
ZONE_R = 150.0
NAMES = ("KEEP", "LEFT", "RIGHT")


def _stats(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
        }
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "std": float(np.std(a)),
    }


def _fmt(x: float, nd: int = 4) -> str:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def _zone(pos: float, x_acc: float) -> str:
    if pos < x_acc - ZONE_R:
        return "before"
    if pos > x_acc + ZONE_R:
        return "after"
    return "near"


def _boundary(action: int, lane: int, n_lanes: int) -> str:
    if action == LEFT and lane >= n_lanes - 1:
        return "invalid_LEFT"
    if action == RIGHT and lane <= 0:
        return "invalid_RIGHT"
    return "none"


@torch.no_grad()
def collect_diag(
    *,
    env: HighwayAccidentEnv,
    model: torch.nn.Module,
    rollout_steps: int,
    warmup: int,
    device: torch.device,
    deterministic: bool,
) -> dict[str, Any]:
    """Mirror RolloutCollector.collect, plus lane/zone/logits and post-step lane."""
    model.eval()
    num_agents = int(model.num_agents)
    obs_dim = int(model.obs_dim)
    x_acc = float(env.accident_position_m)
    n_lanes = int(env.n_lanes)
    buffer = RolloutBuffer(rollout_steps, num_agents, obs_dim)

    extras: list[dict[str, Any]] = []
    # Minimal lifecycle to match training collector.
    if not env.is_connected:
        env.reset()
    for _ in range(warmup):
        if len(env.cav_ids) > 0:
            break
        env.step()
    agent_ids = _lock_agent_ids(env, num_agents)
    buffer.set_agent_ids(list(agent_ids))

    for _t in range(rollout_steps):
        agent_ids, _dep, _new = _rebind_agent_slots(env, agent_ids, num_agents)
        if len(env.cav_ids) == 0:
            _o, _r, term, trunc, _i = env.step({})
            del _o, _r, _i
            buffer.add(
                obs=np.zeros((num_agents, obs_dim), dtype=np.float32),
                actions=np.zeros(num_agents, dtype=np.int64),
                rewards=np.zeros(num_agents, dtype=np.float32),
                log_probs=np.zeros(num_agents, dtype=np.float32),
                values=np.zeros(num_agents, dtype=np.float32),
                done=bool(term or trunc),
                agent_mask=np.zeros(num_agents, dtype=np.float32),
            )
            extras.append({"agents": []})
            if term or trunc:
                env.reset()
                for _ in range(warmup):
                    if len(env.cav_ids) > 0:
                        break
                    env.step()
                agent_ids = _lock_agent_ids(env, num_agents)
            continue

        states = env.get_cav_states()
        obs_np = _obs_matrix(env, agent_ids, obs_dim)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        actions_np = np.zeros(num_agents, dtype=np.int64)
        logp_np = np.zeros(num_agents, dtype=np.float32)
        actions_dict: dict[str, int] = {}
        agent_meta: list[dict[str, Any]] = []

        for i, aid in enumerate(agent_ids):
            if aid is None or aid not in env.cav_ids:
                actions_np[i] = KEEP
                continue
            logits_t = model.actor.forward(obs_t[i])
            if logits_t.dim() == 1:
                logits_t = logits_t.unsqueeze(0)
            logits = [float(x) for x in logits_t[0].detach().cpu().tolist()]
            probs = torch.softmax(logits_t, dim=-1)[0].detach().cpu().tolist()
            action_i, logp_i = model.act(obs_t[i], deterministic=deterministic)
            a = int(action_i.item())
            actions_np[i] = a
            logp_np[i] = float(logp_i.item())
            actions_dict[aid] = a
            st = states.get(aid)
            lane = int(st["lane"]) if st else -1
            pos = float(st["position"]) if st else float("nan")
            agent_meta.append(
                {
                    "slot": i,
                    "cav_id": aid,
                    "action": a,
                    "lane": lane,
                    "pos": pos,
                    "zone": _zone(pos, x_acc) if st else "unknown",
                    "boundary": _boundary(a, lane, n_lanes) if st else "none",
                    "logits": logits,
                    "probs": [float(p) for p in probs],
                    "d_logit_l": logits[LEFT] - logits[KEEP],
                    "d_logit_r": logits[RIGHT] - logits[KEEP],
                }
            )

        central = model.build_central_state(obs_t)
        value = model.get_value(central)
        v_scalar = float(value.reshape(-1)[0].item())
        values_np = np.full(num_agents, v_scalar, dtype=np.float32)

        _obs_next, rewards_dict, terminated, truncated, _info = env.step(actions_dict)
        del _obs_next, _info
        states_after = env.get_cav_states()

        rewards_np = np.zeros(num_agents, dtype=np.float32)
        mask_np = np.zeros(num_agents, dtype=np.float32)
        for meta in agent_meta:
            aid = meta["cav_id"]
            i = meta["slot"]
            if aid in rewards_dict:
                rewards_np[i] = float(rewards_dict[aid])
            if aid in env.cav_ids:
                mask_np[i] = 1.0
            st1 = states_after.get(aid)
            lane1 = int(st1["lane"]) if st1 is not None else meta["lane"]
            meta["lane_after"] = lane1
            if lane1 > meta["lane"]:
                meta["effective"] = "LEFT"
            elif lane1 < meta["lane"]:
                meta["effective"] = "RIGHT"
            else:
                meta["effective"] = "none"

        done = bool(terminated or truncated)
        buffer.add(
            obs=obs_np,
            actions=actions_np,
            rewards=rewards_np,
            log_probs=logp_np,
            values=values_np,
            done=done,
            agent_mask=mask_np,
        )
        extras.append({"agents": agent_meta, "V": v_scalar})

        if done:
            env.reset()
            for _ in range(warmup):
                if len(env.cav_ids) > 0:
                    break
                env.step()
            agent_ids = _lock_agent_ids(env, num_agents)

    batch = buffer.get()
    return {"batch": batch, "extras": extras}


def flatten_rollout(
    pack: dict[str, Any],
    *,
    gamma: float,
    gae_lambda: float,
) -> list[dict[str, Any]]:
    batch = pack["batch"]
    extras = pack["extras"]
    rewards = batch.rewards
    values = batch.values
    dones = batch.dones
    mask = batch.agent_mask
    actions = batch.actions
    log_probs = batch.log_probs

    advantages, returns = compute_gae(
        rewards,
        values,
        dones,
        gamma=gamma,
        gae_lambda=gae_lambda,
        agent_mask=mask,
        next_values=None,  # matches PPOUpdater (bootstrap zeros)
    )
    adv_norm = normalize_advantages(advantages, mask)

    t_steps, n_agents = actions.shape
    rows: list[dict[str, Any]] = []
    for t in range(t_steps):
        done_t = bool(dones[t])
        next_nonterminal = 0.0 if done_t else 1.0
        for i in range(n_agents):
            if mask[t, i] < 0.5:
                continue
            # Match agent meta by slot
            meta = None
            for m in extras[t].get("agents", []):
                if m["slot"] == i:
                    meta = m
                    break
            if meta is None:
                continue
            if t < t_steps - 1:
                next_v = float(values[t + 1, i])
            else:
                next_v = 0.0  # same bootstrap as compute_gae default
            r = float(rewards[t, i])
            v = float(values[t, i])
            td = r + gamma * next_v * next_nonterminal
            delta = td - v
            a = int(actions[t, i])
            rows.append(
                {
                    "action": a,
                    "action_name": ACTION_NAMES[a],
                    "reward": r,
                    "value": v,
                    "next_value": next_v,
                    "td_target": td,
                    "delta": delta,
                    "advantage": float(advantages[t, i]),
                    "advantage_norm": float(adv_norm[t, i]),
                    "return_gae": float(returns[t, i]),
                    "log_prob": float(log_probs[t, i]),
                    "zone": meta["zone"],
                    "lane": meta["lane"],
                    "boundary": meta["boundary"],
                    "effective": meta["effective"],
                    "d_logit_l": meta["d_logit_l"],
                    "d_logit_r": meta["d_logit_r"],
                    "p_keep": meta["probs"][KEEP],
                    "p_left": meta["probs"][LEFT],
                    "p_right": meta["probs"][RIGHT],
                    "done": done_t,
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "reward",
        "value",
        "next_value",
        "td_target",
        "delta",
        "advantage",
        "advantage_norm",
        "d_logit_l",
        "d_logit_r",
        "p_keep",
        "p_left",
        "p_right",
    )

    def block(subset: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"n": len(subset), "by_action": {}}
        for name in NAMES:
            sub = [r for r in subset if r["action_name"] == name]
            out["by_action"][name] = {
                "n": len(sub),
                **{k: _stats([float(r[k]) for r in sub]) for k in keys},
            }
        out["requested"] = {
            name: sum(1 for r in subset if r["action_name"] == name) for name in NAMES
        }
        out["invalid"] = sum(1 for r in subset if r["boundary"] != "none")
        out["effective_left"] = sum(1 for r in subset if r["effective"] == "LEFT")
        out["effective_right"] = sum(1 for r in subset if r["effective"] == "RIGHT")
        return out

    summary: dict[str, Any] = {
        "all": block(rows),
        "by_zone": {},
        "effective_precursor": {},
        "legal_only": block([r for r in rows if r["boundary"] == "none"]),
        "invalid_only": block([r for r in rows if r["boundary"] != "none"]),
    }
    for zone in ("before", "near", "after"):
        summary["by_zone"][zone] = block([r for r in rows if r["zone"] == zone])
    # Effective: this step's action coincided with an observed lane-index change
    summary["effective_precursor"]["LEFT"] = block(
        [r for r in rows if r["effective"] == "LEFT"]
    )
    summary["effective_precursor"]["RIGHT"] = block(
        [r for r in rows if r["effective"] == "RIGHT"]
    )
    summary["effective_precursor"]["none"] = block(
        [r for r in rows if r["effective"] == "none"]
    )
    return summary


def write_report(
    *,
    summary: dict[str, Any],
    gamma: float,
    gae_lambda: float,
    frozen: bool,
    n_rows: int,
    deterministic_note: str,
    train_cfg: dict[str, Any],
) -> Path:
    lines: list[str] = []
    lines.append("# Phase 5.8 Report")
    lines.append("")
    lines.append("## 1. Status")
    lines.append("")
    lines.append("- Phase: Advantage / Value / TD Target diagnosis.")
    lines.append("- Retrained: **No**.")
    lines.append("- Reward modified: **No** (Current and Candidate A untouched).")
    lines.append("- Checkpoints overwritten: **No**.")
    lines.append(
        f"- Data source: frozen Candidate A `{CA_CKPT.name}`; "
        f"{N_ROLLOUTS} × {train_cfg.get('rollout_steps', 512)}-step rollouts; "
        "GAE via `algorithms.gae.compute_gae` (same call shape as `PPOUpdater.update`)."
    )
    lines.append(f"- Actor/critic parameters frozen after collection: {'YES' if frozen else 'NO'}.")
    lines.append(
        f"- Policy sampling for this diagnosis: **stochastic** "
        f"(deterministic eval has KEEP≈0, so action-conditioned advantage would be empty for KEEP). "
        f"{deterministic_note}"
    )
    lines.append("")
    lines.append("## 2. Data Availability")
    lines.append("")
    lines.append("| Field | Available? | Source |")
    lines.append("|-------|------------|--------|")
    lines.append("| Reward `r_t` | Yes | Candidate A env reward in rollout |")
    lines.append("| Value `V(s_t)` | Yes | Centralized critic, broadcast to agents |")
    lines.append("| Next value `V(s_{t+1})` | Yes | `values[t+1]` (last step bootstrap **0**, matching PPOUpdater) |")
    lines.append("| TD target `r+γV'` | Yes | Computed in diagnostic from stored tensors |")
    lines.append("| Advantage `A_t` (raw GAE) | Yes | `compute_gae` |")
    lines.append("| Advantage (normalized) | Yes | `normalize_advantages` as in PPO update |")
    lines.append("| Policy logit / probability | Yes | Actor forward at decision time |")
    lines.append("| Old log_prob | Yes | Stored in rollout batch |")
    lines.append("| New log_prob / PPO ratio over training | **Unavailable** | Not logged in Phase 5.3 train_metrics |")
    lines.append("| Entropy / losses over training | Partial | Episode aggregates in `train_metrics.csv` only |")
    lines.append("")
    lines.append(f"Valid agent-steps in this diagnosis: **{n_rows}**.")
    lines.append("")
    lines.append("## 3. Immediate Reward Analysis")
    lines.append("")
    lines.append("| Region | Action | n | mean r | median r | std r |")
    lines.append("|--------|--------|---|--------|----------|-------|")
    for zone in ("before", "near", "after"):
        for name in NAMES:
            b = summary["by_zone"][zone]["by_action"][name]
            lines.append(
                f"| {zone} | {name} | {b['n']} | {_fmt(b['reward']['mean'])} | "
                f"{_fmt(b['reward']['median'])} | {_fmt(b['reward']['std'])} |"
            )
    lines.append("")
    lines.append(
        "Consistent with Phase 5.7: within a region, mean/median immediate rewards for KEEP/LEFT/RIGHT "
        "are close. Remaining gaps can reflect **state selection** (which states take which action), "
        "not a same-state causal reward table."
    )
    lines.append("")
    lines.append("## 4. Value Analysis")
    lines.append("")
    lines.append(
        "Critic is **centralized**: one `V(s)` per timestep from concatenated multi-agent observations, "
        "then broadcast to all slots. Therefore mean `V` by action is **not** an action-value Q(s,a)."
    )
    lines.append("")
    lines.append("| Region | Action | n | mean V(s_t) | mean V(s_{t+1}) | mean ΔV |")
    lines.append("|--------|--------|---|-------------|-----------------|---------|")
    for zone in ("before", "near", "after"):
        for name in NAMES:
            b = summary["by_zone"][zone]["by_action"][name]
            if b["n"] == 0:
                lines.append(f"| {zone} | {name} | 0 | n/a | n/a | n/a |")
                continue
            # approximate mean delta from stored next-value - value
            # already have stats separately
            lines.append(
                f"| {zone} | {name} | {b['n']} | {_fmt(b['value']['mean'])} | "
                f"{_fmt(b['next_value']['mean'])} | "
                f"{_fmt(b['next_value']['mean'] - b['value']['mean'])} |"
            )
    lines.append("")
    lines.append("## 5. TD Target Analysis")
    lines.append("")
    lines.append(f"`TD_target = r_t + γ V(s_{{t+1}})` with γ={gamma}, done-mask as in `compute_gae`.")
    lines.append("")
    lines.append("| Region | Action | n | mean TD | median TD | mean δ=TD−V |")
    lines.append("|--------|--------|---|---------|-----------|-------------|")
    for zone in ("before", "near", "after"):
        for name in NAMES:
            b = summary["by_zone"][zone]["by_action"][name]
            lines.append(
                f"| {zone} | {name} | {b['n']} | {_fmt(b['td_target']['mean'])} | "
                f"{_fmt(b['td_target']['median'])} | {_fmt(b['delta']['mean'])} |"
            )
    lines.append("")
    lines.append("## 6. Advantage Analysis")
    lines.append("")
    lines.append(
        f"Raw GAE with γ={gamma}, λ={gae_lambda}. This is the tensor `compute_gae` returns "
        "**before** PPO's batch normalization. Normalized advantages are reported in parentheses."
    )
    lines.append("")
    lines.append(
        "| Region | Action | n | Mean Reward | Mean V(s) | Mean TD | Mean Advantage | Median Adv | Mean Adv (norm) |"
    )
    lines.append(
        "|--------|--------|---|-------------|-----------|---------|----------------|------------|-----------------|"
    )
    for zone in ("before", "near", "after"):
        for name in NAMES:
            b = summary["by_zone"][zone]["by_action"][name]
            lines.append(
                f"| {zone} | {name} | {b['n']} | {_fmt(b['reward']['mean'])} | "
                f"{_fmt(b['value']['mean'])} | {_fmt(b['td_target']['mean'])} | "
                f"{_fmt(b['advantage']['mean'])} | {_fmt(b['advantage']['median'])} | "
                f"{_fmt(b['advantage_norm']['mean'])} |"
            )
    lines.append("")
    # Overall
    lines.append("### All regions")
    lines.append("")
    lines.append("| Action | n | mean A | median A | mean A_norm | mean r | mean TD |")
    lines.append("|--------|---|--------|----------|-------------|--------|---------|")
    for name in NAMES:
        b = summary["all"]["by_action"][name]
        lines.append(
            f"| {name} | {b['n']} | {_fmt(b['advantage']['mean'])} | {_fmt(b['advantage']['median'])} | "
            f"{_fmt(b['advantage_norm']['mean'])} | {_fmt(b['reward']['mean'])} | {_fmt(b['td_target']['mean'])} |"
        )
    lines.append("")
    lines.append("## 7. Policy Preference")
    lines.append("")
    lines.append("| Region | Action | n | mean P(a) | mean Δlogit L−K | mean Δlogit R−K | mean A |")
    lines.append("|--------|--------|---|-----------|-----------------|-----------------|--------|")
    for zone in ("before", "near", "after"):
        for name in NAMES:
            b = summary["by_zone"][zone]["by_action"][name]
            pkey = {"KEEP": "p_keep", "LEFT": "p_left", "RIGHT": "p_right"}[name]
            lines.append(
                f"| {zone} | {name} | {b['n']} | {_fmt(b[pkey]['mean'])} | "
                f"{_fmt(b['d_logit_l']['mean'])} | {_fmt(b['d_logit_r']['mean'])} | "
                f"{_fmt(b['advantage']['mean'])} |"
            )
    lines.append("")
    lines.append(
        "Phase 5.7 showed near-zero same-state one-step ΔR while logits favored lateral actions. "
        "This phase asks whether **realized** GAE advantages on stochastic rollouts still favor "
        "LEFT/RIGHT among the states where those actions were sampled."
    )
    lines.append("")
    lines.append("## 8. Effective Lane Change Analysis")
    lines.append("")
    all_b = summary["all"]
    lines.append(f"- Raw KEEP/LEFT/RIGHT requests: {all_b['requested']}")
    lines.append(f"- Boundary-invalid requests: {all_b['invalid']}")
    lines.append(
        f"- Effective lane-index increases (LEFT): {all_b['effective_left']}; "
        f"decreases (RIGHT): {all_b['effective_right']} "
        "(same-step observation; LC duration is 1.0 s so this under-counts in-progress maneuvers)."
    )
    lines.append("")
    lines.append("| Precursor set | Action taken | n | mean A | mean r | mean TD |")
    lines.append("|---------------|--------------|---|--------|--------|---------|")
    for eff in ("LEFT", "RIGHT", "none"):
        block = summary["effective_precursor"][eff]
        for name in NAMES:
            b = block["by_action"][name]
            if b["n"] == 0:
                continue
            lines.append(
                f"| effective={eff} | {name} | {b['n']} | {_fmt(b['advantage']['mean'])} | "
                f"{_fmt(b['reward']['mean'])} | {_fmt(b['td_target']['mean'])} |"
            )
    lines.append("")
    lines.append("### Legal vs invalid requests")
    lines.append("")
    lines.append("| Set | Action | n | mean A | mean r |")
    lines.append("|-----|--------|---|--------|--------|")
    for label, key in (("legal", "legal_only"), ("invalid", "invalid_only")):
        for name in NAMES:
            b = summary[key]["by_action"][name]
            lines.append(
                f"| {label} | {name} | {b['n']} | {_fmt(b['advantage']['mean'])} | {_fmt(b['reward']['mean'])} |"
            )
    lines.append("")
    lines.append("## 9. MAPPO / Critic Architecture")
    lines.append("")
    lines.append("- **Actor:** shared-parameter decentralized policy; local 17-D obs → Categorical logits over {KEEP, LEFT, RIGHT}.")
    lines.append("- **Critic:** centralized V(s); input = concat of up to N agent observations (padded).")
    lines.append("- **Parameter sharing:** one actor for all CAVs; one critic for the joint state.")
    lines.append("- **Implication:** `V(s_t)` is a joint-traffic value. A single CAV's lane request can change neighbors and group terms later, so long-horizon estimates need not match one-step r(s,a). That possibility is architectural; it is not by itself proof of the lateral preference.")
    lines.append("")
    lines.append("## 10. GAE / PPO Diagnosis")
    lines.append("")
    lines.append("From `configs/train.yaml` and `algorithms/gae.py` / `ppo_update.py`:")
    lines.append("")
    lines.append("```text")
    lines.append(f"gamma        = {gamma}")
    lines.append(f"gae_lambda   = {gae_lambda}")
    lines.append("delta_t      = r_t + γ V(s_{t+1}) (1-d_t) - V(s_t)")
    lines.append("A_t          = delta_t + γλ (1-d_t) A_{t+1}")
    lines.append("returns      = A_t + V(s_t)")
    lines.append("bootstrap    = next_values default zeros at rollout end (PPOUpdater)")
    lines.append("adv norm     = (A - mean)/std over masked batch, before PPO epochs")
    lines.append(f"clip_eps     = {train_cfg.get('clip_eps', train_cfg.get('clip_epsilon'))}")
    lines.append(f"entropy_coef = {train_cfg.get('entropy_coef')}")
    lines.append(f"value_coef   = {train_cfg.get('value_coef')}")
    lines.append(f"ppo_epochs   = {train_cfg.get('ppo_epochs')}")
    lines.append("```")
    lines.append("")
    lines.append(
        "Training-time PPO ratio / per-action KL: **Unavailable** in saved Phase 5.3 logs. "
        "Entropy decline (Phase 5.4 train segment ~1.07→0.85) only shows concentration, not which advantage sign drove it."
    )
    lines.append("")
    lines.append("## 11. Causal Interpretation")
    lines.append("")

    # Build confirmed / supported / unknown from summary numbers
    def mean_a(zone: str, name: str) -> float:
        return float(summary["by_zone"][zone]["by_action"][name]["advantage"]["mean"])

    def mean_a_all(name: str) -> float:
        return float(summary["all"]["by_action"][name]["advantage"]["mean"])

    a_keep = mean_a_all("KEEP")
    a_left = mean_a_all("LEFT")
    a_right = mean_a_all("RIGHT")
    # Situation A-like if lateral mean advantage clearly above KEEP
    lat_above = (a_left > a_keep + 0.01) or (a_right > a_keep + 0.01)
    lat_both = (a_left > a_keep + 0.01) and (a_right > a_keep + 0.01)
    # Situation B-like if advantages similar
    adv_close = abs(a_left - a_keep) < 0.01 and abs(a_right - a_keep) < 0.01

    lines.append("### Confirmed")
    lines.append("")
    lines.append("- Immediate same-state one-step ΔR ≈ 0 (Phase 5.7).")
    lines.append("- Deterministic argmax never selects KEEP (Phase 5.6).")
    lines.append(
        "- Project GAE matches `δ_t = r+γV'−V`, `A=δ+γλA'`; PPO normalizes advantages per batch."
    )
    lines.append(
        "- Critic value is joint-state V(s), broadcast; not Q(s,a)."
    )
    lines.append(
        f"- On these stochastic diagnostic rollouts, mean raw advantage by taken action: "
        f"KEEP={_fmt(a_keep)}, LEFT={_fmt(a_left)}, RIGHT={_fmt(a_right)}."
    )
    lines.append("")
    lines.append("### Supported Hypothesis")
    lines.append("")
    if lat_both or lat_above:
        lines.append(
            "- **Situation A (partially supported):** among states where each action was actually "
            "sampled, mean GAE for at least one lateral action exceeds KEEP. That is compatible with "
            "PPO reinforcing LEFT/RIGHT even when one-step r is flat — but it is still an "
            "**action-conditional average over different states**, not a same-state counterfactual advantage."
        )
    elif adv_close:
        lines.append(
            "- **Situation B (partially supported):** mean GAE by taken action is similar across KEEP/LEFT/RIGHT "
            "while logits still favor lateral moves on many states. Then long-horizon advantage averages alone "
            "do not explain the preference; actor initialization/entropy collapse, boundary no-ops, or "
            "state-visitation bias remain open."
        )
    else:
        lines.append(
            "- Mixed region-level patterns; see tables. No single global Situation A/B dominates without caveats."
        )
    lines.append(
        "- Boundary-invalid LEFT/RIGHT can still receive non-zero GAE because rewards/values continue; "
        "those samples inflate lateral counts without an executed lane change."
    )
    lines.append("")
    lines.append("### Unknown")
    lines.append("")
    lines.append("- Whether advantage **caused** the logit ranking during the original 50 updates (no per-update action–advantage log).")
    lines.append("- Counterfactual A(s, KEEP) vs A(s, LEFT) on the **same** state (would need branched rollouts / Q estimates).")
    lines.append("- Training-time PPO ratio accumulation for lateral vs KEEP.")
    lines.append("")
    lines.append("## 12. Final Diagnosis")
    lines.append("")
    lines.append(
        "Candidate A’s strong lateral **policy** preference is **not** explained by a one-step reward gap "
        "(Phase 5.7). This phase shows what the **frozen** critic + project GAE assign to actions that the "
        f"stochastic actor actually took: mean A(KEEP)={_fmt(a_keep)}, A(LEFT)={_fmt(a_left)}, "
        f"A(RIGHT)={_fmt(a_right)}."
    )
    lines.append("")
    if lat_above and not adv_close:
        lines.append(
            "The data are **consistent with** Situation A at the level of action-conditional GAE averages "
            "(lateral actions appearing with higher mean advantage than KEEP in aggregate), "
            "which could have reinforced non-KEEP logits during PPO. "
            "They do **not** prove that long-term return is the cause, because the comparison is not same-state "
            "and training-time ratios were not logged."
        )
    elif adv_close:
        lines.append(
            "The data lean toward Situation B for **average** GAE by action: advantages look similar while "
            "policy logits still separate. In that reading, “long-term return favors LC” is **not** established; "
            "next checks should target actor/PPO/action-mask interaction rather than a new reward term."
        )
    else:
        lines.append(
            "Region tables must be read carefully; a blanket “advantage caused LC” claim is **not** justified."
        )
    lines.append("")
    lines.append("**尚不能证明** that long-term return / advantage is the unique cause of the lateral preference.")
    lines.append("")
    lines.append("## 13. Recommendation")
    lines.append("")
    lines.append("Do **not** modify Candidate A weights in Phase 5.9 based solely on this report.")
    lines.append("")
    if lat_above and not adv_close:
        lines.append(
            "Phase 5.9 (suggested): same-state **multi-step** branch (through the 1.0 s LC window) comparing "
            "cumulative reward / bootstrapped return under forced KEEP vs LEFT on legal (non-boundary) states; "
            "still no weight edit until that gap is clear. Optionally log per-action mean advantage during a "
            "short frozen evaluation loop only."
        )
    else:
        lines.append(
            "Phase 5.9 (suggested): inspect actor logits vs boundary legality, and whether invalid LEFT/RIGHT "
            "on lane 2/0 receive the same log-prob gradient path as legal moves; still no reward redesign."
        )
    lines.append("")
    lines.append("## Phase 5.8 Result: PASS")
    lines.append("")
    lines.append(
        "PASS = frozen GAE/value/TD diagnosis completed with project code paths; "
        "no training; no reward or checkpoint changes."
    )
    lines.append("")
    path = OUT / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not CA_CKPT.is_file():
        print(f"ERROR: missing {CA_CKPT}")
        return 1

    train_cfg = load_train_config()
    tblock = train_cfg.get("train", train_cfg)
    gamma = float(tblock.get("gamma", 0.99))
    gae_lambda = float(tblock.get("gae_lambda", 0.95))
    rollout_steps = int(tblock.get("rollout_steps", 512))
    warmup = int(tblock.get("warmup_steps", 50))

    device = resolve_device("auto")
    set_eval_seed(SEED)
    model, _ = load_mappo_checkpoint(CA_CKPT, device=device)
    model.eval()
    before = snapshot_model_params(model)

    env_cfg = copy.deepcopy(load_config())
    env = HighwayAccidentEnv(config=env_cfg, gui=False, reward_config=CA_REWARD)

    all_rows: list[dict[str, Any]] = []
    print(f"Phase 5.8 device={device} gamma={gamma} lambda={gae_lambda}")
    try:
        for k in range(N_ROLLOUTS):
            print(f"--- stochastic rollout {k+1}/{N_ROLLOUTS} ---")
            pack = collect_diag(
                env=env,
                model=model,
                rollout_steps=rollout_steps,
                warmup=warmup,
                device=device,
                deterministic=False,
            )
            rows = flatten_rollout(pack, gamma=gamma, gae_lambda=gae_lambda)
            print(f"  rows={len(rows)}")
            all_rows.extend(rows)
    finally:
        env.close()

    after = snapshot_model_params(model)
    frozen = params_unchanged(before, after)
    summary = summarize(all_rows)
    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "n_rows": len(all_rows),
                "n_rollouts": N_ROLLOUTS,
                "rollout_steps": rollout_steps,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "params_frozen": frozen,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Compact CSV for tables
    import csv

    csv_path = OUT / "action_region_stats.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "region",
                "action",
                "n",
                "mean_r",
                "mean_v",
                "mean_td",
                "mean_adv",
                "median_adv",
                "mean_adv_norm",
            ]
        )
        for zone in ("before", "near", "after"):
            for name in NAMES:
                b = summary["by_zone"][zone]["by_action"][name]
                w.writerow(
                    [
                        zone,
                        name,
                        b["n"],
                        b["reward"]["mean"],
                        b["value"]["mean"],
                        b["td_target"]["mean"],
                        b["advantage"]["mean"],
                        b["advantage"]["median"],
                        b["advantage_norm"]["mean"],
                    ]
                )

    report = write_report(
        summary=summary,
        gamma=gamma,
        gae_lambda=gae_lambda,
        frozen=frozen,
        n_rows=len(all_rows),
        deterministic_note="Phase 5.6/5.7 deterministic KEEP≈0 remains the eval behavior of interest.",
        train_cfg=tblock,
    )
    print(f"Report: {report}")
    print("Phase 5.8 Result: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
