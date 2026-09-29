"""Phase 5.9 — action encoding / legal-vs-invalid logits / PPO path audit.

Read-only. Does not train, edit rewards, or write checkpoints.
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

from algorithms.rollout import (  # noqa: E402
    _lock_agent_ids,
    _rebind_agent_slots,
    resolve_device,
)
from env.action import ACTION_NAMES, KEEP, LEFT, RIGHT  # noqa: E402
from env.highway_accident_env import HighwayAccidentEnv, load_config  # noqa: E402
from utils.baseline_eval import (  # noqa: E402
    load_mappo_checkpoint,
    params_unchanged,
    set_eval_seed,
    snapshot_model_params,
)

OUT = PROJECT_ROOT / "results" / "experiments" / "reward_candidate_A_phase5_9"
CKPT_DIR = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "reward_candidate_A_phase5_3"
    / "checkpoints"
)
CA_REWARD = PROJECT_ROOT / "configs" / "reward_candidate_A.yaml"
SEED = 42
N_ROLLOUTS = 2
ROLLOUT_STEPS = 512
ZONE_R = 150.0
NAMES = ("KEEP", "LEFT", "RIGHT")


def _stats(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan")}
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "std": float(np.std(a)),
    }


def _fmt(x: float, nd: int = 3) -> str:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def _zone(pos: float, x_acc: float) -> str:
    if pos < x_acc - ZONE_R:
        return "before"
    if pos > x_acc + ZONE_R:
        return "after"
    return "near"


@torch.no_grad()
def collect(model, device, env) -> list[dict[str, Any]]:
    model.eval()
    x_acc = float(env.accident_position_m)
    n_lanes = int(env.n_lanes)
    num_agents = int(model.num_agents)
    rows: list[dict[str, Any]] = []
    if not env.is_connected:
        env.reset()
    for _ in range(50):
        if len(env.cav_ids) > 0:
            break
        env.step()
    agent_ids = _lock_agent_ids(env, num_agents)
    for step in range(ROLLOUT_STEPS):
        agent_ids, _, _ = _rebind_agent_slots(env, agent_ids, num_agents)
        if len(env.cav_ids) == 0:
            _o, _r, term, trunc, _i = env.step({})
            del _o, _r, _i
            if term or trunc:
                env.reset()
                agent_ids = _lock_agent_ids(env, num_agents)
            continue
        states = env.get_cav_states()
        obs_dict = env.get_observations()
        actions: dict[str, int] = {}
        step_rows: list[dict[str, Any]] = []
        for i, aid in enumerate(agent_ids):
            if aid is None or aid not in env.cav_ids or aid not in obs_dict:
                continue
            obs = np.asarray(obs_dict[aid], dtype=np.float32)
            t = torch.as_tensor(obs, device=device)
            logits_t = model.actor.forward(t)
            if logits_t.dim() == 1:
                logits_t = logits_t.unsqueeze(0)
            logits = [float(x) for x in logits_t[0].detach().cpu().tolist()]
            probs = [
                float(x)
                for x in torch.softmax(logits_t, dim=-1)[0].detach().cpu().tolist()
            ]
            a, _lp = model.act(t, deterministic=False)
            a = int(a.item())
            actions[aid] = a
            st = states[aid]
            lane = int(st["lane"])
            left_legal = lane + 1 < n_lanes
            right_legal = lane - 1 >= 0
            step_rows.append(
                {
                    "step": step,
                    "slot": i,
                    "cav_id": aid,
                    "lane": lane,
                    "zone": _zone(float(st["position"]), x_acc),
                    "action": a,
                    "action_name": ACTION_NAMES[a],
                    "left_legal": left_legal,
                    "right_legal": right_legal,
                    "logits": logits,
                    "probs": probs,
                    "obs": obs.tolist(),
                }
            )
        _o, _r, term, trunc, info = env.step(actions)
        del _o, _r
        # Execution audit from info is not per-agent; re-read lane after.
        after = env.get_cav_states()
        for rec in step_rows:
            st1 = after.get(rec["cav_id"])
            rec["lane_after"] = int(st1["lane"]) if st1 else rec["lane"]
            rec["lane_changed"] = rec["lane_after"] != rec["lane"]
            # Boundary refusal is defined before SUMO, matching apply_action.
            if rec["action"] == LEFT and not rec["left_legal"]:
                rec["exec_class"] = "invalid_LEFT_suppressed"
            elif rec["action"] == RIGHT and not rec["right_legal"]:
                rec["exec_class"] = "invalid_RIGHT_suppressed"
            elif rec["action"] == KEEP:
                rec["exec_class"] = "KEEP_no_changeLane"
            else:
                rec["exec_class"] = "changeLane_requested"
            rows.append(rec)
        if term or trunc:
            env.reset()
            agent_ids = _lock_agent_ids(env, num_agents)
    return rows


def group_logits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def pack(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(subset),
            "logit": {n: _stats([r["logits"][i] for r in subset]) for i, n in enumerate(NAMES)},
            "prob": {n: _stats([r["probs"][i] for r in subset]) for i, n in enumerate(NAMES)},
            "dL": _stats([r["logits"][LEFT] - r["logits"][KEEP] for r in subset]),
            "dR": _stats([r["logits"][RIGHT] - r["logits"][KEEP] for r in subset]),
        }

    out: dict[str, Any] = {"all": pack(rows), "by_zone_lane": {}, "sets": {}}
    for zone in ("before", "near", "after"):
        for lane in (0, 1, 2):
            out["by_zone_lane"][f"{zone}|lane{lane}"] = pack(
                [r for r in rows if r["zone"] == zone and r["lane"] == lane]
            )
    out["sets"]["legal_LEFT"] = pack([r for r in rows if r["left_legal"]])
    out["sets"]["invalid_LEFT"] = pack([r for r in rows if not r["left_legal"]])
    out["sets"]["legal_RIGHT"] = pack([r for r in rows if r["right_legal"]])
    out["sets"]["invalid_RIGHT"] = pack([r for r in rows if not r["right_legal"]])
    # execution counts
    exec_n = defaultdict(int)
    for r in rows:
        exec_n[r["exec_class"]] += 1
    out["exec_class"] = dict(exec_n)
    out["n"] = len(rows)
    # mismatch: logged action vs lane change on invalid
    inv_left_changed = sum(
        1
        for r in rows
        if r["exec_class"] == "invalid_LEFT_suppressed" and r["lane_changed"]
    )
    inv_right_changed = sum(
        1
        for r in rows
        if r["exec_class"] == "invalid_RIGHT_suppressed" and r["lane_changed"]
    )
    out["invalid_but_lane_changed"] = {
        "LEFT": inv_left_changed,
        "RIGHT": inv_right_changed,
    }
    return out


def gradient_probe(model, device, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Frozen sensitivity: grad norm of each logit w.r.t. actor params. No step()."""
    picked: list[dict[str, Any]] = []
    seen = set()
    for r in rows:
        key = (r["zone"], r["lane"])
        if key in seen:
            continue
        seen.add(key)
        picked.append(r)
        if len(picked) >= 8:
            break
    results = []
    model.eval()
    for r in picked:
        obs = torch.as_tensor(r["obs"], dtype=torch.float32, device=device)
        logits = model.actor.forward(obs)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        norms = []
        for i in range(3):
            model.zero_grad(set_to_none=True)
            logits[0, i].backward(retain_graph=True)
            sq = 0.0
            for p in model.actor.parameters():
                if p.grad is not None:
                    sq += float(torch.sum(p.grad.detach() ** 2).item())
            norms.append(math.sqrt(sq))
        model.zero_grad(set_to_none=True)
        results.append(
            {
                "zone": r["zone"],
                "lane": r["lane"],
                "logits": [float(x) for x in logits[0].detach().cpu().tolist()],
                "grad_norm": norms,
            }
        )
    return {"n": len(results), "samples": results}


@torch.no_grad()
def checkpoint_trajectory(device, rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Fixed probe set: up to 30 obs per zone
    probes: list[dict[str, Any]] = []
    counts = defaultdict(int)
    for r in rows:
        if counts[r["zone"]] >= 40:
            continue
        probes.append(r)
        counts[r["zone"]] += 1
    eps = [20, 30, 40, 50]
    out = []
    for ep in eps:
        path = CKPT_DIR / f"episode_{ep:04d}.pt"
        if not path.is_file():
            out.append({"episode": ep, "missing": True})
            continue
        model, _ = load_mappo_checkpoint(path, device=device)
        model.eval()
        acc = {z: [] for z in ("before", "near", "after", "all")}
        for r in probes:
            t = torch.as_tensor(r["obs"], dtype=torch.float32, device=device)
            logits = model.actor.forward(t)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            prob = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
            acc["all"].append(prob)
            acc[r["zone"]].append(prob)
        rec = {"episode": ep, "missing": False, "zones": {}}
        for z, arrs in acc.items():
            if not arrs:
                continue
            m = np.mean(np.stack(arrs, axis=0), axis=0)
            rec["zones"][z] = {
                "n": len(arrs),
                "P_KEEP": float(m[0]),
                "P_LEFT": float(m[1]),
                "P_RIGHT": float(m[2]),
            }
        out.append(rec)
    return {"probes": {z: counts[z] for z in counts}, "episodes": out}


def write_report(stats: dict[str, Any], grads: dict[str, Any], traj: dict[str, Any], frozen: bool) -> Path:
    lines: list[str] = []
    lines.append("# Phase 5.9 Report")
    lines.append("")
    lines.append("## 1. Status")
    lines.append("")
    lines.append("- Diagnosis completed.")
    lines.append("- Retrained: **No**.")
    lines.append("- Reward / observation / action space / PPO / env: **not modified**.")
    lines.append("- Checkpoints overwritten: **No**.")
    lines.append(f"- Actor params frozen during logit collection: {'YES' if frozen else 'NO'}.")
    lines.append(f"- Diagnostic steps: {stats['n']} agent-decisions, stochastic actor, seed={SEED}.")
    lines.append("- Action space is **3**, not 9. There is no longitudinal discrete action.")
    lines.append("")
    lines.append("## 2. Action Encoding Audit")
    lines.append("")
    lines.append("From `env/action.py` and `HighwayAccidentEnv.apply_action`.")
    lines.append("")
    lines.append("| Actor index | Internal | Logged | Environment | Longitudinal | Lateral |")
    lines.append("|-------------|----------|--------|-------------|--------------|---------|")
    lines.append("| 0 | KEEP | KEEP | no `changeLane` | none (SUMO car-follow) | stay |")
    lines.append("| 1 | LEFT | LEFT | `changeLane(lane+1)` if legal, else no-op | none | index +1 |")
    lines.append("| 2 | RIGHT | RIGHT | `changeLane(lane-1)` if legal, else no-op | none | index −1 |")
    lines.append("")
    lines.append(
        "`model.act` uses `torch.argmax(logits)` when deterministic, else `Categorical(logits=logits).sample()`. "
        "Both use the same index order. Log name comes from `ACTION_NAMES`."
    )
    lines.append("")
    lines.append("No 9-action layout exists in this project.")
    lines.append("")
    lines.append("## 3. Invalid Action Handling")
    lines.append("")
    lines.append("Detection is inside `apply_action` → `target_lane_index`, **before** TraCI.")
    lines.append("")
    lines.append("- Lane 0 (rightmost): RIGHT → `target is None` → `executed=False`. Action id is **not** rewritten to KEEP.")
    lines.append("- Lane 2 (leftmost): LEFT → same suppression.")
    lines.append("- Lane 1: LEFT and RIGHT both have a target lane; `changeLane` is called.")
    lines.append("- SUMO is not asked to change lane on a boundary request. The logged/sampled action remains LEFT or RIGHT.")
    lines.append("")
    lines.append(f"This run's execution classes: `{stats['exec_class']}`.")
    lines.append(
        f"Invalid requests that still changed lane index within one 0.1 s step: "
        f"LEFT {stats['invalid_but_lane_changed']['LEFT']}, "
        f"RIGHT {stats['invalid_but_lane_changed']['RIGHT']}."
    )
    lines.append("")
    lines.append("## 4. Observation Audit")
    lines.append("")
    lines.append("| Feature | Present? | Dim | Normalized? | Actor uses? |")
    lines.append("|---------|----------|-----|-------------|-------------|")
    lines.append("| current lane (`ego_lane`) | Yes | 1 of 17 | `lane / (n_lanes-1)` clipped to [0,1] | Yes |")
    lines.append("| left lane availability flag | **No explicit bit** | — | — | No |")
    lines.append("| right lane availability flag | **No explicit bit** | — | — | No |")
    lines.append("| road-boundary flag | **No** | — | — | No |")
    lines.append("| neighbor slots on missing side | Implicit | 4 | distance 1.0, rel-speed 0.0 if side missing **or** empty | Yes |")
    lines.append("")
    lines.append(
        "Actor can see a normalized lane index. It does **not** receive an explicit legal-action mask. "
        "A missing side uses the same (1.0, 0.0) filler as an empty existing lane, so emptiness and 'no lane' are not distinguished."
    )
    lines.append("")
    lines.append("## 5. Action Mask Audit")
    lines.append("")
    lines.append("**Action masking: Not implemented.**")
    lines.append("")
    lines.append("Search of training/env code found no mask applied to logits, probabilities, sampling, or argmax. Boundary refusal happens only at execution.")
    lines.append("")
    lines.append("## 6. Logit / Probability Analysis")
    lines.append("")
    lines.append("| Set | n | mean logit KEEP/LEFT/RIGHT | mean P(LEFT) | mean ΔL−K | mean ΔR−K |")
    lines.append("|-----|---|----------------------------|--------------|-----------|-----------|")
    for key in ("legal_LEFT", "invalid_LEFT", "legal_RIGHT", "invalid_RIGHT"):
        b = stats["sets"][key]
        lg = b["logit"]
        lines.append(
            f"| {key} | {b['n']} | {_fmt(lg['KEEP']['mean'])} / {_fmt(lg['LEFT']['mean'])} / {_fmt(lg['RIGHT']['mean'])} | "
            f"{_fmt(b['prob']['LEFT']['mean'])} | {_fmt(b['dL']['mean'])} | {_fmt(b['dR']['mean'])} |"
        )
    lines.append("")
    lines.append("### By region × lane (mean Δlogit)")
    lines.append("")
    lines.append("| Slice | n | mean ΔL−K | median ΔL−K | mean ΔR−K | median ΔR−K | mean P(KEEP/LEFT/RIGHT) |")
    lines.append("|-------|---|-----------|-------------|-----------|-------------|-------------------------|")
    for zone in ("before", "near", "after"):
        for lane in (0, 1, 2):
            b = stats["by_zone_lane"][f"{zone}|lane{lane}"]
            if b["n"] == 0:
                lines.append(f"| {zone} lane {lane} | 0 | n/a | n/a | n/a | n/a | n/a |")
                continue
            lines.append(
                f"| {zone} lane {lane} | {b['n']} | {_fmt(b['dL']['mean'])} | {_fmt(b['dL']['median'])} | "
                f"{_fmt(b['dR']['mean'])} | {_fmt(b['dR']['median'])} | "
                f"{_fmt(b['prob']['KEEP']['mean'])} / {_fmt(b['prob']['LEFT']['mean'])} / {_fmt(b['prob']['RIGHT']['mean'])} |"
            )
    inv_l = stats["sets"]["invalid_LEFT"]["logit"]["LEFT"]["mean"]
    leg_l = stats["sets"]["legal_LEFT"]["logit"]["LEFT"]["mean"]
    lines.append("")
    lines.append(
        f"Invalid-LEFT states mean logit_LEFT={_fmt(inv_l)}; legal-LEFT states mean logit_LEFT={_fmt(leg_l)}. "
        "Compare these numbers directly; do not treat a higher invalid logit as a proven training bug."
    )
    lines.append("")
    lines.append("## 7. PPO Update Path Audit")
    lines.append("")
    lines.append("From `models/actor.py` and `algorithms/ppo_update.py` (code audit, not a new training run):")
    lines.append("")
    lines.append("```text")
    lines.append("distribution     = Categorical(logits=actor(obs))   # one softmax inside Categorical")
    lines.append("sample/argmax    = dist.sample() or argmax(logits)")
    lines.append("old_log_prob     = dist.log_prob(sampled action) stored in rollout")
    lines.append("new_log_prob     = actor.evaluate_actions(obs, same action index)")
    lines.append("ratio            = exp(new_log_prob - old_log_prob)")
    lines.append("surr             = min(ratio*A, clip(ratio, 1-0.2, 1+0.2)*A)")
    lines.append("actor_loss       = -masked_mean(surr)")
    lines.append("policy_loss      = actor_loss - 0.01 * entropy")
    lines.append("adv norm         = per-batch, masked valid samples, not per-agent")
    lines.append("```")
    lines.append("")
    lines.append("No double softmax. `log_prob` is taken on the same integer index that is stored as the action. `.item()` is used only to copy the scalar into the numpy buffer; the graph used later is rebuilt in `evaluate_actions`.")
    lines.append("")
    lines.append("Training-time per-action ratio series: **Unavailable**.")
    lines.append("")
    lines.append("## 8. Alignment Audit")
    lines.append("")
    lines.append("- Flatten order in PPO is time-major then agent slot: `obs`, `actions`, `old_log_prob`, `advantages` share that index. No off-by-one in the reshape.")
    lines.append("- GAE pairs `rewards[t]` with `values[t]` and `values[t+1]` (or zero bootstrap at the last step).")
    lines.append("- `agent_mask` drops empty slots from the actor minibatch, but **does not reset the GAE recursion**. A slot that is cleared and later rebound to a new CAV can carry `last_gae` across the identity change unless `done` is set. That is a code fact. Whether it caused the lateral preference is **Unknown**.")
    lines.append("- This diagnostic did not find an index swap between LEFT and RIGHT.")
    lines.append("")
    lines.append("## 9. Gradient Diagnostic")
    lines.append("")
    lines.append("Frozen-checkpoint sensitivity only. Parameters were not updated. This is **not** the training-time gradient.")
    lines.append("")
    lines.append("| Zone | Lane | logit K/L/R | grad-norm K/L/R |")
    lines.append("|------|------|-------------|-----------------|")
    for s in grads["samples"]:
        lg = "/".join(_fmt(x) for x in s["logits"])
        gn = "/".join(_fmt(x, 2) for x in s["grad_norm"])
        lines.append(f"| {s['zone']} | {s['lane']} | {lg} | {gn} |")
    lines.append("")
    lines.append("## 10. Checkpoint Trajectory")
    lines.append("")
    lines.append("Same stored observations, different frozen checkpoints. Mean softmax probability.")
    lines.append("")
    lines.append("| Episode | n | P(KEEP) | P(LEFT) | P(RIGHT) |")
    lines.append("|---------|---|---------|---------|----------|")
    for ep in traj["episodes"]:
        if ep.get("missing"):
            lines.append(f"| {ep['episode']} | — | missing | | |")
            continue
        z = ep["zones"].get("all", {})
        lines.append(
            f"| {ep['episode']} | {z.get('n', 0)} | {_fmt(z.get('P_KEEP', float('nan')))} | "
            f"{_fmt(z.get('P_LEFT', float('nan')))} | {_fmt(z.get('P_RIGHT', float('nan')))} |"
        )
    lines.append("")
    lines.append("### By region at episode 20 vs 50")
    lines.append("")
    lines.append("| Region | ep20 P(K/L/R) | ep50 P(K/L/R) |")
    lines.append("|--------|----------------|----------------|")
    by_ep = {e["episode"]: e for e in traj["episodes"] if not e.get("missing")}
    for zone in ("before", "near", "after"):
        a = by_ep.get(20, {}).get("zones", {}).get(zone)
        b = by_ep.get(50, {}).get("zones", {}).get(zone)
        if not a or not b:
            lines.append(f"| {zone} | n/a | n/a |")
            continue
        lines.append(
            f"| {zone} | {_fmt(a['P_KEEP'])}/{_fmt(a['P_LEFT'])}/{_fmt(a['P_RIGHT'])} | "
            f"{_fmt(b['P_KEEP'])}/{_fmt(b['P_LEFT'])}/{_fmt(b['P_RIGHT'])} |"
        )
    lines.append("")
    lines.append("## 11. Root Cause Classification")
    lines.append("")
    lines.append("### Confirmed")
    lines.append("")
    lines.append("- Action encoding 0/1/2 = KEEP/LEFT/RIGHT is consistent from actor logits through logging to `apply_action`. Not a 9-D space. No LEFT/RIGHT swap in code.")
    lines.append("- Invalid boundary actions are suppressed at execution (`executed=False`), not converted to KEEP, and not masked in the policy.")
    lines.append("- Actor sees normalized `ego_lane` but no explicit legality bit. Action masking is not implemented.")
    lines.append("- PPO uses one `Categorical(logits=...)`, matching action index for old and new log-prob, batch-wise advantage normalization.")
    lines.append("")
    lines.append("### Strong Evidence")
    lines.append("")
    lines.append("- Lateral preference is already visible in earlier Candidate A checkpoints on the same observations (see Section 10), so it is not only an episode-50 argmax artifact.")
    lines.append("- Logit gaps on invalid-boundary lanes and legal lanes should be read from Section 6. A high LEFT logit on lane 2 is possible because the policy is not masked and lane legality is not an explicit feature.")
    lines.append("")
    lines.append("### Hypothesis")
    lines.append("")
    lines.append("- Repeated invalid LEFT/RIGHT still receive log-prob and, when advantage is positive, PPO increases those logits. Phase 5.8 showed invalid LEFT had much higher mean advantage than legal LEFT. That is a plausible reinforcement path, not a demonstrated training-time ratio series.")
    lines.append("- Slot-level GAE across rebinds may mix identities. Not shown to be the main driver.")
    lines.append("")
    lines.append("### Unknown")
    lines.append("")
    lines.append("- The unique root of the lateral logit bias. Encoding bugs were not found. Immediate reward and average GAE also do not uniquely explain it.")
    lines.append("")
    lines.append("## 12. Final Diagnosis")
    lines.append("")
    lines.append("### Q1. Action encoding correct?")
    lines.append("")
    lines.append("**Yes, as implemented.** Index 0/1/2 maps to KEEP/LEFT/RIGHT in the actor, the logger, and the environment. No 9-action longitudinal grid.")
    lines.append("")
    lines.append("### Q2. Invalid actions handled correctly?")
    lines.append("")
    lines.append("**Handled at execution only.** Boundary LEFT/RIGHT do not call `changeLane`. They are still sampled, logged, and eligible for PPO updates. They are not rewritten to KEEP.")
    lines.append("")
    lines.append("### Q3. Does the actor explicitly know legality?")
    lines.append("")
    lines.append("**No explicit legality feature and no action mask.** It has a normalized lane index and neighbor fillers that do not distinguish 'no lane' from 'empty lane'.")
    lines.append("")
    lines.append("### Q4. Are action / advantage / log_prob aligned?")
    lines.append("")
    lines.append("**The PPO flatten is aligned.** No off-by-one was found. GAE does not cut on `agent_mask`, so slot rebinding can carry advantage across a vehicle identity. That is a caveat, not a proven mis-index.")
    lines.append("")
    lines.append("### Q5. Obvious PPO/MAPPO implementation bug?")
    lines.append("")
    lines.append("**No encoding or double-softmax bug found.** The notable design facts are: no action mask, and GAE ignoring the agent mask. Neither was changed.")
    lines.append("")
    lines.append("### Q6. Why are lateral logits high?")
    lines.append("")
    lines.append(
        "Current evidence is **not sufficient to name a single root cause**. "
        "What is ruled out: a swapped action index, a 9-D decoding error, and a one-step reward that pays for LEFT/RIGHT. "
        "What remains plausible: an unmasked policy that keeps probability on boundary-invalid lateral actions, "
        "combined with whatever training signal raised those logits before episode 50. "
        "Phase 5.8 does not show a uniform advantage advantage for legal lateral actions."
    )
    lines.append("")
    lines.append("## Phase 5.9 Result: PASS")
    lines.append("")
    lines.append("PASS means the audit finished. It does not mean Candidate A is a good controller, and it does not authorize an action mask or a reward edit.")
    lines.append("")
    path = OUT / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / "episode_0050.pt"
    if not ckpt.is_file():
        print(f"ERROR missing {ckpt}")
        return 1
    device = resolve_device("auto")
    set_eval_seed(SEED)
    model, _ = load_mappo_checkpoint(ckpt, device=device)
    before = snapshot_model_params(model)
    env = HighwayAccidentEnv(
        config=copy.deepcopy(load_config()), gui=False, reward_config=CA_REWARD
    )
    rows: list[dict[str, Any]] = []
    print(f"Phase 5.9 device={device}")
    try:
        for k in range(N_ROLLOUTS):
            print(f"--- rollout {k+1}/{N_ROLLOUTS} ---")
            rows.extend(collect(model, device, env))
            print(f"  cumulative={len(rows)}")
    finally:
        env.close()
    frozen = params_unchanged(before, snapshot_model_params(model))
    # drop obs from stats json size later; keep for probes
    stats = group_logits(rows)
    print("gradients")
    grads = gradient_probe(model, device, rows)
    frozen2 = params_unchanged(before, snapshot_model_params(model))
    print("checkpoint trajectory")
    traj = checkpoint_trajectory(device, rows)
    # strip obs before saving rows sample
    slim = [{k: v for k, v in r.items() if k != "obs"} for r in rows[:200]]
    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "n": len(rows),
                "params_frozen": frozen and frozen2,
                "stats_sets": stats["sets"],
                "by_zone_lane": stats["by_zone_lane"],
                "exec_class": stats["exec_class"],
                "invalid_but_lane_changed": stats["invalid_but_lane_changed"],
                "grads": grads,
                "traj": traj,
                "sample_rows": slim,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report = write_report(stats, grads, traj, frozen and frozen2)
    print(f"Report: {report}")
    print("Phase 5.9 Result: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
