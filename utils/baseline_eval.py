"""Shared closed-loop evaluation for No Control / MAPPO (no training).

Does not modify Reward, Observation, SUMO, or MAPPO training code.
"""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from env.action import KEEP
from env.action_mask import legal_action_mask
from env.highway_accident_env import HighwayAccidentEnv, load_config
from models.mappo_model import MAPPOModel

Method = Literal["NoControl", "MAPPO", "RuleDivert"]

COMPARISON_FIELDS = [
    "seed",
    "method",
    "mean_speed",
    "mean_speed_near_accident",
    "throughput",
    "travel_time",
    "queue_length",
    "queue_length_max",
    "collision",
    "max_pos",
    "near_accident_samples",
    "entered_accident_zone",
]


@dataclass
class BaselineRow:
    seed: int
    method: str
    mean_speed: float
    mean_speed_near_accident: float
    throughput: float
    travel_time: float
    queue_length: float
    collision: int
    # Extra diagnostics (not required in comparison CSV)
    mean_individual_reward: float = float("nan")
    mean_group_reward: float = float("nan")
    mean_total_reward: float = float("nan")
    keep_ratio: float = float("nan")
    left_ratio: float = float("nan")
    right_ratio: float = float("nan")
    near_keep_ratio: float = float("nan")
    near_left_ratio: float = float("nan")
    near_right_ratio: float = float("nan")
    near_action_n: int = 0
    steps: int = 0
    sim_seconds: float = 0.0
    # Phase 5 coverage / queue
    queue_length_max: float = float("nan")
    min_pos: float = float("nan")
    max_pos: float = float("nan")
    mean_ego_position: float = float("nan")
    near_accident_samples: int = 0
    entered_accident_zone: int = 0
    keep_count: int = 0
    left_count: int = 0
    right_count: int = 0
    # Secondary proxy only (travel_time - distance/vmax); optional in reports
    delay_proxy: float = float("nan")
    deterministic: int = 0
    # Candidate A / reward-component diagnostics (optional)
    mean_s_cav: float = float("nan")
    mean_s_loc: float = float("nan")
    mean_p_block: float = float("nan")
    blocked_lane_near_frac: float = float("nan")
    reward_variant: str = ""

    def comparison_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in COMPARISON_FIELDS}


def set_eval_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _params(env: HighwayAccidentEnv) -> dict[str, float | str]:
    cfg = env.cfg
    accident = cfg.get("accident", {})
    obs = cfg.get("observation", {})
    test = cfg.get("test", {})
    return {
        "accident_pos": float(accident.get("position_m", 1200.0)),
        "near_m": float(obs.get("range_m", 150.0)),
        "upstream_m": float(test.get("congestion_upstream_m", 250.0)),
        "slow_mps": float(test.get("congestion_speed_mps", 15.0)),
        "step_length": float(cfg.get("sumo", {}).get("step_length", 0.1)),
        "vmax": float(
            obs.get(
                "max_speed_mps",
                cfg.get("highway", {}).get("speed_limit_mps", 30.0),
            )
        ),
        "ind_w": float(getattr(env, "reward_individual_weight", 0.7)),
        "grp_w": float(getattr(env, "reward_group_weight", 0.3)),
        "accident_prefix": str(accident.get("vehicle_id_prefix", "ACCIDENT")),
    }


def _queue_length(env: HighwayAccidentEnv, p: dict[str, float | str]) -> int:
    lo = float(p["accident_pos"]) - float(p["upstream_m"])
    accident_pos = float(p["accident_pos"])
    slow_mps = float(p["slow_mps"])
    n = 0
    for st in env.get_vehicle_states().values():
        if st.get("type") == "ACCIDENT":
            continue
        pos = float(st["position"])
        if lo <= pos <= accident_pos and float(st["speed"]) < slow_mps:
            n += 1
    return n


def _is_near_accident(pos: float, p: dict[str, float | str]) -> bool:
    return abs(float(pos) - float(p["accident_pos"])) <= float(p["near_m"])


def _actions_no_control(env: HighwayAccidentEnv) -> dict[str, int]:
    return {cid: KEEP for cid in env.cav_ids}


def _actions_rule_divert(env: HighwayAccidentEnv) -> dict[str, int]:
    """Near accident on a blocked lane → step toward the nearest open lane; else KEEP.

    Intended as a strong non-learning baseline once the scene is congested enough
    that diversion has traffic value.
    """
    from env.action import LEFT, RIGHT

    blocked = {int(x) for x in env.accident_lanes}
    open_lanes = [i for i in range(int(env.n_lanes)) if i not in blocked]
    near_m = float(env.cfg.get("observation", {}).get("range_m", 150.0))
    acc_pos = float(env.accident_position_m)
    left_up = bool(env.left_increases_index)
    out: dict[str, int] = {}
    states = env.get_cav_states()
    for cid in env.cav_ids:
        st = states.get(cid)
        if st is None:
            out[cid] = KEEP
            continue
        lane = int(st["lane"])
        pos = float(st["position"])
        if abs(pos - acc_pos) > near_m or lane not in blocked or not open_lanes:
            out[cid] = KEEP
            continue
        target = min(open_lanes, key=lambda L: abs(L - lane))
        if target > lane:
            out[cid] = LEFT if left_up else RIGHT
        elif target < lane:
            out[cid] = RIGHT if left_up else LEFT
        else:
            out[cid] = KEEP
    return out


@torch.no_grad()
def _actions_mappo(
    env: HighwayAccidentEnv,
    model: MAPPOModel,
    device: torch.device,
    *,
    deterministic: bool = False,
    use_action_mask: bool = False,
) -> dict[str, int]:
    model.eval()
    states = env.get_cav_states() if use_action_mask else {}
    out: dict[str, int] = {}
    for cid, obs in env.get_observations().items():
        t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        am = None
        if use_action_mask and cid in states:
            am = torch.as_tensor(
                legal_action_mask(
                    int(states[cid]["lane"]),
                    n_lanes=int(env.n_lanes),
                    left_increases_index=bool(env.left_increases_index),
                ),
                dtype=torch.float32,
                device=device,
            )
        action, _lp = model.act(t, deterministic=deterministic, action_mask=am)
        out[cid] = int(action.item())
    return out


def run_baseline_episode(
    *,
    method: Method,
    seed: int,
    model: MAPPOModel | None = None,
    device: torch.device | str = "cpu",
    gui: bool = False,
    max_steps: int | None = None,
    deterministic: bool = False,
    reward_config: dict[str, Any] | Path | str | None = None,
    use_action_mask: bool = False,
    env_config: dict[str, Any] | None = None,
) -> BaselineRow:
    """One evaluation episode. No training. Same scene/config for all methods."""
    import copy

    set_eval_seed(seed)
    device_t = torch.device(device)

    env_cfg = copy.deepcopy(env_config if env_config is not None else load_config())
    # Keep SUMO / episode horizon identical across methods.
    if max_steps is not None:
        env_cfg.setdefault("episode", {})["max_episode_steps"] = int(max_steps)

    env = HighwayAccidentEnv(
        config=env_cfg, gui=gui, reward_config=reward_config
    )
    p = _params(env)
    horizon = int(env.max_episode_steps)
    sim_seconds = horizon * float(p["step_length"])
    accident_pos = float(p["accident_pos"])
    vmax = float(p["vmax"])

    speeds: list[float] = []
    speeds_near: list[float] = []
    queues: list[float] = []
    cav_positions: list[float] = []
    ind_rs: list[float] = []
    grp_rs: list[float] = []
    tot_rs: list[float] = []
    s_cav_rs: list[float] = []
    s_loc_rs: list[float] = []
    p_block_rs: list[float] = []
    near_cav_n = 0
    near_cav_blocked_n = 0
    act_counts = np.zeros(3, dtype=np.int64)
    near_counts = np.zeros(3, dtype=np.int64)
    near_vehicle_sightings = 0
    entered_zone = 0
    collisions = 0
    arrived = 0
    first_seen: dict[str, float] = {}
    first_pos: dict[str, float] = {}
    travel_times: list[float] = []
    delay_proxies: list[float] = []
    prev_ids: set[str] = set()
    prefix = str(p["accident_prefix"])
    blocked_lanes = set(int(x) for x in env.accident_lanes)
    steps_run = 0

    try:
        env.reset()
        for step_i in range(horizon):
            traci = env._require_traci()
            sim_t = float(traci.simulation.getTime())
            cur_ids = set(traci.vehicle.getIDList())

            for vid in cur_ids - prev_ids:
                if not str(vid).startswith(prefix):
                    first_seen.setdefault(vid, sim_t)
                    try:
                        first_pos[vid] = float(traci.vehicle.getLanePosition(vid))
                    except Exception:  # noqa: BLE001
                        first_pos[vid] = 0.0
            for vid in prev_ids - cur_ids:
                if vid in first_seen:
                    tt = sim_t - first_seen.pop(vid)
                    travel_times.append(tt)
                    arrived += 1
                    start_p = float(first_pos.pop(vid, 0.0))
                    road_len = float(env.cfg.get("highway", {}).get("length_m", 1200.0))
                    trip_m = max(0.0, road_len - start_p)
                    if vmax > 1e-6:
                        delay_proxies.append(max(0.0, tt - trip_m / vmax))
            prev_ids = cur_ids

            collisions += int(traci.simulation.getCollidingVehiclesNumber())

            vehicles = env.get_vehicle_states()
            for vid, st in vehicles.items():
                if st.get("type") == "ACCIDENT":
                    continue
                spd = float(st["speed"])
                pos = float(st["position"])
                speeds.append(spd)
                if _is_near_accident(pos, p):
                    speeds_near.append(spd)
                    near_vehicle_sightings += 1
                if st.get("type") == "CAV":
                    cav_positions.append(pos)
                    if pos >= accident_pos:
                        entered_zone = 1
                    if _is_near_accident(pos, p):
                        entered_zone = 1
                        near_cav_n += 1
                        if int(st.get("lane", -1)) in blocked_lanes:
                            near_cav_blocked_n += 1

            for _cid, detail in env.get_rewards().items():
                ind_rs.append(float(detail["individual"]))
                grp_rs.append(float(detail["group"]))
                tot_rs.append(float(detail["total"]))
                if "s_cav" in detail:
                    s_cav_rs.append(float(detail["s_cav"]))
                if "s_loc" in detail:
                    s_loc_rs.append(float(detail["s_loc"]))
                if "p_block" in detail:
                    p_block_rs.append(float(detail["p_block"]))

            queues.append(float(_queue_length(env, p)))

            if method == "NoControl":
                actions = _actions_no_control(env)
            elif method == "RuleDivert":
                actions = _actions_rule_divert(env)
            else:
                if model is None:
                    raise ValueError("MAPPO evaluation requires a loaded model")
                actions = _actions_mappo(
                    env,
                    model,
                    device_t,
                    deterministic=bool(deterministic),
                    use_action_mask=bool(use_action_mask),
                )

            for cid, a in actions.items():
                a_i = int(a)
                act_counts[a_i] += 1
                st = vehicles.get(cid)
                if st is not None and _is_near_accident(float(st["position"]), p):
                    near_counts[a_i] += 1

            _obs, _rew, _term, truncated, _info = env.step(actions)
            del _obs, _rew, _term, _info
            steps_run = step_i + 1
            if truncated:
                break
    finally:
        env.close()

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    act_n = int(act_counts.sum())
    near_n = int(near_counts.sum())
    min_pos = float(min(cav_positions)) if cav_positions else float("nan")
    max_pos = float(max(cav_positions)) if cav_positions else float("nan")
    # If CAV reached near-band, count as entered
    if cav_positions and any(
        abs(pos - accident_pos) <= float(p["near_m"]) for pos in cav_positions
    ):
        entered_zone = 1

    return BaselineRow(
        seed=seed,
        method=method,
        mean_speed=_mean(speeds),
        mean_speed_near_accident=_mean(speeds_near),
        throughput=float(arrived),
        travel_time=_mean(travel_times),
        queue_length=_mean(queues),
        collision=int(collisions),
        mean_individual_reward=_mean(ind_rs),
        mean_group_reward=_mean(grp_rs),
        mean_total_reward=_mean(tot_rs),
        keep_ratio=float(act_counts[0] / act_n) if act_n else float("nan"),
        left_ratio=float(act_counts[1] / act_n) if act_n else float("nan"),
        right_ratio=float(act_counts[2] / act_n) if act_n else float("nan"),
        near_keep_ratio=float(near_counts[0] / near_n) if near_n else float("nan"),
        near_left_ratio=float(near_counts[1] / near_n) if near_n else float("nan"),
        near_right_ratio=float(near_counts[2] / near_n) if near_n else float("nan"),
        near_action_n=near_n,
        steps=steps_run,
        sim_seconds=float(steps_run) * float(p["step_length"]),
        queue_length_max=float(max(queues)) if queues else 0.0,
        min_pos=min_pos,
        max_pos=max_pos,
        mean_ego_position=_mean(cav_positions),
        near_accident_samples=int(near_vehicle_sightings),
        entered_accident_zone=int(entered_zone),
        keep_count=int(act_counts[0]),
        left_count=int(act_counts[1]),
        right_count=int(act_counts[2]),
        delay_proxy=_mean(delay_proxies),
        deterministic=int(bool(deterministic)) if method == "MAPPO" else 0,
        mean_s_cav=_mean(s_cav_rs),
        mean_s_loc=_mean(s_loc_rs),
        mean_p_block=_mean(p_block_rs),
        blocked_lane_near_frac=(
            float(near_cav_blocked_n) / float(near_cav_n)
            if near_cav_n
            else float("nan")
        ),
        reward_variant=str(getattr(env, "reward_variant", "current")),
    )


def write_csv(path: Path, rows: list[BaselineRow], *, comparison_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if comparison_only:
        fields = COMPARISON_FIELDS
        data = [r.comparison_dict() for r in rows]
    else:
        fields = list(asdict(rows[0]).keys())
        data = [asdict(r) for r in rows]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def load_mappo_checkpoint(
    path: Path | str,
    *,
    device: torch.device | str = "cpu",
) -> tuple[MAPPOModel, dict[str, Any]]:
    device_t = torch.device(device)
    ckpt = torch.load(Path(path), map_location=device_t, weights_only=False)
    meta = ckpt.get("model_meta", {})
    model = MAPPOModel(
        obs_dim=int(meta.get("obs_dim", 17)),
        action_dim=int(meta.get("action_dim", 3)),
        num_agents=int(meta.get("num_agents", 4)),
        hidden_dim=int(meta.get("hidden_dim", 128)),
    ).to(device_t)
    model.actor.load_state_dict(ckpt["actor_state_dict"])
    model.critic.load_state_dict(ckpt["critic_state_dict"])
    model.eval()
    return model, ckpt


def snapshot_model_params(model: MAPPOModel) -> list[torch.Tensor]:
    return [p.detach().cpu().clone() for p in model.parameters()]


def params_unchanged(before: list[torch.Tensor], after: list[torch.Tensor]) -> bool:
    if len(before) != len(after):
        return False
    return all(torch.equal(a, b) for a, b in zip(before, after))
