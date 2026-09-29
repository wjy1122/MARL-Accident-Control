"""Phase 2.1–2.2: TraCI wrapper + vehicle state reading for the highway accident scene."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml

from env.action import (
    ACTION_NAMES,
    KEEP,
    LEFT,
    RIGHT,
    target_lane_index,
    validate_action,
)
from env.action_mask import legal_action_mask
from env.observation import (
    OBS_DIM,
    build_observation,
    build_observation_dict,
    observations_for_cavs,
)
from env.reward import (
    compute_block_penalty,
    compute_candidate_a_group,
    compute_candidate_a_total,
    compute_group_reward,
    compute_individual_reward,
    compute_total_reward,
    find_cav_neighbors,
    find_local_vehicle_ids,
    mean_speed_ratio,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "env.yaml"


def _ensure_sumo_tools() -> None:
    sumo_home = os.environ.get("SUMO_HOME", "")
    if not sumo_home:
        raise RuntimeError(
            "SUMO_HOME is not set. Install SUMO and set SUMO_HOME to its install root."
        )
    tools = os.path.join(sumo_home, "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class HighwayAccidentEnv:
    """Minimal SUMO + TraCI environment (no RL / reward / observation design).

    Lifecycle:
      env = HighwayAccidentEnv()
      state = env.reset()
      state = env.step()
      env.close()
    """

    def __init__(
        self,
        config: dict[str, Any] | Path | str | None = None,
        *,
        gui: bool | None = None,
        delay_ms: float | None = None,
        reward_config: dict[str, Any] | Path | str | None = None,
    ) -> None:
        if isinstance(config, (str, Path)):
            self.cfg = load_config(config)
        elif isinstance(config, dict):
            self.cfg = config
        else:
            self.cfg = load_config()

        sumo_cfg = self.cfg.get("sumo", {})
        gui_block = self.cfg.get("gui", {}) or {}

        # GUI switch: constructor > cfg.gui.enabled > cfg.sumo.gui
        if gui is not None:
            self.gui = bool(gui)
        elif "enabled" in gui_block:
            self.gui = bool(gui_block["enabled"])
        else:
            self.gui = bool(sumo_cfg.get("gui", False))

        self.step_length = float(sumo_cfg.get("step_length", 0.1))

        # Delay only in GUI mode (omit --delay when headless).
        if delay_ms is not None:
            self.delay_ms = float(delay_ms) if self.gui else None
        elif self.gui:
            self.delay_ms = float(
                gui_block.get("delay_ms", sumo_cfg.get("gui_delay_ms", 50))
            )
        else:
            self.delay_ms = None

        self.sumocfg_path = (PROJECT_ROOT / sumo_cfg["config_file"]).resolve()
        if not self.sumocfg_path.is_file():
            raise FileNotFoundError(f"SUMO config not found: {self.sumocfg_path}")

        self.accident_prefix = str(
            self.cfg.get("accident", {}).get("vehicle_id_prefix", "ACCIDENT")
        )
        self.accident_position_m = float(
            self.cfg.get("accident", {}).get("position_m", 1200.0)
        )
        self.accident_lanes = [
            int(x) for x in self.cfg.get("accident", {}).get("blocked_lane_indices", [0])
        ]

        # Reward config (separate YAML; does not alter observation).
        # Default: configs/reward.yaml (current). Override via reward_config=
        # for Candidate A isolation (Phase 5.3) without touching the default file.
        if isinstance(reward_config, dict):
            self.reward_cfg = reward_config
            self.reward_config_path = None
        else:
            if reward_config is None:
                reward_path = PROJECT_ROOT / "configs" / "reward.yaml"
            else:
                reward_path = Path(reward_config)
                if not reward_path.is_absolute():
                    reward_path = PROJECT_ROOT / reward_path
            self.reward_config_path = reward_path
            if reward_path.is_file():
                with reward_path.open("r", encoding="utf-8") as f:
                    self.reward_cfg = yaml.safe_load(f) or {}
            else:
                self.reward_cfg = {}
        reward_block = self.reward_cfg.get("reward", self.reward_cfg)
        obs_range = float(self.cfg.get("observation", {}).get("range_m", 150.0))
        self.reward_variant = str(reward_block.get("variant", "current")).strip()
        self.reward_max_speed_mps = float(reward_block.get("max_speed_mps", 30.0))
        self.reward_neighbor_range_m = float(
            reward_block.get("neighbor_range_m", obs_range)
        )
        self.reward_zone_range_m = float(
            reward_block.get("zone_range_m", self.reward_neighbor_range_m)
        )
        self.reward_group_alpha = float(reward_block.get("group_alpha", 0.6))
        self.reward_individual_weight = float(
            reward_block.get("individual_weight", 0.7)
        )
        self.reward_group_weight = float(reward_block.get("group_weight", 0.3))
        self.reward_block_weight = float(reward_block.get("block_weight", 0.0))
        # Phase 6.5 Candidate A2 extras (0 = Candidate A unchanged).
        self.reward_lc_weight = float(reward_block.get("lc_weight", 0.0))
        self.reward_rev_weight = float(reward_block.get("rev_weight", 0.0))
        self.reward_progress_weight = float(
            reward_block.get("progress_weight", 0.0)
        )

        action_cfg = self.cfg.get("action", {})
        self.n_lanes = int(self.cfg.get("highway", {}).get("lanes", 3))
        self.lane_change_duration_s = float(
            action_cfg.get("lane_change_duration_s", 1.0)
        )
        # Suppress LEFT/RIGHT while a lane change is already in progress.
        self.suppress_mid_lane_change = bool(
            action_cfg.get("suppress_mid_lane_change", False)
        )
        self.mid_lane_change_lateral_m = float(
            action_cfg.get("mid_lane_change_lateral_m", 0.2)
        )
        # Learning-mask mid-LC (Phase 6.5); env-side suppress can still run alone.
        self.mask_mid_lane_change = bool(
            action_cfg.get("mask_mid_lane_change", False)
        )
        # After an effective lane-index change, forbid opposite LC for this long.
        self.lc_cooldown_s = float(action_cfg.get("lc_cooldown_s", 0.0))
        # Confirmed from highway.net.xml shapes (see apply_action docstring).
        self.left_increases_index = bool(
            action_cfg.get("left_increases_index", True)
        )

        episode_cfg = self.cfg.get("episode", {})
        self.max_episode_steps = int(episode_cfg.get("max_episode_steps", 1000))

        self.traci: Any | None = None
        self._connected = False
        self._step_count = 0
        # Oscillation / LC bookkeeping (cleared on reset).
        self._last_lc_dir: dict[str, int] = {}
        self._cooldown_left: dict[str, int] = {}
        self._pos_before: dict[str, float] = {}

    @property
    def is_connected(self) -> bool:
        return self._connected and self.traci is not None

    @property
    def step_count(self) -> int:
        return int(self._step_count)

    @property
    def cav_ids(self) -> list[str]:
        """CAV vehicle IDs currently on the road (excludes ACCIDENT/HDV)."""
        if not self.is_connected:
            return []
        return list(self.get_cav_states().keys())

    def _require_traci(self) -> Any:
        if not self.is_connected:
            raise RuntimeError("Environment is not connected. Call reset() first.")
        assert self.traci is not None
        return self.traci

    def _start_sumo(self) -> None:
        _ensure_sumo_tools()
        import traci

        binary = "sumo-gui" if self.gui else "sumo"
        cmd = [
            binary,
            "-c",
            str(self.sumocfg_path),
            "--step-length",
            str(self.step_length),
            "--start",
            "--quit-on-end",
        ]
        # GUI-only animation delay; headless never adds --delay.
        if self.gui and self.delay_ms is not None:
            cmd.extend(["--delay", str(self.delay_ms)])
        if self._connected:
            self.close()

        self._print_sumo_mode_banner(binary)
        traci.start(cmd)
        self.traci = traci
        self._connected = True
        self._step_count = 0

    def _print_sumo_mode_banner(self, binary: str) -> None:
        traffic = self.cfg.get("traffic", {})
        demand = traffic.get("vehs_per_hour", "?")
        cav_pen = float(traffic.get("cav_penetration", 0.0))
        lanes = ",".join(str(x) for x in getattr(self, "accident_lanes", [0]))
        print("========================================")
        print("SUMO GUI Mode")
        print("========================================")
        print(f"GUI: {'ON' if self.gui else 'OFF'}")
        print(f"SUMO executable: {binary}")
        if self.gui and self.delay_ms is not None:
            print(f"GUI delay: {self.delay_ms:g} ms")
        else:
            print("GUI delay: (none)")
        print(f"Demand: {demand} veh/h")
        print("Accident: ON" if self.accident_lanes else "Accident: OFF")
        print(f"Accident position: {self.accident_position_m:g} m")
        print(f"Accident lane: {lanes}")
        print(f"CAV penetration: {cav_pen:.2f}")
        print("========================================")

    def classify_vehicle(self, vehicle_id: str) -> str:
        """Return role label: CAV / HDV / ACCIDENT / other type id."""
        traci = self._require_traci()
        if vehicle_id.startswith(self.accident_prefix):
            return "ACCIDENT"
        vtype = traci.vehicle.getTypeID(vehicle_id)
        if vtype == "ACCIDENT":
            return "ACCIDENT"
        if vtype in ("CAV", "HDV"):
            return vtype
        return str(vtype)

    def get_vehicle_state(self, vehicle_id: str) -> dict[str, Any]:
        """Read basic TraCI state for one vehicle currently in the network."""
        traci = self._require_traci()
        role = self.classify_vehicle(vehicle_id)
        return {
            "id": vehicle_id,
            "type": role,
            "sumo_type": traci.vehicle.getTypeID(vehicle_id),
            "lane": int(traci.vehicle.getLaneIndex(vehicle_id)),
            "lane_id": str(traci.vehicle.getLaneID(vehicle_id)),
            "position": float(traci.vehicle.getLanePosition(vehicle_id)),
            "speed": float(traci.vehicle.getSpeed(vehicle_id)),
        }

    def get_vehicle_states(self) -> dict[str, dict[str, Any]]:
        """Read basic state for all vehicles currently in the network."""
        traci = self._require_traci()
        states: dict[str, dict[str, Any]] = {}
        for vid in traci.vehicle.getIDList():
            states[vid] = self.get_vehicle_state(vid)
        return states

    def get_cav_states(self) -> dict[str, dict[str, Any]]:
        """Subset of get_vehicle_states() with type == CAV (excludes ACCIDENT)."""
        return {
            vid: st
            for vid, st in self.get_vehicle_states().items()
            if st["type"] == "CAV"
        }

    def get_observation(self, cav_id: str) -> Any:
        """Normalized fixed-dim local observation for one CAV (numpy array)."""
        vehicles = self.get_vehicle_states()
        return build_observation(cav_id=cav_id, vehicles=vehicles, cfg=self.cfg)

    def get_observation_dict(self, cav_id: str) -> dict[str, float]:
        """Named normalized observation features for one CAV."""
        vehicles = self.get_vehicle_states()
        return build_observation_dict(cav_id=cav_id, vehicles=vehicles, cfg=self.cfg)

    def get_observations(self) -> dict[str, Any]:
        """Observations for all CAVs currently on the road."""
        vehicles = self.get_vehicle_states()
        cav_ids = [vid for vid, st in vehicles.items() if st["type"] == "CAV"]
        return observations_for_cavs(
            cav_ids=cav_ids, vehicles=vehicles, cfg=self.cfg
        )

    def get_reward(self, cav_id: str) -> dict[str, float]:
        """Individual / group / total reward for one CAV."""
        rewards = self.get_rewards([cav_id])
        if cav_id not in rewards:
            raise KeyError(f"CAV not on road or not a CAV: {cav_id}")
        return rewards[cav_id]

    def get_rewards(
        self, cav_ids: list[str] | None = None
    ) -> dict[str, dict[str, float]]:
        """Rewards for selected CAVs (default: all CAVs currently on road)."""
        if str(self.reward_variant).lower() in ("candidate_a", "candidate_A", "a"):
            return self._get_rewards_candidate_a(cav_ids)
        return self._get_rewards_current(cav_ids)

    def _get_rewards_current(
        self, cav_ids: list[str] | None = None
    ) -> dict[str, dict[str, float]]:
        """Phase 2.4 / current: 0.7 ind + 0.3 neighbor-CAV group."""
        cav_states = self.get_cav_states()
        if cav_ids is None:
            target_ids = list(cav_states.keys())
        else:
            target_ids = list(cav_ids)

        individuals = {
            vid: compute_individual_reward(
                float(st["speed"]), self.reward_max_speed_mps
            )
            for vid, st in cav_states.items()
        }
        positions = {vid: float(st["position"]) for vid, st in cav_states.items()}

        out: dict[str, dict[str, float]] = {}
        for ego_id in target_ids:
            if ego_id not in cav_states:
                continue
            neighbor_ids = find_cav_neighbors(
                ego_id=ego_id,
                cav_positions=positions,
                neighbor_range_m=self.reward_neighbor_range_m,
            )
            ind = float(individuals[ego_id])
            grp = compute_group_reward(individuals, neighbor_ids, ego_id)
            tot = compute_total_reward(
                ind,
                grp,
                individual_weight=self.reward_individual_weight,
                group_weight=self.reward_group_weight,
            )
            out[ego_id] = {
                "individual": ind,
                "group": grp,
                "total": tot,
                "s_cav": grp,
                "s_loc": ind,
                "p_block": 0.0,
            }
        return out

    def _get_rewards_candidate_a(
        self, cav_ids: list[str] | None = None
    ) -> dict[str, dict[str, float]]:
        """Candidate A: denser group + near-zone blocked-lane tax (no LC bonus)."""
        vehicles = self.get_vehicle_states()
        cav_states = {
            vid: st for vid, st in vehicles.items() if st.get("type") == "CAV"
        }
        if cav_ids is None:
            target_ids = list(cav_states.keys())
        else:
            target_ids = list(cav_ids)

        vmax = self.reward_max_speed_mps
        positions = {vid: float(st["position"]) for vid, st in vehicles.items()}
        types = {vid: str(st.get("type", "")) for vid, st in vehicles.items()}
        speeds = {vid: float(st["speed"]) for vid, st in vehicles.items()}
        cav_positions = {
            vid: float(st["position"]) for vid, st in cav_states.items()
        }

        out: dict[str, dict[str, float]] = {}
        for ego_id in target_ids:
            if ego_id not in cav_states:
                continue
            st = cav_states[ego_id]
            ind = compute_individual_reward(float(st["speed"]), vmax)

            cav_nids = find_cav_neighbors(
                ego_id=ego_id,
                cav_positions=cav_positions,
                neighbor_range_m=self.reward_neighbor_range_m,
            )
            s_cav = mean_speed_ratio(
                [speeds[j] for j in cav_nids if j in speeds],
                vmax,
                fallback=ind,
            )

            loc_ids = find_local_vehicle_ids(
                ego_id=ego_id,
                positions=positions,
                types=types,
                neighbor_range_m=self.reward_neighbor_range_m,
            )
            s_loc = mean_speed_ratio(
                [speeds[j] for j in loc_ids if j in speeds],
                vmax,
                fallback=ind,
            )

            grp = compute_candidate_a_group(
                s_cav=s_cav, s_loc=s_loc, alpha=self.reward_group_alpha
            )
            p_block = compute_block_penalty(
                ego_position=float(st["position"]),
                ego_lane=int(st["lane"]),
                accident_position_m=self.accident_position_m,
                blocked_lanes=self.accident_lanes,
                zone_range_m=self.reward_zone_range_m,
            )
            tot = compute_candidate_a_total(
                individual_reward=ind,
                group_reward=grp,
                p_block=p_block,
                individual_weight=self.reward_individual_weight,
                group_weight=self.reward_group_weight,
                block_weight=self.reward_block_weight,
            )
            out[ego_id] = {
                "individual": float(ind),
                "group": float(grp),
                "total": float(tot),
                "s_cav": float(s_cav),
                "s_loc": float(s_loc),
                "p_block": float(p_block),
                "lc_event": 0.0,
                "rev_event": 0.0,
                "progress": 0.0,
            }
        return out

    def _is_mid_lane_change(self, cav_id: str) -> bool:
        traci = self._require_traci()
        try:
            lat = abs(float(traci.vehicle.getLateralLanePosition(cav_id)))
            return lat > float(self.mid_lane_change_lateral_m)
        except Exception:  # noqa: BLE001
            return False

    def get_action_mask(self, cav_id: str) -> Any:
        """Boundary + optional mid-LC + opposite-direction cooldown mask."""
        traci = self._require_traci()
        if cav_id not in traci.vehicle.getIDList():
            return legal_action_mask(
                1,
                n_lanes=self.n_lanes,
                left_increases_index=self.left_increases_index,
            )
        lane = int(traci.vehicle.getLaneIndex(cav_id))
        mid = bool(self.mask_mid_lane_change and self._is_mid_lane_change(cav_id))
        cd = int(self._cooldown_left.get(cav_id, 0))
        last = self._last_lc_dir.get(cav_id)
        forbid_left = False
        forbid_right = False
        if cd > 0 and last is not None:
            # Forbid the opposite of the last completed LC direction.
            if int(last) == LEFT:
                forbid_right = True
            elif int(last) == RIGHT:
                forbid_left = True
        return legal_action_mask(
            lane,
            n_lanes=self.n_lanes,
            left_increases_index=self.left_increases_index,
            mid_lane_change=mid,
            forbid_left=forbid_left,
            forbid_right=forbid_right,
        )

    def apply_action(self, cav_id: str, action: Any) -> dict[str, Any]:
        """Apply discrete lane-change action to one CAV via TraCI.

        Lane orientation (from ``sumo/highway.net.xml``):
          edge e0 travels +x; lane shapes y = -8.0, -4.8, -1.6 for indices 0,1,2.
          With right-hand traffic, lane 0 = rightmost, lane 2 = leftmost.
          Therefore LEFT → lane_index + 1, RIGHT → lane_index - 1.

        Boundary: LEFT on leftmost / RIGHT on rightmost → no-op (no error).
        Illegal action values → ValueError.
        """
        traci = self._require_traci()
        action_i = validate_action(action)

        if cav_id not in traci.vehicle.getIDList():
            raise KeyError(f"Vehicle not in simulation: {cav_id}")
        if self.classify_vehicle(cav_id) != "CAV":
            raise ValueError(f"apply_action only supports CAV, got {cav_id}")

        lane_before = int(traci.vehicle.getLaneIndex(cav_id))
        target = target_lane_index(
            current_lane=lane_before,
            action=action_i,
            n_lanes=self.n_lanes,
            left_increases_index=self.left_increases_index,
        )

        executed = False
        suppressed_mid = False
        if action_i == KEEP:
            executed = False
        elif target is None:
            executed = False  # boundary: illegal LC suppressed
        else:
            mid = False
            if self.suppress_mid_lane_change:
                try:
                    lat = abs(float(traci.vehicle.getLateralLanePosition(cav_id)))
                    mid = lat > float(self.mid_lane_change_lateral_m)
                except Exception:  # noqa: BLE001
                    mid = False
            if mid:
                executed = False
                suppressed_mid = True
            else:
                # Duration from config (seconds); TraCI changeLane API.
                traci.vehicle.changeLane(
                    cav_id, int(target), float(self.lane_change_duration_s)
                )
                executed = True

        return {
            "cav_id": cav_id,
            "action": action_i,
            "action_name": ACTION_NAMES[action_i],
            "lane_before": lane_before,
            "target_lane": target if target is not None else lane_before,
            "executed": executed,
            "suppressed_mid_lane_change": suppressed_mid,
        }

    def _snapshot(self) -> dict[str, Any]:
        """Sim snapshot including vehicle states (observations/rewards on demand)."""
        traci = self._require_traci()
        vehicles = self.get_vehicle_states()
        cav_ids: list[str] = []
        hdv_ids: list[str] = []
        accident_ids: list[str] = []

        for vid, st in vehicles.items():
            if st["type"] == "CAV":
                cav_ids.append(vid)
            elif st["type"] == "HDV":
                hdv_ids.append(vid)
            elif st["type"] == "ACCIDENT":
                accident_ids.append(vid)

        return {
            "time": float(traci.simulation.getTime()),
            "step_count": int(self._step_count),
            "vehicle_ids": list(vehicles.keys()),
            "cav_ids": cav_ids,
            "hdv_ids": hdv_ids,
            "accident_ids": accident_ids,
            "n_vehicles": len(vehicles),
            "n_cav": len(cav_ids),
            "n_hdv": len(hdv_ids),
            "vehicles": vehicles,
            "obs_dim": OBS_DIM,
        }

    def reset(self) -> dict[str, Any]:
        """Start (or restart) SUMO via TraCI and return the initial snapshot.

        ``self._step_count`` is reset to 0. Snapshot still includes ``cav_ids`` /
        ``vehicles`` for Phase 2.x tests; call ``get_observations()`` for RL obs.
        """
        self._start_sumo()
        assert self._step_count == 0
        self._last_lc_dir.clear()
        self._cooldown_left.clear()
        self._pos_before.clear()
        return self._snapshot()

    def step(
        self, actions: dict[str, Any] | None = None
    ) -> dict[str, Any] | tuple[dict[str, Any], dict[str, float], bool, bool, dict[str, Any]]:
        """Advance the environment.

        Compatibility:
          * ``step()`` / ``step(None)`` → legacy snapshot dict (Phase 2.x tests).
          * ``step({cav_id: action, ...})`` → Phase 3.2 closed loop:
                observations, rewards, terminated, truncated, info
        """
        self._require_traci()
        assert self.traci is not None

        # ---- Legacy path: no joint actions (tests / probing) ----
        if actions is None:
            self.traci.simulationStep()
            self._step_count += 1
            return self._snapshot()

        # ---- Phase 3.2 closed loop ----
        if not isinstance(actions, dict):
            raise TypeError(
                f"actions must be a dict[cav_id -> action], got {type(actions)}"
            )

        # Validate all provided action values first.
        normalized: dict[str, int] = {}
        for cav_id, action in actions.items():
            if not isinstance(cav_id, str):
                raise TypeError(f"action key must be cav_id str, got {cav_id!r}")
            normalized[cav_id] = validate_action(action)

        present = set(self.cav_ids)
        # Snapshot positions before LC + sim for progress term.
        self._pos_before = {}
        for cav_id in present:
            try:
                self._pos_before[cav_id] = float(
                    self.traci.vehicle.getLanePosition(cav_id)
                )
            except Exception:  # noqa: BLE001
                pass

        # Apply all actions for current CAVs in the SAME decision step, then
        # one shared simulationStep (not one step per CAV).
        action_details: dict[str, Any] = {}
        for cav_id in present:
            action_i = normalized.get(cav_id, KEEP)
            if cav_id in self.traci.vehicle.getIDList():
                action_details[cav_id] = self.apply_action(cav_id, action_i)

        self.traci.simulationStep()
        self._step_count += 1

        # Update cooldown / last LC from effective lane-index changes.
        cooldown_steps = 0
        if self.lc_cooldown_s > 0.0 and self.step_length > 0.0:
            cooldown_steps = max(
                1, int(round(float(self.lc_cooldown_s) / float(self.step_length)))
            )
        alive = set(self.cav_ids)
        # Drop departed agents from bookkeeping.
        for stale in list(self._cooldown_left.keys()):
            if stale not in alive:
                self._cooldown_left.pop(stale, None)
                self._last_lc_dir.pop(stale, None)
        for cav_id in list(self._cooldown_left.keys()):
            if self._cooldown_left[cav_id] > 0:
                self._cooldown_left[cav_id] -= 1

        lc_events: dict[str, float] = {}
        rev_events: dict[str, float] = {}
        progress_vals: dict[str, float] = {}
        vmax_dt = float(self.reward_max_speed_mps) * float(self.step_length)
        for cav_id, det in action_details.items():
            if cav_id not in alive:
                continue
            try:
                lane_after = int(self.traci.vehicle.getLaneIndex(cav_id))
            except Exception:  # noqa: BLE001
                continue
            lane_before = int(det.get("lane_before", lane_after))
            changed = lane_after != lane_before
            lc_i = 1.0 if changed else 0.0
            rev_i = 0.0
            if changed:
                direction = LEFT if lane_after > lane_before else RIGHT
                prev_dir = self._last_lc_dir.get(cav_id)
                if prev_dir is not None and int(prev_dir) != int(direction):
                    rev_i = 1.0
                self._last_lc_dir[cav_id] = int(direction)
                if cooldown_steps > 0:
                    self._cooldown_left[cav_id] = int(cooldown_steps)
            lc_events[cav_id] = lc_i
            rev_events[cav_id] = rev_i

            pos0 = self._pos_before.get(cav_id)
            prog = 0.0
            if pos0 is not None and vmax_dt > 0.0:
                try:
                    pos1 = float(self.traci.vehicle.getLanePosition(cav_id))
                    prog = max(0.0, min(1.0, (pos1 - float(pos0)) / vmax_dt))
                except Exception:  # noqa: BLE001
                    prog = 0.0
            progress_vals[cav_id] = prog

        observations = self.get_observations()
        reward_details = self.get_rewards()
        # Fold Phase 6.5 extras into total (Candidate A weights stay 0 → no-op).
        need_extras = (
            self.reward_lc_weight > 0.0
            or self.reward_rev_weight > 0.0
            or self.reward_progress_weight > 0.0
        )
        if need_extras:
            for cav_id, detail in reward_details.items():
                lc_i = float(lc_events.get(cav_id, 0.0))
                rev_i = float(rev_events.get(cav_id, 0.0))
                prog = float(progress_vals.get(cav_id, 0.0))
                detail["lc_event"] = lc_i
                detail["rev_event"] = rev_i
                detail["progress"] = prog
                detail["total"] = compute_candidate_a_total(
                    individual_reward=float(detail["individual"]),
                    group_reward=float(detail["group"]),
                    p_block=float(detail.get("p_block", 0.0)),
                    individual_weight=self.reward_individual_weight,
                    group_weight=self.reward_group_weight,
                    block_weight=self.reward_block_weight,
                    lc_event=lc_i,
                    rev_event=rev_i,
                    progress=prog,
                    lc_weight=self.reward_lc_weight,
                    rev_weight=self.reward_rev_weight,
                    progress_weight=self.reward_progress_weight,
                )

        rewards = {
            cav_id: float(detail["total"])
            for cav_id, detail in reward_details.items()
        }

        truncated = self._step_count >= self.max_episode_steps
        # No natural task termination defined yet in Phase 3.2.
        terminated = False

        info: dict[str, Any] = {
            "step": self._step_count,
            "reward_details": reward_details,
            "action_details": action_details,
            "n_cav": len(observations),
            "lc_events": lc_events,
            "rev_events": rev_events,
        }
        return observations, rewards, terminated, truncated, info

    def close(self) -> None:
        """Close TraCI and quit the SUMO process if still connected."""
        if self.traci is not None and self._connected:
            try:
                self.traci.close()
            except Exception:  # noqa: BLE001
                pass
        self.traci = None
        self._connected = False

    def __enter__(self) -> HighwayAccidentEnv:
        self.reset()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()
