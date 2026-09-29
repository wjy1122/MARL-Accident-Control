"""MAPPO GUI short observation (5 updates) on Phase 4.4.9 frozen scene.

Uses existing scripts/train_mappo.py train() + --gui path.
Does NOT modify MAPPO / Reward / Observation / PPO.

Example:
  .\\.venv\\Scripts\\python.exe scripts\\run_mappo_gui_observation.py
"""

from __future__ import annotations

import json
import shutil
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_mappo import train  # noqa: E402

OUT_DIR = PROJECT_ROOT / "results" / "experiments" / "mappo_gui_observation_1200m_700m"
DIAG_DIR = PROJECT_ROOT / "results" / "diagnostics"


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def _check_consistency() -> dict[str, Any]:
    env = yaml.safe_load((PROJECT_ROOT / "configs" / "env.yaml").read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "train.yaml").read_text(encoding="utf-8")
    )
    reward = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "reward.yaml").read_text(encoding="utf-8")
    )
    tblock = train_cfg.get("train", train_cfg)
    rb = reward.get("reward", reward)

    nod = ET.parse(PROJECT_ROOT / "sumo" / "highway.nod.xml").getroot()
    n1 = nod.find("node[@id='n1']")
    nod_x = float(n1.get("x", "nan")) if n1 is not None else float("nan")

    edg = ET.parse(PROJECT_ROOT / "sumo" / "highway.edg.xml").getroot()
    e0 = edg.find("edge[@id='e0']")
    edg_len = float(e0.get("length", "nan")) if e0 is not None else float("nan")
    edg_lanes = int(e0.get("numLanes", "-1")) if e0 is not None else -1
    edg_spd = float(e0.get("speed", "nan")) if e0 is not None else float("nan")

    net = ET.parse(PROJECT_ROOT / "sumo" / "highway.net.xml").getroot()
    edge = None
    for e in net.iter("edge"):
        if e.get("id") == "e0" and e.get("function") != "internal":
            edge = e
            break
    lanes = list(edge.findall("lane")) if edge is not None else []
    net_lens = [float(ln.get("length", "nan")) for ln in lanes]
    net_spds = [float(ln.get("speed", "nan")) for ln in lanes]

    rou = ET.parse(PROJECT_ROOT / "sumo" / "highway.rou.xml").getroot()
    flows = {
        f.get("id"): int(float(f.get("vehsPerHour", "0"))) for f in rou.findall("flow")
    }
    demand_total = int(sum(flows.values()))
    acc = rou.find("vehicle[@id='ACCIDENT_0']")
    acc_pos = float(acc.get("departPos", "nan")) if acc is not None else float("nan")
    acc_lane = int(acc.get("departLane", "-1")) if acc is not None else -1

    checks = {
        "road_length_env": float(env["highway"]["length_m"]),
        "road_length_nod": nod_x,
        "road_length_edg": edg_len,
        "road_length_net_mean": float(sum(net_lens) / len(net_lens)) if net_lens else float("nan"),
        "accident_pos_env": float(env["accident"]["position_m"]),
        "accident_pos_rou": acc_pos,
        "accident_lane_rou": acc_lane,
        "lanes_env": int(env["highway"]["lanes"]),
        "lanes_edg": edg_lanes,
        "lanes_net": len(lanes),
        "lane_speed_edg": edg_spd,
        "lane_speeds_net": net_spds,
        "demand_env": int(env["traffic"]["vehs_per_hour"]),
        "demand_rou_total": demand_total,
        "demand_rou_flows": flows,
        "cav_pen": float(env["traffic"]["cav_penetration"]),
        "seed_env": int(env.get("seed", -1)),
        "seed_train": int(train_cfg.get("seed", -1)),
        "rollout_steps": int(tblock.get("rollout_steps", -1)),
        "reward_ind": float(rb.get("individual_weight", -1)),
        "reward_grp": float(rb.get("group_weight", -1)),
        "gui_delay_ms": float(env.get("sumo", {}).get("gui_delay_ms", 50)),
        "sumo_files_exist": all(
            (PROJECT_ROOT / "sumo" / name).is_file()
            for name in (
                "highway.nod.xml",
                "highway.edg.xml",
                "highway.net.xml",
                "highway.rou.xml",
                "highway.add.xml",
                "highway.sumocfg",
            )
        ),
    }
    ok = (
        abs(checks["road_length_env"] - 1200.0) < 1e-6
        and abs(checks["road_length_nod"] - 1200.0) < 1e-3
        and abs(checks["road_length_net_mean"] - 1200.0) < 1e-3
        and abs(checks["accident_pos_env"] - 700.0) < 1e-6
        and abs(checks["accident_pos_rou"] - 700.0) < 1e-3
        and checks["accident_lane_rou"] == 0
        and checks["lanes_env"] == 3
        and checks["lanes_net"] == 3
        and checks["demand_env"] == 3600
        and checks["demand_rou_total"] == 3600
        and abs(checks["cav_pen"] - 0.2) < 1e-9
        and checks["seed_env"] == 42
        and checks["seed_train"] == 42
        and checks["rollout_steps"] == 512
        and abs(checks["reward_ind"] - 0.7) < 1e-9
        and abs(checks["reward_grp"] - 0.3) < 1e-9
        and checks["sumo_files_exist"]
    )
    checks["ok"] = bool(ok)
    return checks


def _assert_diagnostics_untouched(before: dict[str, tuple[int, int]]) -> None:
    after: dict[str, tuple[int, int]] = {}
    for p in DIAG_DIR.glob("phase_44*"):
        if p.is_file():
            st = p.stat()
            after[p.name] = (st.st_size, st.st_mtime_ns)
    for name, fp in before.items():
        if name in after and after[name] != fp:
            raise RuntimeError(f"Diagnostics history modified: {name}")


def _snapshot_configs(dst: Path) -> None:
    snap = dst / "config_snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    for rel in (
        "configs/env.yaml",
        "configs/train.yaml",
        "configs/reward.yaml",
        "configs/model.yaml",
        "sumo/highway.rou.xml",
        "sumo/highway.sumocfg",
        "sumo/highway.nod.xml",
        "sumo/highway.edg.xml",
    ):
        src = PROJECT_ROOT / rel
        if src.is_file():
            shutil.copy2(src, snap / src.name)


def _assess(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    if not rows:
        return "FAIL"
    last = rows[-1]
    entered = any(int(r.get("reached_accident_zone", 0)) == 1 for r in rows)
    max_pos = max(float(r.get("max_pos", 0.0)) for r in rows)
    near_frac = float(last.get("near_frac", 0.0))
    actor_ok = bool(summary.get("actor_changed"))
    critic_ok = bool(summary.get("critic_changed"))
    keep_pct = float(last.get("KEEP_pct", 100.0))
    lc_pct = float(last.get("LEFT_pct", 0.0)) + float(last.get("RIGHT_pct", 0.0))

    if not (actor_ok and critic_ok and entered and max_pos >= 700.0):
        return "FAIL"
    if near_frac <= 0.0:
        return "FAIL"
    # Observation-only: extreme KEEP is noted but not auto-fail if zone reached + update OK
    if keep_pct >= 95.0 and lc_pct < 5.0:
        return "BORDERLINE"
    if near_frac < 0.05:
        return "BORDERLINE"
    return "PASS"


def _write_report(
    *,
    out: Path,
    checks: dict[str, Any],
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    result: str,
    device: str,
    gpu: str | None,
) -> None:
    last = rows[-1] if rows else {}
    mean_keep = float(sum(float(r.get("KEEP_pct", 0)) for r in rows) / max(len(rows), 1))
    mean_left = float(sum(float(r.get("LEFT_pct", 0)) for r in rows) / max(len(rows), 1))
    mean_right = float(sum(float(r.get("RIGHT_pct", 0)) for r in rows) / max(len(rows), 1))
    mean_rew = float(sum(float(r.get("mean_reward", 0)) for r in rows) / max(len(rows), 1))
    mean_actor = float(sum(float(r.get("actor_loss", 0)) for r in rows) / max(len(rows), 1))
    mean_critic = float(
        sum(float(r.get("critic_loss", 0)) for r in rows) / max(len(rows), 1)
    )
    mean_ent = float(sum(float(r.get("entropy", 0)) for r in rows) / max(len(rows), 1))
    mean_spd = float(
        sum(float(r.get("mean_speed_mps", 0)) for r in rows) / max(len(rows), 1)
    )
    mean_zone = float(
        sum(float(r.get("mean_speed_near_accident", 0)) for r in rows) / max(len(rows), 1)
    )
    mean_up = float(
        sum(float(r.get("speed_before_mps", 0)) for r in rows) / max(len(rows), 1)
    )
    max_pos = max((float(r.get("max_pos", 0)) for r in rows), default=0.0)
    entered = any(int(r.get("reached_accident_zone", 0)) == 1 for r in rows)

    md = f"""# MAPPO GUI Observation Report

## 1. Configuration

- Road length: {checks['road_length_env']} m (net≈{checks['road_length_net_mean']} m)
- Accident: {checks['accident_pos_env']} m / lane {checks['accident_lane_rou']}
- Lanes: {checks['lanes_net']}
- Demand: {checks['demand_rou_total']} veh/h (HDV={checks['demand_rou_flows'].get('flow_HDV')}, CAV={checks['demand_rou_flows'].get('flow_CAV')})
- CAV penetration: {checks['cav_pen']}
- Seed: {checks['seed_train']}
- Reward: {checks['reward_ind']} * R_ind + {checks['reward_grp']} * R_group
- Observation: 17-D (unchanged)
- GUI: ON (sumo-gui), delay={checks['gui_delay_ms']} ms
- Consistency check: {'PASS' if checks['ok'] else 'FAIL'}
- Entry: `scripts/train_mappo.py` / `train(gui=True)` (no second GUI path)
- Output: `{out.as_posix()}`

## 2. Training

- updates = 5
- rollout_steps = 512
- total_steps = 2560
- seed = {checks['seed_train']}
- device = {device}
- GPU = {gpu or 'n/a'}
- Actor params updated: {summary.get('actor_changed')}
- Critic params updated: {summary.get('critic_changed')}

## 3. Traffic

(Logging metrics from existing train diagnostics; throughput not instrumented in train loop.)

- mean speed (obs ego, mean over updates): {mean_spd:.3f} m/s
- upstream / before-accident speed proxy (`speed_before_mps`): {mean_up:.3f} m/s
- accident-zone speed (`mean_speed_near_accident`): {mean_zone:.3f} m/s
- after-accident speed proxy (`speed_after_mps`, last): {float(last.get('speed_after_mps', float('nan'))):.3f} m/s
- queue proxy mean (last): {float(last.get('mean_queue_length', float('nan'))):.2f}
- max_pos across updates: {max_pos:.1f} m
- entered accident zone: {'YES' if entered else 'NO'}
- throughput: N/A (not in train_mappo metrics)

## 4. MAPPO

- mean actor loss: {mean_actor:.6f}
- mean critic loss: {mean_critic:.6f}
- mean entropy: {mean_ent:.6f}
- mean reward: {mean_rew:.6f}
- last update: actor={float(last.get('actor_loss', float('nan'))):.6f}, critic={float(last.get('critic_loss', float('nan'))):.6f}, ent={float(last.get('entropy', float('nan'))):.6f}, reward={float(last.get('mean_reward', float('nan'))):.6f}

## 5. Actions

Mean over 5 updates:

- KEEP: {mean_keep:.1f}%
- LEFT: {mean_left:.1f}%
- RIGHT: {mean_right:.1f}%

Last update:

- KEEP: {float(last.get('KEEP_pct', float('nan'))):.1f}% (n={last.get('KEEP_count')})
- LEFT: {float(last.get('LEFT_pct', float('nan'))):.1f}% (n={last.get('LEFT_count')})
- RIGHT: {float(last.get('RIGHT_pct', float('nan'))):.1f}% (n={last.get('RIGHT_count')})
- Near-zone KEEP/LEFT/RIGHT: {float(last.get('near_KEEP_pct', float('nan'))):.1f}% / {float(last.get('near_LEFT_pct', float('nan'))):.1f}% / {float(last.get('near_RIGHT_pct', float('nan'))):.1f}%

Lane-change execution counters (raw / boundary blocked / actual):

- Not logged inside current `train_mappo` loop (available in `zone_behavior_diagnostics` / Phase 4.4.4 only). Recorded as N/A for this observation run — no algorithm change.

## 6. GUI Observation

- Accident vehicle present at ~700 m / lane 0: expected YES (config+rou verified; GUI banner Accident position 700 m)
- Entered accident zone during rollouts: {'YES' if entered else 'NO'} (max_pos={max_pos:.1f} m)
- Lane-change actions issued: {'YES' if (mean_left + mean_right) > 5 else 'LIMITED / mostly KEEP'} (LEFT+RIGHT≈{mean_left + mean_right:.1f}%)
- Obvious congestion: queue proxy last={float(last.get('mean_queue_length', 0)):.2f}; zone speed {mean_zone:.2f} vs free-flow 30 m/s
- Abnormal / freeze: none reported by training process (completed + SUMO closed)

## 7. Result

**{result}**

Notes (observation only, no tuning):

- Reward remaining near 1.0 is a known Phase 4.4 finding; not modified here.
- Formal long MAPPO training was NOT started.
"""
    (out / "report.md").write_text(md, encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hist_before = {
        p.name: (p.stat().st_size, p.stat().st_mtime_ns)
        for p in DIAG_DIR.glob("phase_44*")
        if p.is_file()
    }

    log_path = OUT_DIR / "stdout.log"
    log_f = log_path.open("w", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = Tee(old_out, log_f)
    sys.stderr = Tee(old_err, log_f)

    try:
        print("========================================")
        print("MAPPO GUI Observation (Phase 4.4.9 scene)")
        print("========================================")
        print(f"Output: {OUT_DIR}")
        print(f"Started: {datetime.now(timezone.utc).isoformat()}")
        print()

        print("--- Consistency check ---")
        checks = _check_consistency()
        for k in (
            "road_length_env",
            "road_length_net_mean",
            "accident_pos_env",
            "accident_pos_rou",
            "demand_rou_total",
            "cav_pen",
            "rollout_steps",
            "seed_train",
        ):
            print(f"  {k}: {checks[k]}")
        if not checks["ok"]:
            print("CONSISTENCY FAIL — abort (no algorithm changes).")
            (OUT_DIR / "consistency_check.json").write_text(
                json.dumps(checks, indent=2), encoding="utf-8"
            )
            return 1
        print("Consistency: PASS")
        (OUT_DIR / "consistency_check.json").write_text(
            json.dumps(checks, indent=2), encoding="utf-8"
        )

        _snapshot_configs(OUT_DIR)
        print("Config snapshot saved.")

        print()
        print("--- Training: 5 updates × 512 steps, GUI ON ---")
        summary = train(
            num_episodes=5,
            rollout_steps=512,
            gui=True,
            checkpoint_dir=OUT_DIR / "checkpoints",
            metrics_dir=OUT_DIR,
            run_title="MAPPO GUI Observation (1200m / accident@700m, 5 updates)",
        )
        rows = list(summary.get("log_rows") or [])
        result = _assess(rows, summary)

        device = str(summary.get("device", "unknown"))
        gpu = None
        try:
            import torch

            if device.startswith("cuda") and torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            gpu = None

        meta = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "updates": 5,
            "rollout_steps": 512,
            "total_steps": 2560,
            "result": result,
            "summary": {
                k: summary[k]
                for k in (
                    "device",
                    "actor_changed",
                    "critic_changed",
                    "coverage_ok",
                    "metrics_csv",
                    "checkpoints",
                )
                if k in summary
            },
            "consistency": checks,
            "log_rows": rows,
        }
        (OUT_DIR / "run_meta.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )
        _write_report(
            out=OUT_DIR,
            checks=checks,
            summary=summary,
            rows=rows,
            result=result,
            device=device,
            gpu=gpu,
        )
        _assert_diagnostics_untouched(hist_before)

        print()
        print("========================================")
        print(f"Observation Result: {result}")
        print(f"Report: {OUT_DIR / 'report.md'}")
        print("MAPPO formal long training: NOT STARTED")
        print("========================================")
        return 0 if result != "FAIL" else 1
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        log_f.close()


if __name__ == "__main__":
    raise SystemExit(main())
