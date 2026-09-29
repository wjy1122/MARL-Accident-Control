"""Phase 4.5: MAPPO preflight smoke (3600 veh/h + accident, 2 updates).

Temporarily rewrites highway.rou.xml to demand=3600 + accident, then restores
env.yaml demand (1800). Writes outputs under results/preflight_3600_accident/
so Phase 4.4.x diagnostics and prior smoke checkpoints are not overwritten.

Example:
  .\\.venv\\Scripts\\python.exe scripts\\preflight_mappo_smoke.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from build_sumo_scenario import build_routes, load_config as load_build_config  # noqa: E402
from compare_accident_scenarios import write_routes_for_scenario  # noqa: E402

from algorithms.rollout import load_train_config  # noqa: E402
from scripts.train_mappo import train  # noqa: E402


def main() -> int:
    train_cfg = load_train_config()
    tblock = train_cfg.get("train", train_cfg)
    r_steps = int(tblock.get("rollout_steps", -1))

    out_root = PROJECT_ROOT / "results" / "preflight_3600_accident"
    ckpt_dir = out_root / "checkpoints"
    out_root.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("Phase 4.5 MAPPO Preflight Smoke")
    print("========================================")
    print("Scene: demand=3600, accident=ON (temporary rou.xml)")
    print(f"Config rollout_steps = {r_steps}")
    print("Updates = 2")
    print(f"Output dir = {out_root}")
    print()

    if r_steps != 512:
        print("FAIL: configs/train.yaml rollout_steps must be 512")
        return 1

    try:
        write_routes_for_scenario(demand=3600, with_accident=True)
        summary = train(
            num_episodes=2,
            rollout_steps=None,
            gui=False,
            checkpoint_dir=ckpt_dir,
            metrics_dir=out_root,
            run_title="Phase 4.5 Preflight: 3600 + accident + MAPPO (2 updates)",
        )
    finally:
        build_routes(load_build_config(PROJECT_ROOT / "configs" / "env.yaml"))
        print("Restored highway.rou.xml to env.yaml demand (1800 + accident).")

    rows = summary.get("log_rows") or []
    ok_chain = True
    for row in rows:
        cov = (
            int(row["rollout_steps"]) == 512
            and float(row["max_pos"]) > 1200.0
            and float(row["near_frac"]) > 0.0
            and int(row["num_near_accident_steps"]) > 0
        )
        print(
            f"update={row['episode']}: "
            f"actor={row['actor_loss']:.4f} critic={row['critic_loss']:.4f} "
            f"ent={row['entropy']:.4f} mean_R={row['mean_reward']:.4f} "
            f"min_pos={row['min_pos']:.1f} max_pos={row['max_pos']:.1f} "
            f"near_frac={row['near_frac']:.3f} "
            f"[{'PASS' if cov else 'FAIL'}]"
        )
        ok_chain = ok_chain and cov

    params_ok = bool(summary.get("actor_changed")) and bool(
        summary.get("critic_changed")
    )
    ok = ok_chain and params_ok and len(rows) == 2

    report = {
        "ok": ok,
        "demand": 3600,
        "accident": True,
        "rollout_steps": r_steps,
        "updates": 2,
        "device": summary.get("device"),
        "actor_changed": summary.get("actor_changed"),
        "critic_changed": summary.get("critic_changed"),
        "log_rows": rows,
        "checkpoints": summary.get("checkpoints"),
        "metrics_csv": summary.get("metrics_csv"),
    }
    report_path = out_root / "preflight_summary.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print()
    print("Phase 4.5 Smoke PASSED" if ok else "Phase 4.5 Smoke FAILED")
    print(f"Saved: {report_path}")
    if ok:
        print("可以进入正式 MAPPO 训练。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
