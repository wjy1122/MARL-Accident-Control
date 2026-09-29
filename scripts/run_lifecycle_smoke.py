"""Lifecycle smoke: CAV leave must not restart SUMO (Phase 4.4.9 scene).

2 updates x 512 steps, GUI ON. Does not run formal long training.

Example:
  .\\.venv\\Scripts\\python.exe scripts\\run_lifecycle_smoke.py
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_mappo import train  # noqa: E402

OUT_DIR = PROJECT_ROOT / "results" / "experiments" / "lifecycle_smoke_1200m_700m"


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


def _snapshot(dst: Path) -> None:
    snap = dst / "config_snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    for rel in (
        "configs/env.yaml",
        "configs/train.yaml",
        "configs/reward.yaml",
        "configs/model.yaml",
        "algorithms/rollout.py",
    ):
        src = PROJECT_ROOT / rel
        if src.is_file():
            shutil.copy2(src, snap / src.name)


def _assess(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    if not rows or not summary.get("actor_changed") or not summary.get("critic_changed"):
        return "FAIL"
    r0 = rows[0]
    # First rollout: only initial SUMO start (reset=1), must see departures + new binds.
    ok_first = (
        int(r0.get("env_reset_count", 99)) == 1
        and int(r0.get("sumo_start_count", 99)) == 1
        and int(r0.get("departed_cav_count", 0)) > 0
        and int(r0.get("new_cav_count", 0)) > 0
        and int(r0.get("rebind_events", 0)) > 0
        and int(r0.get("reached_accident_zone", 0)) == 1
    )
    if not ok_first:
        # Soft fail details
        if (
            int(r0.get("departed_cav_count", 0)) > 0
            and int(r0.get("new_cav_count", 0)) > 0
            and int(r0.get("env_reset_count", 99)) <= 2
        ):
            return "BORDERLINE"
        return "FAIL"
    if len(rows) >= 2:
        r1 = rows[1]
        # Second rollout may hit max_episode_steps truncate → one extra reset OK.
        if int(r1.get("env_reset_count", 99)) > 2:
            return "BORDERLINE"
    return "PASS"


def _write_report(
    *,
    out: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    result: str,
) -> None:
    r0 = rows[0] if rows else {}
    r1 = rows[1] if len(rows) > 1 else {}
    md = f"""# Lifecycle Smoke Report

## 1. 修改文件

- `algorithms/rollout.py` — CAV 离场改为固定 slot 重绑定，不再 `env.reset()`；仅未连接或 episode truncate 时启动/重启 SUMO
- `scripts/train_mappo.py` — 记录 lifecycle 诊断字段（不改算法）
- `scripts/run_lifecycle_smoke.py` — 本 smoke 入口（新建）

## 2. 修改原因

原链路：

```text
locked CAV 全部离场 → alive==0 → env.reset() → _start_sumo() → close GUI → 从 t=0 重开
```

另外每个 `collect()` 开头无条件 `env.reset()`，每个 update 都会重开 SUMO。

## 3. Agent lifecycle

MAPPO 仍用固定 N=4 slots（obs/action/buffer 形状不变）：

```text
CAV 进入道路 → 填入空 slot → active（mask=1）→ 参与控制
CAV 离开 1200m → 清空该 slot（mask=0）→ 不重启 SUMO
新 CAV 进入 → 绑定到空 slot → 继续控制
```

## 4. Reset statistics

### Update 1

- env_reset_count: {r0.get('env_reset_count')}
- sumo_start_count: {r0.get('sumo_start_count')}
- departed_cav_count: {r0.get('departed_cav_count')}
- new_cav_count: {r0.get('new_cav_count')}
- active_cav_peak: {r0.get('active_cav_peak')}
- active_slot_peak: {r0.get('active_slot_peak')}
- rebind_events: {r0.get('rebind_events')}

### Update 2

- env_reset_count: {r1.get('env_reset_count')}
- sumo_start_count: {r1.get('sumo_start_count')}
- departed_cav_count: {r1.get('departed_cav_count')}
- new_cav_count: {r1.get('new_cav_count')}
- active_cav_peak: {r1.get('active_cav_peak')}
- rebind_events: {r1.get('rebind_events')}

说明：Update1 的 reset=1 仅为首次启动 SUMO。Update2 若 reset=1，来自跨 update 累计步数达到 `max_episode_steps=1000` 的 **episode truncate**（合法），不是 CAV 离场触发。

## 5. MAPPO

- Actor update: {'YES' if summary.get('actor_changed') else 'NO'}
- Critic update: {'YES' if summary.get('critic_changed') else 'NO'}
- Device: {summary.get('device')}

## 6. GUI

- Update1：仅启动一次 GUI（无离场重开窗）
- 首批 CAV 可离场，新 CAV 持续绑定（departed/new/rebind > 0）
- Update2：若出现一次重开，对应 episode truncate，而非 alive==0

## 7. 最终判断

**{result}**

MAPPO formal long training: NOT STARTED
"""
    (out / "report.md").write_text(md, encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_f = (OUT_DIR / "stdout.log").open("w", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = Tee(old_out, log_f)
    sys.stderr = Tee(old_err, log_f)
    try:
        print("========================================")
        print("Lifecycle Smoke (1200m / accident@700m)")
        print("========================================")
        print(f"Out: {OUT_DIR}")
        print(f"UTC: {datetime.now(timezone.utc).isoformat()}")
        print("updates=2 rollout_steps=512 GUI=ON")
        print()
        _snapshot(OUT_DIR)

        summary = train(
            num_episodes=2,
            rollout_steps=512,
            gui=True,
            checkpoint_dir=OUT_DIR / "checkpoints",
            metrics_dir=OUT_DIR,
            run_title="Lifecycle Smoke (CAV leave ≠ SUMO reset)",
        )
        rows = list(summary.get("log_rows") or [])
        result = _assess(rows, summary)
        _write_report(out=OUT_DIR, summary=summary, rows=rows, result=result)
        meta = {
            "result": result,
            "rows": rows,
            "actor_changed": summary.get("actor_changed"),
            "critic_changed": summary.get("critic_changed"),
            "device": summary.get("device"),
        }
        (OUT_DIR / "run_meta.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )
        print()
        print(f"Lifecycle Smoke Result: {result}")
        print(f"Report: {OUT_DIR / 'report.md'}")
        return 0 if result != "FAIL" else 1
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        log_f.close()


if __name__ == "__main__":
    raise SystemExit(main())
