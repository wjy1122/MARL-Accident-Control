"""Phase 3.2: env.step(actions) closed-loop tests."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.action import KEEP, LEFT, RIGHT  # noqa: E402
from env.highway_accident_env import HighwayAccidentEnv  # noqa: E402
from env.observation import OBS_DIM  # noqa: E402


def _wait_for_cavs(env: HighwayAccidentEnv, min_n: int = 1, max_steps: int = 2000) -> None:
    for _ in range(max_steps):
        if len(env.cav_ids) >= min_n:
            return
        env.step()  # legacy advance
    raise RuntimeError(f"Expected >= {min_n} CAV(s), got {len(env.cav_ids)}")


def main() -> int:
    results: list[tuple[str, bool, str]] = []
    env: HighwayAccidentEnv | None = None

    try:
        env = HighwayAccidentEnv(gui=False)
        snap = env.reset()
        results.append(
            (
                "reset step_count==0",
                env.step_count == 0 and snap["step_count"] == 0,
                f"step_count={env.step_count}",
            )
        )

        _wait_for_cavs(env, min_n=1)

        # Test 1: basic step(actions)
        actions = {cav_id: KEEP for cav_id in env.cav_ids}
        out = env.step(actions)
        results.append(
            (
                "Test1 step returns 5-tuple",
                isinstance(out, tuple) and len(out) == 5,
                f"type={type(out)}",
            )
        )
        obs, rewards, terminated, truncated, info = out  # type: ignore[misc]
        results.append(
            (
                "Test1 fields",
                isinstance(obs, dict)
                and isinstance(rewards, dict)
                and isinstance(terminated, bool)
                and isinstance(truncated, bool)
                and isinstance(info, dict),
                f"keys info={list(info.keys())}",
            )
        )

        # Test 2: observation dim
        dim_ok = all(
            hasattr(v, "__len__") and len(v) == OBS_DIM for v in obs.values()
        ) and len(obs) > 0
        results.append(
            (
                "Test2 obs dim==17",
                dim_ok,
                f"n={len(obs)} dims={[len(v) for v in list(obs.values())[:3]]}",
            )
        )

        # Test 3: rewards in [0,1]
        reward_ok = len(rewards) > 0 and all(
            isinstance(r, float) and 0.0 <= r <= 1.0 for r in rewards.values()
        )
        results.append(
            (
                "Test3 rewards in [0,1]",
                reward_ok,
                f"sample={list(rewards.items())[:3]}",
            )
        )

        # Test 4: step_count increments with closed-loop steps
        # Reset count baseline after previous steps: record current, do 3 more.
        base = env.step_count
        for _ in range(3):
            cur_actions = {cav_id: KEEP for cav_id in env.cav_ids}
            env.step(cur_actions)
        results.append(
            (
                "Test4 step_count +3",
                env.step_count == base + 3,
                f"base={base} now={env.step_count}",
            )
        )

        # Test 5: mixed actions without crash
        _wait_for_cavs(env, min_n=1)
        ids = list(env.cav_ids)
        mixed: dict[str, int] = {}
        for i, cav_id in enumerate(ids):
            mixed[cav_id] = [KEEP, LEFT, RIGHT][i % 3]
        try:
            obs2, rewards2, term2, trunc2, info2 = env.step(mixed)
            results.append(
                (
                    "Test5 mixed actions",
                    isinstance(obs2, dict) and isinstance(rewards2, dict),
                    f"actions={mixed} n_obs={len(obs2)}",
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(("Test5 mixed actions", False, str(exc)))

        # Test 6: max_episode_steps → truncated
        env.close()
        from env.highway_accident_env import load_config

        cfg = copy.deepcopy(load_config())
        cfg.setdefault("episode", {})["max_episode_steps"] = 3
        env = HighwayAccidentEnv(config=cfg, gui=False)
        env.reset()
        trunc_flags: list[bool] = []
        info: dict = {}
        for _ in range(3):
            acts = {cav_id: KEEP for cav_id in env.cav_ids}
            _, _, _, trunc, info = env.step(acts)
            trunc_flags.append(bool(trunc))
        results.append(
            (
                "Test6 truncated at max_episode_steps",
                trunc_flags[-1] is True and env.step_count == 3,
                f"flags={trunc_flags} step_count={env.step_count} info_step={info.get('step')}",
            )
        )

        env.close()
        results.append(("close()", True, "ok"))

    except Exception as exc:  # noqa: BLE001
        import traceback

        results.append(("test_step run", False, f"{exc}\n{traceback.format_exc()}"))
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
    print("\nAll Phase 3.2 step checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
