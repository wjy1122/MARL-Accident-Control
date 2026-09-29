"""Phase 2.3: CAV local observation tests (headless)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.highway_accident_env import HighwayAccidentEnv  # noqa: E402
from env.observation import OBS_DIM, OBS_KEYS, build_observation  # noqa: E402


def main() -> int:
    results: list[tuple[str, bool, str]] = []
    env: HighwayAccidentEnv | None = None

    try:
        env = HighwayAccidentEnv(gui=False)
        env.reset()

        # Advance until at least one CAV exists.
        last = None
        for _ in range(2000):
            last = env.step()
            if last["n_cav"] > 0:
                break
        assert last is not None

        results.append(("find CAV", last["n_cav"] > 0, f"cav_ids={last['cav_ids'][:5]}"))

        cav_id = last["cav_ids"][0]
        obs = env.get_observation(cav_id)
        named = env.get_observation_dict(cav_id)

        results.append(
            ("generate observation", obs is not None and len(named) == OBS_DIM, cav_id)
        )
        results.append(
            (
                "fixed dimension",
                isinstance(obs, np.ndarray) and obs.shape == (OBS_DIM,),
                f"shape={getattr(obs, 'shape', None)} expected=({OBS_DIM},)",
            )
        )
        results.append(
            (
                "OBS_KEYS alignment",
                list(named.keys()) == OBS_KEYS,
                f"n_keys={len(named)}",
            )
        )
        results.append(
            ("no NaN", bool(np.isfinite(obs).all() and not np.isnan(obs).any()), str(obs[:5]))
        )
        results.append(
            ("no Inf", bool(np.isfinite(obs).all()), f"finite={np.isfinite(obs).all()}")
        )

        # Missing side-lane / missing neighbors must not raise.
        try:
            for vid in list(last["cav_ids"]):
                _ = env.get_observation(vid)
            # Also force edge cases via synthetic vehicles map.
            vehicles = dict(last["vehicles"])
            ego = dict(vehicles[cav_id])
            # Put ego on leftmost lane (no left neighbor lane).
            ego["lane"] = 2
            vehicles[cav_id] = ego
            edge = build_observation(cav_id=cav_id, vehicles=vehicles, cfg=env.cfg)
            results.append(
                (
                    "no left lane safe",
                    edge.shape == (OBS_DIM,) and np.isfinite(edge).all(),
                    f"lane=2 left_front={edge[7]:.2f}",
                )
            )
            # Put ego on rightmost lane (no right neighbor lane).
            ego2 = dict(ego)
            ego2["lane"] = 0
            vehicles[cav_id] = ego2
            edge2 = build_observation(cav_id=cav_id, vehicles=vehicles, cfg=env.cfg)
            results.append(
                (
                    "no right lane safe",
                    edge2.shape == (OBS_DIM,) and np.isfinite(edge2).all(),
                    f"lane=0 right_front={edge2[11]:.2f}",
                )
            )
            # Ego alone on road (no neighbors).
            solo = {cav_id: ego2}
            solo_obs = build_observation(cav_id=cav_id, vehicles=solo, cfg=env.cfg)
            # Missing neighbors -> distance normalized to 1.0
            ok_pad = (
                abs(float(solo_obs[3]) - 1.0) < 1e-5  # front_distance
                and abs(float(solo_obs[4])) < 1e-5  # front_relative_speed
            )
            results.append(
                (
                    "no neighbors safe (padding)",
                    solo_obs.shape == (OBS_DIM,) and ok_pad,
                    f"front_d={solo_obs[3]:.3f} front_rs={solo_obs[4]:.3f}",
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(("edge-case observations", False, str(exc)))

        # Normalized ranges: distances/speeds/lanes in [-1, 1] or [0, 1].
        in_range = bool(np.all(obs >= -1.0 - 1e-5) and np.all(obs <= 1.0 + 1e-5))
        results.append(("normalized to [-1, 1]", in_range, f"min={obs.min():.3f} max={obs.max():.3f}"))

        # Multiple CAVs.
        for _ in range(500):
            last = env.step()
            if last["n_cav"] >= 2:
                break
        multi = env.get_observations()
        results.append(
            (
                "multiple CAV observations",
                len(multi) == last["n_cav"]
                and all(v.shape == (OBS_DIM,) and np.isfinite(v).all() for v in multi.values()),
                f"n={len(multi)}",
            )
        )

        env.close()
        results.append(("close()", not env.is_connected, "ok"))

    except Exception as exc:  # noqa: BLE001
        results.append(("observation run", False, str(exc)))
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
    print("\nAll Phase 2.3 observation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
