"""Phase 4.3: GAE unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.gae import compute_gae  # noqa: E402


def _approx(a: float, b: float, tol: float = 1e-5) -> bool:
    return abs(float(a) - float(b)) <= tol


def main() -> int:
    results: list[tuple[str, bool, str]] = []

    # Shape test
    t, n = 5, 3
    rewards = np.random.randn(t, n).astype(np.float32)
    values = np.random.randn(t, n).astype(np.float32)
    dones = np.zeros(t, dtype=np.bool_)
    adv, ret = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)
    results.append(
        (
            "shape",
            adv.shape == (t, n) and ret.shape == (t, n),
            f"adv={adv.shape} ret={ret.shape}",
        )
    )
    results.append(
        (
            "returns = adv + values",
            np.allclose(ret, adv + values, atol=1e-5),
            "ok",
        )
    )

    # Terminal blocks bootstrap
    rewards = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
    values = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
    dones = np.array([False, True, False], dtype=np.bool_)
    next_values = np.array([100.0], dtype=np.float32)  # should not leak across done
    adv, ret = compute_gae(
        rewards,
        values,
        dones,
        gamma=0.5,
        gae_lambda=1.0,
        next_values=next_values,
    )
    # At t=1 done=True => next_nonterminal=0, so delta_1 = 1 + 0 - 0 = 1, A_1=1
    # At t=0: delta_0 = 1 + 0.5*V1*(1-0) - 0 = 1 + 0 = 1, A_0 = 1 + 0.5*1*A_1 = 1.5
    results.append(
        (
            "terminal cuts bootstrap",
            _approx(adv[1, 0], 1.0) and _approx(adv[0, 0], 1.5),
            f"A={adv.ravel().tolist()}",
        )
    )

    # Hand-calculated T=3, N=1, gamma=1, lambda=1, values=0, no done until end
    # r = [1, 1, 1], V=0, done=[F,F,T], next_values=0
    # delta_t = r_t for all
    # A_2 = 1
    # A_1 = 1 + 1*1*1 = 2
    # A_0 = 1 + 1*1*2 = 3
    rewards = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
    values = np.zeros((3, 1), dtype=np.float32)
    dones = np.array([False, False, True], dtype=np.bool_)
    adv, ret = compute_gae(
        rewards, values, dones, gamma=1.0, gae_lambda=1.0, next_values=np.zeros(1)
    )
    expected = np.array([[3.0], [2.0], [1.0]], dtype=np.float32)
    results.append(
        (
            "hand calc T=3 N=1",
            np.allclose(adv, expected, atol=1e-5),
            f"got={adv.ravel().tolist()} expected={expected.ravel().tolist()}",
        )
    )
    results.append(
        (
            "hand returns",
            np.allclose(ret, expected, atol=1e-5),
            f"ret={ret.ravel().tolist()}",
        )
    )

    # Mask-aware GAE: vacant middle step must not pollute earlier advantages.
    # Slot 0: active, vacant, active. Rewards/values only matter on active steps.
    rewards = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    values = np.array([[0.0, 9.0], [9.0, 9.0], [0.0, 9.0]], dtype=np.float32)
    dones = np.array([False, False, False], dtype=np.bool_)
    mask = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    adv_m, _ret_m = compute_gae(
        rewards,
        values,
        dones,
        gamma=1.0,
        gae_lambda=1.0,
        next_values=np.zeros(2),
        agent_mask=mask,
    )
    vacant_zero = _approx(adv_m[1, 0], 0.0) and _approx(adv_m[0, 1], 0.0)
    # t=0 must not carry through vacant t=1: A_0 = delta_0 with slot_continue=0
    # delta_0 = 1 + 1*V1*0 - 0 = 1, A_0 = 1
    t0_ok = _approx(adv_m[0, 0], 1.0)
    results.append(
        (
            "mask zeros vacant and cuts carry",
            vacant_zero and t0_ok,
            f"A={adv_m[:, 0].tolist()}",
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
    print("\nAll GAE checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
