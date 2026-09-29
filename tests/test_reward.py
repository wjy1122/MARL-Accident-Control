"""Phase 2.4: reward unit tests (no SUMO required for formula checks)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.reward import (  # noqa: E402
    compute_group_reward,
    compute_individual_reward,
    compute_total_reward,
)


def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def main() -> int:
    results: list[tuple[str, bool, str]] = []

    # Test 1: Individual Reward
    cases = [(0.0, 0.0), (15.0, 0.5), (30.0, 1.0), (-5.0, 0.0), (45.0, 1.0)]
    ind_ok = True
    detail = []
    for speed, expected in cases:
        got = compute_individual_reward(speed, 30.0)
        detail.append(f"{speed}->{got}")
        if not _approx(got, expected):
            ind_ok = False
    results.append(("Test1 individual reward", ind_ok, ", ".join(detail)))

    # Test 2: Group Reward with neighbors
    individuals = {"ego": 0.8, "n1": 0.4, "n2": 0.6}
    group = compute_group_reward(individuals, ["n1", "n2"], "ego")
    results.append(
        ("Test2 group reward mean", _approx(group, 0.5), f"group={group}")
    )

    # Test 3: No neighbor
    group_alone = compute_group_reward({"ego": 0.8}, [], "ego")
    results.append(
        (
            "Test3 no neighbor fallback",
            _approx(group_alone, 0.8),
            f"group={group_alone}",
        )
    )

    # Test 4: Total Reward
    total = compute_total_reward(
        0.8, 0.5, individual_weight=0.7, group_weight=0.3
    )
    results.append(
        ("Test4 total reward", _approx(total, 0.71), f"total={total}")
    )

    # Test 5: Range under random inputs
    rng = random.Random(42)
    range_ok = True
    for _ in range(500):
        speed = rng.uniform(-20.0, 60.0)
        ind = compute_individual_reward(speed, 30.0)
        n = rng.randint(0, 5)
        inds = {"ego": ind}
        neighbors: list[str] = []
        for k in range(n):
            nid = f"n{k}"
            inds[nid] = compute_individual_reward(rng.uniform(-10.0, 50.0), 30.0)
            neighbors.append(nid)
        grp = compute_group_reward(inds, neighbors, "ego")
        tot = compute_total_reward(ind, grp, individual_weight=0.7, group_weight=0.3)
        if not (0.0 <= ind <= 1.0 and 0.0 <= grp <= 1.0 and 0.0 <= tot <= 1.0):
            range_ok = False
            break
    results.append(("Test5 range [0,1]", range_ok, "500 random samples"))

    failed = 0
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
        if not ok:
            failed += 1

    if failed:
        print(f"\n{failed} check(s) failed.")
        return 1
    print("\nAll Phase 2.4 reward checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
