"""Phase 4.4: smoke test for closed-loop MAPPO training (2–3 episodes)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_mappo import train  # noqa: E402


def main() -> int:
    results: list[tuple[str, bool, str]] = []

    # Short rollouts keep the smoke test fast; pipeline must still be complete.
    summary = train(num_episodes=3, rollout_steps=32, gui=False)

    results.append(
        (
            "episodes ran",
            summary["num_episodes"] == 3 and len(summary["log_rows"]) == 3,
            f"n={summary['num_episodes']} rows={len(summary['log_rows'])}",
        )
    )
    results.append(
        (
            "device set",
            summary["device"] in ("cpu", "cuda"),
            f"device={summary['device']}",
        )
    )
    results.append(
        (
            "actor params changed",
            bool(summary["actor_changed"]),
            str(summary["actor_changed"]),
        )
    )
    results.append(
        (
            "critic params changed",
            bool(summary["critic_changed"]),
            str(summary["critic_changed"]),
        )
    )

    finite = all(
        math.isfinite(x)
        for x in (
            summary["actor_loss"],
            summary["critic_loss"],
            summary["entropy"],
            summary["first_reward"],
            summary["final_reward"],
        )
    )
    results.append(("no NaN/Inf", finite, "ok" if finite else "bad"))

    ckpts = [Path(p) for p in summary["checkpoints"]]
    results.append(
        (
            "checkpoint saved",
            len(ckpts) > 0 and all(p.is_file() for p in ckpts),
            f"n={len(ckpts)} last={ckpts[-1] if ckpts else None}",
        )
    )

    # Verify checkpoint contents
    if ckpts:
        import torch

        ckpt = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        keys_ok = all(
            k in ckpt
            for k in (
                "actor_state_dict",
                "critic_state_dict",
                "optimizer_state_dict",
                "episode",
                "config",
            )
        )
        results.append(("checkpoint keys", keys_ok, str(sorted(ckpt.keys()))))
    else:
        results.append(("checkpoint keys", False, "no checkpoint"))

    failed = 0
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
        if not ok:
            failed += 1

    if failed:
        print(f"\n{failed} check(s) failed.")
        return 1
    print("\nPhase 4.4 smoke test PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
