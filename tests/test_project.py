"""Phase 0: project skeleton and Python environment checks."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRS = [
    "env",
    "agents",
    "models",
    "algorithms",
    "configs",
    "scripts",
    "experiments",
    "results",
    "tests",
    "sumo",
]

REQUIRED_CONFIGS = [
    "configs/env.yaml",
    "configs/model.yaml",
    "configs/train.yaml",
    "configs/reward.yaml",
]


def test_project_directories() -> None:
    missing = [d for d in REQUIRED_DIRS if not (PROJECT_ROOT / d).is_dir()]
    assert not missing, f"Missing directories: {missing}"


def test_config_files() -> None:
    missing = [p for p in REQUIRED_CONFIGS if not (PROJECT_ROOT / p).is_file()]
    assert not missing, f"Missing config files: {missing}"


def test_python_version() -> None:
    assert sys.version_info[:2] == (3, 11), (
        f"Expected Python 3.11, got {sys.version_info.major}.{sys.version_info.minor}"
    )


def test_pytorch() -> None:
    import torch

    assert torch.__version__, "torch version missing"
    # CUDA detection must not raise even if unavailable.
    _ = torch.cuda.is_available()


def test_numpy() -> None:
    import numpy as np

    assert np.__version__, "numpy version missing"


def main() -> int:
    checks = [
        ("project directories", test_project_directories),
        ("config files", test_config_files),
        ("Python 3.11", test_python_version),
        ("PyTorch", test_pytorch),
        ("NumPy", test_numpy),
    ]
    failed = 0
    for name, fn in checks:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as exc:  # noqa: BLE001 - report all Phase 0 failures
            failed += 1
            print(f"[FAIL] {name}: {exc}")
    if failed:
        print(f"\n{failed} check(s) failed.")
        return 1
    print("\nAll Phase 0 project checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
