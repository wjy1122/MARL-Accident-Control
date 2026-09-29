"""Algorithms package (Phase 4.2–4.4)."""

from algorithms.gae import compute_gae
from algorithms.mappo_trainer import MAPPOTrainer
from algorithms.ppo_update import PPOUpdater, normalize_advantages
from algorithms.rollout import RolloutCollector, resolve_device
from algorithms.rollout_buffer import RolloutBatch, RolloutBuffer

__all__ = [
    "MAPPOTrainer",
    "PPOUpdater",
    "RolloutBatch",
    "RolloutBuffer",
    "RolloutCollector",
    "compute_gae",
    "normalize_advantages",
    "resolve_device",
]
