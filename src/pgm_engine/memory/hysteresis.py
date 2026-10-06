"""Dual-Threshold Hysteresis & Refractory Cooldown Controller.

Eliminates thrashing (rapid push-pop oscillations across consecutive steps)
using asymmetric utility thresholds and temporal refractory penalties.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class HysteresisConfig:
    """Parameters governing working buffer admission and eviction."""
    tau_push: float = 0.75
    tau_pop: float = 0.25
    t_cooldown: int = 5
    refractory_penalty: float = 0.50  # Additional threshold increment during cooldown


class HysteresisController:
    """Manages push/pop decisions with hysteresis and refractory quarantine."""

    def __init__(self, config: Optional[HysteresisConfig] = None):
        self.config = config or HysteresisConfig()
        # entity_id -> (eviction_step, eviction_count)
        self.eviction_history: Dict[str, tuple[int, int]] = {}
        self.current_step: int = 0
        self.total_thrash_preventions: int = 0

    def step(self) -> None:
        """Advances the iteration step counter."""
        self.current_step += 1

    def get_effective_push_threshold(self, entity_id: str) -> float:
        """Calculates effective push threshold incorporating refractory cooldown penalty."""
        base_threshold = self.config.tau_push
        if entity_id not in self.eviction_history:
            return base_threshold

        evict_step, _ = self.eviction_history[entity_id]
        elapsed = self.current_step - evict_step

        if elapsed >= self.config.t_cooldown:
            return base_threshold

        # Linear decay of refractory penalty over cooldown window
        decay = (self.config.t_cooldown - elapsed) / float(self.config.t_cooldown)
        penalty = self.config.refractory_penalty * decay
        return base_threshold + penalty

    def should_push(self, entity_id: str, utility: float) -> bool:
        """Evaluates if an entity qualifies for admission to L1."""
        threshold = self.get_effective_push_threshold(entity_id)
        qualifies = utility >= threshold
        if not qualifies and utility >= self.config.tau_push:
            # Blocked specifically by refractory penalty
            self.total_thrash_preventions += 1
        return qualifies

    def should_pop(self, utility: float) -> bool:
        """Evaluates if an active L1 entity should be evicted."""
        return utility < self.config.tau_pop

    def record_eviction(self, entity_id: str) -> None:
        """Logs an eviction event and places the entity into refractory cooldown."""
        prev_count = 0
        if entity_id in self.eviction_history:
            _, prev_count = self.eviction_history[entity_id]
        self.eviction_history[entity_id] = (self.current_step, prev_count + 1)

    def is_in_cooldown(self, entity_id: str) -> bool:
        if entity_id not in self.eviction_history:
            return False
        evict_step, _ = self.eviction_history[entity_id]
        return (self.current_step - evict_step) < self.config.t_cooldown

    def get_cooldown_remaining(self, entity_id: str) -> int:
        if entity_id not in self.eviction_history:
            return 0
        evict_step, _ = self.eviction_history[entity_id]
        remaining = self.config.t_cooldown - (self.current_step - evict_step)
        return max(0, remaining)
