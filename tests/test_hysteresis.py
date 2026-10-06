"""Unit tests for Dual-Threshold Hysteresis and Refractory Cooldown."""

import pytest
from pgm_engine.memory.hysteresis import HysteresisConfig, HysteresisController


def test_hysteresis_thresholds():
    config = HysteresisConfig(tau_push=0.75, tau_pop=0.25, t_cooldown=5)
    controller = HysteresisController(config)

    # Initial admission tests
    assert controller.should_push("node_1", utility=0.80) is True
    assert controller.should_push("node_2", utility=0.70) is False

    # Pop eviction tests
    assert controller.should_pop(utility=0.20) is True
    assert controller.should_pop(utility=0.30) is False


def test_refractory_quarantine_anti_thrashing():
    config = HysteresisConfig(tau_push=0.75, tau_pop=0.25, t_cooldown=5, refractory_penalty=0.50)
    controller = HysteresisController(config)

    # Admit and then evict node_1 at step 0
    assert controller.should_push("node_1", utility=0.80) is True
    controller.record_eviction("node_1")

    # Immediate re-admission at same step should be penalized:
    # tau_push_eff = 0.75 + 0.50 = 1.25.
    # Utility 0.80 was sufficient before, but fails now
    assert controller.is_in_cooldown("node_1") is True
    assert controller.should_push("node_1", utility=0.80) is False
    assert controller.total_thrash_preventions >= 1

    # Advance steps through cooldown window
    for _ in range(3):
        controller.step()

    # Still in cooldown (elapsed 3 < 5)
    assert controller.is_in_cooldown("node_1") is True
    # Penalty decayed: (5 - 3)/5 * 0.50 = 0.20 -> effective threshold 0.95
    assert controller.should_push("node_1", utility=0.90) is False

    # Advance past cooldown window
    controller.step()
    controller.step()  # step 5
    assert controller.is_in_cooldown("node_1") is False
    # Threshold restored to base 0.75
    assert controller.should_push("node_1", utility=0.80) is True
