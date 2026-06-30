"""Metric containers for gate-traversal simulation tasks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from moewe.sim.frames import GRAVITY_M_S2
from moewe.sim.state import FlightState


class FailureReason(StrEnum):
    """Explicit terminal status for task evaluation."""

    NONE = "none"
    FLOOR = "floor"
    CEILING = "ceiling"
    WALL = "wall"
    STALL_LIMIT = "stall_limit"
    TIMEOUT = "timeout"
    NON_FINITE_STATE = "non_finite_state"


@dataclass(frozen=True)
class GateTaskMetrics:
    """Public metrics for one gate-traversal rollout."""

    success: bool
    gate_crossed: bool
    gate_miss_distance_m: float
    min_safety_margin_m: float
    flight_time_s: float
    terminal_specific_energy_margin_j_kg: float
    max_angle_of_attack_rad: float
    failure_reason: FailureReason


def specific_energy_j_kg(state: FlightState, g_m_s2: float = GRAVITY_M_S2) -> float:
    """Return specific mechanical energy using world z-up height."""

    speed2 = float(np.dot(state.velocity_b_m_s, state.velocity_b_m_s))
    return 0.5 * speed2 + float(g_m_s2) * float(state.position_w_m[2])
