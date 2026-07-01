"""Task interfaces and metrics for Moewe simulation rollouts."""

from .gate import GatePlane, GateTraversalTask, front_exit_gate
from .metrics import FailureReason, GateTaskMetrics, specific_energy_j_kg
from .scenario import (
    FlightVolume,
    FOUR_FAN_CENTERS_XY_M,
    FRONT_EXIT_GATE_CENTRE_W_M,
    FRONT_EXIT_GATE_HEIGHT_M,
    FRONT_EXIT_GATE_NORMAL_W,
    FRONT_EXIT_GATE_WIDTH_M,
    LAUNCH_GATE_NOMINAL_POSITION_W_M,
    SINGLE_FAN_CENTER_XY_M,
    FixedInitialState,
    Scenario,
    TRUE_SAFE_FLIGHT_VOLUME,
    TRUE_SAFE_X_W_M,
    TRUE_SAFE_Y_W_M,
    TRUE_SAFE_Z_W_M,
    UniformInitialStateSampler,
    state_is_launch_gate_compliant,
)

__all__ = [
    "FailureReason",
    "FixedInitialState",
    "FlightVolume",
    "FOUR_FAN_CENTERS_XY_M",
    "FRONT_EXIT_GATE_CENTRE_W_M",
    "FRONT_EXIT_GATE_HEIGHT_M",
    "FRONT_EXIT_GATE_NORMAL_W",
    "FRONT_EXIT_GATE_WIDTH_M",
    "GatePlane",
    "GateTaskMetrics",
    "GateTraversalTask",
    "LAUNCH_GATE_NOMINAL_POSITION_W_M",
    "Scenario",
    "SINGLE_FAN_CENTER_XY_M",
    "TRUE_SAFE_FLIGHT_VOLUME",
    "TRUE_SAFE_X_W_M",
    "TRUE_SAFE_Y_W_M",
    "TRUE_SAFE_Z_W_M",
    "UniformInitialStateSampler",
    "front_exit_gate",
    "state_is_launch_gate_compliant",
    "specific_energy_j_kg",
]
