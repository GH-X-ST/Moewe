"""Scenario dataclasses for deterministic and seeded task rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moewe.sim.state import FlightState

TRACKER_LIMIT_X_W_M = (0.0, 8.0)
TRACKER_LIMIT_Y_W_M = (0.0, 4.8)
TRACKER_LIMIT_Z_W_M = (0.0, 3.5)

TRUE_SAFE_X_W_M = (1.2, 6.6)
TRUE_SAFE_Y_W_M = (0.0, 4.4)
TRUE_SAFE_Z_W_M = (0.4, 3.5)

LAUNCH_GATE_X_W_M = (1.2, 1.4)
LAUNCH_GATE_Y_W_M = (1.8, 2.2)
LAUNCH_GATE_Z_W_M = (1.3, 1.8)
LAUNCH_GATE_NOMINAL_POSITION_W_M = (
    0.5 * (LAUNCH_GATE_X_W_M[0] + LAUNCH_GATE_X_W_M[1]),
    0.5 * (LAUNCH_GATE_Y_W_M[0] + LAUNCH_GATE_Y_W_M[1]),
    0.5 * (LAUNCH_GATE_Z_W_M[0] + LAUNCH_GATE_Z_W_M[1]),
)
LAUNCH_GATE_ROLL_LIMIT_RAD = float(np.deg2rad(20.0))
LAUNCH_GATE_PITCH_MIN_RAD = float(np.deg2rad(-10.0))
LAUNCH_GATE_PITCH_MAX_RAD = float(np.deg2rad(20.0))
LAUNCH_GATE_YAW_LIMIT_RAD = float(np.deg2rad(20.0))
LAUNCH_GATE_FORWARD_SPEED_M_S = (4.0, 8.0)
LAUNCH_GATE_SIDE_VELOCITY_LIMIT_M_S = 1.5
LAUNCH_GATE_VERTICAL_BODY_VELOCITY_LIMIT_M_S = 0.9
LAUNCH_GATE_ROLL_RATE_LIMIT_RAD_S = 1.2
LAUNCH_GATE_PITCH_RATE_LIMIT_RAD_S = 1.2
LAUNCH_GATE_YAW_RATE_LIMIT_RAD_S = 1.8

FRONT_EXIT_X_W_M = TRUE_SAFE_X_W_M[1]
FRONT_EXIT_GATE_CENTRE_W_M = (
    FRONT_EXIT_X_W_M,
    0.5 * (TRUE_SAFE_Y_W_M[0] + TRUE_SAFE_Y_W_M[1]),
    0.5 * (TRUE_SAFE_Z_W_M[0] + TRUE_SAFE_Z_W_M[1]),
)
FRONT_EXIT_GATE_NORMAL_W = (1.0, 0.0, 0.0)
FRONT_EXIT_GATE_WIDTH_M = TRUE_SAFE_Y_W_M[1] - TRUE_SAFE_Y_W_M[0]
FRONT_EXIT_GATE_HEIGHT_M = TRUE_SAFE_Z_W_M[1] - TRUE_SAFE_Z_W_M[0]

SINGLE_FAN_CENTER_XY_M = (4.2, 2.4)
FOUR_FAN_CENTERS_XY_M = (
    (3.0, 3.6),
    (5.4, 3.6),
    (3.0, 1.2),
    (5.4, 1.2),
)


def _vec3(value: np.ndarray | tuple[float, float, float] | list[float]) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(3)


@dataclass(frozen=True)
class FlightVolume:
    """Axis-aligned flight volume in world z-up coordinates."""

    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    z_min_m: float
    z_max_m: float

    def validate(self) -> None:
        if self.x_min_m >= self.x_max_m:
            raise ValueError("x_min_m must be smaller than x_max_m.")
        if self.y_min_m >= self.y_max_m:
            raise ValueError("y_min_m must be smaller than y_max_m.")
        if self.z_min_m >= self.z_max_m:
            raise ValueError("z_min_m must be smaller than z_max_m.")

    def margins_m(self, position_w_m: np.ndarray) -> np.ndarray:
        self.validate()
        x, y, z = _vec3(position_w_m)
        return np.array(
            [
                x - self.x_min_m,
                self.x_max_m - x,
                y - self.y_min_m,
                self.y_max_m - y,
                z - self.z_min_m,
                self.z_max_m - z,
            ],
            dtype=float,
        )

    def min_margin_m(self, position_w_m: np.ndarray) -> float:
        return float(np.min(self.margins_m(position_w_m)))

    def failure_reason(self, position_w_m: np.ndarray) -> str | None:
        x, y, z = _vec3(position_w_m)
        if z < self.z_min_m:
            return "floor"
        if z > self.z_max_m:
            return "ceiling"
        if x < self.x_min_m or x > self.x_max_m or y < self.y_min_m or y > self.y_max_m:
            return "wall"
        return None


TRACKER_LIMIT_FLIGHT_VOLUME = FlightVolume(
    x_min_m=TRACKER_LIMIT_X_W_M[0],
    x_max_m=TRACKER_LIMIT_X_W_M[1],
    y_min_m=TRACKER_LIMIT_Y_W_M[0],
    y_max_m=TRACKER_LIMIT_Y_W_M[1],
    z_min_m=TRACKER_LIMIT_Z_W_M[0],
    z_max_m=TRACKER_LIMIT_Z_W_M[1],
)
TRUE_SAFE_FLIGHT_VOLUME = FlightVolume(
    x_min_m=TRUE_SAFE_X_W_M[0],
    x_max_m=TRUE_SAFE_X_W_M[1],
    y_min_m=TRUE_SAFE_Y_W_M[0],
    y_max_m=TRUE_SAFE_Y_W_M[1],
    z_min_m=TRUE_SAFE_Z_W_M[0],
    z_max_m=TRUE_SAFE_Z_W_M[1],
)


def state_is_launch_gate_compliant(state: FlightState) -> bool:
    """Return whether a state lies inside the measured physical launch gate."""

    if not state.finite():
        return False
    x_w, y_w, z_w = state.position_w_m
    roll, pitch, yaw = state.euler_rad
    u_b, v_b, w_b = state.velocity_b_m_s
    p_b, q_b, r_b = state.rates_b_rad_s
    return bool(
        LAUNCH_GATE_X_W_M[0] <= x_w <= LAUNCH_GATE_X_W_M[1]
        and LAUNCH_GATE_Y_W_M[0] <= y_w <= LAUNCH_GATE_Y_W_M[1]
        and LAUNCH_GATE_Z_W_M[0] <= z_w <= LAUNCH_GATE_Z_W_M[1]
        and -LAUNCH_GATE_ROLL_LIMIT_RAD <= roll <= LAUNCH_GATE_ROLL_LIMIT_RAD
        and LAUNCH_GATE_PITCH_MIN_RAD <= pitch <= LAUNCH_GATE_PITCH_MAX_RAD
        and -LAUNCH_GATE_YAW_LIMIT_RAD <= yaw <= LAUNCH_GATE_YAW_LIMIT_RAD
        and LAUNCH_GATE_FORWARD_SPEED_M_S[0] <= u_b <= LAUNCH_GATE_FORWARD_SPEED_M_S[1]
        and abs(float(v_b)) <= LAUNCH_GATE_SIDE_VELOCITY_LIMIT_M_S
        and abs(float(w_b)) <= LAUNCH_GATE_VERTICAL_BODY_VELOCITY_LIMIT_M_S
        and abs(float(p_b)) <= LAUNCH_GATE_ROLL_RATE_LIMIT_RAD_S
        and abs(float(q_b)) <= LAUNCH_GATE_PITCH_RATE_LIMIT_RAD_S
        and abs(float(r_b)) <= LAUNCH_GATE_YAW_RATE_LIMIT_RAD_S
    )


@dataclass(frozen=True)
class FixedInitialState:
    """Deterministic initial-state sampler."""

    state: FlightState

    def sample(self, seed: int | None = None) -> FlightState:
        del seed
        return self.state


@dataclass(frozen=True)
class UniformInitialStateSampler:
    """Seeded uniform sampler around a nominal state.

    Half-width arrays are ordered as the canonical 15-state vector.
    """

    nominal_state: FlightState
    half_width: np.ndarray

    def __post_init__(self) -> None:
        half_width = np.asarray(self.half_width, dtype=float).reshape(15)
        if np.any(half_width < 0.0):
            raise ValueError("Initial-state half widths must be non-negative.")
        object.__setattr__(self, "half_width", half_width)

    def sample(self, seed: int | None = None) -> FlightState:
        rng = np.random.default_rng(seed)
        delta = rng.uniform(-self.half_width, self.half_width)
        return FlightState.from_vector(self.nominal_state.as_vector() + delta)


@dataclass(frozen=True)
class Scenario:
    """Small container binding a task, initial state, and optional wind model."""

    name: str
    initial_state_sampler: FixedInitialState | UniformInitialStateSampler
    wind_model: object | None = None
    wind_mode: str = "panel"
    seed: int | None = None

    def initial_state(self) -> FlightState:
        return self.initial_state_sampler.sample(self.seed)
