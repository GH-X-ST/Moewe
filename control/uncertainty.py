"""Simultaneous aircraft, flow, sensing, and geometry bounds."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
import numpy.typing as npt

from control.flow import FlowBounds

FAST_PERIOD_S = 0.020
GOVERNOR_PERIOD_S = 0.100
PREDICTION_PERIOD_S = 0.200
PREDICTION_STAGES = 10
NEXT_UPDATE_STAGE = 5


@dataclass(frozen=True)
class Bounds:
    """Immutable bounds used by every generated certificate."""

    flow: FlowBounds
    density_kg_m3: tuple[float, float]
    aerodynamic_scale: tuple[float, float]
    force_residual_abs_n: npt.ArrayLike
    moment_residual_abs_n_m: npt.ArrayLike
    mass_kg: tuple[float, float]
    cg_residual_abs_m: npt.ArrayLike
    inertia_residual_abs_kg_m2: npt.ArrayLike
    actuator_tau_lower_s: npt.ArrayLike
    actuator_tau_upper_s: npt.ArrayLike
    command_error_abs_rad: npt.ArrayLike
    state_estimation_abs: npt.ArrayLike
    command_delay_s: tuple[float, float]
    nonlinear_remainder_abs: npt.ArrayLike
    numerical_remainder_abs: npt.ArrayLike
    body_inflation_m: float
    mission_position_error_abs_m: float
    mission_attitude_error_abs_rad: float
    roll_abs_max_rad: float
    pitch_abs_max_rad: float
    airspeed_m_s: tuple[float, float]
    alpha_abs_max_rad: float
    body_rate_abs_max_rad_s: float

    def __post_init__(self) -> None:
        arrays = (
            ("force_residual_abs_n", (3,)),
            ("moment_residual_abs_n_m", (3,)),
            ("cg_residual_abs_m", (3,)),
            ("inertia_residual_abs_kg_m2", (3, 3)),
            ("actuator_tau_lower_s", (3,)),
            ("actuator_tau_upper_s", (3,)),
            ("command_error_abs_rad", (3,)),
            ("state_estimation_abs", (15,)),
            ("nonlinear_remainder_abs", (15,)),
            ("numerical_remainder_abs", (15,)),
        )
        for name, shape in arrays:
            value = np.asarray(getattr(self, name), dtype=float).reshape(shape).copy()
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    @property
    def queue_length(self) -> int:
        """Return the number of issued commands spanning the delay bound."""

        return ceil(self.command_delay_s[1] / FAST_PERIOD_S)

    @property
    def delay_step_bounds(self) -> tuple[int, int]:
        """Return inclusive command-age indices at a fast update."""

        lower = int(np.floor(self.command_delay_s[0] / FAST_PERIOD_S))
        upper = ceil(self.command_delay_s[1] / FAST_PERIOD_S)
        return lower, upper

    @property
    def stage_remainder_abs(self) -> np.ndarray:
        """Return the combined generated and numerical stage remainder."""

        return self.nonlinear_remainder_abs + self.numerical_remainder_abs
