"""Common local-controller interface and command-limit handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from moewe.sim.actuator import (
    NAUSICAA_SURFACE_LOWER_LIMITS_RAD,
    NAUSICAA_SURFACE_UPPER_LIMITS_RAD,
)
from moewe.sim.state import FlightState


def _command_vector(value: np.ndarray | tuple[float, float, float] | list[float]) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(3)


@dataclass(frozen=True)
class CommandLimits:
    """Command limits for aileron, elevator, and rudder in radians."""

    lower_rad: tuple[float, float, float] = NAUSICAA_SURFACE_LOWER_LIMITS_RAD
    upper_rad: tuple[float, float, float] = NAUSICAA_SURFACE_UPPER_LIMITS_RAD

    def __post_init__(self) -> None:
        lower = _command_vector(self.lower_rad)
        upper = _command_vector(self.upper_rad)
        if np.any(lower > upper):
            raise ValueError("Command lower limits must not exceed upper limits.")

    @property
    def lower(self) -> np.ndarray:
        return _command_vector(self.lower_rad)

    @property
    def upper(self) -> np.ndarray:
        return _command_vector(self.upper_rad)

    def clip(self, command_rad: np.ndarray) -> np.ndarray:
        return np.clip(_command_vector(command_rad), self.lower, self.upper)


@dataclass(frozen=True)
class ControllerMetadata:
    """Small metadata record for later ablations and primitive generation."""

    controller_type: str
    description: str
    reference: str = "local_linear_model"


class LocalController(Protocol):
    """Protocol shared by LQR, PD, and later local controllers."""

    reference_state: FlightState
    reference_command_rad: np.ndarray
    command_limits: CommandLimits
    metadata: ControllerMetadata

    def command(self, time_s: float, state: FlightState) -> np.ndarray:
        """Return an aileron, elevator, rudder command vector in radians."""
