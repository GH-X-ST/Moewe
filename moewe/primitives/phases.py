"""Deterministic phase definitions for structured manoeuvre primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from moewe.sim.state import FlightState


def _command_vector(value: np.ndarray | tuple[float, float, float] | list[float]) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(3)


def _finite_positive_duration(duration_s: float) -> float:
    value = float(duration_s)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("Primitive phase duration_s must be positive and finite.")
    return value


def _fraction(local_time_s: float, duration_s: float) -> float:
    return float(np.clip(float(local_time_s) / float(duration_s), 0.0, 1.0))


def _with_euler_and_surfaces(state: FlightState, euler_rad: np.ndarray, command_rad: np.ndarray) -> FlightState:
    return FlightState(
        position_w_m=state.position_w_m,
        euler_rad=np.asarray(euler_rad, dtype=float).reshape(3),
        velocity_b_m_s=state.velocity_b_m_s,
        rates_b_rad_s=state.rates_b_rad_s,
        surfaces_rad=_command_vector(command_rad),
    )


def interpolate_state(
    start_state: FlightState,
    end_state: FlightState,
    fraction: float,
    command_rad: np.ndarray | tuple[float, float, float] | list[float],
) -> FlightState:
    """Linearly interpolate a reference state in canonical state coordinates."""

    blend = float(np.clip(fraction, 0.0, 1.0))
    start = start_state.as_vector()
    end = end_state.as_vector()
    vector = (1.0 - blend) * start + blend * end
    vector[12:15] = _command_vector(command_rad)
    return FlightState.from_vector(vector)


@dataclass(frozen=True)
class PhaseSample:
    """Reference state and command at a local phase time.

    Command order is always ``[aileron, elevator, rudder]`` in radians.
    """

    state: FlightState
    command_rad: np.ndarray
    phase_name: str
    local_time_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_rad", _command_vector(self.command_rad))


class PrimitivePhase(Protocol):
    """Protocol shared by all deterministic primitive phases."""

    duration_s: float
    name: str

    def sample(self, local_time_s: float) -> PhaseSample:
        """Return the reference state and command at local phase time."""


@dataclass(frozen=True)
class HoldPhase:
    """Maintain a local operating point for a fixed duration."""

    duration_s: float
    reference_state: FlightState
    command_rad: np.ndarray | tuple[float, float, float] | list[float]
    name: str = "hold"

    def __post_init__(self) -> None:
        object.__setattr__(self, "duration_s", _finite_positive_duration(self.duration_s))
        object.__setattr__(self, "command_rad", _command_vector(self.command_rad))

    def sample(self, local_time_s: float) -> PhaseSample:
        del local_time_s
        state = self.reference_state.with_surfaces(self.command_rad)
        return PhaseSample(state=state, command_rad=self.command_rad, phase_name=self.name, local_time_s=0.0)


@dataclass(frozen=True)
class BankTransitionPhase:
    """Move from the current bank angle to a target bank angle."""

    duration_s: float
    start_state: FlightState
    command_rad: np.ndarray | tuple[float, float, float] | list[float]
    target_bank_rad: float
    name: str = "bank_transition"

    def __post_init__(self) -> None:
        object.__setattr__(self, "duration_s", _finite_positive_duration(self.duration_s))
        object.__setattr__(self, "command_rad", _command_vector(self.command_rad))
        if not np.isfinite(float(self.target_bank_rad)):
            raise ValueError("target_bank_rad must be finite.")

    def sample(self, local_time_s: float) -> PhaseSample:
        blend = _fraction(local_time_s, self.duration_s)
        euler = self.start_state.euler_rad.copy()
        euler[0] = (1.0 - blend) * float(self.start_state.euler_rad[0]) + blend * float(self.target_bank_rad)
        state = _with_euler_and_surfaces(self.start_state, euler, self.command_rad)
        return PhaseSample(state=state, command_rad=self.command_rad, phase_name=self.name, local_time_s=local_time_s)


@dataclass(frozen=True)
class PitchPulsePhase:
    """Apply a smooth pitch reference pulse that returns to the entry pitch."""

    duration_s: float
    reference_state: FlightState
    command_rad: np.ndarray | tuple[float, float, float] | list[float]
    delta_pitch_rad: float
    name: str = "pitch_pulse"

    def __post_init__(self) -> None:
        object.__setattr__(self, "duration_s", _finite_positive_duration(self.duration_s))
        object.__setattr__(self, "command_rad", _command_vector(self.command_rad))
        if not np.isfinite(float(self.delta_pitch_rad)):
            raise ValueError("delta_pitch_rad must be finite.")

    def sample(self, local_time_s: float) -> PhaseSample:
        blend = _fraction(local_time_s, self.duration_s)
        euler = self.reference_state.euler_rad.copy()
        euler[1] = float(self.reference_state.euler_rad[1]) + float(self.delta_pitch_rad) * np.sin(np.pi * blend)
        state = _with_euler_and_surfaces(self.reference_state, euler, self.command_rad)
        return PhaseSample(state=state, command_rad=self.command_rad, phase_name=self.name, local_time_s=local_time_s)


@dataclass(frozen=True)
class DwellPhase:
    """Remain at the current local reference state for a useful dwell period."""

    duration_s: float
    reference_state: FlightState
    command_rad: np.ndarray | tuple[float, float, float] | list[float]
    name: str = "dwell"

    def __post_init__(self) -> None:
        object.__setattr__(self, "duration_s", _finite_positive_duration(self.duration_s))
        object.__setattr__(self, "command_rad", _command_vector(self.command_rad))

    def sample(self, local_time_s: float) -> PhaseSample:
        del local_time_s
        state = self.reference_state.with_surfaces(self.command_rad)
        return PhaseSample(state=state, command_rad=self.command_rad, phase_name=self.name, local_time_s=0.0)


@dataclass(frozen=True)
class RecoveryPhase:
    """Return from a manoeuvre reference toward a nominal operating point."""

    duration_s: float
    start_state: FlightState
    target_state: FlightState
    command_rad: np.ndarray | tuple[float, float, float] | list[float]
    name: str = "recovery"

    def __post_init__(self) -> None:
        object.__setattr__(self, "duration_s", _finite_positive_duration(self.duration_s))
        object.__setattr__(self, "command_rad", _command_vector(self.command_rad))

    def sample(self, local_time_s: float) -> PhaseSample:
        blend = _fraction(local_time_s, self.duration_s)
        state = interpolate_state(self.start_state, self.target_state, blend, self.command_rad)
        return PhaseSample(state=state, command_rad=self.command_rad, phase_name=self.name, local_time_s=local_time_s)
