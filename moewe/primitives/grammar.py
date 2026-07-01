"""Structured primitive grammar specifications and safety checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moewe.sim.actuator import NAUSICAA_MAX_COMMAND_ABS_RAD
from moewe.sim.glider_model import GliderModel, NAUSICAA_OPERATIONAL_ALPHA_LIMIT_RAD
from moewe.sim.state import FlightState
from moewe.tasks.scenario import (
    LAUNCH_GATE_NOMINAL_POSITION_W_M,
    TRUE_SAFE_X_W_M,
    TRUE_SAFE_Y_W_M,
    TRUE_SAFE_Z_W_M,
)


def _finite_tuple(values: tuple[float, ...] | list[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.all(np.isfinite(np.asarray(result, dtype=float))):
        raise ValueError(f"{name} values must be finite.")
    return result


def _speed_m_s(state: FlightState) -> float:
    return float(np.linalg.norm(state.velocity_b_m_s))


@dataclass(frozen=True)
class OperatingPointSpec:
    """Pseudo-trim operating-point enumeration in SI units and radians."""

    airspeed_m_s: tuple[float, ...] = (7.0,)
    flight_path_angle_rad: tuple[float, ...] = (0.0,)
    altitude_m: tuple[float, ...] = (LAUNCH_GATE_NOMINAL_POSITION_W_M[2],)
    x_w_m: float = LAUNCH_GATE_NOMINAL_POSITION_W_M[0]
    y_w_m: float = LAUNCH_GATE_NOMINAL_POSITION_W_M[1]
    heading_rad: float = 0.0
    turn_rate_rad_s: float = 0.0
    command_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    wind_mode: str = "panel"

    def validate(self) -> None:
        speeds = _finite_tuple(self.airspeed_m_s, "airspeed_m_s")
        gammas = _finite_tuple(self.flight_path_angle_rad, "flight_path_angle_rad")
        altitudes = _finite_tuple(self.altitude_m, "altitude_m")
        if any(speed <= 0.0 for speed in speeds):
            raise ValueError("airspeed_m_s values must be positive.")
        if any(altitude <= 0.0 for altitude in altitudes):
            raise ValueError("altitude_m values must be positive.")
        finite_scalars = (self.x_w_m, self.y_w_m, self.heading_rad, self.turn_rate_rad_s)
        if not np.isfinite(finite_scalars).all():
            raise ValueError("x_w_m, y_w_m, heading_rad, and turn_rate_rad_s must be finite.")
        if np.asarray(self.command_rad, dtype=float).reshape(3).shape != (3,):
            raise ValueError("command_rad must be ordered as [aileron, elevator, rudder].")
        if self.wind_mode not in {"cg", "panel"}:
            raise ValueError("wind_mode must be 'cg' or 'panel'.")
        del gammas


@dataclass(frozen=True)
class BankTransitionSpec:
    """Bank transition grammar factors."""

    target_bank_rad: tuple[float, ...] = (-0.2, 0.0, 0.2)
    duration_s: float = 0.2

    def validate(self) -> None:
        _finite_tuple(self.target_bank_rad, "target_bank_rad")
        if not np.isfinite(float(self.duration_s)) or float(self.duration_s) <= 0.0:
            raise ValueError("bank transition duration_s must be positive and finite.")


@dataclass(frozen=True)
class PitchPulseSpec:
    """Pitch pulse grammar factors."""

    delta_pitch_rad: tuple[float, ...] = (-0.05, 0.0, 0.05)
    duration_s: float = 0.2

    def validate(self) -> None:
        _finite_tuple(self.delta_pitch_rad, "delta_pitch_rad")
        if not np.isfinite(float(self.duration_s)) or float(self.duration_s) <= 0.0:
            raise ValueError("pitch pulse duration_s must be positive and finite.")


@dataclass(frozen=True)
class DwellSpec:
    """Dwell-duration grammar factors."""

    duration_s: tuple[float, ...] = (0.1, 0.2)

    def validate(self) -> None:
        durations = _finite_tuple(self.duration_s, "dwell duration_s")
        if any(duration <= 0.0 for duration in durations):
            raise ValueError("dwell duration_s values must be positive.")


@dataclass(frozen=True)
class RecoverySpec:
    """Recovery phase grammar factors."""

    duration_s: float = 0.2
    mode: str = "nominal"

    def validate(self) -> None:
        if not np.isfinite(float(self.duration_s)) or float(self.duration_s) <= 0.0:
            raise ValueError("recovery duration_s must be positive and finite.")
        if self.mode != "nominal":
            raise ValueError("Only nominal recovery is implemented in the smoke grammar.")


@dataclass(frozen=True)
class PrimitiveEntryCondition:
    """State-family gate for entering a primitive."""

    min_airspeed_m_s: float = 3.0
    max_airspeed_m_s: float = 12.0
    min_altitude_m: float = TRUE_SAFE_Z_W_M[0]
    max_altitude_m: float = TRUE_SAFE_Z_W_M[1]
    max_abs_bank_rad: float = 0.8
    max_abs_pitch_rad: float = 0.8

    def rejection_reason(self, state: FlightState) -> str | None:
        if not state.finite():
            return "non_finite_state"
        speed = _speed_m_s(state)
        if speed < float(self.min_airspeed_m_s) or speed > float(self.max_airspeed_m_s):
            return "entry_airspeed"
        altitude = float(state.position_w_m[2])
        if altitude < float(self.min_altitude_m) or altitude > float(self.max_altitude_m):
            return "entry_altitude"
        if abs(float(state.euler_rad[0])) > float(self.max_abs_bank_rad):
            return "entry_bank"
        if abs(float(state.euler_rad[1])) > float(self.max_abs_pitch_rad):
            return "entry_pitch"
        return None

    def accepts(self, state: FlightState) -> bool:
        return self.rejection_reason(state) is None


@dataclass(frozen=True)
class PrimitiveSafetyReport:
    """Result of a primitive safety-limit check."""

    ok: bool
    failure_reason: str | None
    min_margin_m: float
    max_angle_of_attack_rad: float
    max_command_abs_rad: float


@dataclass(frozen=True)
class PrimitiveSafetyLimits:
    """Axis-aligned volume, attitude, angle-of-attack, and command limits."""

    x_min_m: float = TRUE_SAFE_X_W_M[0]
    x_max_m: float = TRUE_SAFE_X_W_M[1]
    y_min_m: float = TRUE_SAFE_Y_W_M[0]
    y_max_m: float = TRUE_SAFE_Y_W_M[1]
    z_min_m: float = TRUE_SAFE_Z_W_M[0]
    z_max_m: float = TRUE_SAFE_Z_W_M[1]
    max_abs_bank_rad: float = 1.0
    max_abs_angle_of_attack_rad: float = NAUSICAA_OPERATIONAL_ALPHA_LIMIT_RAD
    max_abs_command_rad: float = NAUSICAA_MAX_COMMAND_ABS_RAD

    def validate(self) -> None:
        if self.x_min_m >= self.x_max_m:
            raise ValueError("x_min_m must be smaller than x_max_m.")
        if self.y_min_m >= self.y_max_m:
            raise ValueError("y_min_m must be smaller than y_max_m.")
        if self.z_min_m >= self.z_max_m:
            raise ValueError("z_min_m must be smaller than z_max_m.")
        limits = (self.max_abs_bank_rad, self.max_abs_angle_of_attack_rad, self.max_abs_command_rad)
        if any(not np.isfinite(float(limit)) or float(limit) <= 0.0 for limit in limits):
            raise ValueError("Safety angular and command limits must be positive and finite.")

    def margins_m(self, state: FlightState) -> np.ndarray:
        self.validate()
        x, y, z = state.position_w_m
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

    def angle_of_attack_rad(
        self,
        state: FlightState,
        model: GliderModel | None = None,
        wind_model: object | None = None,
        wind_mode: str = "panel",
    ) -> float:
        if model is None:
            return float(np.arctan2(state.velocity_b_m_s[2], state.velocity_b_m_s[0]))
        return float(model.evaluate_aero(state, wind_model=wind_model, wind_mode=wind_mode).alpha_rad)

    def check_state(
        self,
        state: FlightState,
        model: GliderModel | None = None,
        wind_model: object | None = None,
        wind_mode: str = "panel",
    ) -> PrimitiveSafetyReport:
        self.validate()
        if not state.finite():
            return PrimitiveSafetyReport(
                ok=False,
                failure_reason="non_finite_state",
                min_margin_m=float("nan"),
                max_angle_of_attack_rad=float("nan"),
                max_command_abs_rad=float("nan"),
            )
        margin = float(np.min(self.margins_m(state)))
        alpha_abs = abs(self.angle_of_attack_rad(state, model=model, wind_model=wind_model, wind_mode=wind_mode))
        command_abs = float(np.max(np.abs(state.surfaces_rad)))
        reason: str | None = None
        x, y, z = state.position_w_m
        if z < self.z_min_m:
            reason = "floor"
        elif z > self.z_max_m:
            reason = "ceiling"
        elif x < self.x_min_m or x > self.x_max_m or y < self.y_min_m or y > self.y_max_m:
            reason = "wall"
        elif abs(float(state.euler_rad[0])) > float(self.max_abs_bank_rad):
            reason = "bank_limit"
        elif alpha_abs > float(self.max_abs_angle_of_attack_rad):
            reason = "stall_limit"
        elif command_abs > float(self.max_abs_command_rad):
            reason = "command_limit"
        return PrimitiveSafetyReport(
            ok=reason is None,
            failure_reason=reason,
            min_margin_m=margin,
            max_angle_of_attack_rad=alpha_abs,
            max_command_abs_rad=command_abs,
        )

    def check_rollout(
        self,
        states: list[FlightState] | tuple[FlightState, ...],
        commands_rad: np.ndarray,
        model: GliderModel | None = None,
        wind_model: object | None = None,
        wind_mode: str = "panel",
    ) -> PrimitiveSafetyReport:
        if not states:
            return PrimitiveSafetyReport(False, "empty_rollout", float("nan"), float("nan"), float("nan"))
        reports = [
            self.check_state(state, model=model, wind_model=wind_model, wind_mode=wind_mode)
            for state in states
        ]
        first_failure = next((report.failure_reason for report in reports if not report.ok), None)
        min_margin = float(np.nanmin([report.min_margin_m for report in reports]))
        max_alpha = float(np.nanmax([report.max_angle_of_attack_rad for report in reports]))
        commands = np.asarray(commands_rad, dtype=float).reshape(-1, 3)
        if commands.size:
            if not np.isfinite(commands).all():
                command_failure = "non_finite_command"
                max_command = float("nan")
            else:
                max_command = float(np.max(np.abs(commands)))
                command_failure = "command_limit" if max_command > float(self.max_abs_command_rad) else None
        else:
            max_command = 0.0
            command_failure = None
        failure = first_failure or command_failure
        return PrimitiveSafetyReport(
            ok=failure is None,
            failure_reason=failure,
            min_margin_m=min_margin,
            max_angle_of_attack_rad=max_alpha,
            max_command_abs_rad=max_command,
        )


@dataclass(frozen=True)
class PrimitiveGrammarSpec:
    """Small deterministic primitive grammar specification."""

    operating_point: OperatingPointSpec = OperatingPointSpec()
    bank_transition: BankTransitionSpec = BankTransitionSpec()
    pitch_pulse: PitchPulseSpec = PitchPulseSpec()
    dwell: DwellSpec = DwellSpec()
    recovery: RecoverySpec = RecoverySpec()
    entry_condition: PrimitiveEntryCondition = PrimitiveEntryCondition()
    safety_limits: PrimitiveSafetyLimits = PrimitiveSafetyLimits()
    hold_duration_s: float = 0.1
    controller_types: tuple[str, ...] = ("pd",)
    family: str = "bank_pitch_dwell_recovery"

    @classmethod
    def smoke(cls) -> "PrimitiveGrammarSpec":
        """Return the repository smoke-scale grammar."""

        return cls()

    def validate(self) -> None:
        self.operating_point.validate()
        self.bank_transition.validate()
        self.pitch_pulse.validate()
        self.dwell.validate()
        self.recovery.validate()
        self.safety_limits.validate()
        if not np.isfinite(float(self.hold_duration_s)) or float(self.hold_duration_s) <= 0.0:
            raise ValueError("hold_duration_s must be positive and finite.")
        if not self.controller_types:
            raise ValueError("At least one controller type is required.")
        for controller_type in self.controller_types:
            if controller_type not in {"pd", "lqr"}:
                raise ValueError("controller_types may contain only 'pd' or 'lqr'.")
