"""Deterministic reference-tracking baseline for gate tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, pi

import numpy as np

from moewe.control import CommandLimits, ControllerMetadata, PDController, PDGains, run_closed_loop
from moewe.sim.actuator import ActuatorModel
from moewe.sim.glider_model import GliderModel
from moewe.sim.integrator import IntegratorConfig
from moewe.sim.state import FlightState
from moewe.tasks import FailureReason, GateTraversalTask

Vector3Like = np.ndarray | tuple[float, float, float] | list[float]

BASELINE_NAME = "reference_tracking_pd"


def _vector3(value: Vector3Like, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(3)
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain finite values.")
    return vector


def _finite_scalar(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    return scalar


def _wrap_pi(angle_rad: float) -> float:
    return float((float(angle_rad) + pi) % (2.0 * pi) - pi)


def _list3(vector: Vector3Like) -> list[float]:
    return [float(value) for value in _vector3(vector, "vector")]


@dataclass(frozen=True)
class ReferenceTrackingConfig:
    """Configuration for a transparent reference-tracking baseline rollout."""

    dt_s: float = 0.01
    horizon_s: float = 0.30
    target_speed_m_s: float = 7.0
    heading_to_roll_gain: float = 0.6
    altitude_to_pitch_gain: float = 0.25
    max_reference_roll_rad: float = 0.45
    max_reference_pitch_rad: float = 0.25
    command_limits: CommandLimits = CommandLimits()
    gains: PDGains = PDGains()
    wind_mode: str = "panel"

    def __post_init__(self) -> None:
        positive_fields = ("dt_s", "horizon_s", "target_speed_m_s")
        finite_fields = (
            "heading_to_roll_gain",
            "altitude_to_pitch_gain",
            "max_reference_roll_rad",
            "max_reference_pitch_rad",
        )
        for name in positive_fields + finite_fields:
            _finite_scalar(getattr(self, name), name)
        for name in positive_fields:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        for name in ("heading_to_roll_gain", "altitude_to_pitch_gain"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if self.max_reference_roll_rad < 0.0:
            raise ValueError("max_reference_roll_rad must be non-negative.")
        if self.max_reference_pitch_rad < 0.0:
            raise ValueError("max_reference_pitch_rad must be non-negative.")
        if not np.isfinite(self.command_limits.lower).all() or not np.isfinite(self.command_limits.upper).all():
            raise ValueError("command_limits must be finite.")
        gain_names = (
            "roll_p",
            "roll_d",
            "pitch_p",
            "pitch_d",
            "yaw_p",
            "yaw_d",
            "speed_p",
            "altitude_p",
            "vertical_speed_p",
        )
        for name in gain_names:
            _finite_scalar(getattr(self.gains, name), f"gains.{name}")
        if self.wind_mode not in {"cg", "panel"}:
            raise ValueError("wind_mode must be 'cg' or 'panel'.")

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serialisable configuration record."""

        gain_names = (
            "roll_p",
            "roll_d",
            "pitch_p",
            "pitch_d",
            "yaw_p",
            "yaw_d",
            "speed_p",
            "altitude_p",
            "vertical_speed_p",
        )
        return {
            "dt_s": float(self.dt_s),
            "horizon_s": float(self.horizon_s),
            "target_speed_m_s": float(self.target_speed_m_s),
            "heading_to_roll_gain": float(self.heading_to_roll_gain),
            "altitude_to_pitch_gain": float(self.altitude_to_pitch_gain),
            "max_reference_roll_rad": float(self.max_reference_roll_rad),
            "max_reference_pitch_rad": float(self.max_reference_pitch_rad),
            "command_limits": {
                "lower_rad": _list3(self.command_limits.lower),
                "upper_rad": _list3(self.command_limits.upper),
            },
            "gains": {name: float(getattr(self.gains, name)) for name in gain_names},
            "wind_mode": self.wind_mode,
        }


@dataclass(frozen=True)
class ReferenceTrackingController:
    """Gate-target reference generator using the local controller interface."""

    target_position_w_m: Vector3Like
    config: ReferenceTrackingConfig = ReferenceTrackingConfig()
    metadata: ControllerMetadata = ControllerMetadata(
        controller_type=BASELINE_NAME,
        description="transparent reference-tracking baseline for gate tasks",
        reference="gate_or_target_position",
    )
    reference_state: FlightState = field(init=False)
    reference_command_rad: np.ndarray = field(init=False)
    command_limits: CommandLimits = field(init=False)

    def __post_init__(self) -> None:
        target = _vector3(self.target_position_w_m, "target_position_w_m")
        object.__setattr__(self, "target_position_w_m", target)
        object.__setattr__(self, "command_limits", self.config.command_limits)
        object.__setattr__(self, "reference_command_rad", np.zeros(3))
        object.__setattr__(
            self,
            "reference_state",
            FlightState(
                position_w_m=target,
                euler_rad=np.zeros(3),
                velocity_b_m_s=np.array([self.config.target_speed_m_s, 0.0, 0.0]),
                rates_b_rad_s=np.zeros(3),
                surfaces_rad=np.zeros(3),
            ),
        )

    def reference_for_state(self, state: FlightState) -> FlightState:
        """Build the transient reference state for the current vehicle state."""

        if not state.finite():
            raise ValueError("state must be finite.")
        target_offset = self.target_position_w_m - state.position_w_m
        horizontal_norm = float(np.linalg.norm(target_offset[:2]))
        if horizontal_norm <= 1e-12:
            heading_error = 0.0
        else:
            target_yaw = float(np.arctan2(target_offset[1], target_offset[0]))
            heading_error = _wrap_pi(target_yaw - float(state.euler_rad[2]))
        desired_roll = float(
            np.clip(
                self.config.heading_to_roll_gain * heading_error,
                -self.config.max_reference_roll_rad,
                self.config.max_reference_roll_rad,
            )
        )
        altitude_error = float(self.target_position_w_m[2] - state.position_w_m[2])
        desired_pitch = float(
            np.clip(
                self.config.altitude_to_pitch_gain * altitude_error,
                -self.config.max_reference_pitch_rad,
                self.config.max_reference_pitch_rad,
            )
        )
        reference_position = state.position_w_m.copy()
        reference_position[2] = self.target_position_w_m[2]
        return FlightState(
            position_w_m=reference_position,
            euler_rad=np.array([desired_roll, desired_pitch, state.euler_rad[2] + heading_error]),
            velocity_b_m_s=np.array([self.config.target_speed_m_s, 0.0, 0.0]),
            rates_b_rad_s=np.zeros(3),
            surfaces_rad=np.zeros(3),
        )

    def command(self, time_s: float, state: FlightState) -> np.ndarray:
        """Return bounded aileron, elevator, rudder commands in radians."""

        reference = self.reference_for_state(state)
        controller = PDController(
            reference_state=reference,
            reference_command_rad=self.reference_command_rad,
            gains=self.config.gains,
            command_limits=self.command_limits,
            metadata=self.metadata,
        )
        command = controller.command(time_s, state)
        if not np.isfinite(command).all():
            raise ValueError("reference tracking command must be finite.")
        return self.command_limits.clip(command)


@dataclass(frozen=True)
class ReferenceTrackingRolloutRecord:
    """In-memory reference-tracking rollout summary."""

    baseline_name: str
    target_position_w_m: tuple[float, float, float]
    initial_position_w_m: tuple[float, float, float]
    terminal_position_w_m: tuple[float, float, float]
    state_count: int
    command_count: int
    controller_failed: bool
    controller_failure_reason: str | None
    gate_success: bool | None = None
    gate_crossed: bool | None = None
    gate_miss_distance_m: float | None = None
    min_safety_margin_m: float | None = None
    terminal_specific_energy_margin_j_kg: float | None = None
    max_angle_of_attack_rad: float | None = None
    failure_reason: str | None = None

    def to_record(self) -> dict[str, object]:
        """Return a plain JSON-serialisable rollout record."""

        return {
            "baseline_name": self.baseline_name,
            "target_position_w_m": list(self.target_position_w_m),
            "initial_position_w_m": list(self.initial_position_w_m),
            "terminal_position_w_m": list(self.terminal_position_w_m),
            "state_count": int(self.state_count),
            "command_count": int(self.command_count),
            "controller_failed": bool(self.controller_failed),
            "controller_failure_reason": self.controller_failure_reason,
            "gate_success": self.gate_success,
            "gate_crossed": self.gate_crossed,
            "gate_miss_distance_m": self.gate_miss_distance_m,
            "min_safety_margin_m": self.min_safety_margin_m,
            "terminal_specific_energy_margin_j_kg": self.terminal_specific_energy_margin_j_kg,
            "max_angle_of_attack_rad": self.max_angle_of_attack_rad,
            "failure_reason": self.failure_reason,
        }


def build_gate_tracking_target(
    task: GateTraversalTask | None = None,
    target_position_w_m: Vector3Like | None = None,
) -> np.ndarray:
    """Return an explicit target position, or the supplied gate centre."""

    if target_position_w_m is not None:
        return _vector3(target_position_w_m, "target_position_w_m")
    if task is None:
        raise ValueError("Provide either task or target_position_w_m.")
    return _vector3(task.gate.centre_w_m, "task.gate.centre_w_m")


def _tuple3(vector: Vector3Like) -> tuple[float, float, float]:
    values = _vector3(vector, "vector")
    return (float(values[0]), float(values[1]), float(values[2]))


def _metric_failure_reason(metrics_failure: FailureReason | str | None) -> str | None:
    if metrics_failure is None or metrics_failure == FailureReason.NONE:
        return None
    if isinstance(metrics_failure, FailureReason):
        return metrics_failure.value
    return str(metrics_failure)


def _optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def run_reference_tracking_rollout(
    initial_state: FlightState,
    config: ReferenceTrackingConfig | None = None,
    task: GateTraversalTask | None = None,
    target_position_w_m: Vector3Like | None = None,
    model: GliderModel | None = None,
    actuator: ActuatorModel | None = None,
    wind_model: object | None = None,
) -> ReferenceTrackingRolloutRecord:
    """Run an in-memory reference-tracking rollout through the existing simulator."""

    if not initial_state.finite():
        raise ValueError("initial_state must be finite.")
    cfg = ReferenceTrackingConfig() if config is None else config
    target = build_gate_tracking_target(task=task, target_position_w_m=target_position_w_m)
    controller = ReferenceTrackingController(target_position_w_m=target, config=cfg)
    steps = max(1, int(ceil(float(cfg.horizon_s) / float(cfg.dt_s))))
    result = run_closed_loop(
        initial_state=initial_state,
        controller=controller,
        steps=steps,
        task=task,
        model=model,
        actuator=actuator,
        wind_model=wind_model,
        config=IntegratorConfig(dt_s=cfg.dt_s, wind_mode=cfg.wind_mode),
    )
    terminal_state = result.states[-1]
    metrics = result.metrics
    return ReferenceTrackingRolloutRecord(
        baseline_name=BASELINE_NAME,
        target_position_w_m=_tuple3(target),
        initial_position_w_m=_tuple3(initial_state.position_w_m),
        terminal_position_w_m=_tuple3(terminal_state.position_w_m),
        state_count=len(result.states),
        command_count=int(result.commands_rad.shape[0]),
        controller_failed=bool(result.controller_failed),
        controller_failure_reason=result.controller_failure_reason,
        gate_success=None if metrics is None else bool(metrics.success),
        gate_crossed=None if metrics is None else bool(metrics.gate_crossed),
        gate_miss_distance_m=None if metrics is None else float(metrics.gate_miss_distance_m),
        min_safety_margin_m=None if metrics is None else float(metrics.min_safety_margin_m),
        terminal_specific_energy_margin_j_kg=None
        if metrics is None
        else float(metrics.terminal_specific_energy_margin_j_kg),
        max_angle_of_attack_rad=None if metrics is None else _optional_float(metrics.max_angle_of_attack_rad),
        failure_reason=None if metrics is None else _metric_failure_reason(metrics.failure_reason),
    )


__all__ = [
    "ReferenceTrackingConfig",
    "ReferenceTrackingController",
    "ReferenceTrackingRolloutRecord",
    "build_gate_tracking_target",
    "run_reference_tracking_rollout",
]
