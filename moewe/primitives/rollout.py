"""Primitive rollout adapter over the existing simulator and controllers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from moewe.control.interface import CommandLimits, ControllerMetadata
from moewe.control.linearise import finite_difference_linearisation, to_euler_discrete
from moewe.control.lqr import solve_discrete_lqr
from moewe.control.pd import PDController, PDGains
from moewe.control.rollout import run_closed_loop
from moewe.sim.actuator import ActuatorConfig, ActuatorModel
from moewe.sim.glider_model import GliderModel, nominal_glider
from moewe.sim.integrator import IntegratorConfig
from moewe.sim.state import FlightState
from moewe.tasks.gate import GateTraversalTask
from moewe.tasks.metrics import GateTaskMetrics, specific_energy_j_kg

from .evidence import PrimitiveEvidence
from .generate import PrimitiveCandidate
from .reference import PrimitiveReference


@dataclass(frozen=True)
class PrimitiveRolloutConfig:
    """Smoke-rollout configuration for a structured primitive."""

    dt_s: float = 0.01
    wind_mode: str = "panel"
    max_duration_s: float | None = None
    command_limits: CommandLimits = CommandLimits()
    pd_gains: PDGains = PDGains()
    lqr_active_state_indices: tuple[int, ...] = (3, 4, 5, 9, 10, 11, 12, 13, 14)
    lqr_q_weights: tuple[float, ...] = (4.0, 6.0, 2.0, 0.4, 0.6, 0.3, 0.2, 0.2, 0.2)
    lqr_r_weights: tuple[float, float, float] = (0.1, 0.1, 0.1)
    scenario_id: str = "primitive_smoke"
    seed: int | None = 0


@dataclass(frozen=True)
class ReferencePDController:
    """Time-varying PD tracker for a primitive reference."""

    reference: PrimitiveReference
    gains: PDGains = PDGains()
    command_limits: CommandLimits = CommandLimits()
    metadata: ControllerMetadata = ControllerMetadata(
        controller_type="pd",
        description="time-varying primitive reference PD tracker",
        reference="primitive_reference",
    )

    @property
    def reference_state(self) -> FlightState:
        return self.reference.state_at(0.0)

    @property
    def reference_command_rad(self) -> np.ndarray:
        return self.reference.command_at(0.0)

    def command(self, time_s: float, state: FlightState) -> np.ndarray:
        ref_state = self.reference.state_at(time_s)
        ref_command = self.reference.command_at(time_s)
        controller = PDController(
            reference_state=ref_state,
            reference_command_rad=ref_command,
            gains=self.gains,
            command_limits=self.command_limits,
            metadata=self.metadata,
        )
        return controller.command(time_s, state)


@dataclass(frozen=True)
class ReferenceLQRController:
    """Time-invariant LQR stabiliser tracking a primitive reference."""

    reference: PrimitiveReference
    gain: np.ndarray
    active_state_indices: tuple[int, ...]
    command_limits: CommandLimits = CommandLimits()
    metadata: ControllerMetadata = ControllerMetadata(
        controller_type="lqr",
        description="time-invariant primitive-local LQR stabiliser",
        reference="finite_difference_local_linear_model",
    )

    def __post_init__(self) -> None:
        gain = np.asarray(self.gain, dtype=float)
        if gain.shape != (3, len(self.active_state_indices)):
            raise ValueError("Primitive LQR gain must have shape (3, active_state_count).")
        if not np.isfinite(gain).all():
            raise ValueError("Primitive LQR gain must be finite.")
        object.__setattr__(self, "gain", gain)

    @property
    def reference_state(self) -> FlightState:
        return self.reference.state_at(0.0)

    @property
    def reference_command_rad(self) -> np.ndarray:
        return self.reference.command_at(0.0)

    def command(self, time_s: float, state: FlightState) -> np.ndarray:
        ref_state = self.reference.state_at(time_s)
        ref_command = self.reference.command_at(time_s)
        error = state.as_vector() - ref_state.as_vector()
        error = error[np.asarray(self.active_state_indices, dtype=int)]
        return self.command_limits.clip(ref_command - self.gain @ error)


@dataclass(frozen=True)
class PrimitiveRolloutResult:
    """Closed-loop rollout output and primitive evidence."""

    primitive_id: str
    states: list[FlightState]
    commands_rad: np.ndarray
    metrics: GateTaskMetrics | None
    evidence: PrimitiveEvidence
    controller_failed: bool
    failure_reason: str | None


def build_primitive_controller(
    primitive: PrimitiveCandidate,
    config: PrimitiveRolloutConfig | None = None,
    model: GliderModel | None = None,
    actuator_config: ActuatorConfig | None = None,
    wind_model: object | None = None,
) -> ReferencePDController | ReferenceLQRController:
    """Build the supported local controller for a primitive."""

    cfg = PrimitiveRolloutConfig() if config is None else config
    if primitive.controller_type == "pd":
        return ReferencePDController(
            reference=primitive.reference,
            gains=cfg.pd_gains,
            command_limits=cfg.command_limits,
        )
    if primitive.controller_type == "lqr":
        state0 = primitive.reference.state_at(0.0)
        command0 = primitive.reference.command_at(0.0)
        linearisation = finite_difference_linearisation(
            state=state0,
            command_rad=command0,
            model=model,
            actuator_config=ActuatorConfig() if actuator_config is None else actuator_config,
            wind_model=wind_model,
            wind_mode=cfg.wind_mode,
        )
        discrete = to_euler_discrete(linearisation, cfg.dt_s)
        idx = np.asarray(cfg.lqr_active_state_indices, dtype=int)
        q_weights = np.asarray(cfg.lqr_q_weights, dtype=float)
        r_weights = np.asarray(cfg.lqr_r_weights, dtype=float)
        if q_weights.shape != (idx.size,):
            raise ValueError("lqr_q_weights must match lqr_active_state_indices.")
        if r_weights.shape != (3,):
            raise ValueError("lqr_r_weights must contain three command weights.")
        gain = solve_discrete_lqr(
            discrete.a[np.ix_(idx, idx)],
            discrete.b[idx, :],
            np.diag(q_weights),
            np.diag(r_weights),
        )
        return ReferenceLQRController(
            reference=primitive.reference,
            gain=gain,
            active_state_indices=tuple(int(value) for value in idx),
            command_limits=cfg.command_limits,
        )
    raise ValueError(f"Unsupported primitive controller_type: {primitive.controller_type}")


def _failure_result(
    primitive: PrimitiveCandidate,
    initial_state: FlightState,
    reason: str,
    config: PrimitiveRolloutConfig,
) -> PrimitiveRolloutResult:
    safety = primitive.safety_limits.check_state(initial_state)
    evidence = PrimitiveEvidence(
        primitive_id=primitive.primitive_id,
        family=primitive.family,
        controller_type=primitive.controller_type,
        rollout_success=False,
        min_safety_margin_m=safety.min_margin_m,
        terminal_specific_energy_change_j_kg=0.0,
        terminal_specific_energy_margin_j_kg=None,
        max_angle_of_attack_rad=safety.max_angle_of_attack_rad,
        max_command_abs_rad=0.0,
        gate_miss_distance_m=None,
        failure_reason=reason,
        scenario_id=config.scenario_id,
        seed=config.seed,
    )
    return PrimitiveRolloutResult(
        primitive_id=primitive.primitive_id,
        states=[initial_state],
        commands_rad=np.empty((0, 3)),
        metrics=None,
        evidence=evidence,
        controller_failed=True,
        failure_reason=reason,
    )


def _evidence_from_rollout(
    primitive: PrimitiveCandidate,
    states: list[FlightState],
    commands_rad: np.ndarray,
    metrics: GateTaskMetrics | None,
    config: PrimitiveRolloutConfig,
    model: GliderModel,
    wind_model: object | None,
    controller_failure_reason: str | None,
) -> PrimitiveEvidence:
    safety = primitive.safety_limits.check_rollout(
        states,
        commands_rad,
        model=model,
        wind_model=wind_model,
        wind_mode=config.wind_mode,
    )
    if metrics is None:
        min_margin = safety.min_margin_m
        terminal_energy_change = specific_energy_j_kg(states[-1]) - specific_energy_j_kg(states[0])
        terminal_energy_margin = None
        max_alpha = safety.max_angle_of_attack_rad
        gate_miss = None
        task_failure = None
        task_success = True
    else:
        min_margin = min(float(metrics.min_safety_margin_m), safety.min_margin_m)
        terminal_energy_change = specific_energy_j_kg(states[-1]) - specific_energy_j_kg(states[0])
        terminal_energy_margin = float(metrics.terminal_specific_energy_margin_j_kg)
        max_alpha = max(float(metrics.max_angle_of_attack_rad), safety.max_angle_of_attack_rad)
        gate_miss = float(metrics.gate_miss_distance_m)
        task_failure = None if metrics.success else str(metrics.failure_reason)
        task_success = bool(metrics.success)
    failure_reason = controller_failure_reason or safety.failure_reason or task_failure
    success = controller_failure_reason is None and safety.ok and task_success
    return PrimitiveEvidence(
        primitive_id=primitive.primitive_id,
        family=primitive.family,
        controller_type=primitive.controller_type,
        rollout_success=success,
        min_safety_margin_m=float(min_margin),
        terminal_specific_energy_change_j_kg=float(terminal_energy_change),
        terminal_specific_energy_margin_j_kg=terminal_energy_margin,
        max_angle_of_attack_rad=float(max_alpha),
        max_command_abs_rad=float(safety.max_command_abs_rad),
        gate_miss_distance_m=gate_miss,
        failure_reason=failure_reason,
        scenario_id=config.scenario_id,
        seed=config.seed,
        rollout_duration_s=float(max(len(states) - 1, 0) * config.dt_s),
    )


def rollout_primitive(
    primitive: PrimitiveCandidate,
    initial_state: FlightState | None = None,
    task: GateTraversalTask | None = None,
    model: GliderModel | None = None,
    actuator: ActuatorModel | None = None,
    wind_model: object | None = None,
    config: PrimitiveRolloutConfig | None = None,
) -> PrimitiveRolloutResult:
    """Run a short closed-loop primitive rollout and return explicit evidence."""

    cfg = PrimitiveRolloutConfig() if config is None else config
    if cfg.dt_s <= 0.0:
        raise ValueError("Primitive rollout dt_s must be positive.")
    state0 = primitive.reference.state_at(0.0) if initial_state is None else initial_state
    entry_failure = primitive.entry_condition.rejection_reason(state0)
    if entry_failure is not None:
        return _failure_result(primitive, state0, entry_failure, cfg)
    duration = primitive.reference.total_duration_s
    if cfg.max_duration_s is not None:
        duration = min(duration, float(cfg.max_duration_s))
    steps = max(1, int(math.ceil(duration / float(cfg.dt_s))))
    glider = nominal_glider() if model is None else model
    try:
        actuator_config = actuator.config if actuator is not None else ActuatorConfig()
        controller = build_primitive_controller(
            primitive,
            cfg,
            model=glider,
            actuator_config=actuator_config,
            wind_model=wind_model,
        )
    except ValueError as exc:
        return _failure_result(primitive, state0, str(exc), cfg)
    closed_loop = run_closed_loop(
        initial_state=state0,
        controller=controller,
        steps=steps,
        task=task,
        model=glider,
        actuator=actuator,
        wind_model=wind_model,
        config=IntegratorConfig(dt_s=cfg.dt_s, wind_mode=cfg.wind_mode),
    )
    evidence = _evidence_from_rollout(
        primitive=primitive,
        states=closed_loop.states,
        commands_rad=closed_loop.commands_rad,
        metrics=closed_loop.metrics,
        config=cfg,
        model=glider,
        wind_model=wind_model,
        controller_failure_reason=closed_loop.controller_failure_reason,
    )
    return PrimitiveRolloutResult(
        primitive_id=primitive.primitive_id,
        states=closed_loop.states,
        commands_rad=closed_loop.commands_rad,
        metrics=closed_loop.metrics,
        evidence=evidence,
        controller_failed=closed_loop.controller_failed,
        failure_reason=evidence.failure_reason,
    )
