"""Closed-loop rollout adapter for local controllers and gate tasks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moewe.sim.actuator import ActuatorConfig, ActuatorModel
from moewe.sim.glider_model import GliderModel, nominal_glider
from moewe.sim.integrator import IntegratorConfig, step_fixed
from moewe.sim.state import FlightState
from moewe.tasks.gate import GateTraversalTask
from moewe.tasks.metrics import GateTaskMetrics

from .interface import LocalController


@dataclass(frozen=True)
class ClosedLoopResult:
    """In-memory closed-loop rollout output."""

    states: list[FlightState]
    commands_rad: np.ndarray
    metrics: GateTaskMetrics | None
    controller_failed: bool
    controller_failure_reason: str | None = None


def run_closed_loop(
    initial_state: FlightState,
    controller: LocalController,
    steps: int,
    task: GateTraversalTask | None = None,
    model: GliderModel | None = None,
    actuator: ActuatorModel | None = None,
    wind_model: object | None = None,
    config: IntegratorConfig | None = None,
) -> ClosedLoopResult:
    """Run a controller through the simulator and optional task metrics."""

    if steps < 0:
        raise ValueError("Closed-loop steps must be non-negative.")
    glider = nominal_glider() if model is None else model
    cfg = IntegratorConfig() if config is None else config
    cfg.validate()
    actuator_model = ActuatorModel(ActuatorConfig()) if actuator is None else actuator
    states = [initial_state]
    commands: list[np.ndarray] = []
    state = initial_state
    controller_failed = False
    failure_reason: str | None = None
    for index in range(int(steps)):
        time_s = index * cfg.dt_s
        try:
            command = np.asarray(controller.command(time_s, state), dtype=float).reshape(3)
        except Exception as exc:  # pragma: no cover - defensive failure path.
            controller_failed = True
            failure_reason = f"controller_exception:{type(exc).__name__}"
            break
        if not np.isfinite(command).all():
            controller_failed = True
            failure_reason = "non_finite_command"
            break
        commands.append(command)
        state = step_fixed(
            state=state,
            command_rad=command,
            model=glider,
            actuator=actuator_model,
            wind_model=wind_model,
            config=cfg,
        )
        states.append(state)
        if not state.finite():
            controller_failed = True
            failure_reason = "non_finite_state"
            break
    metrics = None
    if task is not None:
        metrics = task.evaluate(
            states,
            dt_s=cfg.dt_s,
            model=glider,
            wind_model=wind_model,
            wind_mode=cfg.wind_mode,
        )
    command_history = np.asarray(commands, dtype=float).reshape(-1, 3) if commands else np.empty((0, 3))
    return ClosedLoopResult(
        states=states,
        commands_rad=command_history,
        metrics=metrics,
        controller_failed=controller_failed,
        controller_failure_reason=failure_reason,
    )
