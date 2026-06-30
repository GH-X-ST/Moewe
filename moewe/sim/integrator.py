"""Fixed-step integration for simulator smoke tests and future rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .actuator import ActuatorConfig, ActuatorModel
from .glider_model import GliderModel, nominal_glider
from .rigid_body import rigid_body_derivative
from .state import FlightState


@dataclass(frozen=True)
class IntegratorConfig:
    dt_s: float = 0.01
    wind_mode: str = "panel"
    rho_kg_m3: float = 1.225

    def validate(self) -> None:
        if self.dt_s <= 0.0:
            raise ValueError("Integrator dt_s must be positive.")
        if self.wind_mode not in {"cg", "panel"}:
            raise ValueError("Integrator wind_mode must be 'cg' or 'panel'.")


CommandSource = np.ndarray | Callable[[float, FlightState], np.ndarray]


def _command_at(command: CommandSource, time_s: float, state: FlightState) -> np.ndarray:
    if callable(command):
        return np.asarray(command(float(time_s), state), dtype=float).reshape(3)
    return np.asarray(command, dtype=float).reshape(3)


def step_fixed(
    state: FlightState,
    command_rad: np.ndarray,
    model: GliderModel | None = None,
    actuator: ActuatorModel | None = None,
    wind_model: object | None = None,
    config: IntegratorConfig | None = None,
) -> FlightState:
    """Advance state by one fixed Euler step with explicit actuator stepping."""

    cfg = IntegratorConfig() if config is None else config
    cfg.validate()
    glider = nominal_glider() if model is None else model
    actuator_model = ActuatorModel(ActuatorConfig()) if actuator is None else actuator
    loads = glider.evaluate_loads(
        state,
        wind_model=wind_model,
        wind_mode=cfg.wind_mode,
        rho_kg_m3=cfg.rho_kg_m3,
    )
    derivative = rigid_body_derivative(
        state=state,
        force_b_n=loads.force_b_n,
        moment_b_n_m=loads.moment_b_n_m,
        mass_kg=glider.mass_kg,
        inertia_b_kg_m2=glider.inertia_b_kg_m2,
    )
    next_vector = state.as_vector() + cfg.dt_s * derivative
    next_surfaces = actuator_model.step(state.surfaces_rad, command_rad, cfg.dt_s)
    next_vector[12:15] = next_surfaces
    return FlightState.from_vector(next_vector)


def simulate_fixed_step(
    initial_state: FlightState,
    command: CommandSource,
    steps: int,
    model: GliderModel | None = None,
    actuator: ActuatorModel | None = None,
    wind_model: object | None = None,
    config: IntegratorConfig | None = None,
) -> list[FlightState]:
    """Run a deterministic fixed-step rollout and return every state."""

    if steps < 0:
        raise ValueError("steps must be non-negative.")
    cfg = IntegratorConfig() if config is None else config
    cfg.validate()
    glider = nominal_glider() if model is None else model
    actuator_model = ActuatorModel(ActuatorConfig()) if actuator is None else actuator
    states = [initial_state]
    state = initial_state
    for index in range(int(steps)):
        time_s = index * cfg.dt_s
        cmd = _command_at(command, time_s, state)
        state = step_fixed(
            state=state,
            command_rad=cmd,
            model=glider,
            actuator=actuator_model,
            wind_model=wind_model,
            config=cfg,
        )
        states.append(state)
    return states
