"""Finite-difference linearisation for the Moewe simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moewe.sim.actuator import ActuatorConfig, ActuatorModel
from moewe.sim.glider_model import GliderModel, nominal_glider
from moewe.sim.rigid_body import rigid_body_derivative
from moewe.sim.state import FlightState, STATE_SIZE

INPUT_SIZE = 3


@dataclass(frozen=True)
class Linearisation:
    """Continuous or discrete local linear model around ``x_ref, u_ref``."""

    a: np.ndarray
    b: np.ndarray
    x_ref: np.ndarray
    u_ref: np.ndarray
    f_ref: np.ndarray
    mode: str
    dt_s: float | None = None
    state_step: float = 1e-5
    input_step: float = 1e-5

    def __post_init__(self) -> None:
        a = np.asarray(self.a, dtype=float)
        b = np.asarray(self.b, dtype=float)
        if a.shape != (STATE_SIZE, STATE_SIZE):
            raise ValueError("Linearisation A matrix must have shape (15, 15).")
        if b.shape != (STATE_SIZE, INPUT_SIZE):
            raise ValueError("Linearisation B matrix must have shape (15, 3).")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Linearisation matrices must be finite.")


@dataclass(frozen=True)
class LinearisationCheck:
    """One-step nonlinear versus linear prediction check."""

    nonlinear_next: np.ndarray
    linear_next: np.ndarray
    error_norm: float


def continuous_dynamics(
    state_vector: np.ndarray,
    command_rad: np.ndarray,
    model: GliderModel | None = None,
    actuator_config: ActuatorConfig | None = None,
    wind_model: object | None = None,
    wind_mode: str = "panel",
    rho_kg_m3: float = 1.225,
) -> np.ndarray:
    """Return the 15-state derivative with first-order actuator states.

    Fixed actuator delay is intentionally omitted from this continuous-time
    local model; delay remains a rollout-level actuator property.
    """

    state = FlightState.from_vector(state_vector)
    glider = nominal_glider() if model is None else model
    config = ActuatorConfig() if actuator_config is None else actuator_config
    config.validate()
    command = ActuatorModel(config).prepare_command(command_rad)
    tau = np.asarray(config.time_constant_s, dtype=float).reshape(3)
    surface_rate = np.zeros(3)
    dynamic = tau > 0.0
    surface_rate[dynamic] = (command[dynamic] - state.surfaces_rad[dynamic]) / tau[dynamic]
    loads = glider.evaluate_loads(
        state,
        wind_model=wind_model,
        wind_mode=wind_mode,
        rho_kg_m3=float(rho_kg_m3),
    )
    derivative = rigid_body_derivative(
        state=state,
        force_b_n=loads.force_b_n,
        moment_b_n_m=loads.moment_b_n_m,
        mass_kg=glider.mass_kg,
        inertia_b_kg_m2=glider.inertia_b_kg_m2,
        surface_rates_rad_s=surface_rate,
    )
    if not np.isfinite(derivative).all():
        raise FloatingPointError("Continuous dynamics produced non-finite values.")
    return derivative


def finite_difference_linearisation(
    state: FlightState,
    command_rad: np.ndarray,
    model: GliderModel | None = None,
    actuator_config: ActuatorConfig | None = None,
    wind_model: object | None = None,
    wind_mode: str = "panel",
    rho_kg_m3: float = 1.225,
    state_step: float = 1e-5,
    input_step: float = 1e-5,
) -> Linearisation:
    """Central finite-difference continuous-time linearisation."""

    if state_step <= 0.0 or input_step <= 0.0:
        raise ValueError("Finite-difference steps must be positive.")
    x_ref = state.as_vector()
    u_ref = np.asarray(command_rad, dtype=float).reshape(INPUT_SIZE)
    f_ref = continuous_dynamics(
        x_ref,
        u_ref,
        model=model,
        actuator_config=actuator_config,
        wind_model=wind_model,
        wind_mode=wind_mode,
        rho_kg_m3=rho_kg_m3,
    )
    a = np.zeros((STATE_SIZE, STATE_SIZE), dtype=float)
    b = np.zeros((STATE_SIZE, INPUT_SIZE), dtype=float)
    for index in range(STATE_SIZE):
        delta = np.zeros(STATE_SIZE)
        delta[index] = float(state_step)
        f_plus = continuous_dynamics(
            x_ref + delta,
            u_ref,
            model=model,
            actuator_config=actuator_config,
            wind_model=wind_model,
            wind_mode=wind_mode,
            rho_kg_m3=rho_kg_m3,
        )
        f_minus = continuous_dynamics(
            x_ref - delta,
            u_ref,
            model=model,
            actuator_config=actuator_config,
            wind_model=wind_model,
            wind_mode=wind_mode,
            rho_kg_m3=rho_kg_m3,
        )
        a[:, index] = (f_plus - f_minus) / (2.0 * float(state_step))
    for index in range(INPUT_SIZE):
        delta_u = np.zeros(INPUT_SIZE)
        delta_u[index] = float(input_step)
        f_plus = continuous_dynamics(
            x_ref,
            u_ref + delta_u,
            model=model,
            actuator_config=actuator_config,
            wind_model=wind_model,
            wind_mode=wind_mode,
            rho_kg_m3=rho_kg_m3,
        )
        f_minus = continuous_dynamics(
            x_ref,
            u_ref - delta_u,
            model=model,
            actuator_config=actuator_config,
            wind_model=wind_model,
            wind_mode=wind_mode,
            rho_kg_m3=rho_kg_m3,
        )
        b[:, index] = (f_plus - f_minus) / (2.0 * float(input_step))
    return Linearisation(
        a=a,
        b=b,
        x_ref=x_ref,
        u_ref=u_ref,
        f_ref=f_ref,
        mode="continuous",
        state_step=float(state_step),
        input_step=float(input_step),
    )


def to_euler_discrete(linearisation: Linearisation, dt_s: float) -> Linearisation:
    """Convert a continuous model to an explicit-Euler discrete model."""

    if dt_s <= 0.0:
        raise ValueError("Discrete linearisation dt_s must be positive.")
    if linearisation.mode != "continuous":
        raise ValueError("Only continuous linearisations can be Euler-discretised.")
    return Linearisation(
        a=np.eye(STATE_SIZE) + float(dt_s) * linearisation.a,
        b=float(dt_s) * linearisation.b,
        x_ref=linearisation.x_ref,
        u_ref=linearisation.u_ref,
        f_ref=linearisation.f_ref,
        mode="discrete_euler",
        dt_s=float(dt_s),
        state_step=linearisation.state_step,
        input_step=linearisation.input_step,
    )


def linearisation_step_check(
    linearisation: Linearisation,
    state_perturbation: np.ndarray,
    command_perturbation: np.ndarray,
    dt_s: float,
    model: GliderModel | None = None,
    actuator_config: ActuatorConfig | None = None,
    wind_model: object | None = None,
    wind_mode: str = "panel",
    rho_kg_m3: float = 1.225,
) -> LinearisationCheck:
    """Compare one explicit-Euler nonlinear step with the local linear model."""

    dx = np.asarray(state_perturbation, dtype=float).reshape(STATE_SIZE)
    du = np.asarray(command_perturbation, dtype=float).reshape(INPUT_SIZE)
    if dt_s <= 0.0:
        raise ValueError("Check dt_s must be positive.")
    x = linearisation.x_ref + dx
    u = linearisation.u_ref + du
    nonlinear_next = x + float(dt_s) * continuous_dynamics(
        x,
        u,
        model=model,
        actuator_config=actuator_config,
        wind_model=wind_model,
        wind_mode=wind_mode,
        rho_kg_m3=rho_kg_m3,
    )
    linear_next = (
        linearisation.x_ref
        + float(dt_s) * linearisation.f_ref
        + dx
        + float(dt_s) * (linearisation.a @ dx + linearisation.b @ du)
    )
    return LinearisationCheck(
        nonlinear_next=nonlinear_next,
        linear_next=linear_next,
        error_norm=float(np.linalg.norm(nonlinear_next - linear_next)),
    )
