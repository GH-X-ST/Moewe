"""Discrete local LQR controller for linearised Moewe dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moewe.sim.state import FlightState

from .interface import CommandLimits, ControllerMetadata
from .linearise import Linearisation, to_euler_discrete


def solve_discrete_lqr(
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    max_iterations: int = 500,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Solve a discrete LQR gain by Riccati fixed-point iteration."""

    a_m = np.asarray(a, dtype=float)
    b_m = np.asarray(b, dtype=float)
    q_m = np.asarray(q, dtype=float)
    r_m = np.asarray(r, dtype=float)
    if a_m.ndim != 2 or a_m.shape[0] != a_m.shape[1]:
        raise ValueError("A must be square for LQR.")
    if b_m.ndim != 2 or b_m.shape[0] != a_m.shape[0]:
        raise ValueError("B row count must match A dimension.")
    if q_m.shape != a_m.shape:
        raise ValueError("Q must match A shape.")
    if r_m.shape != (b_m.shape[1], b_m.shape[1]):
        raise ValueError("R must match input dimension.")
    if not np.isfinite(a_m).all() or not np.isfinite(b_m).all():
        raise ValueError("LQR dynamics matrices must be finite.")

    p = q_m.copy()
    for _ in range(int(max_iterations)):
        s = r_m + b_m.T @ p @ b_m
        try:
            gain = np.linalg.solve(s, b_m.T @ p @ a_m)
        except np.linalg.LinAlgError as exc:
            raise ValueError("LQR gain solve is ill-conditioned.") from exc
        p_next = a_m.T @ p @ a_m - a_m.T @ p @ b_m @ gain + q_m
        if not np.isfinite(p_next).all():
            raise ValueError("Riccati iteration produced non-finite values.")
        if np.linalg.norm(p_next - p, ord="fro") < float(tolerance):
            return gain
        p = p_next
    raise ValueError("Riccati iteration did not converge.")


@dataclass(frozen=True)
class LQRController:
    """Local state-feedback controller around a reference state and command."""

    gain: np.ndarray
    reference_state: FlightState
    reference_command_rad: np.ndarray
    command_limits: CommandLimits = CommandLimits()
    active_state_indices: tuple[int, ...] | None = None
    metadata: ControllerMetadata = ControllerMetadata(
        controller_type="lqr",
        description="discrete local LQR around a finite-difference linear model",
    )

    def __post_init__(self) -> None:
        gain = np.asarray(self.gain, dtype=float)
        reference_command = np.asarray(self.reference_command_rad, dtype=float).reshape(3)
        active_count = 15 if self.active_state_indices is None else len(self.active_state_indices)
        if gain.shape != (3, active_count):
            raise ValueError("LQR gain must have shape (3, active_state_count).")
        if not np.isfinite(gain).all():
            raise ValueError("LQR gain must be finite.")
        object.__setattr__(self, "gain", gain)
        object.__setattr__(self, "reference_command_rad", reference_command)

    def command(self, time_s: float, state: FlightState) -> np.ndarray:
        del time_s
        error = state.as_vector() - self.reference_state.as_vector()
        if self.active_state_indices is not None:
            error = error[np.asarray(self.active_state_indices, dtype=int)]
        command = self.reference_command_rad - self.gain @ error
        return self.command_limits.clip(command)


def build_lqr_controller(
    linearisation: Linearisation,
    q: np.ndarray,
    r: np.ndarray,
    reference_state: FlightState,
    command_limits: CommandLimits | None = None,
    active_state_indices: tuple[int, ...] | None = None,
    dt_s: float | None = None,
) -> LQRController:
    """Build an LQR controller from a continuous or discrete linearisation."""

    discrete = (
        linearisation
        if linearisation.mode.startswith("discrete")
        else to_euler_discrete(linearisation, 0.01 if dt_s is None else float(dt_s))
    )
    if active_state_indices is None:
        a = discrete.a
        b = discrete.b
    else:
        idx = np.asarray(active_state_indices, dtype=int)
        a = discrete.a[np.ix_(idx, idx)]
        b = discrete.b[idx, :]
    gain = solve_discrete_lqr(a, b, q, r)
    return LQRController(
        gain=gain,
        reference_state=reference_state,
        reference_command_rad=linearisation.u_ref,
        command_limits=CommandLimits() if command_limits is None else command_limits,
        active_state_indices=active_state_indices,
    )
