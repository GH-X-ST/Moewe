"""Continuous nominal terminal planning and local feedback design."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, minimize

from control.flow import AffineFlow
from control.missions import Mission
from control.tube import FeedbackSegment
from models.aircraft import Aircraft
from models.state import as_state


class PlanningTimeout(RuntimeError):
    """Raised when candidate generation reaches its wall-clock deadline."""


@dataclass
class NominalPlanner:
    """Optimise continuous-valued hold commands and local tracking feedback."""

    aircraft: Aircraft
    flow: AffineFlow
    horizons: tuple[int, ...] = (50,)
    dt_s: float = 0.02
    control_knots: int = 8
    feedback_knots: int = 1
    max_iterations: int = 200
    finite_difference_step: float = 1.0e-5
    constraint_tolerance: float = 1.0e-7
    command_change_weight: float = 0.01
    lqr_state_weight: npt.ArrayLike = (1.0,) * 15
    lqr_control_weight: npt.ArrayLike = (1.0,) * 3
    _center_flow: np.ndarray = field(init=False, repr=False)
    _strip_flow: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._center_flow = np.asarray(
            self.flow.center_b_m_s,
            dtype=float,
        ).reshape(3)
        self._strip_flow = self.flow.strip_flow(self.aircraft.strip_table.r_b_m)

    def plan(
        self,
        initial_state: npt.ArrayLike,
        mission: Mission,
        horizons: int | tuple[int, ...] | None = None,
        initial_control: npt.ArrayLike | None = None,
        time_limit_s: float | None = None,
    ) -> tuple[FeedbackSegment, ...]:
        """Return the least-cost feasible nominal plan and tracking gains."""

        deadline = None if time_limit_s is None else monotonic() + time_limit_s
        state = as_state(initial_state)
        options = self.horizons if horizons is None else horizons
        if isinstance(options, int):
            options = (options,)
        control = (
            state[12:15]
            if initial_control is None
            else np.asarray(initial_control, dtype=float).reshape(3)
        )
        control = self.aircraft.clip_control(control)

        best: tuple[float, np.ndarray, np.ndarray] | None = None
        for horizon in options:
            candidate = self._optimise(
                state,
                control,
                mission,
                horizon,
                deadline,
            )
            if candidate is not None and (best is None or candidate[0] < best[0]):
                best = candidate
        if best is None:
            raise RuntimeError("nominal planner found no feasible horizon")

        _, states, controls = best
        _check_deadline(deadline)
        gains = self._feedback_gains(states, controls, deadline)
        return tuple(
            FeedbackSegment(states[index], controls[index], gains[index])
            for index in range(controls.shape[0])
        )

    def _optimise(
        self,
        state: np.ndarray,
        initial_control: np.ndarray,
        mission: Mission,
        horizon: int,
        deadline: float | None,
    ) -> tuple[float, np.ndarray, np.ndarray] | None:
        knot_count = min(self.control_knots, horizon)
        guess = np.tile(initial_control, knot_count)
        lower = np.tile(self.aircraft.control_lower_rad, knot_count)
        upper = np.tile(self.aircraft.control_upper_rad, knot_count)
        knot_index = np.minimum(
            np.arange(horizon) * knot_count // horizon,
            knot_count - 1,
        )
        cached_controls = np.empty(0)
        cached_states = np.empty((0, 15))

        def expand(flat_controls: np.ndarray) -> np.ndarray:
            return flat_controls.reshape(knot_count, 3)[knot_index]

        def states_from(flat_controls: np.ndarray) -> np.ndarray:
            nonlocal cached_controls, cached_states
            _check_deadline(deadline)
            if cached_controls.shape != flat_controls.shape or not np.array_equal(
                cached_controls, flat_controls
            ):
                cached_controls = flat_controls.copy()
                controls = expand(flat_controls)
                cached_states = self._rollout(state, controls)
            return cached_states

        def objective(flat_controls: np.ndarray) -> float:
            controls = expand(flat_controls)
            states = states_from(flat_controls)
            changes = np.diff(
                np.vstack((state[12:15], controls)),
                axis=0,
            )
            return float(
                sum(
                    mission.running_cost(states[index], controls[index])
                    for index in range(horizon)
                )
                + self.command_change_weight * np.sum(changes * changes)
            )

        def constraints(flat_controls: np.ndarray) -> np.ndarray:
            values = mission.nominal_constraints(states_from(flat_controls))
            return np.asarray(values, dtype=float).reshape(-1)

        result = minimize(
            objective,
            guess,
            method="SLSQP",
            bounds=Bounds(lower, upper),
            constraints=({"type": "ineq", "fun": constraints},),
            options={
                "ftol": 1.0e-8,
                "maxiter": self.max_iterations,
                "disp": False,
            },
        )
        _check_deadline(deadline)
        controls = expand(np.asarray(result.x, dtype=float))
        states = self._rollout(state, controls)
        _check_deadline(deadline)
        feasible = np.all(
            np.asarray(
                mission.nominal_constraints(states),
                dtype=float,
            )
            >= -self.constraint_tolerance
        )
        if not result.success or not feasible:
            return None
        return float(result.fun), states, controls

    def _rollout(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray,
    ) -> np.ndarray:
        states = np.empty((controls.shape[0] + 1, 15))
        states[0] = initial_state
        for index, control in enumerate(controls):
            states[index + 1] = self._step(states[index], control)
        return states

    def _step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        step = self.dt_s

        def derivative(value: np.ndarray) -> np.ndarray:
            return self.aircraft.derivative_local_flow(
                value,
                control,
                self._center_flow,
                self._strip_flow,
            )

        k1 = derivative(state)
        k2 = derivative(state + 0.5 * step * k1)
        return state + step * k2

    def _feedback_gains(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        deadline: float | None,
    ) -> tuple[np.ndarray, ...]:
        count = controls.shape[0]
        sample_indices = np.unique(
            np.linspace(
                0,
                count - 1,
                min(self.feedback_knots, count),
                dtype=int,
            )
        )
        sampled = {}
        for index in sample_indices:
            _check_deadline(deadline)
            sampled[index] = self._linearise(
                states[index],
                controls[index],
                deadline,
            )
        linearisation = []
        sample = sample_indices[0]
        for index in range(count):
            candidates = sample_indices[sample_indices <= index]
            if candidates.size:
                sample = int(candidates[-1])
            linearisation.append(sampled[sample])
        state_weight = np.diag(
            np.asarray(self.lqr_state_weight, dtype=float).reshape(15)
        )
        control_weight = np.diag(
            np.asarray(self.lqr_control_weight, dtype=float).reshape(3)
        )
        value = state_weight.copy()
        gains = [np.zeros((3, 15)) for _ in linearisation]
        for index in range(len(linearisation) - 1, -1, -1):
            _check_deadline(deadline)
            state_matrix, control_matrix = linearisation[index]
            hessian = control_weight + control_matrix.T @ value @ control_matrix
            gain = -np.linalg.solve(
                hessian,
                control_matrix.T @ value @ state_matrix,
            )
            closed_loop = state_matrix + control_matrix @ gain
            value = (
                state_weight
                + gain.T @ control_weight @ gain
                + closed_loop.T @ value @ closed_loop
            )
            value = 0.5 * (value + value.T)
            gains[index] = gain
        return tuple(gains)

    def _linearise(
        self,
        state: np.ndarray,
        control: np.ndarray,
        deadline: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        state_jacobian = np.empty((15, 15))
        control_jacobian = np.empty((15, 3))

        def derivative(
            state_value: np.ndarray,
            control_value: np.ndarray,
        ) -> np.ndarray:
            return self.aircraft.derivative_local_flow(
                state_value,
                control_value,
                self._center_flow,
                self._strip_flow,
            )

        for index in range(15):
            _check_deadline(deadline)
            step = self.finite_difference_step * max(1.0, abs(state[index]))
            offset = np.zeros(15)
            offset[index] = step
            state_jacobian[:, index] = (
                derivative(state + offset, control)
                - derivative(state - offset, control)
            ) / (2.0 * step)

        for index in range(3):
            _check_deadline(deadline)
            step = self.finite_difference_step * max(1.0, abs(control[index]))
            lower = max(
                self.aircraft.control_lower_rad[index],
                control[index] - step,
            )
            upper = min(
                self.aircraft.control_upper_rad[index],
                control[index] + step,
            )
            lower_control = control.copy()
            upper_control = control.copy()
            lower_control[index] = lower
            upper_control[index] = upper
            control_jacobian[:, index] = (
                derivative(state, upper_control) - derivative(state, lower_control)
            ) / (upper - lower)
        return (
            np.eye(15) + self.dt_s * state_jacobian,
            self.dt_s * control_jacobian,
        )


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and monotonic() >= deadline:
        raise PlanningTimeout("nominal planning deadline expired")
