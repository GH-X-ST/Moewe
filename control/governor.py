"""Deterministic active-set solver for the three-input reference governor."""

from __future__ import annotations

from math import sqrt
from time import perf_counter

import numpy as np
import numpy.typing as npt

from control.capture import POLYTOPE_TOLERANCE, ActiveSets, CaptureCertificate
from control.predictor import FastPredictor, GeneratedAircraft
from control.uncertainty import NEXT_UPDATE_STAGE, PREDICTION_STAGES


TRACKING_WEIGHT = 1.0
SMOOTHING_WEIGHT = 0.1

_RANK_TOLERANCE = 64.0 * np.finfo(float).eps
_INFEASIBLE = -1
_TIMED_OUT = 0
_SOLVED = 1


class _StateBelief:
    def __init__(self, estimation_abs: np.ndarray) -> None:
        self.center = np.empty(15)
        self.generators = np.diag(estimation_abs)


class GovernorSolver:
    """Solve fixed-weight, bound-constrained quadratic programs in three inputs."""

    def __init__(
        self,
        reference_lower: npt.ArrayLike,
        reference_upper: npt.ArrayLike,
        max_constraints: int,
    ) -> None:
        lower = np.asarray(reference_lower, dtype=float).reshape(3)
        upper = np.asarray(reference_upper, dtype=float).reshape(3)
        if np.any(upper <= lower):
            raise ValueError("reference upper bounds must exceed lower bounds")
        if max_constraints < 0:
            raise ValueError("maximum constraint count must be nonnegative")

        self.max_constraints = max_constraints
        self._center = 0.5 * (lower + upper)
        self._scale = 0.5 * (upper - lower)
        self._hessian = TRACKING_WEIGHT + SMOOTHING_WEIGHT
        self._inverse_hessian = 1.0 / self._hessian

        row_count = max_constraints + 6
        self._a = np.empty((row_count, 3))
        self._b = np.empty(row_count)
        self._a[:6] = 0.0
        for axis in range(3):
            self._a[2 * axis, axis] = 1.0
            self._a[2 * axis + 1, axis] = -1.0
        self._b[:6] = 1.0 - POLYTOPE_TOLERANCE
        self._nominal = np.empty(3)
        self._previous = np.empty(3)
        self._unconstrained = np.empty(3)
        self._candidate = np.empty(3)
        self._best = np.empty(3)
        self._best_cost = 0.0
        self._has_best = False
        self._deadline: float | None = None
        self._timed_out = False

    def solve_into(
        self,
        nominal: npt.ArrayLike,
        previous: npt.ArrayLike,
        a: np.ndarray,
        b: np.ndarray,
        result: np.ndarray,
        active_sets: ActiveSets,
        deadline: float | None = None,
    ) -> int:
        """Write the optimal reference and return a fixed runtime status."""

        inequalities = a
        limits = b

        self._deadline = deadline
        self._timed_out = False
        if _deadline_reached(deadline):
            return _TIMED_OUT
        row_count = inequalities.shape[0] + 6
        self._prepare_constraints(inequalities, limits)
        if self._timed_out:
            return _TIMED_OUT
        self._normalise_reference(nominal, self._nominal)
        self._normalise_reference(previous, self._previous)
        np.multiply(
            self._previous,
            SMOOTHING_WEIGHT,
            out=self._candidate,
        )
        self._candidate += self._nominal
        np.multiply(
            self._candidate,
            self._inverse_hessian,
            out=self._unconstrained,
        )

        if self._feasible(self._unconstrained, row_count):
            if _deadline_reached(deadline):
                return _TIMED_OUT
            self._result_into(self._unconstrained, result)
            return _SOLVED

        self._has_best = False
        for active in active_sets.one:
            if _deadline_reached(deadline):
                return _TIMED_OUT
            self._solve_one(int(active[0]), row_count)
        for active in active_sets.two:
            if _deadline_reached(deadline):
                return _TIMED_OUT
            self._solve_two(int(active[0]), int(active[1]), row_count)
        for vertex in active_sets.vertices:
            if _deadline_reached(deadline):
                return _TIMED_OUT
            self._candidate[:] = vertex
            self._consider_feasible()

        if _deadline_reached(deadline):
            return _TIMED_OUT
        if not self._has_best:
            return _INFEASIBLE
        self._result_into(self._best, result)
        return _SOLVED

    def _prepare_constraints(
        self,
        inequalities: np.ndarray,
        limits: np.ndarray,
    ) -> None:
        for source in range(inequalities.shape[0]):
            if source % 16 == 0 and _deadline_reached(self._deadline):
                self._timed_out = True
                return
            target = source + 6
            a0 = inequalities[source, 0]
            a1 = inequalities[source, 1]
            a2 = inequalities[source, 2]
            c0 = a0 * self._scale[0]
            c1 = a1 * self._scale[1]
            c2 = a2 * self._scale[2]
            limit = limits[source] - (
                a0 * self._center[0] + a1 * self._center[1] + a2 * self._center[2]
            )
            norm = sqrt(c0 * c0 + c1 * c1 + c2 * c2)
            if norm > 0.0:
                inverse_norm = 1.0 / norm
                c0 *= inverse_norm
                c1 *= inverse_norm
                c2 *= inverse_norm
                limit = limit * inverse_norm - POLYTOPE_TOLERANCE
            self._a[target, 0] = c0
            self._a[target, 1] = c1
            self._a[target, 2] = c2
            self._b[target] = limit

    def _normalise_reference(
        self,
        reference: npt.ArrayLike,
        result: np.ndarray,
    ) -> None:
        np.subtract(reference, self._center, out=result)
        result /= self._scale

    def _solve_one(self, first: int, row_count: int) -> None:
        a0 = self._a[first]
        # The scalar normalized Hessian cancels from the projected KKT system.
        m00 = float(a0 @ a0)
        if m00 <= _RANK_TOLERANCE:
            return
        rhs0 = float(a0 @ self._unconstrained - self._b[first])
        multiplier0 = rhs0 / m00
        if multiplier0 < 0.0:
            return
        for axis in range(3):
            self._candidate[axis] = self._unconstrained[axis] - a0[axis] * multiplier0
        self._consider(row_count)

    def _solve_two(self, first: int, second: int, row_count: int) -> None:
        a0 = self._a[first]
        a1 = self._a[second]
        m00 = float(a0 @ a0)
        m01 = float(a0 @ a1)
        m11 = float(a1 @ a1)
        determinant = m00 * m11 - m01 * m01
        if determinant <= _RANK_TOLERANCE * m00 * m11:
            return

        rhs0 = float(a0 @ self._unconstrained - self._b[first])
        rhs1 = float(a1 @ self._unconstrained - self._b[second])
        multiplier0 = (m11 * rhs0 - m01 * rhs1) / determinant
        multiplier1 = (m00 * rhs1 - m01 * rhs0) / determinant
        if multiplier0 < 0.0 or multiplier1 < 0.0:
            return
        for axis in range(3):
            self._candidate[axis] = self._unconstrained[axis] - (
                a0[axis] * multiplier0 + a1[axis] * multiplier1
            )
        self._consider(row_count)

    def _consider(self, row_count: int) -> None:
        if not self._feasible(self._candidate, row_count):
            return
        self._consider_feasible()

    def _consider_feasible(self) -> None:
        difference0 = self._candidate[0] - self._unconstrained[0]
        difference1 = self._candidate[1] - self._unconstrained[1]
        difference2 = self._candidate[2] - self._unconstrained[2]
        cost = (
            0.5
            * self._hessian
            * (
                difference0 * difference0
                + difference1 * difference1
                + difference2 * difference2
            )
        )
        if (
            not self._has_best
            or cost < self._best_cost
            or (cost == self._best_cost and self._lexicographically_first())
        ):
            self._best[:] = self._candidate
            self._best_cost = cost
            self._has_best = True

    def _feasible(self, candidate: np.ndarray, row_count: int) -> bool:
        for row in range(row_count):
            if row % 16 == 0 and _deadline_reached(self._deadline):
                self._timed_out = True
                return False
            if self._a[row] @ candidate - self._b[row] > POLYTOPE_TOLERANCE:
                return False
        return True

    def _lexicographically_first(self) -> bool:
        for axis in range(3):
            if self._candidate[axis] < self._best[axis]:
                return True
            if self._candidate[axis] > self._best[axis]:
                return False
        return False

    def _result_into(self, normalised: np.ndarray, result: np.ndarray) -> None:
        np.multiply(normalised, self._scale, out=result)
        result += self._center


class JointFlowCaptureGovernor:
    """Run the certified governor at 100 ms and inner feedback at 20 ms."""

    def __init__(self, generated: GeneratedAircraft) -> None:
        if not isinstance(generated, GeneratedAircraft):
            raise TypeError("capture control requires an oracle-verified aircraft core")
        self.status = "inactive"
        self.current_reference = np.empty(3)
        self.gain_index = -1
        self.issued_queue = np.empty((generated.bounds.queue_length, 3))
        self._issued_flat = self.issued_queue.reshape(-1)
        self.latest_belief = _StateBelief(generated.bounds.state_estimation_abs)
        self._predictor = FastPredictor(generated)
        self._stage = NEXT_UPDATE_STAGE
        self._terminal_cell = False
        self._measured_segment = np.empty((PREDICTION_STAGES + 1, 15))
        self._measured_count = 0
        self._augmented_center = np.empty(15 + 3 * generated.bounds.queue_length)
        self._augmented_normalized = np.empty(self._augmented_center.size)
        self._state_lower = np.empty(15)
        self._state_upper = np.empty(15)
        self._prediction_lower = np.empty(15)
        self._prediction_upper = np.empty(15)
        self._nominal_state = np.empty(15)
        self._state_error = np.empty(15)
        self._nominal_command = np.empty(3)
        self._feedback_command = np.empty(3)
        self._command = np.empty(3)
        self._issued_lower = (
            np.asarray(generated.aircraft.control_lower_rad, dtype=float)
            + generated.bounds.command_error_abs_rad
        )
        self._issued_upper = (
            np.asarray(generated.aircraft.control_upper_rad, dtype=float)
            - generated.bounds.command_error_abs_rad
        )

    def activate(
        self,
        certificate: CaptureCertificate,
        state: npt.ArrayLike,
        reference: npt.ArrayLike,
        issued_queue: npt.ArrayLike,
    ) -> None:
        """Activate one compiled mission from its measured command history."""

        if certificate.generated is not self._predictor.generated:
            raise ValueError("capture certificate belongs to another aircraft core")
        measurement = np.asarray(state, dtype=float).reshape(15)
        self.mission = certificate.mission
        self.certificate = certificate
        self.current_reference[:] = np.asarray(reference, dtype=float).reshape(3)
        self.issued_queue[:] = np.asarray(issued_queue, dtype=float).reshape(
            self.issued_queue.shape
        )
        self.latest_belief.center[:] = measurement
        self.gain_index = -1
        self._stage = NEXT_UPDATE_STAGE
        self._terminal_cell = False
        self._measured_segment[0] = measurement
        self._measured_count = 1
        self._normalized_coordinates = np.empty(certificate.coordinate_offset.size)
        self._simplex_weights = np.empty(certificate.coordinate_offset.size + 1)
        self._backup_reference = np.empty(3)
        self._governor_reference = np.empty(3)
        self._solver = GovernorSolver(
            certificate.reference_lower,
            certificate.reference_upper,
            certificate.max_constraints,
        )
        self.status = (
            "active"
            if np.all(self.issued_queue >= self._issued_lower)
            and np.all(self.issued_queue <= self._issued_upper)
            else "out_of_envelope"
        )

    def command(
        self,
        state: npt.ArrayLike,
        nominal_reference: npt.ArrayLike,
        deadline: float,
    ) -> np.ndarray | None:
        """Return one command before an absolute ``perf_counter`` deadline."""

        if self.status != "active":
            raise RuntimeError("capture governor is not active")
        measurement = np.asarray(state, dtype=float).reshape(15)
        event_checked = False

        if self._terminal_cell and self._stage == PREDICTION_STAGES:
            if not self._inside_prediction(measurement, self._stage):
                self.status = "out_of_envelope"
                return None
            if self._terminal(measurement):
                return None
            self.status = "out_of_envelope"
            return None

        if self._stage == NEXT_UPDATE_STAGE and not self._terminal_cell:
            if self.gain_index >= 0:
                if not self._inside_prediction(measurement, self._stage):
                    self.status = "out_of_envelope"
                    return None
                if self._terminal(measurement):
                    return None
                event_checked = True
            if not self._update(measurement, nominal_reference, deadline):
                return None
            self._stage = 0
            event_checked = True

        if not self._inside_prediction(measurement, self._stage):
            self.status = "out_of_envelope"
            return None
        if not event_checked and self._terminal(measurement):
            return None

        prediction = self._predictor.prediction
        np.matmul(
            prediction.issued_reference[self._stage],
            self.current_reference,
            out=self._nominal_command,
        )
        self._nominal_command += prediction.issued_center[self._stage]
        np.subtract(measurement, self._nominal_state, out=self._state_error)
        np.matmul(
            self._predictor.generated.cells[self.gain_index].gain,
            self._state_error,
            out=self._feedback_command,
        )
        np.add(
            self._nominal_command,
            self._feedback_command,
            out=self._command,
        )
        for axis in range(3):
            if (
                self._command[axis] < self._issued_lower[axis]
                or self._command[axis] > self._issued_upper[axis]
            ):
                self.status = "out_of_envelope"
                return None
        np.clip(
            self._command,
            self._predictor.generated.aircraft.control_lower_rad,
            self._predictor.generated.aircraft.control_upper_rad,
            out=self._command,
        )
        for index in range(self.issued_queue.shape[0] - 1):
            self.issued_queue[index] = self.issued_queue[index + 1]
        self.issued_queue[-1] = self._command
        self._stage += 1
        return self._command

    def _inside_prediction(self, measurement: np.ndarray, stage: int) -> bool:
        prediction = self._predictor.prediction
        np.matmul(
            prediction.state_reference[stage],
            self.current_reference,
            out=self._nominal_state,
        )
        self._nominal_state += prediction.state_center[stage]
        np.subtract(
            self._nominal_state,
            prediction.state_radius[stage],
            out=self._prediction_lower,
        )
        np.add(
            self._nominal_state,
            prediction.state_radius[stage],
            out=self._prediction_upper,
        )
        np.subtract(
            measurement,
            self._predictor.generated.bounds.state_estimation_abs,
            out=self._state_lower,
        )
        np.add(
            measurement,
            self._predictor.generated.bounds.state_estimation_abs,
            out=self._state_upper,
        )
        for index in range(15):
            if (
                self._state_lower[index] < self._prediction_lower[index]
                or self._state_upper[index] > self._prediction_upper[index]
            ):
                return False
        return True

    def _terminal(self, measurement: np.ndarray) -> bool:
        if not self._terminal_cell:
            return False
        self._measured_segment[self._measured_count] = measurement
        self._measured_count += 1
        if self._measured_count <= PREDICTION_STAGES:
            return False
        realized = self.mission.realized(self._measured_segment[: self._measured_count])
        if realized:
            self.status = "terminal"
            return True
        return False

    def _update(
        self,
        state: np.ndarray,
        nominal_reference: npt.ArrayLike,
        deadline: float,
    ) -> bool:
        self.latest_belief.center[:] = state
        np.subtract(
            self.latest_belief.center,
            self._predictor.generated.bounds.state_estimation_abs,
            out=self._state_lower,
        )
        np.add(
            self.latest_belief.center,
            self._predictor.generated.bounds.state_estimation_abs,
            out=self._state_upper,
        )
        domain = self.mission.approach_domain
        for index in range(15):
            if (
                self._state_lower[index] < domain.lower[index]
                or self._state_upper[index] > domain.upper[index]
            ):
                self.status = "out_of_envelope"
                return False
        for command in self.issued_queue:
            for axis in range(3):
                if (
                    command[axis] < self._issued_lower[axis]
                    or command[axis] > self._issued_upper[axis]
                ):
                    self.status = "out_of_envelope"
                    return False

        self._augmented_center[:15] = self.latest_belief.center
        self._augmented_center[15:] = self._issued_flat
        simplex_index = self.certificate.locate_augmented_into(
            self._augmented_center,
            self._augmented_normalized,
            self._normalized_coordinates,
            self._simplex_weights,
        )
        if simplex_index < 0:
            self.status = "out_of_envelope"
            return False
        simplex = self.certificate.simplices[simplex_index]
        simplex.backup_into(self._backup_reference)
        if _deadline_reached(deadline):
            self.status = "out_of_envelope"
            return False
        self._predictor.predict(
            self.latest_belief,
            self.issued_queue,
            simplex.gain_index,
            populate_geometry=False,
        )
        matrix = simplex.constraint_matrix
        bounds = simplex.constraint_bounds[0]
        status = self._solver.solve_into(
            nominal_reference,
            self.current_reference,
            matrix,
            bounds,
            self._governor_reference,
            self.certificate.active_sets(simplex_index),
            deadline,
        )
        if status == _SOLVED:
            if not _reference_feasible(
                self._governor_reference,
                matrix,
                bounds,
                self.certificate.reference_lower,
                self.certificate.reference_upper,
            ):
                self.status = "out_of_envelope"
                return False
            reference = self._governor_reference
        elif status == _TIMED_OUT:
            reference = self._backup_reference
        else:
            self.status = "out_of_envelope"
            return False

        self.current_reference[:] = reference
        self.gain_index = simplex.gain_index
        self._terminal_cell = simplex.terminal
        self._measured_segment[0] = state
        self._measured_count = 1
        return True


def _deadline_reached(deadline: float | None) -> bool:
    return deadline is not None and perf_counter() >= deadline


def _reference_feasible(
    reference: np.ndarray,
    matrix: np.ndarray,
    bounds: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> bool:
    for axis in range(3):
        if reference[axis] < lower[axis] or reference[axis] > upper[axis]:
            return False
    for index, row in enumerate(matrix):
        if (
            row[0] * reference[0]
            + row[1] * reference[1]
            + row[2] * reference[2]
            - bounds[index]
            > 0.0
        ):
            return False
    return True
