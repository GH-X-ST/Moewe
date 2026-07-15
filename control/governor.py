"""Deterministic active-set solver for the three-input reference governor."""

from __future__ import annotations

from math import sqrt
from time import perf_counter

import numpy as np
import numpy.typing as npt

from control.capture import (
    POLYTOPE_TOLERANCE,
    CaptureCertificate,
    CaptureConstraintBuilder,
)
from control.observer import StateFlowEstimate
from control.predictor import FastPredictor, GeneratedAircraft
from control.uncertainty import NEXT_UPDATE_STAGE, PREDICTION_STAGES


TRACKING_WEIGHT = 1.0
SMOOTHING_WEIGHT = 0.1

_RANK_TOLERANCE = 64.0 * np.finfo(float).eps
_INFEASIBLE = -1
_TIMED_OUT = 0
_SOLVED = 1
_CONTINUE = 2


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
        self._iterate = np.empty(3)
        self._active: np.ndarray = np.empty(3, dtype=int)
        self._gram = np.empty((3, 3))
        self._rhs = np.empty(3)
        self._multipliers = np.empty(3)
        self._active_count = 0
        self._deadline: float | None = None
        self._timed_out = False

    def solve_into(
        self,
        nominal: npt.ArrayLike,
        previous: npt.ArrayLike,
        a: np.ndarray,
        b: np.ndarray,
        result: np.ndarray,
        backup: npt.ArrayLike,
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

        self._normalise_reference(backup, self._iterate)
        if not self._feasible(self._iterate, row_count):
            return _INFEASIBLE
        self._active_count = 0
        for _ in range(8 * row_count + 16):
            if _deadline_reached(deadline):
                return _TIMED_OUT
            status = self._active_step(row_count)
            if status == _SOLVED:
                self._result_into(self._iterate, result)
                return _SOLVED
            if status == _TIMED_OUT:
                return _TIMED_OUT
            if status == _INFEASIBLE:
                return _INFEASIBLE
        return _TIMED_OUT

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

    def _active_step(self, row_count: int) -> int:
        if self._active_count and self._stationary(row_count):
            return _SOLVED
        if self._timed_out:
            return _TIMED_OUT
        np.subtract(self._unconstrained, self._iterate, out=self._candidate)
        if self._active_count:
            active = self._active[: self._active_count]
            rows = self._a[active]
            gram = self._gram[: self._active_count, : self._active_count]
            np.matmul(rows, rows.T, out=gram)
            rhs = self._rhs[: self._active_count]
            np.matmul(rows, self._candidate, out=rhs)
            try:
                multipliers = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                return _INFEASIBLE
            self._multipliers[: self._active_count] = multipliers
            self._candidate -= rows.T @ multipliers

        if float(self._candidate @ self._candidate) <= _RANK_TOLERANCE**2:
            if not self._active_count:
                return _SOLVED
            multipliers = self._multipliers[: self._active_count]
            remove = int(np.argmin(multipliers))
            if multipliers[remove] >= -POLYTOPE_TOLERANCE:
                return _SOLVED
            self._remove_active(remove)
            return _CONTINUE

        step = 1.0
        blocker = -1
        for index in range(row_count):
            if index % 16 == 0 and _deadline_reached(self._deadline):
                return _TIMED_OUT
            if self._is_active(index):
                continue
            rate = float(self._a[index] @ self._candidate)
            if rate <= _RANK_TOLERANCE:
                continue
            distance = float(self._b[index] - self._a[index] @ self._iterate)
            candidate = max(0.0, distance / rate)
            if candidate < step - POLYTOPE_TOLERANCE or (
                abs(candidate - step) <= POLYTOPE_TOLERANCE
                and (blocker < 0 or index < blocker)
            ):
                step = candidate
                blocker = index
        self._iterate += step * self._candidate
        if blocker < 0:
            return _CONTINUE
        if not self._append_active(blocker):
            return _CONTINUE
        return _CONTINUE

    def _stationary(self, row_count: int) -> bool:
        gradient = self._unconstrained - self._iterate
        tolerance = POLYTOPE_TOLERANCE
        tight = [
            index
            for index in range(row_count)
            if abs(float(self._a[index] @ self._iterate - self._b[index])) <= tolerance
            and float(self._a[index] @ self._a[index]) > _RANK_TOLERANCE
        ]
        for first, first_index in enumerate(tight):
            if first % 8 == 0 and _deadline_reached(self._deadline):
                self._timed_out = True
                return False
            row = self._a[first_index]
            multiplier = float(row @ gradient / (row @ row))
            if (
                multiplier >= -tolerance
                and np.linalg.norm(gradient - multiplier * row) <= tolerance
            ):
                return True
            for second in range(first + 1, len(tight)):
                rows = self._a[[first_index, tight[second]]]
                gram = rows @ rows.T
                if np.linalg.det(gram) <= _RANK_TOLERANCE:
                    continue
                multipliers = np.linalg.solve(gram, rows @ gradient)
                if (
                    np.all(multipliers >= -tolerance)
                    and np.linalg.norm(gradient - rows.T @ multipliers) <= tolerance
                ):
                    return True
                for third in range(second + 1, len(tight)):
                    rows = self._a[[first_index, tight[second], tight[third]]]
                    determinant = float(np.linalg.det(rows))
                    if abs(determinant) <= _RANK_TOLERANCE:
                        continue
                    multipliers = np.linalg.solve(rows.T, gradient)
                    if np.all(multipliers >= -tolerance):
                        return True
        return False

    def _is_active(self, index: int) -> bool:
        for position in range(self._active_count):
            if self._active[position] == index:
                return True
        return False

    def _append_active(self, index: int) -> bool:
        row = self._a[index]
        if self._active_count == 1:
            first = self._a[self._active[0]]
            if float(np.linalg.norm(np.cross(first, row))) <= _RANK_TOLERANCE:
                return False
        elif self._active_count == 2:
            rows = self._a[self._active[:2]]
            if abs(float(np.linalg.det(np.vstack((rows, row))))) <= _RANK_TOLERANCE:
                return False
        elif self._active_count == 3:
            return False
        self._active[self._active_count] = index
        self._active_count += 1
        return True

    def _remove_active(self, position: int) -> None:
        for index in range(position, self._active_count - 1):
            self._active[index] = self._active[index + 1]
        self._active_count -= 1

    def _feasible(self, candidate: np.ndarray, row_count: int) -> bool:
        for row in range(row_count):
            if row % 16 == 0 and _deadline_reached(self._deadline):
                self._timed_out = True
                return False
            if self._a[row] @ candidate - self._b[row] > POLYTOPE_TOLERANCE:
                return False
        return True

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
        self._predictor = FastPredictor(generated)
        self._stage = NEXT_UPDATE_STAGE
        self._terminal_cell = False
        self._last_estimate_timestamp_s = -np.inf
        self._measured_segment = np.empty((2, 15))
        self._augmented_center = np.empty(15 + 3 * generated.bounds.queue_length)
        self._augmented_normalized = np.empty(self._augmented_center.size)
        self._state_lower = np.empty(15)
        self._state_upper = np.empty(15)
        self._belief_radius = np.empty(15)
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
        estimate: StateFlowEstimate,
        reference: npt.ArrayLike,
        issued_queue: npt.ArrayLike,
    ) -> None:
        """Activate one compiled mission from a containing observer set."""

        if certificate.generated is not self._predictor.generated:
            raise ValueError("capture certificate belongs to another aircraft core")
        measurement = estimate.state.center
        self.mission = certificate.mission
        self.certificate = certificate
        self.current_reference[:] = np.asarray(reference, dtype=float).reshape(3)
        self.issued_queue[:] = np.asarray(issued_queue, dtype=float).reshape(
            self.issued_queue.shape
        )
        self.gain_index = -1
        self._stage = NEXT_UPDATE_STAGE
        self._terminal_cell = False
        self._last_estimate_timestamp_s = -np.inf
        self._measured_segment[0] = measurement
        self._normalized_coordinates = np.empty(certificate.coordinate_offset.size)
        self._simplex_weights = np.empty(certificate.coordinate_offset.size + 1)
        self._backup_reference = np.empty(3)
        self._governor_reference = np.empty(3)
        self._solver = GovernorSolver(
            certificate.reference_lower,
            certificate.reference_upper,
            certificate.max_constraints,
        )
        self._constraint_builder = CaptureConstraintBuilder(
            certificate.generated,
            certificate.mission,
            certificate.domain,
        )
        self.status = (
            "active"
            if self._valid_estimate(estimate)
            and np.all(self.issued_queue >= self._issued_lower)
            and np.all(self.issued_queue <= self._issued_upper)
            else "out_of_envelope"
        )

    def command(
        self,
        estimate: StateFlowEstimate,
        nominal_reference: npt.ArrayLike,
        deadline: float,
    ) -> np.ndarray | None:
        """Return one command under the deployment deadline contract."""

        if self.status != "active":
            raise RuntimeError("capture governor is not active")
        if (
            estimate.timestamp_s <= self._last_estimate_timestamp_s
            or not self._valid_estimate(estimate)
        ):
            self.status = "out_of_envelope"
            return None
        self._last_estimate_timestamp_s = estimate.timestamp_s
        measurement = estimate.state.center
        event_checked = False

        if self._terminal_cell and self._stage == PREDICTION_STAGES:
            if not self._inside_prediction(estimate, self._stage):
                self.status = "out_of_envelope"
                return None
            if self._terminal(measurement):
                return None
            self.status = "out_of_envelope"
            return None

        if self._stage == NEXT_UPDATE_STAGE and not self._terminal_cell:
            if self.gain_index >= 0:
                if not self._inside_prediction(estimate, self._stage):
                    self.status = "out_of_envelope"
                    return None
                if self._terminal(measurement):
                    return None
                event_checked = True
            if not self._update(estimate, nominal_reference, deadline):
                return None
            self._stage = 0
            event_checked = True

        if not self._inside_prediction(estimate, self._stage):
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

    def _inside_prediction(self, estimate: StateFlowEstimate, stage: int) -> bool:
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
        self._belief_bounds(estimate)
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
        self._measured_segment[1] = measurement
        realized = self.mission.realized(
            self._measured_segment,
            self._predictor.generated,
        )
        if realized:
            self.status = "terminal"
            return True
        self._measured_segment[0] = measurement
        return False

    def _update(
        self,
        estimate: StateFlowEstimate,
        nominal_reference: npt.ArrayLike,
        deadline: float,
    ) -> bool:
        belief = estimate.state
        self._belief_bounds(estimate)
        domain = self.certificate.domain
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

        self._augmented_center[:15] = belief.center
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
            belief,
            self.issued_queue,
            simplex.gain_index,
            local_flow=estimate.local_flow,
        )
        reference = self._backup_reference
        if not _deadline_reached(deadline):
            matrix, bounds = self._constraint_builder.rows(
                self._predictor.prediction,
                self.certificate.beta,
                simplex.progress_m,
                simplex.terminal,
            )
            if simplex.terminal:
                matrix = np.vstack((matrix, np.eye(3), -np.eye(3)))
                bounds = np.concatenate(
                    (
                        bounds,
                        simplex.reference_upper,
                        -simplex.reference_lower,
                    )
                )
            if matrix.shape[0] != simplex.constraint_count:
                self.status = "out_of_envelope"
                return False
            if not _deadline_reached(deadline):
                status = self._solver.solve_into(
                    nominal_reference,
                    self.current_reference,
                    matrix,
                    bounds,
                    self._governor_reference,
                    self._backup_reference,
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
                elif status not in (_TIMED_OUT, _INFEASIBLE):
                    self.status = "out_of_envelope"
                    return False

        self.current_reference[:] = reference
        self.gain_index = simplex.gain_index
        self._terminal_cell = simplex.terminal
        self._measured_segment[0] = estimate.state.center
        return True

    def _valid_estimate(self, estimate: StateFlowEstimate) -> bool:
        self._belief_radius[:] = estimate.state.radius
        return bool(
            np.all(
                self._belief_radius
                <= self._predictor.generated.bounds.state_estimation_abs
            )
            and estimate.local_flow.subset(
                self._predictor.generated.bounds.flow
            )
            and estimate.state.generators.shape[1]
            <= self._predictor.max_initial_generators
        )

    def _belief_bounds(self, estimate: StateFlowEstimate) -> None:
        np.subtract(
            estimate.state.center,
            self._belief_radius,
            out=self._state_lower,
        )
        np.add(
            estimate.state.center,
            self._belief_radius,
            out=self._state_upper,
        )


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
