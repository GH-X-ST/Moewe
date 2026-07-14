"""Generated affine predictor for the 20 ms and 100 ms control paths."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from math import cos, sin, sqrt

import numpy as np
import numpy.typing as npt

from control.interval import Interval, Zonotope
from control.uncertainty import Bounds, FAST_PERIOD_S, PREDICTION_STAGES
from models.aircraft import Aircraft
from models.geometry import (
    RigidBodyGeometry,
)

INTRINSIC_FEEDBACK_INDICES = np.array(
    [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    dtype=int,
)
LQR_STATE_WEIGHT = np.ones(INTRINSIC_FEEDBACK_INDICES.size)
LQR_CONTROL_WEIGHT = np.ones(3)
REFERENCE_IDENTITY = np.eye(3)
LQR_STATE_WEIGHT.flags.writeable = False
LQR_CONTROL_WEIGHT.flags.writeable = False
REFERENCE_IDENTITY.flags.writeable = False
MIN_GAIN_CELL_WIDTH = 0.125
MAX_GAIN_CELL_DEPTH = 8


@dataclass(frozen=True)
class _CellVerification:
    stage_remainder_abs: npt.ArrayLike
    swept_position_remainder_abs_m: npt.ArrayLike
    contact_velocity_remainder_abs_m_s: npt.ArrayLike

    def __post_init__(self) -> None:
        _set_array(self, "stage_remainder_abs", self.stage_remainder_abs, (15,))
        _set_array(
            self,
            "swept_position_remainder_abs_m",
            self.swept_position_remainder_abs_m,
            (3,),
        )
        _set_array(
            self,
            "contact_velocity_remainder_abs_m_s",
            self.contact_velocity_remainder_abs_m_s,
            (3,),
        )
        for name in (
            "stage_remainder_abs",
            "swept_position_remainder_abs_m",
            "contact_velocity_remainder_abs_m_s",
        ):
            if np.any(getattr(self, name) < 0.0):
                raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True)
class GainCell:
    """One generated affine model and stabilizing inner gain."""

    lower: npt.ArrayLike
    upper: npt.ArrayLike
    anchor: npt.ArrayLike
    control_anchor: npt.ArrayLike
    state_matrix: npt.ArrayLike
    control_matrix: npt.ArrayLike
    offset: npt.ArrayLike
    gain: npt.ArrayLike
    flow_generators: npt.ArrayLike
    model_generators: npt.ArrayLike
    stage_remainder_abs: npt.ArrayLike
    swept_position_remainder_abs_m: npt.ArrayLike
    contact_velocity_remainder_abs_m_s: npt.ArrayLike

    def __post_init__(self) -> None:
        arrays = (
            ("lower", (INTRINSIC_FEEDBACK_INDICES.size,)),
            ("upper", (INTRINSIC_FEEDBACK_INDICES.size,)),
            ("anchor", (15,)),
            ("control_anchor", (3,)),
            ("state_matrix", (15, 15)),
            ("control_matrix", (15, 3)),
            ("offset", (15,)),
            ("gain", (3, 15)),
            ("stage_remainder_abs", (15,)),
            ("swept_position_remainder_abs_m", (3,)),
            ("contact_velocity_remainder_abs_m_s", (3,)),
        )
        for name, shape in arrays:
            _set_array(self, name, getattr(self, name), shape)
        flow = np.asarray(self.flow_generators, dtype=float).reshape(15, -1).copy()
        model = np.asarray(self.model_generators, dtype=float).reshape(15, -1).copy()
        flow.flags.writeable = False
        model.flags.writeable = False
        object.__setattr__(self, "flow_generators", flow)
        object.__setattr__(self, "model_generators", model)


@dataclass(frozen=True)
class RejectedCell:
    """One normalized aircraft-domain cell excluded by generation."""

    anchor: npt.ArrayLike
    lower: npt.ArrayLike
    upper: npt.ArrayLike

    def __post_init__(self) -> None:
        _set_array(self, "anchor", self.anchor, (15,))
        shape = (INTRINSIC_FEEDBACK_INDICES.size,)
        _set_array(self, "lower", self.lower, shape)
        _set_array(self, "upper", self.upper, shape)


@dataclass(frozen=True)
class _AircraftModel:
    aircraft: Aircraft
    geometry: RigidBodyGeometry
    bounds: Bounds
    state_scale: npt.ArrayLike
    domain_anchor: npt.ArrayLike
    reference_center: npt.ArrayLike
    reference_scale: npt.ArrayLike
    cells: tuple[GainCell, ...]
    rejected_cells: tuple[RejectedCell, ...]

    def __post_init__(self) -> None:
        _set_array(self, "state_scale", self.state_scale, (15,))
        _set_array(self, "domain_anchor", self.domain_anchor, (15,))
        _set_array(self, "reference_center", self.reference_center, (3,))
        _set_array(self, "reference_scale", self.reference_scale, (3,))

    def cell(self, state: npt.ArrayLike) -> tuple[int, GainCell]:
        """Return the generated cell containing an intrinsic state."""

        value = np.asarray(state, dtype=float).reshape(15)
        for index, cell in enumerate(self.cells):
            intrinsic = (
                value[INTRINSIC_FEEDBACK_INDICES]
                - cell.anchor[INTRINSIC_FEEDBACK_INDICES]
            ) / self.state_scale[INTRINSIC_FEEDBACK_INDICES]
            if np.all(intrinsic >= cell.lower) and np.all(intrinsic <= cell.upper):
                return index, cell
        raise ValueError("state is outside the generated aircraft domain")


@dataclass(frozen=True)
class GeneratedAircraft(_AircraftModel):
    """Verified aircraft core accepted by capture and runtime control."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.cells:
            raise ValueError("oracle verification produced no aircraft gain cell")


@dataclass
class Prediction:
    """Preallocated affine state and command prediction arrays."""

    max_generators: int
    reference_center: np.ndarray = field(init=False)
    reference_radius: np.ndarray = field(init=False)
    flow_center: np.ndarray = field(init=False)
    flow_radius: np.ndarray = field(init=False)
    state_center: np.ndarray = field(init=False)
    state_reference: np.ndarray = field(init=False)
    state_generators: np.ndarray = field(init=False)
    state_radius: np.ndarray = field(init=False)
    generator_count: np.ndarray = field(init=False)
    issued_center: np.ndarray = field(init=False)
    issued_reference: np.ndarray = field(init=False)
    issued_radius: np.ndarray = field(init=False)
    applied_center: np.ndarray = field(init=False)
    applied_reference: np.ndarray = field(init=False)
    applied_radius: np.ndarray = field(init=False)
    body_center: np.ndarray = field(init=False)
    body_reference: np.ndarray = field(init=False)
    body_radius: np.ndarray = field(init=False)
    contact_center: np.ndarray = field(init=False)
    contact_reference: np.ndarray = field(init=False)
    contact_radius: np.ndarray = field(init=False)
    footprint_center: np.ndarray = field(init=False)
    footprint_reference: np.ndarray = field(init=False)
    footprint_radius: np.ndarray = field(init=False)
    contact_velocity_center: np.ndarray = field(init=False)
    contact_velocity_reference: np.ndarray = field(init=False)
    contact_velocity_radius: np.ndarray = field(init=False)
    body_count: int = 0
    contact_count: int = 0
    footprint_count: int = 0

    def __post_init__(self) -> None:
        stages = PREDICTION_STAGES
        self.reference_center = np.empty(3)
        self.reference_radius = np.empty(3)
        self.flow_center = np.zeros(3)
        self.flow_radius = np.zeros(3)
        self.state_center = np.empty((stages + 1, 15))
        self.state_reference = np.empty((stages + 1, 15, 3))
        self.state_generators = np.zeros((stages + 1, 15, self.max_generators))
        self.state_radius = np.empty((stages + 1, 15))
        self.generator_count = np.empty(stages + 1, dtype=int)
        self.issued_center = np.empty((stages, 3))
        self.issued_reference = np.empty((stages, 3, 3))
        self.issued_radius = np.empty((stages, 3))
        self.applied_center = np.empty((stages, 3))
        self.applied_reference = np.empty((stages, 3, 3))
        self.applied_radius = np.empty((stages, 3))
        self.body_center = np.empty((stages, self.body_count, 3))
        self.body_reference = np.empty((stages, self.body_count, 3, 3))
        self.body_radius = np.empty((stages, self.body_count, 3))
        self.contact_center = np.empty((stages, self.contact_count, 3))
        self.contact_reference = np.empty((stages, self.contact_count, 3, 3))
        self.contact_radius = np.empty((stages, self.contact_count, 3))
        self.footprint_center = np.empty((stages, self.footprint_count, 3))
        self.footprint_reference = np.empty((stages, self.footprint_count, 3, 3))
        self.footprint_radius = np.empty((stages, self.footprint_count, 3))
        self.contact_velocity_center = np.empty((stages, self.contact_count, 3))
        self.contact_velocity_reference = np.empty((stages, self.contact_count, 3, 3))
        self.contact_velocity_radius = np.empty((stages, self.contact_count, 3))

    def state_support(
        self,
        stage: int,
        direction: npt.ArrayLike,
    ) -> tuple[float, np.ndarray]:
        """Return affine upper-support coefficients in the reference."""

        row = np.asarray(direction, dtype=float).reshape(15)
        count = self.generator_count[stage]
        offset = float(row @ self.state_center[stage])
        offset += float(np.sum(np.abs(row @ self.state_generators[stage, :, :count])))
        return offset, row @ self.state_reference[stage]

    def state_interval(
        self,
        stage: int,
        reference_lower: npt.ArrayLike,
        reference_upper: npt.ArrayLike,
    ) -> Interval:
        """Return the interval hull over a physical reference box."""

        lower = np.asarray(reference_lower, dtype=float).reshape(3)
        upper = np.asarray(reference_upper, dtype=float).reshape(3)
        reference = Interval(lower, upper).affine_map(self.state_reference[stage])
        return Interval.from_midpoint(
            self.state_center[stage] + reference.center,
            self.state_radius[stage] + reference.radius,
        )

    def issued_support(
        self,
        stage: int,
        direction: npt.ArrayLike,
    ) -> tuple[float, np.ndarray]:
        """Return affine issued-command support coefficients."""

        row = np.asarray(direction, dtype=float).reshape(3)
        offset = float(row @ self.issued_center[stage])
        offset += float(np.abs(row) @ self.issued_radius[stage])
        return offset, row @ self.issued_reference[stage]

    def applied_support(
        self,
        stage: int,
        direction: npt.ArrayLike,
    ) -> tuple[float, np.ndarray]:
        """Return affine delayed-command support coefficients."""

        row = np.asarray(direction, dtype=float).reshape(3)
        offset = float(row @ self.applied_center[stage])
        offset += float(np.abs(row) @ self.applied_radius[stage])
        return offset, row @ self.applied_reference[stage]

    def air_velocity_support(
        self,
        stage: int,
        direction: npt.ArrayLike,
    ) -> tuple[float, np.ndarray]:
        """Return affine support of body-frame CG air velocity."""

        axis = np.asarray(direction, dtype=float).reshape(3)
        row = np.zeros(15)
        row[6:9] = axis
        offset, reference = self.state_support(stage, row)
        offset -= float(axis @ self.flow_center)
        offset += float(np.abs(axis) @ self.flow_radius)
        return offset, reference

    def airspeed_support(
        self,
        stage: int,
        coefficient: float,
    ) -> tuple[float, np.ndarray]:
        """Return affine support of a signed CG airspeed."""

        if coefficient < 0.0:
            direction = np.array((coefficient, 0.0, 0.0))
            return self.air_velocity_support(stage, direction)
        center = (
            self.state_center[stage, 6:9]
            + self.state_reference[stage, 6:9] @ self.reference_center
            - self.flow_center
        )
        speed = float(np.linalg.norm(center))
        direction = center / speed if speed > 0.0 else np.array((1.0, 0.0, 0.0))
        offset, reference = self.air_velocity_support(stage, direction)
        count = self.generator_count[stage]
        deviation = np.sum(
            np.abs(self.state_generators[stage, 6:9, :count]),
            axis=1,
        )
        deviation += self.flow_radius
        deviation += np.abs(self.state_reference[stage, 6:9]) @ self.reference_radius
        offset += 2.0 * float(np.linalg.norm(deviation))
        return coefficient * offset, coefficient * reference

    def body_support(
        self,
        stage: int,
        point: int,
        direction: npt.ArrayLike,
    ) -> tuple[float, np.ndarray]:
        """Return affine swept-body support coefficients."""

        return self._point_support(
            self.body_center,
            self.body_reference,
            self.body_radius,
            stage,
            point,
            direction,
        )

    def footprint_support(
        self,
        stage: int,
        point: int,
        direction: npt.ArrayLike,
    ) -> tuple[float, np.ndarray]:
        """Return affine landing-footprint support coefficients."""

        return self._point_support(
            self.footprint_center,
            self.footprint_reference,
            self.footprint_radius,
            stage,
            point,
            direction,
        )

    def contact_velocity_support(
        self,
        stage: int,
        point: int,
        direction: npt.ArrayLike,
    ) -> tuple[float, np.ndarray]:
        """Return affine contact-point velocity support coefficients."""

        return self._point_support(
            self.contact_velocity_center,
            self.contact_velocity_reference,
            self.contact_velocity_radius,
            stage,
            point,
            direction,
        )

    @staticmethod
    def _point_support(
        center: np.ndarray,
        reference: np.ndarray,
        radius: np.ndarray,
        stage: int,
        point: int,
        direction: npt.ArrayLike,
    ) -> tuple[float, np.ndarray]:
        row = np.asarray(direction, dtype=float).reshape(3)
        offset = float(row @ center[stage, point])
        offset += float(np.abs(row) @ radius[stage, point])
        return offset, row @ reference[stage, point]


@dataclass
class FastPredictor:
    """Evaluate one ten-stage affine prediction without runtime allocation."""

    generated: _AircraftModel
    max_initial_generators: int = 30
    prediction: Prediction = field(init=False)
    _history_center: np.ndarray = field(init=False, repr=False)
    _history_reference: np.ndarray = field(init=False, repr=False)
    _history_generators: np.ndarray = field(init=False, repr=False)
    _history_radius: np.ndarray = field(init=False, repr=False)
    _history_count: np.ndarray = field(init=False, repr=False)
    _reference_abs: np.ndarray = field(init=False, repr=False)
    _measurement_radius: np.ndarray = field(init=False, repr=False)
    _issued_generators: np.ndarray = field(init=False, repr=False)
    _factor_abs: np.ndarray = field(init=False, repr=False)
    _state_vector: np.ndarray = field(init=False, repr=False)
    _state_reference_product: np.ndarray = field(init=False, repr=False)
    _applied_residual: np.ndarray = field(init=False, repr=False)
    _applied_difference: np.ndarray = field(init=False, repr=False)
    _applied_support: np.ndarray = field(init=False, repr=False)
    _stage_rotations: np.ndarray = field(init=False, repr=False)
    _contact_indices: np.ndarray = field(init=False, repr=False)
    _footprint_indices: np.ndarray = field(init=False, repr=False)
    _body_norms: np.ndarray = field(init=False, repr=False)
    _contact_norms: np.ndarray = field(init=False, repr=False)
    _point_center: np.ndarray = field(init=False, repr=False)
    _point_reference: np.ndarray = field(init=False, repr=False)
    _point_radius: np.ndarray = field(init=False, repr=False)
    _point_jacobian: np.ndarray = field(init=False, repr=False)
    _point_generators: np.ndarray = field(init=False, repr=False)
    _point_generator_abs: np.ndarray = field(init=False, repr=False)
    _angle_generator_abs: np.ndarray = field(init=False, repr=False)
    _frame_rotation: np.ndarray = field(init=False, repr=False)
    _frame_rotation_abs: np.ndarray = field(init=False, repr=False)
    _frame_origin: np.ndarray = field(init=False, repr=False)
    _frame_reference: np.ndarray = field(init=False, repr=False)
    _frame_generators: np.ndarray = field(init=False, repr=False)
    _frame_position_remainder: np.ndarray = field(init=False, repr=False)
    _frame_velocity_remainder: np.ndarray = field(init=False, repr=False)
    _frame_yaw_shift: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        cell = self.generated.cells[0]
        per_stage = cell.flow_generators.shape[1] + 3 + 15
        maximum = (
            self.max_initial_generators
            + cell.model_generators.shape[1]
            + PREDICTION_STAGES * per_stage
        )
        geometry = self.generated.geometry
        self.prediction = Prediction(
            maximum,
            body_count=np.asarray(geometry.body_b_m).reshape(-1, 3).shape[0],
            contact_count=np.asarray(geometry.contact_b_m).reshape(-1, 3).shape[0],
            footprint_count=np.asarray(geometry.footprint_b_m).reshape(-1, 3).shape[0],
        )
        self.prediction.reference_center[:] = self.generated.reference_center
        self.prediction.reference_radius[:] = self.generated.reference_scale
        flow = self.generated.bounds.flow.joint_zonotope(
            self.generated.aircraft.strip_table.r_b_m
        )
        self.prediction.flow_center[:] = flow.center[:3]
        self.prediction.flow_radius[:] = np.sum(
            np.abs(flow.generators[:3]),
            axis=1,
        )
        history = self.generated.bounds.queue_length + PREDICTION_STAGES
        self._history_center = np.zeros((history, 3))
        self._history_reference = np.zeros((history, 3, 3))
        self._history_generators = np.zeros((history, 3, maximum))
        self._history_radius = np.zeros((history, 3))
        self._history_count = np.zeros(history, dtype=int)
        self._reference_abs = np.maximum(
            np.abs(self.generated.aircraft.control_lower_rad),
            np.abs(self.generated.aircraft.control_upper_rad),
        )
        self._measurement_radius = np.stack(
            [
                np.abs(gain_cell.gain) @ self.generated.bounds.state_estimation_abs
                for gain_cell in self.generated.cells
            ]
        )
        self._issued_generators = np.empty((3, maximum))
        self._factor_abs = np.empty(maximum)
        self._state_vector = np.empty(15)
        self._state_reference_product = np.empty((15, 3))
        self._applied_residual = np.empty(3)
        self._applied_difference = np.empty((3, 3))
        self._applied_support = np.empty(3)
        point_count = max(
            self.prediction.body_count,
            self.prediction.contact_count,
            self.prediction.footprint_count,
        )
        self._stage_rotations = np.empty((PREDICTION_STAGES + 1, 4, 3, 3))
        body = np.asarray(geometry.body_b_m).reshape(-1, 3)
        contact = np.asarray(geometry.contact_b_m).reshape(-1, 3)
        footprint = np.asarray(geometry.footprint_b_m).reshape(-1, 3)
        self._contact_indices = _point_indices(body, contact)
        self._footprint_indices = _point_indices(body, footprint)
        self._body_norms = np.linalg.norm(body, axis=1)
        self._contact_norms = self._body_norms[self._contact_indices]
        self._point_center = np.empty((2, point_count, 3))
        self._point_reference = np.empty((2, point_count, 3, 3))
        self._point_radius = np.empty((2, point_count, 3))
        self._point_jacobian = np.empty((point_count, 3, 15))
        self._point_generators = np.empty((point_count, 3, maximum))
        self._point_generator_abs = np.empty((point_count, 3, maximum))
        self._angle_generator_abs = np.empty(maximum)
        self._frame_rotation = np.empty((3, 3))
        self._frame_rotation_abs = np.empty((3, 3))
        self._frame_origin = np.empty(3)
        self._frame_reference = np.empty((3, 3))
        self._frame_generators = np.empty((3, maximum))
        self._frame_position_remainder = np.empty(3)
        self._frame_velocity_remainder = np.empty(3)

    def predict(
        self,
        belief: Zonotope,
        issued_queue: npt.ArrayLike,
        cell_index: int,
        issued_queue_radius: npt.ArrayLike | None = None,
        populate_geometry: bool = True,
    ) -> Prediction:
        """Populate and return the affine prediction for one gain cell."""

        cell = self.generated.cells[cell_index]
        output = self.prediction
        initial_count = belief.generators.shape[1]
        model_count = cell.model_generators.shape[1]
        model_start = initial_count
        count = initial_count + model_count
        self._canonical_initial(output, belief, cell, initial_count)
        output.state_reference[0].fill(0.0)
        output.generator_count[0] = count

        queue = np.asarray(issued_queue, dtype=float).reshape(
            self.generated.bounds.queue_length,
            3,
        )
        queue_length = queue.shape[0]
        self._history_center.fill(0.0)
        self._history_reference.fill(0.0)
        self._history_radius.fill(0.0)
        self._history_count.fill(0)
        self._history_center[:queue_length] = queue
        self._history_radius[:queue_length] = (
            self.generated.bounds.command_error_abs_rad
        )
        if issued_queue_radius is not None:
            self._history_radius[:queue_length] += np.asarray(
                issued_queue_radius,
                dtype=float,
            ).reshape(queue.shape)

        reference_abs = self._reference_abs
        measurement_radius = self._measurement_radius[cell_index]
        for stage in range(PREDICTION_STAGES):
            state_count = int(output.generator_count[stage])
            history_index = queue_length + stage
            state_center = output.state_center[stage]
            state_reference = output.state_reference[stage]
            state_generators = output.state_generators[
                stage,
                :,
                :state_count,
            ]
            issued_center = output.issued_center[stage]
            issued_center.fill(0.0)
            issued_reference = output.issued_reference[stage]
            issued_reference[:] = REFERENCE_IDENTITY
            issued_generators = self._issued_generators[:, :state_count]
            np.matmul(cell.gain, state_generators, out=issued_generators)
            issued_radius = output.issued_radius[stage]
            np.add(
                measurement_radius,
                self.generated.bounds.command_error_abs_rad,
                out=issued_radius,
            )
            self._generator_radii(
                issued_generators,
                state_count,
                self._applied_support,
            )
            issued_radius += self._applied_support
            self._history_center[history_index] = issued_center
            self._history_reference[history_index] = issued_reference
            self._history_generators[
                history_index,
                :,
                :state_count,
            ] = issued_generators
            self._history_radius[history_index] = measurement_radius
            self._history_radius[history_index] += (
                self.generated.bounds.command_error_abs_rad
            )
            self._history_count[history_index] = state_count

            self._applied(
                stage,
                queue_length,
                reference_abs,
            )
            applied_center = output.applied_center[stage]
            applied_reference = output.applied_reference[stage]
            applied_radius = output.applied_radius[stage]

            next_generators = output.state_generators[stage + 1]
            np.matmul(
                cell.state_matrix,
                state_generators,
                out=next_generators[:, :state_count],
            )
            next_generators[
                :,
                model_start : model_start + model_count,
            ] += cell.model_generators
            column = state_count
            flow_count = cell.flow_generators.shape[1]
            next_generators[:, column : column + flow_count] = cell.flow_generators
            column += flow_count
            for state_index in range(15):
                for control_index in range(3):
                    next_generators[state_index, column + control_index] = (
                        cell.control_matrix[state_index, control_index]
                        * applied_radius[control_index]
                    )
            column += 3
            next_generators[:, column : column + 15].fill(0.0)
            for index in range(15):
                next_generators[index, column + index] = cell.stage_remainder_abs[index]
            column += 15
            next_center = output.state_center[stage + 1]
            np.matmul(cell.state_matrix, state_center, out=next_center)
            np.matmul(
                cell.control_matrix,
                applied_center,
                out=self._state_vector,
            )
            next_center += self._state_vector
            next_center += cell.offset
            next_reference = output.state_reference[stage + 1]
            np.matmul(cell.state_matrix, state_reference, out=next_reference)
            np.matmul(
                cell.control_matrix,
                applied_reference,
                out=self._state_reference_product,
            )
            next_reference += self._state_reference_product
            output.generator_count[stage + 1] = column
        self._world_prediction(output, cell)
        self._state_radii(output)
        if populate_geometry:
            self._populate_geometry(output, cell)
        return output

    def _state_radii(self, output: Prediction) -> None:
        for stage in range(PREDICTION_STAGES + 1):
            count = int(output.generator_count[stage])
            self._generator_radii(
                output.state_generators[stage],
                count,
                output.state_radius[stage],
            )

    def _generator_radii(
        self,
        generators: np.ndarray,
        count: int,
        result: np.ndarray,
    ) -> None:
        absolute = self._factor_abs[:count]
        for row in range(generators.shape[0]):
            np.abs(generators[row, :count], out=absolute)
            result[row] = np.add.reduce(absolute)

    def _canonical_initial(
        self,
        output: Prediction,
        belief: Zonotope,
        cell: GainCell,
        initial_count: int,
    ) -> None:
        yaw_shift = float(belief.center[5] - cell.anchor[5])
        cosine = cos(yaw_shift)
        sine = sin(yaw_shift)
        rotation = self._frame_rotation
        rotation[:] = (
            (cosine, sine, 0.0),
            (-sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        )
        np.abs(rotation, out=self._frame_rotation_abs)
        self._frame_origin[:] = belief.center[:3]
        self._frame_yaw_shift = yaw_shift

        center = output.state_center[0]
        center[:] = belief.center
        center[:3] = cell.anchor[:3]
        center[5] = cell.anchor[5]
        generators = output.state_generators[0]
        np.matmul(
            rotation.T,
            belief.generators[:3],
            out=generators[:3, :initial_count],
        )
        generators[3:, :initial_count] = belief.generators[3:]
        model_count = cell.model_generators.shape[1]
        generators[:, initial_count : initial_count + model_count].fill(0.0)

    def _world_prediction(self, output: Prediction, cell: GainCell) -> None:
        rotation = self._frame_rotation
        for stage in range(PREDICTION_STAGES + 1):
            center = output.state_center[stage]
            np.subtract(center[:3], cell.anchor[:3], out=self._state_vector[:3])
            np.matmul(rotation, self._state_vector[:3], out=center[:3])
            center[:3] += self._frame_origin
            center[5] += self._frame_yaw_shift

            reference = output.state_reference[stage]
            np.matmul(rotation, reference[:3], out=self._frame_reference)
            reference[:3] = self._frame_reference

            count = output.generator_count[stage]
            generators = output.state_generators[stage, :, :count]
            np.matmul(
                rotation,
                generators[:3],
                out=self._frame_generators[:, :count],
            )
            generators[:3] = self._frame_generators[:, :count]

    def _applied(
        self,
        stage: int,
        queue_length: int,
        reference_abs: np.ndarray,
    ) -> None:
        lower_age, upper_age = self.generated.bounds.delay_step_bounds
        first = queue_length + stage - upper_age
        last = queue_length + stage - lower_age
        count = last - first + 1
        center = self.prediction.applied_center[stage]
        reference = self.prediction.applied_reference[stage]
        radius = self.prediction.applied_radius[stage]
        center.fill(0.0)
        reference.fill(0.0)
        for index in range(first, last + 1):
            center += self._history_center[index]
            reference += self._history_reference[index]
        center /= count
        reference /= count
        radius.fill(0.0)
        for index in range(first, last + 1):
            generator_count = int(self._history_count[index])
            np.subtract(
                self._history_center[index],
                center,
                out=self._applied_residual,
            )
            np.abs(self._applied_residual, out=self._applied_residual)
            np.subtract(
                self._history_reference[index],
                reference,
                out=self._applied_difference,
            )
            np.abs(self._applied_difference, out=self._applied_difference)
            np.matmul(
                self._applied_difference,
                reference_abs,
                out=self._applied_support,
            )
            self._applied_residual += self._applied_support
            generators = self._history_generators[index, :, :generator_count]
            self._generator_radii(
                generators,
                generator_count,
                self._applied_support,
            )
            self._applied_residual += self._applied_support
            self._applied_residual += self._history_radius[index]
            np.maximum(radius, self._applied_residual, out=radius)

    def _populate_geometry(self, output: Prediction, cell: GainCell) -> None:
        geometry = self.generated.geometry
        reference_abs = self._reference_abs
        np.matmul(
            self._frame_rotation_abs,
            cell.swept_position_remainder_abs_m,
            out=self._frame_position_remainder,
        )
        np.matmul(
            self._frame_rotation_abs,
            cell.contact_velocity_remainder_abs_m_s,
            out=self._frame_velocity_remainder,
        )
        for stage in range(PREDICTION_STAGES + 1):
            _rotation_derivatives(
                output.state_center[stage, 3:6],
                self._stage_rotations[stage],
            )
        self._swept_points(
            output,
            geometry.body_b_m,
            self._body_norms,
            output.body_center,
            output.body_reference,
            output.body_radius,
            reference_abs,
            self._frame_position_remainder,
        )
        np.take(
            output.body_center,
            self._contact_indices,
            axis=1,
            out=output.contact_center,
        )
        np.take(
            output.body_reference,
            self._contact_indices,
            axis=1,
            out=output.contact_reference,
        )
        np.take(
            output.body_radius,
            self._contact_indices,
            axis=1,
            out=output.contact_radius,
        )
        np.take(
            output.body_center,
            self._footprint_indices,
            axis=1,
            out=output.footprint_center,
        )
        np.take(
            output.body_reference,
            self._footprint_indices,
            axis=1,
            out=output.footprint_reference,
        )
        np.take(
            output.body_radius,
            self._footprint_indices,
            axis=1,
            out=output.footprint_radius,
        )
        self._swept_velocities(
            output,
            reference_abs,
            self._frame_velocity_remainder,
        )

    def _swept_points(
        self,
        output: Prediction,
        points: npt.ArrayLike,
        point_norms: np.ndarray,
        centers: np.ndarray,
        references: np.ndarray,
        radii: np.ndarray,
        reference_abs: np.ndarray,
        swept_remainder: np.ndarray,
    ) -> None:
        points_array = np.asarray(points, dtype=float).reshape(-1, 3)
        inflation = (
            self.generated.bounds.body_inflation_m
            + self.generated.bounds.mission_position_error_abs_m
        )
        attitude_error = self.generated.bounds.mission_attitude_error_abs_rad
        point_count = points_array.shape[0]
        first = 0
        second = 1
        self._point_affine_into(output, 0, points_array, first, reference_abs)
        for stage in range(PREDICTION_STAGES):
            self._point_affine_into(
                output,
                stage + 1,
                points_array,
                second,
                reference_abs,
            )
            np.add(
                self._point_center[first, :point_count],
                self._point_center[second, :point_count],
                out=centers[stage],
            )
            centers[stage] *= 0.5
            np.add(
                self._point_reference[first, :point_count],
                self._point_reference[second, :point_count],
                out=references[stage],
            )
            references[stage] *= 0.5
            for point in range(point_count):
                attitude = point_norms[point] * attitude_error
                for axis in range(3):
                    first_residual = self._point_radius[first, point, axis] + abs(
                        self._point_center[first, point, axis]
                        - centers[stage, point, axis]
                    )
                    second_residual = self._point_radius[second, point, axis] + abs(
                        self._point_center[second, point, axis]
                        - centers[stage, point, axis]
                    )
                    for reference_axis in range(3):
                        first_residual += (
                            abs(
                                self._point_reference[
                                    first,
                                    point,
                                    axis,
                                    reference_axis,
                                ]
                                - references[stage, point, axis, reference_axis]
                            )
                            * reference_abs[reference_axis]
                        )
                        second_residual += (
                            abs(
                                self._point_reference[
                                    second,
                                    point,
                                    axis,
                                    reference_axis,
                                ]
                                - references[stage, point, axis, reference_axis]
                            )
                            * reference_abs[reference_axis]
                        )
                    radii[stage, point, axis] = (
                        max(first_residual, second_residual)
                        + inflation
                        + attitude
                        + swept_remainder[axis]
                    )
            first, second = second, first

    def _swept_velocities(
        self,
        output: Prediction,
        reference_abs: np.ndarray,
        velocity_remainder: np.ndarray,
    ) -> None:
        points = np.asarray(
            self.generated.geometry.contact_b_m,
            dtype=float,
        ).reshape(-1, 3)
        centers = output.contact_velocity_center
        references = output.contact_velocity_reference
        radii = output.contact_velocity_radius
        point_count = points.shape[0]
        attitude_error = self.generated.bounds.mission_attitude_error_abs_rad
        first = 0
        second = 1
        self._velocity_affine_into(output, 0, points, first, reference_abs)
        for stage in range(PREDICTION_STAGES):
            self._velocity_affine_into(
                output,
                stage + 1,
                points,
                second,
                reference_abs,
            )
            np.add(
                self._point_center[first, :point_count],
                self._point_center[second, :point_count],
                out=centers[stage],
            )
            centers[stage] *= 0.5
            np.add(
                self._point_reference[first, :point_count],
                self._point_reference[second, :point_count],
                out=references[stage],
            )
            references[stage] *= 0.5
            for point in range(point_count):
                point_norm = self._contact_norms[point]
                speed = self.generated.bounds.airspeed_m_s[1]
                speed += self.generated.bounds.body_rate_abs_max_rad_s * point_norm
                for axis in range(3):
                    first_residual = self._point_radius[first, point, axis] + abs(
                        self._point_center[first, point, axis]
                        - centers[stage, point, axis]
                    )
                    second_residual = self._point_radius[second, point, axis] + abs(
                        self._point_center[second, point, axis]
                        - centers[stage, point, axis]
                    )
                    for reference_axis in range(3):
                        first_residual += (
                            abs(
                                self._point_reference[
                                    first,
                                    point,
                                    axis,
                                    reference_axis,
                                ]
                                - references[stage, point, axis, reference_axis]
                            )
                            * reference_abs[reference_axis]
                        )
                        second_residual += (
                            abs(
                                self._point_reference[
                                    second,
                                    point,
                                    axis,
                                    reference_axis,
                                ]
                                - references[stage, point, axis, reference_axis]
                            )
                            * reference_abs[reference_axis]
                        )
                    radii[stage, point, axis] = (
                        max(first_residual, second_residual)
                        + speed * attitude_error
                        + velocity_remainder[axis]
                    )
            first, second = second, first

    def _point_affine_into(
        self,
        prediction: Prediction,
        stage: int,
        points: np.ndarray,
        slot: int,
        reference_abs: np.ndarray,
    ) -> None:
        state = prediction.state_center[stage]
        rotation = self._stage_rotations[stage]
        point_count = points.shape[0]
        center = self._point_center[slot, :point_count]
        jacobian = self._point_jacobian[:point_count]
        jacobian.fill(0.0)
        for point in range(point_count):
            for axis in range(3):
                center[point, axis] = state[axis]
                jacobian[point, axis, axis] = 1.0
                for body_axis in range(3):
                    center[point, axis] += (
                        rotation[0, axis, body_axis] * points[point, body_axis]
                    )
                for angle in range(3):
                    for body_axis in range(3):
                        jacobian[point, axis, 3 + angle] += (
                            rotation[1 + angle, axis, body_axis]
                            * points[point, body_axis]
                        )
        self._project_geometry(
            prediction,
            stage,
            slot,
            point_count,
            reference_abs,
            points,
            self._body_norms,
            False,
        )

    def _velocity_affine_into(
        self,
        prediction: Prediction,
        stage: int,
        points: np.ndarray,
        slot: int,
        reference_abs: np.ndarray,
    ) -> None:
        state = prediction.state_center[stage]
        rotation = self._stage_rotations[stage]
        point_count = points.shape[0]
        center = self._point_center[slot, :point_count]
        jacobian = self._point_jacobian[:point_count]
        jacobian.fill(0.0)
        for point in range(point_count):
            x, y, z = points[point]
            wx, wy, wz = state[9:12]
            rigid0 = state[6] + wy * z - wz * y
            rigid1 = state[7] + wz * x - wx * z
            rigid2 = state[8] + wx * y - wy * x
            for axis in range(3):
                center[point, axis] = (
                    rotation[0, axis, 0] * rigid0
                    + rotation[0, axis, 1] * rigid1
                    + rotation[0, axis, 2] * rigid2
                )
                jacobian[point, axis, 6:9] = rotation[0, axis]
                for angle in range(3):
                    jacobian[point, axis, 3 + angle] = (
                        rotation[1 + angle, axis, 0] * rigid0
                        + rotation[1 + angle, axis, 1] * rigid1
                        + rotation[1 + angle, axis, 2] * rigid2
                    )
                jacobian[point, axis, 9] = (
                    -rotation[0, axis, 1] * z + rotation[0, axis, 2] * y
                )
                jacobian[point, axis, 10] = (
                    rotation[0, axis, 0] * z - rotation[0, axis, 2] * x
                )
                jacobian[point, axis, 11] = (
                    -rotation[0, axis, 0] * y + rotation[0, axis, 1] * x
                )
        self._project_geometry(
            prediction,
            stage,
            slot,
            point_count,
            reference_abs,
            points,
            self._contact_norms,
            True,
        )

    def _project_geometry(
        self,
        prediction: Prediction,
        stage: int,
        slot: int,
        point_count: int,
        reference_abs: np.ndarray,
        points: np.ndarray,
        point_norms: np.ndarray,
        velocity: bool,
    ) -> None:
        jacobian = self._point_jacobian[:point_count]
        reference = self._point_reference[slot, :point_count]
        np.matmul(
            jacobian,
            prediction.state_reference[stage],
            out=reference,
        )
        count = prediction.generator_count[stage]
        generators = self._point_generators[:point_count, :, :count]
        np.matmul(
            jacobian,
            prediction.state_generators[stage, :, :count],
            out=generators,
        )
        generator_abs = self._point_generator_abs[:point_count, :, :count]
        np.abs(generators, out=generator_abs)
        radius = self._point_radius[slot, :point_count]
        np.add.reduce(generator_abs, axis=2, out=radius)

        angle_radius = 0.0
        for angle in range(3, 6):
            generator_abs = self._angle_generator_abs[:count]
            np.abs(
                prediction.state_generators[stage, angle, :count],
                out=generator_abs,
            )
            value = float(np.add.reduce(generator_abs))
            reference_value = 0.0
            for reference_axis in range(3):
                reference_value += (
                    abs(
                        prediction.state_reference[
                            stage,
                            angle,
                            reference_axis,
                        ]
                    )
                    * reference_abs[reference_axis]
                )
            angle_radius = max(angle_radius, value + reference_value)

        if velocity:
            state = prediction.state_center[stage]
            velocity_norm = sqrt(float(state[6:9] @ state[6:9]))
            rate_norm = sqrt(float(state[9:12] @ state[9:12]))
            maximum_point_norm = float(np.max(point_norms))
            nonlinear = (
                velocity_norm + rate_norm * maximum_point_norm
            ) * angle_radius**2
            radius += nonlinear
            return

        for point in range(point_count):
            nonlinear = point_norms[point] * angle_radius**2
            for axis in range(3):
                reference_magnitude = 0.0
                for reference_axis in range(3):
                    reference_magnitude = max(
                        reference_magnitude,
                        float(abs(reference[point, axis, reference_axis])),
                    )
                radius[point, axis] += nonlinear + 1.0e-12 * reference_magnitude


def _point_indices(body: np.ndarray, subset: np.ndarray) -> np.ndarray:
    return np.asarray(
        [np.flatnonzero(np.all(body == point, axis=1))[0] for point in subset],
        dtype=int,
    )


def _rotation_derivatives(attitude: np.ndarray, result: np.ndarray) -> None:
    phi, theta, psi = attitude
    c_phi, s_phi = cos(phi), sin(phi)
    c_theta, s_theta = cos(theta), sin(theta)
    c_psi, s_psi = cos(psi), sin(psi)
    rotation, d_phi, d_theta, d_psi = result

    rotation[:] = (
        (
            c_theta * c_psi,
            s_phi * s_theta * c_psi - c_phi * s_psi,
            c_phi * s_theta * c_psi + s_phi * s_psi,
        ),
        (
            -c_theta * s_psi,
            -s_phi * s_theta * s_psi - c_phi * c_psi,
            -c_phi * s_theta * s_psi + s_phi * c_psi,
        ),
        (s_theta, -s_phi * c_theta, -c_phi * c_theta),
    )
    d_phi[:] = (
        (
            0.0,
            c_phi * s_theta * c_psi + s_phi * s_psi,
            -s_phi * s_theta * c_psi + c_phi * s_psi,
        ),
        (
            0.0,
            -c_phi * s_theta * s_psi + s_phi * c_psi,
            s_phi * s_theta * s_psi + c_phi * c_psi,
        ),
        (0.0, -c_phi * c_theta, s_phi * c_theta),
    )
    d_theta[:] = (
        (-s_theta * c_psi, s_phi * c_theta * c_psi, c_phi * c_theta * c_psi),
        (s_theta * s_psi, -s_phi * c_theta * s_psi, -c_phi * c_theta * s_psi),
        (c_theta, s_phi * s_theta, c_phi * s_theta),
    )
    d_psi[:] = (
        (
            -c_theta * s_psi,
            -s_phi * s_theta * s_psi - c_phi * c_psi,
            -c_phi * s_theta * s_psi + s_phi * c_psi,
        ),
        (
            -c_theta * c_psi,
            -s_phi * s_theta * c_psi + c_phi * s_psi,
            -c_phi * s_theta * c_psi - s_phi * s_psi,
        ),
        (0.0, 0.0, 0.0),
    )


def _generate_aircraft(
    aircraft: Aircraft,
    geometry: RigidBodyGeometry,
    bounds: Bounds,
    anchor: npt.ArrayLike,
    control_anchor: npt.ArrayLike,
    cell_verifier: Callable[[GainCell], _CellVerification | None] | None = None,
) -> _AircraftModel:

    domain_anchor = np.asarray(anchor, dtype=float).reshape(15)
    control = np.asarray(control_anchor, dtype=float).reshape(3)
    state_scale = _state_scale(aircraft, bounds, domain_anchor)
    reference_center = 0.5 * (aircraft.control_lower_rad + aircraft.control_upper_rad)
    reference_scale = 0.5 * (aircraft.control_upper_rad - aircraft.control_lower_rad)
    control_scale = np.maximum(
        np.abs(aircraft.control_lower_rad - control),
        np.abs(aircraft.control_upper_rad - control),
    )
    size = INTRINSIC_FEEDBACK_INDICES.size
    pending = [(-np.ones(size), np.ones(size), 0)]
    cells: list[GainCell] = []
    rejected: list[RejectedCell] = []
    while pending:
        lower, upper, depth = pending.pop()
        center = 0.5 * (lower + upper)
        state = domain_anchor.copy()
        state[INTRINSIC_FEEDBACK_INDICES] += (
            state_scale[INTRINSIC_FEEDBACK_INDICES] * center
        )
        try:
            cell = _gain_cell(
                aircraft,
                bounds,
                state,
                control,
                state_scale,
                control_scale,
                lower - center,
                upper - center,
            )
        except (ValueError, np.linalg.LinAlgError):
            cell = None
        accepted = cell is not None and cell_verifier is None
        if cell is not None and cell_verifier is not None:
            verification = cell_verifier(cell)
            accepted = verification is not None
            if verification is not None:
                cell = replace(
                    cell,
                    stage_remainder_abs=np.maximum(
                        bounds.stage_remainder_abs,
                        verification.stage_remainder_abs,
                    ),
                    swept_position_remainder_abs_m=(
                        verification.swept_position_remainder_abs_m
                    ),
                    contact_velocity_remainder_abs_m_s=(
                        verification.contact_velocity_remainder_abs_m_s
                    ),
                )
        if accepted and cell is not None:
            cells.append(cell)
            continue
        widths = upper - lower
        split = int(np.argmax(widths))
        if depth == MAX_GAIN_CELL_DEPTH or widths[split] <= MIN_GAIN_CELL_WIDTH:
            rejected.append(RejectedCell(state, lower - center, upper - center))
            continue
        midpoint = 0.5 * (lower[split] + upper[split])
        left_upper = upper.copy()
        left_upper[split] = midpoint
        right_lower = lower.copy()
        right_lower[split] = midpoint
        pending.append((right_lower, upper, depth + 1))
        pending.append((lower, left_upper, depth + 1))
    values = dict(
        aircraft=aircraft,
        geometry=geometry,
        bounds=bounds,
        state_scale=state_scale,
        domain_anchor=domain_anchor,
        reference_center=reference_center,
        reference_scale=reference_scale,
        cells=tuple(cells),
        rejected_cells=tuple(rejected),
    )
    if cell_verifier is None:
        return _AircraftModel(**values)
    return GeneratedAircraft(**values)


def _gain_cell(
    aircraft: Aircraft,
    bounds: Bounds,
    state: np.ndarray,
    control: np.ndarray,
    state_scale: np.ndarray,
    control_scale: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> GainCell:
    state_matrix, control_matrix, offset = _linear_model(
        aircraft,
        bounds,
        state,
        control,
        state_scale,
        control_scale,
    )
    gain = _lqr_gain(
        state_matrix,
        control_matrix,
        state_scale,
        control_scale,
    )
    flow_generators = _flow_generators(aircraft, bounds, state, control)
    model_generators = _model_generators(aircraft, bounds, state, control)
    cell = GainCell(
        lower,
        upper,
        state,
        control,
        state_matrix,
        control_matrix,
        offset,
        gain,
        flow_generators,
        model_generators,
        bounds.stage_remainder_abs,
        np.zeros(3),
        np.zeros(3),
    )
    return cell


def _linear_model(
    aircraft: Aircraft,
    bounds: Bounds,
    state: np.ndarray,
    control: np.ndarray,
    state_scale: np.ndarray,
    control_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint = bounds.flow.joint_zonotope(aircraft.strip_table.r_b_m)
    flow = joint.center.reshape(-1, 3)
    density = 0.5 * sum(bounds.density_kg_m3)

    def step(state_value: np.ndarray, control_value: np.ndarray) -> np.ndarray:
        return _rk4(
            aircraft,
            state_value,
            control_value,
            flow[0],
            flow[1:],
            density,
        )

    state_matrix = np.empty((15, 15))
    for index in range(15):
        delta = max(1.0e-7, 1.0e-5 * state_scale[index])
        offset = np.zeros(15)
        offset[index] = delta
        state_matrix[:, index] = (
            step(state + offset, control) - step(state - offset, control)
        ) / (2.0 * delta)
    control_matrix = np.empty((15, 3))
    for index in range(3):
        delta = max(1.0e-7, 1.0e-5 * control_scale[index])
        offset = np.zeros(3)
        offset[index] = delta
        control_matrix[:, index] = (
            step(state, control + offset) - step(state, control - offset)
        ) / (2.0 * delta)
    successor = step(state, control)
    affine_offset = successor - state_matrix @ state - control_matrix @ control
    return state_matrix, control_matrix, affine_offset


def _flow_generators(
    aircraft: Aircraft,
    bounds: Bounds,
    state: np.ndarray,
    control: np.ndarray,
) -> np.ndarray:
    joint = bounds.flow.joint_zonotope(aircraft.strip_table.r_b_m)
    center = joint.center.reshape(-1, 3)
    density = 0.5 * sum(bounds.density_kg_m3)
    generators = np.empty((15, joint.generators.shape[1]))
    for index, column in enumerate(joint.generators.T):
        delta = column.reshape(center.shape)
        upper = _rk4(
            aircraft,
            state,
            control,
            center[0] + delta[0],
            center[1:] + delta[1:],
            density,
        )
        lower = _rk4(
            aircraft,
            state,
            control,
            center[0] - delta[0],
            center[1:] - delta[1:],
            density,
        )
        generators[:, index] = 0.5 * (upper - lower)
    return generators


def _model_generators(
    aircraft: Aircraft,
    bounds: Bounds,
    state: np.ndarray,
    control: np.ndarray,
) -> np.ndarray:
    joint = bounds.flow.joint_zonotope(aircraft.strip_table.r_b_m)
    flow = joint.center.reshape(-1, 3)
    density_center = 0.5 * sum(bounds.density_kg_m3)
    density_lower = _rk4(
        aircraft,
        state,
        control,
        flow[0],
        flow[1:],
        bounds.density_kg_m3[0],
    )
    density_upper = _rk4(
        aircraft,
        state,
        control,
        flow[0],
        flow[1:],
        bounds.density_kg_m3[1],
    )
    columns = [0.5 * (density_upper - density_lower)]
    force, moment = aircraft.aero_loads_local_flow(
        state,
        flow[0],
        flow[1:],
        density_center,
    )
    coefficient_radius = max(
        abs(bounds.aerodynamic_scale[0] - 1.0),
        abs(bounds.aerodynamic_scale[1] - 1.0),
    )
    coefficient = np.zeros(15)
    coefficient[6:9] = FAST_PERIOD_S * coefficient_radius * force / aircraft.mass_kg
    coefficient[9:12] = (
        FAST_PERIOD_S * coefficient_radius * aircraft.inertia_inv_b @ moment
    )
    columns.append(coefficient)
    inverse_mass_radius = max(
        abs(1.0 / bounds.mass_kg[0] - 1.0 / aircraft.mass_kg),
        abs(1.0 / bounds.mass_kg[1] - 1.0 / aircraft.mass_kg),
    )
    mass = np.zeros(15)
    mass[6:9] = FAST_PERIOD_S * force * inverse_mass_radius
    columns.append(mass)
    for index in range(3):
        column = np.zeros(15)
        column[6 + index] = (
            FAST_PERIOD_S * bounds.force_residual_abs_n[index] / bounds.mass_kg[0]
        )
        columns.append(column)
    for index in range(3):
        column = np.zeros(15)
        column[9:12] = (
            FAST_PERIOD_S
            * aircraft.inertia_inv_b[:, index]
            * bounds.moment_residual_abs_n_m[index]
        )
        columns.append(column)
    axes = np.eye(3)
    for index in range(3):
        column = np.zeros(15)
        shifted_moment = -np.cross(
            axes[index] * bounds.cg_residual_abs_m[index],
            force,
        )
        column[9:12] = FAST_PERIOD_S * aircraft.inertia_inv_b @ shifted_moment
        columns.append(column)
    inertia_radius = float(
        np.linalg.norm(aircraft.inertia_inv_b, ord=np.inf)
        * np.sum(bounds.inertia_residual_abs_kg_m2)
        * (
            np.linalg.norm(state[9:12]) ** 2
            + np.linalg.norm(aircraft.inertia_inv_b @ moment)
        )
    )
    for index in range(3):
        column = np.zeros(15)
        column[9 + index] = FAST_PERIOD_S * inertia_radius
        columns.append(column)
    command_span = aircraft.control_upper_rad - aircraft.control_lower_rad
    nominal_inverse_tau = 1.0 / np.asarray(
        aircraft.config.actuator_tau_s,
        dtype=float,
    )
    tau_radius = np.maximum(
        np.abs(1.0 / bounds.actuator_tau_lower_s - nominal_inverse_tau),
        np.abs(1.0 / bounds.actuator_tau_upper_s - nominal_inverse_tau),
    )
    for index in range(3):
        column = np.zeros(15)
        column[12 + index] = FAST_PERIOD_S * command_span[index] * tau_radius[index]
        columns.append(column)
    return np.column_stack(columns)


def _lqr_gain(
    state_matrix: np.ndarray,
    control_matrix: np.ndarray,
    state_scale: np.ndarray,
    control_scale: np.ndarray,
) -> np.ndarray:
    indices = INTRINSIC_FEEDBACK_INDICES
    scale = state_scale[indices]
    state_normalized = (
        state_matrix[np.ix_(indices, indices)] * scale[None, :] / scale[:, None]
    )
    control_normalized = (
        control_matrix[indices] * control_scale[None, :] / scale[:, None]
    )
    state_weight = np.diag(LQR_STATE_WEIGHT)
    control_weight = np.diag(LQR_CONTROL_WEIGHT)
    value = state_weight.copy()
    for _ in range(1000):
        hessian = control_weight + control_normalized.T @ value @ control_normalized
        gain = -np.linalg.solve(
            hessian,
            control_normalized.T @ value @ state_normalized,
        )
        closed = state_normalized + control_normalized @ gain
        updated = (
            state_weight + gain.T @ control_weight @ gain + closed.T @ value @ closed
        )
        if np.max(np.abs(updated - value)) <= 1.0e-12:
            value = updated
            break
        value = updated
    else:
        raise ValueError("discrete LQR iteration did not converge")
    hessian = control_weight + control_normalized.T @ value @ control_normalized
    normalized = -np.linalg.solve(
        hessian,
        control_normalized.T @ value @ state_normalized,
    )
    closed = state_normalized + control_normalized @ normalized
    if np.max(np.abs(np.linalg.eigvals(closed))) >= 1.0:
        raise ValueError("discrete LQR gain does not stabilize the local model")
    gain = np.zeros((3, 15))
    gain[:, indices] = control_scale[:, None] * normalized / scale[None, :]
    return gain


def _state_scale(
    aircraft: Aircraft,
    bounds: Bounds,
    anchor: np.ndarray,
) -> np.ndarray:
    flow = bounds.flow.joint_zonotope(aircraft.strip_table.r_b_m).interval_hull()
    flow_lower = flow.lower[:3]
    flow_upper = flow.upper[:3]
    velocity_lower = flow_lower - bounds.airspeed_m_s[1]
    velocity_upper = flow_upper + bounds.airspeed_m_s[1]
    velocity_lower[0] = flow_lower[0] + bounds.airspeed_m_s[0]
    attitude_lower = np.array((-bounds.roll_abs_max_rad, -bounds.pitch_abs_max_rad))
    attitude_upper = -attitude_lower
    rate_lower = np.full(3, -bounds.body_rate_abs_max_rad_s)
    rate_upper = -rate_lower
    scale = np.ones(15)
    scale[3:5] = np.maximum(
        np.abs(attitude_lower - anchor[3:5]),
        np.abs(attitude_upper - anchor[3:5]),
    )
    scale[5] = np.pi
    scale[6:9] = np.maximum(
        np.abs(velocity_lower - anchor[6:9]),
        np.abs(velocity_upper - anchor[6:9]),
    )
    scale[9:12] = np.maximum(
        np.abs(rate_lower - anchor[9:12]),
        np.abs(rate_upper - anchor[9:12]),
    )
    scale[12:15] = np.maximum(
        np.abs(aircraft.control_lower_rad - anchor[12:15]),
        np.abs(aircraft.control_upper_rad - anchor[12:15]),
    )
    return scale


def _rk4(
    aircraft: Aircraft,
    state: np.ndarray,
    control: np.ndarray,
    center_flow: np.ndarray,
    strip_flow: np.ndarray,
    density: float,
) -> np.ndarray:
    def derivative(value: np.ndarray) -> np.ndarray:
        return aircraft.derivative_local_flow(
            value,
            control,
            center_flow,
            strip_flow,
            density,
        )

    step = FAST_PERIOD_S
    k1 = derivative(state)
    k2 = derivative(state + 0.5 * step * k1)
    k3 = derivative(state + 0.5 * step * k2)
    k4 = derivative(state + step * k3)
    return state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _set_array(
    instance: object,
    name: str,
    value: npt.ArrayLike,
    shape: tuple[int, ...],
) -> None:
    array = np.asarray(value, dtype=float).reshape(shape).copy()
    array.flags.writeable = False
    object.__setattr__(instance, name, array)
