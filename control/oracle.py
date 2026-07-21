"""Offline validated propagation for generated predictors."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from math import pi, radians
from typing import TypedDict, cast

import numpy as np
import numpy.typing as npt

from control.flow import (
    CENTER_FLOW_GENERATOR_COUNT,
    SHARED_FLOW_GENERATOR_COUNT,
    FlowBounds,
    JointFlow,
)
from control.interval import AffineForm, Interval, Zonotope
from control.missions import GateMission, LandingMission, Mission
from control.predictor import (
    INTRINSIC_FEEDBACK_INDICES,
    _AircraftModel,
    FastPredictor,
    GainCell,
    GeneratedAircraft,
    Prediction,
    _CellVerification,
    _generate_aircraft,
    _state_scale,
)
from control.uncertainty import Bounds, FAST_PERIOD_S, PREDICTION_STAGES
from models.aircraft import G_M_S2, SMOOTH_ABS_EPS, Aircraft, flat_plate_coefficients
from models.geometry import RigidBodyGeometry

NUMERICAL_TOLERANCE = 64.0 * np.finfo(float).eps


class _Realization(TypedDict):
    state: np.ndarray
    queue_factors: np.ndarray
    flows: tuple[np.ndarray, ...]
    density: float
    aerodynamic_scale: float
    mass: float
    force_error: np.ndarray
    moment_error: np.ndarray
    cg: np.ndarray
    inertia: np.ndarray
    tau: np.ndarray
    measurement: np.ndarray
    command_error: np.ndarray
    delay_age: int


class OracleValidationError(RuntimeError):
    """Raised when validated propagation cannot enclose a stage."""


def generate_aircraft(
    aircraft: Aircraft,
    geometry: RigidBodyGeometry,
    bounds: Bounds,
    anchor: npt.ArrayLike,
    control_anchor: npt.ArrayLike,
) -> GeneratedAircraft:
    """Generate the gain schedule through complete nonlinear cell verification."""

    domain_anchor = np.asarray(anchor, dtype=float).reshape(15)
    state_scale = _state_scale(aircraft, bounds, domain_anchor)
    reference_center = 0.5 * (aircraft.control_lower_rad + aircraft.control_upper_rad)
    reference_scale = 0.5 * (aircraft.control_upper_rad - aircraft.control_lower_rad)

    def verify(cell: GainCell) -> _CellVerification | None:
        generated = _AircraftModel(
            aircraft,
            geometry,
            bounds,
            state_scale,
            domain_anchor,
            reference_center,
            reference_scale,
            (cell,),
        )
        return _verify_gain_cell(generated, cell)

    generated = _generate_aircraft(
        aircraft,
        geometry,
        bounds,
        domain_anchor,
        control_anchor,
        verify,
    )
    return cast(GeneratedAircraft, generated)


@dataclass(frozen=True)
class GeometryEnclosure:
    """Full-body, contact, footprint, and contact-velocity enclosures."""

    occupied: Interval
    contact: Interval
    footprint: Interval
    contact_velocity: Interval


@dataclass(frozen=True)
class OracleStage:
    """Validated boundary and continuous-time sets for one fast stage."""

    initial: Zonotope
    successor: Zonotope
    continuous_state: Interval
    boundary_geometry: GeometryEnclosure
    continuous_geometry: GeometryEnclosure
    issued_command: Interval
    applied_command: Interval
    durations_s: tuple[float, ...]
    joint_flow: Zonotope


@dataclass(frozen=True)
class OraclePrediction:
    """Ten-stage nonlinear enclosure aligned with the fast predictor."""

    initial: Zonotope
    initial_geometry: GeometryEnclosure
    stages: tuple[OracleStage, ...]

    def state_interval(self, stage: int) -> Interval:
        """Return the discrete state hull at a predictor stage."""

        if stage == 0:
            return self.initial.interval_hull()
        return self.stages[stage - 1].successor.interval_hull()

    def geometry(self, stage: int) -> GeometryEnclosure:
        """Return the discrete geometry enclosure at a predictor stage."""

        if stage == 0:
            return self.initial_geometry
        return self.stages[stage - 1].boundary_geometry


@dataclass(frozen=True)
class RemainderBounds:
    """Generated one-step nonlinear and floating-point remainder bounds."""

    nonlinear_abs: npt.ArrayLike
    numerical_abs: npt.ArrayLike

    def __post_init__(self) -> None:
        _set_array(
            self,
            "nonlinear_abs",
            self.nonlinear_abs,
            (PREDICTION_STAGES, 15),
        )
        _set_array(
            self,
            "numerical_abs",
            self.numerical_abs,
            (PREDICTION_STAGES, 15),
        )

    def within(self, bounds: Bounds) -> bool:
        """Return whether one declared stage bound covers every stage."""

        return bool(
            np.all(self.nonlinear_abs <= bounds.nonlinear_remainder_abs)
            and np.all(self.numerical_abs <= bounds.numerical_remainder_abs)
        )


@dataclass
class NonlinearOracle:
    """Validated affine/Picard oracle for one generated aircraft."""

    generated: _AircraftModel
    substeps: int = 4
    picard_iterations: int = 16
    max_subdivisions: int = 4
    max_generators: int = 120
    flow_bounds: FlowBounds | None = None
    _flow: JointFlow = field(init=False, repr=False)
    _flow_form: AffineForm = field(init=False, repr=False)
    _flow_noncenter_radius: np.ndarray = field(init=False, repr=False)
    _flow_shape: tuple[int, int] = field(init=False, repr=False)
    _locations: Interval = field(init=False, repr=False)
    _spans: Interval = field(init=False, repr=False)
    _projection: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        table = self.aircraft.strip_table
        flow = self.bounds.flow if self.flow_bounds is None else self.flow_bounds
        self._flow = flow.joint_flow(table.r_b_m)
        self._flow_form = self._flow.affine_form()
        noncenter_generators = self._flow_form.generators[
            ...,
            CENTER_FLOW_GENERATOR_COUNT:,
        ].reshape(-1, self._flow_form.generator_count - CENTER_FLOW_GENERATOR_COUNT)
        self._flow_noncenter_radius = Zonotope(
            np.zeros(self._flow_form.center.size),
            noncenter_generators,
        ).radius.reshape(self._flow_form.center.shape)
        self._flow_shape = (table.r_b_m.shape[0] + 1, 3)
        self._locations = Interval.point(table.r_b_m)
        self._spans = Interval.point(table.span_axis_b)
        self._projection = (
            np.eye(3)[None, :, :]
            - table.span_axis_b[:, :, None] * table.span_axis_b[:, None, :]
        )

    @property
    def aircraft(self) -> Aircraft:
        """Return the generated aircraft model."""

        return self.generated.aircraft

    @property
    def bounds(self) -> Bounds:
        """Return the declared uncertainty bounds."""

        return self.generated.bounds

    @property
    def joint_flow(self) -> Zonotope:
        """Return the exact shared centre, gradient, and strip flow set."""

        return self._flow.zonotope()

    def propagate(
        self,
        initial: Interval | Zonotope,
        issued_queue: npt.ArrayLike | tuple[Interval, ...],
        cell: GainCell,
        reference: npt.ArrayLike | Interval,
    ) -> OraclePrediction:
        """Validate ten delayed nonlinear stages from predictor stage sets."""

        state = (
            initial
            if isinstance(initial, Zonotope)
            else Zonotope.from_interval(initial)
        )
        queue = self._command_queue(issued_queue)
        reference_box = (
            reference
            if isinstance(reference, Interval)
            else Interval.point(np.asarray(reference, dtype=float).reshape(3))
        )
        cell_index = next(
            index
            for index, generated_cell in enumerate(self.generated.cells)
            if generated_cell is cell
        )
        queue_centers = np.stack([command.center for command in queue])
        queue_radii = np.stack([command.radius for command in queue])
        fast = FastPredictor(self.generated).predict(
            state,
            queue_centers,
            cell_index,
            issued_queue_radius=queue_radii,
            local_flow=(
                self.bounds.flow if self.flow_bounds is None else self.flow_bounds
            ),
        )
        initial_geometry = self._geometry(state.interval_hull())
        stages = []
        command_error = self.bounds.command_error_abs_rad
        history = [
            (command + Interval(-command_error, command_error)).clip(
                self.aircraft.control_lower_rad,
                self.aircraft.control_upper_rad,
            )
            for command in queue
        ]
        for stage_index in range(PREDICTION_STAGES):
            if stage_index:
                state = _prediction_state(fast, stage_index, reference_box)
            nominal = reference_box.affine_map(
                fast.state_reference[stage_index],
                fast.state_center[stage_index],
            )
            issued = self._issued_command(
                state.interval_hull(),
                nominal,
                cell,
                reference_box,
            )
            history.append(issued)
            applied = self._applied_command(history, stage_index)
            stage = self._propagate_stage(state, issued, applied)
            stages.append(stage)
        return OraclePrediction(
            stages[0].initial,
            initial_geometry,
            tuple(stages),
        )

    def remainder_bounds(
        self,
        initial: Interval | Zonotope,
        issued_queue: npt.ArrayLike | tuple[Interval, ...],
        cell: GainCell,
        reference: npt.ArrayLike | Interval,
    ) -> RemainderBounds:
        """Generate complete-box local remainder bounds for every stage."""

        result = self.propagate(initial, issued_queue, cell, reference)
        state = (
            initial
            if isinstance(initial, Zonotope)
            else Zonotope.from_interval(initial)
        )
        queue = self._command_queue(issued_queue)
        reference_box = (
            reference
            if isinstance(reference, Interval)
            else Interval.point(np.asarray(reference, dtype=float).reshape(3))
        )
        cell_index = next(
            index
            for index, generated_cell in enumerate(self.generated.cells)
            if generated_cell is cell
        )
        fast = FastPredictor(self.generated).predict(
            state,
            np.stack([command.center for command in queue]),
            cell_index,
            issued_queue_radius=np.stack([command.radius for command in queue]),
            local_flow=(
                self.bounds.flow if self.flow_bounds is None else self.flow_bounds
            ),
        )
        nonlinear = np.empty((PREDICTION_STAGES, 15))
        numerical = np.empty_like(nonlinear)
        for index, stage in enumerate(result.stages):
            state_box = stage.initial.interval_hull()
            nonlinear[index] = self._flow_remainder(
                fast,
                index,
                reference_box,
                stage.applied_command,
                cell,
            )
            numerical[index] = _roundoff_bound(
                cell,
                state_box,
                stage.applied_command,
            )
        return RemainderBounds(nonlinear, numerical)

    def _flow_remainder(
        self,
        prediction: Prediction,
        stage: int,
        reference: Interval,
        command: Interval,
        cell: GainCell,
    ) -> np.ndarray:
        """Bound one stage for every certified center-flow sub-box."""

        flow_count = cell.flow_generators.shape[1]
        per_stage = flow_count + 3 + 15
        base = int(prediction.generator_count[0])
        factor_count = stage * CENTER_FLOW_GENERATOR_COUNT
        basis = object()

        state_count = int(prediction.generator_count[stage])
        state_flow = np.zeros((15, factor_count))
        state_nonflow = np.ones(state_count, dtype=bool)
        for index in range(stage):
            source = base + index * per_stage
            target = index * CENTER_FLOW_GENERATOR_COUNT
            state_flow[
                :,
                target : target + CENTER_FLOW_GENERATOR_COUNT,
            ] = prediction.state_generators[
                stage,
                :,
                source : source + CENTER_FLOW_GENERATOR_COUNT,
            ]
            state_nonflow[source : source + CENTER_FLOW_GENERATOR_COUNT] = False
        state_reference = prediction.state_reference[stage] @ np.diag(reference.radius)
        state_generators = np.column_stack(
            (
                prediction.state_generators[stage, :, :state_count][
                    :,
                    state_nonflow,
                ],
                state_reference,
            )
        )
        state_center = (
            prediction.state_center[stage]
            + prediction.state_reference[stage] @ reference.center
        )
        state_form = AffineForm(
            state_center,
            state_flow,
            Zonotope(np.zeros(15), state_generators).radius,
            basis,
        )
        successor, durations = self._flow_affine_stage(state_form, command)

        next_count = int(prediction.generator_count[stage + 1])
        fast_flow = np.zeros((15, successor.generator_count))
        next_nonflow = np.ones(next_count, dtype=bool)
        for index in range(stage):
            source = base + index * per_stage
            target = index * CENTER_FLOW_GENERATOR_COUNT
            fast_flow[
                :,
                target : target + CENTER_FLOW_GENERATOR_COUNT,
            ] = prediction.state_generators[
                stage + 1,
                :,
                source : source + CENTER_FLOW_GENERATOR_COUNT,
            ]
            next_nonflow[source : source + CENTER_FLOW_GENERATOR_COUNT] = False
            next_nonflow[source + flow_count : source + flow_count + 3] = False
        source = base + stage * per_stage
        target = stage * CENTER_FLOW_GENERATOR_COUNT
        for duration in durations:
            fast_flow[
                :,
                target : target + CENTER_FLOW_GENERATOR_COUNT,
            ] = prediction.state_generators[
                stage + 1,
                :,
                source : source + CENTER_FLOW_GENERATOR_COUNT,
            ] * (duration / FAST_PERIOD_S)
            target += CENTER_FLOW_GENERATOR_COUNT
        next_nonflow[source : source + CENTER_FLOW_GENERATOR_COUNT] = False
        next_nonflow[source + flow_count : source + flow_count + 3] = False
        remainder_start = base + stage * per_stage + flow_count + 3
        next_nonflow[remainder_start : remainder_start + 15] = False
        next_reference = prediction.state_reference[stage + 1] @ np.diag(
            reference.radius
        )
        next_generators = np.column_stack(
            (
                prediction.state_generators[stage + 1, :, :next_count][
                    :,
                    next_nonflow,
                ],
                next_reference,
            )
        )
        nonflow_radius = Interval.point(np.abs(next_generators)).sum(axis=1).lower
        fast_center = (
            prediction.state_center[stage + 1]
            + prediction.state_reference[stage + 1] @ reference.center
        )
        fast_form = AffineForm(
            fast_center,
            fast_flow,
            np.zeros(15),
            successor.basis,
        )
        difference = (successor - fast_form).interval_hull()
        difference_abs = np.maximum(
            np.abs(difference.lower),
            np.abs(difference.upper),
        )
        required = (Interval.point(difference_abs) - nonflow_radius).upper
        return np.maximum(required, 0.0)

    def _flow_affine_stage(
        self,
        initial: AffineForm,
        command: Interval,
    ) -> tuple[AffineForm, tuple[float, ...]]:
        """Propagate a stage with fresh flow factors at every oracle step."""

        state = initial
        durations = []
        step_s = FAST_PERIOD_S / self.substeps
        for _ in range(self.substeps):
            state, pieces = self._flow_affine_adaptive(
                state,
                command,
                step_s,
                0,
            )
            durations.extend(pieces)
        return state, tuple(durations)

    def _flow_affine_adaptive(
        self,
        initial: AffineForm,
        command: Interval,
        dt_s: float,
        depth: int,
    ) -> tuple[AffineForm, tuple[float, ...]]:
        try:
            state, flow = self._append_flow(initial)
            return self._flow_affine_step(state, command, flow, dt_s), (dt_s,)
        except OracleValidationError:
            if depth >= self.max_subdivisions:
                raise
            midpoint, first = self._flow_affine_adaptive(
                initial,
                command,
                0.5 * dt_s,
                depth + 1,
            )
            successor, second = self._flow_affine_adaptive(
                midpoint,
                command,
                0.5 * dt_s,
                depth + 1,
            )
            return successor, first + second

    def _append_flow(
        self,
        state: AffineForm,
    ) -> tuple[AffineForm, AffineForm]:
        start = state.generator_count
        stop = start + CENTER_FLOW_GENERATOR_COUNT
        basis = object()
        state_generators = np.zeros(state.center.shape + (stop,))
        state_generators[..., :start] = state.generators
        extended = AffineForm(
            state.center,
            state_generators,
            state.remainder,
            basis,
        )
        flow_generators = np.zeros(self._flow_form.center.shape + (stop,))
        flow_generators[..., start:stop] = self._flow_form.generators[
            ...,
            :CENTER_FLOW_GENERATOR_COUNT,
        ]
        flow = AffineForm(
            self._flow_form.center,
            flow_generators,
            self._flow_noncenter_radius,
            basis,
        )
        return extended, flow

    def _flow_affine_step(
        self,
        initial: AffineForm,
        command: Interval,
        flow: AffineForm,
        dt_s: float,
    ) -> AffineForm:
        derivative, _ = self._validated_derivative(
            initial.interval_hull(),
            command,
            flow,
            dt_s,
        )
        return initial + derivative * dt_s

    def certify_fast_prediction(
        self,
        initial: Zonotope,
        issued_queue: tuple[Interval, ...],
        cell: GainCell,
        reference: Interval,
        fast: Prediction,
    ) -> bool:
        """Certify one complete-cell affine state and geometry prediction."""

        try:
            nonlinear = self.propagate(initial, issued_queue, cell, reference)
        except OracleValidationError:
            return False
        return _prediction_contains(fast, nonlinear, reference, True)

    def certify_terminal_event(
        self,
        initial: Interval | Zonotope,
        issued_queue: npt.ArrayLike | tuple[Interval, ...],
        cell: GainCell,
        reference: npt.ArrayLike | Interval,
        mission: Mission,
    ) -> bool:
        """Certify a robust full-body gate or first-contact event."""

        try:
            result = self.propagate(initial, issued_queue, cell, reference)
        except OracleValidationError:
            return False
        center_flow = self.joint_flow.interval_hull()[:3]
        domain = all(
            self._inside_domain(stage.continuous_state, center_flow)
            for stage in result.stages
        )
        if (
            not domain
            or not self._inductive(result)
            or not self._free_space(result, mission)
        ):
            return False
        if isinstance(mission, GateMission):
            return self._gate_event(result, mission)
        if isinstance(mission, LandingMission):
            return self._landing_event(result, mission)
        return False

    @staticmethod
    def _inductive(prediction: OraclePrediction) -> bool:
        return all(
            stage.successor.interval_hull().subset(
                prediction.stages[index + 1].initial.interval_hull()
            )
            for index, stage in enumerate(prediction.stages[:-1])
        )

    def _factor_count(self, initial: Zonotope) -> int:
        state_count = initial.generators.shape[1]
        queue_count = self.bounds.queue_length * 3
        flow_count = PREDICTION_STAGES * self._flow.generators.shape[1]
        measurement_count = PREDICTION_STAGES * 18
        return state_count + queue_count + flow_count + 24 + measurement_count + 1

    def _command_queue(
        self,
        queue: npt.ArrayLike | tuple[Interval, ...],
    ) -> tuple[Interval, ...]:
        if isinstance(queue, tuple) and all(
            isinstance(item, Interval) for item in queue
        ):
            return queue
        values = np.asarray(queue, dtype=float).reshape(self.bounds.queue_length, 3)
        return tuple(Interval.point(row) for row in values)

    def _issued_command(
        self,
        state: Interval,
        nominal: Interval,
        cell: GainCell,
        reference: Interval,
    ) -> Interval:
        command = (state - nominal).affine_map(cell.gain) + reference
        radius = np.abs(cell.gain) @ self.bounds.state_estimation_abs
        radius += self.bounds.command_error_abs_rad
        return (command + Interval(-radius, radius)).clip(
            self.aircraft.control_lower_rad,
            self.aircraft.control_upper_rad,
        )

    def _applied_command(
        self,
        history: list[Interval],
        stage: int,
    ) -> Interval:
        lower_age, upper_age = self.bounds.delay_step_bounds
        queue_length = self.bounds.queue_length
        first = queue_length + stage - upper_age
        last = queue_length + stage - lower_age
        command = history[first]
        for index in range(first + 1, last + 1):
            command = command.hull(history[index])
        return command

    def _propagate_stage(
        self,
        initial: Zonotope,
        issued: Interval,
        applied: Interval,
    ) -> OracleStage:
        state = initial
        continuous = []
        durations = []
        step_s = FAST_PERIOD_S / self.substeps
        for _ in range(self.substeps):
            for successor, enclosure, duration in self._adaptive_step(
                state,
                applied,
                step_s,
                0,
            ):
                state = successor
                continuous.append(enclosure)
                durations.append(duration)
        state_enclosure = _hull(continuous)
        continuous_geometry = self._geometry(state_enclosure)
        boundary_geometry = self._geometry(state.interval_hull())
        return OracleStage(
            initial,
            state,
            state_enclosure,
            boundary_geometry,
            continuous_geometry,
            issued,
            applied,
            tuple(durations),
            self.joint_flow,
        )

    def _adaptive_step(
        self,
        initial: Zonotope,
        command: Interval,
        dt_s: float,
        depth: int,
    ) -> Iterator[tuple[Zonotope, Interval, float]]:
        try:
            successor, continuous = self._step(initial, command, dt_s)
            yield successor, continuous, dt_s
        except OracleValidationError:
            if depth >= self.max_subdivisions:
                raise
            midpoint = initial
            for piece in self._adaptive_step(
                initial,
                command,
                0.5 * dt_s,
                depth + 1,
            ):
                midpoint = piece[0]
                yield piece
            yield from self._adaptive_step(
                midpoint,
                command,
                0.5 * dt_s,
                depth + 1,
            )

    def _step(
        self,
        initial: Zonotope,
        command: Interval,
        dt_s: float,
    ) -> tuple[Zonotope, Interval]:
        initial_box = initial.interval_hull()
        flow = self._flow.affine_form()
        derivative, continuous = self._validated_derivative(
            initial_box,
            command,
            flow,
            dt_s,
        )
        return self._validated_successor(initial, derivative, dt_s), continuous

    def _validated_derivative(
        self,
        initial: Interval,
        command: Interval,
        flow: AffineForm,
        dt_s: float,
    ) -> tuple[AffineForm, Interval]:
        derivative = self._affine_derivative(
            initial,
            command,
            flow[0],
            flow[1:],
        )
        time = Interval(0.0, dt_s)
        continuous = _inflate(
            initial.hull(initial + time * derivative.interval_hull()),
            0.25,
        )
        for _ in range(self.picard_iterations):
            derivative = self._affine_derivative(
                continuous,
                command,
                flow[0],
                flow[1:],
            )
            image = initial + time * derivative.interval_hull()
            if not self._inside_domain(image, flow[0].interval_hull()):
                raise OracleValidationError("reachable set left the hard domain")
            if image.subset(continuous):
                return derivative, continuous
            continuous = _inflate(continuous.hull(image), 0.25)
        raise OracleValidationError("Picard enclosure did not converge")

    def _validated_successor(
        self,
        initial: Zonotope,
        derivative: AffineForm,
        dt_s: float,
    ) -> Zonotope:
        delta = derivative * dt_s
        shifted = Interval.point(initial.center) + delta.center
        radius = _up_sum(shifted.radius, delta.remainder)
        initial_count = initial.generators.shape[1]
        generators = np.concatenate(
            (initial.generators, delta.generators, np.diag(radius)),
            axis=1,
        )
        successor = Zonotope(shifted.center, generators)
        shared_count = min(
            SHARED_FLOW_GENERATOR_COUNT,
            delta.generator_count,
        )
        protected = tuple(range(initial_count, initial_count + shared_count))
        return _reduce(successor, self.max_generators, protected)

    def _inside_domain(self, state: Interval, center_flow: Interval) -> bool:
        bounds = self.bounds
        air_velocity = state[6:9] - center_flow
        airspeed = air_velocity.norm()
        if air_velocity.lower[0] <= 0.0:
            return False
        alpha = (-air_velocity[2]).atan2(air_velocity[0])
        return bool(
            state.lower[3] >= -bounds.roll_abs_max_rad
            and state.upper[3] <= bounds.roll_abs_max_rad
            and state.lower[4] >= -bounds.pitch_abs_max_rad
            and state.upper[4] <= bounds.pitch_abs_max_rad
            and np.all(state.lower[9:12] >= -bounds.body_rate_abs_max_rad_s)
            and np.all(state.upper[9:12] <= bounds.body_rate_abs_max_rad_s)
            and np.all(state.lower[12:15] >= self.aircraft.control_lower_rad)
            and np.all(state.upper[12:15] <= self.aircraft.control_upper_rad)
            and airspeed.lower >= bounds.airspeed_m_s[0]
            and airspeed.upper <= bounds.airspeed_m_s[1]
            and alpha.lower >= -bounds.alpha_abs_max_rad
            and alpha.upper <= bounds.alpha_abs_max_rad
        )

    def _affine_derivative(
        self,
        state: Interval,
        command: Interval,
        center_flow: AffineForm,
        strip_flow: AffineForm,
    ) -> AffineForm:
        if not self._inside_domain(state, center_flow.interval_hull()):
            raise OracleValidationError("state set left the hard domain")
        rotation = _body_to_world(state[3:6])
        velocity = state[6:9]
        omega = state[9:12]
        force, moment = self._affine_aero_loads(
            state,
            center_flow,
            strip_flow,
        )
        gravity_b = _matvec(
            _transpose(rotation),
            Interval.point((0.0, 0.0, -G_M_S2)),
        )
        velocity_dot = (
            force / Interval(*self.bounds.mass_kg) + gravity_b - omega.cross(velocity)
        )
        inertia_omega = omega.affine_map(self.aircraft.inertia_b_kg_m2)
        angular_load = moment - omega.cross(inertia_omega)
        angular_dot = (angular_load[None, :] * self.aircraft.inertia_inv_b).sum(axis=1)
        angular_dot += self._inertia_error(angular_dot, omega)
        position_dot = _matvec(rotation, velocity)
        euler_dot = _matvec(_euler_rate_matrix(state[3], state[4]), omega)
        surface_dot = (command - state[12:15]) / Interval(
            self.bounds.actuator_tau_lower_s,
            self.bounds.actuator_tau_upper_s,
        )
        return _join_affine(
            position_dot,
            euler_dot,
            velocity_dot,
            angular_dot,
            surface_dot,
        )

    def _inertia_error(
        self,
        nominal: AffineForm,
        omega: Interval,
    ) -> Interval:
        inertia_error = self.bounds.inertia_residual_abs_kg_m2
        omega_abs = np.maximum(np.abs(omega.lower), np.abs(omega.upper))
        cross_abs = _cross_abs(omega_abs) @ (inertia_error @ omega_abs)
        inverse_abs = np.abs(self.aircraft.inertia_inv_b)
        coupling = inverse_abs @ inertia_error
        if np.max(np.sum(coupling, axis=1)) >= 1.0:
            raise OracleValidationError("inertia uncertainty is not contractive")
        nominal_box = nominal.interval_hull()
        nominal_abs = np.maximum(
            np.abs(nominal_box.lower),
            np.abs(nominal_box.upper),
        )
        angular_abs = np.linalg.solve(
            np.eye(3) - coupling,
            nominal_abs + inverse_abs @ cross_abs,
        )
        radius = inverse_abs @ (cross_abs + inertia_error @ angular_abs)
        return Interval(-radius, radius)

    def _affine_aero_loads(
        self,
        state: Interval,
        center_flow: AffineForm,
        strip_flow: AffineForm,
    ) -> tuple[AffineForm, AffineForm]:
        table = self.aircraft.strip_table
        velocity = state[6:9]
        omega = state[9:12]
        surface = state[12:15].clip(
            self.aircraft.control_lower_rad,
            self.aircraft.control_upper_rad,
        )
        rigid_velocity = velocity + omega.cross(self._locations)
        air = AffineForm.from_interval(rigid_velocity) - strip_flow
        plane = (air[:, None, :] * self._projection).sum(axis=2)
        speed = plane.norm(axis=1)
        speed_box = speed.interval_hull()
        if (
            np.min(speed_box.lower) < self.bounds.airspeed_m_s[0]
            or np.max(speed_box.upper) > self.bounds.airspeed_m_s[1]
        ):
            raise OracleValidationError("strip airspeed left the validated domain")
        plane_box = plane.interval_hull()
        if np.any(plane_box[:, 0].lower <= 0.0):
            raise OracleValidationError("strip flow left the forward-flight domain")
        drag = -plane / speed[:, None]
        lift = drag.cross(-self._spans)
        lift_box = lift.interval_hull()
        normal_projection = (lift_box * table.normal_b).sum(axis=1)
        negative = normal_projection.upper < 0.0
        uncertain = (normal_projection.lower <= 0.0) & (normal_projection.upper >= 0.0)
        lift = lift * np.where(negative, -1.0, 1.0)[:, None]
        ambiguous = Interval(
            np.minimum(lift_box.lower, -lift_box.upper),
            np.maximum(lift_box.upper, -lift_box.lower),
        )
        lift = lift.replace(uncertain, ambiguous)
        alpha = (-(plane_box * table.normal_b).sum(axis=1)).atan2(plane_box[:, 0])
        if np.max(np.abs((alpha.lower, alpha.upper))) > self.bounds.alpha_abs_max_rad:
            raise OracleValidationError(
                "strip angle of attack left the validated domain"
            )
        delta = surface.affine_map(table.control_mix)
        cl, cd, cm = _flat_plate_interval(
            alpha,
            delta,
            table.aspect_ratio,
            table.chord_m,
            table.flap_chord_fraction,
            table.alpha0_rad,
            table.cd0,
        )
        pressure = speed.square() * (0.5 * Interval(*self.bounds.density_kg_m3))
        force_strips = (
            pressure[:, None]
            * table.area_m2[:, None]
            * (lift * cl[:, None] + drag * cd[:, None])
        )
        moment_strips = force_strips.cross(-self._locations)
        moment_strips += (
            pressure[:, None]
            * table.area_m2[:, None]
            * table.chord_m[:, None]
            * cm[:, None]
            * table.moment_axis_b
        )
        force = force_strips.sum(axis=0)
        moment = moment_strips.sum(axis=0)
        cg_air = AffineForm.from_interval(velocity) - center_flow
        force += (
            cg_air.norm()
            * cg_air
            * (-0.5 * self.aircraft.config.drag_area_fuse_m2)
            * Interval(*self.bounds.density_kg_m3)
        )
        scale = Interval(*self.bounds.aerodynamic_scale)
        force *= scale
        moment *= scale
        force += Interval(
            -self.bounds.force_residual_abs_n,
            self.bounds.force_residual_abs_n,
        )
        moment += Interval(
            -self.bounds.moment_residual_abs_n_m,
            self.bounds.moment_residual_abs_n_m,
        )
        offset = AffineForm.from_interval(
            Interval(
                -self.bounds.cg_residual_abs_m,
                self.bounds.cg_residual_abs_m,
            )
        )
        moment -= offset.cross(force)
        return force, moment

    def _geometry(self, state: Interval) -> GeometryEnclosure:
        geometry = self.generated.geometry
        return GeometryEnclosure(
            _body_points(state, geometry.body_b_m),
            _body_points(state, geometry.contact_b_m),
            _body_points(state, geometry.footprint_b_m),
            _point_velocities(state, geometry.contact_b_m),
        )

    def _free_space(self, prediction: OraclePrediction, mission: Mission) -> bool:
        halfspaces = mission.free_space_halfspaces
        for stage in prediction.stages:
            occupied = stage.continuous_geometry.occupied
            for row, bound in zip(halfspaces.matrix, halfspaces.bounds, strict=True):
                if _point_support(occupied, row) > bound:
                    return False
        return True

    def _gate_event(
        self,
        prediction: OraclePrediction,
        mission: GateMission,
    ) -> bool:
        normal = mission.normal_w
        width = mission.width_axis_w
        height = np.cross(normal, width)
        center = mission.center_w_m
        clearance = mission._clearance(self.generated)
        initial = prediction.geometry(0).occupied
        final = prediction.geometry(PREDICTION_STAGES).occupied
        if _point_support(initial, normal) - normal @ center > -clearance:
            return False
        if _point_support(final, -normal) + normal @ center > -clearance:
            return False
        axes = (width, -width, height, -height)
        limits = (
            0.5 * mission.width_m - clearance,
            0.5 * mission.width_m - clearance,
            0.5 * mission.height_m - clearance,
            0.5 * mission.height_m - clearance,
        )
        crossing = []
        for stage in prediction.stages:
            occupied = stage.continuous_geometry.occupied
            plane = _plane_height(occupied, center, normal)
            if (
                np.min(plane.lower) <= clearance + NUMERICAL_TOLERANCE
                and np.max(plane.upper) >= -clearance - NUMERICAL_TOLERANCE
            ):
                crossing.append(stage)
        if not crossing:
            return False
        center_flow = self.joint_flow.interval_hull()[:3]
        target_heading = float(np.arctan2(-normal[1], normal[0]))
        for stage in crossing:
            occupied = stage.continuous_geometry.occupied
            if any(
                _point_support(occupied, axis) - axis @ center > limit
                for axis, limit in zip(axes, limits, strict=True)
            ):
                return False
            state = stage.continuous_state
            heading = Interval(
                target_heading - state.upper[5],
                target_heading - state.lower[5],
            )
            heading -= 2.0 * pi * round(float(heading.center) / (2.0 * pi))
            airspeed = (state[6:9] - center_flow).norm()
            if (
                heading.lower < -mission.heading_abs_max_rad
                or heading.upper > mission.heading_abs_max_rad
                or state.lower[3] < -mission.roll_abs_max_rad
                or state.upper[3] > mission.roll_abs_max_rad
                or state.lower[4] < mission.pitch_bounds_rad[0]
                or state.upper[4] > mission.pitch_bounds_rad[1]
                or airspeed.lower < mission.airspeed_bounds_m_s[0]
                or airspeed.upper > mission.airspeed_bounds_m_s[1]
            ):
                return False
        return True

    def _landing_event(
        self,
        prediction: OraclePrediction,
        mission: LandingMission,
    ) -> bool:
        length = mission.length_axis_w
        width = mission.width_axis_w
        normal = np.cross(length, width)
        center = mission.center_w_m
        clearance = mission._clearance(self.generated)
        initial_body = _plane_height(
            prediction.geometry(0).occupied,
            center,
            normal,
        )
        initial_contact = _plane_height(
            prediction.geometry(0).contact,
            center,
            normal,
        )
        if (
            np.min(initial_body.lower) <= clearance + NUMERICAL_TOLERANCE
            or np.min(initial_contact.lower) <= clearance + NUMERICAL_TOLERANCE
        ):
            return False
        first_possible = None
        guaranteed = None
        for index, stage in enumerate(prediction.stages):
            contact_height = _plane_height(
                stage.continuous_geometry.contact,
                center,
                normal,
            )
            if (
                first_possible is None
                and np.min(contact_height.lower) > clearance + NUMERICAL_TOLERANCE
            ):
                body_height = _plane_height(
                    stage.continuous_geometry.occupied,
                    center,
                    normal,
                )
                if np.min(body_height.lower) <= clearance + NUMERICAL_TOLERANCE:
                    return False
                continue
            if first_possible is None:
                first_possible = index
            if not self._landing_stage(stage, mission, center, normal, clearance):
                return False
            boundary_height = _plane_height(
                stage.boundary_geometry.contact,
                center,
                normal,
            )
            if np.any(boundary_height.upper <= clearance + NUMERICAL_TOLERANCE):
                guaranteed = index
                break
        if first_possible is None or guaranteed is None:
            return False

        return True

    def _landing_stage(
        self,
        stage: OracleStage,
        mission: LandingMission,
        center: np.ndarray,
        normal: np.ndarray,
        clearance: float,
    ) -> bool:
        length = mission.length_axis_w
        width = mission.width_axis_w
        body = self.generated.geometry.body_b_m
        contact = self.generated.geometry.contact_b_m
        permitted = np.any(
            np.all(body[:, None] == contact[None, :], axis=2),
            axis=1,
        )
        forbidden = _body_points(stage.continuous_state, body[~permitted])
        if (
            forbidden.lower.size
            and np.min(_plane_height(forbidden, center, normal).lower)
            <= clearance + NUMERICAL_TOLERANCE
        ):
            return False
        footprint = stage.continuous_geometry.footprint
        axes = (length, -length, width, -width)
        limits = (
            0.5 * mission.length_m - clearance,
            0.5 * mission.length_m - clearance,
            0.5 * mission.width_m - clearance,
            0.5 * mission.width_m - clearance,
        )
        if any(
            _point_support(footprint, axis) - axis @ center > limit
            for axis, limit in zip(axes, limits, strict=True)
        ):
            return False
        height = ((stage.continuous_state[:3] - center) * normal).sum()
        if (
            height.lower < mission.height_bounds_m[0]
            or height.upper > mission.height_bounds_m[1]
        ):
            return False
        velocity = stage.continuous_geometry.contact_velocity
        normal_velocity = (velocity * normal).sum(axis=1)
        tangent = velocity - normal_velocity[:, None] * normal
        if (
            np.max(normal_velocity.upper) > NUMERICAL_TOLERANCE
            or np.max(-normal_velocity.lower) > mission.normal_speed_max_m_s
            or np.max(tangent.norm(axis=1).upper) > mission.tangential_speed_max_m_s
        ):
            return False
        roll, pitch = _relative_attitude(
            stage.continuous_state[3:6],
            length,
            width,
        )
        return bool(
            roll.lower >= -mission.roll_abs_max_rad
            and roll.upper <= mission.roll_abs_max_rad
            and pitch.lower
            >= mission.touchdown_pitch_rad - mission.pitch_error_abs_max_rad
            and pitch.upper
            <= mission.touchdown_pitch_rad + mission.pitch_error_abs_max_rad
        )

    def _realization(
        self,
        initial: Zonotope,
        factors: np.ndarray,
    ) -> _Realization:
        cursor = 0
        state_count = initial.generators.shape[1]
        state = initial.center + initial.generators @ factors[:state_count]
        cursor += state_count
        queue_count = self.bounds.queue_length * 3
        queue_factors = factors[cursor : cursor + queue_count].reshape(-1, 3)
        cursor += queue_count
        flow_count = self._flow.generators.shape[1]
        flows: list[np.ndarray] = []
        for _ in range(PREDICTION_STAGES):
            flow = self._flow.evaluate(factors[cursor : cursor + flow_count])
            flows.append(flow)
            cursor += flow_count
        model = factors[cursor : cursor + 24]
        cursor += 24
        density = _factor_interval(self.bounds.density_kg_m3, float(model[0]))
        aerodynamic_scale = _factor_interval(
            self.bounds.aerodynamic_scale,
            float(model[1]),
        )
        mass = _factor_interval(self.bounds.mass_kg, float(model[2]))
        force_error = model[3:6] * self.bounds.force_residual_abs_n
        moment_error = model[6:9] * self.bounds.moment_residual_abs_n_m
        cg = model[9:12] * self.bounds.cg_residual_abs_m
        inertia = model[12:21].reshape(3, 3) * self.bounds.inertia_residual_abs_kg_m2
        inertia = self.aircraft.inertia_b_kg_m2 + 0.5 * (inertia + inertia.T)
        tau_factor = model[21:24]
        tau = 0.5 * (
            self.bounds.actuator_tau_lower_s + self.bounds.actuator_tau_upper_s
        ) + 0.5 * tau_factor * (
            self.bounds.actuator_tau_upper_s - self.bounds.actuator_tau_lower_s
        )
        measurement = np.empty((PREDICTION_STAGES, 15))
        command_error = np.empty((PREDICTION_STAGES, 3))
        for stage in range(PREDICTION_STAGES):
            measurement[stage] = (
                factors[cursor : cursor + 15] * self.bounds.state_estimation_abs
            )
            cursor += 15
            command_error[stage] = (
                factors[cursor : cursor + 3] * self.bounds.command_error_abs_rad
            )
            cursor += 3
        lower_age, upper_age = self.bounds.delay_step_bounds
        delay_age = lower_age if factors[cursor] < 0.0 else upper_age
        return {
            "state": state,
            "queue_factors": queue_factors,
            "flows": tuple(flows),
            "density": density,
            "aerodynamic_scale": aerodynamic_scale,
            "mass": mass,
            "force_error": force_error,
            "moment_error": moment_error,
            "cg": cg,
            "inertia": inertia,
            "tau": tau,
            "measurement": measurement,
            "command_error": command_error,
            "delay_age": delay_age,
        }

    def _concrete_derivative(
        self,
        state: np.ndarray,
        command: np.ndarray,
        flow: np.ndarray,
        realization: _Realization,
    ) -> np.ndarray:
        density = realization["density"]
        force, moment = self.aircraft.aero_loads_local_flow(
            state,
            flow[0],
            flow[1:],
            density,
        )
        force *= realization["aerodynamic_scale"]
        moment *= realization["aerodynamic_scale"]
        force += realization["force_error"]
        moment += realization["moment_error"]
        moment -= np.cross(realization["cg"], force)
        rotation = _body_to_world_numpy(state[3:6])
        velocity = state[6:9]
        omega = state[9:12]
        inertia = realization["inertia"]
        position_dot = rotation @ velocity
        euler_dot = _euler_rate_numpy(float(state[3]), float(state[4])) @ omega
        gravity = rotation.T @ np.array((0.0, 0.0, -G_M_S2))
        velocity_dot = force / realization["mass"] + gravity - np.cross(omega, velocity)
        angular_dot = np.linalg.solve(
            inertia,
            moment - np.cross(omega, inertia @ omega),
        )
        surface_dot = (command - state[12:15]) / realization["tau"]
        return np.concatenate(
            (position_dot, euler_dot, velocity_dot, angular_dot, surface_dot)
        )


def _verify_gain_cell(
    generated: _AircraftModel,
    cell: GainCell,
) -> _CellVerification | None:
    initial = _gain_cell_interval(generated, cell)
    queue_lower = (
        generated.aircraft.control_lower_rad + generated.bounds.command_error_abs_rad
    )
    queue_upper = (
        generated.aircraft.control_upper_rad - generated.bounds.command_error_abs_rad
    )
    if np.any(queue_upper < queue_lower):
        return None
    queue = tuple(
        Interval(queue_lower, queue_upper) for _ in range(generated.bounds.queue_length)
    )
    reference = Interval(
        generated.aircraft.control_lower_rad,
        generated.aircraft.control_upper_rad,
    )
    initial_set = Zonotope.from_interval(initial)
    oracle = NonlinearOracle(generated)
    try:
        nonlinear = oracle.propagate(initial_set, queue, cell, reference)
        remainders = oracle.remainder_bounds(initial_set, queue, cell, reference)
    except OracleValidationError:
        return None
    if not remainders.within(generated.bounds):
        return None

    queue_center = np.stack([value.center for value in queue])
    queue_radius = np.stack([value.radius for value in queue])
    fast = FastPredictor(generated).predict(
        initial_set,
        queue_center,
        0,
        queue_radius,
        local_flow=generated.bounds.flow,
    )
    if not _prediction_contains(fast, nonlinear, reference, False):
        return None
    position_remainder, velocity_remainder = _required_geometry_remainders(
        fast,
        nonlinear,
        reference,
        cell,
    )
    verification = _CellVerification(
        generated.bounds.stage_remainder_abs,
        position_remainder,
        velocity_remainder,
    )
    verified_cell = replace(
        cell,
        swept_position_remainder_abs_m=position_remainder,
        contact_velocity_remainder_abs_m_s=velocity_remainder,
    )
    verified_generated = replace(generated, cells=(verified_cell,))
    verified_fast = FastPredictor(verified_generated).predict(
        initial_set,
        queue_center,
        0,
        queue_radius,
        local_flow=generated.bounds.flow,
    )
    if not _prediction_contains(verified_fast, nonlinear, reference, True):
        return None
    return verification


def _gain_cell_interval(
    generated: _AircraftModel,
    cell: GainCell,
) -> Interval:
    lower = cell.anchor.copy()
    upper = cell.anchor.copy()
    scale = generated.state_scale[INTRINSIC_FEEDBACK_INDICES]
    lower[INTRINSIC_FEEDBACK_INDICES] += scale * cell.lower
    upper[INTRINSIC_FEEDBACK_INDICES] += scale * cell.upper
    return Interval(lower, upper)


def _prediction_contains(
    fast: Prediction,
    nonlinear: OraclePrediction,
    reference: Interval,
    include_geometry: bool,
) -> bool:
    if not nonlinear.initial.interval_hull().subset(
        fast.state_interval(0, reference.lower, reference.upper)
    ):
        return False
    for stage_index, stage in enumerate(nonlinear.stages):
        if not stage.successor.interval_hull().subset(
            fast.state_interval(
                stage_index + 1,
                reference.lower,
                reference.upper,
            )
        ):
            return False
        issued = _affine_interval(
            fast.issued_center[stage_index],
            fast.issued_reference[stage_index],
            fast.issued_radius[stage_index],
            reference,
        )
        applied = _affine_interval(
            fast.applied_center[stage_index],
            fast.applied_reference[stage_index],
            fast.applied_radius[stage_index],
            reference,
        )
        if not stage.issued_command.subset(issued):
            return False
        if not stage.applied_command.subset(applied):
            return False
        if include_geometry:
            predicted = _prediction_geometry_interval(
                fast,
                stage_index,
                reference,
            )
            actual = stage.continuous_geometry
            if not actual.occupied.subset(predicted.occupied):
                return False
            if not actual.contact.subset(predicted.contact):
                return False
            if not actual.footprint.subset(predicted.footprint):
                return False
            if not actual.contact_velocity.subset(predicted.contact_velocity):
                return False
    return True


def _required_geometry_remainders(
    fast: Prediction,
    nonlinear: OraclePrediction,
    reference: Interval,
    cell: GainCell,
) -> tuple[np.ndarray, np.ndarray]:
    position_gap = np.zeros(3)
    velocity_gap = np.zeros(3)
    for stage_index, stage in enumerate(nonlinear.stages):
        predicted = _prediction_geometry_interval(fast, stage_index, reference)
        actual = stage.continuous_geometry
        for inner, outer in (
            (actual.occupied, predicted.occupied),
            (actual.contact, predicted.contact),
            (actual.footprint, predicted.footprint),
        ):
            gap = np.maximum(
                np.maximum(outer.lower - inner.lower, inner.upper - outer.upper),
                0.0,
            )
            np.maximum(position_gap, np.max(gap, axis=0), out=position_gap)
        gap = np.maximum(
            np.maximum(
                predicted.contact_velocity.lower - actual.contact_velocity.lower,
                actual.contact_velocity.upper - predicted.contact_velocity.upper,
            ),
            0.0,
        )
        np.maximum(velocity_gap, np.max(gap, axis=0), out=velocity_gap)
    position = cell.swept_position_remainder_abs_m + position_gap
    velocity = cell.contact_velocity_remainder_abs_m_s + velocity_gap
    return np.nextafter(position, np.inf), np.nextafter(velocity, np.inf)


def _affine_interval(
    center: np.ndarray,
    coefficient: np.ndarray,
    radius: np.ndarray,
    reference: Interval,
) -> Interval:
    midpoint = center + coefficient @ reference.center
    support = radius + np.abs(coefficient) @ reference.radius
    return Interval.from_midpoint(midpoint, support)


def _prediction_geometry_interval(
    prediction: Prediction,
    stage: int,
    reference: Interval,
) -> GeometryEnclosure:
    return GeometryEnclosure(
        _affine_interval(
            prediction.body_center[stage],
            prediction.body_reference[stage],
            prediction.body_radius[stage],
            reference,
        ),
        _affine_interval(
            prediction.contact_center[stage],
            prediction.contact_reference[stage],
            prediction.contact_radius[stage],
            reference,
        ),
        _affine_interval(
            prediction.footprint_center[stage],
            prediction.footprint_reference[stage],
            prediction.footprint_radius[stage],
            reference,
        ),
        _affine_interval(
            prediction.contact_velocity_center[stage],
            prediction.contact_velocity_reference[stage],
            prediction.contact_velocity_radius[stage],
            reference,
        ),
    )


def _flat_plate_interval(
    alpha: Interval,
    delta: Interval,
    aspect_ratio: npt.ArrayLike,
    chord_m: npt.ArrayLike,
    flap_chord_fraction: npt.ArrayLike,
    alpha0_rad: npt.ArrayLike,
    cd0: npt.ArrayLike,
) -> tuple[Interval, Interval, Interval]:
    aspect = np.asarray(aspect_ratio, dtype=float).reshape(-1)
    chord = np.asarray(chord_m, dtype=float).reshape(-1)
    flap = np.asarray(flap_chord_fraction, dtype=float).reshape(-1)
    alpha_zero = np.asarray(alpha0_rad, dtype=float).reshape(-1)
    profile_drag = np.asarray(cd0, dtype=float).reshape(-1)
    if np.max(alpha.upper - alpha.lower) <= radians(10.0):
        return _flat_plate_core(
            alpha,
            delta,
            aspect,
            chord,
            flap,
            alpha_zero,
            profile_drag,
        )
    lift = []
    drag = []
    moment = []
    for index in range(aspect.size):
        width = float(alpha.upper[index] - alpha.lower[index])
        count = max(1, int(np.ceil(width / radians(10.0))))
        edges = np.linspace(alpha.lower[index], alpha.upper[index], count + 1)
        bounds = [
            _flat_plate_core(
                Interval(edges[part], edges[part + 1]),
                delta[index],
                aspect[index],
                chord[index],
                flap[index],
                alpha_zero[index],
                profile_drag[index],
            )
            for part in range(count)
        ]
        lift.append(_hull([item[0] for item in bounds]))
        drag.append(_hull([item[1] for item in bounds]))
        moment.append(_hull([item[2] for item in bounds]))
    return _stack(lift), _stack(drag), _stack(moment)


def _flat_plate_core(
    alpha: Interval,
    delta: Interval,
    aspect_ratio: npt.ArrayLike,
    chord_m: npt.ArrayLike,
    flap_chord_fraction: npt.ArrayLike,
    alpha0_rad: npt.ArrayLike,
    cd0: npt.ArrayLike,
) -> tuple[Interval, Interval, Interval]:
    aspect = np.asarray(aspect_ratio, dtype=float)
    chord = np.asarray(chord_m, dtype=float)
    flap_fraction = np.asarray(flap_chord_fraction, dtype=float)
    alpha_zero = np.asarray(alpha0_rad, dtype=float)
    profile_drag = np.asarray(cd0, dtype=float)
    a_le, a_te, alpha_le, alpha_te, alpha_high = flat_plate_coefficients(aspect)
    cl_alpha = 2.0 * pi * (aspect / (aspect + 2.0 * (aspect + 4.0) / (aspect + 2.0)))
    theta_f = np.arccos(2.0 * flap_fraction - 1.0)
    tau_f = 1.0 - (theta_f - np.sin(theta_f)) / pi
    delta_cl = cl_alpha * tau_f * delta
    alpha_lift = alpha - alpha_zero
    abs_alpha = (alpha_lift.square() + SMOOTH_ABS_EPS**2).sqrt()
    f_te = 0.5 * (1.0 - (a_te * (abs_alpha - alpha_te)).tanh())
    f_le = 0.5 * (1.0 - (a_le * (abs_alpha - alpha_le)).tanh())
    sqrt_f_te = f_te.clip(0.0, 1.0).sqrt()
    sin_alpha = alpha_lift.sin()
    cos_alpha = alpha_lift.cos()
    smooth_sin = (sin_alpha.square() + SMOOTH_ABS_EPS**2).sqrt()
    blend = 0.25 * (1.0 + sqrt_f_te).square()
    cl_attached = (
        blend
        * (
            cl_alpha * sin_alpha * cos_alpha.square()
            + f_le.square() * pi * smooth_sin * sin_alpha * cos_alpha
        )
        + delta_cl
    )
    tangent = alpha_lift.clip(
        -0.5 * pi + 1.0e-6,
        0.5 * pi - 1.0e-6,
    ).tan()
    cd_attached = profile_drag + cl_attached * tangent
    cm_attached = -blend * (
        0.0625
        * (-1.0 + 6.0 * sqrt_f_te - 5.0 * f_te)
        * cl_alpha
        * sin_alpha
        * cos_alpha
        + 0.17 * f_le.square() * pi * smooth_sin * sin_alpha
    )
    flap_chord = flap_fraction * chord
    c_prime = (
        (chord - flap_chord) ** 2
        + flap_chord**2
        + 2.0 * flap_chord * (chord - flap_chord) * delta.cos()
    ).sqrt()
    alpha_f = (flap_chord * delta.sin() / c_prime).clip(-1.0, 1.0).asin()
    alpha_post = alpha - alpha_zero + alpha_f
    cd_90 = 1.98 - 4.26e-2 * delta.square() + 2.1e-1 * delta
    sin_post = alpha_post.sin()
    cos_post = alpha_post.cos()
    smooth_post = (sin_post.square() + SMOOTH_ABS_EPS**2).sqrt()
    normal_gain = 1.0 / (0.56 + 0.44 * smooth_post) - 0.41 * (
        1.0 - np.exp(-17.0 / aspect)
    )
    normal = cd_90 * sin_post * normal_gain
    axial = 0.015 * cos_post
    cl_post = normal * cos_post - axial * sin_post
    cd_post = normal * sin_post + axial * cos_post
    cm_post = -normal * (0.25 - 0.175 * (1.0 - 2.0 * alpha_post / pi))
    smooth_alpha = (alpha.square() + SMOOTH_ABS_EPS**2).sqrt()
    sigma = 0.5 * (1.0 + (20.0 * (alpha_high - smooth_alpha)).tanh())
    return (
        sigma * cl_attached + (1.0 - sigma) * cl_post,
        sigma * cd_attached + (1.0 - sigma) * cd_post,
        sigma * cm_attached + (1.0 - sigma) * cm_post,
    )


def _body_points(state: Interval, points_b_m: npt.ArrayLike) -> Interval:
    position = state[:3]
    rotation = _body_to_world(state[3:6])
    points = np.asarray(points_b_m, dtype=float).reshape(-1, 3)
    rotated = _interval_batch_matvec(rotation, Interval.point(points))
    return position + rotated


def _point_velocities(state: Interval, points_b_m: npt.ArrayLike) -> Interval:
    rotation = _body_to_world(state[3:6])
    velocity = state[6:9]
    omega = state[9:12]
    points = np.asarray(points_b_m, dtype=float).reshape(-1, 3)
    point_velocity = velocity + omega.cross(points)
    return _interval_batch_matvec(rotation, point_velocity)


def _body_to_world(attitude: Interval) -> Interval:
    phi, theta, psi = attitude[0], attitude[1], attitude[2]
    c_phi, s_phi = phi.cos(), phi.sin()
    c_theta, s_theta = theta.cos(), theta.sin()
    c_psi, s_psi = psi.cos(), psi.sin()
    return _matrix(
        (
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
    )


def _body_to_world_numpy(attitude: npt.ArrayLike) -> np.ndarray:
    phi, theta, psi = np.asarray(attitude, dtype=float).reshape(3)
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    c_psi, s_psi = np.cos(psi), np.sin(psi)
    return np.array(
        [
            [
                c_theta * c_psi,
                s_phi * s_theta * c_psi - c_phi * s_psi,
                c_phi * s_theta * c_psi + s_phi * s_psi,
            ],
            [
                -c_theta * s_psi,
                -s_phi * s_theta * s_psi - c_phi * c_psi,
                -c_phi * s_theta * s_psi + s_phi * c_psi,
            ],
            [s_theta, -s_phi * c_theta, -c_phi * c_theta],
        ]
    )


def _euler_rate_matrix(phi: Interval, theta: Interval) -> Interval:
    c_phi, s_phi = phi.cos(), phi.sin()
    c_theta = theta.cos()
    t_theta = theta.tan()
    zero = Interval.point(0.0)
    one = Interval.point(1.0)
    return _matrix(
        (
            (one, s_phi * t_theta, c_phi * t_theta),
            (zero, c_phi, -s_phi),
            (zero, s_phi / c_theta, c_phi / c_theta),
        )
    )


def _euler_rate_numpy(phi: float, theta: float) -> np.ndarray:
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_theta = np.cos(theta)
    return np.array(
        [
            [1.0, s_phi * np.tan(theta), c_phi * np.tan(theta)],
            [0.0, c_phi, -s_phi],
            [0.0, s_phi / c_theta, c_phi / c_theta],
        ]
    )


def _relative_attitude(
    attitude: Interval,
    forward: np.ndarray,
    lateral: np.ndarray,
) -> tuple[Interval, Interval]:
    normal = np.cross(forward, lateral)
    desired = np.column_stack((forward, -lateral, -normal))
    relative = (
        Interval.point(desired.T)[:, :, None] * _body_to_world(attitude)[None, :, :]
    ).sum(axis=1)
    pitch = (-relative[2, 0]).clip(-1.0, 1.0).asin()
    roll = relative[2, 1].atan2(relative[2, 2])
    return roll, pitch


def _matrix(rows: tuple[tuple[Interval, ...], ...]) -> Interval:
    return Interval(
        np.array([[float(value.lower) for value in row] for row in rows]),
        np.array([[float(value.upper) for value in row] for row in rows]),
    )


def _transpose(matrix: Interval) -> Interval:
    return Interval(matrix.lower.T, matrix.upper.T)


def _matvec(matrix: Interval, vector: Interval) -> Interval:
    return (matrix * vector).sum(axis=1)


def _interval_batch_matvec(matrix: Interval, vector: Interval) -> Interval:
    return (matrix[None, :, :] * vector[:, None, :]).sum(axis=2)


def _stack(values: list[Interval]) -> Interval:
    return Interval(
        np.stack([value.lower for value in values]),
        np.stack([value.upper for value in values]),
    )


def _hull(values: list[Interval]) -> Interval:
    result = values[0]
    for value in values[1:]:
        result = result.hull(value)
    return result


def _reduce(
    value: Zonotope,
    maximum: int,
    protected: tuple[int, ...],
) -> Zonotope:
    count = value.generators.shape[1]
    if count <= maximum:
        return value
    keep_count = max(maximum - value.center.size, 0)
    norms = np.linalg.norm(value.generators, axis=0)
    protected_indices = np.unique(
        np.asarray(
            [index for index in protected if 0 <= index < count],
            dtype=int,
        )
    )
    if protected_indices.size > keep_count:
        order = np.argsort(norms[protected_indices])
        protected_indices = protected_indices[order[-keep_count:]]
    candidates = np.setdiff1d(
        np.arange(count),
        protected_indices,
        assume_unique=True,
    )
    remaining = keep_count - protected_indices.size
    selected = (
        candidates[np.argsort(norms[candidates])[-remaining:]]
        if remaining > 0
        else np.empty(0, dtype=int)
    )
    kept_indices = np.concatenate((protected_indices, selected))
    discarded_indices = np.setdiff1d(
        np.arange(count),
        kept_indices,
        assume_unique=True,
    )
    discarded = value.generators[:, discarded_indices]
    kept = value.generators[:, kept_indices]
    radius = Zonotope(np.zeros(value.center.size), discarded).radius
    return Zonotope(
        value.center,
        np.concatenate((kept, np.diag(radius)), axis=1),
    )


def _join_affine(*values: AffineForm | Interval) -> AffineForm:
    forms = [
        value if isinstance(value, AffineForm) else AffineForm.from_interval(value)
        for value in values
    ]
    generated = [value for value in forms if value.generator_count]
    if not generated:
        interval = Interval(
            np.concatenate(
                [value.interval_hull().lower.reshape(-1) for value in forms]
            ),
            np.concatenate(
                [value.interval_hull().upper.reshape(-1) for value in forms]
            ),
        )
        return AffineForm.from_interval(interval)
    basis = generated[0].basis
    count = generated[0].generator_count
    aligned = []
    for value in forms:
        if value.generator_count:
            aligned.append(value)
        else:
            aligned.append(
                AffineForm(
                    value.center,
                    np.zeros(value.center.shape + (count,)),
                    value.remainder,
                    basis,
                )
            )
    return AffineForm(
        np.concatenate([value.center.reshape(-1) for value in aligned]),
        np.concatenate(
            [value.generators.reshape(-1, count) for value in aligned],
            axis=0,
        ),
        np.concatenate([value.remainder.reshape(-1) for value in aligned]),
        basis,
    )


def _inflate(value: Interval, fraction: float) -> Interval:
    radius = value.radius
    coupling = 1.0e-3 * float(np.max(radius))
    margin = np.maximum(fraction * (radius + coupling), 1.0e-12)
    return Interval(value.lower - margin, value.upper + margin)


def _up_sum(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    total = np.nextafter(first + second, np.inf)
    return np.where(first == 0.0, second, np.where(second == 0.0, first, total))


def _roundoff_bound(
    cell: GainCell,
    state: Interval,
    command: Interval,
) -> np.ndarray:
    epsilon = np.finfo(float).eps
    state_abs = np.maximum(np.abs(state.lower), np.abs(state.upper))
    command_abs = np.maximum(np.abs(command.lower), np.abs(command.upper))
    state_gamma = 15.0 * epsilon / (1.0 - 15.0 * epsilon)
    command_gamma = 3.0 * epsilon / (1.0 - 3.0 * epsilon)
    error = state_gamma * (np.abs(cell.state_matrix) @ state_abs)
    error += command_gamma * (np.abs(cell.control_matrix) @ command_abs)
    error += epsilon * (
        np.abs(cell.offset)
        + np.sum(np.abs(cell.flow_generators), axis=1)
        + np.sum(np.abs(cell.model_generators), axis=1)
    )
    return np.nextafter(error, np.inf)


def _plane_height(
    points: Interval,
    center: np.ndarray,
    normal: np.ndarray,
) -> Interval:
    return ((points - center) * normal).sum(axis=1)


def _point_support(points: Interval, direction: np.ndarray) -> float:
    return float(np.max((points * direction).sum(axis=1).upper))


def _prediction_state(
    prediction: Prediction,
    stage: int,
    reference: Interval,
) -> Zonotope:
    count = prediction.generator_count[stage]
    center = (
        prediction.state_center[stage]
        + prediction.state_reference[stage] @ reference.center
    )
    generators = prediction.state_generators[stage, :, :count]
    reference_generators = prediction.state_reference[stage] @ np.diag(reference.radius)
    return Zonotope(
        center,
        np.column_stack((generators, reference_generators)),
    )


def _cross_abs(vector_abs: np.ndarray) -> np.ndarray:
    x, y, z = vector_abs
    return np.array(((0.0, z, y), (z, 0.0, x), (y, x, 0.0)))


def _factor_interval(bounds: tuple[float, float], factor: float) -> float:
    return 0.5 * (bounds[0] + bounds[1]) + 0.5 * factor * (bounds[1] - bounds[0])


def _set_array(
    instance: object,
    name: str,
    value: npt.ArrayLike,
    shape: tuple[int, ...],
) -> None:
    array = np.asarray(value, dtype=float).reshape(shape).copy()
    array.flags.writeable = False
    object.__setattr__(instance, name, array)
