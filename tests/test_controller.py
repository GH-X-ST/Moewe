"""Tests for the two-rate joint-flow capture governor."""

from __future__ import annotations

from pathlib import Path
from subprocess import run
import sys
from time import perf_counter

import numpy as np
import pytest

from control.capture import (
    _LookupNode,
    CaptureCertificate,
    CaptureConstraintBuilder,
    CaptureSimplex,
)
from control.flow import FlowBounds
from control.governor import JointFlowCaptureGovernor, _SOLVED, _TIMED_OUT
from control.interval import Zonotope
from control.missions import CompilationDomain, Halfspaces
from control.observer import StateFlowEstimate
from control.predictor import (
    _AircraftModel,
    FastPredictor,
    GeneratedAircraft,
    _generate_aircraft,
)
from control.uncertainty import (
    Bounds,
    FAST_PERIOD_S,
    NEXT_UPDATE_STAGE,
    PREDICTION_STAGES,
)
from models.aircraft import Aircraft
from models.geometry import RigidBodyGeometry


class _Mission:
    def __init__(self) -> None:
        self.terminal_x: float | None = None
        self.last_segment: np.ndarray | None = None
        self.matrix = np.zeros((1, 3))
        self.bounds = np.ones(1)
        self.queue_dependent = False

    def terminal_halfspaces(self, generated: GeneratedAircraft) -> Halfspaces:
        del generated
        return Halfspaces.box((-100.0, -100.0), (100.0, 100.0))

    @property
    def free_space_halfspaces(self) -> Halfspaces:
        return Halfspaces.box((-100.0,) * 3, (100.0,) * 3)

    def error(self, state: np.ndarray, generated: GeneratedAircraft) -> np.ndarray:
        del generated
        return np.asarray(state, dtype=float)[1:3]

    def distance(self, state: np.ndarray) -> float:
        return float(np.asarray(state, dtype=float)[0])

    def realized(self, states: np.ndarray, generated: GeneratedAircraft) -> bool:
        del generated
        self.last_segment = np.asarray(states, dtype=float).copy()
        return self.terminal_x is not None and bool(
            states[0, 0] <= self.terminal_x <= states[-1, 0]
        )

    def terminal_support_constraints(
        self,
        prediction: object,
        generated: GeneratedAircraft,
        domain: CompilationDomain,
    ) -> tuple[np.ndarray, np.ndarray]:
        del prediction, generated, domain
        return np.empty((0, 3)), np.empty(0)

    def online_rows(
        self, prediction: object, *_: object
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.queue_dependent:
            center = prediction.applied_center
            return np.array(((1.0, 0.0, 0.0),)), np.array((0.03 + center[0, 0],))
        return self.matrix.copy(), self.bounds.copy()


@pytest.fixture(scope="module")
def generated() -> GeneratedAircraft:
    aircraft = Aircraft()
    bounds = Bounds(
        flow=FlowBounds(
            (-0.02, -0.02, -0.02),
            (0.02, 0.02, 0.02),
            np.full((3, 3), -0.004),
            np.full((3, 3), 0.004),
            (0.003, 0.003, 0.003),
        ),
        density_kg_m3=(1.224, 1.226),
        aerodynamic_scale=(0.999, 1.001),
        force_residual_abs_n=np.full(3, 1.0e-5),
        moment_residual_abs_n_m=np.full(3, 1.0e-7),
        mass_kg=(0.1429, 0.1431),
        cg_residual_abs_m=np.full(3, 1.0e-5),
        inertia_residual_abs_kg_m2=np.full((3, 3), 1.0e-8),
        actuator_tau_lower_s=np.full(3, 0.059),
        actuator_tau_upper_s=np.full(3, 0.061),
        command_error_abs_rad=np.full(3, 1.0e-5),
        state_estimation_abs=np.array(
            [1.0e-5] * 3 + [1.0e-6] * 3 + [1.0e-4] * 3 + [1.0e-5] * 3 + [1.0e-6] * 3
        ),
        command_delay_s=(0.0, 0.073),
        nonlinear_remainder_abs=np.array(
            [2.0e-5] * 3 + [2.0e-6] * 3 + [2.0e-4] * 3 + [2.0e-4] * 3 + [2.0e-5] * 3
        ),
        numerical_remainder_abs=np.full(15, 1.0e-12),
        body_inflation_m=1.0e-4,
        mission_position_error_abs_m=1.0e-4,
        mission_attitude_error_abs_rad=1.0e-4,
        roll_abs_max_rad=np.deg2rad(60.0),
        pitch_abs_max_rad=np.deg2rad(60.0),
        airspeed_m_s=(1.0, 15.0),
        alpha_abs_max_rad=np.deg2rad(45.0),
        body_rate_abs_max_rad_s=10.0,
    )
    body = np.array(
        (
            (0.1055, 0.382, 0.0),
            (0.1055, -0.382, 0.0),
            (-0.38875, 0.182, 0.014),
            (-0.38875, -0.182, 0.014),
            (-0.37175, 0.0, -0.116),
        )
    )
    geometry = RigidBodyGeometry(body, body[2:4], body[2:4])
    anchor = np.zeros(15)
    anchor[:3] = (2.0, 2.0, 2.0)
    anchor[6] = 6.0
    model = _generate_aircraft(
        aircraft,
        geometry,
        bounds,
        anchor,
        np.zeros(3),
    )
    return GeneratedAircraft(
        model.aircraft,
        model.geometry,
        model.bounds,
        model.state_scale,
        model.domain_anchor,
        model.reference_center,
        model.reference_scale,
        model.cells,
    )


def _mission(generated: GeneratedAircraft) -> _Mission:
    del generated
    return _Mission()


def _estimate(
    generated: GeneratedAircraft,
    state: np.ndarray,
    timestamp_s: float = 0.0,
) -> StateFlowEstimate:
    flow = generated.bounds.flow
    flow_center = 0.5 * (flow.center_lower_m_s + flow.center_upper_m_s)
    flow_radius = 0.25 * (flow.center_upper_m_s - flow.center_lower_m_s)
    center = np.concatenate((np.asarray(state, dtype=float).reshape(15), flow_center))
    generators = np.diag(
        np.concatenate((0.5 * generated.bounds.state_estimation_abs, flow_radius))
    )
    return StateFlowEstimate(timestamp_s, Zonotope(center, generators), flow)


def _certificate(
    generated: GeneratedAircraft,
    matrix: np.ndarray | None = None,
    bounds: np.ndarray | None = None,
    backup: np.ndarray | None = None,
    terminal: bool = False,
    mission: _Mission | None = None,
    reference_lower: np.ndarray | None = None,
    reference_upper: np.ndarray | None = None,
) -> CaptureCertificate:
    rows = np.zeros((1, 3)) if matrix is None else np.asarray(matrix, dtype=float)
    limits = (
        np.ones(rows.shape[0]) if bounds is None else np.asarray(bounds, dtype=float)
    )
    backup_reference = (
        np.zeros(3) if backup is None else np.asarray(backup, dtype=float)
    )
    radius = 400.0
    vertices = radius * np.array(
        (
            (3.0, -1.0, -1.0),
            (-1.0, 3.0, -1.0),
            (-1.0, -1.0, 3.0),
            (-1.0, -1.0, -1.0),
        )
    )
    selected_mission = _mission(generated) if mission is None else mission
    selected_mission.matrix = rows.copy()
    selected_mission.bounds = limits.copy()
    simplex = CaptureSimplex(
        vertices,
        backup_reference,
        rows.shape[0] + (6 if terminal else 0),
        0,
        1.0,
        terminal,
        (
            generated.aircraft.control_lower_rad
            if reference_lower is None
            else reference_lower
        ),
        (
            generated.aircraft.control_upper_rad
            if reference_upper is None
            else reference_upper
        ),
    )
    queue_size = 3 * generated.bounds.queue_length
    augmented_offset = np.concatenate((generated.cells[0].anchor, np.zeros(queue_size)))
    transform = np.zeros((3, augmented_offset.size))
    transform[:, :3] = np.eye(3)
    anchor = generated.cells[0].anchor
    return CaptureCertificate(
        beta=np.ones(2),
        coordinate_offset=np.zeros(3),
        coordinate_scale=np.ones(3),
        reference_lower=generated.aircraft.control_lower_rad,
        reference_upper=generated.aircraft.control_upper_rad,
        simplices=(simplex,),
        terminal_bounds=np.ones(2),
        distance_max=100.0,
        augmented_offset=augmented_offset,
        augmented_scale=np.ones(augmented_offset.size),
        coordinate_transform=transform,
        domain=CompilationDomain(anchor - 100.0, anchor + 100.0),
        mission=selected_mission,
        generated=generated,
        lookup_roots=(_LookupNode(vertices, 0),),
    )


def _activate(
    generated: GeneratedAircraft,
    mission: _Mission,
    certificate: CaptureCertificate,
    queue: np.ndarray | None = None,
    reference: np.ndarray | None = None,
) -> JointFlowCaptureGovernor:
    controller = JointFlowCaptureGovernor(generated)
    assert certificate.mission is mission
    controller.activate(
        certificate,
        _estimate(generated, generated.cells[0].anchor),
        np.zeros(3) if reference is None else reference,
        np.zeros((generated.bounds.queue_length, 3)) if queue is None else queue,
    )
    controller._constraint_builder.rows = mission.online_rows
    return controller


def _nominal_state(controller: JointFlowCaptureGovernor, stage: int) -> np.ndarray:
    prediction = controller._predictor.prediction
    return (
        prediction.state_center[stage]
        + prediction.state_reference[stage] @ controller.current_reference
    )


def test_runtime_rejects_unverified_aircraft_model(
    generated: GeneratedAircraft,
) -> None:
    model = _AircraftModel(
        generated.aircraft,
        generated.geometry,
        generated.bounds,
        generated.state_scale,
        generated.domain_anchor,
        generated.reference_center,
        generated.reference_scale,
        generated.cells,
    )
    with pytest.raises(TypeError, match="oracle-verified"):
        JointFlowCaptureGovernor(model)


def test_capture_certificate_is_bound_to_its_aircraft_core(
    generated: GeneratedAircraft,
) -> None:
    other = GeneratedAircraft(
        generated.aircraft,
        generated.geometry,
        generated.bounds,
        generated.state_scale,
        generated.domain_anchor,
        generated.reference_center,
        generated.reference_scale,
        generated.cells,
    )
    controller = JointFlowCaptureGovernor(other)
    certificate = _certificate(generated)
    with pytest.raises(ValueError, match="another aircraft core"):
        controller.activate(
            certificate,
            _estimate(generated, generated.cells[0].anchor),
            np.zeros(3),
            np.zeros((generated.bounds.queue_length, 3)),
        )


def test_runtime_rejects_an_oversized_observer_state_set(
    generated: GeneratedAircraft,
) -> None:
    mission = _mission(generated)
    certificate = _certificate(generated, mission=mission)
    controller = JointFlowCaptureGovernor(generated)
    estimate = _estimate(generated, generated.cells[0].anchor)
    controller.activate(
        certificate,
        estimate,
        np.zeros(3),
        np.zeros((generated.bounds.queue_length, 3)),
    )
    generators = estimate.joint.generators.copy()
    generators[0, 0] = 2.0 * generated.bounds.state_estimation_abs[0]
    oversized = StateFlowEstimate(
        0.0,
        Zonotope(estimate.joint.center, generators),
        estimate.local_flow,
    )

    assert controller.command(oversized, np.zeros(3), perf_counter() + 1.0) is None
    assert controller.status == "out_of_envelope"


def test_runtime_rejects_flow_outside_the_compiled_envelope(
    generated: GeneratedAircraft,
) -> None:
    mission = _mission(generated)
    certificate = _certificate(generated, mission=mission)
    controller = JointFlowCaptureGovernor(generated)
    estimate = _estimate(generated, generated.cells[0].anchor)
    controller.activate(
        certificate,
        estimate,
        np.zeros(3),
        np.zeros((generated.bounds.queue_length, 3)),
    )
    certified = generated.bounds.flow
    outside = FlowBounds(
        certified.center_lower_m_s - 0.01,
        certified.center_upper_m_s,
        certified.gradient_lower_s,
        certified.gradient_upper_s,
        certified.remainder_abs_m_s,
    )
    invalid = StateFlowEstimate(0.0, estimate.joint, outside)

    assert controller.command(invalid, np.zeros(3), perf_counter() + 1.0) is None
    assert controller.status == "out_of_envelope"


def test_runtime_rejects_a_reused_estimate(
    generated: GeneratedAircraft,
) -> None:
    mission = _mission(generated)
    certificate = _certificate(generated, mission=mission)
    controller = JointFlowCaptureGovernor(generated)
    initial = generated.cells[0].anchor
    controller.activate(
        certificate,
        _estimate(generated, initial),
        np.zeros(3),
        np.zeros((generated.bounds.queue_length, 3)),
    )
    controller._constraint_builder.rows = mission.online_rows
    estimate = _estimate(generated, initial, FAST_PERIOD_S)
    assert controller.command(estimate, np.zeros(3), perf_counter() + 30.0) is not None

    assert controller.command(estimate, np.zeros(3), perf_counter() + 30.0) is None
    assert controller.status == "out_of_envelope"


def test_runtime_rejects_excess_initial_generators(
    generated: GeneratedAircraft,
) -> None:
    state = generated.cells[0].anchor
    center = np.concatenate((state, np.zeros(3)))
    generators = np.zeros((18, 31))
    generators[:15] = generated.bounds.state_estimation_abs[:, None] / 62.0
    estimate = StateFlowEstimate(
        0.0,
        Zonotope(center, generators),
        generated.bounds.flow,
    )
    controller = JointFlowCaptureGovernor(generated)
    controller.activate(
        _certificate(generated),
        estimate,
        np.zeros(3),
        np.zeros((generated.bounds.queue_length, 3)),
    )

    assert controller.status == "out_of_envelope"


def test_current_flow_changes_online_air_velocity_rows(
    generated: GeneratedAircraft,
) -> None:
    mission = _mission(generated)
    certificate = _certificate(generated, mission=mission)
    predictor = FastPredictor(generated)
    builder = CaptureConstraintBuilder(generated, mission, certificate.domain)
    state = _estimate(generated, generated.cells[0].anchor).state
    queue = np.zeros((generated.bounds.queue_length, 3))
    certified = generated.bounds.flow

    def rows(center_x: float) -> np.ndarray:
        local_flow = FlowBounds(
            (center_x - 0.001, -0.001, -0.001),
            (center_x + 0.001, 0.001, 0.001),
            certified.gradient_lower_s,
            certified.gradient_upper_s,
            certified.remainder_abs_m_s,
        )
        prediction = predictor.predict(
            state,
            queue,
            0,
            local_flow=local_flow,
        )
        return np.array(
            [
                limit - prediction.air_velocity_support(1, direction)[0]
                for direction, limit in zip(
                    builder.air_velocity_directions,
                    builder.air_velocity_bounds,
                    strict=True,
                )
            ]
        )

    assert not np.array_equal(rows(-0.01), rows(0.01))


def test_governor_runs_exactly_every_five_fast_commands(
    generated: GeneratedAircraft,
) -> None:
    mission = _mission(generated)
    controller = _activate(
        generated,
        mission,
        _certificate(generated, mission=mission),
    )
    deadline = perf_counter() + 30.0
    first_nominal = np.full(3, 0.02)
    assert (
        controller.command(
            _estimate(generated, generated.cells[0].anchor, FAST_PERIOD_S),
            first_nominal,
            deadline,
        )
        is not None
    )
    first = controller.current_reference.copy()
    for stage in range(1, NEXT_UPDATE_STAGE):
        assert (
            controller.command(
                _estimate(
                    generated,
                    _nominal_state(controller, stage),
                    (stage + 1) * FAST_PERIOD_S,
                ),
                np.full(3, -0.02),
                deadline,
            )
            is not None
        )
        np.testing.assert_array_equal(controller.current_reference, first)

    next_state = _nominal_state(controller, NEXT_UPDATE_STAGE)
    assert (
        controller.command(
            _estimate(
                generated,
                next_state,
                (NEXT_UPDATE_STAGE + 1) * FAST_PERIOD_S,
            ),
            np.full(3, -0.02),
            deadline,
        )
        is not None
    )
    assert not np.array_equal(controller.current_reference, first)


def test_feedback_uses_each_short_prediction_center_and_exact_fifo(
    generated: GeneratedAircraft,
) -> None:
    mission = _mission(generated)
    queue = np.linspace(
        -0.03,
        0.03,
        generated.bounds.queue_length * 3,
    ).reshape(-1, 3)
    original = queue.copy()
    controller = _activate(
        generated,
        mission,
        _certificate(generated, mission=mission),
        queue,
    )
    deadline = perf_counter() + 30.0
    first = controller.command(
        _estimate(generated, generated.cells[0].anchor, FAST_PERIOD_S),
        np.full(3, 0.01),
        deadline,
    )
    assert first is not None
    first = first.copy()
    np.testing.assert_array_equal(controller.issued_queue[:-1], original[1:])
    np.testing.assert_array_equal(controller.issued_queue[-1], first)

    state = _nominal_state(controller, 1)
    prediction = controller._predictor.prediction
    expected = (
        prediction.issued_center[1]
        + prediction.issued_reference[1] @ controller.current_reference
    )
    second = controller.command(
        _estimate(generated, state, 2.0 * FAST_PERIOD_S),
        np.zeros(3),
        deadline,
    )
    assert second is not None
    np.testing.assert_allclose(second, expected, atol=2.0e-15)


def test_complete_fast_belief_must_remain_in_prediction(
    generated: GeneratedAircraft,
) -> None:
    mission = _mission(generated)
    controller = _activate(
        generated,
        mission,
        _certificate(generated, mission=mission),
    )
    deadline = perf_counter() + 30.0
    assert (
        controller.command(
            _estimate(generated, generated.cells[0].anchor, FAST_PERIOD_S),
            np.zeros(3),
            deadline,
        )
        is not None
    )
    escaped = _nominal_state(controller, 1)
    escaped[3] += 1.0
    queue = controller.issued_queue.copy()
    assert (
        controller.command(
            _estimate(generated, escaped, 2.0 * FAST_PERIOD_S),
            np.zeros(3),
            deadline,
        )
        is None
    )
    assert controller.status == "out_of_envelope"
    np.testing.assert_array_equal(controller.issued_queue, queue)


def test_next_governor_sample_must_match_the_previous_stage_five_set(
    generated: GeneratedAircraft,
) -> None:
    mission = _mission(generated)
    controller = _activate(
        generated,
        mission,
        _certificate(generated, mission=mission),
    )
    deadline = perf_counter() + 30.0
    for stage in range(NEXT_UPDATE_STAGE):
        state = (
            generated.cells[0].anchor
            if stage == 0
            else _nominal_state(controller, stage)
        )
        assert (
            controller.command(
                _estimate(generated, state, (stage + 1) * FAST_PERIOD_S),
                np.zeros(3),
                deadline,
            )
            is not None
        )
    escaped = _nominal_state(controller, NEXT_UPDATE_STAGE)
    escaped[3] += 1.0
    assert (
        controller.command(
            _estimate(
                generated,
                escaped,
                (NEXT_UPDATE_STAGE + 1) * FAST_PERIOD_S,
            ),
            np.zeros(3),
            deadline,
        )
        is None
    )
    assert controller.status == "out_of_envelope"


def test_complete_state_domain_is_checked_before_projected_membership(
    generated: GeneratedAircraft,
) -> None:
    mission = _mission(generated)
    certificate = _certificate(generated, mission=mission)
    controller = _activate(
        generated,
        mission,
        certificate,
    )
    state = generated.cells[0].anchor.copy()
    state[3] = certificate.domain.upper[3]
    assert (
        controller.command(
            _estimate(generated, state, FAST_PERIOD_S),
            np.zeros(3),
            perf_counter() + 30.0,
        )
        is None
    )
    assert controller.status == "out_of_envelope"


def test_runtime_does_not_add_estimation_fiber_twice(
    generated: GeneratedAircraft,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mission = _mission(generated)
    certificate = _certificate(generated, mission=mission)
    controller = _activate(generated, mission, certificate)
    original = CaptureCertificate.locate_augmented_into
    calls = 0

    def locate(
        instance: CaptureCertificate,
        center: np.ndarray,
        augmented_normalized: np.ndarray,
        coordinate_normalized: np.ndarray,
        weights: np.ndarray,
    ) -> int:
        nonlocal calls
        calls += 1
        return original(
            instance,
            center,
            augmented_normalized,
            coordinate_normalized,
            weights,
        )

    monkeypatch.setattr(CaptureCertificate, "locate_augmented_into", locate)
    assert (
        controller.command(
            _estimate(generated, generated.cells[0].anchor, FAST_PERIOD_S),
            np.zeros(3),
            perf_counter() + 30.0,
        )
        is not None
    )
    assert calls == 1


def test_timeout_uses_only_the_compiled_backup(
    generated: GeneratedAircraft,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backup = np.array((0.01, -0.01, 0.005))
    mission = _mission(generated)
    controller = _activate(
        generated,
        mission,
        _certificate(generated, backup=backup, mission=mission),
    )
    monkeypatch.setattr(controller._solver, "solve_into", lambda *_: _TIMED_OUT)
    command = controller.command(
        _estimate(generated, generated.cells[0].anchor, FAST_PERIOD_S),
        np.full(3, 0.1),
        perf_counter() + 1.0,
    )
    assert command is not None
    np.testing.assert_array_equal(controller.current_reference, backup)
    assert controller.status == "active"


def test_positive_solver_residual_leaves_the_capture_envelope(
    generated: GeneratedAircraft,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backup = np.array((-0.01, 0.0, 0.0))
    mission = _mission(generated)
    controller = _activate(
        generated,
        mission,
        _certificate(
            generated,
            matrix=np.array(((1.0, 0.0, 0.0),)),
            bounds=np.array((0.0,)),
            backup=backup,
            mission=mission,
        ),
    )

    def unsafe_result(*args: object) -> int:
        result = args[4]
        assert isinstance(result, np.ndarray)
        result[:] = (5.0e-11, 0.0, 0.0)
        return _SOLVED

    monkeypatch.setattr(controller._solver, "solve_into", unsafe_result)
    command = controller.command(
        _estimate(generated, generated.cells[0].anchor, FAST_PERIOD_S),
        np.zeros(3),
        perf_counter() + 1.0,
    )

    assert command is None
    assert controller.status == "out_of_envelope"


def test_expired_update_deadline_stops_before_prediction(
    generated: GeneratedAircraft,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mission = _mission(generated)
    controller = _activate(
        generated,
        mission,
        _certificate(generated, mission=mission),
    )

    def fail(*_: object, **__: object) -> None:
        raise AssertionError("predictor ran after the update deadline")

    monkeypatch.setattr(controller._predictor, "predict", fail)
    command = controller.command(
        _estimate(generated, generated.cells[0].anchor, FAST_PERIOD_S),
        np.zeros(3),
        perf_counter(),
    )
    assert command is None
    assert controller.status == "out_of_envelope"


def test_deadline_after_prediction_uses_backup_without_building_rows(
    generated: GeneratedAircraft,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backup = np.array((0.01, -0.01, 0.005))
    mission = _mission(generated)
    controller = _activate(
        generated,
        mission,
        _certificate(generated, backup=backup, mission=mission),
    )
    checks = iter((False, True))
    monkeypatch.setattr(
        "control.governor._deadline_reached",
        lambda _: next(checks),
    )

    def fail(*_: object, **__: object) -> None:
        raise AssertionError("rows were built after the optimization deadline")

    monkeypatch.setattr(controller._constraint_builder, "rows", fail)
    command = controller.command(
        _estimate(generated, generated.cells[0].anchor, FAST_PERIOD_S),
        np.full(3, 0.1),
        perf_counter() + 1.0,
    )
    assert command is not None
    np.testing.assert_array_equal(controller.current_reference, backup)


def test_deadline_after_rows_uses_backup_without_solving(
    generated: GeneratedAircraft,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backup = np.array((0.01, -0.01, 0.005))
    mission = _mission(generated)
    controller = _activate(
        generated,
        mission,
        _certificate(generated, backup=backup, mission=mission),
    )
    checks = iter((False, False, True))
    monkeypatch.setattr(
        "control.governor._deadline_reached",
        lambda _: next(checks),
    )

    def fail(*_: object, **__: object) -> None:
        raise AssertionError("solver ran after the optimization deadline")

    monkeypatch.setattr(controller._solver, "solve_into", fail)
    command = controller.command(
        _estimate(generated, generated.cells[0].anchor, FAST_PERIOD_S),
        np.full(3, 0.1),
        perf_counter() + 1.0,
    )
    assert command is not None
    np.testing.assert_array_equal(controller.current_reference, backup)


def test_infeasible_online_set_uses_the_compiled_backup(
    generated: GeneratedAircraft,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backup = np.array((0.01, -0.01, 0.005))
    mission = _mission(generated)
    controller = _activate(
        generated,
        mission,
        _certificate(generated, backup=backup, mission=mission),
    )
    monkeypatch.setattr(controller._solver, "solve_into", lambda *_: -1)
    command = controller.command(
        _estimate(generated, generated.cells[0].anchor, FAST_PERIOD_S),
        np.full(3, 0.1),
        perf_counter() + 1.0,
    )
    assert command is not None
    np.testing.assert_array_equal(controller.current_reference, backup)
    assert controller.status == "active"


def test_empty_current_governor_polytope_uses_the_compiled_backup(
    generated: GeneratedAircraft,
) -> None:
    mission = _mission(generated)
    controller = _activate(
        generated,
        mission,
        _certificate(
            generated,
            matrix=np.zeros((1, 3)),
            bounds=np.array((-1.0,)),
            backup=np.full(3, 0.02),
            mission=mission,
        ),
    )
    command = controller.command(
        _estimate(generated, generated.cells[0].anchor, FAST_PERIOD_S),
        np.zeros(3),
        perf_counter() + 30.0,
    )
    assert command is not None
    np.testing.assert_array_equal(controller.current_reference, np.full(3, 0.02))
    assert controller.status == "active"


def test_raw_command_and_initial_fifo_require_command_error_margin(
    generated: GeneratedAircraft,
) -> None:
    mission = _mission(generated)
    certificate = _certificate(generated, mission=mission)
    bad_queue = np.zeros((generated.bounds.queue_length, 3))
    bad_queue[-1, 0] = generated.aircraft.control_upper_rad[0]
    bad_history = _activate(generated, mission, certificate, bad_queue)
    assert bad_history.status == "out_of_envelope"

    unsafe_reference = generated.aircraft.control_upper_rad.copy()
    raw = _activate(generated, mission, certificate, reference=unsafe_reference)
    queue = raw.issued_queue.copy()
    assert (
        raw.command(
            _estimate(generated, generated.cells[0].anchor, FAST_PERIOD_S),
            unsafe_reference,
            perf_counter() + 30.0,
        )
        is None
    )
    assert raw.status == "out_of_envelope"
    np.testing.assert_array_equal(raw.issued_queue, queue)


def test_current_prediction_rows_constrain_the_reference(
    generated: GeneratedAircraft,
) -> None:
    mission = _mission(generated)
    controller = _activate(
        generated,
        mission,
        _certificate(
            generated,
            matrix=np.array(((1.0, 0.0, 0.0),)),
            bounds=np.array((0.01,)),
            mission=mission,
        ),
    )
    command = controller.command(
        _estimate(generated, generated.cells[0].anchor, FAST_PERIOD_S),
        np.array((0.1, 0.0, 0.0)),
        perf_counter() + 30.0,
    )
    assert command is not None
    assert controller.current_reference[0] <= 0.01 + 1.0e-12


def test_measured_queue_changes_the_current_admissible_set(
    generated: GeneratedAircraft,
) -> None:
    references = []
    for value in (-0.02, 0.02):
        mission = _mission(generated)
        mission.queue_dependent = True
        queue = np.full((generated.bounds.queue_length, 3), value)
        controller = _activate(
            generated,
            mission,
            _certificate(generated, mission=mission),
            queue,
        )
        command = controller.command(
            _estimate(generated, generated.cells[0].anchor, FAST_PERIOD_S),
            np.array((0.1, 0.0, 0.0)),
            perf_counter() + 30.0,
        )
        assert command is not None
        references.append(controller.current_reference[0])

    assert references[1] > references[0] + 0.02


def test_terminal_event_is_checked_on_each_measured_segment(
    generated: GeneratedAircraft,
) -> None:
    mission = _mission(generated)
    controller = _activate(
        generated,
        mission,
        _certificate(generated, terminal=True, mission=mission),
    )
    deadline = perf_counter() + 30.0
    initial = generated.cells[0].anchor
    assert (
        controller.command(
            _estimate(generated, initial, FAST_PERIOD_S),
            np.zeros(3),
            deadline,
        )
        is not None
    )
    mission.terminal_x = 0.5 * (initial[0] + _nominal_state(controller, 1)[0])
    queue = controller.issued_queue.copy()
    assert (
        controller.command(
            _estimate(
                generated,
                _nominal_state(controller, 1),
                2.0 * FAST_PERIOD_S,
            ),
            np.zeros(3),
            deadline,
        )
        is None
    )
    assert controller.status == "terminal"
    assert mission.last_segment is not None
    assert mission.last_segment.shape == (2, 15)
    np.testing.assert_array_equal(controller.issued_queue, queue)


def test_terminal_reference_stays_inside_the_oracle_certified_hull(
    generated: GeneratedAircraft,
) -> None:
    mission = _mission(generated)
    controller = _activate(
        generated,
        mission,
        _certificate(
            generated,
            terminal=True,
            mission=mission,
            reference_lower=np.full(3, -0.01),
            reference_upper=np.full(3, 0.01),
        ),
    )

    command = controller.command(
        _estimate(generated, generated.cells[0].anchor, FAST_PERIOD_S),
        np.full(3, 0.1),
        perf_counter() + 30.0,
    )

    assert command is not None
    assert np.all(controller.current_reference >= -0.01)
    assert np.all(controller.current_reference <= 0.01)


def test_terminal_reference_is_held_for_the_complete_event_horizon(
    generated: GeneratedAircraft,
) -> None:
    mission = _mission(generated)
    controller = _activate(
        generated,
        mission,
        _certificate(generated, terminal=True, mission=mission),
    )
    deadline = perf_counter() + 30.0
    initial = generated.cells[0].anchor
    assert (
        controller.command(
            _estimate(generated, initial, FAST_PERIOD_S),
            np.full(3, 0.02),
            deadline,
        )
        is not None
    )
    reference = controller.current_reference.copy()

    for stage in range(1, PREDICTION_STAGES):
        state = _nominal_state(controller, stage)
        assert (
            controller.command(
                _estimate(generated, state, (stage + 1) * FAST_PERIOD_S),
                np.full(3, -0.02),
                deadline,
            )
            is not None
        )
        np.testing.assert_array_equal(controller.current_reference, reference)

    previous_state = _nominal_state(controller, PREDICTION_STAGES - 1)
    terminal_state = _nominal_state(controller, PREDICTION_STAGES)
    mission.terminal_x = 0.5 * (previous_state[0] + terminal_state[0])
    assert (
        controller.command(
            _estimate(
                generated,
                terminal_state,
                (PREDICTION_STAGES + 1) * FAST_PERIOD_S,
            ),
            np.zeros(3),
            deadline,
        )
        is None
    )
    assert controller.status == "terminal"
    assert mission.last_segment is not None
    assert mission.last_segment.shape == (2, 15)


def test_nonterminal_simplex_does_not_run_the_event_monitor(
    generated: GeneratedAircraft,
) -> None:
    mission = _mission(generated)
    mission.terminal_x = generated.cells[0].anchor[0]
    controller = _activate(
        generated,
        mission,
        _certificate(generated, mission=mission),
    )
    command = controller.command(
        _estimate(generated, generated.cells[0].anchor, FAST_PERIOD_S),
        np.zeros(3),
        perf_counter() + 30.0,
    )
    assert command is not None
    assert controller.status == "active"
    assert mission.last_segment is None


def test_clean_runtime_import_does_not_load_oracle_or_scipy() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        "import sys; import control.governor; "
        "assert 'control.oracle' not in sys.modules; "
        "assert 'flint' not in sys.modules; "
        "assert not any(name == 'scipy' or name.startswith('scipy.') "
        "for name in sys.modules)"
    )
    run((sys.executable, "-c", script), cwd=root, check=True)
