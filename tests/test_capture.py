"""Tests for automatic contracting-capture compilation."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from control.capture import (
    _CaptureCompiler,
    CaptureSimplex,
    compile_capture,
    triangulate_box,
)
from control.flow import FlowBounds
from control.missions import CompilationDomain, FreeSpace, Halfspaces


def _exactly_feasible(
    matrix: np.ndarray,
    reference: np.ndarray,
    bounds: np.ndarray,
) -> bool:
    for row, limit in zip(matrix, bounds, strict=True):
        value = sum(
            (
                Fraction.from_float(float(coefficient))
                * Fraction.from_float(float(component))
                for coefficient, component in zip(row, reference, strict=True)
            ),
            Fraction(0),
        )
        if value > Fraction.from_float(float(limit)):
            return False
    return True


class _Prediction:
    def __init__(
        self,
        belief: object,
        queue: np.ndarray,
        queue_radius: np.ndarray,
        approaching: bool,
    ) -> None:
        stages = 11
        self.state_center = np.repeat(belief.center[None, :], stages, axis=0)
        self.state_reference = np.zeros((stages, 15, 3))
        count = belief.generators.shape[1] + 3
        self.state_generators = np.zeros((stages, 15, count))
        self.state_generators[:, :, : belief.generators.shape[1]] = belief.generators
        self.generator_count = np.full(stages, count)
        self.body_count = 1
        direction = -1.0 if approaching else 1.0
        for stage in range(stages):
            fraction = stage / 5.0
            self.state_center[stage, 0] += direction * 0.25 * fraction
            self.state_center[stage, :3] += 0.02 * fraction * queue[0]
            self.state_generators[stage, 0, -3] = 0.02 * fraction * queue_radius[0, 0]
            self.state_reference[stage, 0, 0] = direction * 0.1 * fraction
            self.state_reference[stage, 1, 1] = -0.2 * fraction
            self.state_reference[stage, 2, 2] = -0.4 * fraction

    def state_support(
        self,
        stage: int,
        direction: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        row = np.asarray(direction, dtype=float)
        count = self.generator_count[stage]
        offset = float(row @ self.state_center[stage])
        offset += float(np.sum(np.abs(row @ self.state_generators[stage, :, :count])))
        return offset, row @ self.state_reference[stage]

    def issued_support(
        self,
        stage: int,
        direction: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        del stage
        row = np.asarray(direction, dtype=float)
        return 0.0, row.copy()

    def applied_support(
        self,
        stage: int,
        direction: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        del stage
        row = np.asarray(direction, dtype=float)
        return 0.0, row.copy()

    def body_support(
        self,
        stage: int,
        point: int,
        direction: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        del point
        row = np.zeros(15)
        row[:3] = direction
        return self.state_support(stage, row)


class _Predictor:
    def __init__(self, approaching: bool = True) -> None:
        self.approaching = approaching
        self.queues: list[tuple[float, ...]] = []
        self.queue_radii: list[tuple[float, ...]] = []
        self.prediction_verifications = 0
        self.terminal_verifications = 0

    def predict(
        self,
        belief: object,
        queue: np.ndarray,
        gain_index: int,
        issued_queue_radius: np.ndarray | None = None,
    ) -> _Prediction:
        assert gain_index == 0
        values = np.asarray(queue, dtype=float)
        radius = np.asarray(issued_queue_radius, dtype=float)
        self.queues.append(tuple(values.reshape(-1)))
        self.queue_radii.append(tuple(radius.reshape(-1)))
        return _Prediction(belief, values, radius, self.approaching)


class _Oracle:
    def __init__(self, predictor: _Predictor) -> None:
        self.predictor = predictor

    def certify_fast_prediction(self, *_: object) -> bool:
        self.predictor.prediction_verifications += 1
        return True

    def certify_terminal_event(self, *_: object) -> bool:
        self.predictor.terminal_verifications += 1
        return True


def _compiler(
    generated: object,
    predictor: _Predictor,
    mission: object,
) -> _CaptureCompiler:
    with patch("control.oracle.NonlinearOracle", return_value=_Oracle(predictor)):
        return _CaptureCompiler(generated, predictor, mission, generated.domain)


class _Generated:
    def __init__(self, domain: CompilationDomain) -> None:
        self.domain = domain
        self.domain_anchor = domain.center
        self.aircraft = SimpleNamespace(
            control_lower_rad=-np.ones(3),
            control_upper_rad=np.ones(3),
            strip_table=SimpleNamespace(r_b_m=np.zeros((1, 3))),
        )
        self.bounds = SimpleNamespace(
            flow=FlowBounds(
                np.zeros(3),
                np.zeros(3),
                np.zeros((3, 3)),
                np.zeros((3, 3)),
                np.zeros(3),
            ),
            queue_length=1,
            command_error_abs_rad=np.zeros(3),
            state_estimation_abs=np.zeros(15),
            roll_abs_max_rad=1.0,
            pitch_abs_max_rad=1.0,
            airspeed_m_s=(0.1, 10.0),
            alpha_abs_max_rad=1.0,
            body_rate_abs_max_rad_s=5.0,
        )
        self.reference_center = np.zeros(3)
        self.reference_scale = np.ones(3)
        self.state_scale = np.ones(15)
        self.cells = (
            SimpleNamespace(
                anchor=domain.center,
                lower=-10.0 * np.ones(11),
                upper=10.0 * np.ones(11),
            ),
        )

    def cell(self, state: np.ndarray) -> tuple[int, object]:
        del state
        return 0, self.cells[0]


@dataclass
class _Mission:
    terminal_set: Halfspaces
    free_space_halfspaces: Halfspaces

    def terminal_halfspaces(self, generated: object) -> Halfspaces:
        del generated
        return self.terminal_set

    def error(self, state: np.ndarray, generated: object) -> np.ndarray:
        del generated
        value = np.asarray(state, dtype=float)
        return value[1:3].copy()

    def distance(self, state: np.ndarray) -> float:
        return float(np.asarray(state, dtype=float)[0])

    def realized(self, states: np.ndarray, generated: object) -> bool:
        del generated
        return bool(np.asarray(states)[-1, 0] <= 0.0)

    def terminal_support_constraints(
        self,
        prediction: _Prediction,
        generated: object,
        domain: CompilationDomain,
    ) -> tuple[np.ndarray, np.ndarray]:
        del generated, domain
        direction = np.zeros(15)
        direction[0] = 1.0
        offset, reference = prediction.state_support(10, direction)
        return reference.reshape(1, 3), np.array([-offset])


class _DependentMission(_Mission):
    def error(self, state: np.ndarray, generated: object) -> np.ndarray:
        del generated
        value = np.asarray(state, dtype=float)
        return np.array((-value[0], value[1], value[2]))


def _error_support(
    mission: _Mission,
    prediction: _Prediction,
    stage: int,
    facet: np.ndarray,
    generated: object,
    domain: CompilationDomain,
) -> tuple[float, np.ndarray]:
    del mission, generated, domain
    direction = np.zeros(15)
    direction[1:3] = facet
    return prediction.state_support(stage, direction)


def _distance_support(
    mission: _Mission,
    prediction: _Prediction,
    stage: int,
    sign: float,
) -> tuple[float, np.ndarray]:
    del mission
    direction = np.zeros(15)
    direction[0] = sign
    return prediction.state_support(stage, direction)


def _preterminal_support(
    mission: _Mission,
    prediction: _Prediction,
    generated: object,
) -> tuple[np.ndarray, np.ndarray]:
    del mission, prediction, generated
    return np.empty((0, 3)), np.empty(0)


def _fixture(approaching: bool = True) -> tuple[object, _Predictor, _Mission]:
    lower = -0.05 * np.ones(15)
    upper = 0.05 * np.ones(15)
    lower[0], upper[0] = 0.0, 2.0
    lower[1:3], upper[1:3] = -0.2, 0.2
    lower[6], upper[6] = 0.9, 1.1
    domain = CompilationDomain(lower, upper)
    mission = _Mission(
        Halfspaces.box((-0.35, -0.35), (0.35, 0.35)),
        FreeSpace.box((-100.0,) * 3, (100.0,) * 3).halfspaces,
    )
    return _Generated(domain), _Predictor(approaching), mission


def _compile(approaching: bool = True) -> tuple[object, _Predictor]:
    generated, predictor, mission = _fixture(approaching)

    with (
        patch("control.capture.error_support", _error_support),
        patch("control.capture.distance_support", _distance_support),
        patch("control.capture.preterminal_support_constraints", _preterminal_support),
    ):
        certificate = _compiler(generated, predictor, mission).compile()
    return certificate, predictor


@pytest.fixture(scope="module")
def compiled() -> tuple[object, _Predictor]:
    return _compile()


def test_terminal_equality_and_monotone_expansion(
    compiled: tuple[object, _Predictor],
) -> None:
    """Recover the exact terminal set and expand every generated facet."""

    certificate, _ = compiled
    np.testing.assert_array_equal(
        certificate.capture_bounds(0.0),
        certificate.terminal_bounds,
    )
    near = certificate.capture_bounds(0.25 * certificate.distance_max)
    far = certificate.capture_bounds(0.75 * certificate.distance_max)
    assert np.all(near >= certificate.terminal_bounds)
    assert np.all(far >= near)


def test_every_compiled_cell_uses_prediction_and_terminal_verification(
    compiled: tuple[object, _Predictor],
) -> None:
    certificate, predictor = compiled
    assert predictor.prediction_verifications >= len(certificate.simplices)
    assert predictor.terminal_verifications > 0


def test_terminal_reference_hull_encloses_the_complete_polytope() -> None:
    generated, predictor, mission = _fixture()
    compiler = _compiler(generated, predictor, mission)
    matrix = np.array(
        (
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        )
    )
    bounds = np.array((0.3, 0.4, 0.2, 0.5, 0.6, 0.7))
    hull = compiler._reference_hull(matrix, bounds)
    assert hull is not None
    assert np.all(hull.lower <= (-0.4, -0.5, -0.7))
    assert np.all(hull.upper >= (0.3, 0.2, 0.6))


def test_compiler_retains_numerically_close_nonredundant_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated, predictor, mission = _fixture()
    compiler = _compiler(generated, predictor, mission)
    matrix = np.array(
        (
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
        )
    )
    bounds = np.array((0.0, 0.0, 5.0e-11))
    monkeypatch.setattr(
        compiler.constraint_builder,
        "rows",
        lambda *_: (matrix.copy(), bounds.copy()),
    )

    simplex = compiler._certify_cell(
        triangulate_box(-np.ones(3), np.ones(3))[0],
        np.ones(4),
        0.1,
        False,
    )

    assert simplex is not None
    assert simplex.constraint_count == matrix.shape[0]
    assert _exactly_feasible(matrix, simplex.backup(), bounds)
    point = np.array((5.0e-11, -5.0e-11, 0.0))
    assert np.any(matrix @ point > bounds)


def test_beta_follows_facet_authority_over_worst_approach_rate(
    compiled: tuple[object, _Predictor],
) -> None:
    """Preserve the relative correction authority of the two error axes."""

    certificate, _ = compiled
    assert np.all(certificate.beta > 0.0)
    np.testing.assert_allclose(certificate.beta[2:], certificate.beta[:2])
    np.testing.assert_allclose(
        certificate.beta[1] / certificate.beta[0],
        2.0,
    )


def test_uniform_cell_backups_are_feasible_everywhere() -> None:
    """Verify each cell backup against its complete-cell inequalities."""

    generated, predictor, mission = _fixture()
    with (
        patch("control.capture.error_support", _error_support),
        patch("control.capture.distance_support", _distance_support),
        patch("control.capture.preterminal_support_constraints", _preterminal_support),
    ):
        compiler = _compiler(generated, predictor, mission)
        certificate = compiler.compile()
        for simplex in certificate.simplices:
            belief, gain_index = compiler._cell_belief(simplex.vertices)
            prediction = predictor.predict(
                belief,
                compiler.queue_center,
                gain_index,
                compiler.queue_radius,
            )
            matrix, bounds = compiler.constraint_builder.rows(
                prediction,
                certificate.beta,
                simplex.progress_m,
                simplex.terminal,
            )
            reference = simplex.backup()
            assert simplex.constraint_count == matrix.shape[0]
            np.testing.assert_array_equal(simplex.backup_reference, reference)
            assert np.max(matrix @ reference - bounds) <= 1.0e-15


def test_complete_belief_simplex_containment_is_not_center_only(
    compiled: tuple[object, _Predictor],
) -> None:
    """Accept only coordinate zonotopes wholly contained in one simplex."""

    certificate, _ = compiled
    simplex = certificate.simplices[0]
    normalized = np.mean(simplex.vertices, axis=0)
    coordinates = (
        certificate.coordinate_offset + certificate.coordinate_scale * normalized
    )
    generators = 1.0e-6 * certificate.coordinate_scale[:, None] * np.ones((3, 1))
    index, weights = certificate.locate(coordinates, generators)
    assert index == 0
    assert np.all(weights > 0.0)
    augmented = certificate.augmented_offset.copy()
    augmented[:3] += certificate.coordinate_scale * normalized
    augmented_generators = np.zeros((augmented.size, 1))
    augmented_generators[:3] = generators
    augmented_index, _ = certificate.locate_augmented(
        augmented,
        augmented_generators,
    )
    assert augmented_index == index
    assert np.all(certificate.coordinate_transform[:, 15:] == 0.0)
    with pytest.raises(ValueError, match="outside"):
        certificate.locate(
            coordinates,
            2.0 * certificate.coordinate_scale[:, None] * np.ones((3, 1)),
        )


def test_subdivision_tree_locates_every_compiled_simplex(
    compiled: tuple[object, _Predictor],
) -> None:
    certificate, _ = compiled
    for expected, simplex in enumerate(certificate.simplices):
        normalized = np.mean(simplex.vertices, axis=0)
        coordinates = (
            certificate.coordinate_offset + certificate.coordinate_scale * normalized
        )
        actual, _ = certificate.locate(coordinates)
        assert actual == expected


def test_complete_queue_zonotope_is_compiled_without_corner_sampling(
    compiled: tuple[object, _Predictor],
) -> None:
    """Pass the complete issued-command box as independent generators."""

    _, predictor = compiled
    assert set(predictor.queues) == {(0.0, 0.0, 0.0)}
    assert set(predictor.queue_radii) == {(1.0, 1.0, 1.0)}


def test_online_row_counts_and_positive_nonterminal_progress(
    compiled: tuple[object, _Predictor],
) -> None:
    """Store only row counts and generated progress for running cells."""

    certificate, _ = compiled
    assert certificate.max_constraints == max(
        simplex.constraint_count for simplex in certificate.simplices
    )
    nonterminal = [simplex for simplex in certificate.simplices if not simplex.terminal]
    assert nonterminal
    assert all(simplex.progress_m > 0.0 for simplex in nonterminal)


def test_degenerate_domains_and_simplices_are_rejected() -> None:
    """Reject deterministic zero-volume inputs instead of inverting them."""

    with pytest.raises(ValueError, match="upper"):
        triangulate_box(np.zeros(3), np.ones(3) * (1.0, 0.0, 1.0))
    vertices = np.zeros((4, 3))
    with pytest.raises(ValueError, match="dependent"):
        CaptureSimplex(
            vertices,
            np.zeros(3),
            1,
            0,
            0.1,
            False,
        )

    vertices = np.vstack((np.zeros(3), np.eye(3)))
    with pytest.raises(ValueError, match="constraint count"):
        CaptureSimplex(
            vertices,
            np.zeros(3),
            0,
            0,
            0.1,
            False,
        )


def test_dependent_distance_error_row_is_skipped_automatically() -> None:
    """Select the first independent error pair for a landing-like descriptor."""

    import control.capture as capture

    generated, _, mission = _fixture()
    dependent = _DependentMission(
        Halfspaces.box((-2.0, -0.35, -0.35), (0.0, 0.35, 0.35)),
        mission.free_space_halfspaces,
    )
    coordinates = capture._coordinate_map(generated, dependent, generated.domain)
    assert coordinates.error_indices == (1, 2)
    assert np.linalg.matrix_rank(coordinates.transform[:, :15]) == 3


def test_explicitly_infeasible_aircraft_mission_pair_is_rejected() -> None:
    """Refuse a mission whose predictor cannot make positive approach progress."""

    generated, predictor, mission = _fixture(False)

    with (
        patch("control.capture.error_support", _error_support),
        patch("control.capture.distance_support", _distance_support),
        patch("control.capture.preterminal_support_constraints", _preterminal_support),
        pytest.raises(ValueError, match="approach rate"),
    ):
        _compiler(generated, predictor, mission).compile()


def test_public_compiler_requires_a_verified_aircraft_core() -> None:
    generated, predictor, mission = _fixture()
    with pytest.raises(TypeError, match="oracle-verified"):
        compile_capture(generated, predictor, mission)


def test_hard_airspeed_and_alpha_rows_include_cg_flow() -> None:
    import control.capture as capture

    generated, _, _ = _fixture()
    generated.bounds.flow = FlowBounds(
        (1.0, -0.5, 0.2),
        (2.0, 0.5, 0.4),
        np.zeros((3, 3)),
        np.zeros((3, 3)),
        np.zeros(3),
    )
    matrix, limits = capture._hard_halfspaces(generated)
    direction = matrix[4, 6:9]
    flow = generated.bounds.flow.joint_zonotope(
        generated.aircraft.strip_table.r_b_m
    ).interval_hull()
    flow_support = float(
        (-direction) @ flow.center[:3] + np.abs(direction) @ flow.radius[:3]
    )
    assert limits[4] == pytest.approx(generated.bounds.airspeed_m_s[1] - flow_support)
    assert limits[12] == pytest.approx(
        -generated.bounds.airspeed_m_s[0] - flow.upper[0]
    )
