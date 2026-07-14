"""Automatic compilation of contracting terminal-capture certificates."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations, product
from math import tan

import numpy as np
import numpy.typing as npt

from control.interval import Interval, Zonotope
from control.missions import (
    GateMission,
    LandingMission,
    Mission,
    distance_support,
    error_support,
    preterminal_support_constraints,
)
from control.predictor import (
    INTRINSIC_FEEDBACK_INDICES,
    FastPredictor,
    GeneratedAircraft,
    Prediction,
)
from control.uncertainty import NEXT_UPDATE_STAGE, PREDICTION_STAGES

BOUNDARY_TOLERANCE = 64.0 * np.finfo(float).eps
POLYTOPE_TOLERANCE = 1.0e-10
BETA_BRACKET_LEVELS = 32
BETA_SEARCH_ITERATIONS = 32
MAX_SUBDIVISION_DEPTH = 8


@dataclass(frozen=True)
class CaptureSimplex:
    """Certified simplex with a complete-cell backup and uniform row bounds."""

    vertices: npt.ArrayLike
    backup_references: npt.ArrayLike
    constraint_matrix: npt.ArrayLike
    constraint_bounds: npt.ArrayLike
    gain_index: int
    progress_m: float
    terminal: bool
    _barycentric_matrix: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=float)
        if vertices.ndim != 2 or vertices.shape[0] != vertices.shape[1] + 1:
            raise ValueError("simplex must have dimension plus one vertices")
        dimension = vertices.shape[1]
        vertices = vertices.copy()
        backup = (
            np.asarray(self.backup_references, dtype=float)
            .reshape(
                dimension + 1,
                3,
            )
            .copy()
        )
        if not np.array_equal(backup, np.broadcast_to(backup[0], backup.shape)):
            raise ValueError("simplex backup must be constant over the complete cell")
        matrix = np.asarray(self.constraint_matrix, dtype=float).reshape(-1, 3).copy()
        bounds = (
            np.asarray(self.constraint_bounds, dtype=float)
            .reshape(
                dimension + 1,
                matrix.shape[0],
            )
            .copy()
        )
        if not np.all(bounds == bounds[0]):
            raise ValueError("runtime constraint bounds must be uniform in a simplex")
        augmented = np.column_stack((vertices, np.ones(dimension + 1)))
        singular = np.linalg.svd(augmented, compute_uv=False)
        if singular[-1] <= BOUNDARY_TOLERANCE * singular[0]:
            raise ValueError("simplex vertices are affinely dependent")
        barycentric = np.linalg.inv(augmented).T
        for value in (vertices, backup, matrix, bounds, barycentric):
            value.flags.writeable = False
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "backup_references", backup)
        object.__setattr__(self, "constraint_matrix", matrix)
        object.__setattr__(self, "constraint_bounds", bounds)
        object.__setattr__(self, "_barycentric_matrix", barycentric)

    def barycentric(self, normalized: npt.ArrayLike) -> np.ndarray:
        """Return barycentric coordinates in this simplex."""

        point = np.asarray(normalized, dtype=float).reshape(self.vertices.shape[1])
        return (
            self._barycentric_matrix[:, :-1] @ point + self._barycentric_matrix[:, -1]
        )

    def barycentric_into(
        self,
        normalized: npt.ArrayLike,
        result: np.ndarray,
    ) -> None:
        """Write barycentric coordinates into a fixed runtime buffer."""

        np.matmul(
            self._barycentric_matrix[:, :-1],
            normalized,
            out=result,
        )
        result += self._barycentric_matrix[:, -1]

    def contains(
        self,
        normalized_center: npt.ArrayLike,
        normalized_generators: npt.ArrayLike | None = None,
    ) -> bool:
        """Return whether a normalized belief is wholly in the simplex."""

        weights = self.barycentric(normalized_center)
        if normalized_generators is None:
            return bool(np.min(weights) >= -BOUNDARY_TOLERANCE)
        generators = np.asarray(normalized_generators, dtype=float).reshape(
            self.vertices.shape[1],
            -1,
        )
        linear = self._barycentric_matrix[:, :-1]
        radius = np.sum(np.abs(linear @ generators), axis=1)
        return bool(np.min(weights - radius) >= -BOUNDARY_TOLERANCE)

    def backup(self) -> np.ndarray:
        """Return the complete-cell backup reference."""

        return self.backup_references[0].copy()

    def backup_into(self, result: np.ndarray) -> None:
        """Write the complete-cell backup into a runtime buffer."""

        result[:] = self.backup_references[0]


@dataclass(frozen=True)
class ActiveSets:
    """Feasible face and edge indices plus unique polytope vertices."""

    one: npt.ArrayLike
    two: npt.ArrayLike
    vertices: npt.ArrayLike

    def __post_init__(self) -> None:
        for name, width in (("one", 1), ("two", 2)):
            value = np.asarray(getattr(self, name), dtype=int).reshape(-1, width).copy()
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        vertices = np.asarray(self.vertices, dtype=float).reshape(-1, 3).copy()
        vertices.flags.writeable = False
        object.__setattr__(self, "vertices", vertices)


@dataclass(frozen=True)
class CaptureCertificate:
    """Complete mission capture family and certified simplicial cover."""

    beta: npt.ArrayLike
    coordinate_offset: npt.ArrayLike
    coordinate_scale: npt.ArrayLike
    reference_lower: npt.ArrayLike
    reference_upper: npt.ArrayLike
    simplices: tuple[CaptureSimplex, ...]
    terminal_bounds: npt.ArrayLike
    distance_max: float
    augmented_offset: npt.ArrayLike
    augmented_scale: npt.ArrayLike
    coordinate_transform: npt.ArrayLike
    mission: Mission
    generated: GeneratedAircraft
    _active_sets: tuple[ActiveSets, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.simplices:
            raise ValueError("capture certificate requires a certified simplex")
        beta = np.asarray(self.beta, dtype=float).reshape(-1).copy()
        dimension = self.simplices[0].vertices.shape[1]
        offset = (
            np.asarray(self.coordinate_offset, dtype=float).reshape(dimension).copy()
        )
        scale = np.asarray(self.coordinate_scale, dtype=float).reshape(dimension).copy()
        lower = np.asarray(self.reference_lower, dtype=float).reshape(3).copy()
        upper = np.asarray(self.reference_upper, dtype=float).reshape(3).copy()
        terminal = (
            np.asarray(self.terminal_bounds, dtype=float).reshape(beta.shape).copy()
        )
        augmented_offset = (
            np.asarray(self.augmented_offset, dtype=float).reshape(-1).copy()
        )
        augmented_scale = (
            np.asarray(self.augmented_scale, dtype=float)
            .reshape(augmented_offset.shape)
            .copy()
        )
        transform = (
            np.asarray(self.coordinate_transform, dtype=float)
            .reshape(
                dimension,
                augmented_offset.size,
            )
            .copy()
        )
        if np.any(scale <= 0.0) or np.any(augmented_scale <= 0.0):
            raise ValueError("normalization scales must be positive")
        if np.any(beta < 0.0) or self.distance_max <= 0.0:
            raise ValueError("capture slope and distance must be positive")
        row_count = self.simplices[0].constraint_matrix.shape[0]
        if any(
            simplex.vertices.shape[1] != dimension
            or simplex.constraint_matrix.shape[0] != row_count
            for simplex in self.simplices
        ):
            raise ValueError("certificate simplices must have fixed dimensions")
        for value in (
            beta,
            offset,
            scale,
            lower,
            upper,
            terminal,
            augmented_offset,
            augmented_scale,
            transform,
        ):
            value.flags.writeable = False
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "coordinate_offset", offset)
        object.__setattr__(self, "coordinate_scale", scale)
        object.__setattr__(self, "reference_lower", lower)
        object.__setattr__(self, "reference_upper", upper)
        object.__setattr__(self, "terminal_bounds", terminal)
        object.__setattr__(self, "augmented_offset", augmented_offset)
        object.__setattr__(self, "augmented_scale", augmented_scale)
        object.__setattr__(self, "coordinate_transform", transform)
        cache: dict[tuple[bytes, bytes], ActiveSets] = {}
        active_sets = []
        for simplex in self.simplices:
            key = (
                simplex.constraint_matrix.tobytes(),
                simplex.constraint_bounds[0].tobytes(),
            )
            active = cache.get(key)
            if active is None:
                active = _feasible_active_sets(
                    simplex.constraint_matrix,
                    simplex.constraint_bounds[0],
                    lower,
                    upper,
                )
                cache[key] = active
            active_sets.append(active)
        object.__setattr__(self, "_active_sets", tuple(active_sets))

    @property
    def max_constraints(self) -> int:
        """Return the fixed row count used by the runtime solver."""

        return self.simplices[0].constraint_matrix.shape[0]

    def capture_bounds(self, distance: float) -> np.ndarray:
        """Return the contracting terminal-facet bounds at one distance."""

        return self.terminal_bounds + self.beta * min(distance, self.distance_max)

    def active_sets(self, simplex_index: int) -> ActiveSets:
        """Return the precomputed feasible active sets for one simplex."""

        return self._active_sets[simplex_index]

    def normalize(self, coordinates: npt.ArrayLike) -> np.ndarray:
        """Return normalized mission coordinates."""

        values = np.asarray(coordinates, dtype=float).reshape(
            self.coordinate_offset.shape
        )
        return (values - self.coordinate_offset) / self.coordinate_scale

    def locate(
        self,
        coordinates: npt.ArrayLike,
        coordinate_generators: npt.ArrayLike | None = None,
    ) -> tuple[int, np.ndarray]:
        """Locate a complete coordinate belief with deterministic boundaries."""

        normalized = self.normalize(coordinates)
        generators = None
        if coordinate_generators is not None:
            generators = (
                np.asarray(coordinate_generators, dtype=float)
                / (self.coordinate_scale[:, None])
            )
        for index, simplex in enumerate(self.simplices):
            if simplex.contains(normalized, generators):
                return index, simplex.barycentric(normalized)
        raise ValueError("belief is outside the compiled capture region")

    def locate_augmented(
        self,
        augmented_center: npt.ArrayLike,
        augmented_generators: npt.ArrayLike | None = None,
    ) -> tuple[int, np.ndarray]:
        """Locate a complete physical augmented-state belief."""

        center = np.asarray(augmented_center, dtype=float).reshape(
            self.augmented_offset.shape
        )
        normalized = self.coordinate_transform @ (
            (center - self.augmented_offset) / self.augmented_scale
        )
        generators = None
        if augmented_generators is not None:
            physical = np.asarray(augmented_generators, dtype=float).reshape(
                center.size,
                -1,
            )
            generators = self.coordinate_transform @ (
                physical / self.augmented_scale[:, None]
            )
        for index, simplex in enumerate(self.simplices):
            if simplex.contains(normalized, generators):
                return index, simplex.barycentric(normalized)
        raise ValueError("belief is outside the compiled capture region")

    def locate_augmented_into(
        self,
        augmented_center: np.ndarray,
        augmented_normalized: np.ndarray,
        coordinate_normalized: np.ndarray,
        weights: np.ndarray,
    ) -> int:
        """Locate one runtime center using caller-owned work arrays."""

        np.subtract(
            augmented_center,
            self.augmented_offset,
            out=augmented_normalized,
        )
        augmented_normalized /= self.augmented_scale
        np.matmul(
            self.coordinate_transform,
            augmented_normalized,
            out=coordinate_normalized,
        )
        for index, simplex in enumerate(self.simplices):
            simplex.barycentric_into(coordinate_normalized, weights)
            inside = True
            for value in weights:
                if value < -BOUNDARY_TOLERANCE:
                    inside = False
                    break
            if inside:
                return index
        return -1


@dataclass(frozen=True)
class _CoordinateMap:
    offset: np.ndarray
    scale: np.ndarray
    transform: np.ndarray
    state_lift: np.ndarray
    state_generators: np.ndarray
    augmented_offset: np.ndarray
    augmented_scale: np.ndarray
    error_indices: tuple[int, int]
    distance_jacobian: np.ndarray
    error_jacobian: np.ndarray


class _CaptureCompiler:
    def __init__(
        self,
        generated: GeneratedAircraft,
        predictor: FastPredictor,
        mission: Mission,
    ) -> None:
        from control.oracle import NonlinearOracle

        if isinstance(mission, (GateMission, LandingMission)):
            values = (
                mission.body_inflation_m,
                mission.position_error_m,
                mission.attitude_error_rad,
            )
            expected = (
                generated.bounds.body_inflation_m,
                generated.bounds.mission_position_error_abs_m,
                generated.bounds.mission_attitude_error_abs_rad,
            )
            if values != expected:
                raise ValueError("mission uncertainty differs from aircraft bounds")
            if isinstance(mission, GateMission):
                flow_center = 0.5 * (
                    generated.bounds.flow.center_lower_m_s
                    + generated.bounds.flow.center_upper_m_s
                )
                if not np.array_equal(mission.center_flow_b_m_s, flow_center):
                    raise ValueError("mission flow center differs from aircraft bounds")
        self.generated = generated
        self.predictor = predictor
        self.mission = mission
        from scipy.optimize import linprog  # type: ignore[import-untyped]

        self.linear_program = linprog
        self.oracle = NonlinearOracle(generated)
        self.reference_lower = np.asarray(
            generated.aircraft.control_lower_rad,
            dtype=float,
        )
        self.reference_upper = np.asarray(
            generated.aircraft.control_upper_rad,
            dtype=float,
        )
        self.reference_center = 0.5 * (self.reference_lower + self.reference_upper)
        self.reference_scale = 0.5 * (self.reference_upper - self.reference_lower)
        self.coordinates = _coordinate_map(generated, mission)
        queue_lower = self.reference_lower + generated.bounds.command_error_abs_rad
        queue_upper = self.reference_upper - generated.bounds.command_error_abs_rad
        self.queue_center = np.tile(
            0.5 * (queue_lower + queue_upper),
            (generated.bounds.queue_length, 1),
        )
        self.queue_radius = np.tile(
            0.5 * (queue_upper - queue_lower),
            (generated.bounds.queue_length, 1),
        )
        self.queue_intervals = tuple(
            Interval(lower, upper)
            for lower, upper in zip(
                self.queue_center - self.queue_radius,
                self.queue_center + self.queue_radius,
                strict=True,
            )
        )
        self.reference_interval = Interval(self.reference_lower, self.reference_upper)
        self.hard_matrix, self.hard_bounds = _hard_halfspaces(generated)

    def compile(self) -> CaptureCertificate:
        lower = self.coordinates.offset - self.coordinates.scale
        upper = self.coordinates.offset + self.coordinates.scale
        if lower[0] < -BOUNDARY_TOLERANCE or upper[0] <= 0.0:
            raise ValueError("approach distance must span a positive interval")
        normalized_cover = triangulate_box(-np.ones(3), np.ones(3))
        authority, worst_approach, terminal_reach = self._authority(normalized_cover)
        if worst_approach <= BOUNDARY_TOLERANCE:
            raise ValueError("mission has no certified approach rate")
        slope = authority / worst_approach
        cover = _layered_cover(
            self.coordinates.offset,
            self.coordinates.scale,
            terminal_reach,
        )
        verified = self._compile_cover(
            cover,
            slope,
            terminal_reach,
            worst_approach,
        )
        if verified:
            beta = slope
        else:
            upper_scale = 1.0
            lower_scale = -1.0
            for level in range(1, BETA_BRACKET_LEVELS + 1):
                scale = 1.0 - level / BETA_BRACKET_LEVELS
                candidate = scale * slope
                compiled = self._compile_cover(
                    cover,
                    candidate,
                    terminal_reach,
                    worst_approach,
                )
                if compiled:
                    lower_scale = scale
                    beta = candidate
                    verified = compiled
                    break
                upper_scale = scale
            if lower_scale < 0.0:
                raise ValueError("aircraft and mission have no complete capture cover")
            for _ in range(BETA_SEARCH_ITERATIONS):
                scale = 0.5 * (lower_scale + upper_scale)
                candidate = scale * slope
                compiled = self._compile_cover(
                    cover,
                    candidate,
                    terminal_reach,
                    worst_approach,
                )
                if compiled:
                    lower_scale = scale
                    beta = candidate
                    verified = compiled
                else:
                    upper_scale = scale
        verified = self._compile_cover(
            cover,
            beta,
            terminal_reach,
            worst_approach,
        )
        simplices = _pad_rows(verified)

        terminal = self.mission.terminal_halfspaces
        return CaptureCertificate(
            beta,
            self.coordinates.offset,
            self.coordinates.scale,
            self.reference_lower,
            self.reference_upper,
            simplices,
            terminal.bounds,
            float(upper[0]),
            self.coordinates.augmented_offset,
            self.coordinates.augmented_scale,
            self.coordinates.transform,
            self.mission,
            self.generated,
        )

    def _authority(
        self,
        cover: tuple[np.ndarray, ...],
    ) -> tuple[np.ndarray, float, float]:
        vertices = sorted({tuple(vertex) for simplex in cover for vertex in simplex})
        facets = self.mission.terminal_halfspaces.matrix
        authority = np.full(facets.shape[0], np.inf)
        worst_approach = 0.0
        terminal_reach = np.inf
        for normalized in vertices:
            value = np.asarray(normalized, dtype=float)
            belief, gain_index = self._belief(value)
            distance = self.coordinates.offset[0] + self.coordinates.scale[0] * value[0]
            prediction = self.predictor.predict(
                belief,
                self.queue_center,
                gain_index,
                self.queue_radius,
            )
            for index, facet in enumerate(facets):
                _, coefficient = error_support(
                    self.mission,
                    prediction,
                    NEXT_UPDATE_STAGE,
                    facet,
                )
                correction = 2.0 * float(np.abs(coefficient) @ self.reference_scale)
                authority[index] = min(authority[index], correction)
            offset, coefficient = distance_support(
                self.mission,
                prediction,
                NEXT_UPDATE_STAGE,
                -1.0,
            )
            lower_distance = -offset - _box_support(
                coefficient,
                self.reference_center,
                self.reference_scale,
            )
            worst_approach = max(
                worst_approach,
                distance - lower_distance,
            )
            offset, coefficient = distance_support(
                self.mission,
                prediction,
                PREDICTION_STAGES,
                1.0,
            )
            best_upper = offset + float(
                coefficient @ self.reference_center
                - np.abs(coefficient) @ self.reference_scale
            )
            terminal_reach = min(
                terminal_reach,
                distance - best_upper,
            )
        return authority, worst_approach, terminal_reach

    def _compile_cover(
        self,
        cover: tuple[np.ndarray, ...],
        beta: np.ndarray,
        terminal_reach: float,
        worst_approach: float,
    ) -> tuple[CaptureSimplex, ...]:
        pending = deque((vertices, 0) for vertices in cover)
        accepted: list[CaptureSimplex] = []
        while pending:
            vertices, depth = pending.popleft()
            distance = (
                self.coordinates.offset[0] + self.coordinates.scale[0] * vertices[:, 0]
            )
            terminal_first = float(np.max(distance)) <= (
                terminal_reach + BOUNDARY_TOLERANCE
            )
            modes = (True, False) if terminal_first else (False, True)
            certified = None
            for terminal in modes:
                if not terminal and np.min(distance) <= BOUNDARY_TOLERANCE:
                    continue
                progress = 0.0
                if not terminal:
                    progress = min(
                        0.25 * worst_approach,
                        0.5 * float(np.min(distance)),
                    )
                certified = self._certify_cell(
                    vertices,
                    beta,
                    progress,
                    terminal,
                )
                if certified is not None:
                    break
            if certified is not None:
                accepted.append(certified)
                continue
            if depth == MAX_SUBDIVISION_DEPTH:
                return ()
            left, right = subdivide_simplex(vertices)
            pending.append((left, depth + 1))
            pending.append((right, depth + 1))
        return tuple(accepted)

    def _certify_cell(
        self,
        vertices: np.ndarray,
        beta: np.ndarray,
        progress: float,
        terminal: bool,
    ) -> CaptureSimplex | None:
        try:
            belief, gain_index = self._cell_belief(vertices)
        except ValueError:
            return None
        prediction = self.predictor.predict(
            belief,
            self.queue_center,
            gain_index,
            self.queue_radius,
        )
        cell = self.generated.cells[gain_index]
        if not self.oracle.certify_fast_prediction(
            belief,
            self.queue_intervals,
            cell,
            self.reference_interval,
            prediction,
        ):
            return None
        matrix, limits = self._rows(
            prediction,
            beta,
            progress,
            terminal,
        )
        if not np.all(np.isfinite(limits)):
            return None
        for index, row in enumerate(matrix):
            if np.any(row):
                continue
            if limits[index] < -POLYTOPE_TOLERANCE:
                return None
            limits[index] = 0.0
        backup = self._linear_feasibility(matrix, limits)
        if backup is None:
            return None
        if terminal:
            reference = self._reference_hull(matrix, limits)
            if reference is None:
                return None
            reference = Interval(
                np.minimum(reference.lower, backup),
                np.maximum(reference.upper, backup),
            )
            if not self.oracle.certify_terminal_event(
                belief,
                self.queue_intervals,
                cell,
                reference,
                self.mission,
            ):
                return None
        backups = np.tile(backup, (vertices.shape[0], 1))
        return CaptureSimplex(
            vertices,
            backups,
            matrix,
            np.tile(limits, (vertices.shape[0], 1)),
            gain_index,
            progress,
            terminal,
        )

    def _belief(
        self,
        normalized: np.ndarray,
        required_gain: int | None = None,
    ) -> tuple[Zonotope, int]:
        center = (
            self.mission.approach_domain.center
            + self.coordinates.state_lift @ normalized
        )
        belief = Zonotope(center, self.coordinates.state_generators)
        gain_index, cell = self.generated.cell(center)
        if required_gain is not None and gain_index != required_gain:
            raise ValueError("simplex spans generated gain cells")
        intrinsic = (
            center[INTRINSIC_FEEDBACK_INDICES] - cell.anchor[INTRINSIC_FEEDBACK_INDICES]
        ) / self.generated.state_scale[INTRINSIC_FEEDBACK_INDICES]
        generators = (
            self.coordinates.state_generators[INTRINSIC_FEEDBACK_INDICES]
            / self.generated.state_scale[INTRINSIC_FEEDBACK_INDICES, None]
        )
        radius = np.sum(np.abs(generators), axis=1)
        if np.any(intrinsic - radius < cell.lower - BOUNDARY_TOLERANCE):
            raise ValueError("state fiber leaves generated gain cell")
        if np.any(intrinsic + radius > cell.upper + BOUNDARY_TOLERANCE):
            raise ValueError("state fiber leaves generated gain cell")
        return belief, gain_index

    def _cell_belief(
        self,
        vertices: np.ndarray,
    ) -> tuple[Zonotope, int]:
        lower = np.min(vertices, axis=0)
        upper = np.max(vertices, axis=0)
        normalized = 0.5 * (lower + upper)
        center = (
            self.mission.approach_domain.center
            + self.coordinates.state_lift @ normalized
        )
        deviations = self.coordinates.state_lift @ np.diag(0.5 * (upper - lower))
        generators = np.column_stack((self.coordinates.state_generators, deviations))
        belief = Zonotope(center, generators)
        gain_index, cell = self.generated.cell(center)
        intrinsic = (
            center[INTRINSIC_FEEDBACK_INDICES] - cell.anchor[INTRINSIC_FEEDBACK_INDICES]
        ) / self.generated.state_scale[INTRINSIC_FEEDBACK_INDICES]
        normalized_generators = (
            generators[INTRINSIC_FEEDBACK_INDICES]
            / self.generated.state_scale[INTRINSIC_FEEDBACK_INDICES, None]
        )
        radius = np.sum(np.abs(normalized_generators), axis=1)
        if np.any(intrinsic - radius < cell.lower - BOUNDARY_TOLERANCE):
            raise ValueError("simplex belief leaves generated gain cell")
        if np.any(intrinsic + radius > cell.upper + BOUNDARY_TOLERANCE):
            raise ValueError("simplex belief leaves generated gain cell")
        return belief, gain_index

    def _rows(
        self,
        prediction: Prediction,
        beta: np.ndarray,
        progress: float,
        terminal: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        matrices: list[np.ndarray] = []
        bounds: list[float] = []

        command_axes = np.vstack((np.eye(3), -np.eye(3)))
        command_limits = np.concatenate((self.reference_upper, -self.reference_lower))
        for stage in range(PREDICTION_STAGES):
            for direction, limit in zip(
                command_axes,
                command_limits,
                strict=True,
            ):
                offset, reference = prediction.issued_support(stage, direction)
                matrices.append(reference)
                bounds.append(float(limit - offset))
                offset, reference = prediction.applied_support(stage, direction)
                matrices.append(reference)
                bounds.append(float(limit - offset))

        for stage in range(1, PREDICTION_STAGES + 1):
            for direction, limit in zip(
                self.hard_matrix,
                self.hard_bounds,
                strict=True,
            ):
                offset, reference = prediction.state_support(stage, direction)
                matrices.append(reference)
                bounds.append(float(limit - offset))

        free = self.mission.free_space_halfspaces
        for stage in range(PREDICTION_STAGES):
            for point in range(prediction.body_count):
                for direction, limit in zip(
                    free.matrix,
                    free.bounds,
                    strict=True,
                ):
                    offset, reference = prediction.body_support(
                        stage,
                        point,
                        direction,
                    )
                    matrices.append(reference)
                    bounds.append(float(limit - offset))

        terminal_set = self.mission.terminal_halfspaces
        if not terminal:
            matrix, limits = preterminal_support_constraints(
                self.mission,
                prediction,
            )
            matrices.extend(np.asarray(matrix, dtype=float).reshape(-1, 3))
            bounds.extend(np.asarray(limits, dtype=float).reshape(-1))
            self._append_successor_rows(
                prediction,
                beta,
                matrices,
                bounds,
            )

        current_negative_offset, current_negative_reference = distance_support(
            self.mission,
            prediction,
            0,
            -1.0,
        )
        lower_distance = -current_negative_offset - _box_support(
            current_negative_reference,
            self.reference_center,
            self.reference_scale,
        )
        scheduled_distance = min(
            lower_distance,
            self.coordinates.offset[0] + self.coordinates.scale[0],
        )
        for index, (facet, limit) in enumerate(
            zip(terminal_set.matrix, terminal_set.bounds, strict=True)
        ):
            offset, reference = error_support(
                self.mission,
                prediction,
                0,
                facet,
            )
            upper_error = offset + _box_support(
                reference,
                self.reference_center,
                self.reference_scale,
            )
            matrices.append(np.zeros(3))
            bounds.append(float(limit + beta[index] * scheduled_distance - upper_error))

        if terminal:
            matrix, limits = self.mission.terminal_support_constraints(prediction)
            matrices.extend(np.asarray(matrix, dtype=float).reshape(-1, 3))
            bounds.extend(np.asarray(limits, dtype=float).reshape(-1))
        else:
            offset, reference = self._progress_support(prediction)
            matrices.append(reference)
            bounds.append(float(-progress - offset))
        return np.asarray(matrices, dtype=float), np.asarray(bounds, dtype=float)

    def _progress_support(
        self,
        prediction: Prediction,
    ) -> tuple[float, np.ndarray]:
        row = self.coordinates.distance_jacobian
        stage = NEXT_UPDATE_STAGE
        initial_count = int(prediction.generator_count[0])
        count = int(prediction.generator_count[stage])
        generators = row @ prediction.state_generators[stage, :, :count]
        generators[:initial_count] -= (
            row @ prediction.state_generators[0, :, :initial_count]
        )
        offset = float(
            row @ (prediction.state_center[stage] - prediction.state_center[0])
            + np.sum(np.abs(generators))
        )
        reference = row @ (
            prediction.state_reference[stage] - prediction.state_reference[0]
        )
        return offset, reference

    def _append_successor_rows(
        self,
        prediction: Prediction,
        beta: np.ndarray,
        matrices: list[np.ndarray],
        bounds: list[float],
    ) -> None:
        domain = self.mission.approach_domain
        estimation = self.generated.bounds.state_estimation_abs
        for index in range(15):
            direction = np.zeros(15)
            direction[index] = 1.0
            offset, reference = prediction.state_support(
                NEXT_UPDATE_STAGE,
                direction,
            )
            matrices.append(reference)
            bounds.append(float(domain.upper[index] - offset - estimation[index]))
            offset, reference = prediction.state_support(
                NEXT_UPDATE_STAGE,
                -direction,
            )
            matrices.append(reference)
            bounds.append(float(-domain.lower[index] - offset - estimation[index]))

        coordinate_lower = self.coordinates.offset - self.coordinates.scale
        coordinate_upper = self.coordinates.offset + self.coordinates.scale
        distance_margin = float(np.abs(self.coordinates.distance_jacobian) @ estimation)
        offset, reference = distance_support(
            self.mission,
            prediction,
            NEXT_UPDATE_STAGE,
            1.0,
        )
        matrices.append(reference)
        bounds.append(float(coordinate_upper[0] - offset - distance_margin))
        offset, reference = distance_support(
            self.mission,
            prediction,
            NEXT_UPDATE_STAGE,
            -1.0,
        )
        matrices.append(reference)
        bounds.append(float(-coordinate_lower[0] - offset - distance_margin))

        error_dimension = self.mission.terminal_halfspaces.matrix.shape[1]
        for coordinate, error_index in enumerate(self.coordinates.error_indices):
            facet = np.zeros(error_dimension)
            facet[error_index] = 1.0
            offset, reference = error_support(
                self.mission,
                prediction,
                NEXT_UPDATE_STAGE,
                facet,
            )
            matrices.append(reference)
            error_margin = float(
                np.abs(self.coordinates.error_jacobian[error_index]) @ estimation
            )
            bounds.append(
                float(coordinate_upper[coordinate + 1] - offset - error_margin)
            )
            offset, reference = error_support(
                self.mission,
                prediction,
                NEXT_UPDATE_STAGE,
                -facet,
            )
            matrices.append(reference)
            bounds.append(
                float(-coordinate_lower[coordinate + 1] - offset - error_margin)
            )

        negative_offset, negative_reference = distance_support(
            self.mission,
            prediction,
            NEXT_UPDATE_STAGE,
            -1.0,
        )
        terminal_set = self.mission.terminal_halfspaces
        distance_max = self.coordinates.offset[0] + self.coordinates.scale[0]
        for index, (facet, limit) in enumerate(
            zip(terminal_set.matrix, terminal_set.bounds, strict=True)
        ):
            offset, reference = error_support(
                self.mission,
                prediction,
                NEXT_UPDATE_STAGE,
                facet,
            )
            error_margin = float(
                np.abs(facet @ self.coordinates.error_jacobian) @ estimation
            )
            matrices.append(reference + beta[index] * negative_reference)
            bounds.append(
                float(
                    limit
                    - offset
                    - error_margin
                    - beta[index] * (negative_offset + distance_margin)
                )
            )
            matrices.append(reference)
            bounds.append(
                float(limit + beta[index] * distance_max - offset - error_margin)
            )

    def _linear_feasibility(
        self,
        matrix: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray | None:
        tightened = bounds - POLYTOPE_TOLERANCE * np.linalg.norm(
            matrix * self.reference_scale,
            axis=1,
        )
        lower = self.reference_lower + POLYTOPE_TOLERANCE * self.reference_scale
        upper = self.reference_upper - POLYTOPE_TOLERANCE * self.reference_scale
        result = self.linear_program(
            np.zeros(3),
            A_ub=matrix,
            b_ub=tightened,
            bounds=tuple(zip(lower, upper)),
            method="highs",
        )
        if not result.success:
            return None
        reference = np.asarray(result.x, dtype=float)
        if (
            np.any(matrix @ reference > tightened)
            or np.any(reference < lower)
            or np.any(reference > upper)
        ):
            return None
        return self._repair_reference(reference, matrix, bounds)

    def _repair_reference(
        self,
        reference: np.ndarray,
        matrix: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray | None:
        for _ in range(4 * matrix.shape[0] + 16):
            violated = -1
            value = Fraction(0)
            for index, (row, limit) in enumerate(zip(matrix, bounds, strict=True)):
                value = _exact_dot(row, reference)
                if value > Fraction.from_float(float(limit)):
                    violated = index
                    break
            if violated < 0:
                return reference

            row = matrix[violated]
            limit = Fraction.from_float(float(bounds[violated]))
            moved = False
            for axis in np.argsort(-np.abs(row)):
                coefficient = Fraction.from_float(float(row[axis]))
                if coefficient == 0:
                    continue
                current = Fraction.from_float(float(reference[axis]))
                threshold = (limit - (value - coefficient * current)) / coefficient
                candidate = float(threshold)
                if coefficient > 0:
                    if Fraction.from_float(candidate) > threshold:
                        candidate = float(np.nextafter(candidate, -np.inf))
                    candidate = max(candidate, self.reference_lower[axis])
                    if candidate < reference[axis]:
                        reference[axis] = candidate
                        moved = True
                        break
                else:
                    if Fraction.from_float(candidate) < threshold:
                        candidate = float(np.nextafter(candidate, np.inf))
                    candidate = min(candidate, self.reference_upper[axis])
                    if candidate > reference[axis]:
                        reference[axis] = candidate
                        moved = True
                        break
            if not moved:
                return None
        return None

    def _reference_hull(
        self,
        matrix: np.ndarray,
        bounds: np.ndarray,
    ) -> Interval | None:
        lower = np.empty(3)
        upper = np.empty(3)
        augmented_matrix = np.vstack((matrix, np.eye(3), -np.eye(3)))
        augmented_bounds = np.concatenate(
            (bounds, self.reference_upper, -self.reference_lower)
        )
        for axis in range(3):
            direction = np.zeros(3)
            direction[axis] = 1.0
            maximum = self._dual_upper(
                direction,
                augmented_matrix,
                augmented_bounds,
            )
            negative_minimum = self._dual_upper(
                -direction,
                augmented_matrix,
                augmented_bounds,
            )
            if maximum is None or negative_minimum is None:
                return None
            lower[axis] = -negative_minimum
            upper[axis] = maximum
        return Interval(lower, upper)

    def _dual_upper(
        self,
        direction: np.ndarray,
        matrix: np.ndarray,
        bounds: np.ndarray,
    ) -> float | None:
        result = self.linear_program(
            bounds,
            A_eq=matrix.T,
            b_eq=direction,
            bounds=(0.0, None),
            method="highs",
        )
        if not result.success:
            return None
        multipliers = np.maximum(np.asarray(result.x, dtype=float), 0.0)
        exact = sum(
            (
                Fraction.from_float(float(limit))
                * Fraction.from_float(float(multiplier))
                for limit, multiplier in zip(bounds, multipliers, strict=True)
            ),
            Fraction(0),
        )
        for axis in range(3):
            residual = Fraction.from_float(float(direction[axis])) - sum(
                (
                    Fraction.from_float(float(row[axis]))
                    * Fraction.from_float(float(multiplier))
                    for row, multiplier in zip(matrix, multipliers, strict=True)
                ),
                Fraction(0),
            )
            exact += max(
                residual * Fraction.from_float(float(self.reference_lower[axis])),
                residual * Fraction.from_float(float(self.reference_upper[axis])),
            )
        upper = float(exact)
        if Fraction.from_float(upper) < exact:
            upper = float(np.nextafter(upper, np.inf))
        return upper


def compile_capture(
    generated: GeneratedAircraft,
    predictor: FastPredictor,
    mission: Mission,
) -> CaptureCertificate:
    """Compile the complete deterministic capture certificate for a mission."""

    if not isinstance(generated, GeneratedAircraft):
        raise TypeError("capture compilation requires an oracle-verified aircraft core")
    if not isinstance(predictor, FastPredictor) or predictor.generated is not generated:
        raise ValueError("predictor and capture compiler require the same aircraft")
    return _CaptureCompiler(generated, predictor, mission).compile()


def triangulate_box(
    lower: npt.ArrayLike, upper: npt.ArrayLike
) -> tuple[np.ndarray, ...]:
    """Return the fixed six-tetrahedron cover of a three-dimensional box."""

    low = np.asarray(lower, dtype=float).reshape(3)
    high = np.asarray(upper, dtype=float).reshape(3)
    if np.any(high <= low):
        raise ValueError("box upper bounds must exceed lower bounds")
    vertices = np.array(
        [
            (low[0], low[1], low[2]),
            (high[0], low[1], low[2]),
            (low[0], high[1], low[2]),
            (high[0], high[1], low[2]),
            (low[0], low[1], high[2]),
            (high[0], low[1], high[2]),
            (low[0], high[1], high[2]),
            (high[0], high[1], high[2]),
        ]
    )
    indices = (
        (0, 1, 3, 7),
        (0, 3, 2, 7),
        (0, 2, 6, 7),
        (0, 6, 4, 7),
        (0, 4, 5, 7),
        (0, 5, 1, 7),
    )
    return tuple(vertices[np.asarray(index)] for index in indices)


def subdivide_simplex(vertices: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Bisect a simplex along its longest normalized edge."""

    values = np.asarray(vertices, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] + 1:
        raise ValueError("simplex must have dimension plus one vertices")
    first = 0
    second = 1
    length = -1.0
    for row in range(values.shape[0]):
        for column in range(row + 1, values.shape[0]):
            candidate = float(np.sum((values[row] - values[column]) ** 2))
            if candidate > length:
                first, second, length = row, column, candidate
    if length <= BOUNDARY_TOLERANCE:
        raise ValueError("simplex is degenerate")
    midpoint = 0.5 * (values[first] + values[second])
    left = values.copy()
    right = values.copy()
    left[first] = midpoint
    right[second] = midpoint
    return left, right


def _coordinate_map(
    generated: GeneratedAircraft,
    mission: Mission,
) -> _CoordinateMap:
    domain = mission.approach_domain
    radius = np.asarray(domain.radius, dtype=float)
    if np.any(radius <= 0.0):
        raise ValueError("approach domain must be a full state box")

    center = np.asarray(domain.center, dtype=float)
    center_error = np.asarray(mission.error(center), dtype=float).reshape(-1)
    if center_error.size < 2:
        raise ValueError("mission error requires two capture coordinates")
    distance_jacobian = np.empty(15)
    error_jacobian = np.empty((center_error.size, 15))
    for index in range(15):
        step = np.zeros(15)
        step[index] = radius[index]
        upper_error = np.asarray(
            mission.error(center + step),
            dtype=float,
        ).reshape(center_error.shape)
        lower_error = np.asarray(
            mission.error(center - step),
            dtype=float,
        ).reshape(center_error.shape)
        distance_jacobian[index] = (
            mission.distance(center + step) - mission.distance(center - step)
        ) / (2.0 * radius[index])
        error_jacobian[:, index] = (upper_error - lower_error) / (2.0 * radius[index])

    selected = None
    for first, second in combinations(range(center_error.size), 2):
        jacobian = np.vstack(
            (
                distance_jacobian,
                error_jacobian[first],
                error_jacobian[second],
            )
        )
        physical = jacobian * radius
        scale = np.sum(np.abs(physical), axis=1)
        if np.any(scale <= BOUNDARY_TOLERANCE):
            continue
        transform = physical / scale[:, None]
        singular = np.linalg.svd(transform, compute_uv=False)
        if singular[-1] > BOUNDARY_TOLERANCE * singular[0]:
            selected = first, second
            break
    if selected is None:
        raise ValueError("capture coordinate transform is rank deficient")
    offset = np.array(
        (
            mission.distance(center),
            center_error[selected[0]],
            center_error[selected[1]],
        )
    )
    lift = transform.T @ np.linalg.inv(transform @ transform.T)
    state_lift = radius[:, None] * lift
    residual = np.eye(15) - lift @ transform
    residual = radius[:, None] * residual
    state_generators = np.diag(
        np.sum(np.abs(residual), axis=1) + generated.bounds.state_estimation_abs
    )

    queue_lower = (
        generated.aircraft.control_lower_rad + generated.bounds.command_error_abs_rad
    )
    queue_upper = (
        generated.aircraft.control_upper_rad - generated.bounds.command_error_abs_rad
    )
    queue_center = np.tile(
        0.5 * (queue_lower + queue_upper), generated.bounds.queue_length
    )
    queue_scale = np.tile(
        0.5 * (queue_upper - queue_lower), generated.bounds.queue_length
    )
    augmented_offset = np.concatenate((center, queue_center))
    augmented_scale = np.concatenate((radius, queue_scale))
    augmented_transform = np.zeros((3, augmented_offset.size))
    augmented_transform[:, :15] = transform
    return _CoordinateMap(
        offset,
        scale,
        augmented_transform,
        state_lift,
        state_generators,
        augmented_offset,
        augmented_scale,
        selected,
        distance_jacobian,
        error_jacobian,
    )


def _hard_halfspaces(
    generated: GeneratedAircraft,
) -> tuple[np.ndarray, np.ndarray]:
    bounds = generated.bounds
    aircraft = generated.aircraft
    rows: list[np.ndarray] = []
    limits: list[float] = []
    flow = bounds.flow.joint_zonotope(aircraft.strip_table.r_b_m)
    flow_center = flow.center[:3]
    flow_radius = np.sum(np.abs(flow.generators[:3]), axis=1)

    def flow_support(direction: np.ndarray) -> float:
        return float(direction @ flow_center + np.abs(direction) @ flow_radius)

    def append(index: int, coefficient: float, limit: float) -> None:
        row = np.zeros(15)
        row[index] = coefficient
        rows.append(row)
        limits.append(limit)

    append(3, 1.0, bounds.roll_abs_max_rad)
    append(3, -1.0, bounds.roll_abs_max_rad)
    append(4, 1.0, bounds.pitch_abs_max_rad)
    append(4, -1.0, bounds.pitch_abs_max_rad)
    for signs in product((-1.0, 1.0), repeat=3):
        row = np.zeros(15)
        row[6:9] = signs
        rows.append(row)
        direction = np.asarray(signs)
        limits.append(bounds.airspeed_m_s[1] - flow_support(-direction))
    forward = np.array((1.0, 0.0, 0.0))
    append(
        6,
        -1.0,
        -bounds.airspeed_m_s[0] - flow_support(forward),
    )
    alpha = tan(bounds.alpha_abs_max_rad)
    row = np.zeros(15)
    row[8] = 1.0
    row[6] = -alpha
    rows.append(row)
    limits.append(-flow_support(-row[6:9]))
    row = row.copy()
    row[8] = -1.0
    rows.append(row)
    limits.append(-flow_support(-row[6:9]))
    for index in range(9, 12):
        append(index, 1.0, bounds.body_rate_abs_max_rad_s)
        append(index, -1.0, bounds.body_rate_abs_max_rad_s)
    for index in range(3):
        append(12 + index, 1.0, float(aircraft.control_upper_rad[index]))
        append(12 + index, -1.0, float(-aircraft.control_lower_rad[index]))
    return np.asarray(rows), np.asarray(limits)


def _exact_dot(row: np.ndarray, value: np.ndarray) -> Fraction:
    return sum(
        (
            Fraction.from_float(float(coefficient))
            * Fraction.from_float(float(component))
            for coefficient, component in zip(row, value, strict=True)
        ),
        Fraction(0),
    )


def _box_support(
    row: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> float:
    return float(row @ center + np.abs(row) @ scale)


def _feasible_active_sets(
    matrix: np.ndarray,
    bounds: np.ndarray,
    reference_lower: np.ndarray,
    reference_upper: np.ndarray,
) -> ActiveSets:
    center = 0.5 * (reference_lower + reference_upper)
    scale = 0.5 * (reference_upper - reference_lower)
    row_count = matrix.shape[0] + 6
    normalized = np.zeros((row_count, 3))
    limits = np.empty(row_count)
    for axis in range(3):
        normalized[2 * axis, axis] = 1.0
        normalized[2 * axis + 1, axis] = -1.0
    limits[:6] = 1.0 - POLYTOPE_TOLERANCE
    facets: dict[bytes, tuple[int, float]] = {}
    for source in range(matrix.shape[0]):
        target = source + 6
        row = matrix[source] * scale
        limit = bounds[source] - matrix[source] @ center
        norm = float(np.linalg.norm(row))
        if norm > 0.0:
            row = row / norm
            limit = limit / norm - POLYTOPE_TOLERANCE
        normalized[target] = row
        limits[target] = limit
        if norm == 0.0:
            continue
        key = np.ascontiguousarray(row).tobytes()
        previous = facets.get(key)
        if previous is None or limit < previous[1]:
            facets[key] = target, limit

    candidates = (*range(6), *(value[0] for value in facets.values()))
    vertices: list[np.ndarray] = []
    for active in combinations(candidates, 3):
        rows = normalized[np.asarray(active)]
        if abs(float(np.linalg.det(rows))) <= BOUNDARY_TOLERANCE:
            continue
        point = np.linalg.solve(rows, limits[np.asarray(active)])
        if np.all(normalized @ point <= limits + POLYTOPE_TOLERANCE):
            if any(np.max(np.abs(point - vertex)) <= 1.0e-9 for vertex in vertices):
                continue
            vertices.append(point)
    if not vertices:
        raise ValueError("governor polytope is empty or lower-dimensional")
    points = np.stack(vertices)
    active = np.abs(points @ normalized.T - limits) <= 1.0e-8
    singles = []
    for index in candidates:
        face = points[active[:, index]]
        if (
            face.shape[0] >= 3
            and np.linalg.matrix_rank(
                face[1:] - face[0],
                tol=1.0e-9,
            )
            >= 2
        ):
            singles.append(index)
    pairs = []
    for pair in combinations(singles, 2):
        edge = points[active[:, pair[0]] & active[:, pair[1]]]
        if (
            edge.shape[0] >= 2
            and np.max(np.linalg.norm(edge - edge[0], axis=1)) > 1.0e-9
        ):
            pairs.append(pair)
    return ActiveSets(
        np.asarray(singles, dtype=int)[:, None],
        np.asarray(pairs, dtype=int),
        points,
    )


def _layered_cover(
    offset: np.ndarray,
    scale: np.ndarray,
    terminal_reach: float,
) -> tuple[np.ndarray, ...]:
    split = (terminal_reach - offset[0]) / scale[0]
    if split <= -1.0 + BOUNDARY_TOLERANCE:
        return triangulate_box(-np.ones(3), np.ones(3))
    if split >= 1.0 - BOUNDARY_TOLERANCE:
        return triangulate_box(-np.ones(3), np.ones(3))
    lower = triangulate_box(
        -np.ones(3),
        np.array((split, 1.0, 1.0)),
    )
    upper = triangulate_box(
        np.array((split, -1.0, -1.0)),
        np.ones(3),
    )
    return lower + upper


def _pad_rows(
    simplices: tuple[CaptureSimplex, ...],
) -> tuple[CaptureSimplex, ...]:
    row_count = max(simplex.constraint_matrix.shape[0] for simplex in simplices)
    padded = []
    for simplex in simplices:
        count = simplex.constraint_matrix.shape[0]
        if count == row_count:
            padded.append(simplex)
            continue
        matrix = np.zeros((row_count, 3))
        matrix[:count] = simplex.constraint_matrix
        bounds = np.ones((simplex.vertices.shape[0], row_count))
        bounds[:, :count] = simplex.constraint_bounds
        padded.append(
            CaptureSimplex(
                simplex.vertices,
                simplex.backup_references,
                matrix,
                bounds,
                simplex.gain_index,
                simplex.progress_m,
                simplex.terminal,
            )
        )
    return tuple(padded)
