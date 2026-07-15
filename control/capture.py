"""Automatic compilation of contracting terminal-capture certificates."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations, product
from math import tan

import numpy as np
import numpy.typing as npt

from control.interval import Interval, Zonotope
from control.missions import (
    CompilationDomain,
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
    """Certified state simplex with one complete-cell backup reference."""

    vertices: npt.ArrayLike
    backup_reference: npt.ArrayLike
    constraint_count: int
    gain_index: int
    progress_m: float
    terminal: bool
    reference_lower: npt.ArrayLike
    reference_upper: npt.ArrayLike
    _barycentric_matrix: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=float)
        if vertices.ndim != 2 or vertices.shape[0] != vertices.shape[1] + 1:
            raise ValueError("simplex must have dimension plus one vertices")
        dimension = vertices.shape[1]
        vertices = vertices.copy()
        backup = np.asarray(self.backup_reference, dtype=float).reshape(3).copy()
        reference_lower = np.asarray(self.reference_lower, dtype=float).reshape(3).copy()
        reference_upper = np.asarray(self.reference_upper, dtype=float).reshape(3).copy()
        if self.constraint_count <= 0:
            raise ValueError("constraint count must be positive")
        if np.any(reference_lower > reference_upper):
            raise ValueError("reference lower bound exceeds upper bound")
        if np.any(backup < reference_lower) or np.any(backup > reference_upper):
            raise ValueError("backup lies outside its certified reference bounds")
        augmented = np.column_stack((vertices, np.ones(dimension + 1)))
        singular = np.linalg.svd(augmented, compute_uv=False)
        if singular[-1] <= BOUNDARY_TOLERANCE * singular[0]:
            raise ValueError("simplex vertices are affinely dependent")
        barycentric = np.linalg.inv(augmented).T
        for value in (
            vertices,
            backup,
            reference_lower,
            reference_upper,
            barycentric,
        ):
            value.flags.writeable = False
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "backup_reference", backup)
        object.__setattr__(self, "reference_lower", reference_lower)
        object.__setattr__(self, "reference_upper", reference_upper)
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

        return self.backup_reference.copy()

    def backup_into(self, result: np.ndarray) -> None:
        """Write the complete-cell backup into a runtime buffer."""

        result[:] = self.backup_reference


@dataclass(frozen=True)
class _LookupNode:
    vertices: npt.ArrayLike
    simplex_index: int
    split_edge: tuple[int, int] = (-1, -1)
    left: _LookupNode | None = None
    right: _LookupNode | None = None
    _barycentric_matrix: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=float)
        dimension = vertices.shape[1]
        augmented = np.column_stack((vertices, np.ones(dimension + 1)))
        barycentric = np.linalg.inv(augmented).T
        vertices = vertices.copy()
        vertices.flags.writeable = False
        barycentric.flags.writeable = False
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "_barycentric_matrix", barycentric)

    def locate_into(self, normalized: np.ndarray, weights: np.ndarray) -> int:
        np.matmul(
            self._barycentric_matrix[:, :-1],
            normalized,
            out=weights,
        )
        weights += self._barycentric_matrix[:, -1]
        if np.min(weights) < -BOUNDARY_TOLERANCE:
            return -1
        if self.simplex_index >= 0:
            return self.simplex_index
        first, second = self.split_edge
        child = self.left if weights[first] <= weights[second] else self.right
        if child is None:
            return -1
        return child.locate_into(normalized, weights)


@dataclass(frozen=True)
class CaptureCertificate:
    """Certified capture cells for direct online admissible-set construction."""

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
    domain: CompilationDomain
    mission: Mission
    generated: GeneratedAircraft
    lookup_roots: tuple[_LookupNode, ...]

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
        if any(
            simplex.vertices.shape[1] != dimension or simplex.constraint_count <= 0
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
        if not self.lookup_roots:
            raise ValueError("capture certificate requires a lookup root")

    @property
    def max_constraints(self) -> int:
        """Return the largest online inequality count in the certificate."""

        return max(simplex.constraint_count for simplex in self.simplices)

    def capture_bounds(self, distance: float) -> np.ndarray:
        """Return the contracting terminal-facet bounds at one distance."""

        return self.terminal_bounds + self.beta * min(distance, self.distance_max)

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
        weights = np.empty(normalized.size + 1)
        for root in self.lookup_roots:
            index = root.locate_into(normalized, weights)
            if index >= 0 and self.simplices[index].contains(normalized, generators):
                return index, self.simplices[index].barycentric(normalized)
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
        weights = np.empty(normalized.size + 1)
        for root in self.lookup_roots:
            index = root.locate_into(normalized, weights)
            if index >= 0 and self.simplices[index].contains(normalized, generators):
                return index, self.simplices[index].barycentric(normalized)
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
        for root in self.lookup_roots:
            index = root.locate_into(coordinate_normalized, weights)
            if index >= 0:
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


class CaptureConstraintBuilder:
    """Build the current admissible reference set from one short prediction."""

    def __init__(
        self,
        generated: GeneratedAircraft,
        mission: Mission,
        domain: CompilationDomain,
    ) -> None:
        self.generated = generated
        self.mission = mission
        self.domain = domain
        self.coordinates = _coordinate_map(generated, mission, domain)
        self.terminal_set = mission.terminal_halfspaces(generated)
        lower = np.asarray(generated.aircraft.control_lower_rad, dtype=float)
        upper = np.asarray(generated.aircraft.control_upper_rad, dtype=float)
        self.reference_center = 0.5 * (lower + upper)
        self.reference_scale = 0.5 * (upper - lower)
        self.hard_matrix, self.hard_bounds = _hard_halfspaces(generated)
        alpha = tan(generated.bounds.alpha_abs_max_rad)
        self.air_velocity_directions = np.vstack(
            (
                tuple(product((-1.0, 1.0), repeat=3)),
                (-1.0, 0.0, 0.0),
                (-alpha, 0.0, 1.0),
                (-alpha, 0.0, -1.0),
            )
        )
        self.air_velocity_bounds = np.concatenate(
            (
                np.full(8, generated.bounds.airspeed_m_s[1]),
                (-generated.bounds.airspeed_m_s[0], 0.0, 0.0),
            )
        )

    def rows(
        self,
        prediction: Prediction,
        beta: np.ndarray,
        progress: float,
        terminal: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return current affine inequalities in the governor reference."""

        matrices: list[np.ndarray] = []
        bounds: list[float] = []
        aircraft = self.generated.aircraft
        reference_lower = np.asarray(aircraft.control_lower_rad, dtype=float)
        reference_upper = np.asarray(aircraft.control_upper_rad, dtype=float)

        command_axes = np.vstack((np.eye(3), -np.eye(3)))
        command_limits = np.concatenate((reference_upper, -reference_lower))
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
            for direction, limit in zip(
                self.air_velocity_directions,
                self.air_velocity_bounds,
                strict=True,
            ):
                offset, reference = prediction.air_velocity_support(
                    stage,
                    direction,
                )
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

        if not terminal:
            matrix, limits = preterminal_support_constraints(
                self.mission,
                prediction,
                self.generated,
            )
            matrices.extend(np.asarray(matrix, dtype=float).reshape(-1, 3))
            bounds.extend(np.asarray(limits, dtype=float).reshape(-1))
            self._append_successor_rows(prediction, beta, matrices, bounds)

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
            zip(
                self.terminal_set.matrix,
                self.terminal_set.bounds,
                strict=True,
            )
        ):
            offset, reference = error_support(
                self.mission,
                prediction,
                0,
                facet,
                self.generated,
                self.domain,
            )
            upper_error = offset + _box_support(
                reference,
                self.reference_center,
                self.reference_scale,
            )
            matrices.append(np.zeros(3))
            bounds.append(float(limit + beta[index] * scheduled_distance - upper_error))

        if terminal:
            matrix, limits = self.mission.terminal_support_constraints(
                prediction,
                self.generated,
                self.domain,
            )
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
        estimation = self.generated.bounds.state_estimation_abs
        for index in range(15):
            direction = np.zeros(15)
            direction[index] = 1.0
            offset, reference = prediction.state_support(
                NEXT_UPDATE_STAGE,
                direction,
            )
            matrices.append(reference)
            bounds.append(float(self.domain.upper[index] - offset - estimation[index]))
            offset, reference = prediction.state_support(
                NEXT_UPDATE_STAGE,
                -direction,
            )
            matrices.append(reference)
            bounds.append(float(-self.domain.lower[index] - offset - estimation[index]))

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

        error_dimension = self.terminal_set.matrix.shape[1]
        for coordinate, error_index in enumerate(self.coordinates.error_indices):
            facet = np.zeros(error_dimension)
            facet[error_index] = 1.0
            offset, reference = error_support(
                self.mission,
                prediction,
                NEXT_UPDATE_STAGE,
                facet,
                self.generated,
                self.domain,
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
                self.generated,
                self.domain,
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
        distance_max = self.coordinates.offset[0] + self.coordinates.scale[0]
        for index, (facet, limit) in enumerate(
            zip(
                self.terminal_set.matrix,
                self.terminal_set.bounds,
                strict=True,
            )
        ):
            offset, reference = error_support(
                self.mission,
                prediction,
                NEXT_UPDATE_STAGE,
                facet,
                self.generated,
                self.domain,
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


@dataclass(frozen=True)
class _CompiledCover:
    simplices: tuple[CaptureSimplex, ...]
    roots: tuple[_LookupNode, ...]


class _CaptureCompiler:
    def __init__(
        self,
        generated: GeneratedAircraft,
        predictor: FastPredictor,
        mission: Mission,
        domain: CompilationDomain | None = None,
    ) -> None:
        from control.oracle import NonlinearOracle

        self.generated = generated
        self.predictor = predictor
        self.mission = mission
        self.domain = domain or _automatic_compilation_domain(generated)
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
        self.constraint_builder = CaptureConstraintBuilder(
            generated,
            mission,
            self.domain,
        )
        self.coordinates = self.constraint_builder.coordinates
        self.terminal_set = self.constraint_builder.terminal_set
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
        if verified is not None:
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
                if compiled is not None:
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
                if compiled is not None:
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
        if verified is None:
            raise ValueError("verified capture cover could not be reproduced")
        return CaptureCertificate(
            beta,
            self.coordinates.offset,
            self.coordinates.scale,
            self.reference_lower,
            self.reference_upper,
            verified.simplices,
            self.terminal_set.bounds,
            float(upper[0]),
            self.coordinates.augmented_offset,
            self.coordinates.augmented_scale,
            self.coordinates.transform,
            self.domain,
            self.mission,
            self.generated,
            verified.roots,
        )

    def _authority(
        self,
        cover: tuple[np.ndarray, ...],
    ) -> tuple[np.ndarray, float, float]:
        vertices = sorted({tuple(vertex) for simplex in cover for vertex in simplex})
        facets = self.terminal_set.matrix
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
                    self.generated,
                    self.domain,
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
    ) -> _CompiledCover | None:
        accepted: list[CaptureSimplex] = []
        roots = []
        for vertices in cover:
            root = self._compile_node(
                vertices,
                0,
                beta,
                terminal_reach,
                worst_approach,
                accepted,
            )
            if root is None:
                return None
            roots.append(root)
        return _CompiledCover(tuple(accepted), tuple(roots))

    def _compile_node(
        self,
        vertices: np.ndarray,
        depth: int,
        beta: np.ndarray,
        terminal_reach: float,
        worst_approach: float,
        accepted: list[CaptureSimplex],
    ) -> _LookupNode | None:
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
            index = len(accepted)
            accepted.append(certified)
            return _LookupNode(vertices, index)
        if depth == MAX_SUBDIVISION_DEPTH:
            return None
        first, second = _longest_edge(vertices)
        left_vertices, right_vertices = subdivide_simplex(vertices)
        left = self._compile_node(
            left_vertices,
            depth + 1,
            beta,
            terminal_reach,
            worst_approach,
            accepted,
        )
        if left is None:
            return None
        right = self._compile_node(
            right_vertices,
            depth + 1,
            beta,
            terminal_reach,
            worst_approach,
            accepted,
        )
        if right is None:
            return None
        return _LookupNode(vertices, -1, (first, second), left, right)

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
        matrix, limits = self.constraint_builder.rows(
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
        reference_lower = self.reference_lower
        reference_upper = self.reference_upper
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
            reference_lower = reference.lower
            reference_upper = reference.upper
        return CaptureSimplex(
            vertices,
            backup,
            matrix.shape[0] + (6 if terminal else 0),
            gain_index,
            progress,
            terminal,
            reference_lower,
            reference_upper,
        )

    def _belief(
        self,
        normalized: np.ndarray,
        required_gain: int | None = None,
    ) -> tuple[Zonotope, int]:
        center = self.domain.center + self.coordinates.state_lift @ normalized
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
        center = self.domain.center + self.coordinates.state_lift @ normalized
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
    domain: CompilationDomain | None = None,
) -> CaptureCertificate:
    """Compile the complete deterministic capture certificate for a mission."""

    if not isinstance(generated, GeneratedAircraft):
        raise TypeError("capture compilation requires an oracle-verified aircraft core")
    if not isinstance(predictor, FastPredictor) or predictor.generated is not generated:
        raise ValueError("predictor and capture compiler require the same aircraft")
    return _CaptureCompiler(generated, predictor, mission, domain).compile()


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
    first, second = _longest_edge(values)
    midpoint = 0.5 * (values[first] + values[second])
    left = values.copy()
    right = values.copy()
    left[first] = midpoint
    right[second] = midpoint
    return left, right


def _longest_edge(vertices: np.ndarray) -> tuple[int, int]:
    first = 0
    second = 1
    length = -1.0
    for row in range(vertices.shape[0]):
        for column in range(row + 1, vertices.shape[0]):
            candidate = float(np.sum((vertices[row] - vertices[column]) ** 2))
            if candidate > length:
                first, second, length = row, column, candidate
    if length <= BOUNDARY_TOLERANCE:
        raise ValueError("simplex is degenerate")
    return first, second


def _automatic_compilation_domain(
    generated: GeneratedAircraft,
) -> CompilationDomain:
    center = np.asarray(generated.domain_anchor, dtype=float)
    radius = np.asarray(generated.state_scale, dtype=float)
    return CompilationDomain(center - radius, center + radius)


def _coordinate_map(
    generated: GeneratedAircraft,
    mission: Mission,
    domain: CompilationDomain,
) -> _CoordinateMap:
    radius = np.asarray(domain.radius, dtype=float)
    if np.any(radius <= 0.0):
        raise ValueError("approach domain must be a full state box")

    center = np.asarray(domain.center, dtype=float)
    center_error = np.asarray(mission.error(center, generated), dtype=float).reshape(-1)
    if center_error.size < 2:
        raise ValueError("mission error requires two capture coordinates")
    distance_jacobian = np.empty(15)
    error_jacobian = np.empty((center_error.size, 15))
    for index in range(15):
        step = np.zeros(15)
        step[index] = radius[index]
        upper_error = np.asarray(
            mission.error(center + step, generated),
            dtype=float,
        ).reshape(center_error.shape)
        lower_error = np.asarray(
            mission.error(center - step, generated),
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

    def append(index: int, coefficient: float, limit: float) -> None:
        row = np.zeros(15)
        row[index] = coefficient
        rows.append(row)
        limits.append(limit)

    append(3, 1.0, bounds.roll_abs_max_rad)
    append(3, -1.0, bounds.roll_abs_max_rad)
    append(4, 1.0, bounds.pitch_abs_max_rad)
    append(4, -1.0, bounds.pitch_abs_max_rad)
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
