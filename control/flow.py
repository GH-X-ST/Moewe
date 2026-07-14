"""Joint body-relative local-flow representation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from control.interval import AffineForm, Interval, Zonotope

FLOW_COMPONENT_ORDER = ("x", "y", "z")
GRADIENT_COMPONENT_ORDER = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
)
CENTER_FLOW_GENERATOR_COUNT = 3
GRADIENT_FLOW_GENERATOR_COUNT = 9
SHARED_FLOW_GENERATOR_COUNT = 12


@dataclass(frozen=True)
class AffineFlow:
    """Local flow value, gradient, and non-affine strip remainder."""

    center_b_m_s: npt.ArrayLike
    gradient_b_s: npt.ArrayLike
    remainder_b_m_s: npt.ArrayLike

    def __post_init__(self) -> None:
        _set_array(self, "center_b_m_s", self.center_b_m_s, (3,))
        _set_array(self, "gradient_b_s", self.gradient_b_s, (3, 3))
        _set_array(self, "remainder_b_m_s", self.remainder_b_m_s, (-1, 3))

    def strip_flow(self, strip_b_m: npt.ArrayLike) -> np.ndarray:
        """Return body-frame flow at each aerodynamic strip."""

        locations = np.asarray(strip_b_m, dtype=float).reshape(-1, 3)
        return (
            self.center_b_m_s + locations @ self.gradient_b_s.T + self.remainder_b_m_s
        )


@dataclass(frozen=True)
class JointFlow:
    """Immutable joint-flow factors at the centre and aerodynamic strips.

    Gradient rows use the order ``G00, G01, G02, G10, ..., G22``. Generator
    groups are centre, gradient, then strip-major ``x, y, z`` remainder.
    """

    strip_b_m: npt.ArrayLike
    center_midpoint_b_m_s: npt.ArrayLike
    center_generators_b_m_s: npt.ArrayLike
    gradient_midpoint_b_s: npt.ArrayLike
    gradient_generators_b_s: npt.ArrayLike
    remainder_midpoint_b_m_s: npt.ArrayLike
    remainder_generators_b_m_s: npt.ArrayLike
    center: np.ndarray = field(init=False, repr=False)
    generators: np.ndarray = field(init=False, repr=False)
    center_factor_slice: slice = field(init=False)
    gradient_factor_slice: slice = field(init=False)
    remainder_factor_slice: slice = field(init=False)

    def __post_init__(self) -> None:
        _set_array(self, "strip_b_m", self.strip_b_m, (-1, 3))
        strip_count = self.strip_b_m.shape[0]
        _set_array(
            self,
            "center_midpoint_b_m_s",
            self.center_midpoint_b_m_s,
            (3,),
        )
        _set_matrix(
            self,
            "center_generators_b_m_s",
            self.center_generators_b_m_s,
            3,
        )
        _set_array(
            self,
            "gradient_midpoint_b_s",
            self.gradient_midpoint_b_s,
            (3, 3),
        )
        _set_matrix(
            self,
            "gradient_generators_b_s",
            self.gradient_generators_b_s,
            9,
        )
        _set_array(
            self,
            "remainder_midpoint_b_m_s",
            self.remainder_midpoint_b_m_s,
            (strip_count, 3),
        )
        _set_matrix(
            self,
            "remainder_generators_b_m_s",
            self.remainder_generators_b_m_s,
            3 * strip_count,
        )

        sample_locations = np.vstack((np.zeros(3), self.strip_b_m))
        center_lift = np.tile(np.eye(3), (strip_count + 1, 1))
        # Row-major vec(G) makes each block satisfy Q_i vec(G) = G q_i.
        gradient_lift = np.vstack(
            [np.kron(np.eye(3), location) for location in sample_locations]
        )
        remainder_midpoint = np.concatenate(
            (np.zeros(3), self.remainder_midpoint_b_m_s.reshape(-1))
        )
        remainder_lift = np.vstack(
            (
                np.zeros((3, self.remainder_generators_b_m_s.shape[1])),
                self.remainder_generators_b_m_s,
            )
        )
        center = (
            center_lift @ self.center_midpoint_b_m_s
            + gradient_lift @ self.gradient_midpoint_b_s.reshape(-1)
            + remainder_midpoint
        )
        generators = np.column_stack(
            (
                center_lift @ self.center_generators_b_m_s,
                gradient_lift @ self.gradient_generators_b_s,
                remainder_lift,
            )
        )
        _set_array(self, "center", center, (3 * (strip_count + 1),))
        _set_matrix(self, "generators", generators, self.center.size)

        center_stop = self.center_generators_b_m_s.shape[1]
        gradient_stop = center_stop + self.gradient_generators_b_s.shape[1]
        object.__setattr__(self, "center_factor_slice", slice(0, center_stop))
        object.__setattr__(
            self,
            "gradient_factor_slice",
            slice(center_stop, gradient_stop),
        )
        object.__setattr__(
            self,
            "remainder_factor_slice",
            slice(gradient_stop, self.generators.shape[1]),
        )

    @property
    def strip_count(self) -> int:
        """Return the number of aerodynamic strips."""

        return int(self.strip_b_m.shape[0])

    def evaluate(self, factors: npt.ArrayLike) -> np.ndarray:
        """Evaluate the stacked centre and strip flow for one factor vector."""

        coefficients = np.asarray(factors, dtype=float).reshape(
            self.generators.shape[1]
        )
        return (self.center + self.generators @ coefficients).reshape(-1, 3)

    def realization(self, factors: npt.ArrayLike) -> AffineFlow:
        """Return the affine-flow fields represented by one factor vector."""

        coefficients = np.asarray(factors, dtype=float).reshape(
            self.generators.shape[1]
        )
        center = (
            self.center_midpoint_b_m_s
            + self.center_generators_b_m_s @ coefficients[self.center_factor_slice]
        )
        gradient = (
            self.gradient_midpoint_b_s.reshape(-1)
            + self.gradient_generators_b_s @ coefficients[self.gradient_factor_slice]
        ).reshape(3, 3)
        remainder = (
            self.remainder_midpoint_b_m_s.reshape(-1)
            + self.remainder_generators_b_m_s
            @ coefficients[self.remainder_factor_slice]
        ).reshape(self.strip_count, 3)
        return AffineFlow(center, gradient, remainder)

    def strip_flow(self, factors: npt.ArrayLike) -> np.ndarray:
        """Evaluate body-frame flow at the aerodynamic strips."""

        return self.evaluate(factors)[1:]

    def zonotope(self) -> Zonotope:
        """Return the exact stacked joint-flow zonotope."""

        return Zonotope(self.center, self.generators)

    def affine_form(self) -> AffineForm:
        """Return the stacked joint flow as a shaped affine form."""

        return AffineForm.from_zonotope(
            self.zonotope(),
            (self.strip_count + 1, 3),
        )

    def support(self, direction: npt.ArrayLike) -> float:
        """Return joint-flow support in a stacked direction."""

        return self.zonotope().support(direction)

    def independent_hull(self) -> Interval:
        """Return the box obtained by making every flow sample independent."""

        return self.zonotope().interval_hull()

    def independent_support(self, direction: npt.ArrayLike) -> float:
        """Return support after replacing the joint flow by independent boxes."""

        vector = np.asarray(direction, dtype=float).reshape(self.center.shape)
        return self.independent_hull().support(vector)


@dataclass(frozen=True)
class FlowBounds:
    """Box bounds compiled into the joint-flow factorization."""

    center_lower_m_s: npt.ArrayLike
    center_upper_m_s: npt.ArrayLike
    gradient_lower_s: npt.ArrayLike
    gradient_upper_s: npt.ArrayLike
    remainder_abs_m_s: npt.ArrayLike

    def __post_init__(self) -> None:
        _set_array(self, "center_lower_m_s", self.center_lower_m_s, (3,))
        _set_array(self, "center_upper_m_s", self.center_upper_m_s, (3,))
        _set_array(self, "gradient_lower_s", self.gradient_lower_s, (3, 3))
        _set_array(self, "gradient_upper_s", self.gradient_upper_s, (3, 3))
        _set_array(self, "remainder_abs_m_s", self.remainder_abs_m_s, (3,))

    def joint_flow(self, strip_b_m: npt.ArrayLike) -> JointFlow:
        """Compile box bounds into the exact joint-flow factorization."""

        locations = np.asarray(strip_b_m, dtype=float).reshape(-1, 3)
        center = Interval(self.center_lower_m_s, self.center_upper_m_s)
        gradient = Interval(self.gradient_lower_s, self.gradient_upper_s)
        remainder_radius = np.broadcast_to(
            self.remainder_abs_m_s,
            locations.shape,
        )
        return JointFlow(
            locations,
            center.center,
            np.diag(center.radius),
            gradient.center,
            np.diag(gradient.radius.reshape(-1)),
            np.zeros_like(locations),
            np.diag(remainder_radius.reshape(-1)),
        )

    def strip_bounds(
        self,
        strip_b_m: npt.ArrayLike,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return independent interval bounds for all strip flows."""

        joint = self.joint_flow(strip_b_m)
        hull = joint.independent_hull()
        lower = hull.lower.reshape(joint.strip_count + 1, 3)
        upper = hull.upper.reshape(joint.strip_count + 1, 3)
        return lower[1:], upper[1:]

    def joint_zonotope(self, strip_b_m: npt.ArrayLike) -> Zonotope:
        """Return the stacked joint-flow zonotope."""

        return self.joint_flow(strip_b_m).zonotope()


def _set_array(
    instance: object,
    name: str,
    value: npt.ArrayLike,
    shape: tuple[int, ...],
) -> None:
    array = np.asarray(value, dtype=float).reshape(shape).copy()
    array.flags.writeable = False
    object.__setattr__(instance, name, array)


def _set_matrix(
    instance: object,
    name: str,
    value: npt.ArrayLike,
    row_count: int,
) -> None:
    array = np.asarray(value, dtype=float)
    column_count = array.shape[1] if array.ndim == 2 else array.size // row_count
    _set_array(instance, name, array, (row_count, column_count))
