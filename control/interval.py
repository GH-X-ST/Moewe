"""Outward-rounded interval and affine-generator set arithmetic."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from flint import arb, ctx


def _down(value: npt.ArrayLike) -> np.ndarray:
    return np.nextafter(np.asarray(value, dtype=float), -np.inf)


def _up(value: npt.ArrayLike) -> np.ndarray:
    return np.nextafter(np.asarray(value, dtype=float), np.inf)


def _arb_unary(
    lower: np.ndarray,
    upper: np.ndarray,
    operation: str,
) -> tuple[np.ndarray, np.ndarray]:
    lower_result = np.empty(lower.shape)
    upper_result = np.empty(upper.shape)
    with ctx.workprec(128):
        for index in np.ndindex(lower.shape):
            value = arb(float(lower[index])).union(arb(float(upper[index])))
            result = getattr(value, operation)()
            if not result.is_finite():
                raise ValueError(f"{operation} is unbounded on the interval")
            lower_result[index] = np.nextafter(
                float(result.lower()),
                -np.inf,
            )
            upper_result[index] = np.nextafter(
                float(result.upper()),
                np.inf,
            )
    return lower_result, upper_result


def _arb_atan2(
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    x_lower: np.ndarray,
    x_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower_result = np.empty(y_lower.shape)
    upper_result = np.empty(y_upper.shape)
    with ctx.workprec(128):
        for index in np.ndindex(y_lower.shape):
            y_value = arb(float(y_lower[index])).union(arb(float(y_upper[index])))
            x_value = arb(float(x_lower[index])).union(arb(float(x_upper[index])))
            result = arb.atan2(y_value, x_value)
            lower_result[index] = np.nextafter(
                float(result.lower()),
                -np.inf,
            )
            upper_result[index] = np.nextafter(
                float(result.upper()),
                np.inf,
            )
    return lower_result, upper_result


@dataclass(frozen=True)
class Interval:
    """Immutable axis-aligned interval with finite binary64 endpoints."""

    lower: npt.ArrayLike
    upper: npt.ArrayLike

    __array_priority__ = 1000

    def __post_init__(self) -> None:
        lower, upper = np.broadcast_arrays(
            np.asarray(self.lower, dtype=float),
            np.asarray(self.upper, dtype=float),
        )
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("interval endpoints must be finite")
        if np.any(lower > upper):
            raise ValueError("interval lower endpoint exceeds upper endpoint")
        lower = np.array(lower, copy=True)
        upper = np.array(upper, copy=True)
        lower.flags.writeable = False
        upper.flags.writeable = False
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @classmethod
    def point(cls, value: npt.ArrayLike) -> Interval:
        """Return a degenerate interval at a finite value."""

        array = np.asarray(value, dtype=float)
        return cls(array, array)

    @classmethod
    def from_midpoint(
        cls,
        midpoint: npt.ArrayLike,
        radius: npt.ArrayLike,
    ) -> Interval:
        """Return an interval from its midpoint and nonnegative radius."""

        midpoint_array, radius_array = np.broadcast_arrays(
            np.asarray(midpoint, dtype=float),
            np.asarray(radius, dtype=float),
        )
        if np.any(radius_array < 0.0):
            raise ValueError("interval radius must be nonnegative")
        lower = np.where(
            radius_array == 0.0,
            midpoint_array,
            _down(midpoint_array - radius_array),
        )
        upper = np.where(
            radius_array == 0.0,
            midpoint_array,
            _up(midpoint_array + radius_array),
        )
        return cls(lower, upper)

    @property
    def center(self) -> np.ndarray:
        """Return the midpoint of each component."""

        return self.lower + 0.5 * (self.upper - self.lower)

    @property
    def radius(self) -> np.ndarray:
        """Return an outward-rounded component radius."""

        center = self.center
        radius = _up(np.maximum(center - self.lower, self.upper - center))
        return np.where(self.lower == self.upper, 0.0, radius)

    def contains(self, value: npt.ArrayLike) -> bool:
        """Return whether every value component lies in the interval."""

        array = np.asarray(value, dtype=float)
        return bool(np.all(array >= self.lower) and np.all(array <= self.upper))

    def subset(self, other: Interval) -> bool:
        """Return whether this interval is a subset of another interval."""

        return bool(
            np.all(self.lower >= other.lower) and np.all(self.upper <= other.upper)
        )

    def hull(self, other: Interval) -> Interval:
        """Return the smallest box containing both intervals."""

        return _new_interval(
            np.minimum(self.lower, other.lower),
            np.maximum(self.upper, other.upper),
        )

    def intersection(self, other: Interval) -> Interval:
        """Return the interval intersection, rejecting an empty result."""

        return Interval(
            np.maximum(self.lower, other.lower),
            np.minimum(self.upper, other.upper),
        )

    def support(self, direction: npt.ArrayLike) -> float:
        """Return the support value in a direction."""

        values = np.asarray(direction, dtype=float)
        if values.shape != self.lower.shape:
            raise ValueError("support direction must match interval shape")
        result = (self * values).sum()
        return float(result.upper)

    def affine_map(
        self,
        matrix: npt.ArrayLike,
        offset: npt.ArrayLike = 0.0,
    ) -> Interval:
        """Return the interval image under a fixed scalar or matrix map."""

        values = np.asarray(matrix, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("affine map must be finite")
        if values.ndim == 0:
            return self * values + offset
        if self.lower.ndim != 1:
            raise ValueError("matrix affine maps require a vector interval")
        if values.ndim == 1:
            if values.shape != self.lower.shape:
                raise ValueError("affine map dimension mismatch")
            return self.dot(values) + offset
        if values.ndim != 2 or values.shape[1] != self.lower.size:
            raise ValueError("affine map dimension mismatch")
        rows = [self.dot(row) for row in values]
        result = _new_interval(
            np.array([float(row.lower) for row in rows]),
            np.array([float(row.upper) for row in rows]),
        )
        return result + offset

    def sum(self, axis: int | None = None) -> Interval:
        """Return an outward-rounded component sum."""

        lower = self.lower
        upper = self.upper
        if axis is None:
            lower = lower.reshape(-1)
            upper = upper.reshape(-1)
            axis = 0
        lower = np.moveaxis(lower, axis, 0)
        upper = np.moveaxis(upper, axis, 0)
        if lower.shape[0] == 0:
            return Interval.point(np.zeros(lower.shape[1:]))
        lower_sum = lower[0]
        upper_sum = upper[0]
        for index in range(1, lower.shape[0]):
            lower_sum = _down(lower_sum + lower[index])
            upper_sum = _up(upper_sum + upper[index])
        return _new_interval(lower_sum, upper_sum)

    def dot(self, other: Interval | npt.ArrayLike) -> Interval:
        """Return the outward-rounded sum of component products."""

        operand = _as_interval(other)
        if self.lower.shape != operand.lower.shape:
            raise ValueError("dot operands must have the same shape")
        return (self * operand).sum()

    def cross(self, other: Interval | npt.ArrayLike) -> Interval:
        """Return the interval cross product along the final axis."""

        operand = _as_interval(other)
        if self.lower.shape[-1:] != (3,) or operand.lower.shape[-1:] != (3,):
            raise ValueError("cross operands must end in three components")
        first, second = np.broadcast_arrays(self.lower, operand.lower)
        del first, second
        components = (
            self[..., 1] * operand[..., 2] - self[..., 2] * operand[..., 1],
            self[..., 2] * operand[..., 0] - self[..., 0] * operand[..., 2],
            self[..., 0] * operand[..., 1] - self[..., 1] * operand[..., 0],
        )
        return _new_interval(
            np.stack([component.lower for component in components], axis=-1),
            np.stack([component.upper for component in components], axis=-1),
        )

    def abs(self) -> Interval:
        """Return the componentwise absolute-value interval."""

        lower = np.where(
            self.lower >= 0.0,
            self.lower,
            np.where(self.upper <= 0.0, -self.upper, 0.0),
        )
        upper = np.where(
            self.lower >= 0.0,
            self.upper,
            np.where(
                self.upper <= 0.0,
                -self.lower,
                np.maximum(-self.lower, self.upper),
            ),
        )
        return _new_interval(lower, upper)

    def square(self) -> Interval:
        """Return the componentwise square interval."""

        lower_square = self.lower * self.lower
        upper_square = self.upper * self.upper
        crosses_zero = (self.lower <= 0.0) & (self.upper >= 0.0)
        lower = np.where(
            crosses_zero,
            0.0,
            _down(np.minimum(lower_square, upper_square)),
        )
        upper = _up(np.maximum(lower_square, upper_square))
        return _new_interval(lower, upper)

    def sqrt(self) -> Interval:
        """Return the componentwise square-root interval."""

        if np.any(self.lower < 0.0):
            raise ValueError("square root requires a nonnegative interval")
        lower = np.where(self.lower == 0.0, 0.0, _down(np.sqrt(self.lower)))
        return _new_interval(lower, _up(np.sqrt(self.upper)))

    def sin(self) -> Interval:
        """Return rigorous componentwise sine bounds."""

        return _new_interval(*_arb_unary(self.lower, self.upper, "sin"))

    def cos(self) -> Interval:
        """Return rigorous componentwise cosine bounds."""

        return _new_interval(*_arb_unary(self.lower, self.upper, "cos"))

    def tan(self) -> Interval:
        """Return tangent bounds, rejecting intervals crossing a pole."""

        try:
            bounds = _arb_unary(self.lower, self.upper, "tan")
        except ValueError as error:
            raise ValueError("tangent interval crosses a pole") from error
        return _new_interval(*bounds)

    def tanh(self) -> Interval:
        """Return componentwise hyperbolic-tangent bounds."""

        return _new_interval(*_arb_unary(self.lower, self.upper, "tanh"))

    def asin(self) -> Interval:
        """Return componentwise inverse-sine bounds."""

        if np.any(self.lower < -1.0) or np.any(self.upper > 1.0):
            raise ValueError("inverse sine requires an interval inside [-1, 1]")
        return _new_interval(*_arb_unary(self.lower, self.upper, "asin"))

    def atan2(self, positive_x: Interval | npt.ArrayLike) -> Interval:
        """Return atan2 bounds when every second-argument value is positive."""

        x_interval = _as_interval(positive_x)
        y_lower, x_lower = np.broadcast_arrays(self.lower, x_interval.lower)
        y_upper, x_upper = np.broadcast_arrays(self.upper, x_interval.upper)
        if np.any(x_lower <= 0.0):
            raise ValueError("atan2 requires a strictly positive x interval")
        return _new_interval(*_arb_atan2(y_lower, y_upper, x_lower, x_upper))

    def norm(self, axis: int | None = None) -> Interval:
        """Return the Euclidean norm interval over selected components."""

        squared = self.square().sum(axis=axis)
        squared = _new_interval(np.maximum(squared.lower, 0.0), squared.upper)
        return squared.sqrt()

    def clip(
        self,
        lower: npt.ArrayLike,
        upper: npt.ArrayLike,
    ) -> Interval:
        """Return the componentwise clipped interval image."""

        lower_bound, upper_bound = np.broadcast_arrays(
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
        )
        if np.any(lower_bound > upper_bound):
            raise ValueError("clip lower bound exceeds upper bound")
        if not np.all(np.isfinite(lower_bound)) or not np.all(np.isfinite(upper_bound)):
            raise ValueError("clip bounds must be finite")
        return _new_interval(
            np.clip(self.lower, lower_bound, upper_bound),
            np.clip(self.upper, lower_bound, upper_bound),
        )

    def __getitem__(self, key: object) -> Interval:
        return _new_interval(self.lower[key], self.upper[key])

    def __neg__(self) -> Interval:
        return _new_interval(-self.upper, -self.lower)

    def __abs__(self) -> Interval:
        return self.abs()

    def __add__(self, other: Interval | npt.ArrayLike) -> Interval:
        operand = _as_interval(other)
        self_zero = (self.lower == 0.0) & (self.upper == 0.0)
        operand_zero = (operand.lower == 0.0) & (operand.upper == 0.0)
        lower = _down(self.lower + operand.lower)
        upper = _up(self.upper + operand.upper)
        return _new_interval(
            np.where(
                operand_zero,
                self.lower,
                np.where(self_zero, operand.lower, lower),
            ),
            np.where(
                operand_zero,
                self.upper,
                np.where(self_zero, operand.upper, upper),
            ),
        )

    def __radd__(self, other: Interval | npt.ArrayLike) -> Interval:
        return self + other

    def __sub__(self, other: Interval | npt.ArrayLike) -> Interval:
        operand = _as_interval(other)
        self_zero = (self.lower == 0.0) & (self.upper == 0.0)
        operand_zero = (operand.lower == 0.0) & (operand.upper == 0.0)
        lower = _down(self.lower - operand.upper)
        upper = _up(self.upper - operand.lower)
        return _new_interval(
            np.where(
                operand_zero,
                self.lower,
                np.where(self_zero, -operand.upper, lower),
            ),
            np.where(
                operand_zero,
                self.upper,
                np.where(self_zero, -operand.lower, upper),
            ),
        )

    def __rsub__(self, other: Interval | npt.ArrayLike) -> Interval:
        return _as_interval(other) - self

    def __mul__(self, other: Interval | npt.ArrayLike) -> Interval:
        operand = _as_interval(other)
        exact_zero = ((self.lower == 0.0) & (self.upper == 0.0)) | (
            (operand.lower == 0.0) & (operand.upper == 0.0)
        )
        products = np.stack(
            (
                self.lower * operand.lower,
                self.lower * operand.upper,
                self.upper * operand.lower,
                self.upper * operand.upper,
            )
        )
        return _new_interval(
            np.where(exact_zero, 0.0, _down(np.min(products, axis=0))),
            np.where(exact_zero, 0.0, _up(np.max(products, axis=0))),
        )

    def __rmul__(self, other: Interval | npt.ArrayLike) -> Interval:
        return self * other

    def __truediv__(self, other: Interval | npt.ArrayLike) -> Interval:
        operand = _as_interval(other)
        if np.any((operand.lower <= 0.0) & (operand.upper >= 0.0)):
            raise ValueError("interval divisor contains zero")
        reciprocal = _new_interval(
            _down(1.0 / operand.upper),
            _up(1.0 / operand.lower),
        )
        return self * reciprocal

    def __rtruediv__(self, other: Interval | npt.ArrayLike) -> Interval:
        return _as_interval(other) / self


@dataclass(frozen=True)
class Zonotope:
    """Immutable affine-generator set with generators stored by column."""

    center: npt.ArrayLike
    generators: npt.ArrayLike

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float).reshape(-1)
        generators = np.asarray(self.generators, dtype=float)
        if generators.size == 0:
            generators = np.empty((center.size, 0))
        elif generators.ndim == 1:
            generators = generators.reshape(center.size, 1)
        if generators.ndim != 2 or generators.shape[0] != center.size:
            raise ValueError("zonotope generator dimension mismatch")
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(generators)):
            raise ValueError("zonotope values must be finite")
        center = np.array(center, copy=True)
        generators = np.array(generators, copy=True)
        center.flags.writeable = False
        generators.flags.writeable = False
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "generators", generators)

    @classmethod
    def from_interval(cls, interval: Interval) -> Zonotope:
        """Return the axis-aligned zonotope equal to a vector interval."""

        if interval.lower.ndim != 1:
            raise ValueError("zonotopes require vector intervals")
        return cls(interval.center, np.diag(interval.radius))

    @property
    def radius(self) -> np.ndarray:
        """Return an outward-rounded axis-aligned hull radius."""

        radius = np.zeros(self.center.size)
        for index in range(self.generators.shape[1]):
            contribution = np.abs(self.generators[:, index])
            radius = np.where(
                contribution == 0.0,
                radius,
                _up(radius + contribution),
            )
        return radius

    @property
    def lower(self) -> np.ndarray:
        """Return the lower endpoint of the interval hull."""

        return self.interval_hull().lower

    @property
    def upper(self) -> np.ndarray:
        """Return the upper endpoint of the interval hull."""

        return self.interval_hull().upper

    def interval_hull(self) -> Interval:
        """Return the outward-rounded axis-aligned interval hull."""

        return Interval.from_midpoint(self.center, self.radius)

    def support(self, direction: npt.ArrayLike) -> float:
        """Return a rigorous upper bound on support in a direction."""

        vector = np.asarray(direction, dtype=float).reshape(-1)
        if vector.shape != self.center.shape:
            raise ValueError("support direction must match zonotope dimension")
        result = Interval.point(self.center).dot(vector)
        for generator in self.generators.T:
            result += Interval.point(generator).dot(vector).abs()
        return float(result.upper)

    def contains(self, point: npt.ArrayLike) -> bool:
        """Certify point membership with a bounded least-norm solution."""

        value = np.asarray(point, dtype=float).reshape(-1)
        if value.shape != self.center.shape or not np.all(np.isfinite(value)):
            return False
        if not self.interval_hull().contains(value):
            return False
        delta = value - self.center
        if self.generators.shape[1] == 0:
            return bool(np.array_equal(value, self.center))
        coefficients = np.linalg.lstsq(
            self.generators,
            delta,
            rcond=None,
        )[0]
        residual = self.generators @ coefficients - delta
        scale = max(
            1.0,
            float(np.linalg.norm(delta, ord=np.inf)),
            float(np.linalg.norm(self.generators, ord=np.inf)),
        )
        tolerance = 64.0 * np.finfo(float).eps * scale
        return bool(
            np.linalg.norm(residual, ord=np.inf) <= tolerance
            and np.max(np.abs(coefficients), initial=0.0) <= 1.0 + tolerance
        )

    def affine_map(
        self,
        matrix: npt.ArrayLike,
        offset: npt.ArrayLike = 0.0,
    ) -> Zonotope:
        """Return the zonotope image under a fixed affine map."""

        values = np.asarray(matrix, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.center.size:
            raise ValueError("affine map dimension mismatch")
        center_box = Interval.point(self.center).affine_map(values, offset)
        mapped = []
        error = center_box.radius
        for generator in self.generators.T:
            image = Interval.point(generator).affine_map(values)
            mapped.append(image.center)
            error = _up(error + image.radius)
        generators = (
            np.column_stack(mapped) if mapped else np.empty((values.shape[0], 0))
        )
        if np.any(error > 0.0):
            generators = np.column_stack((generators, np.diag(error)))
        return Zonotope(center_box.center, generators)

    def minkowski(self, other: Zonotope) -> Zonotope:
        """Return the Minkowski sum of two zonotopes."""

        if self.center.shape != other.center.shape:
            raise ValueError("zonotope dimension mismatch")
        center_box = Interval.point(self.center) + other.center
        generators = np.column_stack((self.generators, other.generators))
        if np.any(center_box.radius > 0.0):
            generators = np.column_stack((generators, np.diag(center_box.radius)))
        return Zonotope(center_box.center, generators)


@dataclass(frozen=True)
class AffineForm:
    """Affine generators with an axis-aligned nonlinear remainder."""

    center: npt.ArrayLike
    generators: npt.ArrayLike
    remainder: npt.ArrayLike
    basis: object | None = None

    __array_priority__ = 1000

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        generators = np.asarray(self.generators, dtype=float)
        remainder = np.broadcast_to(
            np.asarray(self.remainder, dtype=float),
            center.shape,
        )
        if generators.size == 0:
            generators = np.empty(center.shape + (0,))
        if generators.ndim != center.ndim + 1:
            raise ValueError("affine-form generators need one trailing axis")
        if generators.shape[:-1] != center.shape:
            raise ValueError("affine-form generator shape mismatch")
        if (
            not np.all(np.isfinite(center))
            or not np.all(np.isfinite(generators))
            or not np.all(np.isfinite(remainder))
            or np.any(remainder < 0.0)
        ):
            raise ValueError("affine-form values must be finite")
        center = np.array(center, copy=True)
        generators = np.array(generators, copy=True)
        remainder = np.array(remainder, copy=True)
        center.flags.writeable = False
        generators.flags.writeable = False
        remainder.flags.writeable = False
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "generators", generators)
        object.__setattr__(self, "remainder", remainder)
        if generators.shape[-1] and self.basis is None:
            object.__setattr__(self, "basis", object())

    @classmethod
    def from_interval(cls, interval: Interval) -> AffineForm:
        """Return an affine form enclosing an interval without correlation."""

        return cls(
            interval.center,
            np.empty(interval.lower.shape + (0,)),
            interval.radius,
        )

    @classmethod
    def from_zonotope(
        cls,
        zonotope: Zonotope,
        shape: tuple[int, ...] | None = None,
    ) -> AffineForm:
        """Return an affine form retaining every zonotope generator."""

        value_shape = (zonotope.center.size,) if shape is None else shape
        if int(np.prod(value_shape)) != zonotope.center.size:
            raise ValueError("affine-form shape must match zonotope dimension")
        count = zonotope.generators.shape[1]
        return cls(
            zonotope.center.reshape(value_shape),
            zonotope.generators.reshape(value_shape + (count,)),
            np.zeros(value_shape),
        )

    @property
    def generator_count(self) -> int:
        """Return the number of shared affine generators."""

        return int(self.generators.shape[-1])

    def interval_hull(self) -> Interval:
        """Return an outward-rounded interval hull."""

        radius = _sum_nonnegative(self.remainder, np.abs(self.generators))
        return Interval.from_midpoint(self.center, radius)

    def sum(self, axis: int | None = None) -> AffineForm:
        """Sum components while retaining shared-generator cancellation."""

        axes: tuple[int, ...]
        if axis is None:
            axes = tuple(range(self.center.ndim))
        else:
            axes = (axis % self.center.ndim,)
        center = Interval.point(self.center)
        generators = Interval.point(self.generators)
        remainder = Interval(-self.remainder, self.remainder)
        for current in sorted(axes, reverse=True):
            center = center.sum(axis=current)
            generators = generators.sum(axis=current)
            remainder = remainder.sum(axis=current)
        return _affine_from_boxes(
            center,
            generators,
            remainder,
            self.basis,
        )

    def cross(self, other: AffineForm | Interval | npt.ArrayLike) -> AffineForm:
        """Return the cross product along the final value axis."""

        operand = _as_affine_form(other)
        if self.center.shape[-1:] != (3,) or operand.center.shape[-1:] != (3,):
            raise ValueError("cross operands must end in three components")
        components = (
            self[..., 1] * operand[..., 2] - self[..., 2] * operand[..., 1],
            self[..., 2] * operand[..., 0] - self[..., 0] * operand[..., 2],
            self[..., 0] * operand[..., 1] - self[..., 1] * operand[..., 0],
        )
        return _stack_affine(components, axis=-1)

    def square(self) -> AffineForm:
        """Return the componentwise square enclosure."""

        return self * self

    def sqrt(self) -> AffineForm:
        """Return a componentwise square-root enclosure."""

        hull = self.interval_hull()
        if np.any(hull.lower < 0.0):
            raise ValueError("square root requires a nonnegative affine form")
        if np.any(hull.lower == 0.0):
            return AffineForm.from_interval(hull.sqrt())

        def derivative(value: Interval) -> Interval:
            return 0.5 / value.sqrt()

        return self._unary(Interval.sqrt, derivative)

    def norm(self, axis: int | None = None) -> AffineForm:
        """Return the Euclidean-norm enclosure."""

        squared = self.square().sum(axis=axis)
        if np.any(squared.interval_hull().lower < 0.0):
            return AffineForm.from_interval(self.interval_hull().norm(axis=axis))
        return squared.sqrt()

    def replace(
        self,
        mask: npt.ArrayLike,
        enclosure: Interval,
    ) -> AffineForm:
        """Replace selected leading components by interval enclosures."""

        selected = np.asarray(mask, dtype=bool)
        if selected.shape != self.center.shape[: selected.ndim]:
            raise ValueError("replacement mask shape mismatch")
        center = np.array(self.center)
        generators = np.array(self.generators)
        remainder = np.array(self.remainder)
        center[selected] = enclosure.center[selected]
        generators[selected] = 0.0
        remainder[selected] = enclosure.radius[selected]
        return AffineForm(center, generators, remainder, self.basis)

    def _unary(
        self,
        operation: Callable[[Interval], Interval],
        derivative: Callable[[Interval], Interval],
    ) -> AffineForm:
        hull = self.interval_hull()
        center_value = operation(Interval.point(self.center))
        derivative_center = derivative(Interval.point(self.center))
        derivative_hull = derivative(hull)
        generators = Interval.point(self.generators) * derivative_center[..., None]
        delta = hull - self.center
        remainder = (
            derivative_center * Interval(-self.remainder, self.remainder)
            + (derivative_hull - derivative_center) * delta
        )
        return _affine_from_boxes(
            center_value,
            generators,
            remainder,
            self.basis,
        )

    def __getitem__(self, key: object) -> AffineForm:
        generator_key = key if isinstance(key, tuple) else (key,)
        return AffineForm(
            self.center[key],
            self.generators[generator_key + (slice(None),)],
            self.remainder[key],
            self.basis,
        )

    def __neg__(self) -> AffineForm:
        return AffineForm(
            -self.center,
            -self.generators,
            self.remainder,
            self.basis,
        )

    def __add__(
        self,
        other: AffineForm | Interval | npt.ArrayLike,
    ) -> AffineForm:
        first, second, basis = _align_affine(self, _as_affine_form(other))
        return _affine_from_boxes(
            Interval.point(first.center) + second.center,
            Interval.point(first.generators) + second.generators,
            Interval(-first.remainder, first.remainder)
            + Interval(-second.remainder, second.remainder),
            basis,
        )

    def __radd__(
        self,
        other: AffineForm | Interval | npt.ArrayLike,
    ) -> AffineForm:
        return self + other

    def __sub__(
        self,
        other: AffineForm | Interval | npt.ArrayLike,
    ) -> AffineForm:
        return self + -_as_affine_form(other)

    def __rsub__(
        self,
        other: AffineForm | Interval | npt.ArrayLike,
    ) -> AffineForm:
        return _as_affine_form(other) - self

    def __mul__(
        self,
        other: AffineForm | Interval | npt.ArrayLike,
    ) -> AffineForm:
        first, second, basis = _align_affine(self, _as_affine_form(other))
        first_delta = first.interval_hull() - first.center
        second_delta = second.interval_hull() - second.center
        center = Interval.point(first.center) * second.center
        generators = (
            Interval.point(first.generators) * second.center[..., None]
            + Interval.point(second.generators) * first.center[..., None]
        )
        linear_remainder = (
            Interval(-first.remainder, first.remainder) * second.center
            + Interval(-second.remainder, second.remainder) * first.center
        )
        nonlinear_remainder = first_delta * second_delta
        return _affine_from_boxes(
            center,
            generators,
            linear_remainder + nonlinear_remainder,
            basis,
        )

    def __rmul__(
        self,
        other: AffineForm | Interval | npt.ArrayLike,
    ) -> AffineForm:
        return self * other

    def __truediv__(
        self,
        other: AffineForm | Interval | npt.ArrayLike,
    ) -> AffineForm:
        operand = _as_affine_form(other)
        hull = operand.interval_hull()
        if np.any((hull.lower <= 0.0) & (hull.upper >= 0.0)):
            raise ZeroDivisionError("affine-form denominator contains zero")

        def reciprocal(value: Interval) -> Interval:
            return 1.0 / value

        def derivative(value: Interval) -> Interval:
            return -1.0 / value.square()

        return self * operand._unary(reciprocal, derivative)

    def __rtruediv__(
        self,
        other: AffineForm | Interval | npt.ArrayLike,
    ) -> AffineForm:
        return _as_affine_form(other) / self


def _align_affine(
    first: AffineForm,
    second: AffineForm,
) -> tuple[AffineForm, AffineForm, object | None]:
    shape = np.broadcast_shapes(first.center.shape, second.center.shape)
    if first.generator_count and second.generator_count:
        if first.basis is not second.basis:
            raise ValueError("affine forms have independent generator bases")
        if first.generator_count != second.generator_count:
            raise ValueError("affine forms have inconsistent generator counts")
        count = first.generator_count
        basis = first.basis
    else:
        count = max(first.generator_count, second.generator_count)
        basis = first.basis if first.generator_count else second.basis

    def broadcast(value: AffineForm) -> AffineForm:
        generators = value.generators
        if value.generator_count == 0 and count:
            generators = np.zeros(value.center.shape + (count,))
        return AffineForm(
            np.broadcast_to(value.center, shape),
            np.broadcast_to(generators, shape + (count,)),
            np.broadcast_to(value.remainder, shape),
            basis,
        )

    return broadcast(first), broadcast(second), basis


def _affine_from_boxes(
    center: Interval,
    generators: Interval,
    remainder: Interval,
    basis: object | None,
) -> AffineForm:
    shifted = center + remainder.center
    error = _sum_nonnegative(
        shifted.radius,
        remainder.radius,
        generators.radius,
    )
    return AffineForm(
        shifted.center,
        generators.center,
        error,
        basis,
    )


def _as_affine_form(
    value: AffineForm | Interval | npt.ArrayLike,
) -> AffineForm:
    if isinstance(value, AffineForm):
        return value
    if isinstance(value, Interval):
        return AffineForm.from_interval(value)
    return AffineForm.from_interval(Interval.point(value))


def _stack_affine(
    values: tuple[AffineForm, ...],
    axis: int,
) -> AffineForm:
    basis = values[0].basis
    count = values[0].generator_count
    if any(value.basis is not basis for value in values):
        raise ValueError("affine forms have independent generator bases")
    output_ndim = values[0].center.ndim + 1
    value_axis = axis % output_ndim
    return AffineForm(
        np.stack([value.center for value in values], axis=value_axis),
        np.stack([value.generators for value in values], axis=value_axis),
        np.stack([value.remainder for value in values], axis=value_axis),
        basis if count else None,
    )


def _sum_nonnegative(*values: np.ndarray) -> np.ndarray:
    shape = np.broadcast_shapes(*(value.shape[:-1] for value in values[-1:]))
    result = np.zeros(shape)
    for position, value in enumerate(values):
        array = np.asarray(value, dtype=float)
        contributions = (
            np.moveaxis(array, -1, 0)
            if position == len(values) - 1 and array.ndim == result.ndim + 1
            else np.broadcast_to(array, shape)[None, ...]
        )
        for contribution in contributions:
            result = np.where(
                contribution == 0.0,
                result,
                _up(result + contribution),
            )
    return result


def _new_interval(lower: npt.ArrayLike, upper: npt.ArrayLike) -> Interval:
    lower_array, upper_array = np.broadcast_arrays(
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
    )
    lower_array.flags.writeable = False
    upper_array.flags.writeable = False
    result = object.__new__(Interval)
    object.__setattr__(result, "lower", lower_array)
    object.__setattr__(result, "upper", upper_array)
    return result


def _as_interval(value: Interval | npt.ArrayLike) -> Interval:
    if isinstance(value, Interval):
        return value
    array = np.asarray(value, dtype=float)
    return _new_interval(array, array)
