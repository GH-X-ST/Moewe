"""Body-relative affine local-flow model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from control.interval import AffineForm, Interval, Zonotope

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
        remainder = np.asarray(self.remainder_b_m_s, dtype=float)
        if remainder.shape[-1:] != (3,):
            raise ValueError("flow remainder must end in three components")
        _set_array(self, "remainder_b_m_s", remainder, remainder.shape)

    def strip_flow(self, strip_b_m: npt.ArrayLike) -> np.ndarray:
        """Return body-frame flow at each aerodynamic strip."""

        locations = np.asarray(strip_b_m, dtype=float).reshape(-1, 3)
        center = np.asarray(self.center_b_m_s, dtype=float).reshape(3)
        gradient = np.asarray(self.gradient_b_s, dtype=float).reshape(3, 3)
        remainder = np.asarray(self.remainder_b_m_s, dtype=float).reshape(
            locations.shape
        )
        return center + locations @ gradient.T + remainder


@dataclass(frozen=True)
class FlowBounds:
    """Amplitude bounds for arbitrarily time-varying affine flow."""

    center_lower_m_s: npt.ArrayLike
    center_upper_m_s: npt.ArrayLike
    gradient_lower_s: npt.ArrayLike
    gradient_upper_s: npt.ArrayLike
    remainder_abs_m_s: npt.ArrayLike

    def __post_init__(self) -> None:
        fields = (
            ("center_lower_m_s", self.center_lower_m_s, (3,)),
            ("center_upper_m_s", self.center_upper_m_s, (3,)),
            ("gradient_lower_s", self.gradient_lower_s, (3, 3)),
            ("gradient_upper_s", self.gradient_upper_s, (3, 3)),
            ("remainder_abs_m_s", self.remainder_abs_m_s, (3,)),
        )
        for name, value, shape in fields:
            _set_array(self, name, value, shape)
        if np.any(self.center_lower_m_s > self.center_upper_m_s):
            raise ValueError("flow center bounds are empty")
        if np.any(self.gradient_lower_s > self.gradient_upper_s):
            raise ValueError("flow gradient bounds are empty")
        if np.any(self.remainder_abs_m_s < 0.0):
            raise ValueError("flow remainder bounds must be nonnegative")

    def strip_bounds(
        self,
        strip_b_m: npt.ArrayLike,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return interval bounds for all body-frame strip flows."""

        locations = np.asarray(strip_b_m, dtype=float).reshape(-1, 3)
        center = Interval(self.center_lower_m_s, self.center_upper_m_s)
        gradient = Interval(self.gradient_lower_s, self.gradient_upper_s)
        remainder = Interval(
            -self.remainder_abs_m_s,
            self.remainder_abs_m_s,
        )
        strips = [
            center + _gradient_at(gradient, location) + remainder
            for location in locations
        ]
        return (
            np.stack([strip.lower for strip in strips]),
            np.stack([strip.upper for strip in strips]),
        )

    def affine_form(self, strip_b_m: npt.ArrayLike) -> AffineForm:
        """Return the joint flow zonotope as a shaped affine form."""

        locations = np.asarray(strip_b_m, dtype=float).reshape(-1, 3)
        zonotope = self.joint_zonotope(locations)
        return AffineForm.from_zonotope(
            zonotope,
            (locations.shape[0] + 1, 3),
        )

    def joint_zonotope(self, strip_b_m: npt.ArrayLike) -> Zonotope:
        """Return the stacked centre-and-strip flow zonotope."""

        locations = np.asarray(strip_b_m, dtype=float).reshape(-1, 3)
        center = Interval(self.center_lower_m_s, self.center_upper_m_s)
        gradient = Interval(self.gradient_lower_s, self.gradient_upper_s)
        midpoint_gradient = Interval.point(gradient.center)
        values = [Interval.point(center.center)]
        values.extend(
            Interval.point(center.center) + _gradient_at(midpoint_gradient, location)
            for location in locations
        )
        value = Interval(
            np.stack([item.lower for item in values]),
            np.stack([item.upper for item in values]),
        )
        strip_count = locations.shape[0]
        generators = np.zeros(
            value.lower.shape + (SHARED_FLOW_GENERATOR_COUNT + 3 * strip_count,)
        )
        numerical_error = np.array(value.radius, copy=True)
        for component in range(3):
            generators[:, component, component] = center.radius[component]
            for coordinate in range(3):
                coefficient = (
                    Interval.point(locations[:, coordinate])
                    * gradient.radius[component, coordinate]
                )
                column = 3 + 3 * coordinate + component
                generators[1:, component, column] = coefficient.center
                numerical_error[1:, component] = _add_nonnegative(
                    numerical_error[1:, component],
                    coefficient.radius,
                )
        remainder_radius = _add_nonnegative(
            np.broadcast_to(
                self.remainder_abs_m_s,
                numerical_error[1:].shape,
            ),
            numerical_error[1:],
        )
        for strip in range(strip_count):
            for component in range(3):
                column = SHARED_FLOW_GENERATOR_COUNT + 3 * strip + component
                generators[strip + 1, component, column] = remainder_radius[
                    strip,
                    component,
                ]
        return Zonotope(
            value.center.reshape(-1),
            generators.reshape(value.center.size, -1),
        )

    def contains(self, flow: AffineFlow) -> bool:
        """Return whether an affine-flow realization is inside the set."""

        center = np.asarray(flow.center_b_m_s, dtype=float).reshape(3)
        gradient = np.asarray(flow.gradient_b_s, dtype=float).reshape(3, 3)
        remainder = np.asarray(flow.remainder_b_m_s, dtype=float)
        return bool(
            np.all(center >= np.asarray(self.center_lower_m_s))
            and np.all(center <= np.asarray(self.center_upper_m_s))
            and np.all(gradient >= np.asarray(self.gradient_lower_s))
            and np.all(gradient <= np.asarray(self.gradient_upper_s))
            and np.all(np.abs(remainder) <= np.asarray(self.remainder_abs_m_s))
        )


def _gradient_at(gradient: Interval, location: np.ndarray) -> Interval:
    components = [gradient[row].dot(location) for row in range(3)]
    return Interval(
        np.array([component.lower for component in components]),
        np.array([component.upper for component in components]),
    )


def _add_nonnegative(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    total = np.nextafter(first + second, np.inf)
    return np.where(first == 0.0, second, np.where(second == 0.0, first, total))


def _set_array(
    instance: object,
    name: str,
    value: npt.ArrayLike,
    shape: tuple[int, ...],
) -> None:
    array = np.asarray(value, dtype=float).reshape(shape).copy()
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.flags.writeable = False
    object.__setattr__(instance, name, array)
