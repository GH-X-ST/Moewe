"""Body-relative affine local-flow model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from control.interval import Interval


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
    """Compact bounds for time-varying body-relative affine flow."""

    center_lower_m_s: npt.ArrayLike
    center_upper_m_s: npt.ArrayLike
    gradient_lower_s: npt.ArrayLike
    gradient_upper_s: npt.ArrayLike
    remainder_abs_m_s: npt.ArrayLike
    rate_abs_m_s2: npt.ArrayLike
    gradient_rate_abs_s2: npt.ArrayLike = ((0.0, 0.0, 0.0),) * 3
    remainder_rate_abs_m_s2: npt.ArrayLike = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        fields = (
            ("center_lower_m_s", self.center_lower_m_s, (3,)),
            ("center_upper_m_s", self.center_upper_m_s, (3,)),
            ("gradient_lower_s", self.gradient_lower_s, (3, 3)),
            ("gradient_upper_s", self.gradient_upper_s, (3, 3)),
            ("remainder_abs_m_s", self.remainder_abs_m_s, (3,)),
            ("rate_abs_m_s2", self.rate_abs_m_s2, (3,)),
            ("gradient_rate_abs_s2", self.gradient_rate_abs_s2, (3, 3)),
            (
                "remainder_rate_abs_m_s2",
                self.remainder_rate_abs_m_s2,
                (3,),
            ),
        )
        for name, value, shape in fields:
            _set_array(self, name, value, shape)
        if np.any(self.center_lower_m_s > self.center_upper_m_s):
            raise ValueError("flow center bounds are empty")
        if np.any(self.gradient_lower_s > self.gradient_upper_s):
            raise ValueError("flow gradient bounds are empty")
        if np.any(self.remainder_abs_m_s < 0.0):
            raise ValueError("flow remainder bounds must be nonnegative")
        if (
            np.any(self.rate_abs_m_s2 < 0.0)
            or np.any(self.gradient_rate_abs_s2 < 0.0)
            or np.any(self.remainder_rate_abs_m_s2 < 0.0)
        ):
            raise ValueError("flow rate bounds must be nonnegative")

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

    def rate_contains(
        self,
        previous: AffineFlow,
        current: AffineFlow,
        dt_s: float,
    ) -> bool:
        """Return whether value, gradient, and remainder rates are bounded."""

        changes = (
            np.asarray(current.center_b_m_s) - np.asarray(previous.center_b_m_s),
            np.asarray(current.gradient_b_s) - np.asarray(previous.gradient_b_s),
            np.asarray(current.remainder_b_m_s) - np.asarray(previous.remainder_b_m_s),
        )
        limits = (
            dt_s * np.asarray(self.rate_abs_m_s2),
            dt_s * np.asarray(self.gradient_rate_abs_s2),
            dt_s * np.asarray(self.remainder_rate_abs_m_s2),
        )
        return all(
            np.all(np.abs(change) <= limit)
            for change, limit in zip(changes, limits, strict=True)
        )


def _gradient_at(gradient: Interval, location: np.ndarray) -> Interval:
    components = [gradient[row].dot(location) for row in range(3)]
    return Interval(
        np.array([component.lower for component in components]),
        np.array([component.upper for component in components]),
    )


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
