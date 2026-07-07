"""Configurable annular Gaussian updraft model."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


PARAMETER_ORDER = ("a_ring_m_s", "r_ring_m", "delta_r_m", "w0_m_s")
MIN_DELTA_R_M = 1.0e-12


def annular_gaussian(
    radius_m: npt.ArrayLike,
    a_ring_m_s: npt.ArrayLike,
    r_ring_m: npt.ArrayLike,
    delta_r_m: npt.ArrayLike,
    w0_m_s: npt.ArrayLike,
) -> np.ndarray:
    """Evaluate the annular Gaussian vertical velocity profile."""
    radius = np.asarray(radius_m, dtype=float)
    delta = np.maximum(np.asarray(delta_r_m, dtype=float), MIN_DELTA_R_M)
    return (
        np.asarray(w0_m_s, dtype=float)
        + np.asarray(a_ring_m_s, dtype=float)
        * np.exp(-((radius - np.asarray(r_ring_m, dtype=float)) / delta) ** 2)
    )


@dataclass(frozen=True)
class AnnularGaussianUpdraft:
    """A configurable sum of annular Gaussian updraft sources."""

    source_xy_m: npt.ArrayLike
    z_axis_m: npt.ArrayLike
    parameters: npt.ArrayLike
    source_strengths: npt.ArrayLike | None = None
    active_sources: npt.ArrayLike | None = None
    name: str = "annular_gaussian_updraft"
    source: str = "configured annular Gaussian updraft field"
    _source_xy_m: np.ndarray = field(init=False, repr=False)
    _z_axis_m: np.ndarray = field(init=False, repr=False)
    _parameters: np.ndarray = field(init=False, repr=False)
    _shared_parameters: bool = field(init=False, repr=False)
    _source_strengths: np.ndarray = field(init=False, repr=False)
    _active_sources: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        source_xy = np.asarray(self.source_xy_m, dtype=float)
        z_axis = np.asarray(self.z_axis_m, dtype=float)
        parameters = np.asarray(self.parameters, dtype=float)
        n_sources = self._validate_axes(source_xy, z_axis)
        shared = self._parameters_are_shared(
            parameters,
            n_sources,
            z_axis.size,
        )

        object.__setattr__(self, "_source_xy_m", source_xy)
        object.__setattr__(self, "_z_axis_m", z_axis)
        object.__setattr__(self, "_parameters", parameters)
        object.__setattr__(self, "_shared_parameters", shared)
        object.__setattr__(
            self,
            "_source_strengths",
            self._optional_array(
                self.source_strengths,
                n_sources,
                fill_value=1.0,
                dtype=float,
                label="source_strengths",
            ),
        )
        object.__setattr__(
            self,
            "_active_sources",
            self._optional_array(
                self.active_sources,
                n_sources,
                fill_value=True,
                dtype=bool,
                label="active_sources",
            ),
        )

    def __call__(self, points_w_up_m: npt.ArrayLike) -> np.ndarray:
        """Return wind vectors in public z-up world coordinates."""
        points = np.asarray(points_w_up_m, dtype=float).reshape(-1, 3)
        wind = np.zeros((points.shape[0], 3), dtype=float)
        z_query = np.clip(points[:, 2], self._z_axis_m[0], self._z_axis_m[-1])
        shared_params = None
        if self._shared_parameters:
            shared_params = self._interpolate_parameters(
                self._parameters,
                z_query,
            )

        for source_idx, source_xy in enumerate(self._source_xy_m):
            if not self._active_sources[source_idx]:
                continue
            params = shared_params
            if params is None:
                params = self._interpolate_parameters(
                    self._parameters[source_idx],
                    z_query,
                )
            radius = np.hypot(
                points[:, 0] - source_xy[0],
                points[:, 1] - source_xy[1],
            )
            vertical = annular_gaussian(
                radius,
                params[:, 0],
                params[:, 1],
                params[:, 2],
                params[:, 3],
            )
            wind[:, 2] += self._source_strengths[source_idx] * vertical
        return wind

    def params_at_z(self, z_m: float, source_idx: int = 0) -> np.ndarray:
        """Return one source's interpolated parameters at a height."""
        z_query = np.asarray([z_m], dtype=float)
        z_query = np.clip(z_query, self._z_axis_m[0], self._z_axis_m[-1])
        if self._shared_parameters:
            params = self._parameters
        else:
            params = self._parameters[int(source_idx)]
        return self._interpolate_parameters(params, z_query)[0]

    @staticmethod
    def _validate_axes(source_xy: np.ndarray, z_axis: np.ndarray) -> int:
        if source_xy.ndim != 2 or source_xy.shape[1] != 2:
            raise ValueError("source_xy_m must have shape (n_sources, 2).")
        if z_axis.ndim != 1 or z_axis.size == 0:
            raise ValueError(
                "z_axis_m must be a non-empty one-dimensional axis."
            )
        if np.any(np.diff(z_axis) <= 0.0):
            raise ValueError("z_axis_m must be strictly increasing.")
        return int(source_xy.shape[0])

    @staticmethod
    def _parameters_are_shared(
        parameters: np.ndarray,
        n_sources: int,
        n_heights: int,
    ) -> bool:
        profile_shape = (n_heights, len(PARAMETER_ORDER))
        field_shape = (n_sources, n_heights, len(PARAMETER_ORDER))
        if parameters.shape == profile_shape:
            return True
        if parameters.shape == field_shape:
            return False
        raise ValueError(
            "parameters must have shape "
            f"{profile_shape} or {field_shape} for {PARAMETER_ORDER}."
        )

    @staticmethod
    def _optional_array(
        values: npt.ArrayLike | None,
        size: int,
        *,
        fill_value: float | bool,
        dtype: type,
        label: str,
    ) -> np.ndarray:
        if values is None:
            return np.full(size, fill_value, dtype=dtype)
        array = np.asarray(values, dtype=dtype)
        if array.shape != (size,):
            raise ValueError(f"{label} must have shape ({size},).")
        return array

    def _interpolate_parameters(
        self,
        parameters: np.ndarray,
        z_query: np.ndarray,
    ) -> np.ndarray:
        out = np.empty((z_query.size, len(PARAMETER_ORDER)), dtype=float)
        for idx in range(len(PARAMETER_ORDER)):
            out[:, idx] = np.interp(
                z_query,
                self._z_axis_m,
                parameters[:, idx],
            )
        return out
