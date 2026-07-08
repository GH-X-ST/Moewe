"""Configurable annular Gaussian updraft model."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


PARAMETER_ORDER = ("a_ring_m_s", "r_ring_m", "delta_r_m")
DEFAULT_OVERLAP_BETA = 0.5
DEFAULT_OVERLAP_EPSILON = 1.0e-12
DEFAULT_Z_AXIS_M = (0.2, 0.35, 0.5, 0.75, 1.1, 1.6, 2.2)
DEFAULT_PROFILE_PARAMETERS = (
    (4.993168672, 0.306463654579, 0.207643648291),
    (4.6907967727, 0.338099122176, 0.259135645401),
    (4.34842187165, 0.504176146278, 0.255494706674),
    (3.44842281242, 0.434609212945, 0.327207377932),
    (2.46803251123, 0.40527002812, 0.413003916537),
    (2.66435258512, 0.427794522449, 0.416874327164),
    (2.06749412407, 0.401702057616, 0.517533266127),
)


@dataclass(frozen=True)
class UpdraftConfig:
    """Height profile and overlap closure for an updraft plug-in."""

    z_axis_m: npt.ArrayLike
    parameters: npt.ArrayLike
    overlap_beta: float = DEFAULT_OVERLAP_BETA
    overlap_epsilon: float = DEFAULT_OVERLAP_EPSILON


def default_updraft_config() -> UpdraftConfig:
    """Return the default annular Gaussian updraft profile."""

    return UpdraftConfig(
        z_axis_m=DEFAULT_Z_AXIS_M,
        parameters=DEFAULT_PROFILE_PARAMETERS,
    )


def annular_gaussian(
    radius_m: npt.ArrayLike,
    a_ring_m_s: npt.ArrayLike,
    r_ring_m: npt.ArrayLike,
    delta_r_m: npt.ArrayLike,
) -> np.ndarray:
    """Evaluate the annular Gaussian vertical velocity profile."""
    radius = np.asarray(radius_m, dtype=float)
    delta = np.asarray(delta_r_m, dtype=float)
    return np.asarray(a_ring_m_s, dtype=float) * np.exp(
        -((radius - np.asarray(r_ring_m, dtype=float)) / delta) ** 2
    )


@dataclass
class AnnularGaussianUpdraft:
    """A configurable annular Gaussian updraft field."""

    source_xy_m: npt.ArrayLike
    source_strengths: npt.ArrayLike | None = None
    config: UpdraftConfig = field(default_factory=default_updraft_config)

    def __post_init__(self) -> None:
        self.source_xy_m = np.asarray(self.source_xy_m, dtype=float)
        self.z_axis_m = np.asarray(self.config.z_axis_m, dtype=float)
        self.parameters = np.asarray(self.config.parameters, dtype=float)
        self.overlap_beta = float(self.config.overlap_beta)
        self.overlap_epsilon = float(self.config.overlap_epsilon)

        if self.source_xy_m.ndim != 2 or self.source_xy_m.shape[1] != 2:
            raise ValueError("source_xy_m must have shape (n_sources, 2).")
        if self.z_axis_m.ndim != 1 or self.z_axis_m.size == 0:
            raise ValueError(
                "z_axis_m must be a non-empty one-dimensional axis."
            )
        if np.any(np.diff(self.z_axis_m) <= 0.0):
            raise ValueError("z_axis_m must be strictly increasing.")

        n_sources = self.source_xy_m.shape[0]
        profile_shape = (self.z_axis_m.size, len(PARAMETER_ORDER))
        field_shape = (n_sources, self.z_axis_m.size, len(PARAMETER_ORDER))
        if self.parameters.shape not in (profile_shape, field_shape):
            raise ValueError(
                "parameters must have shape "
                f"{profile_shape} or {field_shape} for {PARAMETER_ORDER}."
            )
        if np.any(self.parameters[..., 1] < 0.0):
            raise ValueError("r_ring_m must be non-negative.")
        if np.any(self.parameters[..., 2] <= 0.0):
            raise ValueError("delta_r_m must be positive.")

        if self.source_strengths is None:
            self.source_strengths = np.ones(n_sources, dtype=float)
        else:
            self.source_strengths = np.asarray(
                self.source_strengths,
                dtype=float,
            )
            if self.source_strengths.shape != (n_sources,):
                raise ValueError(
                    f"source_strengths must have shape ({n_sources},)."
                )

    def __call__(self, points_w_up_m: npt.ArrayLike) -> np.ndarray:
        """Return wind vectors in public z-up world coordinates."""
        points = np.asarray(points_w_up_m, dtype=float).reshape(-1, 3)
        z_query = np.clip(points[:, 2], self.z_axis_m[0], self.z_axis_m[-1])
        source_sum = np.zeros(points.shape[0], dtype=float)
        source_abs_sum = np.zeros(points.shape[0], dtype=float)
        source_square_sum = np.zeros(points.shape[0], dtype=float)

        shared_params = None
        if self.parameters.ndim == 2:
            shared_params = self._interpolate_parameters(
                self.parameters,
                z_query,
            )

        for source_idx, source_xy in enumerate(self.source_xy_m):
            strength = self.source_strengths[source_idx]
            if strength == 0.0:
                continue
            params = shared_params
            if params is None:
                params = self._interpolate_parameters(
                    self.parameters[source_idx],
                    z_query,
                )
            radius = np.hypot(
                points[:, 0] - source_xy[0],
                points[:, 1] - source_xy[1],
            )
            phi = strength * annular_gaussian(
                radius,
                params[:, 0],
                params[:, 1],
                params[:, 2],
            )
            source_sum += phi
            source_abs_sum += np.abs(phi)
            source_square_sum += phi * phi

        wind = np.zeros((points.shape[0], 3), dtype=float)
        wind[:, 2] = self._combine_sources(
            source_sum,
            source_abs_sum,
            source_square_sum,
        )
        return wind

    def _interpolate_parameters(
        self,
        parameters: np.ndarray,
        z_query: np.ndarray,
    ) -> np.ndarray:
        out = np.empty((z_query.size, len(PARAMETER_ORDER)), dtype=float)
        for idx in range(len(PARAMETER_ORDER)):
            out[:, idx] = np.interp(
                z_query,
                self.z_axis_m,
                parameters[:, idx],
            )
        return out

    def _combine_sources(
        self,
        source_sum: np.ndarray,
        source_abs_sum: np.ndarray,
        source_square_sum: np.ndarray,
    ) -> np.ndarray:
        beta = float(self.overlap_beta)
        if beta == 0.0:
            return source_sum

        n_eff = np.ones_like(source_sum)
        active = source_abs_sum > 0.0
        n_eff[active] = source_abs_sum[active] ** 2 / (
            source_square_sum[active] + float(self.overlap_epsilon)
        )
        return source_sum / np.maximum(n_eff, 1.0) ** beta


def build_updraft(
    source_xy_m: npt.ArrayLike,
    source_strengths: npt.ArrayLike | None = None,
    config: UpdraftConfig | None = None,
) -> AnnularGaussianUpdraft:
    """Build a reusable updraft plug-in instance."""

    return AnnularGaussianUpdraft(
        source_xy_m=source_xy_m,
        source_strengths=source_strengths,
        config=default_updraft_config() if config is None else config,
    )
