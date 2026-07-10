"""Aerodynamic recoverability coordinate map."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import radians

import numpy as np
import numpy.typing as npt

from models.state import (
    G_M_S2,
    as_state,
    sink_rate_down,
)


AirData = Callable[[np.ndarray], tuple[float, float]]
EnergyRate = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class RecoverabilityConfig:
    """Physical scales used by the recoverability coordinates."""

    energy_recovery_m2_s2: float = 12.0
    energy_trim_m2_s2: float = 35.0
    stall_speed_m_s: float = 3.5
    trim_speed_m_s: float = 6.0
    alpha_max_rad: float = radians(30.0)
    alpha_trim_rad: float = radians(6.0)
    pitch_rate_max_rad_s: float = 4.0
    pitch_rate_trim_rad_s: float = 0.5
    z_min_m: float = 0.4
    z_max_m: float = 3.5
    recovery_height_m: float = 0.5
    reaction_time_s: float = 0.25
    boundary_time_cap_s: float = 2.0
    surface_limit_rad: tuple[float, float, float] = (
        radians(19.3),
        radians(23.7),
        radians(33.0),
    )
    sink_energy_rate_m2_s3: float = 5.0


@dataclass(frozen=True)
class RecoverabilityMap:
    """Map raw states into dimensionless recoverability coordinates."""

    air_data: AirData
    energy_rate: EnergyRate
    config: RecoverabilityConfig = RecoverabilityConfig()

    def __call__(self, state: npt.ArrayLike) -> np.ndarray:
        """Return the recoverability coordinate vector."""

        x = as_state(state)
        velocity = x[6:9]
        actuator = x[12:15]
        z_w = float(x[2])
        speed = float(np.linalg.norm(velocity))
        va, alpha = self.air_data(x)
        sink_rate = sink_rate_down(x)
        vz_up = -sink_rate
        floor_time, ceiling_time = _boundary_times(
            z_w,
            vz_up,
            self.config,
        )
        energy = G_M_S2 * z_w + 0.5 * speed * speed
        surface_limit = np.asarray(self.config.surface_limit_rad, dtype=float)
        return np.array(
            [
                _scale(
                    energy,
                    self.config.energy_recovery_m2_s2,
                    self.config.energy_trim_m2_s2,
                ),
                _scale(
                    va,
                    self.config.stall_speed_m_s,
                    self.config.trim_speed_m_s,
                ),
                (
                    self.config.alpha_max_rad - abs(alpha)
                )
                / (
                    self.config.alpha_max_rad
                    - self.config.alpha_trim_rad
                ),
                (
                    self.config.pitch_rate_max_rad_s - abs(float(x[10]))
                )
                / (
                    self.config.pitch_rate_max_rad_s
                    - self.config.pitch_rate_trim_rad_s
                ),
                (z_w - self.config.z_min_m) / self.config.recovery_height_m,
                (self.config.z_max_m - z_w) / self.config.recovery_height_m,
                floor_time / self.config.reaction_time_s,
                ceiling_time / self.config.reaction_time_s,
                float(
                    np.min((surface_limit - np.abs(actuator)) / surface_limit)
                ),
                self.energy_rate(x)
                / abs(self.config.sink_energy_rate_m2_s3),
            ],
            dtype=float,
        )

    def margin(self, state: npt.ArrayLike) -> float:
        """Return the instantaneous recoverability margin."""

        return float(np.min(self(state)))


def _scale(value: float, low: float, high: float) -> float:
    return (value - low) / (high - low)


def _boundary_times(
    z_w: float,
    vz_up: float,
    config: RecoverabilityConfig,
) -> tuple[float, float]:
    floor_time = config.boundary_time_cap_s
    ceiling_time = config.boundary_time_cap_s
    if vz_up < 0.0:
        floor_time = (z_w - config.z_min_m) / -vz_up
    if vz_up > 0.0:
        ceiling_time = (config.z_max_m - z_w) / vz_up
    return floor_time, ceiling_time
