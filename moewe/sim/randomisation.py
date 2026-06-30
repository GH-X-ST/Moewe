"""Seeded bounded randomisation for simulator configuration studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class UniformRange:
    low: float
    high: float

    def validate(self) -> None:
        if not np.isfinite([self.low, self.high]).all():
            raise ValueError("Randomisation bounds must be finite.")
        if self.low > self.high:
            raise ValueError("Randomisation lower bound must not exceed upper bound.")

    def sample(self, rng: np.random.Generator) -> float:
        self.validate()
        return float(rng.uniform(float(self.low), float(self.high)))


RandomSpec = Mapping[str, "UniformRange | RandomSpec | float | int"]


def sample_parameters(spec: RandomSpec, seed: int) -> dict[str, object]:
    """Sample a nested parameter dictionary from explicit bounded ranges."""

    rng = np.random.default_rng(int(seed))

    def _sample(value: object) -> object:
        if isinstance(value, UniformRange):
            return value.sample(rng)
        if isinstance(value, Mapping):
            return {str(key): _sample(child) for key, child in value.items()}
        if isinstance(value, tuple) and len(value) == 2:
            return UniformRange(float(value[0]), float(value[1])).sample(rng)
        if isinstance(value, (float, int)):
            return float(value)
        raise TypeError(f"Unsupported randomisation spec value: {value!r}")

    return {str(key): _sample(value) for key, value in spec.items()}


def default_randomisation_spec() -> dict[str, object]:
    """Return conservative parameter ranges for later logged rollouts."""

    return {
        "airframe": {
            "mass_scale": UniformRange(0.95, 1.05),
            "centre_of_gravity_x_m": UniformRange(-0.010, 0.010),
            "centre_of_gravity_z_m": UniformRange(-0.005, 0.005),
            "inertia_scale": UniformRange(0.90, 1.10),
        },
        "aero": {
            "lift_slope_scale": UniformRange(0.90, 1.10),
            "drag_scale": UniformRange(0.90, 1.20),
            "stall_start_rad": UniformRange(0.18, 0.24),
            "stall_width_rad": UniformRange(0.04, 0.08),
            "lateral_derivative_scale": UniformRange(0.75, 1.25),
        },
        "actuator": {
            "effectiveness_scale": UniformRange(0.80, 1.10),
            "time_constant_scale": UniformRange(0.75, 1.50),
            "delay_s": UniformRange(0.0, 0.04),
        },
        "updraft": {
            "fan_centre_x_m": UniformRange(-0.03, 0.03),
            "fan_centre_y_m": UniformRange(-0.03, 0.03),
            "strength_scale": UniformRange(0.80, 1.20),
            "ring_radius_scale": UniformRange(0.85, 1.15),
            "ring_thickness_scale": UniformRange(0.80, 1.25),
            "background_vertical_m_s": UniformRange(-0.05, 0.05),
        },
        "air": {
            "density_kg_m3": UniformRange(1.15, 1.25),
        },
    }
