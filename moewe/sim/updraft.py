"""Compact annular updraft model for z-up world coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class HeightProfile:
    z_m: tuple[float, ...]
    values: tuple[float, ...]

    def evaluate(self, z_query_m: np.ndarray) -> np.ndarray:
        z_axis = np.asarray(self.z_m, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if z_axis.ndim != 1 or values.ndim != 1 or z_axis.size != values.size:
            raise ValueError("HeightProfile z and value arrays must have matching length.")
        if np.any(np.diff(z_axis) <= 0.0):
            raise ValueError("HeightProfile z values must be strictly increasing.")
        return np.interp(np.asarray(z_query_m, dtype=float), z_axis, values)


ScalarOrProfile = float | HeightProfile


def _evaluate(value: ScalarOrProfile, z_query_m: np.ndarray) -> np.ndarray:
    if isinstance(value, HeightProfile):
        return value.evaluate(z_query_m)
    return np.full_like(np.asarray(z_query_m, dtype=float), float(value), dtype=float)


@dataclass(frozen=True)
class FanUpdraft:
    """One annular vertical-flow contribution in world z-up coordinates."""

    centre_xy_m: tuple[float, float]
    strength_m_s: ScalarOrProfile
    ring_radius_m: ScalarOrProfile
    ring_thickness_m: ScalarOrProfile
    background_m_s: ScalarOrProfile = 0.0
    vertical_decay_m: float | None = None
    reference_z_m: float = 0.0
    harmonic_cos: tuple[float, ...] = ()
    harmonic_sin: tuple[float, ...] = ()
    fluctuation_std_m_s: float = 0.0

    def vertical_speed(self, points_w_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_w_m, dtype=float).reshape(-1, 3)
        dx = points[:, 0] - float(self.centre_xy_m[0])
        dy = points[:, 1] - float(self.centre_xy_m[1])
        z = points[:, 2]
        radius = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)
        strength = _evaluate(self.strength_m_s, z)
        ring_radius = _evaluate(self.ring_radius_m, z)
        thickness = np.maximum(_evaluate(self.ring_thickness_m, z), EPS)
        background = _evaluate(self.background_m_s, z)
        if self.vertical_decay_m is not None:
            decay = np.exp(-np.maximum(z - float(self.reference_z_m), 0.0) / max(float(self.vertical_decay_m), EPS))
            strength = strength * decay
            background = background * decay
        harmonic = np.ones_like(radius)
        for order, coeff in enumerate(self.harmonic_cos, start=1):
            harmonic += float(coeff) * np.cos(order * theta)
        for order, coeff in enumerate(self.harmonic_sin, start=1):
            harmonic += float(coeff) * np.sin(order * theta)
        envelope = np.exp(-((radius - ring_radius) / thickness) ** 2)
        return background + strength * harmonic * envelope


@dataclass(frozen=True)
class AnnularUpdraft:
    """Sum of one or more compact annular fan models."""

    fans: tuple[FanUpdraft, ...]

    @classmethod
    def from_fans(cls, fans: Iterable[FanUpdraft]) -> "AnnularUpdraft":
        fan_tuple = tuple(fans)
        if not fan_tuple:
            raise ValueError("AnnularUpdraft requires at least one fan.")
        return cls(fan_tuple)

    def vertical_speed_at(
        self,
        points_w_m: np.ndarray,
        seed: int | None = None,
        include_perturbation: bool = False,
    ) -> np.ndarray:
        points = np.asarray(points_w_m, dtype=float)
        flat = points.reshape(-1, 3)
        total = np.zeros(flat.shape[0], dtype=float)
        variance = np.zeros_like(total)
        for fan in self.fans:
            total += fan.vertical_speed(flat)
            variance += float(fan.fluctuation_std_m_s) ** 2
        if include_perturbation:
            rng = np.random.default_rng(seed)
            total += rng.normal(0.0, np.sqrt(variance), size=total.shape)
        return total.reshape(points.shape[:-1])

    def uncertainty_at(self, points_w_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_w_m, dtype=float)
        flat_count = points.reshape(-1, 3).shape[0]
        variance = np.zeros(flat_count, dtype=float)
        for fan in self.fans:
            variance += float(fan.fluctuation_std_m_s) ** 2
        return np.sqrt(variance).reshape(points.shape[:-1])

    def velocity_at(
        self,
        points_w_m: np.ndarray,
        seed: int | None = None,
        include_perturbation: bool = False,
    ) -> np.ndarray:
        points = np.asarray(points_w_m, dtype=float)
        vertical = self.vertical_speed_at(points, seed=seed, include_perturbation=include_perturbation)
        velocity = np.zeros_like(points, dtype=float)
        velocity[..., 2] = vertical
        return velocity

    def __call__(self, points_w_m: np.ndarray) -> np.ndarray:
        return self.velocity_at(points_w_m)
