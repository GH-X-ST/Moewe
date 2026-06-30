"""Scenario dataclasses for deterministic and seeded task rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moewe.sim.state import FlightState


def _vec3(value: np.ndarray | tuple[float, float, float] | list[float]) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(3)


@dataclass(frozen=True)
class FlightVolume:
    """Axis-aligned flight volume in world z-up coordinates."""

    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    z_min_m: float
    z_max_m: float

    def validate(self) -> None:
        if self.x_min_m >= self.x_max_m:
            raise ValueError("x_min_m must be smaller than x_max_m.")
        if self.y_min_m >= self.y_max_m:
            raise ValueError("y_min_m must be smaller than y_max_m.")
        if self.z_min_m >= self.z_max_m:
            raise ValueError("z_min_m must be smaller than z_max_m.")

    def margins_m(self, position_w_m: np.ndarray) -> np.ndarray:
        self.validate()
        x, y, z = _vec3(position_w_m)
        return np.array(
            [
                x - self.x_min_m,
                self.x_max_m - x,
                y - self.y_min_m,
                self.y_max_m - y,
                z - self.z_min_m,
                self.z_max_m - z,
            ],
            dtype=float,
        )

    def min_margin_m(self, position_w_m: np.ndarray) -> float:
        return float(np.min(self.margins_m(position_w_m)))

    def failure_reason(self, position_w_m: np.ndarray) -> str | None:
        x, y, z = _vec3(position_w_m)
        if z < self.z_min_m:
            return "floor"
        if z > self.z_max_m:
            return "ceiling"
        if x < self.x_min_m or x > self.x_max_m or y < self.y_min_m or y > self.y_max_m:
            return "wall"
        return None


@dataclass(frozen=True)
class FixedInitialState:
    """Deterministic initial-state sampler."""

    state: FlightState

    def sample(self, seed: int | None = None) -> FlightState:
        del seed
        return self.state


@dataclass(frozen=True)
class UniformInitialStateSampler:
    """Seeded uniform sampler around a nominal state.

    Half-width arrays are ordered as the canonical 15-state vector.
    """

    nominal_state: FlightState
    half_width: np.ndarray

    def __post_init__(self) -> None:
        half_width = np.asarray(self.half_width, dtype=float).reshape(15)
        if np.any(half_width < 0.0):
            raise ValueError("Initial-state half widths must be non-negative.")
        object.__setattr__(self, "half_width", half_width)

    def sample(self, seed: int | None = None) -> FlightState:
        rng = np.random.default_rng(seed)
        delta = rng.uniform(-self.half_width, self.half_width)
        return FlightState.from_vector(self.nominal_state.as_vector() + delta)


@dataclass(frozen=True)
class Scenario:
    """Small container binding a task, initial state, and optional wind model."""

    name: str
    initial_state_sampler: FixedInitialState | UniformInitialStateSampler
    wind_model: object | None = None
    wind_mode: str = "panel"
    seed: int | None = None

    def initial_state(self) -> FlightState:
        return self.initial_state_sampler.sample(self.seed)
