"""Piecewise reference utilities for manoeuvre primitives."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from moewe.sim.state import FlightState

from .phases import PhaseSample, PrimitivePhase


@dataclass(frozen=True)
class PrimitiveReference:
    """Piecewise deterministic primitive reference."""

    phases: tuple[PrimitivePhase, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        phases = tuple(self.phases)
        if not phases:
            raise ValueError("PrimitiveReference requires at least one phase.")
        for phase in phases:
            if not np.isfinite(float(phase.duration_s)) or float(phase.duration_s) <= 0.0:
                raise ValueError("Reference phase durations must be positive and finite.")
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def total_duration_s(self) -> float:
        return float(sum(float(phase.duration_s) for phase in self.phases))

    def sample_times(self, dt_s: float) -> np.ndarray:
        """Return sample times from zero through the final reference time."""

        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("sample dt_s must be positive and finite.")
        total = self.total_duration_s
        count = int(np.floor(total / dt))
        values = [float(index) * dt for index in range(count + 1)]
        values = [value for value in values if value <= total + 1e-12]
        if not values or abs(values[-1] - total) > 1e-12:
            values.append(total)
        else:
            values[-1] = total
        return np.asarray(values, dtype=float)

    def sample(self, time_s: float) -> PhaseSample:
        """Return the phase sample at primitive-local time."""

        t = float(np.clip(float(time_s), 0.0, self.total_duration_s))
        elapsed = 0.0
        for phase in self.phases:
            end = elapsed + float(phase.duration_s)
            if t <= end or phase is self.phases[-1]:
                return phase.sample(t - elapsed)
            elapsed = end
        return self.phases[-1].sample(self.phases[-1].duration_s)

    def state_at(self, time_s: float) -> FlightState:
        """Return the reference state at primitive-local time."""

        return self.sample(time_s).state

    def command_at(self, time_s: float) -> np.ndarray:
        """Return the reference command at primitive-local time.

        Command order is ``[aileron, elevator, rudder]`` in radians.
        """

        return self.sample(time_s).command_rad
