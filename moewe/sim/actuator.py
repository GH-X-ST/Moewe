"""Configurable deterministic surface actuator model."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np

NAUSICAA_SURFACE_LOWER_LIMITS_RAD = (
    -0.32742376767413617,
    -0.3909537524467298,
    -0.4031710572106901,
)
NAUSICAA_SURFACE_UPPER_LIMITS_RAD = (
    0.32742376767413617,
    0.3909537524467298,
    0.4031710572106901,
)
NAUSICAA_MAX_COMMAND_ABS_RAD = max(NAUSICAA_SURFACE_UPPER_LIMITS_RAD)
NAUSICAA_ACTUATOR_TIME_CONSTANT_S = (
    0.1116139610371975,
    0.110121589022133,
    0.10122971500671,
)
NAUSICAA_COMMAND_DELAY_S = 0.02


def _surface_vector(value: np.ndarray | tuple[float, float, float]) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(3)


@dataclass(frozen=True)
class ActuatorConfig:
    """Actuator limits, optional lattice, fixed delay, and first-order response."""

    lower_limits_rad: tuple[float, float, float] = NAUSICAA_SURFACE_LOWER_LIMITS_RAD
    upper_limits_rad: tuple[float, float, float] = NAUSICAA_SURFACE_UPPER_LIMITS_RAD
    time_constant_s: tuple[float, float, float] = NAUSICAA_ACTUATOR_TIME_CONSTANT_S
    delay_s: float = NAUSICAA_COMMAND_DELAY_S
    lattice_step_rad: float | None = None

    def validate(self) -> None:
        lower = _surface_vector(self.lower_limits_rad)
        upper = _surface_vector(self.upper_limits_rad)
        tau = _surface_vector(self.time_constant_s)
        if np.any(lower > upper):
            raise ValueError("Actuator lower limits must not exceed upper limits.")
        if np.any(tau < 0.0):
            raise ValueError("Actuator time constants must be non-negative.")
        if self.delay_s < 0.0:
            raise ValueError("Actuator delay must be non-negative.")
        if self.lattice_step_rad is not None and self.lattice_step_rad <= 0.0:
            raise ValueError("Actuator lattice step must be positive when enabled.")


class ActuatorModel:
    """Stateful delay buffer plus pure first-order surface response."""

    def __init__(
        self,
        config: ActuatorConfig | None = None,
        initial_command_rad: np.ndarray | tuple[float, float, float] | None = None,
    ) -> None:
        self.config = ActuatorConfig() if config is None else config
        self.config.validate()
        initial = np.zeros(3) if initial_command_rad is None else _surface_vector(initial_command_rad)
        self._initial_command = self.prepare_command(initial)
        self._queue: list[np.ndarray] = []
        self._queue_dt_s: float | None = None

    def reset(self, initial_command_rad: np.ndarray | tuple[float, float, float] | None = None) -> None:
        initial = self._initial_command if initial_command_rad is None else _surface_vector(initial_command_rad)
        self._initial_command = self.prepare_command(initial)
        self._queue.clear()
        self._queue_dt_s = None

    def prepare_command(self, command_rad: np.ndarray | tuple[float, float, float]) -> np.ndarray:
        command = _surface_vector(command_rad)
        if self.config.lattice_step_rad is not None:
            step = float(self.config.lattice_step_rad)
            command = np.round(command / step) * step
        return np.clip(
            command,
            _surface_vector(self.config.lower_limits_rad),
            _surface_vector(self.config.upper_limits_rad),
        )

    def _delay_steps(self, dt_s: float) -> int:
        if dt_s <= 0.0:
            raise ValueError("Actuator time step must be positive.")
        return int(ceil(float(self.config.delay_s) / float(dt_s)))

    def _delayed_command(self, command_rad: np.ndarray, dt_s: float) -> np.ndarray:
        steps = self._delay_steps(dt_s)
        command = self.prepare_command(command_rad)
        if steps == 0:
            return command

        if self._queue_dt_s != float(dt_s) or len(self._queue) != steps:
            self._queue = [self._initial_command.copy() for _ in range(steps)]
            self._queue_dt_s = float(dt_s)

        self._queue.append(command)
        return self._queue.pop(0)

    def step(
        self,
        surface_rad: np.ndarray | tuple[float, float, float],
        command_rad: np.ndarray | tuple[float, float, float],
        dt_s: float,
    ) -> np.ndarray:
        """Advance actual surfaces by one deterministic actuator step."""

        surface = _surface_vector(surface_rad)
        target = self._delayed_command(_surface_vector(command_rad), float(dt_s))
        tau = _surface_vector(self.config.time_constant_s)
        alpha = np.ones(3)
        dynamic = tau > 0.0
        alpha[dynamic] = 1.0 - np.exp(-float(dt_s) / tau[dynamic])
        next_surface = surface + alpha * (target - surface)
        return np.clip(
            next_surface,
            _surface_vector(self.config.lower_limits_rad),
            _surface_vector(self.config.upper_limits_rad),
        )
