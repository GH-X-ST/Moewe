"""Transparent PD-style local stabiliser for inner-loop ablations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moewe.sim.frames import body_to_world
from moewe.sim.state import FlightState

from .interface import CommandLimits, ControllerMetadata


@dataclass(frozen=True)
class PDGains:
    """Transparent gains mapping local errors to surface commands."""

    roll_p: float = 0.8
    roll_d: float = 0.15
    pitch_p: float = 0.8
    pitch_d: float = 0.18
    yaw_p: float = 0.2
    yaw_d: float = 0.08
    speed_p: float = 0.03
    altitude_p: float = 0.0
    vertical_speed_p: float = 0.0


@dataclass(frozen=True)
class PDController:
    """Local PD controller sharing the same command interface as LQR."""

    reference_state: FlightState
    reference_command_rad: np.ndarray
    gains: PDGains = PDGains()
    command_limits: CommandLimits = CommandLimits()
    metadata: ControllerMetadata = ControllerMetadata(
        controller_type="pd",
        description="transparent local PD stabiliser for ablation studies",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference_command_rad", np.asarray(self.reference_command_rad, dtype=float).reshape(3))

    def command(self, time_s: float, state: FlightState) -> np.ndarray:
        del time_s
        attitude_error = state.euler_rad - self.reference_state.euler_rad
        rate_error = state.rates_b_rad_s - self.reference_state.rates_b_rad_s
        speed = float(np.linalg.norm(state.velocity_b_m_s))
        reference_speed = float(np.linalg.norm(self.reference_state.velocity_b_m_s))
        speed_error = speed - reference_speed
        vertical_speed = float(body_to_world(state.velocity_b_m_s, state.euler_rad)[2])
        reference_vertical_speed = float(
            body_to_world(self.reference_state.velocity_b_m_s, self.reference_state.euler_rad)[2]
        )
        vertical_speed_error = vertical_speed - reference_vertical_speed
        altitude_error = float(state.position_w_m[2] - self.reference_state.position_w_m[2])

        command = self.reference_command_rad.copy()
        command[0] += -self.gains.roll_p * attitude_error[0] - self.gains.roll_d * rate_error[0]
        command[1] += (
            -self.gains.pitch_p * attitude_error[1]
            - self.gains.pitch_d * rate_error[1]
            - self.gains.speed_p * speed_error
            + self.gains.altitude_p * altitude_error
            + self.gains.vertical_speed_p * vertical_speed_error
        )
        command[2] += -self.gains.yaw_p * attitude_error[2] - self.gains.yaw_d * rate_error[2]
        return self.command_limits.clip(command)
