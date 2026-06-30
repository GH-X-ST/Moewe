"""State representation for the Moewe glider simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frames import GRAVITY_M_S2

STATE_SIZE = 15
SURFACE_SIZE = 3


def _vector3(value: np.ndarray | tuple[float, float, float] | list[float]) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(3)


@dataclass(frozen=True)
class FlightState:
    """Canonical 15-state glider state.

    State order is position in world z-up coordinates, Euler attitude, body
    velocity, body rates, then actual surface angles:
    ``[x, y, z, phi, theta, psi, u, v, w, p, q, r, da, de, dr]``.
    Surface order is aileron, elevator, rudder.
    """

    position_w_m: np.ndarray
    euler_rad: np.ndarray
    velocity_b_m_s: np.ndarray
    rates_b_rad_s: np.ndarray
    surfaces_rad: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_w_m", _vector3(self.position_w_m))
        object.__setattr__(self, "euler_rad", _vector3(self.euler_rad))
        object.__setattr__(self, "velocity_b_m_s", _vector3(self.velocity_b_m_s))
        object.__setattr__(self, "rates_b_rad_s", _vector3(self.rates_b_rad_s))
        object.__setattr__(self, "surfaces_rad", _vector3(self.surfaces_rad))

    @classmethod
    def zero(cls) -> "FlightState":
        return cls(
            position_w_m=np.zeros(3),
            euler_rad=np.zeros(3),
            velocity_b_m_s=np.zeros(3),
            rates_b_rad_s=np.zeros(3),
            surfaces_rad=np.zeros(3),
        )

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "FlightState":
        values = np.asarray(vector, dtype=float).reshape(STATE_SIZE)
        return cls(
            position_w_m=values[0:3],
            euler_rad=values[3:6],
            velocity_b_m_s=values[6:9],
            rates_b_rad_s=values[9:12],
            surfaces_rad=values[12:15],
        )

    def as_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                self.position_w_m,
                self.euler_rad,
                self.velocity_b_m_s,
                self.rates_b_rad_s,
                self.surfaces_rad,
            ]
        )

    def with_surfaces(self, surfaces_rad: np.ndarray) -> "FlightState":
        return FlightState(
            position_w_m=self.position_w_m,
            euler_rad=self.euler_rad,
            velocity_b_m_s=self.velocity_b_m_s,
            rates_b_rad_s=self.rates_b_rad_s,
            surfaces_rad=np.asarray(surfaces_rad, dtype=float).reshape(3),
        )

    def finite(self) -> bool:
        return bool(np.all(np.isfinite(self.as_vector())))

    def mechanical_energy_j(self, mass_kg: float, g_m_s2: float = GRAVITY_M_S2) -> float:
        speed2 = float(np.dot(self.velocity_b_m_s, self.velocity_b_m_s))
        return 0.5 * float(mass_kg) * speed2 + float(mass_kg) * float(g_m_s2) * float(
            self.position_w_m[2]
        )
