"""Deterministic entry and exit state classification for primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moewe.sim.state import FlightState
from moewe.tasks.metrics import specific_energy_j_kg


def _bin_abs(value: float, low_limit: float, high_limit: float, labels: tuple[str, str, str]) -> str:
    magnitude = abs(float(value))
    if magnitude <= float(low_limit):
        return labels[0]
    if magnitude <= float(high_limit):
        return labels[1]
    return labels[2]


def _bin_signed(value: float, low_limit: float, high_limit: float, labels: tuple[str, str, str]) -> str:
    scalar = float(value)
    if scalar < float(low_limit):
        return labels[0]
    if scalar <= float(high_limit):
        return labels[1]
    return labels[2]


@dataclass(frozen=True)
class StateClassLabel:
    """Stable serialisable label for a primitive entry or exit state."""

    label: str
    airspeed_bin: str
    altitude_bin: str
    bank_bin: str
    pitch_bin: str
    lateral_bin: str
    energy_bin: str
    valid: bool = True
    failure_reason: str | None = None

    def to_record(self) -> dict[str, object]:
        return {
            "label": self.label,
            "airspeed_bin": self.airspeed_bin,
            "altitude_bin": self.altitude_bin,
            "bank_bin": self.bank_bin,
            "pitch_bin": self.pitch_bin,
            "lateral_bin": self.lateral_bin,
            "energy_bin": self.energy_bin,
            "valid": self.valid,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class StateClassifierSpec:
    """Binning thresholds for primitive entry and exit classification."""

    low_airspeed_m_s: float = 5.0
    high_airspeed_m_s: float = 9.0
    low_altitude_m: float = 0.7
    high_altitude_m: float = 1.5
    level_bank_rad: float = 0.15
    moderate_bank_rad: float = 0.5
    level_pitch_rad: float = 0.08
    moderate_pitch_rad: float = 0.3
    centred_lateral_m: float = 0.25
    offset_lateral_m: float = 1.0
    low_specific_energy_j_kg: float = 25.0
    high_specific_energy_j_kg: float = 80.0

    def validate(self) -> None:
        ordered_pairs = (
            (self.low_airspeed_m_s, self.high_airspeed_m_s, "airspeed"),
            (self.low_altitude_m, self.high_altitude_m, "altitude"),
            (self.level_bank_rad, self.moderate_bank_rad, "bank"),
            (self.level_pitch_rad, self.moderate_pitch_rad, "pitch"),
            (self.centred_lateral_m, self.offset_lateral_m, "lateral"),
            (self.low_specific_energy_j_kg, self.high_specific_energy_j_kg, "specific energy"),
        )
        for lower, upper, name in ordered_pairs:
            if not np.isfinite([lower, upper]).all():
                raise ValueError(f"{name} classifier thresholds must be finite.")
            if float(lower) < 0.0 or float(lower) > float(upper):
                raise ValueError(f"{name} classifier thresholds must be non-negative and ordered.")


@dataclass(frozen=True)
class PrimitiveStateClassifier:
    """Small inspectable classifier for primitive validation smoke records."""

    spec: StateClassifierSpec = StateClassifierSpec()

    def classify(self, state: FlightState) -> StateClassLabel:
        self.spec.validate()
        if not state.finite():
            return StateClassLabel(
                label="invalid:non_finite_state",
                airspeed_bin="invalid",
                altitude_bin="invalid",
                bank_bin="invalid",
                pitch_bin="invalid",
                lateral_bin="invalid",
                energy_bin="invalid",
                valid=False,
                failure_reason="non_finite_state",
            )
        speed = float(np.linalg.norm(state.velocity_b_m_s))
        airspeed = _bin_signed(
            speed,
            self.spec.low_airspeed_m_s,
            self.spec.high_airspeed_m_s,
            ("slow", "nominal_speed", "fast"),
        )
        altitude = _bin_signed(
            float(state.position_w_m[2]),
            self.spec.low_altitude_m,
            self.spec.high_altitude_m,
            ("low_altitude", "nominal_altitude", "high_altitude"),
        )
        bank = _bin_abs(
            float(state.euler_rad[0]),
            self.spec.level_bank_rad,
            self.spec.moderate_bank_rad,
            ("level_bank", "moderate_bank", "steep_bank"),
        )
        pitch = _bin_abs(
            float(state.euler_rad[1]),
            self.spec.level_pitch_rad,
            self.spec.moderate_pitch_rad,
            ("level_pitch", "moderate_pitch", "steep_pitch"),
        )
        lateral = _bin_abs(
            float(state.position_w_m[1]),
            self.spec.centred_lateral_m,
            self.spec.offset_lateral_m,
            ("centred_lateral", "offset_lateral", "wide_lateral"),
        )
        energy = _bin_signed(
            specific_energy_j_kg(state),
            self.spec.low_specific_energy_j_kg,
            self.spec.high_specific_energy_j_kg,
            ("low_energy", "nominal_energy", "high_energy"),
        )
        label = "|".join((airspeed, altitude, bank, pitch, lateral, energy))
        return StateClassLabel(
            label=label,
            airspeed_bin=airspeed,
            altitude_bin=altitude,
            bank_bin=bank,
            pitch_bin=pitch,
            lateral_bin=lateral,
            energy_bin=energy,
        )
