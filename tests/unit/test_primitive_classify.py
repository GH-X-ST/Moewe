from __future__ import annotations

import numpy as np

from moewe.primitives import PrimitiveStateClassifier
from moewe.sim.state import FlightState


def _state(
    position: tuple[float, float, float] = (0.0, 0.0, 1.0),
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0),
    velocity: tuple[float, float, float] = (7.0, 0.0, 0.0),
) -> FlightState:
    return FlightState(
        position_w_m=np.array(position),
        euler_rad=np.array(euler),
        velocity_b_m_s=np.array(velocity),
        rates_b_rad_s=np.zeros(3),
        surfaces_rad=np.zeros(3),
    )


def test_state_classifier_outputs_stable_label_for_known_state() -> None:
    label = PrimitiveStateClassifier().classify(_state())

    assert label.valid
    assert label.label == (
        "nominal_speed|nominal_altitude|level_bank|level_pitch|centred_lateral|nominal_energy"
    )
    assert label.to_record()["label"] == label.label


def test_state_classifier_bins_bank_pitch_and_lateral_position() -> None:
    label = PrimitiveStateClassifier().classify(
        _state(position=(0.0, 0.6, 2.0), euler=(0.4, -0.2, 0.0), velocity=(10.0, 0.0, 0.0))
    )

    assert label.airspeed_bin == "fast"
    assert label.altitude_bin == "high_altitude"
    assert label.bank_bin == "moderate_bank"
    assert label.pitch_bin == "moderate_pitch"
    assert label.lateral_bin == "offset_lateral"


def test_state_classifier_reports_non_finite_state_explicitly() -> None:
    state = _state()
    state.position_w_m[0] = np.nan

    label = PrimitiveStateClassifier().classify(state)

    assert not label.valid
    assert label.label == "invalid:non_finite_state"
    assert label.failure_reason == "non_finite_state"
