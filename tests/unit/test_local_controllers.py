from __future__ import annotations

import numpy as np
import pytest

from moewe.control import (
    CommandLimits,
    LQRController,
    Linearisation,
    PDController,
    PDGains,
    build_lqr_controller,
    solve_discrete_lqr,
)
from moewe.sim.state import FlightState


def _reference_state() -> FlightState:
    return FlightState(
        position_w_m=np.array([0.0, 0.0, 1.0]),
        euler_rad=np.zeros(3),
        velocity_b_m_s=np.array([7.0, 0.0, 0.0]),
        rates_b_rad_s=np.zeros(3),
        surfaces_rad=np.zeros(3),
    )


def test_discrete_lqr_gain_has_expected_shape_and_sign() -> None:
    a = np.array([[1.0, 0.1], [0.0, 1.0]])
    b = np.array([[0.0], [0.1]])
    q = np.diag([1.0, 0.1])
    r = np.array([[0.2]])

    gain = solve_discrete_lqr(a, b, q, r)

    assert gain.shape == (1, 2)
    assert np.isfinite(gain).all()
    assert gain[0, 0] > 0.0


def test_lqr_controller_saturates_command() -> None:
    reference = _reference_state()
    gain = np.zeros((3, 15))
    gain[0, 3] = 10.0
    controller = LQRController(
        gain=gain,
        reference_state=reference,
        reference_command_rad=np.zeros(3),
        command_limits=CommandLimits(lower_rad=(-0.2, -0.2, -0.2), upper_rad=(0.2, 0.2, 0.2)),
    )
    perturbed = FlightState.from_vector(reference.as_vector() + np.eye(15)[3])

    command = controller.command(0.0, perturbed)

    np.testing.assert_allclose(command, [-0.2, 0.0, 0.0])


def test_build_lqr_controller_from_discrete_linearisation_subset() -> None:
    reference = _reference_state()
    a = np.eye(15)
    a[12, 12] = 0.8
    a[13, 13] = 0.8
    a[14, 14] = 0.8
    b = np.zeros((15, 3))
    b[12, 0] = 0.2
    b[13, 1] = 0.2
    b[14, 2] = 0.2
    lin = Linearisation(
        a=a,
        b=b,
        x_ref=reference.as_vector(),
        u_ref=np.zeros(3),
        f_ref=np.zeros(15),
        mode="discrete_euler",
        dt_s=0.01,
    )

    controller = build_lqr_controller(
        lin,
        q=np.eye(3),
        r=0.1 * np.eye(3),
        reference_state=reference,
        active_state_indices=(12, 13, 14),
    )

    assert controller.gain.shape == (3, 3)
    assert np.isfinite(controller.gain).all()


def test_pd_controller_uses_same_command_limits() -> None:
    reference = _reference_state()
    controller = PDController(
        reference_state=reference,
        reference_command_rad=np.zeros(3),
        gains=PDGains(roll_p=10.0),
        command_limits=CommandLimits(lower_rad=(-0.3, -0.3, -0.3), upper_rad=(0.3, 0.3, 0.3)),
    )
    perturbed = FlightState.from_vector(reference.as_vector() + 0.1 * np.eye(15)[3])

    command = controller.command(0.0, perturbed)

    np.testing.assert_allclose(command, [-0.3, 0.0, 0.0])


def test_lqr_reports_bad_dimensions() -> None:
    with pytest.raises(ValueError, match="shape"):
        LQRController(
            gain=np.zeros((2, 15)),
            reference_state=_reference_state(),
            reference_command_rad=np.zeros(3),
        )
