from __future__ import annotations

import numpy as np

from moewe.sim.actuator import ActuatorConfig, ActuatorModel


def test_saturation_and_command_lattice() -> None:
    model = ActuatorModel(
        ActuatorConfig(
            lower_limits_rad=(-0.5, -0.5, -0.5),
            upper_limits_rad=(0.5, 0.5, 0.5),
            time_constant_s=(0.0, 0.0, 0.0),
            lattice_step_rad=0.1,
        )
    )

    next_surface = model.step(np.zeros(3), np.array([0.56, -0.54, 0.04]), 0.1)

    np.testing.assert_allclose(next_surface, [0.5, -0.5, 0.0])


def test_delay_buffer_applies_fixed_delay() -> None:
    model = ActuatorModel(
        ActuatorConfig(
            lower_limits_rad=(-2.0, -2.0, -2.0),
            upper_limits_rad=(2.0, 2.0, 2.0),
            time_constant_s=(0.0, 0.0, 0.0),
            delay_s=0.2,
        )
    )
    surface = np.zeros(3)
    command = np.array([1.0, 0.0, 0.0])

    surface = model.step(surface, command, 0.1)
    np.testing.assert_allclose(surface, [0.0, 0.0, 0.0])
    surface = model.step(surface, command, 0.1)
    np.testing.assert_allclose(surface, [0.0, 0.0, 0.0])
    surface = model.step(surface, command, 0.1)
    np.testing.assert_allclose(surface, [1.0, 0.0, 0.0])


def test_first_order_response_reaches_one_time_constant_fraction() -> None:
    model = ActuatorModel(
        ActuatorConfig(
            lower_limits_rad=(-2.0, -2.0, -2.0),
            upper_limits_rad=(2.0, 2.0, 2.0),
            time_constant_s=(0.5, 0.5, 0.5),
        )
    )

    next_surface = model.step(np.zeros(3), np.ones(3), 0.5)

    np.testing.assert_allclose(next_surface, np.full(3, 1.0 - np.exp(-1.0)), rtol=1e-12)
