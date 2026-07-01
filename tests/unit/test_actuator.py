from __future__ import annotations

import numpy as np

from moewe.control.interface import CommandLimits
from moewe.primitives.grammar import PrimitiveSafetyLimits
from moewe.primitives.validate import AcceptanceThresholds
from moewe.sim.actuator import (
    ActuatorConfig,
    ActuatorModel,
    NAUSICAA_ACTUATOR_TIME_CONSTANT_S,
    NAUSICAA_COMMAND_DELAY_S,
    NAUSICAA_MAX_COMMAND_ABS_RAD,
    NAUSICAA_SURFACE_LOWER_LIMITS_RAD,
    NAUSICAA_SURFACE_UPPER_LIMITS_RAD,
)


def test_default_actuator_config_uses_nausicaa_calibration() -> None:
    config = ActuatorConfig()

    np.testing.assert_allclose(config.lower_limits_rad, NAUSICAA_SURFACE_LOWER_LIMITS_RAD)
    np.testing.assert_allclose(config.upper_limits_rad, NAUSICAA_SURFACE_UPPER_LIMITS_RAD)
    np.testing.assert_allclose(config.time_constant_s, NAUSICAA_ACTUATOR_TIME_CONSTANT_S)
    assert config.delay_s == NAUSICAA_COMMAND_DELAY_S


def test_default_command_and_validation_limits_match_actuator_envelope() -> None:
    command_limits = CommandLimits()

    np.testing.assert_allclose(command_limits.lower, NAUSICAA_SURFACE_LOWER_LIMITS_RAD)
    np.testing.assert_allclose(command_limits.upper, NAUSICAA_SURFACE_UPPER_LIMITS_RAD)
    assert PrimitiveSafetyLimits().max_abs_command_rad == NAUSICAA_MAX_COMMAND_ABS_RAD
    assert AcceptanceThresholds().max_command_abs_rad == NAUSICAA_MAX_COMMAND_ABS_RAD


def test_saturation_and_command_lattice() -> None:
    model = ActuatorModel(
        ActuatorConfig(
            lower_limits_rad=(-0.5, -0.5, -0.5),
            upper_limits_rad=(0.5, 0.5, 0.5),
            time_constant_s=(0.0, 0.0, 0.0),
            delay_s=0.0,
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
            delay_s=0.0,
        )
    )

    next_surface = model.step(np.zeros(3), np.ones(3), 0.5)

    np.testing.assert_allclose(next_surface, np.full(3, 1.0 - np.exp(-1.0)), rtol=1e-12)
