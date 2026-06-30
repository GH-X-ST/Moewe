from __future__ import annotations

import numpy as np

from moewe.control import CommandLimits, PDController, PDGains, run_closed_loop
from moewe.sim.actuator import ActuatorConfig, ActuatorModel
from moewe.sim.glider_model import nominal_glider
from moewe.sim.integrator import IntegratorConfig
from moewe.sim.state import FlightState
from moewe.tasks import FlightVolume, GatePlane, GateTraversalTask


def _reference_state() -> FlightState:
    return FlightState(
        position_w_m=np.array([0.0, 0.0, 1.0]),
        euler_rad=np.zeros(3),
        velocity_b_m_s=np.array([7.0, 0.0, 0.0]),
        rates_b_rad_s=np.zeros(3),
        surfaces_rad=np.zeros(3),
    )


def _task() -> GateTraversalTask:
    return GateTraversalTask(
        gate=GatePlane(
            centre_w_m=np.array([2.0, 0.0, 1.0]),
            normal_w=np.array([1.0, 0.0, 0.0]),
            width_m=1.0,
            height_m=0.8,
        ),
        flight_volume=FlightVolume(
            x_min_m=-1.0,
            x_max_m=4.0,
            y_min_m=-1.0,
            y_max_m=1.0,
            z_min_m=0.1,
            z_max_m=3.0,
        ),
        timeout_s=1.0,
        angle_of_attack_limit_rad=0.8,
    )


def test_closed_loop_rollout_returns_history_commands_and_metrics() -> None:
    reference = _reference_state()
    controller = PDController(
        reference_state=reference,
        reference_command_rad=np.zeros(3),
        gains=PDGains(),
        command_limits=CommandLimits(lower_rad=(-0.4, -0.4, -0.4), upper_rad=(0.4, 0.4, 0.4)),
    )

    result = run_closed_loop(
        initial_state=reference,
        controller=controller,
        steps=20,
        task=_task(),
        model=nominal_glider(),
        actuator=ActuatorModel(ActuatorConfig()),
        config=IntegratorConfig(dt_s=0.01),
    )

    assert not result.controller_failed
    assert len(result.states) == 21
    assert result.commands_rad.shape == (20, 3)
    assert result.metrics is not None
    assert np.isfinite(result.commands_rad).all()


def test_closed_loop_rollout_records_non_finite_command_failure() -> None:
    class BadController:
        reference_state = _reference_state()
        reference_command_rad = np.zeros(3)
        command_limits = CommandLimits()
        metadata = None

        def command(self, time_s: float, state: FlightState) -> np.ndarray:
            del time_s, state
            return np.array([np.nan, 0.0, 0.0])

    result = run_closed_loop(
        initial_state=_reference_state(),
        controller=BadController(),
        steps=5,
        task=_task(),
        model=nominal_glider(),
        config=IntegratorConfig(dt_s=0.01),
    )

    assert result.controller_failed
    assert result.controller_failure_reason == "non_finite_command"
    assert result.commands_rad.shape == (0, 3)
