from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from moewe.baselines import (
    ReferenceTrackingConfig,
    ReferenceTrackingController,
    build_gate_tracking_target,
    run_reference_tracking_rollout,
)
from moewe.baselines import reference_tracking
from moewe.control import CommandLimits
from moewe.sim.state import FlightState
from moewe.tasks import FlightVolume, GatePlane, GateTraversalTask


def _state() -> FlightState:
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
            centre_w_m=np.array([2.0, 0.0, 1.2]),
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


@pytest.mark.parametrize(
    "kwargs",
    (
        {"dt_s": 0.0},
        {"horizon_s": -0.1},
        {"heading_to_roll_gain": float("inf")},
        {"heading_to_roll_gain": -0.1},
        {"altitude_to_pitch_gain": 0.0},
    ),
)
def test_invalid_reference_tracking_config_fails(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ReferenceTrackingConfig(**kwargs)


def test_reference_tracking_controller_returns_finite_clipped_commands() -> None:
    limits = CommandLimits(lower_rad=(-0.05, -0.05, -0.05), upper_rad=(0.05, 0.05, 0.05))
    config = ReferenceTrackingConfig(
        heading_to_roll_gain=20.0,
        altitude_to_pitch_gain=20.0,
        command_limits=limits,
    )
    controller = ReferenceTrackingController(
        target_position_w_m=np.array([1.0, 10.0, 4.0]),
        config=config,
    )

    command = controller.command(0.0, _state())
    reference = controller.reference_for_state(_state())

    assert np.isfinite(command).all()
    assert np.all(command <= limits.upper + 1e-12)
    assert np.all(command >= limits.lower - 1e-12)
    assert abs(reference.euler_rad[0]) <= config.max_reference_roll_rad
    assert abs(reference.euler_rad[1]) <= config.max_reference_pitch_rad


def test_gate_tracking_target_uses_gate_centre_or_explicit_target() -> None:
    task = _task()
    explicit_target = np.array([1.0, -0.2, 1.4])

    assert np.allclose(build_gate_tracking_target(task=task), task.gate.centre_w_m)
    assert np.allclose(build_gate_tracking_target(target_position_w_m=explicit_target), explicit_target)
    with pytest.raises(ValueError):
        build_gate_tracking_target()


def test_reference_tracking_rollout_records_are_deterministic_and_json_serialisable() -> None:
    config = ReferenceTrackingConfig(dt_s=0.02, horizon_s=0.06)
    target = np.array([1.0, 0.2, 1.1])

    first = run_reference_tracking_rollout(_state(), config=config, target_position_w_m=target).to_record()
    second = run_reference_tracking_rollout(_state(), config=config, target_position_w_m=target).to_record()

    assert first == second
    assert first["baseline_name"] == "reference_tracking_pd"
    assert first["state_count"] == 4
    assert first["command_count"] == 3
    assert first["controller_failed"] is False
    json.dumps(first, sort_keys=True)


def test_reference_tracking_baseline_does_not_require_selector_evidence_objects() -> None:
    source = inspect.getsource(reference_tracking)

    assert "moewe.primitives" not in source
    assert "moewe.returnability" not in source
    assert "moewe.governor" not in source

    record = run_reference_tracking_rollout(
        _state(),
        config=ReferenceTrackingConfig(dt_s=0.02, horizon_s=0.04),
        target_position_w_m=np.array([1.0, 0.0, 1.0]),
    ).to_record()

    assert record["gate_success"] is None
    assert record["gate_crossed"] is None
