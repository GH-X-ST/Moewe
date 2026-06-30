from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from moewe.baselines import ReferenceTrackingConfig, run_reference_tracking_rollout
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


def test_reference_tracking_baseline_smoke_builds_metric_record() -> None:
    config_path = Path("config/baselines/reference_tracking_smoke.yaml")
    assert config_path.exists()
    config_text = config_path.read_text(encoding="utf-8")
    assert "writes_files_by_default: false" in config_text
    assert "not a final autopilot" in config_text

    record = run_reference_tracking_rollout(
        initial_state=_state(),
        config=ReferenceTrackingConfig(dt_s=0.01, horizon_s=0.05),
        task=_task(),
    ).to_record()

    assert record["state_count"] >= 2
    assert record["command_count"] >= 1
    assert record["gate_success"] in {True, False}
    assert record["gate_crossed"] in {True, False}
    assert np.isfinite(record["gate_miss_distance_m"])
    assert np.isfinite(record["min_safety_margin_m"])
    assert np.isfinite(record["terminal_specific_energy_margin_j_kg"])
    assert np.isfinite(record["max_angle_of_attack_rad"])
    json.dumps(record, sort_keys=True)
