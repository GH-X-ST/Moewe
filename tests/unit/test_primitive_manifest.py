from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from moewe.primitives import (
    AcceptanceThresholds,
    PrimitiveRolloutConfig,
    PrimitiveValidationManifest,
    ValidationScenario,
    run_validation_sweep,
)


def _report():
    scenario = ValidationScenario(
        scenario_id="manifest_smoke",
        rollout_config=PrimitiveRolloutConfig(dt_s=0.01, max_duration_s=0.05),
        thresholds=AcceptanceThresholds(
            min_safety_margin_m=-10.0,
            max_angle_of_attack_rad=10.0,
            max_command_abs_rad=10.0,
            min_terminal_specific_energy_change_j_kg=-100.0,
        ),
    )
    return run_validation_sweep(scenarios=(scenario,), max_primitives=2)


def test_manifest_export_is_serialisable_without_trajectories() -> None:
    report = _report()
    records = report.to_records()

    assert records
    for record in records:
        assert "states" not in record
        assert "commands_rad" not in record
    json.dumps({"summary": report.to_summary(), "records": records})

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp_dir:
        path = Path(tmp_dir) / "primitive_validation_manifest.json"
        report.write_json(path)
        payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["summary"]["rollout_count"] == len(records)
    assert payload["summary"]["retained_count"] == sum(1 for record in records if record["retained"])
    assert payload["records"][0]["primitive_id"] == records[0]["primitive_id"]


def test_manifest_rejects_raw_arrays() -> None:
    with pytest.raises(TypeError, match="raw trajectory arrays"):
        PrimitiveValidationManifest.from_records(({"states": np.zeros((2, 15))},))


def test_manifest_summary_counts_match_records() -> None:
    report = _report()
    summary = report.to_summary()
    records = report.to_records()

    assert summary["candidate_count"] == 2
    assert summary["scenario_count"] == 1
    assert summary["rollout_count"] == len(records)
    assert summary["retained_count"] == sum(1 for record in records if record["retained"])
    assert summary["entry_class_coverage"] == sorted({record["entry_class"] for record in records})
    assert summary["exit_class_coverage"] == sorted({record["exit_class"] for record in records})
