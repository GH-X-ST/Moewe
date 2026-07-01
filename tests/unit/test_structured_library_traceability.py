from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from moewe.primitives import (
    build_random_challenge_cases,
    build_structured_library_design_cases,
    build_structured_primitive_library,
)


def test_structured_library_build_records_library_design_traceability() -> None:
    design_cases = build_structured_library_design_cases()[:2]
    report = build_structured_primitive_library(design_cases=design_cases, max_primitives=3)

    assert report.case_count == 2
    assert report.primitive_candidate_count == 3
    assert report.validated_rollout_count == 6

    records = report.to_records()
    assert {record["design_case_id"] for record in records} == {"design_still_air_trim_recovery", "design_still_air_gate_alignment"}
    assert {record["case_set"] for record in records} == {"library_design"}
    assert all(record["primitive_id"] for record in records)
    assert all(record["family"] == "bank_pitch_dwell_recovery" for record in records)
    assert all(record["controller_type"] == "lqr" for record in records)
    assert all(record["entry_class"] for record in records)
    assert all(record["exit_class"] for record in records)

    retained = [record for record in records if record["retained"]]
    assert retained
    assert all(record["entry_class"] and record["exit_class"] for record in retained)

    with TemporaryDirectory(prefix=".tmp_structured_report_", dir=Path.cwd()) as directory:
        output = Path(directory) / "structured_report.json"
        report.write_json(output)
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["metadata"]["case_set"] == "library_design"
        assert "states" not in output.read_text(encoding="utf-8")


def test_structured_library_build_rejects_random_challenge_cases() -> None:
    random_cases = build_random_challenge_cases(controller_frozen=True)

    with pytest.raises(ValueError, match="structured library design cases only"):
        build_structured_primitive_library(design_cases=random_cases, max_primitives=1)


def test_default_structured_library_build_uses_deterministic_library_design_cases_only() -> None:
    report = build_structured_primitive_library(
        design_cases=build_structured_library_design_cases()[:1],
        max_primitives=1,
    )

    assert all(case.case_set == "library_design" for case in report.design_cases)
    assert all(not case.randomized for case in report.design_cases)
    assert all(case.seed is None for case in report.design_cases)
    assert not any(record["design_case_id"].startswith("challenge_") for record in report.to_records())


def test_structured_library_design_cases_validate_complete_primitives_by_default() -> None:
    scenario = build_structured_library_design_cases()[0].to_validation_scenario()

    assert scenario.rollout_config.max_duration_s is None
