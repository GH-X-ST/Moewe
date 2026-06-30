"""Structured primitive-library build reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json

from moewe.sim.actuator import ActuatorModel
from moewe.sim.glider_model import GliderModel

from .classify import PrimitiveStateClassifier
from .grammar import PrimitiveGrammarSpec
from .library import LIBRARY_DESIGN_CASE_SET, StructuredDesignCase, build_structured_library_design_cases, dense_smoke_grammar
from .manifest import PrimitiveValidationManifest
from .sweep import PrimitiveValidationReport, run_validation_sweep


def _count(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _write_csv(path: str | Path, records: list[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for record in records for key in record})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for record in records:
            row = {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in record.items()
            }
            writer.writerow(row)


def _validate_library_design_cases(cases: tuple[StructuredDesignCase, ...]) -> None:
    if not cases:
        raise ValueError("At least one structured design case is required.")
    for case in cases:
        if case.case_set != LIBRARY_DESIGN_CASE_SET:
            raise ValueError("Structured library builds must use structured library design cases only.")
        if case.randomized or case.seed is not None:
            raise ValueError("Structured library builds must not use randomized cases.")
        if not case.used_for_library_construction:
            raise ValueError("Structured library design cases must be marked for library construction.")


@dataclass(frozen=True)
class StructuredLibraryBuildReport:
    """In-memory report for deterministic primitive-library construction."""

    validation_report: PrimitiveValidationReport
    design_cases: tuple[StructuredDesignCase, ...]
    source: str = "structured_library_design"

    def __post_init__(self) -> None:
        _validate_library_design_cases(self.design_cases)
        case_ids = {case.case_id for case in self.design_cases}
        missing = {result.scenario_id for result in self.validation_report.results} - case_ids
        if missing:
            raise ValueError(f"Validation results are not traced to structured library design cases: {sorted(missing)}")

    @property
    def case_count(self) -> int:
        return len(self.design_cases)

    @property
    def primitive_candidate_count(self) -> int:
        return self.validation_report.candidate_count

    @property
    def validated_rollout_count(self) -> int:
        return self.validation_report.rollout_count

    @property
    def retained_count(self) -> int:
        return self.validation_report.retained_count

    def to_records(self) -> list[dict[str, object]]:
        case_by_id = {case.case_id: case for case in self.design_cases}
        records: list[dict[str, object]] = []
        for result in self.validation_report.results:
            case = case_by_id[result.scenario_id]
            record = result.to_record()
            record.update(
                {
                    "design_case_id": case.case_id,
                    "case_set": case.case_set,
                    "source": self.source,
                }
            )
            records.append(record)
        return records

    def to_manifest(self) -> PrimitiveValidationManifest:
        return PrimitiveValidationManifest.from_records(
            self.to_records(),
            metadata={
                "source": self.source,
                "case_set": LIBRARY_DESIGN_CASE_SET,
                "case_count": self.case_count,
                "candidate_count": self.primitive_candidate_count,
                "scenario_count": self.validation_report.scenario_count,
            },
        )

    def to_summary(self) -> dict[str, object]:
        records = self.to_records()
        retained = [record for record in records if bool(record["retained"])]
        return {
            "source": self.source,
            "case_set": LIBRARY_DESIGN_CASE_SET,
            "case_count": self.case_count,
            "primitive_candidate_count": self.primitive_candidate_count,
            "validated_rollout_count": self.validated_rollout_count,
            "retained_count": self.retained_count,
            "entry_class_coverage": sorted({str(record["entry_class"]) for record in retained}),
            "exit_class_coverage": sorted({str(record["exit_class"]) for record in retained}),
            "family_coverage": sorted({f"{record['family']}:{record['controller_type']}" for record in records}),
            "design_case_coverage": _count([str(record["design_case_id"]) for record in records]),
            "retention_reason_counts": _count([str(record["retention_reason"]) for record in records]),
        }

    def write_json(self, path: str | Path) -> None:
        self.to_manifest().write_json(path)

    def write_csv(self, path: str | Path) -> None:
        _write_csv(path, self.to_records())


def build_structured_primitive_library(
    grammar_spec: PrimitiveGrammarSpec | None = None,
    design_cases: tuple[StructuredDesignCase, ...] | list[StructuredDesignCase] | None = None,
    classifier: PrimitiveStateClassifier | None = None,
    model: GliderModel | None = None,
    actuator: ActuatorModel | None = None,
    max_primitives: int | None = None,
) -> StructuredLibraryBuildReport:
    """Build a deterministic structured primitive-library evidence report."""

    cases = tuple(build_structured_library_design_cases() if design_cases is None else design_cases)
    _validate_library_design_cases(cases)
    scenarios = tuple(case.to_validation_scenario() for case in cases)
    grammar = dense_smoke_grammar() if grammar_spec is None else grammar_spec
    validation_report = run_validation_sweep(
        grammar_spec=grammar,
        scenarios=scenarios,
        classifier=classifier,
        model=model,
        actuator=actuator,
        max_primitives=max_primitives,
    )
    return StructuredLibraryBuildReport(validation_report=validation_report, design_cases=cases)
