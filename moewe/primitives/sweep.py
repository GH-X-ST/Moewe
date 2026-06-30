"""Lightweight primitive validation sweep execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from moewe.sim.actuator import ActuatorModel
from moewe.sim.glider_model import GliderModel

from .classify import PrimitiveStateClassifier
from .generate import generate_primitives
from .grammar import PrimitiveGrammarSpec
from .manifest import PrimitiveValidationManifest
from .validate import PrimitiveValidationResult, ValidationScenario, validate_primitive


@dataclass(frozen=True)
class PrimitiveValidationReport:
    """In-memory primitive validation sweep report."""

    results: tuple[PrimitiveValidationResult, ...]
    candidate_count: int
    scenario_count: int

    @property
    def rollout_count(self) -> int:
        return len(self.results)

    @property
    def retained_count(self) -> int:
        return sum(1 for result in self.results if result.retention.retained)

    def to_records(self) -> list[dict[str, object]]:
        return [result.to_record() for result in self.results]

    def to_manifest(self) -> PrimitiveValidationManifest:
        return PrimitiveValidationManifest.from_records(
            self.to_records(),
            metadata={
                "candidate_count": self.candidate_count,
                "scenario_count": self.scenario_count,
            },
        )

    def to_summary(self) -> dict[str, object]:
        return self.to_manifest().to_summary()

    def write_json(self, path: str | Path) -> None:
        self.to_manifest().write_json(path)

    def write_csv(self, path: str | Path) -> None:
        self.to_manifest().write_csv(path)


def run_validation_sweep(
    grammar_spec: PrimitiveGrammarSpec | None = None,
    scenarios: tuple[ValidationScenario, ...] | list[ValidationScenario] | None = None,
    classifier: PrimitiveStateClassifier | None = None,
    model: GliderModel | None = None,
    actuator: ActuatorModel | None = None,
    wind_model: object | None = None,
    max_primitives: int | None = None,
) -> PrimitiveValidationReport:
    """Generate primitives and validate them against one or more smoke scenarios."""

    scenario_list = tuple(scenarios or (ValidationScenario(scenario_id="validation_smoke"),))
    if not scenario_list:
        raise ValueError("At least one validation scenario is required.")
    for scenario in scenario_list:
        scenario.validate()
    primitives = generate_primitives(grammar_spec, model=model, wind_model=wind_model)
    if max_primitives is not None:
        if max_primitives <= 0:
            raise ValueError("max_primitives must be positive when set.")
        primitives = primitives[: int(max_primitives)]
    state_classifier = PrimitiveStateClassifier() if classifier is None else classifier
    results: list[PrimitiveValidationResult] = []
    for primitive in primitives:
        for scenario in scenario_list:
            results.append(
                validate_primitive(
                    primitive=primitive,
                    scenario=scenario,
                    classifier=state_classifier,
                    model=model,
                    actuator=actuator,
                    wind_model=wind_model,
                )
            )
    return PrimitiveValidationReport(
        results=tuple(results),
        candidate_count=len(primitives),
        scenario_count=len(scenario_list),
    )
