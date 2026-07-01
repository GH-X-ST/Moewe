"""Primitive validation scenarios, retention rules, and records."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256

import numpy as np

from moewe.sim.actuator import ActuatorModel, NAUSICAA_MAX_COMMAND_ABS_RAD
from moewe.sim.glider_model import GliderModel, NAUSICAA_OPERATIONAL_ALPHA_LIMIT_RAD
from moewe.sim.state import FlightState, STATE_SIZE
from moewe.tasks.gate import GateTraversalTask

from .classify import PrimitiveStateClassifier, StateClassLabel
from .evidence import PrimitiveEvidence
from .generate import PrimitiveCandidate
from .rollout import PrimitiveRolloutConfig, PrimitiveRolloutResult, rollout_primitive


def _vector15(value: tuple[float, ...] | list[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(STATE_SIZE)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _stable_seed(base_seed: int | None, scenario_id: str, primitive_id: str) -> int | None:
    if base_seed is None:
        return None
    payload = f"{int(base_seed)}|{scenario_id}|{primitive_id}".encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:4], byteorder="big", signed=False)


@dataclass(frozen=True)
class EntryPerturbationSpec:
    """Deterministic offset and seeded uniform perturbation in canonical state order."""

    offset: tuple[float, ...] = (0.0,) * STATE_SIZE
    half_width: tuple[float, ...] = (0.0,) * STATE_SIZE

    def validate(self) -> None:
        _vector15(self.offset, "entry offset")
        half_width = _vector15(self.half_width, "entry half_width")
        if np.any(half_width < 0.0):
            raise ValueError("entry half_width values must be non-negative.")

    def sample(self, nominal_state: FlightState, seed: int | None = None) -> FlightState:
        self.validate()
        offset = _vector15(self.offset, "entry offset")
        half_width = _vector15(self.half_width, "entry half_width")
        if np.any(half_width > 0.0):
            if seed is None:
                raise ValueError("Seeded entry perturbations require an explicit seed.")
            rng = np.random.default_rng(int(seed))
            delta = rng.uniform(-half_width, half_width)
        else:
            delta = np.zeros(STATE_SIZE)
        return FlightState.from_vector(nominal_state.as_vector() + offset + delta)


@dataclass(frozen=True)
class AcceptanceThresholds:
    """Transparent primitive retention thresholds for smoke validation."""

    min_safety_margin_m: float = 0.0
    max_angle_of_attack_rad: float = NAUSICAA_OPERATIONAL_ALPHA_LIMIT_RAD
    max_command_abs_rad: float = NAUSICAA_MAX_COMMAND_ABS_RAD
    min_terminal_specific_energy_change_j_kg: float = -10.0
    min_terminal_specific_energy_margin_j_kg: float | None = None

    def validate(self) -> None:
        finite_values = (
            self.min_safety_margin_m,
            self.max_angle_of_attack_rad,
            self.max_command_abs_rad,
            self.min_terminal_specific_energy_change_j_kg,
        )
        if not np.isfinite(finite_values).all():
            raise ValueError("Acceptance thresholds must be finite.")
        if self.max_angle_of_attack_rad <= 0.0:
            raise ValueError("max_angle_of_attack_rad must be positive.")
        if self.max_command_abs_rad <= 0.0:
            raise ValueError("max_command_abs_rad must be positive.")
        if self.min_terminal_specific_energy_margin_j_kg is not None and not np.isfinite(
            self.min_terminal_specific_energy_margin_j_kg
        ):
            raise ValueError("min_terminal_specific_energy_margin_j_kg must be finite when set.")

    def to_record(self) -> dict[str, object]:
        return {
            "min_safety_margin_m": float(self.min_safety_margin_m),
            "max_angle_of_attack_rad": float(self.max_angle_of_attack_rad),
            "max_command_abs_rad": float(self.max_command_abs_rad),
            "min_terminal_specific_energy_change_j_kg": float(self.min_terminal_specific_energy_change_j_kg),
            "min_terminal_specific_energy_margin_j_kg": (
                None
                if self.min_terminal_specific_energy_margin_j_kg is None
                else float(self.min_terminal_specific_energy_margin_j_kg)
            ),
        }


@dataclass(frozen=True)
class RetentionDecision:
    """Transparent primitive retention result."""

    retained: bool
    reason: str
    thresholds: AcceptanceThresholds

    def to_record(self) -> dict[str, object]:
        return {
            "retained": self.retained,
            "reason": self.reason,
            "thresholds": self.thresholds.to_record(),
        }


@dataclass(frozen=True)
class ValidationScenario:
    """Compact primitive validation scenario specification."""

    scenario_id: str
    seed: int | None = 0
    entry_perturbation: EntryPerturbationSpec = EntryPerturbationSpec()
    rollout_config: PrimitiveRolloutConfig = PrimitiveRolloutConfig()
    task: GateTraversalTask | None = None
    wind_model: object | None = None
    wind_mode: str = "panel"
    thresholds: AcceptanceThresholds = AcceptanceThresholds()

    def validate(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must be non-empty.")
        self.entry_perturbation.validate()
        self.thresholds.validate()
        if self.wind_mode not in {"cg", "panel"}:
            raise ValueError("wind_mode must be 'cg' or 'panel'.")

    def effective_seed(self, primitive_id: str) -> int | None:
        return _stable_seed(self.seed, self.scenario_id, primitive_id)

    def initial_state_for(self, primitive: PrimitiveCandidate) -> tuple[FlightState, int | None]:
        self.validate()
        seed = self.effective_seed(primitive.primitive_id)
        return self.entry_perturbation.sample(primitive.reference.state_at(0.0), seed=seed), seed

    def rollout_config_for(self, seed: int | None) -> PrimitiveRolloutConfig:
        self.validate()
        return replace(self.rollout_config, scenario_id=self.scenario_id, seed=seed, wind_mode=self.wind_mode)


@dataclass(frozen=True)
class PrimitiveValidationResult:
    """Validation record plus the rollout object kept in memory only."""

    primitive_id: str
    family: str
    controller_type: str
    scenario_id: str
    seed: int | None
    entry_class: StateClassLabel
    exit_class: StateClassLabel
    evidence: PrimitiveEvidence
    retention: RetentionDecision
    rollout: PrimitiveRolloutResult

    def to_record(self) -> dict[str, object]:
        record = self.evidence.to_record()
        record.update(
            {
                "entry_class": self.entry_class.label,
                "exit_class": self.exit_class.label,
                "entry_class_record": self.entry_class.to_record(),
                "exit_class_record": self.exit_class.to_record(),
                "retained": self.retention.retained,
                "retention_reason": self.retention.reason,
                "retention_thresholds": self.retention.thresholds.to_record(),
            }
        )
        return record


def decide_retention(evidence: PrimitiveEvidence, thresholds: AcceptanceThresholds) -> RetentionDecision:
    """Apply the smoke validation retention rule to one evidence record."""

    thresholds.validate()
    if not evidence.rollout_success:
        return RetentionDecision(False, f"rollout_failure:{evidence.failure_reason or 'unknown'}", thresholds)
    if evidence.min_safety_margin_m < thresholds.min_safety_margin_m:
        return RetentionDecision(False, "min_safety_margin", thresholds)
    if evidence.max_angle_of_attack_rad > thresholds.max_angle_of_attack_rad:
        return RetentionDecision(False, "angle_of_attack", thresholds)
    if evidence.max_command_abs_rad > thresholds.max_command_abs_rad:
        return RetentionDecision(False, "command_magnitude", thresholds)
    if evidence.terminal_specific_energy_change_j_kg < thresholds.min_terminal_specific_energy_change_j_kg:
        return RetentionDecision(False, "terminal_specific_energy_change", thresholds)
    if thresholds.min_terminal_specific_energy_margin_j_kg is not None:
        if evidence.terminal_specific_energy_margin_j_kg is None:
            return RetentionDecision(False, "terminal_specific_energy_margin_missing", thresholds)
        if evidence.terminal_specific_energy_margin_j_kg < thresholds.min_terminal_specific_energy_margin_j_kg:
            return RetentionDecision(False, "terminal_specific_energy_margin", thresholds)
    return RetentionDecision(True, "retained", thresholds)


def validate_primitive(
    primitive: PrimitiveCandidate,
    scenario: ValidationScenario,
    classifier: PrimitiveStateClassifier | None = None,
    model: GliderModel | None = None,
    actuator: ActuatorModel | None = None,
    wind_model: object | None = None,
) -> PrimitiveValidationResult:
    """Roll out one primitive in one scenario and return a serialisable validation record."""

    state_classifier = PrimitiveStateClassifier() if classifier is None else classifier
    entry_state, seed = scenario.initial_state_for(primitive)
    entry_class = state_classifier.classify(entry_state)
    selected_wind_model = scenario.wind_model if scenario.wind_model is not None else wind_model
    rollout = rollout_primitive(
        primitive=primitive,
        initial_state=entry_state,
        task=scenario.task,
        model=model,
        actuator=actuator,
        wind_model=selected_wind_model,
        config=scenario.rollout_config_for(seed),
    )
    exit_state = rollout.states[-1] if rollout.states else entry_state
    exit_class = state_classifier.classify(exit_state)
    decision = decide_retention(rollout.evidence, scenario.thresholds)
    evidence = replace(
        rollout.evidence,
        entry_class=entry_class.label,
        exit_class=exit_class.label,
        retained=decision.retained,
        retention_reason=decision.reason,
    )
    rollout_with_evidence = replace(rollout, evidence=evidence, failure_reason=evidence.failure_reason)
    return PrimitiveValidationResult(
        primitive_id=primitive.primitive_id,
        family=primitive.family,
        controller_type=primitive.controller_type,
        scenario_id=scenario.scenario_id,
        seed=seed,
        entry_class=entry_class,
        exit_class=exit_class,
        evidence=evidence,
        retention=decision,
        rollout=rollout_with_evidence,
    )
