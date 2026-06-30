"""Central baseline and ablation registry."""

from __future__ import annotations

from dataclasses import dataclass, field


B0_OPEN_LOOP_DIAGNOSTIC = "B0_open_loop_diagnostic"
B1_UNGOVERNED_PRIMITIVE_SELECTOR = "B1_ungoverned_primitive_selector"
B2_FILTER_ONLY_NO_DEGRADATION = "B2_filter_only_no_degradation"
B3_NO_RETURNABILITY_SELECTOR = "B3_no_returnability_selector"
B4_TRAJECTORY_TRACKING_LQR_TVLQR = "B4_trajectory_tracking_lqr_tvlqr"
B5_ENERGY_LATERAL_GUIDANCE = "B5_energy_lateral_guidance"
B6_WIND_AWARE_GUIDANCE = "B6_wind_aware_guidance"
B7_INNER_LOOP_SUBSTITUTION = "B7_inner_loop_substitution"
B8_LIFT_EVIDENCE_REMOVED = "B8_lift_evidence_removed"


BASELINE_COMMON_SCHEMA_FIELDS = (
    "method_name",
    "case_id",
    "case_family",
    "scenario_seed",
    "controller_config_id",
    "primitive_library_id",
    "returnability_graph_id",
    "request_id",
    "decision_type",
    "selected_primitive_id",
    "requested_primitive_id",
    "degradation_level",
    "rejection_reasons",
    "active_constraints",
    "predicted_successor_class",
    "predicted_returnability",
    "predicted_safety_margin",
    "predicted_energy_delta",
    "predicted_lift_exposure",
    "actual_successor_class",
    "rollout_success",
    "gate_crossed",
    "gate_miss_distance_m",
    "min_safety_margin_m",
    "terminal_specific_energy_margin_j_kg",
    "terminal_specific_energy_change_j_kg",
    "useful_lift_exposure",
    "max_angle_of_attack_rad",
    "max_command_abs_rad",
    "failure_reason",
    "failure_mode",
    "runtime_ms",
    "candidate_count",
    "admissible_candidate_count",
    "filtered_count_by_stage",
    "fallback_used",
)


@dataclass(frozen=True)
class BaselineSpec:
    """Static public definition of one benchmark baseline or ablation."""

    baseline_id: str
    method_name: str
    description: str
    uses_returnability: bool
    uses_degradation: bool
    uses_wind_information: bool
    scaffold_level: str = "preflight"
    removed_objective_terms: tuple[str, ...] = ()

    def to_record(self) -> dict[str, object]:
        return {
            "method_name": self.method_name,
            "description": self.description,
            "uses_returnability": bool(self.uses_returnability),
            "uses_degradation": bool(self.uses_degradation),
            "uses_wind_information": bool(self.uses_wind_information),
            "scaffold_level": self.scaffold_level,
            "removed_objective_terms": list(self.removed_objective_terms),
        }


@dataclass(frozen=True)
class BaselineSmokeRecord:
    """Common-schema in-memory baseline smoke record."""

    method_name: str
    case_id: str
    case_family: str
    scenario_seed: int | None
    controller_config_id: str
    primitive_library_id: str | None = None
    returnability_graph_id: str | None = None
    request_id: str | None = None
    decision_type: str | None = "baseline_scaffold"
    selected_primitive_id: str | None = None
    requested_primitive_id: str | None = None
    degradation_level: int = 0
    rejection_reasons: tuple[str, ...] = ()
    active_constraints: tuple[str, ...] = ()
    predicted_successor_class: str | None = None
    predicted_returnability: float | None = None
    predicted_safety_margin: float | None = None
    predicted_energy_delta: float | None = None
    predicted_lift_exposure: float | None = None
    actual_successor_class: str | None = None
    rollout_success: bool = False
    gate_crossed: bool | None = None
    gate_miss_distance_m: float | None = None
    min_safety_margin_m: float | None = None
    terminal_specific_energy_margin_j_kg: float | None = None
    terminal_specific_energy_change_j_kg: float | None = None
    useful_lift_exposure: float | None = None
    max_angle_of_attack_rad: float | None = None
    max_command_abs_rad: float | None = None
    failure_reason: str | None = None
    failure_mode: str | None = None
    runtime_ms: float = 0.0
    candidate_count: int = 0
    admissible_candidate_count: int = 0
    filtered_count_by_stage: dict[str, int] = field(default_factory=dict)
    fallback_used: bool = False
    same_initial_state: bool = True
    same_scenario_seed: bool = True
    same_actuator_limits: bool = True
    wind_information_available: bool = False

    def to_record(self) -> dict[str, object]:
        return {
            "method_name": self.method_name,
            "case_id": self.case_id,
            "case_family": self.case_family,
            "scenario_seed": self.scenario_seed,
            "controller_config_id": self.controller_config_id,
            "primitive_library_id": self.primitive_library_id,
            "returnability_graph_id": self.returnability_graph_id,
            "request_id": self.request_id,
            "decision_type": self.decision_type,
            "selected_primitive_id": self.selected_primitive_id,
            "requested_primitive_id": self.requested_primitive_id,
            "degradation_level": int(self.degradation_level),
            "rejection_reasons": list(self.rejection_reasons),
            "active_constraints": list(self.active_constraints),
            "predicted_successor_class": self.predicted_successor_class,
            "predicted_returnability": self.predicted_returnability,
            "predicted_safety_margin": self.predicted_safety_margin,
            "predicted_energy_delta": self.predicted_energy_delta,
            "predicted_lift_exposure": self.predicted_lift_exposure,
            "actual_successor_class": self.actual_successor_class,
            "rollout_success": bool(self.rollout_success),
            "gate_crossed": self.gate_crossed,
            "gate_miss_distance_m": self.gate_miss_distance_m,
            "min_safety_margin_m": self.min_safety_margin_m,
            "terminal_specific_energy_margin_j_kg": self.terminal_specific_energy_margin_j_kg,
            "terminal_specific_energy_change_j_kg": self.terminal_specific_energy_change_j_kg,
            "useful_lift_exposure": self.useful_lift_exposure,
            "max_angle_of_attack_rad": self.max_angle_of_attack_rad,
            "max_command_abs_rad": self.max_command_abs_rad,
            "failure_reason": self.failure_reason,
            "failure_mode": self.failure_mode,
            "runtime_ms": float(self.runtime_ms),
            "candidate_count": int(self.candidate_count),
            "admissible_candidate_count": int(self.admissible_candidate_count),
            "filtered_count_by_stage": dict(sorted(self.filtered_count_by_stage.items())),
            "fallback_used": bool(self.fallback_used),
            "same_initial_state": bool(self.same_initial_state),
            "same_scenario_seed": bool(self.same_scenario_seed),
            "same_actuator_limits": bool(self.same_actuator_limits),
            "wind_information_available": bool(self.wind_information_available),
        }


@dataclass(frozen=True)
class BaselineScaffold:
    """Minimal deterministic baseline object used by preflight checks."""

    spec: BaselineSpec

    def evaluate_smoke_case(
        self,
        *,
        case_id: str = "still_air_easy",
        case_family: str = "still_air",
        scenario_seed: int | None = 0,
        wind_information_available: bool | None = None,
    ) -> BaselineSmokeRecord:
        return BaselineSmokeRecord(
            method_name=self.spec.method_name,
            case_id=case_id,
            case_family=case_family,
            scenario_seed=scenario_seed,
            controller_config_id=f"{self.spec.method_name}_preflight",
            request_id=f"baseline:{self.spec.method_name}:{case_id}",
            decision_type="baseline_scaffold",
            failure_reason=None,
            failure_mode=None,
            wind_information_available=self.spec.uses_wind_information
            if wind_information_available is None
            else bool(wind_information_available),
        )


BASELINE_REGISTRY: dict[str, BaselineSpec] = {
    B0_OPEN_LOOP_DIAGNOSTIC: BaselineSpec(
        baseline_id=B0_OPEN_LOOP_DIAGNOSTIC,
        method_name="open_loop_diagnostic",
        description="fixed trim or zero-command diagnostic reference",
        uses_returnability=False,
        uses_degradation=False,
        uses_wind_information=False,
    ),
    B1_UNGOVERNED_PRIMITIVE_SELECTOR: BaselineSpec(
        baseline_id=B1_UNGOVERNED_PRIMITIVE_SELECTOR,
        method_name="ungoverned_primitive_selector",
        description="same primitive library and objective score terms without returnability filtering or degradation",
        uses_returnability=False,
        uses_degradation=False,
        uses_wind_information=False,
    ),
    B2_FILTER_ONLY_NO_DEGRADATION: BaselineSpec(
        baseline_id=B2_FILTER_ONLY_NO_DEGRADATION,
        method_name="filter_only_no_degradation",
        description="entry, actuator, safety, and returnability filters without structural degradation",
        uses_returnability=True,
        uses_degradation=False,
        uses_wind_information=False,
    ),
    B3_NO_RETURNABILITY_SELECTOR: BaselineSpec(
        baseline_id=B3_NO_RETURNABILITY_SELECTOR,
        method_name="no_returnability_selector",
        description="local safety and objective scoring while ignoring successor returnability",
        uses_returnability=False,
        uses_degradation=False,
        uses_wind_information=False,
    ),
    B4_TRAJECTORY_TRACKING_LQR_TVLQR: BaselineSpec(
        baseline_id=B4_TRAJECTORY_TRACKING_LQR_TVLQR,
        method_name="trajectory_tracking_lqr_tvlqr",
        description="trajectory tracking interface with LQR-compatible smoke implementation",
        uses_returnability=False,
        uses_degradation=False,
        uses_wind_information=False,
    ),
    B5_ENERGY_LATERAL_GUIDANCE: BaselineSpec(
        baseline_id=B5_ENERGY_LATERAL_GUIDANCE,
        method_name="energy_lateral_guidance",
        description="simplified fixed-wing energy and lateral guidance scaffold",
        uses_returnability=False,
        uses_degradation=False,
        uses_wind_information=False,
    ),
    B6_WIND_AWARE_GUIDANCE: BaselineSpec(
        baseline_id=B6_WIND_AWARE_GUIDANCE,
        method_name="wind_aware_guidance",
        description="guidance scaffold using the same updraft information available to Moewe",
        uses_returnability=False,
        uses_degradation=False,
        uses_wind_information=True,
    ),
    B7_INNER_LOOP_SUBSTITUTION: BaselineSpec(
        baseline_id=B7_INNER_LOOP_SUBSTITUTION,
        method_name="inner_loop_substitution",
        description="local-controller substitution ablation scaffold",
        uses_returnability=True,
        uses_degradation=True,
        uses_wind_information=False,
    ),
    B8_LIFT_EVIDENCE_REMOVED: BaselineSpec(
        baseline_id=B8_LIFT_EVIDENCE_REMOVED,
        method_name="lift_evidence_removed",
        description="returnability-supervised selection with lift and energy opportunity terms removed",
        uses_returnability=True,
        uses_degradation=True,
        uses_wind_information=False,
        removed_objective_terms=("useful_lift_exposure", "terminal_energy_opportunity"),
    ),
}

BASELINE_METHOD_NAMES = tuple(spec.method_name for spec in BASELINE_REGISTRY.values())
BASELINE_ALIASES = {
    "unfiltered": "ungoverned_primitive_selector",
    "reference_tracking_pd": "trajectory_tracking_lqr_tvlqr",
}


def baseline_spec(identifier: str) -> BaselineSpec:
    """Return a baseline spec by B identifier, semantic method name, or legacy alias."""

    if identifier in BASELINE_REGISTRY:
        return BASELINE_REGISTRY[identifier]
    method_name = BASELINE_ALIASES.get(identifier, identifier)
    for spec in BASELINE_REGISTRY.values():
        if spec.method_name == method_name:
            return spec
    raise KeyError(f"Unknown baseline identifier: {identifier}")


def instantiate_baseline(identifier: str) -> BaselineScaffold:
    """Instantiate a deterministic preflight baseline scaffold."""

    return BaselineScaffold(spec=baseline_spec(identifier))


def baseline_registry_records() -> list[dict[str, object]]:
    """Return deterministic public registry records."""

    return [BASELINE_REGISTRY[key].to_record() for key in sorted(BASELINE_REGISTRY)]


__all__ = [
    "B0_OPEN_LOOP_DIAGNOSTIC",
    "B1_UNGOVERNED_PRIMITIVE_SELECTOR",
    "B2_FILTER_ONLY_NO_DEGRADATION",
    "B3_NO_RETURNABILITY_SELECTOR",
    "B4_TRAJECTORY_TRACKING_LQR_TVLQR",
    "B5_ENERGY_LATERAL_GUIDANCE",
    "B6_WIND_AWARE_GUIDANCE",
    "B7_INNER_LOOP_SUBSTITUTION",
    "B8_LIFT_EVIDENCE_REMOVED",
    "BASELINE_ALIASES",
    "BASELINE_COMMON_SCHEMA_FIELDS",
    "BASELINE_METHOD_NAMES",
    "BASELINE_REGISTRY",
    "BaselineScaffold",
    "BaselineSmokeRecord",
    "BaselineSpec",
    "baseline_registry_records",
    "baseline_spec",
    "instantiate_baseline",
]
