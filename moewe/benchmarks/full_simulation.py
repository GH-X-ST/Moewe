"""Decision-centric first full simulation campaign runner."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter

from moewe.baselines import (
    BASELINE_COMMON_SCHEMA_FIELDS,
    ReferenceTrackingConfig,
    UnfilteredPrimitiveSelector,
    UnfilteredPrimitiveSelectorConfig,
    run_reference_tracking_rollout,
)
from moewe.campaigns import (
    RandomUpdraftChallengeCase,
    RandomUpdraftChallengeConfig,
    RandomUpdraftChallengeMethodRecord,
    build_random_updraft_challenge_cases,
)
from moewe.governor import (
    DegradationPolicy,
    ManoeuvrePrimitiveGovernor,
    OnlineGovernorConfig,
    PrimitiveRequest,
)
from moewe.objectives import GateTraversalProposer, LiftExploitationProposer, RecoveryProposer
from moewe.primitives import PrimitiveLibrary, PrimitiveLibraryCandidate, PrimitiveRolloutConfig, rollout_primitive
from moewe.returnability import ReturnabilityGraph
from moewe.sim.actuator import NAUSICAA_MAX_COMMAND_ABS_RAD
from moewe.sim.glider_model import NAUSICAA_OPERATIONAL_ALPHA_LIMIT_RAD
from moewe.tasks import specific_energy_j_kg

FIRST_FULL_SIMULATION_METHODS = (
    "governor",
    "ungoverned_primitive_selector",
    "filter_only_no_degradation",
    "no_returnability_selector",
    "trajectory_tracking_lqr_tvlqr",
    "lift_evidence_removed",
)
FIRST_FULL_SIMULATION_CASE_FAMILIES = (
    "weak_random_single_source",
    "hard_random_single_source",
    "random_two_source",
    "random_four_source",
)
SCAFFOLD_ONLY_METHODS = (
    "open_loop_diagnostic",
    "energy_lateral_guidance",
    "wind_aware_guidance",
    "inner_loop_substitution",
)
_GOVERNED_METHODS = frozenset(
    {
        "governor",
        "filter_only_no_degradation",
        "lift_evidence_removed",
    }
)


@dataclass(frozen=True)
class FirstFullSimulationConfig:
    """Configuration for the first decision-centric full simulation campaign."""

    full_case_count: int = 120
    base_seed: int = 91
    tier: str = "balanced"
    methods: tuple[str, ...] = FIRST_FULL_SIMULATION_METHODS
    case_families: tuple[str, ...] = FIRST_FULL_SIMULATION_CASE_FAMILIES
    dt_s: float = 0.01
    max_duration_s: float = 0.10
    reference_horizon_s: float = 0.30
    wind_mode: str = "panel"
    local_safety_min_margin_m: float = 0.0
    local_safety_max_angle_of_attack_rad: float = NAUSICAA_OPERATIONAL_ALPHA_LIMIT_RAD
    local_safety_max_command_abs_rad: float = NAUSICAA_MAX_COMMAND_ABS_RAD
    run_command: str | None = None
    tests_status: str | None = None

    def __post_init__(self) -> None:
        if int(self.full_case_count) <= 0:
            raise ValueError("full_case_count must be positive.")
        if int(self.base_seed) < 0:
            raise ValueError("base_seed must be non-negative.")
        if not self.tier:
            raise ValueError("tier must be non-empty.")
        if not self.methods:
            raise ValueError("At least one method is required.")
        allowed = set(FIRST_FULL_SIMULATION_METHODS) | set(SCAFFOLD_ONLY_METHODS)
        unknown = sorted(set(self.methods) - allowed)
        if unknown:
            raise ValueError(f"Unknown first full simulation methods: {unknown}")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("methods must not contain duplicates.")
        if not self.case_families:
            raise ValueError("At least one case family is required.")
        unknown_families = sorted(set(self.case_families) - set(FIRST_FULL_SIMULATION_CASE_FAMILIES))
        if unknown_families:
            raise ValueError(f"Unknown case families: {unknown_families}")
        if len(set(self.case_families)) != len(self.case_families):
            raise ValueError("case_families must not contain duplicates.")
        for name in (
            "dt_s",
            "max_duration_s",
            "reference_horizon_s",
            "local_safety_min_margin_m",
            "local_safety_max_angle_of_attack_rad",
            "local_safety_max_command_abs_rad",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        for name in ("dt_s", "max_duration_s", "reference_horizon_s"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        for name in ("local_safety_max_angle_of_attack_rad", "local_safety_max_command_abs_rad"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if self.wind_mode not in {"cg", "panel"}:
            raise ValueError("wind_mode must be 'cg' or 'panel'.")

    @property
    def performance_methods(self) -> tuple[str, ...]:
        return tuple(method for method in self.methods if method not in SCAFFOLD_ONLY_METHODS)

    @property
    def expected_record_count(self) -> int:
        return int(self.full_case_count) * len(self.performance_methods)

    def to_record(self) -> dict[str, object]:
        return {
            "full_case_count": int(self.full_case_count),
            "base_seed": int(self.base_seed),
            "tier": self.tier,
            "methods": list(self.methods),
            "performance_methods": list(self.performance_methods),
            "case_families": list(self.case_families),
            "dt_s": float(self.dt_s),
            "max_duration_s": float(self.max_duration_s),
            "reference_horizon_s": float(self.reference_horizon_s),
            "wind_mode": self.wind_mode,
            "local_safety_min_margin_m": float(self.local_safety_min_margin_m),
            "local_safety_max_angle_of_attack_rad": float(self.local_safety_max_angle_of_attack_rad),
            "local_safety_max_command_abs_rad": float(self.local_safety_max_command_abs_rad),
            "expected_record_count": self.expected_record_count,
            "run_command": self.run_command,
            "tests_status": self.tests_status,
        }


@dataclass(frozen=True)
class FirstFullSimulationReport:
    """In-memory report for a completed or explicitly partial first full run."""

    config: FirstFullSimulationConfig
    output_dir: Path
    records: tuple[RandomUpdraftChallengeMethodRecord, ...]
    method_capability_audit: dict[str, object]
    summary_by_method: dict[str, object]
    summary_by_case_family: dict[str, object]
    decision_diagnostics: list[dict[str, object]]
    failure_taxonomy: dict[str, object]
    runtime_summary: dict[str, object]
    manifest: dict[str, object]
    partial_run: bool = False
    stop_reason: str | None = None

    @property
    def case_count(self) -> int:
        return len({record.case_id for record in self.records})

    @property
    def record_count(self) -> int:
        return len(self.records)

    @property
    def methods(self) -> tuple[str, ...]:
        return _unique_ordered(record.selector_name for record in self.records)

    def to_summary(self) -> dict[str, object]:
        return {
            "partial_run": bool(self.partial_run),
            "stop_reason": self.stop_reason,
            "output_dir": str(self.output_dir),
            "case_count": self.case_count,
            "record_count": self.record_count,
            "methods": list(self.methods),
            "excluded_scaffold_methods": self.method_capability_audit["excluded_scaffold_methods"],
        }


@dataclass(frozen=True)
class _MethodDecision:
    method_name: str
    selected_candidate_id: str | None
    selected_primitive_id: str | None
    fallback_used: bool
    fallback_reason: str | None
    decision_record: dict[str, object]


@dataclass(frozen=True)
class _FullSimulationContext:
    governed: ManoeuvrePrimitiveGovernor
    filter_only: ManoeuvrePrimitiveGovernor
    lift_removed: ManoeuvrePrimitiveGovernor
    ungoverned_selector: UnfilteredPrimitiveSelector


def require_first_full_simulation_guards(
    *,
    controller_frozen: bool,
    write_results: bool,
    output_dir: str | Path | None,
    public_repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Validate explicit full-run guards before executing or writing outputs."""

    if not controller_frozen:
        raise ValueError("First full simulation requires controller_frozen=True.")
    if not write_results:
        raise ValueError("First full simulation requires write_results=True.")
    if output_dir is None:
        raise ValueError("First full simulation requires an explicit output_dir.")
    output_path = Path(output_dir).resolve()
    repo_root = _resolve_public_repo_root(public_repo_root)
    if repo_root is not None and (output_path == repo_root or repo_root in output_path.parents):
        raise ValueError("First full simulation output_dir must be outside the public repository.")
    return {
        "controller_frozen": True,
        "write_results": True,
        "output_dir": str(output_path),
        "public_repo_root": None if repo_root is None else str(repo_root),
        "runs_full_simulation": True,
    }


def run_first_full_simulation_campaign(
    *,
    library: PrimitiveLibrary,
    graph: ReturnabilityGraph,
    output_dir: str | Path,
    config: FirstFullSimulationConfig | None = None,
    controller_frozen: bool = False,
    write_results: bool = False,
    public_repo_root: str | Path | None = None,
) -> FirstFullSimulationReport:
    """Run the first decision-centric full simulation and write private artifacts."""

    cfg = FirstFullSimulationConfig() if config is None else config
    guard_record = require_first_full_simulation_guards(
        controller_frozen=controller_frozen,
        write_results=write_results,
        output_dir=output_dir,
        public_repo_root=public_repo_root,
    )
    output_path = Path(output_dir).resolve()
    method_audit = _method_capability_audit(cfg.methods)
    cases = _first_full_simulation_cases(cfg)
    context = _full_simulation_context(library, graph, cfg)
    records: list[RandomUpdraftChallengeMethodRecord] = []

    for case in cases:
        for method_name in cfg.performance_methods:
            records.append(_run_method_case(method_name, case, library, context, cfg))

    record_tuple = tuple(records)
    _validate_first_full_records(record_tuple, cfg)
    record_dicts = [record.to_record() for record in record_tuple]
    summary_by_method = _summary_by_method(record_dicts, method_audit)
    summary_by_case_family = _summary_by_case_family(record_dicts)
    failure_taxonomy = _failure_taxonomy(record_dicts)
    runtime_summary = _runtime_summary(record_dicts)
    decision_diagnostics = _decision_diagnostics(record_dicts)
    manifest = _manifest(
        cfg=cfg,
        guard_record=guard_record,
        output_dir=output_path,
        records=record_dicts,
        method_audit=method_audit,
        public_repo_root=public_repo_root,
        partial_run=False,
        stop_reason=None,
    )
    report = FirstFullSimulationReport(
        config=cfg,
        output_dir=output_path,
        records=record_tuple,
        method_capability_audit=method_audit,
        summary_by_method=summary_by_method,
        summary_by_case_family=summary_by_case_family,
        decision_diagnostics=decision_diagnostics,
        failure_taxonomy=failure_taxonomy,
        runtime_summary=runtime_summary,
        manifest=manifest,
        partial_run=False,
        stop_reason=None,
    )
    _write_private_outputs(report)
    return report


def _resolve_public_repo_root(public_repo_root: str | Path | None) -> Path | None:
    if public_repo_root is not None:
        return Path(public_repo_root).resolve()
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _first_full_simulation_cases(config: FirstFullSimulationConfig) -> tuple[RandomUpdraftChallengeCase, ...]:
    generated_count = int(config.full_case_count) * len(FIRST_FULL_SIMULATION_CASE_FAMILIES)
    random_config = RandomUpdraftChallengeConfig(
        case_count=generated_count,
        base_seed=int(config.base_seed),
        tier=config.tier,
        selectors=("governor",),
        dt_s=float(config.dt_s),
        max_duration_s=float(config.max_duration_s),
        reference_horizon_s=float(config.reference_horizon_s),
        wind_mode=config.wind_mode,
        controller_frozen=True,
    )
    candidates = build_random_updraft_challenge_cases(random_config)
    families = set(config.case_families)
    selected = [case for case in candidates if case.environment_family in families]
    if len(selected) < int(config.full_case_count):
        raise ValueError("Could not build enough first full simulation cases.")
    return tuple(selected[: int(config.full_case_count)])


def _full_simulation_context(
    library: PrimitiveLibrary,
    graph: ReturnabilityGraph,
    config: FirstFullSimulationConfig,
) -> _FullSimulationContext:
    governor_config = OnlineGovernorConfig(tier=config.tier)
    return _FullSimulationContext(
        governed=ManoeuvrePrimitiveGovernor(
            library,
            graph,
            governor_config,
            degradation_policy=DegradationPolicy(enabled=True),
        ),
        filter_only=ManoeuvrePrimitiveGovernor(
            library,
            graph,
            governor_config,
            degradation_policy=DegradationPolicy(enabled=False),
        ),
        lift_removed=ManoeuvrePrimitiveGovernor(
            library,
            graph,
            governor_config,
            degradation_policy=DegradationPolicy(enabled=True),
        ),
        ungoverned_selector=UnfilteredPrimitiveSelector(
            library,
            UnfilteredPrimitiveSelectorConfig(tier=config.tier),
        ),
    )


def _run_method_case(
    method_name: str,
    case: RandomUpdraftChallengeCase,
    library: PrimitiveLibrary,
    context: _FullSimulationContext,
    config: FirstFullSimulationConfig,
) -> RandomUpdraftChallengeMethodRecord:
    if method_name == "trajectory_tracking_lqr_tvlqr":
        return _reference_tracking_record(case, config)
    request = _request_for_case(method_name, case, library, config)
    if method_name == "governor":
        decision = context.governed.decide(case.initial_state, request, tier=config.tier)
        method_decision = _decision_from_governor_record(method_name, decision.to_record())
    elif method_name == "filter_only_no_degradation":
        decision = context.filter_only.decide(case.initial_state, request, tier=config.tier)
        record = decision.to_record()
        record["degradation_enabled"] = False
        method_decision = _decision_from_governor_record(method_name, record)
    elif method_name == "lift_evidence_removed":
        decision = context.lift_removed.decide(case.initial_state, request, tier=config.tier)
        record = decision.to_record()
        record["removed_objective_terms"] = ["useful_lift_exposure", "terminal_energy"]
        method_decision = _decision_from_governor_record(method_name, record)
    elif method_name == "ungoverned_primitive_selector":
        method_decision = _objective_selector_decision(
            method_name,
            case,
            library,
            request,
            config,
            local_safety_filter=False,
        )
    elif method_name == "no_returnability_selector":
        method_decision = _objective_selector_decision(
            method_name,
            case,
            library,
            request,
            config,
            local_safety_filter=True,
        )
    else:
        raise ValueError(f"Method is not implemented for performance records: {method_name}")
    return _primitive_rollout_record(case, method_decision, library, config)


def _request_for_case(
    method_name: str,
    case: RandomUpdraftChallengeCase,
    library: PrimitiveLibrary,
    config: FirstFullSimulationConfig,
) -> PrimitiveRequest:
    proposer_reason = "gate_default"
    if _low_energy(case) or _near_boundary(case):
        proposer = RecoveryProposer()
        proposer_reason = "state_recovery"
    elif case.environment_family in {"hard_random_single_source", "random_two_source", "random_four_source"}:
        proposer = LiftExploitationProposer(memory_enabled=False)
        proposer_reason = "strong_or_multi_source_updraft"
    else:
        proposer = GateTraversalProposer()
    request = proposer.propose(case.initial_state)
    query = library.query(case.initial_state, tier=config.tier)
    if not query.candidates and proposer_reason != "state_recovery":
        request = RecoveryProposer().propose(case.initial_state)
        proposer_reason = "empty_candidate_set_recovery"

    score_terms = dict(request.objective_score_terms)
    removed_terms: list[str] = []
    if method_name == "lift_evidence_removed":
        for term in ("useful_lift_exposure", "terminal_energy"):
            if term in score_terms:
                score_terms[term] = 0.0
                removed_terms.append(term)
        if "terminal_energy_opportunity" not in score_terms:
            score_terms["terminal_energy_opportunity"] = 0.0
        removed_terms.append("terminal_energy_opportunity")

    requested_id, candidate_ids = _objective_candidate_set(
        library,
        case.initial_state,
        tier=config.tier,
        preferred_family=request.preferred_family,
        requested_aggressiveness=request.requested_aggressiveness,
        objective_score_terms=score_terms,
    )
    metadata = {
        **dict(request.metadata),
        "case_id": case.case_id,
        "case_family": case.environment_family,
        "method_name": method_name,
        "proposer_policy": proposer_reason,
        "spatial_memory": "off",
        "removed_objective_terms": removed_terms,
    }
    return PrimitiveRequest(
        request_id=f"objective:{metadata['proposer']}:{case.case_id}:{method_name}",
        task_intent=request.task_intent,
        preferred_family=request.preferred_family,
        requested_aggressiveness=request.requested_aggressiveness,
        target_region=request.target_region,
        objective_score_terms=score_terms,
        requested_primitive_id=requested_id,
        candidate_ids=candidate_ids,
        metadata=metadata,
    )


def _low_energy(case: RandomUpdraftChallengeCase) -> bool:
    return specific_energy_j_kg(case.initial_state) < float(case.task.required_terminal_specific_energy_j_kg)


def _near_boundary(case: RandomUpdraftChallengeCase) -> bool:
    state = case.initial_state
    volume = case.task.flight_volume
    return (
        float(state.position_w_m[2]) < float(volume.z_min_m) + 0.15
        or float(state.position_w_m[1]) < float(volume.y_min_m) + 0.10
        or float(state.position_w_m[1]) > float(volume.y_max_m) - 0.10
    )


def _objective_candidate_set(
    library: PrimitiveLibrary,
    state: object,
    *,
    tier: str,
    preferred_family: str,
    requested_aggressiveness: int,
    objective_score_terms: dict[str, float],
) -> tuple[str | None, tuple[str, ...]]:
    query = library.query(state, tier=tier)
    candidates = tuple(candidate for candidate in query.candidates if candidate.family == preferred_family)
    if not candidates:
        candidates = tuple(query.candidates)
    candidate_ids = tuple(candidate.primitive_id for candidate in candidates)
    if not candidates:
        return None, ()
    selected = max(
        candidates,
        key=lambda candidate: (
            _objective_score(candidate, objective_score_terms)
            - 0.01 * abs(_candidate_aggressiveness(candidate) - int(requested_aggressiveness)),
            -_candidate_aggressiveness(candidate),
            candidate.primitive_id,
        ),
    )
    return selected.primitive_id, candidate_ids


def _objective_selector_decision(
    method_name: str,
    case: RandomUpdraftChallengeCase,
    library: PrimitiveLibrary,
    request: PrimitiveRequest,
    config: FirstFullSimulationConfig,
    *,
    local_safety_filter: bool,
) -> _MethodDecision:
    start_s = perf_counter()
    query = library.query(case.initial_state, tier=config.tier)
    candidates = tuple(
        candidate
        for candidate in query.candidates
        if not local_safety_filter or _local_safety_ok(candidate, config)
    )
    selected = None
    if candidates and (not query.fallback_used or method_name == "ungoverned_primitive_selector"):
        allowed = set(request.candidate_ids)
        ranked = tuple(
            candidate
            for candidate in candidates
            if not allowed or candidate.primitive_id in allowed or bool(allowed & set(candidate.represented_primitive_ids))
        )
        if not ranked:
            ranked = candidates
        selected = max(
            ranked,
            key=lambda candidate: (
                _objective_score(candidate, request.objective_score_terms),
                -_candidate_aggressiveness(candidate),
                candidate.primitive_id,
            ),
        )
    rejection_reasons: list[str] = []
    if query.fallback_used and method_name != "ungoverned_primitive_selector":
        rejection_reasons.append("retrieval_fallback_not_allowed")
    if not query.candidates:
        rejection_reasons.append("no_retrieval_candidate")
    if local_safety_filter and query.candidates and not candidates:
        rejection_reasons.append("no_local_safety_candidate")
    record = {
        "request_id": request.request_id,
        "decision_type": "rank" if selected is not None else "reject",
        "selected_candidate_id": None if selected is None else selected.primitive_id,
        "selected_primitive_id": None if selected is None else selected.primitive_id,
        "requested_primitive_id": request.requested_primitive_id,
        "degradation_level": 0,
        "degradation_stage": None,
        "rejection_reasons": _unique_ordered(rejection_reasons),
        "active_constraints": ["local_safety"] if local_safety_filter else [],
        "entry_class": query.entry_class,
        "predicted_successor_class": None if selected is None else selected.exit_class,
        "predicted_returnability": None,
        "predicted_safety_margin": None if selected is None else _feature_float(selected, "min_safety_margin_m"),
        "predicted_energy_delta": None
        if selected is None
        else _feature_float(selected, "terminal_specific_energy_change_j_kg"),
        "predicted_lift_exposure": None
        if selected is None
        else _feature_float(selected, "mean_positive_vertical_wind_m_s"),
        "candidate_count": len(query.candidates),
        "admissible_candidate_count": len(candidates),
        "filtered_count_by_stage": {"local_safety": max(0, len(query.candidates) - len(candidates))},
        "fallback_used": bool(query.fallback_used),
        "fallback_reason": query.fallback_reason,
        "runtime_ms": _elapsed_ms(start_s),
        "returnability_filter_enabled": False,
        "degradation_enabled": False,
        "local_safety_filter_enabled": bool(local_safety_filter),
        "objective_scores": [
            {
                "candidate_id": candidate.primitive_id,
                "score": _objective_score(candidate, request.objective_score_terms),
            }
            for candidate in sorted(candidates, key=lambda item: item.primitive_id)
        ],
        "request_summary": request.to_record(),
    }
    return _MethodDecision(
        method_name=method_name,
        selected_candidate_id=None if selected is None else selected.primitive_id,
        selected_primitive_id=None if selected is None else selected.primitive_id,
        fallback_used=bool(query.fallback_used),
        fallback_reason=query.fallback_reason,
        decision_record=record,
    )


def _decision_from_governor_record(method_name: str, record: dict[str, object]) -> _MethodDecision:
    return _MethodDecision(
        method_name=method_name,
        selected_candidate_id=_optional_str(record.get("selected_candidate_id")),
        selected_primitive_id=_optional_str(record.get("selected_primitive_id")),
        fallback_used=bool(record.get("fallback_used", False)),
        fallback_reason=_optional_str(record.get("fallback_reason")),
        decision_record=dict(record),
    )


def _primitive_rollout_record(
    case: RandomUpdraftChallengeCase,
    decision: _MethodDecision,
    library: PrimitiveLibrary,
    config: FirstFullSimulationConfig,
) -> RandomUpdraftChallengeMethodRecord:
    primitive_id, primitive = _selected_primitive(library, decision)
    if primitive is None:
        reason = "no_selected_primitive" if primitive_id is None else "missing_selected_primitive"
        return _non_rollout_record(case, decision, reason, config)

    rollout_config = PrimitiveRolloutConfig(
        dt_s=float(config.dt_s),
        wind_mode=case.wind_mode,
        max_duration_s=float(config.max_duration_s),
        scenario_id=f"{case.case_id}:{decision.method_name}",
        seed=case.seed,
    )
    result = rollout_primitive(
        primitive=primitive,
        initial_state=case.initial_state,
        task=case.task,
        wind_model=case.wind_model,
        config=rollout_config,
    )
    evidence = result.evidence
    metrics = result.metrics
    record = dict(decision.decision_record)
    record["actual_successor_class"] = _actual_successor_class(library, result.states[-1])
    return RandomUpdraftChallengeMethodRecord(
        case_id=case.case_id,
        case_set=case.case_set,
        environment_family=case.environment_family,
        selector_name=decision.method_name,
        selected_candidate_id=decision.selected_candidate_id,
        selected_primitive_id=primitive_id,
        rollout_success=bool(evidence.rollout_success),
        gate_crossed=None if metrics is None else bool(metrics.gate_crossed),
        gate_miss_distance_m=_optional_float(evidence.gate_miss_distance_m),
        min_safety_margin_m=float(evidence.min_safety_margin_m),
        terminal_specific_energy_margin_j_kg=_optional_float(evidence.terminal_specific_energy_margin_j_kg),
        terminal_specific_energy_change_j_kg=float(evidence.terminal_specific_energy_change_j_kg),
        max_angle_of_attack_rad=float(evidence.max_angle_of_attack_rad),
        max_command_abs_rad=float(evidence.max_command_abs_rad),
        failure_reason=evidence.failure_reason,
        fallback_used=decision.fallback_used,
        fallback_reason=decision.fallback_reason,
        decision_record=record,
        scenario_seed=case.seed,
        controller_config_id=f"{decision.method_name}_first_full_simulation",
        primitive_library_id=_primitive_library_id(library),
        returnability_graph_id="first_full_simulation_returnability_graph",
        useful_lift_exposure=_optional_float(record.get("predicted_lift_exposure")),
        runtime_ms=_optional_float(record.get("runtime_ms")) or 0.0,
        filtered_count_by_stage=record.get("filtered_count_by_stage")
        if isinstance(record.get("filtered_count_by_stage"), dict)
        else {},
    )


def _reference_tracking_record(
    case: RandomUpdraftChallengeCase,
    config: FirstFullSimulationConfig,
) -> RandomUpdraftChallengeMethodRecord:
    start_s = perf_counter()
    tracking_record = run_reference_tracking_rollout(
        initial_state=case.initial_state,
        task=case.task,
        wind_model=case.wind_model,
        config=ReferenceTrackingConfig(
            dt_s=float(config.dt_s),
            horizon_s=float(config.reference_horizon_s),
            wind_mode=case.wind_mode,
        ),
    ).to_record()
    terminal_margin = _optional_float(tracking_record["terminal_specific_energy_margin_j_kg"])
    terminal_energy_change = None
    if terminal_margin is not None:
        terminal_energy_change = (
            terminal_margin
            + float(case.task.required_terminal_specific_energy_j_kg)
            - specific_energy_j_kg(case.initial_state)
        )
    controller_failed = bool(tracking_record["controller_failed"])
    failure_reason = _optional_str(tracking_record["controller_failure_reason"]) or _optional_str(
        tracking_record["failure_reason"]
    )
    decision_record = {
        **tracking_record,
        "decision_type": "baseline_rollout",
        "request_id": f"baseline:trajectory_tracking_lqr_tvlqr:{case.case_id}",
        "requested_primitive_id": None,
        "degradation_level": 0,
        "rejection_reasons": [],
        "active_constraints": [],
        "runtime_ms": _elapsed_ms(start_s),
        "candidate_count": 0,
        "admissible_candidate_count": 0,
        "filtered_count_by_stage": {},
        "implementation_limit": "implemented gate reference-tracking rollout adapter for the LQR/TVLQR benchmark slot",
    }
    return RandomUpdraftChallengeMethodRecord(
        case_id=case.case_id,
        case_set=case.case_set,
        environment_family=case.environment_family,
        selector_name="trajectory_tracking_lqr_tvlqr",
        selected_candidate_id=None,
        selected_primitive_id=None,
        rollout_success=bool(tracking_record["gate_success"]) and not controller_failed,
        gate_crossed=_optional_bool(tracking_record["gate_crossed"]),
        gate_miss_distance_m=_optional_float(tracking_record["gate_miss_distance_m"]),
        min_safety_margin_m=_optional_float(tracking_record["min_safety_margin_m"]),
        terminal_specific_energy_margin_j_kg=terminal_margin,
        terminal_specific_energy_change_j_kg=terminal_energy_change,
        max_angle_of_attack_rad=_optional_float(tracking_record["max_angle_of_attack_rad"]),
        max_command_abs_rad=None,
        failure_reason=failure_reason,
        fallback_used=False,
        fallback_reason=None,
        decision_record=decision_record,
        scenario_seed=case.seed,
        controller_config_id="trajectory_tracking_lqr_tvlqr_first_full_simulation",
    )


def _non_rollout_record(
    case: RandomUpdraftChallengeCase,
    decision: _MethodDecision,
    reason: str,
    config: FirstFullSimulationConfig,
) -> RandomUpdraftChallengeMethodRecord:
    return RandomUpdraftChallengeMethodRecord(
        case_id=case.case_id,
        case_set=case.case_set,
        environment_family=case.environment_family,
        selector_name=decision.method_name,
        selected_candidate_id=decision.selected_candidate_id,
        selected_primitive_id=decision.selected_primitive_id,
        rollout_success=False,
        gate_crossed=None,
        gate_miss_distance_m=None,
        min_safety_margin_m=None,
        terminal_specific_energy_margin_j_kg=None,
        terminal_specific_energy_change_j_kg=None,
        max_angle_of_attack_rad=None,
        max_command_abs_rad=None,
        failure_reason=reason,
        fallback_used=decision.fallback_used,
        fallback_reason=decision.fallback_reason,
        decision_record=decision.decision_record,
        scenario_seed=case.seed,
        controller_config_id=f"{decision.method_name}_first_full_simulation",
        runtime_ms=_optional_float(decision.decision_record.get("runtime_ms")) or 0.0,
        filtered_count_by_stage=decision.decision_record.get("filtered_count_by_stage")
        if isinstance(decision.decision_record.get("filtered_count_by_stage"), dict)
        else {},
    )


def _selected_primitive(
    library: PrimitiveLibrary,
    decision: _MethodDecision,
) -> tuple[str | None, object | None]:
    candidate_index = getattr(getattr(library, "compressed", None), "candidate_index", {})
    if not isinstance(candidate_index, dict):
        return decision.selected_primitive_id, None
    for selected_id in _unique_ordered((decision.selected_primitive_id, decision.selected_candidate_id)):
        primitive = candidate_index.get(selected_id)
        if primitive is not None:
            return selected_id, primitive
    return decision.selected_primitive_id or decision.selected_candidate_id, None


def _actual_successor_class(library: PrimitiveLibrary, state: object) -> str | None:
    classifier = getattr(library, "classifier", None)
    if classifier is None or not hasattr(classifier, "classify"):
        return None
    return str(classifier.classify(state).label)


def _objective_score(candidate: PrimitiveLibraryCandidate, terms: dict[str, float]) -> float:
    gate_miss = _feature_float(candidate, "gate_miss_distance_m") or 0.0
    energy = _feature_float(candidate, "terminal_specific_energy_change_j_kg") or 0.0
    lift = _feature_float(candidate, "mean_positive_vertical_wind_m_s") or 0.0
    safety = _feature_float(candidate, "min_safety_margin_m") or 0.0
    score = 0.0
    score += float(terms.get("gate_alignment", 0.0)) * -gate_miss
    score += float(terms.get("terminal_energy", 0.0)) * energy
    score += float(terms.get("terminal_energy_opportunity", 0.0)) * energy
    score += float(terms.get("useful_lift_exposure", 0.0)) * lift
    score += float(terms.get("safety_margin", 0.0)) * safety
    return float(score)


def _candidate_aggressiveness(candidate: PrimitiveLibraryCandidate) -> int:
    for key in ("aggressiveness_level", "requested_aggressiveness"):
        value = candidate.feature_record.get(key)
        if isinstance(value, int):
            return max(0, int(value))
        if isinstance(value, float) and math.isfinite(value):
            return max(0, int(value))
    return 0


def _local_safety_ok(candidate: PrimitiveLibraryCandidate, config: FirstFullSimulationConfig) -> bool:
    safety = _feature_float(candidate, "min_safety_margin_m")
    alpha = _feature_float(candidate, "max_angle_of_attack_rad")
    command = _feature_float(candidate, "max_command_abs_rad")
    if safety is None or alpha is None or command is None:
        return False
    return (
        safety >= float(config.local_safety_min_margin_m)
        and alpha <= float(config.local_safety_max_angle_of_attack_rad)
        and command <= float(config.local_safety_max_command_abs_rad)
    )


def _feature_float(candidate: PrimitiveLibraryCandidate, key: str) -> float | None:
    value = candidate.feature_record.get(key)
    return _optional_float(value)


def _method_capability_audit(methods: tuple[str, ...]) -> dict[str, object]:
    requested = set(methods)
    records: list[dict[str, object]] = []
    for method in FIRST_FULL_SIMULATION_METHODS:
        records.append(
            {
                "method_name": method,
                "requested": method in requested,
                "scaffold_only": False,
                "eligible_for_performance_claims": method in requested,
                "exclusion_reason": None if method in requested else "not_requested",
                "semantics": _method_semantics(method),
            }
        )
    for method in SCAFFOLD_ONLY_METHODS:
        records.append(
            {
                "method_name": method,
                "requested": method in requested,
                "scaffold_only": True,
                "eligible_for_performance_claims": False,
                "exclusion_reason": "scaffold_only_rollout_logic_not_implemented",
                "semantics": _method_semantics(method),
            }
        )
    excluded = [
        str(record["method_name"])
        for record in records
        if bool(record["scaffold_only"]) and (bool(record["requested"]) or record["method_name"] in SCAFFOLD_ONLY_METHODS)
    ]
    return {
        "records": records,
        "performance_methods": [method for method in methods if method not in SCAFFOLD_ONLY_METHODS],
        "excluded_scaffold_methods": excluded,
    }


def _method_semantics(method_name: str) -> str:
    return {
        "governor": "manoeuvre primitive governor with returnability filtering and degradation enabled",
        "ungoverned_primitive_selector": "same primitive library and objective scoring without returnability filtering or degradation",
        "filter_only_no_degradation": "governor filters enabled with structural degradation disabled",
        "no_returnability_selector": "local safety and objective scoring while ignoring successor returnability",
        "trajectory_tracking_lqr_tvlqr": "implemented reference-tracking rollout path for the tracking baseline slot",
        "lift_evidence_removed": "governed path with useful lift and terminal energy opportunity terms zeroed in the request",
        "open_loop_diagnostic": "scaffold-only diagnostic, excluded from performance records",
        "energy_lateral_guidance": "scaffold-only guidance baseline, excluded from performance records",
        "wind_aware_guidance": "scaffold-only wind-aware guidance baseline, excluded from performance records",
        "inner_loop_substitution": "scaffold-only inner-loop ablation, excluded from performance records",
    }[method_name]


def _validate_first_full_records(
    records: tuple[RandomUpdraftChallengeMethodRecord, ...],
    config: FirstFullSimulationConfig,
) -> None:
    materialised = [record.to_record() for record in records]
    if len(materialised) != config.expected_record_count:
        raise ValueError(
            f"Expected {config.expected_record_count} method-case records, got {len(materialised)}."
        )
    required = set(BASELINE_COMMON_SCHEMA_FIELDS)
    for record in materialised:
        missing = required - set(record)
        if missing:
            raise ValueError(f"Result record missing common schema fields: {sorted(missing)}")
        decision = record.get("decision_record")
        if not isinstance(decision, dict):
            raise ValueError("Every result record requires a nested decision_record.")
        if record["method_name"] in _GOVERNED_METHODS:
            for field in (
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
                "runtime_ms",
                "candidate_count",
                "admissible_candidate_count",
                "filtered_count_by_stage",
            ):
                if field not in record:
                    raise ValueError(f"Governed result missing decision-centric field: {field}")
            if record["method_name"] == "governor" and (
                record.get("request_id") is None or record.get("decision_type") is None
            ):
                raise ValueError("Main governor records must include PrimitiveRequest and decision type fields.")
        if record["method_name"] == "filter_only_no_degradation" and int(record["degradation_level"]) > 0:
            raise ValueError("filter_only_no_degradation emitted a positive degradation level.")
        if record["method_name"] == "lift_evidence_removed":
            removed = decision.get("removed_objective_terms")
            if not isinstance(removed, list) or "useful_lift_exposure" not in removed:
                raise ValueError("lift_evidence_removed must record removed lift or energy objective terms.")


def _summary_by_method(
    records: list[dict[str, object]],
    method_audit: dict[str, object],
) -> dict[str, object]:
    methods = _unique_ordered(str(record["method_name"]) for record in records)
    method_summaries = {method: _record_summary(_records_for(records, method_name=method)) for method in methods}
    paired = _paired_success_deltas(records)
    return {
        "total_method_case_record_count": len(records),
        "record_count_by_method": {method: method_summaries[method]["record_count"] for method in methods},
        "methods": method_summaries,
        "excluded_scaffold_methods": method_audit["excluded_scaffold_methods"],
        "paired_comparisons": paired,
        "governor_vs_ungoverned_same_case_success_delta": paired["governor_vs_ungoverned_same_case_success_delta"],
        "governor_vs_no_returnability_same_case_success_delta": paired[
            "governor_vs_no_returnability_same_case_success_delta"
        ],
        "governor_vs_filter_only_same_case_success_delta": paired[
            "governor_vs_filter_only_same_case_success_delta"
        ],
        "governor_vs_tracking_same_case_success_delta": paired[
            "governor_vs_tracking_same_case_success_delta"
        ],
    }


def _summary_by_case_family(records: list[dict[str, object]]) -> dict[str, object]:
    families = _unique_ordered(str(record["case_family"]) for record in records)
    return {
        family: {
            "record_count": len(_records_for(records, case_family=family)),
            "method_summaries": {
                method: _record_summary(_records_for(records, method_name=method, case_family=family))
                for method in _unique_ordered(str(record["method_name"]) for record in _records_for(records, case_family=family))
            },
        }
        for family in families
    }


def _record_summary(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        return {"record_count": 0}
    by_family: dict[str, dict[str, object]] = {}
    for family in _unique_ordered(str(record["case_family"]) for record in records):
        family_records = _records_for(records, case_family=family)
        by_family[family] = {
            "record_count": len(family_records),
            "success_rate": _rate(record.get("rollout_success") is True for record in family_records),
            "gate_crossed_rate": _rate(record.get("gate_crossed") is True for record in family_records),
        }
    return {
        "record_count": len(records),
        "case_count": len({record["case_id"] for record in records}),
        "case_family_count": {family: len(_records_for(records, case_family=family)) for family in by_family},
        "success_rate": _rate(record.get("rollout_success") is True for record in records),
        "success_rate_by_case_family": by_family,
        "gate_crossed_rate": _rate(record.get("gate_crossed") is True for record in records),
        "gate_miss_distance_m": _numeric_summary(record.get("gate_miss_distance_m") for record in records),
        "min_safety_margin_m": _numeric_summary(record.get("min_safety_margin_m") for record in records),
        "terminal_specific_energy_change_j_kg": _numeric_summary(
            record.get("terminal_specific_energy_change_j_kg") for record in records
        ),
        "useful_lift_exposure": _numeric_summary(record.get("useful_lift_exposure") for record in records),
        "failure_mode_histogram": _histogram(
            _optional_str(record.get("failure_mode") or record.get("failure_reason")) for record in records
        ),
        "decision_type_histogram": _histogram(_optional_str(record.get("decision_type")) for record in records),
        "degradation_level_histogram": _histogram(str(record.get("degradation_level", 0)) for record in records),
        "degradation_stage_histogram": _histogram(
            _optional_str(_decision(record).get("degradation_stage")) for record in records
        ),
        "rejection_reason_histogram": _histogram(
            str(reason)
            for record in records
            for reason in _as_list(record.get("rejection_reasons"))
        ),
        "runtime_ms": _numeric_summary(record.get("runtime_ms") for record in records),
        "candidate_count": _numeric_summary(record.get("candidate_count") for record in records),
        "admissible_candidate_count": _numeric_summary(
            record.get("admissible_candidate_count") for record in records
        ),
    }


def _paired_success_deltas(records: list[dict[str, object]]) -> dict[str, object]:
    pairs = {
        "governor_vs_ungoverned_same_case_success_delta": "ungoverned_primitive_selector",
        "governor_vs_no_returnability_same_case_success_delta": "no_returnability_selector",
        "governor_vs_filter_only_same_case_success_delta": "filter_only_no_degradation",
        "governor_vs_tracking_same_case_success_delta": "trajectory_tracking_lqr_tvlqr",
    }
    by_key = {
        (record["case_id"], record["scenario_seed"], record["method_name"]): record
        for record in records
    }
    result: dict[str, object] = {}
    for field, baseline in pairs.items():
        deltas: list[int] = []
        for record in records:
            if record["method_name"] != "governor":
                continue
            other = by_key.get((record["case_id"], record["scenario_seed"], baseline))
            if other is None:
                continue
            deltas.append(int(record.get("rollout_success") is True) - int(other.get("rollout_success") is True))
        result[field] = {
            "paired_record_count": len(deltas),
            "paired_on": ["case_id", "scenario_seed"],
            "mean_success_delta": None if not deltas else float(sum(deltas) / len(deltas)),
            "success_delta_sum": int(sum(deltas)),
        }
    return result


def _failure_taxonomy(records: list[dict[str, object]]) -> dict[str, object]:
    methods = _unique_ordered(str(record["method_name"]) for record in records)
    families = _unique_ordered(str(record["case_family"]) for record in records)
    failed_records = [record for record in records if record.get("rollout_success") is not True]
    return {
        "record_count": len(records),
        "failed_record_count": len(failed_records),
        "histogram_denominator": "failed records with non-null failure_mode or failure_reason",
        "global_failure_mode_histogram": _histogram(
            _optional_str(record.get("failure_mode") or record.get("failure_reason")) for record in records
        ),
        "by_method": {
            method: {
                "record_count": len(_records_for(records, method_name=method)),
                "failed_record_count": len(
                    [
                        record
                        for record in _records_for(records, method_name=method)
                        if record.get("rollout_success") is not True
                    ]
                ),
                "failure_mode_histogram": _histogram(
                    _optional_str(record.get("failure_mode") or record.get("failure_reason"))
                    for record in _records_for(records, method_name=method)
                ),
            }
            for method in methods
        },
        "by_case_family": {
            family: {
                "record_count": len(_records_for(records, case_family=family)),
                "failed_record_count": len(
                    [
                        record
                        for record in _records_for(records, case_family=family)
                        if record.get("rollout_success") is not True
                    ]
                ),
                "failure_mode_histogram": _histogram(
                    _optional_str(record.get("failure_mode") or record.get("failure_reason"))
                    for record in _records_for(records, case_family=family)
                ),
            }
            for family in families
        },
    }


def _runtime_summary(records: list[dict[str, object]]) -> dict[str, object]:
    methods = _unique_ordered(str(record["method_name"]) for record in records)
    return {
        method: {
            "runtime_ms": _numeric_summary(record.get("runtime_ms") for record in _records_for(records, method_name=method)),
            "candidate_count": _numeric_summary(
                record.get("candidate_count") for record in _records_for(records, method_name=method)
            ),
            "admissible_candidate_count": _numeric_summary(
                record.get("admissible_candidate_count") for record in _records_for(records, method_name=method)
            ),
        }
        for method in methods
    }


def _decision_diagnostics(records: list[dict[str, object]]) -> list[dict[str, object]]:
    by_case_method = {(record["case_id"], record["method_name"]): record for record in records}
    diagnostics: list[dict[str, object]] = []
    for record in records:
        if record["method_name"] != "governor":
            continue
        ungoverned = by_case_method.get((record["case_id"], "ungoverned_primitive_selector"))
        decision = _decision(record)
        ungoverned_decision = {} if ungoverned is None else _decision(ungoverned)
        blocked = (
            ungoverned is not None
            and ungoverned.get("selected_primitive_id") is not None
            and ungoverned.get("selected_primitive_id") != record.get("selected_primitive_id")
            and (
                record.get("decision_type") in {"degrade", "reject"}
                or bool(record.get("rejection_reasons"))
            )
        )
        diagnostics.append(
            {
                "case_id": record["case_id"],
                "case_family": record["case_family"],
                "objective_request": decision.get("request_summary"),
                "governor_decision": decision,
                "ungoverned_selector_counterfactual": ungoverned_decision,
                "governor_blocked_ungoverned_selection": bool(blocked),
                "reason": decision.get("degradation_stage") or decision.get("rejection_reason"),
                "predicted_successor_class": record.get("predicted_successor_class"),
                "actual_successor_class": record.get("actual_successor_class"),
            }
        )
    return diagnostics


def _manifest(
    *,
    cfg: FirstFullSimulationConfig,
    guard_record: dict[str, object],
    output_dir: Path,
    records: list[dict[str, object]],
    method_audit: dict[str, object],
    public_repo_root: str | Path | None,
    partial_run: bool,
    stop_reason: str | None,
) -> dict[str, object]:
    version = _repo_version(public_repo_root)
    method_names = _unique_ordered(str(record["method_name"]) for record in records)
    return {
        "run_id": "first_full_simulation_campaign",
        "task": "first_full_simulation_campaign",
        "git_commit": version.get("commit_hash"),
        "branch": version.get("branch_name"),
        "dirty_tree_status": version.get("dirty_tree_status"),
        "controller_frozen": bool(guard_record["controller_frozen"]),
        "write_results": bool(guard_record["write_results"]),
        "runs_full_simulation": True,
        "partial_run": bool(partial_run),
        "stop_reason": stop_reason,
        "output_dir": str(output_dir),
        "public_repo_root": guard_record.get("public_repo_root"),
        "method_names": list(method_names),
        "case_families": list(cfg.case_families),
        "base_seed": int(cfg.base_seed),
        "tier": cfg.tier,
        "wind_mode": cfg.wind_mode,
        "config": cfg.to_record(),
        "guard_record": guard_record,
        "case_count": len({record["case_id"] for record in records}),
        "record_count": len(records),
        "record_count_by_method": _histogram(str(record["method_name"]) for record in records),
        "case_family_count_by_method": _case_family_count_by_method(records),
        "method_capability_audit": method_audit,
        "source_version": version,
        "software_versions": {"python": sys.version},
        "output_files": [
            "manifest.json",
            "resolved_config.json",
            "method_capability_audit.json",
            "records.jsonl",
            "summary_by_method.json",
            "summary_by_case_family.json",
            "decision_diagnostics.json",
            "failure_taxonomy.json",
            "runtime_summary.json",
            "run_notes.md",
        ],
    }


def _write_private_outputs(report: FirstFullSimulationReport) -> None:
    output_dir = report.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "manifest.json", report.manifest)
    _write_json(output_dir / "resolved_config.json", report.config.to_record())
    _write_json(output_dir / "method_capability_audit.json", report.method_capability_audit)
    _write_json(output_dir / "summary_by_method.json", report.summary_by_method)
    _write_json(output_dir / "summary_by_case_family.json", report.summary_by_case_family)
    _write_json(output_dir / "decision_diagnostics.json", report.decision_diagnostics)
    _write_json(output_dir / "failure_taxonomy.json", report.failure_taxonomy)
    _write_json(output_dir / "runtime_summary.json", report.runtime_summary)
    with (output_dir / "records.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in report.records:
            handle.write(json.dumps(record.to_record(), sort_keys=True, separators=(",", ":")) + "\n")
    (output_dir / "run_notes.md").write_text(_run_notes(report), encoding="utf-8")


def _run_notes(report: FirstFullSimulationReport) -> str:
    manifest = report.manifest
    version = manifest["source_version"]
    config = report.config.to_record()
    lines = [
        "# First Full Simulation Run Notes",
        "",
        f"- git commit hash: {version.get('commit_hash')}",
        f"- branch name: {version.get('branch_name')}",
        f"- dirty tree status: {version.get('dirty_tree_status')}",
        f"- exact command used: {config.get('run_command') or 'not recorded by API caller'}",
        f"- tests passed or failed: {config.get('tests_status') or 'not recorded by API caller'}",
        f"- complete or partial: {'partial' if report.partial_run else 'complete'}",
        f"- stop reason: {report.stop_reason}",
        f"- total case count: {report.case_count}",
        f"- total method-case record count: {report.record_count}",
        f"- method list: {', '.join(report.methods)}",
        f"- scaffold-only methods excluded: {', '.join(report.method_capability_audit['excluded_scaffold_methods'])}",
        "",
        "## Resolved Configuration",
        "",
        "```json",
        json.dumps(config, indent=2, sort_keys=True),
        "```",
        "",
        "## Case Family Counts",
        "",
        "```json",
        json.dumps(manifest["case_family_count_by_method"], indent=2, sort_keys=True),
        "```",
        "",
        "## Warnings",
        "",
        "- This first run is decision-centric simulation evidence, not a manuscript-ready statistical claim.",
        "- The trajectory_tracking_lqr_tvlqr method uses the currently implemented reference-tracking rollout adapter.",
        "- Scaffold-only methods are listed in the capability audit and excluded from performance summaries.",
    ]
    return "\n".join(lines) + "\n"


def _repo_version(public_repo_root: str | Path | None) -> dict[str, object]:
    root = _resolve_public_repo_root(public_repo_root)
    if root is None:
        return {"branch_name": None, "commit_hash": None, "dirty_tree_status": None}
    return {
        "branch_name": _git_output(root, ("rev-parse", "--abbrev-ref", "HEAD")),
        "commit_hash": _git_output(root, ("rev-parse", "HEAD")),
        "dirty_tree_status": "dirty" if _git_output(root, ("status", "--short")) else "clean",
    }


def _git_output(root: Path, args: tuple[str, ...]) -> str | None:
    completed = subprocess.run(
        ("git", *args),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    return output or None


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _case_family_count_by_method(records: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    methods = _unique_ordered(str(record["method_name"]) for record in records)
    return {
        method: _histogram(
            str(record["case_family"]) for record in _records_for(records, method_name=method)
        )
        for method in methods
    }


def _records_for(
    records: list[dict[str, object]],
    *,
    method_name: str | None = None,
    case_family: str | None = None,
) -> list[dict[str, object]]:
    selected = records
    if method_name is not None:
        selected = [record for record in selected if record["method_name"] == method_name]
    if case_family is not None:
        selected = [record for record in selected if record["case_family"] == case_family]
    return selected


def _decision(record: dict[str, object]) -> dict[str, object]:
    decision = record.get("decision_record")
    return decision if isinstance(decision, dict) else {}


def _numeric_summary(values: Iterable[object]) -> dict[str, float | int | None]:
    finite = sorted(value for value in (_optional_float(item) for item in values) if value is not None)
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p95": None,
            "p99": None,
        }
    return {
        "count": len(finite),
        "mean": float(sum(finite) / len(finite)),
        "median": float(median(finite)),
        "min": float(finite[0]),
        "max": float(finite[-1]),
        "p95": _percentile(finite, 0.95),
        "p99": _percentile(finite, 0.99),
    }


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * float(fraction)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _histogram(values: Iterable[str | None]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _rate(values: Iterable[bool]) -> float | None:
    materialised = list(values)
    if not materialised:
        return None
    return float(sum(1 for value in materialised if value) / len(materialised))


def _primitive_library_id(library: PrimitiveLibrary) -> str:
    metadata = getattr(getattr(library, "compressed", None), "metadata", {})
    if isinstance(metadata, dict):
        method = metadata.get("compression_method")
        count = metadata.get("retained_candidate_count")
        if method is not None and count is not None:
            return f"{method}:{count}"
    return "first_full_simulation_library"


def _elapsed_ms(start_s: float) -> float:
    return max(0.0, (perf_counter() - start_s) * 1000.0)


def _as_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None:
        return []
    return [value]


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_bool(value: object) -> bool | None:
    return None if value is None else bool(value)


def _unique_ordered(values: Iterable[str | None]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value is None or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


__all__ = [
    "FIRST_FULL_SIMULATION_CASE_FAMILIES",
    "FIRST_FULL_SIMULATION_METHODS",
    "FirstFullSimulationConfig",
    "FirstFullSimulationReport",
    "SCAFFOLD_ONLY_METHODS",
    "require_first_full_simulation_guards",
    "run_first_full_simulation_campaign",
]
