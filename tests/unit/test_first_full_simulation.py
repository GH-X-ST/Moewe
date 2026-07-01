from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

import moewe.benchmarks.full_simulation as full_simulation
from moewe.benchmarks import (
    FIRST_FULL_SIMULATION_METHODS,
    FirstFullSimulationConfig,
    require_first_full_simulation_guards,
    run_first_full_simulation_campaign,
)
from moewe.campaigns import RandomUpdraftChallengeCase
from moewe.primitives import PrimitiveEvidence, PrimitiveLibraryCandidate, PrimitiveLibraryQuery, PrimitiveRolloutResult
from moewe.returnability import PrimitiveTransition, ReturnabilityGraph
from moewe.sim.state import FlightState
from moewe.tasks import FlightVolume, GatePlane, GateTraversalTask


@pytest.fixture
def external_output_root(request: pytest.FixtureRequest) -> Path:
    root = Path.cwd().parent / ".pytest_moewe_first_full_simulation" / request.node.name
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@dataclass(frozen=True)
class _CompressedStub:
    candidate_index: dict[str, object]
    metadata: dict[str, object]


@dataclass(frozen=True)
class _LibraryStub:
    query_result: PrimitiveLibraryQuery
    compressed: _CompressedStub

    def query(self, state: FlightState, tier: str = "balanced") -> PrimitiveLibraryQuery:
        assert state.finite()
        assert tier == self.query_result.tier
        return self.query_result


def _feature(
    *,
    energy: float = 1.0,
    safety: float = 1.0,
    gate_miss: float = 0.0,
    lift: float = 0.0,
    alpha: float = 0.1,
    command: float = 0.1,
    aggressiveness: int = 1,
) -> dict[str, object]:
    return {
        "terminal_specific_energy_change_j_kg": energy,
        "min_safety_margin_m": safety,
        "gate_miss_distance_m": gate_miss,
        "mean_positive_vertical_wind_m_s": lift,
        "max_angle_of_attack_rad": alpha,
        "max_command_abs_rad": command,
        "aggressiveness_level": aggressiveness,
    }


def _candidate(
    primitive_id: str,
    *,
    exit_class: str = "terminal",
    feature: dict[str, object] | None = None,
) -> PrimitiveLibraryCandidate:
    return PrimitiveLibraryCandidate(
        primitive_id=primitive_id,
        family="bank_pitch_dwell_recovery",
        controller_type="pd",
        entry_class="entry",
        exit_class=exit_class,
        represented_primitive_ids=(primitive_id,),
        feature_record=_feature() if feature is None else feature,
    )


def _transition(
    primitive_id: str,
    exit_class: str,
    *,
    retained: bool = True,
    rollout_success: bool = True,
    failure_reason: str | None = None,
) -> PrimitiveTransition:
    return PrimitiveTransition(
        primitive_id=primitive_id,
        design_case_id="design",
        case_set="library_design",
        family="bank_pitch_dwell_recovery",
        controller_type="pd",
        entry_class="entry",
        exit_class=exit_class,
        retained=retained,
        rollout_success=rollout_success,
        min_safety_margin_m=1.0,
        terminal_specific_energy_change_j_kg=1.0,
        terminal_specific_energy_margin_j_kg=None,
        max_angle_of_attack_rad=0.1,
        max_command_abs_rad=0.1,
        failure_reason=failure_reason,
        retention_reason="retained" if retained else "rollout_failure",
        scenario_id="design",
    )


def _library_and_graph() -> tuple[_LibraryStub, ReturnabilityGraph]:
    hot = _candidate(
        "prim_hot",
        exit_class="forbidden",
        feature=_feature(energy=10.0, lift=2.0, aggressiveness=2),
    )
    mild = _candidate(
        "prim_mild",
        exit_class="terminal",
        feature=_feature(energy=1.0, lift=0.0, aggressiveness=0),
    )
    graph = ReturnabilityGraph(
        transitions=(
            _transition("prim_hot", "forbidden", rollout_success=False, failure_reason="wall"),
            _transition("prim_mild", "terminal"),
        ),
        safe_classes=frozenset({"entry", "terminal"}),
        forbidden_classes=frozenset({"forbidden"}),
        terminal_success_classes=frozenset({"terminal"}),
        recoverable_classes=frozenset({"entry", "terminal"}),
        dead_end_classes=frozenset(),
        entry_supported_classes=frozenset({"entry"}),
        exit_observed_classes=frozenset({"terminal", "forbidden"}),
    )
    library = _LibraryStub(
        query_result=PrimitiveLibraryQuery(
            tier="balanced",
            entry_class="entry",
            candidates=(hot, mild),
            fallback_used=False,
            fallback_reason=None,
            available_entry_classes=("entry",),
        ),
        compressed=_CompressedStub(
            candidate_index={},
            metadata={"compression_method": "stub", "retained_candidate_count": 2},
        ),
    )
    return library, graph


def _mission_state(x_m: float) -> FlightState:
    return FlightState(
        position_w_m=np.array([x_m, 0.0, 1.0]),
        euler_rad=np.zeros(3),
        velocity_b_m_s=np.array([5.0, 0.0, 0.0]),
        rates_b_rad_s=np.zeros(3),
        surfaces_rad=np.zeros(3),
    )


def _mission_case() -> RandomUpdraftChallengeCase:
    task = GateTraversalTask(
        gate=GatePlane(
            centre_w_m=np.array([0.25, 0.0, 1.0]),
            normal_w=np.array([1.0, 0.0, 0.0]),
            width_m=2.0,
            height_m=2.0,
        ),
        flight_volume=FlightVolume(
            x_min_m=-1.0,
            x_max_m=0.25,
            y_min_m=-1.0,
            y_max_m=1.0,
            z_min_m=0.2,
            z_max_m=2.0,
        ),
        timeout_s=1.0,
        angle_of_attack_limit_rad=1.0,
    )
    return RandomUpdraftChallengeCase(
        case_id="mission_loop_case",
        case_set="random_challenge_after_freeze",
        environment_family="weak_random_single_source",
        seed=7,
        initial_state=_mission_state(0.0),
        task=task,
        wind_model=None,
        wind_mode="panel",
        factor_record={},
    )


def _fake_segment_rollout(**kwargs: object) -> PrimitiveRolloutResult:
    initial_state = kwargs["initial_state"]
    assert isinstance(initial_state, FlightState)
    next_state = FlightState(
        position_w_m=initial_state.position_w_m + np.array([0.1, 0.0, 0.0]),
        euler_rad=initial_state.euler_rad,
        velocity_b_m_s=initial_state.velocity_b_m_s,
        rates_b_rad_s=initial_state.rates_b_rad_s,
        surfaces_rad=initial_state.surfaces_rad,
    )
    config = kwargs["config"]
    evidence = PrimitiveEvidence(
        primitive_id="prim_mild",
        family="bank_pitch_dwell_recovery",
        controller_type="pd",
        rollout_success=True,
        min_safety_margin_m=0.1,
        terminal_specific_energy_change_j_kg=0.0,
        terminal_specific_energy_margin_j_kg=None,
        max_angle_of_attack_rad=0.0,
        max_command_abs_rad=0.0,
        gate_miss_distance_m=None,
        failure_reason=None,
        scenario_id=getattr(config, "scenario_id"),
        seed=getattr(config, "seed"),
        rollout_duration_s=getattr(config, "dt_s"),
    )
    return PrimitiveRolloutResult(
        primitive_id="prim_mild",
        states=[initial_state, next_state],
        commands_rad=np.zeros((1, 3)),
        metrics=None,
        evidence=evidence,
        controller_failed=False,
        failure_reason=None,
    )


def test_first_full_simulation_guards_require_freeze_write_and_external_output(
    external_output_root: Path,
) -> None:
    repo_root = Path.cwd()
    external_output = external_output_root / "first_full"

    with pytest.raises(ValueError):
        require_first_full_simulation_guards(
            controller_frozen=False,
            write_results=True,
            output_dir=external_output,
            public_repo_root=repo_root,
        )
    with pytest.raises(ValueError):
        require_first_full_simulation_guards(
            controller_frozen=True,
            write_results=False,
            output_dir=external_output,
            public_repo_root=repo_root,
        )
    with pytest.raises(ValueError):
        require_first_full_simulation_guards(
            controller_frozen=True,
            write_results=True,
            output_dir=repo_root / "results",
            public_repo_root=repo_root,
        )

    record = require_first_full_simulation_guards(
        controller_frozen=True,
        write_results=True,
        output_dir=external_output,
        public_repo_root=repo_root,
    )

    assert record["runs_full_simulation"] is True
    assert record["output_dir"] == str(external_output.resolve())


def test_first_full_simulation_routes_governor_through_decision_interface(
    external_output_root: Path,
) -> None:
    library, graph = _library_and_graph()
    report = run_first_full_simulation_campaign(
        library=library,
        graph=graph,
        output_dir=external_output_root / "run",
        config=FirstFullSimulationConfig(
            full_case_count=1,
            methods=FIRST_FULL_SIMULATION_METHODS,
            tests_status="unit smoke",
        ),
        controller_frozen=True,
        write_results=True,
        public_repo_root=Path.cwd(),
    )

    assert report.case_count == 1
    assert report.record_count == len(FIRST_FULL_SIMULATION_METHODS)
    governor = next(record for record in report.records if record.selector_name == "governor").to_record()
    assert governor["request_id"] is not None
    assert governor["decision_type"] in {"accept", "degrade", "reject", "rank"}
    assert isinstance(governor["decision_record"]["request_summary"], dict)
    assert governor["decision_record"]["request_summary"]["metadata"]["spatial_memory"] == "off"
    assert (external_output_root / "run" / "manifest.json").exists()
    assert (external_output_root / "run" / "records.jsonl").exists()
    manifest = json.loads((external_output_root / "run" / "manifest.json").read_text(encoding="utf-8"))
    for field in (
        "git_commit",
        "branch",
        "controller_frozen",
        "write_results",
        "runs_full_simulation",
        "case_count",
        "record_count",
        "method_names",
        "case_families",
        "base_seed",
        "tier",
        "wind_mode",
        "public_repo_root",
        "output_dir",
        "partial_run",
    ):
        assert field in manifest
    assert manifest["controller_frozen"] is True
    assert manifest["write_results"] is True
    json.dumps(report.to_summary(), sort_keys=True)


def test_first_full_simulation_repeats_primitive_decisions_until_exit_gate(
    external_output_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library, graph = _library_and_graph()
    library.compressed.candidate_index["prim_mild"] = object()
    monkeypatch.setattr(full_simulation, "_first_full_simulation_cases", lambda config: (_mission_case(),))
    monkeypatch.setattr(full_simulation, "rollout_primitive", _fake_segment_rollout)

    report = run_first_full_simulation_campaign(
        library=library,
        graph=graph,
        output_dir=external_output_root / "run",
        config=FirstFullSimulationConfig(
            full_case_count=1,
            methods=("governor",),
            dt_s=0.1,
            max_duration_s=0.1,
        ),
        controller_frozen=True,
        write_results=True,
        public_repo_root=Path.cwd(),
    )

    record = report.records[0].to_record()
    decision = record["decision_record"]

    assert record["rollout_success"] is True
    assert record["gate_crossed"] is True
    assert decision["mission_rollout"] is True
    assert decision["mission_decision_count"] == 3
    assert decision["selected_primitive_sequence"] == ["prim_mild", "prim_mild", "prim_mild"]


def test_first_full_simulation_ablation_records_have_required_semantics(
    external_output_root: Path,
) -> None:
    library, graph = _library_and_graph()
    report = run_first_full_simulation_campaign(
        library=library,
        graph=graph,
        output_dir=external_output_root / "run",
        config=FirstFullSimulationConfig(
            full_case_count=1,
            methods=("filter_only_no_degradation", "lift_evidence_removed", "no_returnability_selector"),
        ),
        controller_frozen=True,
        write_results=True,
        public_repo_root=Path.cwd(),
    )
    records = {record.selector_name: record.to_record() for record in report.records}

    assert records["filter_only_no_degradation"]["degradation_level"] == 0
    assert records["filter_only_no_degradation"]["decision_record"]["degradation_enabled"] is False
    assert "useful_lift_exposure" in records["lift_evidence_removed"]["decision_record"]["removed_objective_terms"]
    assert records["no_returnability_selector"]["decision_record"]["returnability_filter_enabled"] is False
    assert records["no_returnability_selector"]["decision_record"]["local_safety_filter_enabled"] is True
    taxonomy = json.loads((external_output_root / "run" / "failure_taxonomy.json").read_text(encoding="utf-8"))
    assert taxonomy["record_count"] == report.record_count
    assert "histogram_denominator" in taxonomy


def test_scaffold_only_methods_are_excluded_from_performance_records(
    external_output_root: Path,
) -> None:
    library, graph = _library_and_graph()
    report = run_first_full_simulation_campaign(
        library=library,
        graph=graph,
        output_dir=external_output_root / "run",
        config=FirstFullSimulationConfig(
            full_case_count=1,
            methods=("governor", "open_loop_diagnostic", "wind_aware_guidance"),
        ),
        controller_frozen=True,
        write_results=True,
        public_repo_root=Path.cwd(),
    )

    assert report.record_count == 1
    assert report.methods == ("governor",)
    assert "open_loop_diagnostic" in report.method_capability_summary["excluded_scaffold_methods"]
    assert "wind_aware_guidance" in report.method_capability_summary["excluded_scaffold_methods"]
    assert all(record.selector_name not in {"open_loop_diagnostic", "wind_aware_guidance"} for record in report.records)
    summary = json.loads((external_output_root / "run" / "summary_by_method.json").read_text(encoding="utf-8"))
    paired = summary["paired_comparisons"]["governor_vs_ungoverned_same_case_success_delta"]
    assert paired["paired_on"] == ["case_id", "scenario_seed"]


def test_first_full_simulation_tests_do_not_create_public_output_dirs(
    external_output_root: Path,
) -> None:
    library, graph = _library_and_graph()
    run_first_full_simulation_campaign(
        library=library,
        graph=graph,
        output_dir=external_output_root / "run",
        config=FirstFullSimulationConfig(full_case_count=1, methods=("governor",)),
        controller_frozen=True,
        write_results=True,
        public_repo_root=Path.cwd(),
    )

    for name in ("results", "data", "figures", "videos"):
        assert not (Path.cwd() / name).exists()
