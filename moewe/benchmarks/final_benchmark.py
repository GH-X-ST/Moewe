"""Final benchmark plan and preflight checks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from moewe.baselines import BASELINE_COMMON_SCHEMA_FIELDS, BASELINE_METHOD_NAMES, UnfilteredPrimitiveSelector
from moewe.campaigns import (
    RandomUpdraftChallengeConfig,
    RandomUpdraftChallengeReport,
    build_random_updraft_challenge_cases,
    run_random_updraft_challenge_campaign,
)
from moewe.governor import OnlineGovernor
from moewe.primitives import PrimitiveLibrary

BENCHMARK_METHODS = ("governor", *BASELINE_METHOD_NAMES)
BENCHMARK_CASE_FAMILIES = (
    "weak_random_single_source",
    "hard_random_single_source",
    "random_two_source",
    "random_four_source",
)
BENCHMARK_SCHEMA_FIELDS = BASELINE_COMMON_SCHEMA_FIELDS


@dataclass(frozen=True)
class BenchmarkMethodSpec:
    """One method included in the final benchmark plan."""

    name: str

    def __post_init__(self) -> None:
        if self.name not in BENCHMARK_METHODS:
            raise ValueError(f"Unknown benchmark method: {self.name}")

    def to_record(self) -> dict[str, object]:
        return {"method_name": self.name}


@dataclass(frozen=True)
class BenchmarkCaseSetSpec:
    """Case-family set for the final benchmark plan."""

    case_set: str = "random_challenge_after_freeze"
    case_families: tuple[str, ...] = BENCHMARK_CASE_FAMILIES
    full_case_count: int = 120

    def __post_init__(self) -> None:
        if self.case_set != "random_challenge_after_freeze":
            raise ValueError("Final benchmark case_set must be random_challenge_after_freeze.")
        if int(self.full_case_count) <= 0:
            raise ValueError("full_case_count must be positive.")
        if not self.case_families:
            raise ValueError("At least one case family is required.")
        unknown = sorted(set(self.case_families) - set(BENCHMARK_CASE_FAMILIES))
        if unknown:
            raise ValueError(f"Unknown case families: {unknown}")
        if len(set(self.case_families)) != len(self.case_families):
            raise ValueError("case_families must not contain duplicates.")

    def to_record(self) -> dict[str, object]:
        return {
            "case_set": self.case_set,
            "case_families": list(self.case_families),
            "full_case_count": int(self.full_case_count),
        }


@dataclass(frozen=True)
class FinalBenchmarkConfig:
    """Configuration for final benchmark planning and smoke preflight."""

    tier: str = "balanced"
    methods: tuple[str, ...] = BENCHMARK_METHODS
    case_families: tuple[str, ...] = BENCHMARK_CASE_FAMILIES
    full_case_count: int = 120
    preflight_case_count: int = 4
    base_seed: int = 91
    dt_s: float = 0.01
    max_duration_s: float = 0.10
    reference_horizon_s: float = 0.30
    wind_mode: str = "panel"
    writes_files_by_default: bool = False
    runs_full_simulation: bool = False

    def __post_init__(self) -> None:
        if not self.tier:
            raise ValueError("tier must be non-empty.")
        if not self.methods:
            raise ValueError("At least one benchmark method is required.")
        unknown_methods = sorted(set(self.methods) - set(BENCHMARK_METHODS))
        if unknown_methods:
            raise ValueError(f"Unknown benchmark methods: {unknown_methods}")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("methods must not contain duplicates.")
        BenchmarkCaseSetSpec(case_families=self.case_families, full_case_count=self.full_case_count)
        if int(self.preflight_case_count) <= 0:
            raise ValueError("preflight_case_count must be positive.")
        if int(self.preflight_case_count) > int(self.full_case_count):
            raise ValueError("preflight_case_count must not exceed full_case_count.")
        if int(self.base_seed) < 0:
            raise ValueError("base_seed must be non-negative.")
        for name in ("dt_s", "max_duration_s", "reference_horizon_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        if self.wind_mode not in {"cg", "panel"}:
            raise ValueError("wind_mode must be 'cg' or 'panel'.")
        if self.writes_files_by_default:
            raise ValueError("Final benchmark preflight must not write files by default.")
        if self.runs_full_simulation:
            raise ValueError("Final benchmark preflight must not run full simulation.")

    @property
    def method_count(self) -> int:
        return len(self.methods)

    @property
    def expected_full_record_count(self) -> int:
        return int(self.full_case_count) * self.method_count

    @property
    def expected_preflight_record_count(self) -> int:
        return int(self.preflight_case_count) * self.method_count

    def to_record(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "methods": list(self.methods),
            "case_families": list(self.case_families),
            "full_case_count": int(self.full_case_count),
            "preflight_case_count": int(self.preflight_case_count),
            "base_seed": int(self.base_seed),
            "dt_s": float(self.dt_s),
            "max_duration_s": float(self.max_duration_s),
            "reference_horizon_s": float(self.reference_horizon_s),
            "wind_mode": self.wind_mode,
            "writes_files_by_default": bool(self.writes_files_by_default),
            "runs_full_simulation": bool(self.runs_full_simulation),
            "expected_full_record_count": self.expected_full_record_count,
            "expected_preflight_record_count": self.expected_preflight_record_count,
        }


@dataclass(frozen=True)
class FinalBenchmarkPlan:
    """Deterministic final benchmark plan without generated results."""

    config: FinalBenchmarkConfig
    methods: tuple[BenchmarkMethodSpec, ...]
    case_set: BenchmarkCaseSetSpec
    schema_fields: tuple[str, ...] = BENCHMARK_SCHEMA_FIELDS
    full_simulation_executed: bool = False
    writes_files_by_default: bool = False

    @property
    def expected_full_record_count(self) -> int:
        return self.config.expected_full_record_count

    @property
    def expected_preflight_record_count(self) -> int:
        return self.config.expected_preflight_record_count

    def to_record(self) -> dict[str, object]:
        return {
            "config": self.config.to_record(),
            "methods": [method.to_record() for method in self.methods],
            "case_set": self.case_set.to_record(),
            "schema_fields": list(self.schema_fields),
            "expected_full_record_count": self.expected_full_record_count,
            "expected_preflight_record_count": self.expected_preflight_record_count,
            "full_simulation_executed": bool(self.full_simulation_executed),
            "writes_files_by_default": bool(self.writes_files_by_default),
        }


@dataclass(frozen=True)
class FinalBenchmarkPreflightReport:
    """Preflight report for final benchmark readiness."""

    plan: FinalBenchmarkPlan
    preflight_report: RandomUpdraftChallengeReport
    ready_for_full_simulation: bool
    full_simulation_executed: bool = False
    writes_files_by_default: bool = False

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(method.name for method in self.plan.methods)

    @property
    def case_families(self) -> tuple[str, ...]:
        return self.plan.case_set.case_families

    @property
    def preflight_case_count(self) -> int:
        return len({record.case_id for record in self.preflight_report.records})

    @property
    def preflight_record_count(self) -> int:
        return len(self.preflight_report.records)

    def to_summary(self) -> dict[str, object]:
        return {
            "ready_for_full_simulation": bool(self.ready_for_full_simulation),
            "full_simulation_executed": bool(self.full_simulation_executed),
            "writes_files_by_default": bool(self.writes_files_by_default),
            "method_names": list(self.methods),
            "case_families": list(self.case_families),
            "expected_full_record_count": self.plan.expected_full_record_count,
            "preflight_case_count": self.preflight_case_count,
            "preflight_record_count": self.preflight_record_count,
            "schema_fields": list(self.plan.schema_fields),
        }

    def to_record(self) -> dict[str, object]:
        return {
            "ready_for_full_simulation": bool(self.ready_for_full_simulation),
            "full_simulation_executed": bool(self.full_simulation_executed),
            "writes_files_by_default": bool(self.writes_files_by_default),
            "method_names": list(self.methods),
            "case_families": list(self.case_families),
            "expected_full_record_count": self.plan.expected_full_record_count,
            "preflight_case_count": self.preflight_case_count,
            "preflight_record_count": self.preflight_record_count,
            "schema_fields": list(self.plan.schema_fields),
            "config": self.plan.config.to_record(),
            "plan": self.plan.to_record(),
            "preflight_summary": self.preflight_report.to_summary(),
        }


def build_final_benchmark_plan(config: FinalBenchmarkConfig | None = None) -> FinalBenchmarkPlan:
    """Build the final benchmark plan without executing the full benchmark."""

    cfg = FinalBenchmarkConfig() if config is None else config
    return FinalBenchmarkPlan(
        config=cfg,
        methods=tuple(BenchmarkMethodSpec(name) for name in cfg.methods),
        case_set=BenchmarkCaseSetSpec(case_families=cfg.case_families, full_case_count=cfg.full_case_count),
    )


def run_final_benchmark_preflight(
    governor: OnlineGovernor,
    selector: UnfilteredPrimitiveSelector,
    library: PrimitiveLibrary,
    config: FinalBenchmarkConfig | None = None,
) -> FinalBenchmarkPreflightReport:
    """Run the small preflight sample and stop before the full benchmark."""

    cfg = FinalBenchmarkConfig() if config is None else config
    plan = build_final_benchmark_plan(cfg)
    cases = _preflight_cases(cfg)
    random_config = RandomUpdraftChallengeConfig(
        case_count=len(cases),
        base_seed=int(cfg.base_seed),
        tier=cfg.tier,
        selectors=cfg.methods,
        dt_s=float(cfg.dt_s),
        max_duration_s=float(cfg.max_duration_s),
        reference_horizon_s=float(cfg.reference_horizon_s),
        wind_mode=cfg.wind_mode,
    )
    preflight_report = run_random_updraft_challenge_campaign(cases, governor, selector, library, random_config)
    schema_ok = _records_match_schema(preflight_report, plan.schema_fields)
    expected_count_ok = len(preflight_report.records) == cfg.expected_preflight_record_count
    case_count_ok = len({record.case_id for record in preflight_report.records}) == int(cfg.preflight_case_count)
    return FinalBenchmarkPreflightReport(
        plan=plan,
        preflight_report=preflight_report,
        ready_for_full_simulation=schema_ok and expected_count_ok and case_count_ok,
    )


def require_full_benchmark_guards(
    *,
    controller_frozen: bool,
    write_results: bool,
    output_dir: str | Path | None,
    public_repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Validate explicit full-benchmark guards without running the campaign."""

    if not controller_frozen:
        raise ValueError("Full benchmark execution requires controller_frozen=True.")
    if not write_results:
        raise ValueError("Full benchmark execution requires write_results=True.")
    if output_dir is None:
        raise ValueError("Full benchmark execution requires an explicit output_dir.")
    output_path = Path(output_dir).resolve()
    if public_repo_root is not None:
        repo_root = Path(public_repo_root).resolve()
        if output_path == repo_root or repo_root in output_path.parents:
            raise ValueError("Full benchmark output_dir must be outside the public repository.")
    return {
        "controller_frozen": True,
        "write_results": True,
        "output_dir": str(output_path),
        "runs_full_simulation": False,
        "guard_only": True,
    }


def _preflight_cases(config: FinalBenchmarkConfig) -> tuple[object, ...]:
    generated_count = len(BENCHMARK_CASE_FAMILIES) * int(config.preflight_case_count)
    random_config = RandomUpdraftChallengeConfig(
        case_count=generated_count,
        base_seed=int(config.base_seed),
        tier=config.tier,
        selectors=config.methods,
        dt_s=float(config.dt_s),
        max_duration_s=float(config.max_duration_s),
        reference_horizon_s=float(config.reference_horizon_s),
        wind_mode=config.wind_mode,
    )
    candidates = build_random_updraft_challenge_cases(random_config)
    selected = [case for case in candidates if case.environment_family in set(config.case_families)]
    if len(selected) < int(config.preflight_case_count):
        raise ValueError("Could not build enough preflight cases for the requested case families.")
    return tuple(selected[: int(config.preflight_case_count)])


def _records_match_schema(report: RandomUpdraftChallengeReport, schema_fields: tuple[str, ...]) -> bool:
    for record in report.records:
        materialised = record.to_record()
        if any(field not in materialised for field in schema_fields):
            return False
    return True


__all__ = [
    "BenchmarkCaseSetSpec",
    "BenchmarkMethodSpec",
    "FinalBenchmarkConfig",
    "FinalBenchmarkPlan",
    "FinalBenchmarkPreflightReport",
    "build_final_benchmark_plan",
    "require_full_benchmark_guards",
    "run_final_benchmark_preflight",
]
