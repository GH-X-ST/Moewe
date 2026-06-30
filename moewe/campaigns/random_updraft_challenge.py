"""Random updraft challenge campaign utilities."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from moewe.baselines import (
    ReferenceTrackingConfig,
    UnfilteredPrimitiveSelector,
    UnfilteredSelectionDecision,
    run_reference_tracking_rollout,
)
from moewe.governor import GovernorDecision, OnlineGovernor
from moewe.primitives import PrimitiveLibrary, PrimitiveRolloutConfig, rollout_primitive
from moewe.sim.state import FlightState
from moewe.sim.updraft import AnnularUpdraft, FanUpdraft
from moewe.tasks import FlightVolume, GatePlane, GateTraversalTask, specific_energy_j_kg

SELECTOR_GOVERNOR = "governor"
SELECTOR_UNFILTERED = "unfiltered"
SELECTOR_REFERENCE_TRACKING = "reference_tracking_pd"
SUPPORTED_SELECTORS = frozenset({SELECTOR_GOVERNOR, SELECTOR_UNFILTERED, SELECTOR_REFERENCE_TRACKING})
RANDOM_CHALLENGE_CASE_SET = "random_challenge_after_freeze"
ENVIRONMENT_FAMILIES = (
    "weak_random_single_source",
    "hard_random_single_source",
    "random_two_source",
    "random_four_source",
)


@dataclass(frozen=True)
class RandomUpdraftChallengeConfig:
    """Configuration for after-freeze random updraft challenge cases."""

    case_count: int = 4
    base_seed: int = 91
    tier: str = "balanced"
    selectors: tuple[str, ...] = (SELECTOR_GOVERNOR, SELECTOR_UNFILTERED, SELECTOR_REFERENCE_TRACKING)
    dt_s: float = 0.01
    max_duration_s: float = 0.10
    reference_horizon_s: float = 0.30
    wind_mode: str = "panel"
    case_set: str = RANDOM_CHALLENGE_CASE_SET

    def __post_init__(self) -> None:
        if int(self.case_count) <= 0:
            raise ValueError("case_count must be positive.")
        if int(self.base_seed) < 0:
            raise ValueError("base_seed must be non-negative.")
        if not self.tier:
            raise ValueError("tier must be non-empty.")
        selectors = tuple(self.selectors)
        if not selectors:
            raise ValueError("At least one selector is required.")
        unknown = sorted(set(selectors) - SUPPORTED_SELECTORS)
        if unknown:
            raise ValueError(f"Unsupported selector names: {unknown}")
        if len(set(selectors)) != len(selectors):
            raise ValueError("selectors must not contain duplicates.")
        for name in ("dt_s", "max_duration_s", "reference_horizon_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        if self.wind_mode not in {"cg", "panel"}:
            raise ValueError("wind_mode must be 'cg' or 'panel'.")
        if self.case_set != RANDOM_CHALLENGE_CASE_SET:
            raise ValueError("Random updraft challenge cases must use the after-freeze case set.")

    def to_record(self) -> dict[str, object]:
        return {
            "case_count": int(self.case_count),
            "base_seed": int(self.base_seed),
            "tier": self.tier,
            "selectors": list(self.selectors),
            "dt_s": float(self.dt_s),
            "max_duration_s": float(self.max_duration_s),
            "reference_horizon_s": float(self.reference_horizon_s),
            "wind_mode": self.wind_mode,
            "case_set": self.case_set,
        }


@dataclass(frozen=True)
class RandomUpdraftChallengeCase:
    """One deterministic after-freeze challenge case."""

    case_id: str
    case_set: str
    environment_family: str
    seed: int
    initial_state: FlightState
    task: GateTraversalTask
    wind_model: AnnularUpdraft | None
    wind_mode: str
    factor_record: dict[str, object]

    def __post_init__(self) -> None:
        for name, value in (
            ("case_id", self.case_id),
            ("case_set", self.case_set),
            ("environment_family", self.environment_family),
            ("wind_mode", self.wind_mode),
        ):
            if not value:
                raise ValueError(f"{name} must be non-empty.")
        if self.case_set != RANDOM_CHALLENGE_CASE_SET:
            raise ValueError("Random updraft challenge cases must use the after-freeze case set.")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative.")
        if not self.initial_state.finite():
            raise ValueError("initial_state must be finite.")
        if self.wind_mode not in {"cg", "panel"}:
            raise ValueError("wind_mode must be 'cg' or 'panel'.")
        self.task.flight_volume.validate()
        if self.wind_model is not None and not self.wind_model.fans:
            raise ValueError("wind_model must contain at least one fan when supplied.")

    @property
    def fan_count(self) -> int:
        return 0 if self.wind_model is None else len(self.wind_model.fans)

    def to_record(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "case_set": self.case_set,
            "environment_family": self.environment_family,
            "seed": int(self.seed),
            "initial_state_vector": _list(self.initial_state.as_vector()),
            "wind_mode": self.wind_mode,
            "fan_count": self.fan_count,
            "updraft_factor_record": _jsonable(self.factor_record),
            "task": _task_record(self.task),
        }


@dataclass(frozen=True)
class RandomUpdraftChallengeMethodRecord:
    """One method result on one random updraft challenge case."""

    case_id: str
    case_set: str
    environment_family: str
    selector_name: str
    selected_candidate_id: str | None
    selected_primitive_id: str | None
    rollout_success: bool
    gate_crossed: bool | None
    gate_miss_distance_m: float | None
    min_safety_margin_m: float | None
    terminal_specific_energy_margin_j_kg: float | None
    terminal_specific_energy_change_j_kg: float | None
    max_angle_of_attack_rad: float | None
    max_command_abs_rad: float | None
    failure_reason: str | None
    fallback_used: bool
    fallback_reason: str | None
    decision_record: dict[str, object]

    def to_record(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "case_set": self.case_set,
            "environment_family": self.environment_family,
            "selector_name": self.selector_name,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_primitive_id": self.selected_primitive_id,
            "rollout_success": bool(self.rollout_success),
            "gate_crossed": self.gate_crossed,
            "gate_miss_distance_m": self.gate_miss_distance_m,
            "min_safety_margin_m": self.min_safety_margin_m,
            "terminal_specific_energy_margin_j_kg": self.terminal_specific_energy_margin_j_kg,
            "terminal_specific_energy_change_j_kg": self.terminal_specific_energy_change_j_kg,
            "max_angle_of_attack_rad": self.max_angle_of_attack_rad,
            "max_command_abs_rad": self.max_command_abs_rad,
            "failure_reason": self.failure_reason,
            "fallback_used": bool(self.fallback_used),
            "fallback_reason": self.fallback_reason,
            "decision_record": _jsonable(self.decision_record),
        }


@dataclass(frozen=True)
class RandomUpdraftChallengeReport:
    """Compact in-memory random updraft challenge report."""

    records: tuple[RandomUpdraftChallengeMethodRecord, ...]
    config: RandomUpdraftChallengeConfig

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("Random updraft challenge report requires at least one record.")

    def to_summary(self) -> dict[str, object]:
        selectors = _unique_ordered(record.selector_name for record in self.records)
        by_selector = {
            selector: _selector_summary(tuple(record for record in self.records if record.selector_name == selector))
            for selector in selectors
        }
        return {
            "case_count": len({record.case_id for record in self.records}),
            "record_count": len(self.records),
            "environment_families": _unique_ordered(record.environment_family for record in self.records),
            "selectors": list(selectors),
            "by_selector": by_selector,
        }

    def to_record(self) -> dict[str, object]:
        return {
            "config": self.config.to_record(),
            "summary": self.to_summary(),
            "records": [record.to_record() for record in self.records],
        }


def build_random_updraft_challenge_cases(
    config: RandomUpdraftChallengeConfig | None = None,
) -> tuple[RandomUpdraftChallengeCase, ...]:
    """Build deterministic after-freeze random updraft challenge cases."""

    cfg = RandomUpdraftChallengeConfig() if config is None else config
    cases: list[RandomUpdraftChallengeCase] = []
    for index in range(int(cfg.case_count)):
        family = ENVIRONMENT_FAMILIES[index % len(ENVIRONMENT_FAMILIES)]
        seed = int(cfg.base_seed) + index
        cases.append(_build_case(family, seed, cfg))
    return tuple(cases)


def run_random_updraft_challenge_campaign(
    cases: tuple[RandomUpdraftChallengeCase, ...] | list[RandomUpdraftChallengeCase],
    governor: OnlineGovernor,
    selector: UnfilteredPrimitiveSelector,
    library: PrimitiveLibrary,
    config: RandomUpdraftChallengeConfig | None = None,
) -> RandomUpdraftChallengeReport:
    """Evaluate public selectors and the reference tracker on identical cases."""

    cfg = RandomUpdraftChallengeConfig() if config is None else config
    case_tuple = tuple(cases)
    if not case_tuple:
        raise ValueError("At least one random updraft challenge case is required.")

    records: list[RandomUpdraftChallengeMethodRecord] = []
    for case in case_tuple:
        for selector_name in cfg.selectors:
            if selector_name == SELECTOR_REFERENCE_TRACKING:
                records.append(_reference_tracking_record(case, cfg))
                continue
            decision = _selector_decision(selector_name, case, governor, selector, cfg.tier)
            records.append(_primitive_rollout_record(case, decision, library, cfg))
    return RandomUpdraftChallengeReport(records=tuple(records), config=cfg)


@dataclass(frozen=True)
class _SelectedPrimitiveDecision:
    selector_name: str
    selected_candidate_id: str | None
    selected_primitive_id: str | None
    fallback_used: bool
    fallback_reason: str | None
    decision_record: dict[str, object]


def _selector_decision(
    selector_name: str,
    case: RandomUpdraftChallengeCase,
    governor: OnlineGovernor,
    selector: UnfilteredPrimitiveSelector,
    tier: str,
) -> _SelectedPrimitiveDecision:
    if selector_name == SELECTOR_GOVERNOR:
        return _governor_selection(governor.decide(case.initial_state, tier=tier))
    if selector_name == SELECTOR_UNFILTERED:
        return _unfiltered_selection(selector.decide(case.initial_state, tier=tier))
    raise ValueError(f"Unsupported selector name: {selector_name}")


def _governor_selection(decision: GovernorDecision) -> _SelectedPrimitiveDecision:
    return _SelectedPrimitiveDecision(
        selector_name=SELECTOR_GOVERNOR,
        selected_candidate_id=decision.selected_candidate_id,
        selected_primitive_id=decision.selected_primitive_id or decision.selected_candidate_id,
        fallback_used=decision.fallback_used,
        fallback_reason=decision.fallback_reason,
        decision_record=decision.to_record(),
    )


def _unfiltered_selection(decision: UnfilteredSelectionDecision) -> _SelectedPrimitiveDecision:
    return _SelectedPrimitiveDecision(
        selector_name=SELECTOR_UNFILTERED,
        selected_candidate_id=decision.selected_candidate_id,
        selected_primitive_id=decision.selected_candidate_id,
        fallback_used=decision.fallback_used,
        fallback_reason=decision.fallback_reason,
        decision_record=decision.to_record(),
    )


def _primitive_rollout_record(
    case: RandomUpdraftChallengeCase,
    decision: _SelectedPrimitiveDecision,
    library: PrimitiveLibrary,
    config: RandomUpdraftChallengeConfig,
) -> RandomUpdraftChallengeMethodRecord:
    if decision.selected_primitive_id is None and decision.selected_candidate_id is None:
        return _non_rollout_record(case, decision, "no_selected_primitive")

    candidate_index = library.compressed.candidate_index
    primitive_id = None
    primitive = None
    for selected_id in _unique_ordered((decision.selected_primitive_id, decision.selected_candidate_id)):
        primitive = candidate_index.get(selected_id)
        if primitive is not None:
            primitive_id = selected_id
            break
    if primitive is None:
        return _non_rollout_record(case, decision, "missing_selected_primitive")

    rollout_config = PrimitiveRolloutConfig(
        dt_s=float(config.dt_s),
        wind_mode=case.wind_mode,
        max_duration_s=float(config.max_duration_s),
        scenario_id=f"{case.case_id}:{decision.selector_name}",
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
    return RandomUpdraftChallengeMethodRecord(
        case_id=case.case_id,
        case_set=case.case_set,
        environment_family=case.environment_family,
        selector_name=decision.selector_name,
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
        decision_record=decision.decision_record,
    )


def _reference_tracking_record(
    case: RandomUpdraftChallengeCase,
    config: RandomUpdraftChallengeConfig,
) -> RandomUpdraftChallengeMethodRecord:
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
    return RandomUpdraftChallengeMethodRecord(
        case_id=case.case_id,
        case_set=case.case_set,
        environment_family=case.environment_family,
        selector_name=SELECTOR_REFERENCE_TRACKING,
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
        decision_record=tracking_record,
    )


def _non_rollout_record(
    case: RandomUpdraftChallengeCase,
    decision: _SelectedPrimitiveDecision,
    reason: str,
) -> RandomUpdraftChallengeMethodRecord:
    return RandomUpdraftChallengeMethodRecord(
        case_id=case.case_id,
        case_set=case.case_set,
        environment_family=case.environment_family,
        selector_name=decision.selector_name,
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
    )


def _build_case(
    family: str,
    seed: int,
    config: RandomUpdraftChallengeConfig,
) -> RandomUpdraftChallengeCase:
    factors = _sample_factors(family, seed)
    initial_state = FlightState(
        position_w_m=np.array(factors["initial_position_w_m"], dtype=float),
        euler_rad=np.array(factors["initial_attitude_rad"], dtype=float),
        velocity_b_m_s=np.array(factors["initial_velocity_b_m_s"], dtype=float),
        rates_b_rad_s=np.zeros(3),
        surfaces_rad=np.zeros(3),
    )
    wind_model = _wind_from_factors(factors)
    case_id = f"challenge_{family}_seed_{seed}"
    return RandomUpdraftChallengeCase(
        case_id=case_id,
        case_set=config.case_set,
        environment_family=family,
        seed=seed,
        initial_state=initial_state,
        task=_gate_task(),
        wind_model=wind_model,
        wind_mode=config.wind_mode,
        factor_record={
            **factors,
            "used_for_library_construction": False,
            "used_for_governor_tuning": False,
        },
    )


def _sample_factors(family: str, seed: int) -> dict[str, object]:
    if family not in ENVIRONMENT_FAMILIES:
        raise ValueError(f"Unsupported environment family: {family}")
    rng = np.random.default_rng(int(seed))
    family_spec = _family_spec(family)
    centre_x_m = float(family_spec["centre_x_m"]) + _uniform(rng, -0.10, 0.10)
    centre_y_m = float(family_spec["centre_y_m"]) + _uniform(rng, -0.20, 0.20)
    strength_scale = _uniform(rng, *family_spec["strength_scale"])
    ring_radius_scale = _uniform(rng, 0.85, 1.20)
    ring_thickness_scale = _uniform(rng, 0.80, 1.30)
    background_vertical_m_s = _uniform(rng, -0.05, 0.08)
    source_count = int(family_spec["source_count"])
    fan_centres = _fan_centres(source_count, centre_x_m, centre_y_m, rng)
    factors = {
        "environment_family": family,
        "seed": int(seed),
        "source_count": source_count,
        "initial_position_w_m": [
            _uniform(rng, -0.05, 0.05),
            _uniform(rng, -0.08, 0.08),
            1.0 + _uniform(rng, -0.08, 0.08),
        ],
        "initial_velocity_b_m_s": [
            7.0 + _uniform(rng, -0.40, 0.40),
            _uniform(rng, -0.15, 0.15),
            _uniform(rng, -0.15, 0.15),
        ],
        "initial_attitude_rad": [
            _uniform(rng, -0.04, 0.04),
            _uniform(rng, -0.04, 0.04),
            _uniform(rng, -0.04, 0.04),
        ],
        "fan_centres_xy_m": fan_centres,
        "strength_m_s": float(family_spec["strength_m_s"]) * strength_scale,
        "strength_scale": strength_scale,
        "ring_radius_m": 0.35 * ring_radius_scale,
        "ring_radius_scale": ring_radius_scale,
        "ring_thickness_m": 0.12 * ring_thickness_scale,
        "ring_thickness_scale": ring_thickness_scale,
        "background_vertical_m_s": background_vertical_m_s,
    }
    _validate_factors(factors)
    return factors


def _family_spec(family: str) -> dict[str, object]:
    specs = {
        "weak_random_single_source": {
            "source_count": 1,
            "centre_x_m": 2.40,
            "centre_y_m": 0.00,
            "strength_m_s": 0.70,
            "strength_scale": (0.80, 1.10),
        },
        "hard_random_single_source": {
            "source_count": 1,
            "centre_x_m": 2.60,
            "centre_y_m": 0.00,
            "strength_m_s": 1.00,
            "strength_scale": (1.00, 1.50),
        },
        "random_two_source": {
            "source_count": 2,
            "centre_x_m": 2.80,
            "centre_y_m": 0.00,
            "strength_m_s": 0.75,
            "strength_scale": (0.90, 1.35),
        },
        "random_four_source": {
            "source_count": 4,
            "centre_x_m": 3.00,
            "centre_y_m": 0.00,
            "strength_m_s": 0.55,
            "strength_scale": (0.90, 1.30),
        },
    }
    return specs[family]


def _fan_centres(
    source_count: int,
    centre_x_m: float,
    centre_y_m: float,
    rng: np.random.Generator,
) -> list[list[float]]:
    offsets = {
        1: ((0.0, 0.0),),
        2: ((0.0, -0.18), (0.0, 0.18)),
        4: ((-0.10, -0.12), (-0.10, 0.12), (0.12, -0.12), (0.12, 0.12)),
    }
    centres: list[list[float]] = []
    for offset_x, offset_y in offsets[source_count]:
        centres.append(
            [
                float(centre_x_m + offset_x + _uniform(rng, -0.05, 0.05)),
                float(centre_y_m + offset_y + _uniform(rng, -0.05, 0.05)),
            ]
        )
    return centres


def _wind_from_factors(factors: dict[str, object]) -> AnnularUpdraft:
    centres = factors["fan_centres_xy_m"]
    if not isinstance(centres, list):
        raise ValueError("fan_centres_xy_m must be a list.")
    fans = []
    for index, centre in enumerate(centres):
        centre_values = np.asarray(centre, dtype=float).reshape(2)
        fans.append(
            FanUpdraft(
                centre_xy_m=(float(centre_values[0]), float(centre_values[1])),
                strength_m_s=float(factors["strength_m_s"]),
                ring_radius_m=float(factors["ring_radius_m"]),
                ring_thickness_m=float(factors["ring_thickness_m"]),
                background_m_s=float(factors["background_vertical_m_s"]) if index == 0 else 0.0,
            )
        )
    return AnnularUpdraft.from_fans(fans)


def _gate_task() -> GateTraversalTask:
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
            y_min_m=-1.2,
            y_max_m=1.2,
            z_min_m=0.1,
            z_max_m=3.0,
        ),
        timeout_s=1.0,
        angle_of_attack_limit_rad=0.8,
    )


def _validate_factors(factors: dict[str, object]) -> None:
    numeric_keys = (
        "strength_m_s",
        "strength_scale",
        "ring_radius_m",
        "ring_radius_scale",
        "ring_thickness_m",
        "ring_thickness_scale",
        "background_vertical_m_s",
    )
    for key in numeric_keys:
        value = factors[key]
        if not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f"{key} must be finite.")
    for key in ("strength_m_s", "ring_radius_m", "ring_thickness_m"):
        if float(factors[key]) <= 0.0:
            raise ValueError(f"{key} must be positive.")
    for key in ("initial_position_w_m", "initial_velocity_b_m_s", "initial_attitude_rad"):
        values = np.asarray(factors[key], dtype=float).reshape(3)
        if not np.isfinite(values).all():
            raise ValueError(f"{key} must contain finite values.")
    centres = np.asarray(factors["fan_centres_xy_m"], dtype=float).reshape(-1, 2)
    if not np.isfinite(centres).all():
        raise ValueError("fan_centres_xy_m must contain finite values.")


def _task_record(task: GateTraversalTask) -> dict[str, object]:
    volume = task.flight_volume
    return {
        "gate": {
            "centre_w_m": _list(task.gate.centre_w_m),
            "normal_w": _list(task.gate.normal_w),
            "width_m": float(task.gate.width_m),
            "height_m": float(task.gate.height_m),
        },
        "flight_volume": {
            "x_min_m": float(volume.x_min_m),
            "x_max_m": float(volume.x_max_m),
            "y_min_m": float(volume.y_min_m),
            "y_max_m": float(volume.y_max_m),
            "z_min_m": float(volume.z_min_m),
            "z_max_m": float(volume.z_max_m),
        },
        "timeout_s": float(task.timeout_s),
        "required_terminal_specific_energy_j_kg": float(task.required_terminal_specific_energy_j_kg),
        "angle_of_attack_limit_rad": _optional_float(task.angle_of_attack_limit_rad),
    }


def _selector_summary(records: tuple[RandomUpdraftChallengeMethodRecord, ...]) -> dict[str, object]:
    selected = [record for record in records if record.selected_candidate_id is not None or record.selected_primitive_id is not None]
    successful = [record for record in records if record.rollout_success]
    crossed = [record for record in records if record.gate_crossed is True]
    failures = [record.failure_reason for record in records if record.failure_reason is not None]
    return {
        "record_count": len(records),
        "selection_count": len(selected),
        "rollout_success_count": len(successful),
        "gate_crossed_count": len(crossed),
        "mean_gate_miss_distance_m": _finite_mean(record.gate_miss_distance_m for record in records),
        "mean_min_safety_margin_m": _finite_mean(record.min_safety_margin_m for record in records),
        "mean_terminal_specific_energy_margin_j_kg": _finite_mean(
            record.terminal_specific_energy_margin_j_kg for record in records
        ),
        "failure_reason_counts": _count(failures),
    }


def _count(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _finite_mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def _unique_ordered(values: Iterable[str | None]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value is None or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _uniform(rng: np.random.Generator, low: float, high: float) -> float:
    value = float(rng.uniform(float(low), float(high)))
    if not math.isfinite(value):
        raise ValueError("Sampled value must be finite.")
    return value


def _list(values: object) -> list[float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError("Record vectors must contain finite values.")
    return [float(value) for value in array]


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return result


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_bool(value: object) -> bool | None:
    return None if value is None else bool(value)


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "RandomUpdraftChallengeCase",
    "RandomUpdraftChallengeConfig",
    "RandomUpdraftChallengeMethodRecord",
    "RandomUpdraftChallengeReport",
    "build_random_updraft_challenge_cases",
    "run_random_updraft_challenge_campaign",
]
