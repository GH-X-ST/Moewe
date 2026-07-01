"""Primitive grammar expansion and structured evidence case builders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moewe.sim.actuator import NAUSICAA_MAX_COMMAND_ABS_RAD
from moewe.sim.glider_model import NAUSICAA_OPERATIONAL_ALPHA_LIMIT_RAD
from moewe.sim.randomisation import UniformRange, sample_parameters
from moewe.sim.updraft import AnnularUpdraft, FanUpdraft
from moewe.tasks.scenario import SINGLE_FAN_CENTER_XY_M

from .generate import PrimitiveCandidate, generate_primitives
from .grammar import (
    BankTransitionSpec,
    DwellSpec,
    OperatingPointSpec,
    PitchPulseSpec,
    PrimitiveGrammarSpec,
    RecoverySpec,
)
from .rollout import PrimitiveRolloutConfig
from .validate import AcceptanceThresholds, EntryPerturbationSpec, ValidationScenario


@dataclass(frozen=True)
class ExpandedPrimitiveGrammar:
    """Deterministic primitive grammar expansion wrapper."""

    spec: PrimitiveGrammarSpec

    def __post_init__(self) -> None:
        self.spec.validate()

    @property
    def candidate_count(self) -> int:
        op = self.spec.operating_point
        return int(
            len(op.airspeed_m_s)
            * len(op.flight_path_angle_rad)
            * len(op.altitude_m)
            * len(self.spec.bank_transition.target_bank_rad)
            * len(self.spec.pitch_pulse.delta_pitch_rad)
            * len(self.spec.dwell.duration_s)
            * len(self.spec.controller_types)
        )

    def generate(self, max_primitives: int | None = None) -> list[PrimitiveCandidate]:
        """Generate candidates, optionally truncating for smoke tests."""

        if max_primitives is not None and int(max_primitives) <= 0:
            raise ValueError("max_primitives must be positive when set.")
        candidates = generate_primitives(self.spec)
        if max_primitives is None:
            return candidates
        return candidates[: int(max_primitives)]

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_count": self.candidate_count,
            "family": self.spec.family,
            "airspeed_count": len(self.spec.operating_point.airspeed_m_s),
            "flight_path_angle_count": len(self.spec.operating_point.flight_path_angle_rad),
            "altitude_count": len(self.spec.operating_point.altitude_m),
            "bank_target_count": len(self.spec.bank_transition.target_bank_rad),
            "pitch_pulse_count": len(self.spec.pitch_pulse.delta_pitch_rad),
            "dwell_count": len(self.spec.dwell.duration_s),
            "controller_type_count": len(self.spec.controller_types),
        }


def dense_smoke_grammar() -> PrimitiveGrammarSpec:
    """Return a denser-than-smoke grammar that remains small enough for review."""

    return PrimitiveGrammarSpec(
        operating_point=OperatingPointSpec(
            airspeed_m_s=(6.5, 7.0),
            flight_path_angle_rad=(-0.03, 0.0),
            altitude_m=(1.45, 1.65),
        ),
        bank_transition=BankTransitionSpec(target_bank_rad=(-0.3, -0.15, 0.0, 0.15, 0.3), duration_s=0.2),
        pitch_pulse=PitchPulseSpec(delta_pitch_rad=(-0.08, -0.04, 0.0, 0.04, 0.08), duration_s=0.2),
        dwell=DwellSpec(duration_s=(0.1, 0.2)),
        recovery=RecoverySpec(duration_s=0.2, mode="nominal"),
        controller_types=("lqr",),
        family="bank_pitch_dwell_recovery",
    )


def expand_primitive_grammar(spec: PrimitiveGrammarSpec | None = None) -> ExpandedPrimitiveGrammar:
    """Create a deterministic expansion wrapper without running validation."""

    return ExpandedPrimitiveGrammar(PrimitiveGrammarSpec.smoke() if spec is None else spec)


LIBRARY_DESIGN_CASE_SET = "library_design"
FROZEN_HOLDOUT_CASE_SET = "frozen_structured_holdout"
RANDOM_CHALLENGE_CASE_SET = "random_challenge_after_freeze"


@dataclass(frozen=True)
class StructuredCase:
    """Structured evidence case record for library design or evaluation."""

    case_id: str
    case_set: str
    name: str
    flow_source_count: int
    updraft_longitudinal_relation: str
    updraft_lateral_relation: str
    updraft_strength_bin: str
    ring_radius_bin: str
    ring_thickness_bin: str
    gate_size_bin: str
    entry_speed_bin: str
    entry_lateral_bin: str
    entry_altitude_bin: str
    entry_attitude_bin: str
    role: str
    randomized: bool = False
    used_for_library_construction: bool = False
    used_for_governor_tuning: bool = False
    seed: int | None = None
    randomized_factors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be non-empty.")
        if self.case_set not in {LIBRARY_DESIGN_CASE_SET, FROZEN_HOLDOUT_CASE_SET, RANDOM_CHALLENGE_CASE_SET}:
            raise ValueError("Unknown structured evidence case_set.")
        if self.case_set == LIBRARY_DESIGN_CASE_SET and (self.randomized or self.seed is not None):
            raise ValueError("structured library design cases must be deterministic and unseeded.")
        if self.case_set == LIBRARY_DESIGN_CASE_SET and not self.used_for_library_construction:
            raise ValueError("library-design cases must be marked for library construction.")
        if self.case_set in {FROZEN_HOLDOUT_CASE_SET, RANDOM_CHALLENGE_CASE_SET} and self.used_for_library_construction:
            raise ValueError("Evaluation cases must not be used for library construction.")
        if self.case_set == RANDOM_CHALLENGE_CASE_SET and not self.randomized:
            raise ValueError("random-challenge cases must be marked randomized.")

    def to_record(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "case_set": self.case_set,
            "name": self.name,
            "flow_source_count": self.flow_source_count,
            "updraft_longitudinal_relation": self.updraft_longitudinal_relation,
            "updraft_lateral_relation": self.updraft_lateral_relation,
            "updraft_strength_bin": self.updraft_strength_bin,
            "ring_radius_bin": self.ring_radius_bin,
            "ring_thickness_bin": self.ring_thickness_bin,
            "gate_size_bin": self.gate_size_bin,
            "entry_speed_bin": self.entry_speed_bin,
            "entry_lateral_bin": self.entry_lateral_bin,
            "entry_altitude_bin": self.entry_altitude_bin,
            "entry_attitude_bin": self.entry_attitude_bin,
            "role": self.role,
            "randomized": self.randomized,
            "used_for_library_construction": self.used_for_library_construction,
            "used_for_governor_tuning": self.used_for_governor_tuning,
            "seed": self.seed,
            "randomized_factors": list(self.randomized_factors),
        }

    def to_validation_scenario(
        self,
        rollout_config: PrimitiveRolloutConfig | None = None,
        thresholds: AcceptanceThresholds | None = None,
    ) -> ValidationScenario:
        """Convert a case to the existing rollout scenario container."""

        return ValidationScenario(
            scenario_id=self.case_id,
            seed=self.seed,
            entry_perturbation=_entry_perturbation_for_case(self),
            rollout_config=PrimitiveRolloutConfig(dt_s=0.01, max_duration_s=None)
            if rollout_config is None
            else rollout_config,
            thresholds=_default_thresholds() if thresholds is None else thresholds,
            wind_model=_wind_for_case(self),
            wind_mode="panel",
        )


StructuredDesignCase = StructuredCase


@dataclass(frozen=True)
class StructuredDesignMatrixSpec:
    """Structured library design matrix settings."""

    rollout_config: PrimitiveRolloutConfig = PrimitiveRolloutConfig(dt_s=0.01, max_duration_s=None)
    thresholds: AcceptanceThresholds = AcceptanceThresholds()
    nominal_strength_m_s: float = 1.0
    nominal_ring_radius_m: float = 0.35
    nominal_ring_thickness_m: float = 0.12

    def validate(self) -> None:
        self.thresholds.validate()
        values = (self.nominal_strength_m_s, self.nominal_ring_radius_m, self.nominal_ring_thickness_m)
        if not np.isfinite(values).all() or any(float(value) <= 0.0 for value in values):
            raise ValueError("Nominal updraft parameters must be positive and finite.")


@dataclass(frozen=True)
class StructuredHoldoutMatrixSpec(StructuredDesignMatrixSpec):
    """Frozen structured holdout matrix settings."""


@dataclass(frozen=True)
class RandomChallengeMatrixSpec(StructuredDesignMatrixSpec):
    """Random challenge settings, available only after controller freeze."""

    seed: int = 91

    def validate(self) -> None:
        super().validate()
        if int(self.seed) < 0:
            raise ValueError("Random challenge seed must be non-negative.")


@dataclass(frozen=True)
class ValidationLadderSpec(StructuredDesignMatrixSpec):
    """Deprecated compatibility alias for structured design matrix settings."""


def _default_thresholds() -> AcceptanceThresholds:
    return AcceptanceThresholds(
        min_safety_margin_m=0.0,
        max_angle_of_attack_rad=NAUSICAA_OPERATIONAL_ALPHA_LIMIT_RAD,
        max_command_abs_rad=NAUSICAA_MAX_COMMAND_ABS_RAD,
        min_terminal_specific_energy_change_j_kg=-100.0,
    )


def _entry_vector_offset(
    y_m: float = 0.0,
    z_m: float = 0.0,
    speed_m_s: float = 0.0,
    roll_rad: float = 0.0,
    pitch_rad: float = 0.0,
) -> tuple[float, ...]:
    values = [0.0] * 15
    values[1] = y_m
    values[2] = z_m
    values[3] = roll_rad
    values[4] = pitch_rad
    values[6] = speed_m_s
    return tuple(values)


def _entry_half_width(
    xyz_m: float = 0.0,
    attitude_rad: float = 0.0,
    speed_m_s: float = 0.0,
) -> tuple[float, ...]:
    values = [0.0] * 15
    values[0] = xyz_m
    values[1] = xyz_m
    values[2] = xyz_m
    values[3] = attitude_rad
    values[4] = attitude_rad
    values[5] = attitude_rad
    values[6] = speed_m_s
    return tuple(values)


def _annular_wind(
    centre_xy_m: tuple[float, float],
    strength_m_s: float,
    ring_radius_m: float,
    ring_thickness_m: float,
    background_m_s: float = 0.0,
) -> AnnularUpdraft:
    return AnnularUpdraft.from_fans(
        (
            FanUpdraft(
                centre_xy_m=centre_xy_m,
                strength_m_s=float(strength_m_s),
                ring_radius_m=float(ring_radius_m),
                ring_thickness_m=float(ring_thickness_m),
                background_m_s=float(background_m_s),
            ),
        )
    )


def _multi_source_wind(
    source_count: int,
    strength_m_s: float,
    ring_radius_m: float,
    ring_thickness_m: float,
    centre_x_m: float = 0.0,
    centre_y_m: float = 0.0,
) -> AnnularUpdraft | None:
    if source_count <= 0:
        return None
    offsets = {
        1: ((centre_x_m, centre_y_m),),
        2: ((centre_x_m, centre_y_m - 0.18), (centre_x_m, centre_y_m + 0.18)),
        3: ((centre_x_m - 0.10, centre_y_m), (centre_x_m + 0.08, centre_y_m - 0.14), (centre_x_m + 0.08, centre_y_m + 0.14)),
        4: (
            (centre_x_m - 0.10, centre_y_m - 0.12),
            (centre_x_m - 0.10, centre_y_m + 0.12),
            (centre_x_m + 0.12, centre_y_m - 0.12),
            (centre_x_m + 0.12, centre_y_m + 0.12),
        ),
    }
    centres = offsets.get(source_count, offsets[4])
    return AnnularUpdraft.from_fans(
        FanUpdraft(
            centre_xy_m=centre,
            strength_m_s=float(strength_m_s),
            ring_radius_m=float(ring_radius_m),
            ring_thickness_m=float(ring_thickness_m),
        )
        for centre in centres
    )


def _randomised_annular_wind(spec: RandomChallengeMatrixSpec, seed: int, hard: bool) -> AnnularUpdraft:
    spread = 0.12 if hard else 0.06
    strength_range = (0.6, 1.5) if hard else (0.8, 1.2)
    radius_range = (0.75, 1.30) if hard else (0.9, 1.1)
    thickness_range = (0.7, 1.4) if hard else (0.85, 1.15)
    background_range = (-0.10, 0.10) if hard else (-0.04, 0.04)
    sample = sample_parameters(
        {
            "centre_x_m": UniformRange(SINGLE_FAN_CENTER_XY_M[0] - spread, SINGLE_FAN_CENTER_XY_M[0] + spread),
            "centre_y_m": UniformRange(SINGLE_FAN_CENTER_XY_M[1] - spread, SINGLE_FAN_CENTER_XY_M[1] + spread),
            "strength_scale": UniformRange(*strength_range),
            "ring_radius_scale": UniformRange(*radius_range),
            "ring_thickness_scale": UniformRange(*thickness_range),
            "background_vertical_m_s": UniformRange(*background_range),
        },
        seed=seed,
    )
    return _annular_wind(
        centre_xy_m=(float(sample["centre_x_m"]), float(sample["centre_y_m"])),
        strength_m_s=spec.nominal_strength_m_s * float(sample["strength_scale"]),
        ring_radius_m=spec.nominal_ring_radius_m * float(sample["ring_radius_scale"]),
        ring_thickness_m=spec.nominal_ring_thickness_m * float(sample["ring_thickness_scale"]),
        background_m_s=float(sample["background_vertical_m_s"]),
    )


def _case(
    case_id: str,
    case_set: str,
    name: str,
    flow_source_count: int,
    updraft_longitudinal_relation: str,
    updraft_lateral_relation: str,
    updraft_strength_bin: str,
    ring_radius_bin: str,
    ring_thickness_bin: str,
    gate_size_bin: str,
    entry_speed_bin: str,
    entry_lateral_bin: str,
    entry_altitude_bin: str,
    entry_attitude_bin: str,
    role: str,
    randomized: bool = False,
    seed: int | None = None,
    randomized_factors: tuple[str, ...] = (),
) -> StructuredCase:
    return StructuredCase(
        case_id=case_id,
        case_set=case_set,
        name=name,
        flow_source_count=flow_source_count,
        updraft_longitudinal_relation=updraft_longitudinal_relation,
        updraft_lateral_relation=updraft_lateral_relation,
        updraft_strength_bin=updraft_strength_bin,
        ring_radius_bin=ring_radius_bin,
        ring_thickness_bin=ring_thickness_bin,
        gate_size_bin=gate_size_bin,
        entry_speed_bin=entry_speed_bin,
        entry_lateral_bin=entry_lateral_bin,
        entry_altitude_bin=entry_altitude_bin,
        entry_attitude_bin=entry_attitude_bin,
        role=role,
        randomized=randomized,
        used_for_library_construction=case_set == LIBRARY_DESIGN_CASE_SET,
        used_for_governor_tuning=case_set == LIBRARY_DESIGN_CASE_SET,
        seed=seed,
        randomized_factors=randomized_factors,
    )


def build_structured_library_design_cases(
    spec: StructuredDesignMatrixSpec | None = None,
) -> tuple[StructuredDesignCase, ...]:
    """Build deterministic physical cases for primitive library construction."""

    design = StructuredDesignMatrixSpec() if spec is None else spec
    design.validate()
    del design
    return (
        _case("design_still_air_trim_recovery", LIBRARY_DESIGN_CASE_SET, "still air trim and recovery", 0, "none", "centreline", "none", "nominal", "nominal", "nominal", "nominal", "centre", "nominal", "level", "baseline primitive stability"),
        _case("design_still_air_gate_alignment", LIBRARY_DESIGN_CASE_SET, "still air gate alignment", 0, "none", "centreline", "none", "nominal", "nominal", "nominal", "nominal", "left_right", "nominal", "level", "basic geometric correction"),
        _case("design_lift_before_gate", LIBRARY_DESIGN_CASE_SET, "single centred lift before gate", 1, "upstream", "centreline", "nominal", "nominal", "nominal", "nominal", "low_nominal", "centre", "nominal", "level", "energy capture before gate"),
        _case("design_lift_at_gate", LIBRARY_DESIGN_CASE_SET, "single centred lift at gate", 1, "gate_centred", "centreline", "nominal", "nominal", "nominal", "nominal", "nominal", "centre", "nominal", "level", "lift and gate coupling"),
        _case("design_lift_after_gate", LIBRARY_DESIGN_CASE_SET, "single centred lift after gate", 1, "downstream", "centreline", "nominal", "nominal", "nominal", "nominal", "nominal_high", "centre", "nominal", "level", "avoid destabilising post gate lift"),
        _case("design_lateral_lift", LIBRARY_DESIGN_CASE_SET, "single lateral lift", 1, "gate_centred", "left_right_offset", "nominal", "nominal", "nominal", "nominal", "nominal", "left_right", "nominal", "mild_bank", "cross track and roll recovery"),
        _case("design_strong_lift", LIBRARY_DESIGN_CASE_SET, "single strong lift", 1, "upstream_or_gate", "centreline", "strong", "nominal", "nominal", "nominal", "low_nominal_high", "centre", "nominal", "level", "helpful lift versus stall risk"),
        _case("design_broad_lift", LIBRARY_DESIGN_CASE_SET, "single broad lift", 1, "upstream_or_gate", "centreline", "nominal", "broad", "diffuse", "nominal", "nominal", "centre", "nominal", "level", "long exposure energy effect"),
        _case("design_compact_lift", LIBRARY_DESIGN_CASE_SET, "single compact lift", 1, "upstream_or_gate", "centreline", "nominal", "compact", "thin", "nominal", "nominal", "centre", "nominal", "level", "sharp gradient timing effect"),
        _case("design_two_source_symmetric_lift", LIBRARY_DESIGN_CASE_SET, "two source symmetric lift", 2, "centred_around_gate", "symmetric", "nominal", "nominal", "nominal", "nominal", "nominal", "centre", "nominal", "level", "simple multi plume interaction"),
        _case("design_two_source_asymmetric_lift", LIBRARY_DESIGN_CASE_SET, "two source asymmetric lift", 2, "one_source_shifted", "asymmetric", "nominal", "nominal", "nominal", "nominal", "nominal", "left_right", "nominal", "mild_bank", "lateral interaction stress"),
        _case("design_recovery_only", LIBRARY_DESIGN_CASE_SET, "recovery only", 1, "any", "centreline", "nominal", "nominal", "nominal", "nominal", "low", "centre", "low", "mild_pitch", "terminal recovery actions"),
    )


def build_frozen_structured_holdout_cases(
    spec: StructuredHoldoutMatrixSpec | None = None,
) -> tuple[StructuredDesignCase, ...]:
    """Build deterministic holdout cases excluded from construction and tuning."""

    holdout = StructuredHoldoutMatrixSpec() if spec is None else spec
    holdout.validate()
    del holdout
    return (
        _case("holdout_still_air_easy", FROZEN_HOLDOUT_CASE_SET, "easy still air holdout", 0, "none", "centreline", "none", "nominal", "nominal", "easy", "nominal", "centre", "nominal", "level", "easy baseline competence"),
        _case("holdout_shifted_single_fan", FROZEN_HOLDOUT_CASE_SET, "shifted single fan", 1, "upstream", "left_offset", "nominal", "nominal", "nominal", "nominal", "nominal", "left", "nominal", "level", "lateral interpolation"),
        _case("holdout_strong_fan_at_gate", FROZEN_HOLDOUT_CASE_SET, "strong fan at gate", 1, "gate_centred", "centreline", "strong", "nominal", "nominal", "nominal", "nominal", "centre", "nominal", "level", "unrecoverable lift entry check"),
        _case("holdout_broad_fan_downstream", FROZEN_HOLDOUT_CASE_SET, "broad fan downstream", 1, "downstream", "centreline", "nominal", "broad", "diffuse", "nominal", "high", "centre", "nominal", "level", "post gate exit stability"),
        _case("holdout_two_source_asymmetric", FROZEN_HOLDOUT_CASE_SET, "two source asymmetric", 2, "upstream_and_gate", "asymmetric", "nominal", "nominal", "nominal", "nominal", "nominal", "right", "nominal", "mild_bank", "composition holdout"),
        _case("holdout_three_source_structured", FROZEN_HOLDOUT_CASE_SET, "three source structured", 3, "triangular", "structured", "nominal", "nominal", "nominal", "nominal", "nominal", "centre", "nominal", "level", "multi source bridge"),
        _case("holdout_four_source_structured", FROZEN_HOLDOUT_CASE_SET, "four source structured", 4, "staggered", "structured", "nominal", "nominal", "nominal", "nominal", "nominal", "centre", "nominal", "level", "multi source reality gap exposure"),
        _case("holdout_tight_gate_strong_lift", FROZEN_HOLDOUT_CASE_SET, "tight gate strong lift", 2, "gate_centred", "centreline", "strong", "nominal", "thin", "tight", "nominal", "centre", "nominal", "level", "combined geometry and energy stress"),
        _case("holdout_low_energy_launch", FROZEN_HOLDOUT_CASE_SET, "low energy launch", 1, "upstream", "centreline", "nominal", "nominal", "nominal", "nominal", "low", "centre", "low", "level", "launch energy sensitivity"),
        _case("holdout_lateral_upset_entry", FROZEN_HOLDOUT_CASE_SET, "lateral upset entry", 2, "offset", "asymmetric", "nominal", "nominal", "nominal", "nominal", "nominal", "left_right", "nominal", "mild_bank", "lateral returnability"),
    )


def build_random_challenge_cases(
    spec: RandomChallengeMatrixSpec | None = None,
    *,
    controller_frozen: bool = False,
) -> tuple[StructuredDesignCase, ...]:
    """Build random challenge cases only after an explicit freeze guard."""

    if not controller_frozen:
        raise ValueError("Random challenge cases require controller_frozen=True.")
    random_spec = RandomChallengeMatrixSpec() if spec is None else spec
    random_spec.validate()
    base = int(random_spec.seed)
    factors = ("updraft_centre", "updraft_strength", "ring_radius", "ring_thickness")
    return (
        _case("challenge_light_random_updraft", RANDOM_CHALLENGE_CASE_SET, "light random updraft", 1, "random", "random", "random_light", "random_light", "nominal", "nominal", "nominal", "centre", "nominal", "level", "mild robustness after freeze", True, base, factors[:-1]),
        _case("challenge_hard_random_updraft", RANDOM_CHALLENGE_CASE_SET, "hard random updraft", 1, "random", "random", "random_hard", "random_hard", "random_hard", "nominal_tight", "nominal", "centre", "nominal", "level", "recoverability stress after freeze", True, base + 1, factors),
        _case("challenge_random_three_source", RANDOM_CHALLENGE_CASE_SET, "random three source", 3, "random", "random", "random", "random", "random", "nominal", "nominal", "centre", "nominal", "level", "three source robustness after freeze", True, base + 2, ("source_positions", "source_strengths")),
        _case("challenge_random_four_source", RANDOM_CHALLENGE_CASE_SET, "random four source", 4, "random", "random", "random", "random", "random", "nominal", "nominal", "centre", "nominal", "level", "four source robustness after freeze", True, base + 3, ("source_positions", "source_strengths")),
        _case("challenge_initial_state_random", RANDOM_CHALLENGE_CASE_SET, "initial state random", 1, "fixed", "centreline", "nominal", "nominal", "nominal", "nominal", "random", "random", "random", "random_mild", "launch sensitivity after freeze", True, base + 4, ("initial_position", "initial_speed", "initial_attitude")),
        _case("challenge_limited_plant_random", RANDOM_CHALLENGE_CASE_SET, "limited plant random", 1, "fixed", "centreline", "nominal", "nominal", "nominal", "nominal", "nominal", "centre", "nominal", "level", "plant sensitivity after review", True, base + 5, ("mass", "cg", "control_effectiveness", "latency")),
    )


def _entry_perturbation_for_case(case: StructuredCase) -> EntryPerturbationSpec:
    if case.case_set == RANDOM_CHALLENGE_CASE_SET and case.randomized:
        return EntryPerturbationSpec(half_width=_entry_half_width(xyz_m=0.03, attitude_rad=0.02, speed_m_s=0.1))
    lateral = {"left": -0.12, "right": 0.12, "left_right": 0.10}.get(case.entry_lateral_bin, 0.0)
    altitude = {"low": -0.12, "high": 0.12}.get(case.entry_altitude_bin, 0.0)
    speed = -0.4 if "low" in case.entry_speed_bin else 0.4 if "high" in case.entry_speed_bin else 0.0
    roll = 0.08 if "bank" in case.entry_attitude_bin else 0.0
    pitch = 0.06 if "pitch" in case.entry_attitude_bin else 0.0
    return EntryPerturbationSpec(offset=_entry_vector_offset(y_m=lateral, z_m=altitude, speed_m_s=speed, roll_rad=roll, pitch_rad=pitch))


def _wind_for_case(case: StructuredCase) -> AnnularUpdraft | None:
    if case.flow_source_count <= 0:
        return None
    strength = {"weak": 0.7, "strong": 1.4, "random_light": 1.0, "random_hard": 1.25}.get(
        case.updraft_strength_bin,
        1.0,
    )
    radius = {"compact": 0.25, "broad": 0.50, "random_light": 0.35, "random_hard": 0.40}.get(
        case.ring_radius_bin,
        0.35,
    )
    thickness = {"thin": 0.08, "diffuse": 0.18, "random_hard": 0.14}.get(case.ring_thickness_bin, 0.12)
    centre_x = {
        "upstream": SINGLE_FAN_CENTER_XY_M[0] - 0.6,
        "upstream_or_gate": SINGLE_FAN_CENTER_XY_M[0] - 0.3,
        "upstream_and_gate": SINGLE_FAN_CENTER_XY_M[0] - 0.3,
        "downstream": SINGLE_FAN_CENTER_XY_M[0] + 0.6,
        "gate_centred": SINGLE_FAN_CENTER_XY_M[0],
        "centred_around_gate": SINGLE_FAN_CENTER_XY_M[0],
        "fixed": SINGLE_FAN_CENTER_XY_M[0],
    }.get(
        case.updraft_longitudinal_relation,
        SINGLE_FAN_CENTER_XY_M[0],
    )
    centre_y = (
        SINGLE_FAN_CENTER_XY_M[1] - 0.12
        if "left" in case.updraft_lateral_relation
        else SINGLE_FAN_CENTER_XY_M[1] + 0.12
        if "right" in case.updraft_lateral_relation
        else SINGLE_FAN_CENTER_XY_M[1]
    )
    if case.case_set == RANDOM_CHALLENGE_CASE_SET and case.seed is not None:
        challenge_spec = RandomChallengeMatrixSpec(seed=case.seed)
        return _randomised_annular_wind(challenge_spec, seed=case.seed, hard="hard" in case.name)
    return _multi_source_wind(case.flow_source_count, strength, radius, thickness, centre_x_m=centre_x, centre_y_m=centre_y)


def build_validation_ladder(spec: ValidationLadderSpec | None = None) -> tuple[ValidationScenario, ...]:
    """Deprecated alias for library-design structured design scenarios."""

    design = ValidationLadderSpec() if spec is None else spec
    design.validate()
    return tuple(
        case.to_validation_scenario(rollout_config=design.rollout_config, thresholds=design.thresholds)
        for case in build_structured_library_design_cases(design)
    )
