"""Deterministic generation of structured primitive candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from itertools import product

from moewe.control.trim import TrimSpec, pseudo_trim
from moewe.sim.glider_model import GliderModel

from .grammar import PrimitiveEntryCondition, PrimitiveGrammarSpec, PrimitiveSafetyLimits
from .phases import BankTransitionPhase, DwellPhase, HoldPhase, PitchPulsePhase, PrimitivePhase, RecoveryPhase
from .reference import PrimitiveReference


def _stable_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _entry_trim_mode(airspeed_m_s: float, gamma_rad: float, altitude_m: float) -> str:
    if airspeed_m_s < 6.8 or altitude_m < 0.9:
        return "low_energy_glide"
    if gamma_rad < -0.02:
        return "descending_glide"
    return "stable_glide"


def _bank_label(bank_rad: float) -> str:
    if abs(bank_rad) < 1e-12:
        return "wings_level"
    side = "left" if bank_rad < 0.0 else "right"
    magnitude = "mild" if abs(bank_rad) <= 0.2 else "energy_retaining"
    return f"{magnitude}_{side}"


def _pitch_label(delta_pitch_rad: float) -> str:
    if abs(delta_pitch_rad) < 1e-12:
        return "neutral"
    if delta_pitch_rad > 0.0:
        return "altitude_recovery" if delta_pitch_rad <= 0.05 else "terminal_flare_preparation"
    return "energy_conserving" if abs(delta_pitch_rad) <= 0.05 else "updraft_entry"


def _dwell_label(duration_s: float) -> str:
    if duration_s <= 0.0:
        return "no_dwell"
    if duration_s <= 0.1:
        return "short"
    if duration_s <= 0.2:
        return "medium"
    return "long"


def _aggressiveness_level(bank_rad: float, pitch_delta_rad: float, dwell_s: float) -> int:
    score = abs(bank_rad) / 0.2 + abs(pitch_delta_rad) / 0.05 + max(0.0, dwell_s - 0.1) / 0.1
    if score < 0.5:
        return 0
    if score < 2.0:
        return 1
    return 2


def _task_intent_label(bank_rad: float, pitch_delta_rad: float, dwell_s: float) -> str:
    if pitch_delta_rad > 0.0:
        return "safe_exit"
    if pitch_delta_rad < 0.0 and dwell_s >= 0.2:
        return "lift_dwell"
    if abs(bank_rad) > 1e-12:
        return "gate_center"
    return "lift_entry"


def _candidate_sort_key(candidate: "PrimitiveCandidate") -> tuple[int, str, str]:
    return (
        int(candidate.metadata.get("aggressiveness_level", 0)),
        str(candidate.metadata.get("task_intent_label", "")),
        candidate.primitive_id,
    )


def _linked_metadata(
    candidate: "PrimitiveCandidate",
    candidates: tuple["PrimitiveCandidate", ...],
) -> dict[str, object]:
    metadata = dict(candidate.metadata)
    aggressiveness = int(metadata.get("aggressiveness_level", 0))
    task_intent = str(metadata.get("task_intent_label", ""))
    same_family = tuple(item for item in candidates if item.family == candidate.family and item.primitive_id != candidate.primitive_id)
    weaker_same_task = tuple(
        item
        for item in same_family
        if int(item.metadata.get("aggressiveness_level", 0)) < aggressiveness
        and str(item.metadata.get("task_intent_label", "")) == task_intent
    )
    weaker_same_family = tuple(
        item
        for item in same_family
        if int(item.metadata.get("aggressiveness_level", 0)) < aggressiveness
    )
    safe_exit = tuple(
        item
        for item in same_family
        if str(item.metadata.get("task_intent_label", "")) == "safe_exit"
    )
    metadata["same_family_degrade_id"] = _first_id(weaker_same_task or weaker_same_family)
    metadata["safer_task_preserving_degrade_id"] = _first_id(weaker_same_task or safe_exit or weaker_same_family)
    metadata["recovery_degrade_id"] = _first_id(safe_exit)
    return metadata


def _first_id(candidates: tuple["PrimitiveCandidate", ...]) -> str | None:
    if not candidates:
        return None
    return min(candidates, key=_candidate_sort_key).primitive_id


def _with_degradation_links(candidates: list["PrimitiveCandidate"]) -> list["PrimitiveCandidate"]:
    candidate_tuple = tuple(candidates)
    linked: list[PrimitiveCandidate] = []
    for candidate in candidate_tuple:
        linked.append(
            PrimitiveCandidate(
                primitive_id=candidate.primitive_id,
                family=candidate.family,
                phases=candidate.phases,
                reference=PrimitiveReference(
                    phases=candidate.reference.phases,
                    metadata=_linked_metadata(candidate, candidate_tuple),
                ),
                entry_condition=candidate.entry_condition,
                safety_limits=candidate.safety_limits,
                controller_type=candidate.controller_type,
                metadata=_linked_metadata(candidate, candidate_tuple),
                generation_hash=candidate.generation_hash,
            )
        )
    return linked


@dataclass(frozen=True)
class PrimitiveCandidate:
    """Inspectable structured primitive record."""

    primitive_id: str
    family: str
    phases: tuple[PrimitivePhase, ...]
    reference: PrimitiveReference
    entry_condition: PrimitiveEntryCondition
    safety_limits: PrimitiveSafetyLimits
    controller_type: str
    metadata: dict[str, object] = field(default_factory=dict)
    generation_hash: str = ""

    def __post_init__(self) -> None:
        if not self.primitive_id:
            raise ValueError("primitive_id must be non-empty.")
        phases = tuple(self.phases)
        if not phases:
            raise ValueError("PrimitiveCandidate requires at least one phase.")
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def summary(self) -> dict[str, object]:
        """Return a compact public summary without simulation histories."""

        return {
            "primitive_id": self.primitive_id,
            "family": self.family,
            "controller_type": self.controller_type,
            "total_duration_s": self.reference.total_duration_s,
            "phase_names": [phase.name for phase in self.phases],
            "metadata": dict(self.metadata),
            "generation_hash": self.generation_hash,
        }


def generate_primitives(
    spec: PrimitiveGrammarSpec | None = None,
    model: GliderModel | None = None,
    wind_model: object | None = None,
) -> list[PrimitiveCandidate]:
    """Generate a deterministic smoke-scale primitive list from a grammar spec."""

    grammar = PrimitiveGrammarSpec.smoke() if spec is None else spec
    grammar.validate()
    primitives: list[PrimitiveCandidate] = []
    op = grammar.operating_point
    command_rad = tuple(float(value) for value in op.command_rad)
    for airspeed, gamma, altitude in product(op.airspeed_m_s, op.flight_path_angle_rad, op.altitude_m):
        trim = pseudo_trim(
            TrimSpec(
                airspeed_m_s=float(airspeed),
                flight_path_angle_rad=float(gamma),
                x_w_m=float(op.x_w_m),
                y_w_m=float(op.y_w_m),
                altitude_m=float(altitude),
                heading_rad=float(op.heading_rad),
                turn_rate_rad_s=float(op.turn_rate_rad_s),
                command_rad=command_rad,
                wind_mode=op.wind_mode,
            ),
            model=model,
            wind_model=wind_model,
        )
        nominal_state = trim.state.with_surfaces(trim.command_rad)
        for bank_rad, pitch_delta_rad, dwell_s, controller_type in product(
            grammar.bank_transition.target_bank_rad,
            grammar.pitch_pulse.delta_pitch_rad,
            grammar.dwell.duration_s,
            grammar.controller_types,
        ):
            factors = {
                "family": grammar.family,
                "airspeed_m_s": float(airspeed),
                "flight_path_angle_rad": float(gamma),
                "altitude_m": float(altitude),
                "target_bank_rad": float(bank_rad),
                "delta_pitch_rad": float(pitch_delta_rad),
                "dwell_duration_s": float(dwell_s),
                "recovery_mode": grammar.recovery.mode,
                "controller_type": str(controller_type),
            }
            generation_hash = _stable_hash(factors)
            primitive_id = f"prim_{generation_hash[:12]}"
            hold = HoldPhase(
                duration_s=grammar.hold_duration_s,
                reference_state=nominal_state,
                command_rad=trim.command_rad,
            )
            bank = BankTransitionPhase(
                duration_s=grammar.bank_transition.duration_s,
                start_state=nominal_state,
                command_rad=trim.command_rad,
                target_bank_rad=float(bank_rad),
            )
            banked_state = bank.sample(bank.duration_s).state
            pitch = PitchPulsePhase(
                duration_s=grammar.pitch_pulse.duration_s,
                reference_state=banked_state,
                command_rad=trim.command_rad,
                delta_pitch_rad=float(pitch_delta_rad),
            )
            post_pitch_state = pitch.sample(pitch.duration_s).state
            dwell = DwellPhase(
                duration_s=float(dwell_s),
                reference_state=post_pitch_state,
                command_rad=trim.command_rad,
            )
            recovery = RecoveryPhase(
                duration_s=grammar.recovery.duration_s,
                start_state=dwell.sample(dwell.duration_s).state,
                target_state=nominal_state,
                command_rad=trim.command_rad,
            )
            phases: tuple[PrimitivePhase, ...] = (hold, bank, pitch, dwell, recovery)
            metadata = {
                "grammar_factors": factors,
                "family": grammar.family,
                "aggressiveness_level": _aggressiveness_level(float(bank_rad), float(pitch_delta_rad), float(dwell_s)),
                "entry_trim_mode": _entry_trim_mode(float(airspeed), float(gamma), float(altitude)),
                "bank_transition_label": _bank_label(float(bank_rad)),
                "pitch_pulse_label": _pitch_label(float(pitch_delta_rad)),
                "dwell_label": _dwell_label(float(dwell_s)),
                "recovery_label": grammar.recovery.mode,
                "task_intent_label": _task_intent_label(float(bank_rad), float(pitch_delta_rad), float(dwell_s)),
                "same_family_degrade_id": None,
                "safer_task_preserving_degrade_id": None,
                "recovery_degrade_id": None,
                "trim_method": trim.method,
                "trim_converged": trim.converged,
                "trim_residual_norm": trim.residual.residual_norm,
                "command_order": "[aileron, elevator, rudder]",
            }
            reference = PrimitiveReference(phases=phases, metadata=metadata)
            primitives.append(
                PrimitiveCandidate(
                    primitive_id=primitive_id,
                    family=grammar.family,
                    phases=phases,
                    reference=reference,
                    entry_condition=grammar.entry_condition,
                    safety_limits=grammar.safety_limits,
                    controller_type=str(controller_type),
                    metadata=metadata,
                    generation_hash=generation_hash,
                )
            )
    return _with_degradation_links(primitives)
