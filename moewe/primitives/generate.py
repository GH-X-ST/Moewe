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
    return primitives
