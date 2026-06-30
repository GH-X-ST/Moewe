"""Objective proposers for closed-loop primitive requests."""

from __future__ import annotations

from dataclasses import dataclass, field

from moewe.governor import ObjectiveProposer, PrimitiveRequest
from moewe.sim.state import FlightState


def _request_id(prefix: str, task_intent: str, preferred_family: str) -> str:
    return f"{prefix}:{task_intent}:{preferred_family}"


@dataclass(frozen=True)
class GateTraversalProposer:
    """Gate-traversal objective proposer without safety filtering."""

    preferred_family: str = "bank_pitch_dwell_recovery"
    requested_aggressiveness: int = 1
    target_region: dict[str, object] = field(default_factory=lambda: {"region_type": "gate"})
    objective_score_terms: dict[str, float] = field(
        default_factory=lambda: {
            "gate_alignment": 1.0,
            "terminal_energy": 0.2,
            "useful_lift_exposure": 0.1,
        }
    )

    def propose(self, state: FlightState | None = None) -> PrimitiveRequest:
        del state
        return PrimitiveRequest(
            request_id=_request_id("objective", "gate_traversal", self.preferred_family),
            task_intent="gate_traversal",
            preferred_family=self.preferred_family,
            requested_aggressiveness=self.requested_aggressiveness,
            target_region=self.target_region,
            objective_score_terms=self.objective_score_terms,
            metadata={"proposer": "gate_traversal", "memory_evidence": "not_used"},
        )


@dataclass(frozen=True)
class LiftExploitationProposer:
    """Lift-exploitation objective proposer without returnability filtering."""

    preferred_family: str = "bank_pitch_dwell_recovery"
    requested_aggressiveness: int = 2
    memory_enabled: bool = False

    def propose(self, state: FlightState | None = None) -> PrimitiveRequest:
        del state
        return PrimitiveRequest(
            request_id=_request_id("objective", "lift_exploitation", self.preferred_family),
            task_intent="lift_exploitation",
            preferred_family=self.preferred_family,
            requested_aggressiveness=self.requested_aggressiveness,
            target_region={"region_type": "local_updraft"},
            objective_score_terms={
                "useful_lift_exposure": 1.0,
                "terminal_energy": 0.5,
                "gate_alignment": 0.1,
            },
            metadata={
                "proposer": "lift_exploitation",
                "memory_evidence": "enabled" if self.memory_enabled else "not_used",
            },
        )


@dataclass(frozen=True)
class RecoveryProposer:
    """Recovery objective proposer without admission filtering."""

    preferred_family: str = "bank_pitch_dwell_recovery"

    def propose(self, state: FlightState | None = None) -> PrimitiveRequest:
        del state
        return PrimitiveRequest(
            request_id=_request_id("objective", "recovery", self.preferred_family),
            task_intent="recovery",
            preferred_family=self.preferred_family,
            requested_aggressiveness=0,
            target_region={"region_type": "safe_exit"},
            objective_score_terms={"safety_margin": 1.0, "terminal_energy": 0.5},
            metadata={"proposer": "recovery", "memory_evidence": "not_used"},
        )


@dataclass(frozen=True)
class TerminalAlignmentProposer:
    """Terminal-alignment proposer scaffold for later benchmark stages."""

    preferred_family: str = "bank_pitch_dwell_recovery"
    requested_aggressiveness: int = 1

    def propose(self, state: FlightState | None = None) -> PrimitiveRequest:
        del state
        return PrimitiveRequest(
            request_id=_request_id("objective", "terminal_alignment", self.preferred_family),
            task_intent="terminal_alignment",
            preferred_family=self.preferred_family,
            requested_aggressiveness=self.requested_aggressiveness,
            target_region={"region_type": "terminal_alignment"},
            objective_score_terms={"alignment": 1.0, "safety_margin": 0.5},
            metadata={"proposer": "terminal_alignment", "scaffold_only": True},
        )


__all__ = [
    "GateTraversalProposer",
    "LiftExploitationProposer",
    "ObjectiveProposer",
    "RecoveryProposer",
    "TerminalAlignmentProposer",
]
