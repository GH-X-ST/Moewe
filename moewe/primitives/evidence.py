"""Lightweight evidence records for primitive smoke rollouts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrimitiveEvidence:
    """Public evidence summary for one primitive rollout."""

    primitive_id: str
    family: str
    controller_type: str
    rollout_success: bool
    min_safety_margin_m: float
    terminal_specific_energy_change_j_kg: float
    terminal_specific_energy_margin_j_kg: float | None
    max_angle_of_attack_rad: float
    max_command_abs_rad: float
    gate_miss_distance_m: float | None
    failure_reason: str | None
    scenario_id: str
    seed: int | None = None
    entry_class: str | None = None
    exit_class: str | None = None
    retained: bool | None = None
    retention_reason: str | None = None
    rollout_duration_s: float = 0.0

    def to_record(self) -> dict[str, object]:
        """Return a serialisable evidence record."""

        return {
            "primitive_id": self.primitive_id,
            "family": self.family,
            "controller_type": self.controller_type,
            "rollout_success": self.rollout_success,
            "min_safety_margin_m": self.min_safety_margin_m,
            "terminal_specific_energy_change_j_kg": self.terminal_specific_energy_change_j_kg,
            "terminal_specific_energy_margin_j_kg": self.terminal_specific_energy_margin_j_kg,
            "max_angle_of_attack_rad": self.max_angle_of_attack_rad,
            "max_command_abs_rad": self.max_command_abs_rad,
            "gate_miss_distance_m": self.gate_miss_distance_m,
            "failure_reason": self.failure_reason,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "entry_class": self.entry_class,
            "exit_class": self.exit_class,
            "retained": self.retained,
            "retention_reason": self.retention_reason,
            "rollout_duration_s": self.rollout_duration_s,
        }
