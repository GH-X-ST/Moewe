"""Behaviour feature extraction for primitive library compression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .validate import PrimitiveValidationResult


def _tuple3(value: np.ndarray) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=float).reshape(3)
    return (float(array[0]), float(array[1]), float(array[2]))


@dataclass(frozen=True)
class PrimitiveBehaviourFeature:
    """Serialisable rollout feature record for medoid compression."""

    primitive_id: str
    family: str
    controller_type: str
    entry_class: str
    exit_class: str
    scenario_id: str
    seed: int | None
    retained: bool
    retention_reason: str | None
    rollout_duration_s: float
    terminal_displacement_w_m: tuple[float, float, float]
    terminal_velocity_delta_b_m_s: tuple[float, float, float]
    terminal_specific_energy_change_j_kg: float
    terminal_specific_energy_margin_j_kg: float | None
    min_safety_margin_m: float
    max_angle_of_attack_rad: float
    max_command_abs_rad: float
    gate_miss_distance_m: float | None
    mean_positive_vertical_wind_m_s: float | None = None

    def to_record(self) -> dict[str, object]:
        return {
            "primitive_id": self.primitive_id,
            "family": self.family,
            "controller_type": self.controller_type,
            "entry_class": self.entry_class,
            "exit_class": self.exit_class,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "retained": self.retained,
            "retention_reason": self.retention_reason,
            "rollout_duration_s": self.rollout_duration_s,
            "terminal_displacement_w_m": list(self.terminal_displacement_w_m),
            "terminal_velocity_delta_b_m_s": list(self.terminal_velocity_delta_b_m_s),
            "terminal_specific_energy_change_j_kg": self.terminal_specific_energy_change_j_kg,
            "terminal_specific_energy_margin_j_kg": self.terminal_specific_energy_margin_j_kg,
            "min_safety_margin_m": self.min_safety_margin_m,
            "max_angle_of_attack_rad": self.max_angle_of_attack_rad,
            "max_command_abs_rad": self.max_command_abs_rad,
            "gate_miss_distance_m": self.gate_miss_distance_m,
            "mean_positive_vertical_wind_m_s": self.mean_positive_vertical_wind_m_s,
            "velocity_frame": "body",
            "displacement_frame": "world_z_up",
        }


@dataclass(frozen=True)
class FeatureScaleSpec:
    """Normalisation scales for feature-distance calculations."""

    displacement_m: float = 1.0
    velocity_m_s: float = 1.0
    energy_j_kg: float = 10.0
    safety_margin_m: float = 1.0
    angle_rad: float = 0.5
    command_rad: float = 0.5
    gate_miss_m: float = 1.0
    useful_lift_m_s: float = 1.0

    def validate(self) -> None:
        values = (
            self.displacement_m,
            self.velocity_m_s,
            self.energy_j_kg,
            self.safety_margin_m,
            self.angle_rad,
            self.command_rad,
            self.gate_miss_m,
            self.useful_lift_m_s,
        )
        if not np.isfinite(values).all() or any(float(value) <= 0.0 for value in values):
            raise ValueError("Feature scales must be positive and finite.")


def _mean_positive_vertical_wind(
    positions_w_m: np.ndarray,
    wind_model: object | None,
) -> float | None:
    if wind_model is None or not hasattr(wind_model, "velocity_at"):
        return None
    wind = np.asarray(wind_model.velocity_at(positions_w_m), dtype=float).reshape(-1, 3)
    return float(np.mean(np.maximum(wind[:, 2], 0.0)))


def extract_behaviour_feature(
    result: PrimitiveValidationResult,
    wind_model: object | None = None,
) -> PrimitiveBehaviourFeature:
    """Extract one serialisable behaviour feature from a validation result."""

    states = result.rollout.states
    if not states:
        raise ValueError("Feature extraction requires at least one rollout state.")
    start = states[0]
    end = states[-1]
    positions = np.asarray([state.position_w_m for state in states], dtype=float)
    displacement = end.position_w_m - start.position_w_m
    velocity_delta = end.velocity_b_m_s - start.velocity_b_m_s
    evidence = result.evidence
    return PrimitiveBehaviourFeature(
        primitive_id=result.primitive_id,
        family=result.family,
        controller_type=result.controller_type,
        entry_class=result.entry_class.label,
        exit_class=result.exit_class.label,
        scenario_id=result.scenario_id,
        seed=result.seed,
        retained=result.retention.retained,
        retention_reason=result.retention.reason,
        rollout_duration_s=evidence.rollout_duration_s,
        terminal_displacement_w_m=_tuple3(displacement),
        terminal_velocity_delta_b_m_s=_tuple3(velocity_delta),
        terminal_specific_energy_change_j_kg=float(evidence.terminal_specific_energy_change_j_kg),
        terminal_specific_energy_margin_j_kg=evidence.terminal_specific_energy_margin_j_kg,
        min_safety_margin_m=float(evidence.min_safety_margin_m),
        max_angle_of_attack_rad=float(evidence.max_angle_of_attack_rad),
        max_command_abs_rad=float(evidence.max_command_abs_rad),
        gate_miss_distance_m=evidence.gate_miss_distance_m,
        mean_positive_vertical_wind_m_s=_mean_positive_vertical_wind(positions, wind_model),
    )


def feature_vector(feature: PrimitiveBehaviourFeature, scale: FeatureScaleSpec | None = None) -> np.ndarray:
    """Return a normalised numeric feature vector for compression."""

    scales = FeatureScaleSpec() if scale is None else scale
    scales.validate()
    gate_miss = 0.0 if feature.gate_miss_distance_m is None else float(feature.gate_miss_distance_m)
    task_energy = (
        0.0
        if feature.terminal_specific_energy_margin_j_kg is None
        else float(feature.terminal_specific_energy_margin_j_kg)
    )
    lift = 0.0 if feature.mean_positive_vertical_wind_m_s is None else float(feature.mean_positive_vertical_wind_m_s)
    return np.asarray(
        [
            *(np.asarray(feature.terminal_displacement_w_m, dtype=float) / scales.displacement_m),
            *(np.asarray(feature.terminal_velocity_delta_b_m_s, dtype=float) / scales.velocity_m_s),
            feature.terminal_specific_energy_change_j_kg / scales.energy_j_kg,
            task_energy / scales.energy_j_kg,
            feature.min_safety_margin_m / scales.safety_margin_m,
            feature.max_angle_of_attack_rad / scales.angle_rad,
            feature.max_command_abs_rad / scales.command_rad,
            gate_miss / scales.gate_miss_m,
            lift / scales.useful_lift_m_s,
        ],
        dtype=float,
    )
