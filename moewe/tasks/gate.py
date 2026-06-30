"""Gate-traversal task geometry and rollout evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from moewe.sim.glider_model import GliderModel
from moewe.sim.state import FlightState

from .metrics import FailureReason, GateTaskMetrics, specific_energy_j_kg
from .scenario import FlightVolume

EPS = 1e-12


def _unit(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(values))
    if norm <= EPS:
        raise ValueError("Gate direction vectors must be non-zero.")
    return values / norm


@dataclass(frozen=True)
class GatePlane:
    """Rectangular gate embedded in a plane in world z-up coordinates."""

    centre_w_m: np.ndarray
    normal_w: np.ndarray
    width_m: float
    height_m: float
    up_hint_w: np.ndarray = field(default_factory=lambda: np.array((0.0, 0.0, 1.0)))

    def __post_init__(self) -> None:
        centre = np.asarray(self.centre_w_m, dtype=float).reshape(3)
        normal = _unit(self.normal_w)
        up_hint = _unit(self.up_hint_w)
        up_axis = up_hint - np.dot(up_hint, normal) * normal
        if np.linalg.norm(up_axis) <= EPS:
            raise ValueError("Gate up_hint_w must not be parallel to normal_w.")
        up_axis = _unit(up_axis)
        lateral_axis = _unit(np.cross(up_axis, normal))
        if self.width_m <= 0.0 or self.height_m <= 0.0:
            raise ValueError("Gate width and height must be positive.")
        object.__setattr__(self, "centre_w_m", centre)
        object.__setattr__(self, "normal_w", normal)
        object.__setattr__(self, "up_hint_w", up_axis)
        object.__setattr__(self, "_lateral_axis_w", lateral_axis)
        object.__setattr__(self, "_vertical_axis_w", up_axis)

    @property
    def lateral_axis_w(self) -> np.ndarray:
        return self._lateral_axis_w

    @property
    def vertical_axis_w(self) -> np.ndarray:
        return self._vertical_axis_w

    def signed_distance_m(self, point_w_m: np.ndarray) -> float:
        return float(np.dot(np.asarray(point_w_m, dtype=float).reshape(3) - self.centre_w_m, self.normal_w))

    def coordinates_m(self, point_w_m: np.ndarray) -> tuple[float, float]:
        offset = np.asarray(point_w_m, dtype=float).reshape(3) - self.centre_w_m
        return float(np.dot(offset, self.lateral_axis_w)), float(np.dot(offset, self.vertical_axis_w))

    def miss_distance_m(self, point_w_m: np.ndarray) -> float:
        lateral, vertical = self.coordinates_m(point_w_m)
        lateral_excess = max(abs(lateral) - 0.5 * float(self.width_m), 0.0)
        vertical_excess = max(abs(vertical) - 0.5 * float(self.height_m), 0.0)
        return float(np.hypot(lateral_excess, vertical_excess))

    def contains_projection(self, point_w_m: np.ndarray) -> bool:
        return self.miss_distance_m(point_w_m) <= EPS

    def segment_crossing(self, start_w_m: np.ndarray, end_w_m: np.ndarray) -> tuple[bool, np.ndarray, float]:
        start = np.asarray(start_w_m, dtype=float).reshape(3)
        end = np.asarray(end_w_m, dtype=float).reshape(3)
        d0 = self.signed_distance_m(start)
        d1 = self.signed_distance_m(end)
        if d0 == 0.0:
            return True, start, 0.0
        if d0 * d1 > 0.0:
            closest = start if abs(d0) < abs(d1) else end
            return False, closest, self.miss_distance_m(closest)
        denom = d0 - d1
        fraction = 0.0 if abs(denom) <= EPS else d0 / denom
        fraction = float(np.clip(fraction, 0.0, 1.0))
        point = start + fraction * (end - start)
        return True, point, self.miss_distance_m(point)


@dataclass(frozen=True)
class GateTraversalTask:
    """Evaluate a trajectory against a rectangular gate and safety envelope."""

    gate: GatePlane
    flight_volume: FlightVolume
    timeout_s: float
    required_terminal_specific_energy_j_kg: float = 0.0
    angle_of_attack_limit_rad: float | None = None

    def _max_angle_of_attack(
        self,
        states: list[FlightState],
        model: GliderModel | None,
        wind_model: object | None,
        wind_mode: str,
    ) -> float:
        values: list[float] = []
        for state in states:
            if model is None:
                alpha = float(np.arctan2(state.velocity_b_m_s[2], state.velocity_b_m_s[0]))
            else:
                alpha = float(model.evaluate_aero(state, wind_model=wind_model, wind_mode=wind_mode).alpha_rad)
            values.append(abs(alpha))
        return float(max(values)) if values else float("nan")

    def evaluate(
        self,
        states: list[FlightState] | tuple[FlightState, ...],
        dt_s: float,
        model: GliderModel | None = None,
        wind_model: object | None = None,
        wind_mode: str = "panel",
    ) -> GateTaskMetrics:
        if dt_s <= 0.0:
            raise ValueError("Task evaluation dt_s must be positive.")
        if not states:
            raise ValueError("Task evaluation requires at least one state.")

        state_list = list(states)
        min_margin = min(self.flight_volume.min_margin_m(state.position_w_m) for state in state_list)
        non_finite = any(not state.finite() for state in state_list)
        max_alpha = self._max_angle_of_attack(state_list, model, wind_model, wind_mode)
        flight_time = float(dt_s) * float(max(len(state_list) - 1, 0))
        terminal_energy_margin = (
            specific_energy_j_kg(state_list[-1]) - float(self.required_terminal_specific_energy_j_kg)
        )

        gate_crossed = False
        best_miss = float("inf")
        for prev, curr in zip(state_list[:-1], state_list[1:]):
            crossed, point, miss = self.gate.segment_crossing(prev.position_w_m, curr.position_w_m)
            best_miss = min(best_miss, float(miss), self.gate.miss_distance_m(prev.position_w_m), self.gate.miss_distance_m(curr.position_w_m))
            if crossed and self.gate.contains_projection(point):
                gate_crossed = True
                best_miss = 0.0
                break
        if len(state_list) == 1:
            best_miss = self.gate.miss_distance_m(state_list[0].position_w_m)

        failure = FailureReason.NONE
        if non_finite:
            failure = FailureReason.NON_FINITE_STATE
        elif self.angle_of_attack_limit_rad is not None and max_alpha > float(self.angle_of_attack_limit_rad):
            failure = FailureReason.STALL_LIMIT
        else:
            for state in state_list:
                reason = self.flight_volume.failure_reason(state.position_w_m)
                if reason is not None:
                    failure = FailureReason(reason)
                    break
            if failure == FailureReason.NONE and not gate_crossed:
                failure = FailureReason.TIMEOUT

        success = gate_crossed and failure == FailureReason.NONE
        return GateTaskMetrics(
            success=success,
            gate_crossed=gate_crossed,
            gate_miss_distance_m=float(best_miss),
            min_safety_margin_m=float(min_margin),
            flight_time_s=flight_time,
            terminal_specific_energy_margin_j_kg=float(terminal_energy_margin),
            max_angle_of_attack_rad=max_alpha,
            failure_reason=failure,
        )
