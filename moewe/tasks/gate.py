"""Gate-traversal task geometry and rollout evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from moewe.sim.frames import body_to_world_rows
from moewe.sim.glider_model import GliderModel, nominal_glider
from moewe.sim.state import FlightState

from .metrics import FailureReason, GateTaskMetrics, specific_energy_j_kg
from .scenario import (
    FRONT_EXIT_GATE_CENTRE_W_M,
    FRONT_EXIT_GATE_HEIGHT_M,
    FRONT_EXIT_GATE_NORMAL_W,
    FRONT_EXIT_GATE_WIDTH_M,
    FlightVolume,
)

EPS = 1e-12
BODY_AXES_B = np.eye(3)


def _unit(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(values))
    if norm <= EPS:
        raise ValueError("Gate direction vectors must be non-zero.")
    return values / norm


@dataclass(frozen=True)
class _AirframeEnvelope:
    half_length_m: float
    half_span_m: float
    half_height_m: float


def _airframe_envelope(model: GliderModel) -> _AirframeEnvelope:
    strips = model.strips
    centres = np.asarray(strips.r_strip_b_m, dtype=float)
    chord = np.asarray(strips.chord_m, dtype=float)
    x_min = float(np.min(centres[:, 0] - chord))
    x_max = float(np.max(centres[:, 0] + chord))
    z_min = float(np.min(centres[:, 2] - 0.5 * chord))
    z_max = float(np.max(centres[:, 2] + 0.5 * chord))
    return _AirframeEnvelope(
        half_length_m=max(abs(x_min), abs(x_max)),
        half_span_m=0.5 * float(model.b_ref_m),
        half_height_m=max(abs(z_min), abs(z_max)),
    )


def _interpolate_state(prev: FlightState, curr: FlightState, fraction: float) -> FlightState:
    alpha = float(np.clip(fraction, 0.0, 1.0))
    return FlightState.from_vector((1.0 - alpha) * prev.as_vector() + alpha * curr.as_vector())


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
    require_airframe_clearance: bool = True
    clearance_margin_m: float = 0.0

    def _crossing_fraction(self, start_w_m: np.ndarray, end_w_m: np.ndarray) -> float | None:
        d0 = self.gate.signed_distance_m(start_w_m)
        d1 = self.gate.signed_distance_m(end_w_m)
        if d0 == 0.0:
            return 0.0
        if d0 * d1 > 0.0:
            return None
        denom = d0 - d1
        return 0.0 if abs(denom) <= EPS else float(np.clip(d0 / denom, 0.0, 1.0))

    def _airframe_gate_miss_distance_m(self, state: FlightState, model: GliderModel) -> float:
        lateral, vertical = self.gate.coordinates_m(state.position_w_m)
        envelope = _airframe_envelope(model)
        body_axes_w = body_to_world_rows(BODY_AXES_B, state.euler_rad)
        half_extents = np.array(
            [envelope.half_length_m, envelope.half_span_m, envelope.half_height_m],
            dtype=float,
        )
        lateral_extent = float(np.sum(np.abs(body_axes_w @ self.gate.lateral_axis_w) * half_extents))
        vertical_extent = float(np.sum(np.abs(body_axes_w @ self.gate.vertical_axis_w) * half_extents))
        margin = max(float(self.clearance_margin_m), 0.0)
        lateral_excess = max(abs(lateral) + lateral_extent + margin - 0.5 * float(self.gate.width_m), 0.0)
        vertical_excess = max(abs(vertical) + vertical_extent + margin - 0.5 * float(self.gate.height_m), 0.0)
        return float(np.hypot(lateral_excess, vertical_excess))

    def _gate_miss_distance_m(self, state: FlightState, model: GliderModel) -> float:
        if self.require_airframe_clearance:
            return self._airframe_gate_miss_distance_m(state, model)
        return self.gate.miss_distance_m(state.position_w_m)

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
        gate_model = nominal_glider() if model is None else model
        gate_crossed = False
        crossing_index: int | None = None
        crossing_point: np.ndarray | None = None
        best_miss = float("inf")
        for index, (prev, curr) in enumerate(zip(state_list[:-1], state_list[1:]), start=1):
            if not prev.finite() or not curr.finite():
                continue
            crossed, point, miss = self.gate.segment_crossing(prev.position_w_m, curr.position_w_m)
            prev_miss = self._gate_miss_distance_m(prev, gate_model)
            curr_miss = self._gate_miss_distance_m(curr, gate_model)
            point_miss = float("inf") if self.require_airframe_clearance else float(miss)
            best_miss = min(best_miss, point_miss, prev_miss, curr_miss)
            crossing_fraction = self._crossing_fraction(prev.position_w_m, curr.position_w_m) if crossed else None
            crossing_state = None if crossing_fraction is None else _interpolate_state(prev, curr, crossing_fraction)
            crossing_miss = float("inf") if crossing_state is None else self._gate_miss_distance_m(crossing_state, gate_model)
            if crossed and crossing_miss <= EPS:
                gate_crossed = True
                crossing_index = index
                crossing_point = point
                best_miss = 0.0
                break
            best_miss = min(best_miss, crossing_miss)
        if len(state_list) == 1:
            best_miss = self._gate_miss_distance_m(state_list[0], gate_model)

        metric_states = state_list if crossing_index is None else state_list[: crossing_index + 1]
        volume_positions = [state.position_w_m for state in metric_states]
        if crossing_point is not None:
            volume_positions[-1] = crossing_point
        min_margin = min(self.flight_volume.min_margin_m(position) for position in volume_positions)
        non_finite = any(not state.finite() for state in metric_states)
        max_alpha = self._max_angle_of_attack(metric_states, model, wind_model, wind_mode)
        flight_time = float(dt_s) * float(max(len(metric_states) - 1, 0))
        terminal_energy_margin = (
            specific_energy_j_kg(metric_states[-1]) - float(self.required_terminal_specific_energy_j_kg)
        )

        failure = FailureReason.NONE
        if non_finite:
            failure = FailureReason.NON_FINITE_STATE
        elif self.angle_of_attack_limit_rad is not None and max_alpha > float(self.angle_of_attack_limit_rad):
            failure = FailureReason.STALL_LIMIT
        else:
            for position in volume_positions:
                reason = self.flight_volume.failure_reason(position)
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


def front_exit_gate() -> GatePlane:
    """Return the measured rectangular exit gate on the front wall."""

    return GatePlane(
        centre_w_m=np.array(FRONT_EXIT_GATE_CENTRE_W_M, dtype=float),
        normal_w=np.array(FRONT_EXIT_GATE_NORMAL_W, dtype=float),
        width_m=FRONT_EXIT_GATE_WIDTH_M,
        height_m=FRONT_EXIT_GATE_HEIGHT_M,
    )
