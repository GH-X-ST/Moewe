"""Motion-only aircraft-state and center-flow observer."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from math import ceil, pi, sin, sqrt

import numpy as np
import numpy.typing as npt

from control.flow import FlowBounds
from control.interval import Interval, Zonotope
from control.predictor import (
    INTRINSIC_FEEDBACK_INDICES,
    MAX_INITIAL_GENERATORS,
    GeneratedAircraft,
)
from control.uncertainty import OBSERVER_PERIOD_S, PREDICTION_PERIOD_S
from models.geometry import body_to_world

AIRCRAFT_STATE_SIZE = 15
OBSERVER_STATE_SIZE = 18
POSE_SIZE = 6

_AIRCRAFT_IDENTITY = np.eye(AIRCRAFT_STATE_SIZE)
_OBSERVER_IDENTITY = np.eye(OBSERVER_STATE_SIZE)
_ATTITUDE_OFFSETS = 1.0e-6 * np.eye(3)
_ZERO_FLOW_DERIVATIVE = np.zeros(3)


@dataclass(frozen=True)
class ObserverCalibration:
    """Filter tuning and calibrated governor-time integrity bounds."""

    process_variance_per_s: npt.ArrayLike
    pose_variance: npt.ArrayLike
    initial_variance: npt.ArrayLike
    integrity_generators: npt.ArrayLike
    innovation_abs: npt.ArrayLike
    flow_change_abs_m_s: npt.ArrayLike
    latency_max_s: float
    sample_gap_max_s: float
    nominal_delay_s: float
    initialization_samples: int

    def __post_init__(self) -> None:
        _set_array(self, "process_variance_per_s", self.process_variance_per_s, 18)
        _set_array(self, "pose_variance", self.pose_variance, 6)
        _set_array(self, "initial_variance", self.initial_variance, 18)
        _set_matrix(self, "integrity_generators", self.integrity_generators, 18)
        _set_array(self, "innovation_abs", self.innovation_abs, 6)
        _set_array(self, "flow_change_abs_m_s", self.flow_change_abs_m_s, 3)
        for name in (
            "process_variance_per_s",
            "pose_variance",
            "initial_variance",
            "innovation_abs",
            "flow_change_abs_m_s",
        ):
            if np.any(getattr(self, name) < 0.0):
                raise ValueError(f"{name} must be nonnegative")
        if not np.isfinite(self.latency_max_s) or self.latency_max_s < 0.0:
            raise ValueError("latency_max_s must be finite and nonnegative")
        if (
            not np.isfinite(self.sample_gap_max_s)
            or self.sample_gap_max_s < OBSERVER_PERIOD_S
        ):
            raise ValueError("sample_gap_max_s must include one observer period")
        if not np.isfinite(self.nominal_delay_s):
            raise ValueError("nominal_delay_s must be finite")
        samples = int(self.initialization_samples)
        if samples < 1 or samples != self.initialization_samples:
            raise ValueError("initialization_samples must be a positive integer")
        object.__setattr__(self, "initialization_samples", samples)


@dataclass(frozen=True)
class IssuedCommandHistory:
    """Timestamped control-surface targets in issue order."""

    timestamps_s: npt.ArrayLike
    commands_rad: npt.ArrayLike

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps_s, dtype=float).reshape(-1).copy()
        commands = np.asarray(self.commands_rad, dtype=float).reshape(
            timestamps.size,
            3,
        ).copy()
        if timestamps.size == 0 or np.any(np.diff(timestamps) <= 0.0):
            raise ValueError("command timestamps must be strictly increasing")
        if not np.all(np.isfinite(timestamps)) or not np.all(np.isfinite(commands)):
            raise ValueError("command history must be finite")
        timestamps.flags.writeable = False
        commands.flags.writeable = False
        object.__setattr__(self, "timestamps_s", timestamps)
        object.__setattr__(self, "commands_rad", commands)

    def command_at(self, timestamp_s: float) -> np.ndarray:
        """Return the latest command issued by a timestamp."""

        index = int(np.searchsorted(self.timestamps_s, timestamp_s, side="right") - 1)
        if index < 0:
            raise ValueError("command history does not cover the delayed timestamp")
        return self.commands_rad[index]


@dataclass(frozen=True)
class StateFlowEstimate:
    """Joint world-frame integrity set and body-frame flow enclosure."""

    timestamp_s: float
    joint: Zonotope
    local_flow: FlowBounds
    state: Zonotope = field(init=False)
    center_flow_w: Zonotope = field(init=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.timestamp_s):
            raise ValueError("estimate timestamp must be finite")
        if self.joint.center.size != OBSERVER_STATE_SIZE:
            raise ValueError("joint state-flow estimate must have 18 components")
        state_columns = np.any(self.joint.generators[:15] != 0.0, axis=0)
        flow_columns = np.any(self.joint.generators[15:] != 0.0, axis=0)
        object.__setattr__(
            self,
            "state",
            Zonotope(
                self.joint.center[:15],
                self.joint.generators[:15, state_columns],
            ),
        )
        object.__setattr__(
            self,
            "center_flow_w",
            Zonotope(
                self.joint.center[15:],
                self.joint.generators[15:, flow_columns],
            ),
        )
        state_radius = self.state.radius
        attitude = Interval.from_midpoint(
            self.state.center[3:6],
            state_radius[3:6],
        )
        flow_radius = self.center_flow_w.radius
        world_flow = Interval.from_midpoint(
            self.center_flow_w.center,
            flow_radius,
        )
        body_flow = _body_flow_bounds(attitude, world_flow)
        local_center = Interval(
            self.local_flow.center_lower_m_s,
            self.local_flow.center_upper_m_s,
        )
        if not body_flow.subset(local_center):
            raise ValueError("local flow must contain the joint-set projection")


class MotionFlowObserver:
    """Estimate aircraft state and world-frame center flow from raw pose."""

    def __init__(
        self,
        generated: GeneratedAircraft,
        calibration: ObserverCalibration,
        initial_state: npt.ArrayLike,
        timestamp_s: float,
    ) -> None:
        if not isinstance(generated, GeneratedAircraft):
            raise TypeError("observer requires an oracle-verified aircraft core")
        bounds = generated.bounds
        flow = bounds.flow
        if not np.array_equal(flow.center_lower_m_s, -flow.center_upper_m_s):
            raise ValueError("observer center-flow bounds must be symmetric")
        if not np.array_equal(flow.gradient_lower_s, -flow.gradient_upper_s):
            raise ValueError("observer gradient bounds must be symmetric")
        if not (
            bounds.command_delay_s[0]
            <= calibration.nominal_delay_s
            <= bounds.command_delay_s[1]
        ):
            raise ValueError("nominal delay must lie inside the certified interval")
        state_radius = Zonotope(
            np.zeros(15),
            calibration.integrity_generators[:15],
        ).radius
        if np.any(state_radius > bounds.state_estimation_abs):
            raise ValueError("observer integrity exceeds the certified state bound")
        state_generator_count = np.count_nonzero(
            np.any(calibration.integrity_generators[:15] != 0.0, axis=0)
        )
        if state_generator_count > MAX_INITIAL_GENERATORS:
            raise ValueError("observer integrity exceeds predictor generator capacity")
        flow_radius = Zonotope(
            np.zeros(3),
            calibration.integrity_generators[15:],
        ).radius
        if np.any(flow_radius > flow.center_upper_m_s):
            raise ValueError("observer flow integrity exceeds the certified envelope")

        self.generated = generated
        self.calibration = calibration
        self._mean = np.concatenate(
            (np.asarray(initial_state, dtype=float).reshape(15), np.zeros(3))
        )
        self._covariance = np.diag(calibration.initial_variance)
        self._timestamp_s = float(timestamp_s)
        if not np.isfinite(self._timestamp_s):
            raise ValueError("observer timestamp must be finite")
        self._innovation_contained = False
        self._contained_updates = 0
        self._has_sample = False
        self._cadence_valid = True
        self._cell_index = generated.cell(self._mean[:15])[0]
        self._density_kg_m3 = 0.5 * sum(bounds.density_kg_m3)
        self._process_covariance_per_s = np.diag(
            calibration.process_variance_per_s
        )
        self._pose_covariance = np.diag(calibration.pose_variance)

    def update(
        self,
        pose: npt.ArrayLike,
        sample_timestamp_s: float,
        commands: IssuedCommandHistory,
    ) -> bool:
        """Process one timestamped raw-pose sample."""

        sample_time = float(sample_timestamp_s)
        if not np.isfinite(sample_time):
            raise ValueError("pose timestamp must be finite")
        if self._has_sample and sample_time <= self._timestamp_s:
            raise ValueError("pose timestamps must be strictly increasing")
        if sample_time < self._timestamp_s:
            raise ValueError("pose timestamp precedes observer initialization")
        if (
            sample_time - self._timestamp_s
            > np.nextafter(self.calibration.sample_gap_max_s, np.inf)
        ):
            self._innovation_contained = False
            self._contained_updates = 0
            self._cadence_valid = False
            return False

        self._propagate_filter(sample_time, commands)
        self._has_sample = True
        measurement = np.asarray(pose, dtype=float).reshape(POSE_SIZE)
        innovation = measurement - self._mean[:POSE_SIZE]
        innovation[3:6] = _wrap_angles(innovation[3:6])
        self._innovation_contained = bool(
            np.all(np.abs(innovation) <= self.calibration.innovation_abs)
        )
        if self._innovation_contained:
            self._correct(innovation)
            self._contained_updates += 1
        else:
            self._contained_updates = 0
        return self._innovation_contained

    def estimate(
        self,
        governor_timestamp_s: float,
        commands: IssuedCommandHistory,
    ) -> StateFlowEstimate | None:
        """Return a containing state-flow set at governor time."""

        governor_time = float(governor_timestamp_s)
        if governor_time < self._timestamp_s:
            raise ValueError("governor time precedes the pose timestamp")
        age = governor_time - self._timestamp_s
        if (
            not self._innovation_contained
            or not self._cadence_valid
            or self._contained_updates < self.calibration.initialization_samples
            or age > self.calibration.latency_max_s
        ):
            return None
        mean = self._propagate_mean(
            self._mean.copy(),
            self._timestamp_s,
            governor_time,
            commands,
        )
        return self._state_flow_estimate(mean, governor_time)

    def _propagate_filter(
        self,
        timestamp_s: float,
        commands: IssuedCommandHistory,
    ) -> None:
        duration = timestamp_s - self._timestamp_s
        for start, step in _steps(self._timestamp_s, duration):
            command = commands.command_at(
                start + 0.5 * step - self.calibration.nominal_delay_s
            )
            transition = self._transition(self._mean, step)
            self._mean = self._step(self._mean, command, step)
            self._covariance = (
                transition @ self._covariance @ transition.T
                + self._process_covariance_per_s * step
            )
            self._covariance = 0.5 * (
                self._covariance + self._covariance.T
            )
        self._timestamp_s = timestamp_s

    def _propagate_mean(
        self,
        mean: np.ndarray,
        start_s: float,
        end_s: float,
        commands: IssuedCommandHistory,
    ) -> np.ndarray:
        for start, step in _steps(start_s, end_s - start_s):
            command = commands.command_at(
                start + 0.5 * step - self.calibration.nominal_delay_s
            )
            mean = self._step(mean, command, step)
        return mean

    def _correct(self, innovation: np.ndarray) -> None:
        residual_covariance = self._covariance[:6, :6] + self._pose_covariance
        gain = np.linalg.solve(
            residual_covariance,
            self._covariance[:6],
        ).T
        self._mean += gain @ innovation
        self._mean[3:6] = _wrap_angles(self._mean[3:6])
        complement = _OBSERVER_IDENTITY.copy()
        complement[:, :6] -= gain
        self._covariance = (
            complement @ self._covariance @ complement.T
            + gain @ self._pose_covariance @ gain.T
        )

    def _transition(self, mean: np.ndarray, step_s: float) -> np.ndarray:
        cell_index = self._cell_index
        cell = self.generated.cells[cell_index]
        intrinsic = (
            mean[INTRINSIC_FEEDBACK_INDICES]
            - cell.anchor[INTRINSIC_FEEDBACK_INDICES]
        ) / self.generated.state_scale[INTRINSIC_FEEDBACK_INDICES]
        if np.any(intrinsic < cell.lower) or np.any(intrinsic > cell.upper):
            cell_index, cell = self.generated.cell(mean[:15])
            self._cell_index = cell_index
        ratio = step_s / OBSERVER_PERIOD_S
        transition = _OBSERVER_IDENTITY.copy()
        transition[:15, :15] = _AIRCRAFT_IDENTITY + ratio * (
            cell.observer_state_matrix - _AIRCRAFT_IDENTITY
        )

        response_b = ratio * cell.observer_flow_matrix
        rotation = body_to_world(mean[3:6])
        transition[:15, 15:] = response_b @ rotation.T
        for axis in range(3):
            offset = _ATTITUDE_OFFSETS[axis]
            upper = body_to_world(mean[3:6] + offset).T @ mean[15:]
            lower = body_to_world(mean[3:6] - offset).T @ mean[15:]
            transition[:15, 3 + axis] += response_b @ (
                (upper - lower) / (2.0e-6)
            )
        return transition

    def _step(
        self,
        mean: np.ndarray,
        command: np.ndarray,
        step_s: float,
    ) -> np.ndarray:
        initial = self._derivative(mean, command)
        midpoint = mean + 0.5 * step_s * initial
        result = mean + step_s * self._derivative(midpoint, command)
        result[3:6] = _wrap_angles(result[3:6])
        return result

    def _derivative(self, mean: np.ndarray, command: np.ndarray) -> np.ndarray:
        state = mean[:15]
        center_b = body_to_world(state[3:6]).T @ mean[15:]
        strip_b = np.broadcast_to(
            center_b,
            self.generated.aircraft.strip_table.r_b_m.shape,
        )
        state_derivative = self.generated.aircraft.derivative_local_flow(
            state,
            command,
            center_b,
            strip_b,
            self._density_kg_m3,
        )
        return np.concatenate((state_derivative, _ZERO_FLOW_DERIVATIVE))

    def _state_flow_estimate(
        self,
        mean: np.ndarray,
        timestamp_s: float,
    ) -> StateFlowEstimate | None:
        joint = Zonotope(mean, self.calibration.integrity_generators)
        state_radius = Zonotope(mean[:15], joint.generators[:15]).radius
        attitude = Interval.from_midpoint(mean[3:6], state_radius[3:6])
        flow_radius = Zonotope(mean[15:], joint.generators[15:]).radius
        flow_steps = ceil(PREDICTION_PERIOD_S / OBSERVER_PERIOD_S)
        flow_change = np.nextafter(
            flow_steps * self.calibration.flow_change_abs_m_s,
            np.inf,
        )
        flow_radius = np.where(
            flow_change == 0.0,
            flow_radius,
            np.nextafter(flow_radius + flow_change, np.inf),
        )
        world_flow = Interval.from_midpoint(mean[15:], flow_radius)
        body_flow = _body_flow_bounds(attitude, world_flow)
        max_world_flow = np.maximum(
            np.abs(world_flow.lower),
            np.abs(world_flow.upper),
        )
        world_norm = float(
            Interval.point(max_world_flow).square().sum().sqrt().upper
        )
        angle = min(
            pi,
            sqrt(3.0)
            * self.generated.bounds.body_rate_abs_max_rad_s
            * PREDICTION_PERIOD_S,
        )
        rotation_change = np.nextafter(
            2.0 * sin(0.5 * angle) * world_norm,
            np.inf,
        )
        body_flow += Interval(
            np.full(3, -rotation_change),
            np.full(3, rotation_change),
        )
        certified = self.generated.bounds.flow
        local_flow = FlowBounds(
            body_flow.lower,
            body_flow.upper,
            certified.gradient_lower_s,
            certified.gradient_upper_s,
            certified.remainder_abs_m_s,
        )
        if not local_flow.subset(certified):
            return None
        return StateFlowEstimate(timestamp_s, joint, local_flow)


def _steps(start_s: float, duration_s: float) -> Iterator[tuple[float, float]]:
    if duration_s == 0.0:
        return
    count = ceil(duration_s / OBSERVER_PERIOD_S)
    step = duration_s / count
    for index in range(count):
        yield start_s + index * step, step


def _body_flow_bounds(attitude: Interval, flow_w: Interval) -> Interval:
    phi, theta, psi = attitude[0], attitude[1], attitude[2]
    c_phi, s_phi = phi.cos(), phi.sin()
    c_theta, s_theta = theta.cos(), theta.sin()
    c_psi, s_psi = psi.cos(), psi.sin()
    x_w, y_w, z_w = flow_w[0], flow_w[1], flow_w[2]
    body = (
        c_theta * c_psi * x_w - c_theta * s_psi * y_w + s_theta * z_w,
        (s_phi * s_theta * c_psi - c_phi * s_psi) * x_w
        + (-s_phi * s_theta * s_psi - c_phi * c_psi) * y_w
        - s_phi * c_theta * z_w,
        (c_phi * s_theta * c_psi + s_phi * s_psi) * x_w
        + (-c_phi * s_theta * s_psi + s_phi * c_psi) * y_w
        - c_phi * c_theta * z_w,
    )
    return Interval(
        np.array([float(component.lower) for component in body]),
        np.array([float(component.upper) for component in body]),
    )


def _wrap_angles(angles: npt.ArrayLike) -> np.ndarray:
    values = np.asarray(angles, dtype=float)
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _set_array(
    instance: object,
    name: str,
    value: npt.ArrayLike,
    size: int,
) -> None:
    array = np.asarray(value, dtype=float).reshape(size).copy()
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.flags.writeable = False
    object.__setattr__(instance, name, array)


def _set_matrix(
    instance: object,
    name: str,
    value: npt.ArrayLike,
    rows: int,
) -> None:
    array = np.asarray(value, dtype=float)
    columns = array.shape[1] if array.ndim == 2 else array.size // rows
    array = array.reshape(rows, columns).copy()
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.flags.writeable = False
    object.__setattr__(instance, name, array)
