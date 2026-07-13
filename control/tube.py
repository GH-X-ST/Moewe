"""Deterministic state and occupied-body tube propagation."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from math import pi, radians
from time import monotonic

import numpy as np
import numpy.typing as npt

from control.flow import FlowBounds
from control.interval import Interval, Zonotope
from models.aircraft import (
    G_M_S2,
    SMOOTH_ABS_EPS,
    Aircraft,
    flat_plate_coefficients,
)
from models.geometry import RigidBodyGeometry


class TubeCertificationError(RuntimeError):
    """Raised when deterministic propagation cannot certify a segment."""


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and monotonic() >= deadline:
        raise TimeoutError("tube verification deadline expired")


@dataclass(frozen=True)
class FlightDomain:
    """Hardware and validated aerodynamic-model limits."""

    roll_abs_max_rad: float
    pitch_abs_max_rad: float
    airspeed_bounds_m_s: tuple[float, float]
    alpha_abs_max_rad: float
    body_rate_abs_max_rad_s: float

    def contains(self, state: Interval, aircraft: Aircraft) -> bool:
        """Return whether a state tube lies inside the hard domain."""

        rates = state[9:12]
        surfaces = state[12:15]
        return bool(
            state.lower[3] >= -self.roll_abs_max_rad
            and state.upper[3] <= self.roll_abs_max_rad
            and state.lower[4] >= -self.pitch_abs_max_rad
            and state.upper[4] <= self.pitch_abs_max_rad
            and np.all(rates.lower >= -self.body_rate_abs_max_rad_s)
            and np.all(rates.upper <= self.body_rate_abs_max_rad_s)
            and np.all(surfaces.lower >= aircraft.control_lower_rad)
            and np.all(surfaces.upper <= aircraft.control_upper_rad)
        )


@dataclass(frozen=True)
class ModelUncertainty:
    """Frozen flow, parameter, actuator, and residual uncertainty set."""

    flow: FlowBounds
    density_kg_m3: tuple[float, float]
    mass_kg: tuple[float, float]
    coefficient_scale: tuple[float, float]
    force_error_abs_n: npt.ArrayLike
    moment_error_abs_n_m: npt.ArrayLike
    angular_accel_error_abs_rad_s2: npt.ArrayLike
    actuator_tau_s: tuple[float, float]
    command_error_abs_rad: npt.ArrayLike
    command_delay_max_s: float = 0.0
    command_rate_abs_rad_s: npt.ArrayLike = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        ranges = (
            "density_kg_m3",
            "mass_kg",
            "coefficient_scale",
            "actuator_tau_s",
        )
        for name in ranges:
            values = tuple(float(value) for value in getattr(self, name))
            if (
                len(values) != 2
                or not np.all(np.isfinite(values))
                or values[0] > values[1]
            ):
                raise ValueError(f"{name} must be a nonempty interval")
            object.__setattr__(self, name, values)
        if (
            self.density_kg_m3[0] <= 0.0
            or self.mass_kg[0] <= 0.0
            or self.coefficient_scale[0] <= 0.0
            or self.actuator_tau_s[0] <= 0.0
        ):
            raise ValueError("physical scale intervals must be positive")
        delay = float(self.command_delay_max_s)
        if not np.isfinite(delay) or delay < 0.0:
            raise ValueError("command delay must be finite and nonnegative")
        object.__setattr__(self, "command_delay_max_s", delay)
        fields = (
            ("force_error_abs_n", self.force_error_abs_n),
            ("moment_error_abs_n_m", self.moment_error_abs_n_m),
            (
                "angular_accel_error_abs_rad_s2",
                self.angular_accel_error_abs_rad_s2,
            ),
            ("command_error_abs_rad", self.command_error_abs_rad),
            ("command_rate_abs_rad_s", self.command_rate_abs_rad_s),
        )
        for name, value in fields:
            array = np.asarray(value, dtype=float).reshape(3).copy()
            if np.any(array < 0.0) or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite and nonnegative")
            array.flags.writeable = False
            object.__setattr__(self, name, array)


@dataclass(frozen=True)
class FeedbackSegment:
    """Continuous command with local full-state feedback."""

    state: npt.ArrayLike
    control_rad: npt.ArrayLike
    gain: npt.ArrayLike

    def __post_init__(self) -> None:
        arrays = (
            ("state", self.state, (15,)),
            ("control_rad", self.control_rad, (3,)),
            ("gain", self.gain, (3, 15)),
        )
        for name, value, shape in arrays:
            array = np.asarray(value, dtype=float).reshape(shape).copy()
            array.flags.writeable = False
            object.__setattr__(self, name, array)

    def command(self, state: npt.ArrayLike, aircraft: Aircraft) -> np.ndarray:
        """Return the clipped feedback command at a measured state."""

        reference = np.asarray(self.state, dtype=float).reshape(15)
        control = np.asarray(self.control_rad, dtype=float).reshape(3)
        gain = np.asarray(self.gain, dtype=float).reshape(3, 15)
        return aircraft.clip_control(
            control + gain @ (np.asarray(state, dtype=float) - reference)
        )

    def command_box(
        self,
        state: Interval,
        aircraft: Aircraft,
        error_abs_rad: npt.ArrayLike,
    ) -> Interval:
        """Return all feedback commands over a state box."""

        reference = np.asarray(self.state, dtype=float).reshape(15)
        control = np.asarray(self.control_rad, dtype=float).reshape(3)
        gain = np.asarray(self.gain, dtype=float).reshape(3, 15)
        command = (state - reference).affine_map(gain, control)
        error = np.asarray(error_abs_rad, dtype=float).reshape(3)
        return (command + Interval(-error, error)).clip(
            aircraft.control_lower_rad,
            aircraft.control_upper_rad,
        )


@dataclass(frozen=True)
class BodyTube:
    """Continuous-time full-body and contact enclosures for one segment."""

    occupied: tuple[Interval, ...]
    contact: tuple[Interval, ...]
    footprint: tuple[Interval, ...]
    contact_velocity: tuple[Interval, ...]


@dataclass(frozen=True)
class SegmentTube:
    """Certified successor and inter-sample tube for one control segment."""

    initial: Zonotope
    successor: Zonotope
    states: tuple[Interval, ...]
    body: BodyTube
    durations_s: tuple[float, ...] = ()


def body_points(
    state: Interval | Zonotope,
    points_b_m: npt.ArrayLike,
) -> Interval:
    """Return a rigorous world enclosure for body-fixed points."""

    state_box = state.interval_hull() if isinstance(state, Zonotope) else state
    position = state_box[:3]
    rotation = _body_to_world(state_box[3:6])
    points = np.asarray(points_b_m, dtype=float).reshape(-1, 3)
    rotated = _interval_batch_matvec(rotation, Interval.point(points))
    return position + rotated


@dataclass
class TubePropagator:
    """Validated interval propagator for the stripwise aircraft model."""

    aircraft: Aircraft
    uncertainty: ModelUncertainty
    domain: FlightDomain
    geometry: RigidBodyGeometry
    substeps: int = 4
    picard_iterations: int = 16
    max_subdivisions: int = 4
    max_generators: int = 120
    _strip_flow: Interval = field(init=False, repr=False)
    _center_flow: Interval = field(init=False, repr=False)
    _locations: Interval = field(init=False, repr=False)
    _spans: Interval = field(init=False, repr=False)
    _projection: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        table = self.aircraft.strip_table
        strip_lower, strip_upper = self.uncertainty.flow.strip_bounds(table.r_b_m)
        self._strip_flow = Interval(strip_lower, strip_upper)
        self._center_flow = Interval(
            self.uncertainty.flow.center_lower_m_s,
            self.uncertainty.flow.center_upper_m_s,
        )
        self._locations = Interval.point(table.r_b_m)
        self._spans = Interval.point(table.span_axis_b)
        self._projection = (
            np.eye(3)[None, :, :]
            - table.span_axis_b[:, :, None] * table.span_axis_b[:, None, :]
        )

    def propagate(
        self,
        initial: Interval | Zonotope,
        segment: FeedbackSegment,
        dt_s: float,
        stop: Callable[[SegmentTube], bool] | None = None,
        deadline: float | None = None,
    ) -> SegmentTube:
        """Propagate one feedback segment with Picard self-enclosures."""

        initial_set = (
            initial
            if isinstance(initial, Zonotope)
            else Zonotope.from_interval(initial)
        )
        state = initial_set
        states = []
        occupied = []
        contact = []
        footprint = []
        contact_velocity = []
        durations = []

        def tube() -> SegmentTube:
            return SegmentTube(
                initial_set,
                state,
                tuple(states),
                BodyTube(
                    tuple(occupied),
                    tuple(contact),
                    tuple(footprint),
                    tuple(contact_velocity),
                ),
                tuple(durations),
            )

        step_s = dt_s / self.substeps
        for _ in range(self.substeps):
            _check_deadline(deadline)
            pieces = self._adaptive_step(state, segment, step_s, 0, deadline)
            for successor, continuous, duration in pieces:
                state = successor
                states.append(continuous)
                durations.append(duration)
                occupied.append(body_points(continuous, self.geometry.body_b_m))
                contact.append(body_points(continuous, self.geometry.contact_b_m))
                footprint.append(body_points(continuous, self.geometry.footprint_b_m))
                contact_velocity.append(
                    self._point_velocities(
                        continuous,
                        self.geometry.contact_b_m,
                    )
                )
                current = tube()
                if stop is not None and stop(current):
                    return current
        return tube()

    def propagate_plan(
        self,
        initial: Interval | Zonotope,
        segments: tuple[FeedbackSegment, ...],
        dt_s: float,
        terminal: Callable[[tuple[SegmentTube, ...]], bool] | None = None,
        deadline: float | None = None,
    ) -> tuple[SegmentTube, ...]:
        """Propagate until the final segment certifies its terminal event."""

        tubes = []
        state = (
            initial
            if isinstance(initial, Zonotope)
            else Zonotope.from_interval(initial)
        )
        final_index = len(segments) - 1
        for index, segment in enumerate(segments):
            prior = tuple(tubes)
            stop = None
            if terminal is not None and index == final_index:

                def stop(
                    tube: SegmentTube, prefix: tuple[SegmentTube, ...] = prior
                ) -> bool:
                    return terminal(prefix + (tube,))

            tube = self.propagate(state, segment, dt_s, stop, deadline)
            tubes.append(tube)
            state = tube.successor
        return tuple(tubes)

    def _adaptive_step(
        self,
        initial: Zonotope,
        segment: FeedbackSegment,
        dt_s: float,
        depth: int,
        deadline: float | None,
    ) -> Iterator[tuple[Zonotope, Interval, float]]:
        _check_deadline(deadline)
        try:
            successor, continuous = self._step(initial, segment, dt_s, deadline)
            yield successor, continuous, dt_s
            return
        except TubeCertificationError:
            if depth >= self.max_subdivisions:
                raise
            midpoint = initial
            for piece in self._adaptive_step(
                initial,
                segment,
                0.5 * dt_s,
                depth + 1,
                deadline,
            ):
                midpoint = piece[0]
                yield piece
            yield from self._adaptive_step(
                midpoint,
                segment,
                0.5 * dt_s,
                depth + 1,
                deadline,
            )

    def _step(
        self,
        initial: Zonotope,
        segment: FeedbackSegment,
        dt_s: float,
        deadline: float | None,
    ) -> tuple[Zonotope, Interval]:
        initial_box = initial.interval_hull()
        command = segment.command_box(
            initial_box,
            self.aircraft,
            self._command_error(),
        )
        derivative = self._derivative(initial_box, command)
        time = Interval(0.0, dt_s)
        continuous = _inflate(
            initial_box.hull(initial_box + time * derivative),
            0.25,
        )
        for _ in range(self.picard_iterations):
            _check_deadline(deadline)
            command = segment.command_box(
                continuous,
                self.aircraft,
                self._command_error(),
            )
            derivative = self._derivative(continuous, command)
            image = initial_box + time * derivative
            if not self.domain.contains(image, self.aircraft):
                raise TubeCertificationError("reachable tube left the hard domain")
            if image.subset(continuous):
                successor = self._validated_successor(
                    initial,
                    derivative,
                    dt_s,
                )
                return successor, continuous
            continuous = _inflate(continuous.hull(image), 0.25)
        raise TubeCertificationError("Picard enclosure did not converge")

    def _command_error(self) -> np.ndarray:
        residual = np.asarray(
            self.uncertainty.command_error_abs_rad,
            dtype=float,
        ).reshape(3)
        rate = np.asarray(
            self.uncertainty.command_rate_abs_rad_s,
            dtype=float,
        ).reshape(3)
        return residual + self.uncertainty.command_delay_max_s * rate

    def _validated_successor(
        self,
        initial: Zonotope,
        derivative: Interval,
        dt_s: float,
    ) -> Zonotope:
        delta = derivative * dt_s
        shifted_center = Interval.point(initial.center) + delta.center
        radius = _up_sum(shifted_center.radius, delta.radius)
        generators = np.concatenate(
            (initial.generators, np.diag(radius)),
            axis=1,
        )
        successor = Zonotope(
            shifted_center.center,
            generators,
        )
        return _reduce(successor, self.max_generators)

    def _derivative(self, state: Interval, command: Interval) -> Interval:
        if not self.domain.contains(state, self.aircraft):
            raise TubeCertificationError("state tube left the hard domain")
        rotation = _body_to_world(state[3:6])
        velocity = state[6:9]
        omega = state[9:12]
        force, moment = self._aero_loads(state)
        mass = Interval(*self.uncertainty.mass_kg)
        gravity_b = _matvec(
            _transpose(rotation),
            Interval.point((0.0, 0.0, -G_M_S2)),
        )
        velocity_dot = force / mass + gravity_b - omega.cross(velocity)
        inertia_omega = omega.affine_map(self.aircraft.inertia_b_kg_m2)
        angular_dot = (moment - omega.cross(inertia_omega)).affine_map(
            self.aircraft.inertia_inv_b
        )
        angular_error = np.asarray(
            self.uncertainty.angular_accel_error_abs_rad_s2,
            dtype=float,
        ).reshape(3)
        angular_dot += Interval(-angular_error, angular_error)
        position_dot = _matvec(rotation, velocity)
        euler_dot = _matvec(_euler_rate_matrix(state[3], state[4]), omega)
        surface = state[12:15]
        tau = Interval(*self.uncertainty.actuator_tau_s)
        surface_dot = (command - surface) / tau
        return _join(
            position_dot,
            euler_dot,
            velocity_dot,
            angular_dot,
            surface_dot,
        )

    def _aero_loads(self, state: Interval) -> tuple[Interval, Interval]:
        table = self.aircraft.strip_table
        velocity = state[6:9]
        omega = state[9:12]
        surface = state[12:15].clip(
            self.aircraft.control_lower_rad,
            self.aircraft.control_upper_rad,
        )
        density = Interval(*self.uncertainty.density_kg_m3)
        scale = Interval(*self.uncertainty.coefficient_scale)
        air = velocity + omega.cross(self._locations) - self._strip_flow
        plane = _batch_matvec(self._projection, air)
        speed = plane.norm(axis=1)
        if (
            np.min(speed.lower) < self.domain.airspeed_bounds_m_s[0]
            or np.max(speed.upper) > self.domain.airspeed_bounds_m_s[1]
        ):
            raise TubeCertificationError("strip airspeed left the validated domain")
        if np.any(plane[:, 0].lower <= 0.0):
            raise TubeCertificationError("strip flow left the forward-flight domain")
        drag = -plane / speed[:, None]
        lift = self._spans.cross(drag)
        normal_projection = (lift * table.normal_b).sum(axis=1)
        flipped = -lift
        negative = normal_projection.upper < 0.0
        uncertain = (normal_projection.lower <= 0.0) & (normal_projection.upper >= 0.0)
        lower = np.where(negative[:, None], flipped.lower, lift.lower)
        upper = np.where(negative[:, None], flipped.upper, lift.upper)
        lift = Interval(
            np.where(
                uncertain[:, None],
                np.minimum(lower, flipped.lower),
                lower,
            ),
            np.where(
                uncertain[:, None],
                np.maximum(upper, flipped.upper),
                upper,
            ),
        )
        alpha = (-(plane * table.normal_b).sum(axis=1)).atan2(plane[:, 0])
        if np.max(np.abs((alpha.lower, alpha.upper))) > (self.domain.alpha_abs_max_rad):
            raise TubeCertificationError(
                "strip angle of attack left the validated domain"
            )
        delta = surface.affine_map(table.control_mix)
        cl, cd, cm = _flat_plate_interval(
            alpha,
            delta,
            table.aspect_ratio,
            table.chord_m,
            table.flap_chord_fraction,
            table.alpha0_rad,
            table.cd0,
        )
        cl *= scale
        cd *= scale
        cm *= scale
        pressure = 0.5 * density * speed.square()
        force_strips = (
            pressure[:, None]
            * table.area_m2[:, None]
            * (cl[:, None] * lift + cd[:, None] * drag)
        )
        moment_strips = self._locations.cross(force_strips)
        moment_strips += (
            pressure[:, None]
            * table.area_m2[:, None]
            * table.chord_m[:, None]
            * cm[:, None]
            * table.moment_axis_b
        )
        force = force_strips.sum(axis=0)
        moment = moment_strips.sum(axis=0)
        cg_air = velocity - self._center_flow
        cg_speed = cg_air.norm()
        force += (
            -0.5 * density * self.aircraft.config.drag_area_fuse_m2 * cg_speed * cg_air
        )
        force_error = np.asarray(
            self.uncertainty.force_error_abs_n,
            dtype=float,
        ).reshape(3)
        moment_error = np.asarray(
            self.uncertainty.moment_error_abs_n_m,
            dtype=float,
        ).reshape(3)
        return (
            force + Interval(-force_error, force_error),
            moment + Interval(-moment_error, moment_error),
        )

    def _point_velocities(
        self,
        state: Interval,
        points_b_m: npt.ArrayLike,
    ) -> Interval:
        rotation = _body_to_world(state[3:6])
        velocity = state[6:9]
        omega = state[9:12]
        points = np.asarray(points_b_m, dtype=float).reshape(-1, 3)
        point_velocity = velocity + omega.cross(points)
        return _interval_batch_matvec(rotation, point_velocity)


def _flat_plate_interval(
    alpha: Interval,
    delta: Interval,
    aspect_ratio: npt.ArrayLike,
    chord_m: npt.ArrayLike,
    flap_chord_fraction: npt.ArrayLike,
    alpha0_rad: npt.ArrayLike,
    cd0: npt.ArrayLike,
) -> tuple[Interval, Interval, Interval]:
    aspect = np.asarray(aspect_ratio, dtype=float).reshape(-1)
    chord = np.asarray(chord_m, dtype=float).reshape(-1)
    flap = np.asarray(flap_chord_fraction, dtype=float).reshape(-1)
    alpha_zero = np.asarray(alpha0_rad, dtype=float).reshape(-1)
    profile_drag = np.asarray(cd0, dtype=float).reshape(-1)
    if np.max(alpha.upper - alpha.lower) <= radians(10.0):
        return _flat_plate_core(
            alpha,
            delta,
            aspect,
            chord,
            flap,
            alpha_zero,
            profile_drag,
        )
    lift = []
    drag = []
    moment = []
    for index in range(aspect.size):
        width = float(alpha.upper[index] - alpha.lower[index])
        count = max(1, int(np.ceil(width / radians(10.0))))
        edges = np.linspace(alpha.lower[index], alpha.upper[index], count + 1)
        bounds = [
            _flat_plate_core(
                Interval(edges[part], edges[part + 1]),
                delta[index],
                aspect[index],
                chord[index],
                flap[index],
                alpha_zero[index],
                profile_drag[index],
            )
            for part in range(count)
        ]
        lift.append(_hull([item[0] for item in bounds]))
        drag.append(_hull([item[1] for item in bounds]))
        moment.append(_hull([item[2] for item in bounds]))
    return _stack(lift), _stack(drag), _stack(moment)


def _flat_plate_core(
    alpha: Interval,
    delta: Interval,
    aspect_ratio: npt.ArrayLike,
    chord_m: npt.ArrayLike,
    flap_chord_fraction: npt.ArrayLike,
    alpha0_rad: npt.ArrayLike,
    cd0: npt.ArrayLike,
) -> tuple[Interval, Interval, Interval]:
    aspect = np.asarray(aspect_ratio, dtype=float)
    chord = np.asarray(chord_m, dtype=float)
    flap_fraction = np.asarray(flap_chord_fraction, dtype=float)
    alpha_zero = np.asarray(alpha0_rad, dtype=float)
    profile_drag = np.asarray(cd0, dtype=float)
    a_le, a_te, alpha_le, alpha_te, alpha_high = flat_plate_coefficients(aspect)
    cl_alpha = 2.0 * pi * (aspect / (aspect + 2.0 * (aspect + 4.0) / (aspect + 2.0)))
    theta_f = np.arccos(2.0 * flap_fraction - 1.0)
    tau_f = 1.0 - (theta_f - np.sin(theta_f)) / pi
    delta_cl = cl_alpha * tau_f * delta
    alpha_lift = alpha - alpha_zero
    abs_alpha = (alpha_lift.square() + SMOOTH_ABS_EPS**2).sqrt()
    f_te = 0.5 * (1.0 - (a_te * (abs_alpha - alpha_te)).tanh())
    f_le = 0.5 * (1.0 - (a_le * (abs_alpha - alpha_le)).tanh())
    sqrt_f_te = f_te.clip(0.0, 1.0).sqrt()
    sin_alpha = alpha_lift.sin()
    cos_alpha = alpha_lift.cos()
    smooth_sin = (sin_alpha.square() + SMOOTH_ABS_EPS**2).sqrt()
    blend = 0.25 * (1.0 + sqrt_f_te).square()
    cl_attached = (
        blend
        * (
            cl_alpha * sin_alpha * cos_alpha.square()
            + f_le.square() * pi * smooth_sin * sin_alpha * cos_alpha
        )
        + delta_cl
    )
    tangent = alpha_lift.clip(
        -0.5 * pi + 1.0e-6,
        0.5 * pi - 1.0e-6,
    ).tan()
    cd_attached = profile_drag + cl_attached * tangent
    cm_attached = -blend * (
        0.0625
        * (-1.0 + 6.0 * sqrt_f_te - 5.0 * f_te)
        * cl_alpha
        * sin_alpha
        * cos_alpha
        + 0.17 * f_le.square() * pi * smooth_sin * sin_alpha
    )
    flap_chord = flap_fraction * chord
    c_prime = (
        (chord - flap_chord) ** 2
        + flap_chord**2
        + 2.0 * flap_chord * (chord - flap_chord) * delta.cos()
    ).sqrt()
    alpha_f = (flap_chord * delta.sin() / c_prime).clip(-1.0, 1.0).asin()
    alpha_post = alpha - alpha_zero + alpha_f
    cd_90 = 1.98 - 4.26e-2 * delta.square() + 2.1e-1 * delta
    sin_post = alpha_post.sin()
    cos_post = alpha_post.cos()
    smooth_post = (sin_post.square() + SMOOTH_ABS_EPS**2).sqrt()
    normal_gain = 1.0 / (0.56 + 0.44 * smooth_post) - 0.41 * (
        1.0 - np.exp(-17.0 / aspect)
    )
    normal = cd_90 * sin_post * normal_gain
    axial = 0.015 * cos_post
    cl_post = normal * cos_post - axial * sin_post
    cd_post = normal * sin_post + axial * cos_post
    cm_post = -normal * (0.25 - 0.175 * (1.0 - 2.0 * alpha_post / pi))
    smooth_alpha = (alpha.square() + SMOOTH_ABS_EPS**2).sqrt()
    sigma = 0.5 * (1.0 + (20.0 * (alpha_high - smooth_alpha)).tanh())
    return (
        sigma * cl_attached + (1.0 - sigma) * cl_post,
        sigma * cd_attached + (1.0 - sigma) * cd_post,
        sigma * cm_attached + (1.0 - sigma) * cm_post,
    )


def _body_to_world(attitude: Interval) -> Interval:
    phi, theta, psi = attitude[0], attitude[1], attitude[2]
    c_phi, s_phi = phi.cos(), phi.sin()
    c_theta, s_theta = theta.cos(), theta.sin()
    c_psi, s_psi = psi.cos(), psi.sin()
    return _matrix(
        (
            (
                c_theta * c_psi,
                s_phi * s_theta * c_psi - c_phi * s_psi,
                c_phi * s_theta * c_psi + s_phi * s_psi,
            ),
            (
                -c_theta * s_psi,
                -s_phi * s_theta * s_psi - c_phi * c_psi,
                -c_phi * s_theta * s_psi + s_phi * c_psi,
            ),
            (s_theta, -s_phi * c_theta, -c_phi * c_theta),
        )
    )


def _euler_rate_matrix(phi: Interval, theta: Interval) -> Interval:
    c_phi, s_phi = phi.cos(), phi.sin()
    c_theta = theta.cos()
    t_theta = theta.tan()
    zero = Interval.point(0.0)
    one = Interval.point(1.0)
    return _matrix(
        (
            (one, s_phi * t_theta, c_phi * t_theta),
            (zero, c_phi, -s_phi),
            (zero, s_phi / c_theta, c_phi / c_theta),
        )
    )


def _matrix(rows: tuple[tuple[Interval, ...], ...]) -> Interval:
    return Interval(
        np.array([[float(value.lower) for value in row] for row in rows]),
        np.array([[float(value.upper) for value in row] for row in rows]),
    )


def _transpose(matrix: Interval) -> Interval:
    return Interval(matrix.lower.T, matrix.upper.T)


def _matvec(matrix: Interval, vector: Interval) -> Interval:
    return (matrix * vector).sum(axis=1)


def _batch_matvec(matrix: np.ndarray, vector: Interval) -> Interval:
    return (vector[:, None, :] * matrix).sum(axis=2)


def _interval_batch_matvec(matrix: Interval, vector: Interval) -> Interval:
    return (matrix[None, :, :] * vector[:, None, :]).sum(axis=2)


def _stack(values: list[Interval]) -> Interval:
    return Interval(
        np.stack([value.lower for value in values]),
        np.stack([value.upper for value in values]),
    )


def _hull(values: list[Interval]) -> Interval:
    result = values[0]
    for value in values[1:]:
        result = result.hull(value)
    return result


def _reduce(value: Zonotope, maximum: int) -> Zonotope:
    count = value.generators.shape[1]
    if count <= maximum:
        return value
    keep_count = max(maximum - value.center.size, 0)
    norms = np.linalg.norm(value.generators, axis=0)
    indices = np.argsort(norms)
    discarded = value.generators[:, indices[: count - keep_count]]
    kept = value.generators[:, indices[count - keep_count :]]
    radius = Zonotope(np.zeros(value.center.size), discarded).radius
    return Zonotope(
        value.center,
        np.concatenate((kept, np.diag(radius)), axis=1),
    )


def _join(*values: Interval) -> Interval:
    return Interval(
        np.concatenate([value.lower.reshape(-1) for value in values]),
        np.concatenate([value.upper.reshape(-1) for value in values]),
    )


def _inflate(value: Interval, fraction: float) -> Interval:
    radius = value.radius
    coupling = 1.0e-3 * float(np.max(radius))
    margin = np.maximum(fraction * (radius + coupling), 1.0e-12)
    return Interval(value.lower - margin, value.upper + margin)


def _up_sum(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    total = np.nextafter(first + second, np.inf)
    return np.where(first == 0.0, second, np.where(second == 0.0, first, total))
