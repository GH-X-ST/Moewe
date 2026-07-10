"""Finite recoverability abstraction and sampled transition construction."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from control.design import DisturbanceSignal, no_disturbance
from models.state import as_state


TransitionTable = tuple[tuple[tuple[int, ...], ...], ...]
StateToCell = Callable[[np.ndarray], int]
Dynamics = Callable[[np.ndarray, np.ndarray, object], np.ndarray]
Margin = Callable[[np.ndarray], float]


@dataclass
class RecoverabilityObject:
    """Offline recoverability object used by the sampled controller."""

    controls: npt.ArrayLike
    sampled_successors: Sequence[Sequence[Sequence[int]]]
    tube_successors: Sequence[Sequence[Sequence[int]]]
    cell_margin: npt.ArrayLike
    state_to_cell: StateToCell
    cell_samples: Sequence[npt.ArrayLike] = ()
    positive_margin_cells: frozenset[int] = field(init=False)
    kernel: frozenset[int] = field(init=False)
    input_indices: tuple[tuple[int, ...], ...] = field(init=False)
    score: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.controls = np.asarray(self.controls, dtype=float).reshape(-1, 3)
        self.cell_margin = np.asarray(self.cell_margin, dtype=float)
        self.sampled_successors = _transition_table(self.sampled_successors)
        self.tube_successors = _transition_table(self.tube_successors)
        self.cell_samples = tuple(
            np.asarray(samples, dtype=float).reshape(-1, 15)
            for samples in self.cell_samples
        )
        n_controls = self.controls.shape[0]
        self.positive_margin_cells = frozenset(
            int(idx) for idx in np.flatnonzero(self.cell_margin >= 0.0)
        )
        self.kernel = fixed_point_kernel(
            self.positive_margin_cells,
            self.sampled_successors,
            self.tube_successors,
            n_controls,
        )
        self.input_indices = _recoverable_inputs(
            self.positive_margin_cells,
            self.kernel,
            self.sampled_successors,
            self.tube_successors,
            n_controls,
        )
        self.score = _score_array(self.cell_margin, self.sampled_successors)

    def cell(self, state: npt.ArrayLike) -> int:
        """Return the abstract cell containing a state."""

        return int(self.state_to_cell(as_state(state)))


def sample_abstraction(
    controls: npt.ArrayLike,
    cell_samples: Sequence[npt.ArrayLike],
    state_to_cell: StateToCell,
    dynamics: Dynamics,
    margin: Margin,
    dt: float,
    disturbance_signals: Sequence[DisturbanceSignal] = (no_disturbance,),
    tube_steps: int = 4,
) -> RecoverabilityObject:
    """Estimate transition tables and margins from finite samples."""

    controls_array = np.asarray(controls, dtype=float).reshape(-1, 3)
    samples = tuple(
        np.asarray(cell, dtype=float).reshape(-1, 15)
        for cell in cell_samples
    )
    sampled = []
    tube = []
    for cell in samples:
        sampled_row = []
        tube_row = []
        for control in controls_array:
            final_cells = set()
            tube_cells = set()
            for state in cell:
                for disturbance in disturbance_signals:
                    end_state, visited = _rollout(
                        state,
                        control,
                        dynamics,
                        disturbance,
                        dt,
                        tube_steps,
                    )
                    final_cells.add(state_to_cell(end_state))
                    tube_cells.update(state_to_cell(x) for x in visited)
            sampled_row.append(tuple(sorted(final_cells)))
            tube_row.append(tuple(sorted(tube_cells)))
        sampled.append(tuple(sampled_row))
        tube.append(tuple(tube_row))
    cell_margin = np.array(
        [min(float(margin(state)) for state in cell) for cell in samples],
        dtype=float,
    )
    return RecoverabilityObject(
        controls=controls_array,
        sampled_successors=tuple(sampled),
        tube_successors=tuple(tube),
        cell_margin=cell_margin,
        state_to_cell=state_to_cell,
        cell_samples=samples,
    )


def fixed_point_kernel(
    positive_margin_cells: frozenset[int],
    sampled_successors: TransitionTable,
    tube_successors: TransitionTable,
    n_controls: int,
) -> frozenset[int]:
    """Return the robust recoverability kernel."""

    sampled_sets = _successor_sets(sampled_successors)
    tube_sets = _successor_sets(tube_successors)
    kernel = positive_margin_cells
    while True:
        next_kernel = frozenset(
            cell
            for cell in kernel
            if any(
                sampled_sets[cell][idx] <= kernel
                and tube_sets[cell][idx] <= positive_margin_cells
                for idx in range(n_controls)
            )
        )
        if next_kernel == kernel:
            return kernel
        kernel = next_kernel


def calibration_error(
    horizon_containment: Sequence[bool],
    confidence: float = 0.99,
) -> float:
    """Return a Hoeffding bound on horizon-containment failure."""

    outcomes = np.asarray(horizon_containment, dtype=bool)
    failure_rate = 1.0 - float(np.mean(outcomes))
    radius = np.sqrt(
        np.log(1.0 / (1.0 - confidence)) / (2.0 * outcomes.size)
    )
    return float(min(1.0, failure_rate + radius))


def _transition_table(
    table: Sequence[Sequence[Sequence[int]]],
) -> TransitionTable:
    return tuple(
        tuple(tuple(int(cell) for cell in cells) for cells in row)
        for row in table
    )


def _successor_sets(
    table: TransitionTable,
) -> tuple[tuple[frozenset[int], ...], ...]:
    return tuple(tuple(frozenset(cells) for cells in row) for row in table)


def _recoverable_inputs(
    positive_margin_cells: frozenset[int],
    kernel: frozenset[int],
    sampled_successors: TransitionTable,
    tube_successors: TransitionTable,
    n_controls: int,
) -> tuple[tuple[int, ...], ...]:
    sampled_sets = _successor_sets(sampled_successors)
    tube_sets = _successor_sets(tube_successors)
    return tuple(
        tuple(
            idx
            for idx in range(n_controls)
            if sampled_sets[cell][idx] <= kernel
            and tube_sets[cell][idx] <= positive_margin_cells
        )
        for cell in range(len(sampled_successors))
    )


def _score_array(
    cell_margin: np.ndarray,
    sampled_successors: TransitionTable,
) -> np.ndarray:
    n_cells = len(sampled_successors)
    n_controls = len(sampled_successors[0])
    score = np.empty((n_cells, n_controls), dtype=float)
    for cell in range(n_cells):
        for idx in range(n_controls):
            score[cell, idx] = min(
                float(cell_margin[next_cell])
                for next_cell in sampled_successors[cell][idx]
            )
    return score


def _rollout(
    state: np.ndarray,
    control: np.ndarray,
    dynamics: Dynamics,
    disturbance: DisturbanceSignal,
    dt: float,
    steps: int,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    h = dt / steps
    x = np.asarray(state, dtype=float).reshape(15)
    visited = [x]
    time = 0.0
    for _ in range(steps):
        x = _rk4(dynamics, x, control, disturbance, time, h)
        visited.append(x)
        time += h
    return x, tuple(visited)


def _rk4(
    dynamics: Dynamics,
    state: np.ndarray,
    control: np.ndarray,
    disturbance: DisturbanceSignal,
    time: float,
    dt: float,
) -> np.ndarray:
    k1 = dynamics(state, control, disturbance(time))
    k2 = dynamics(
        state + 0.5 * dt * k1,
        control,
        disturbance(time + 0.5 * dt),
    )
    k3 = dynamics(
        state + 0.5 * dt * k2,
        control,
        disturbance(time + 0.5 * dt),
    )
    k4 = dynamics(state + dt * k3, control, disturbance(time + dt))
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
