"""Sampled mission-conditioned recoverability controller."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from control.abstraction import RecoverabilityObject
from control.missions import Mission
from models.state import as_state


EventBound = Callable[[np.ndarray, int, int], bool]


class NoSafeControlError(RuntimeError):
    """Raised when a mission has no safety-admissible control."""


@dataclass(frozen=True)
class PreparedMission:
    """Mission bounds evaluated on an abstraction."""

    mission: Mission
    safe_input_indices: tuple[tuple[int, ...], ...]
    progress_bound: np.ndarray
    running_cost_bound: np.ndarray
    event_bound: EventBound


@dataclass
class MissionRecoverabilityController:
    """Mission-conditioned recoverability controller."""

    abstraction: RecoverabilityObject
    lambda_r: float = 1.0
    lambda_u: float = 1.0

    def control(
        self,
        state: npt.ArrayLike,
        previous_control: npt.ArrayLike,
        mission: PreparedMission,
    ) -> np.ndarray:
        """Return the sampled feedback control."""

        x = as_state(state)
        u_prev = np.asarray(previous_control, dtype=float).reshape(3)
        cell = self.abstraction.cell(x)
        safe_indices = mission.safe_input_indices[cell]
        if not safe_indices:
            raise NoSafeControlError
        mission_indices = self._mission_indices(
            x,
            cell,
            safe_indices,
            mission,
        )
        if mission_indices:
            index = min(
                mission_indices,
                key=lambda idx: self._mission_cost(
                    cell,
                    idx,
                    u_prev,
                    mission,
                ),
            )
        else:
            index = self._fallback_index(cell, safe_indices, u_prev)
        return self.abstraction.controls[index]

    def _mission_indices(
        self,
        state: np.ndarray,
        cell: int,
        safe_indices: tuple[int, ...],
        mission: PreparedMission,
    ) -> tuple[int, ...]:
        progress_limit = (
            mission.mission.progress(state) - mission.mission.delta
        )
        return tuple(
            idx
            for idx in safe_indices
            if mission.progress_bound[cell, idx] <= progress_limit
            or mission.event_bound(state, cell, idx)
        )

    def _mission_cost(
        self,
        cell: int,
        input_index: int,
        previous_control: np.ndarray,
        mission: PreparedMission,
    ) -> float:
        control = self.abstraction.controls[input_index]
        control_change = control - previous_control
        return (
            mission.running_cost_bound[cell, input_index]
            - self.lambda_r * self.abstraction.score[cell, input_index]
            + self.lambda_u * float(control_change @ control_change)
        )

    def _fallback_index(
        self,
        cell: int,
        indices: tuple[int, ...],
        previous_control: np.ndarray,
    ) -> int:
        return max(
            indices,
            key=lambda idx: self._recoverability_value(
                cell,
                idx,
                previous_control,
            ),
        )

    def _recoverability_value(
        self,
        cell: int,
        input_index: int,
        previous_control: np.ndarray,
    ) -> float:
        control_change = (
            self.abstraction.controls[input_index] - previous_control
        )
        return (
            self.abstraction.score[cell, input_index]
            - self.lambda_u * float(control_change @ control_change)
        )


def sample_mission(
    abstraction: RecoverabilityObject,
    mission: Mission,
) -> PreparedMission:
    """Estimate mission bounds from representative cell samples."""

    safe_cells = frozenset(
        cell
        for cell, samples in enumerate(abstraction.cell_samples)
        if all(mission.safe(sample) for sample in samples)
    )
    safe_input_indices = tuple(
        tuple(
            idx
            for idx in abstraction.input_indices[cell]
            if all(
                next_cell in safe_cells
                for next_cell in abstraction.tube_successors[cell][idx]
            )
        )
        for cell in range(len(abstraction.sampled_successors))
    )
    progress_bound = np.empty_like(abstraction.score)
    running_cost_bound = np.empty_like(abstraction.score)
    for cell in range(progress_bound.shape[0]):
        for idx, control in enumerate(abstraction.controls):
            samples = tuple(
                sample
                for next_cell in abstraction.sampled_successors[cell][idx]
                for sample in abstraction.cell_samples[next_cell]
            )
            progress_bound[cell, idx] = max(
                mission.progress(sample) for sample in samples
            )
            running_cost_bound[cell, idx] = max(
                mission.running_cost(sample, control)
                for sample in samples
            )

    def event_bound(state: np.ndarray, cell: int, idx: int) -> bool:
        return all(
            mission.event(state, sample)
            for next_cell in abstraction.sampled_successors[cell][idx]
            for sample in abstraction.cell_samples[next_cell]
        )

    return PreparedMission(
        mission,
        safe_input_indices,
        progress_bound,
        running_cost_bound,
        event_bound,
    )
