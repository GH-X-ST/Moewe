"""Design envelopes and regular finite grids."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


Bounds = tuple[tuple[float, float], ...]
DisturbanceSignal = Callable[[float], object]


def no_disturbance(_time: float) -> object:
    """Return the nominal no-disturbance value."""

    return None


@dataclass
class BoxGrid:
    """Axis-aligned finite partition of a box."""

    bounds: Bounds
    shape: tuple[int, ...]
    lower: np.ndarray = field(init=False, repr=False)
    upper: np.ndarray = field(init=False, repr=False)
    step: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        limits = np.asarray(self.bounds, dtype=float)
        self.lower = limits[:, 0]
        self.upper = limits[:, 1]
        self.step = (self.upper - self.lower) / np.asarray(self.shape)

    @property
    def n_cells(self) -> int:
        """Return the number of cells."""

        return int(np.prod(self.shape))

    def cell(self, state: npt.ArrayLike) -> int:
        """Return the cell containing a state."""

        x = np.asarray(state, dtype=float).reshape(len(self.shape))
        index = np.floor((x - self.lower) / self.step).astype(int)
        index = np.where(x == self.upper, np.asarray(self.shape) - 1, index)
        return int(np.ravel_multi_index(tuple(index), self.shape))

    def center_samples(self) -> tuple[np.ndarray, ...]:
        """Return one centre sample for every cell."""

        axes = tuple(
            self.lower[idx]
            + (np.arange(count) + 0.5) * self.step[idx]
            for idx, count in enumerate(self.shape)
        )
        centers = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        flattened = centers.reshape(-1, len(self.shape))
        return tuple(center.reshape(1, -1) for center in flattened)

    def cell_bounds(self, cell: int) -> Bounds:
        """Return the bounds of a cell."""

        index = np.asarray(np.unravel_index(cell, self.shape))
        lower = self.lower + index * self.step
        upper = lower + self.step
        return tuple(zip(lower.tolist(), upper.tolist()))


def control_lattice(bounds: Bounds, shape: tuple[int, ...]) -> np.ndarray:
    """Return a Cartesian control lattice."""

    axes = tuple(
        np.linspace(lower, upper, count)
        if count > 1
        else np.array([0.5 * (lower + upper)])
        for (lower, upper), count in zip(bounds, shape)
    )
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
        -1,
        len(shape),
    )
