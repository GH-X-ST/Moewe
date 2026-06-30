from __future__ import annotations

import pytest

from moewe.sim.randomisation import UniformRange, default_randomisation_spec, sample_parameters


def test_same_seed_produces_same_sample() -> None:
    spec = default_randomisation_spec()

    assert sample_parameters(spec, seed=12) == sample_parameters(spec, seed=12)


def test_sampled_values_remain_inside_declared_bounds() -> None:
    spec = {"mass_scale": UniformRange(0.9, 1.1), "nested": {"delay_s": (0.0, 0.04)}}

    sample = sample_parameters(spec, seed=5)

    assert 0.9 <= sample["mass_scale"] <= 1.1
    assert 0.0 <= sample["nested"]["delay_s"] <= 0.04


def test_invalid_bounds_raise_clear_error() -> None:
    with pytest.raises(ValueError, match="lower bound"):
        sample_parameters({"bad": UniformRange(2.0, 1.0)}, seed=1)
