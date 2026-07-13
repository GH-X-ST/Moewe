"""Focused soundness tests for affine forms with interval remainders."""

from __future__ import annotations

from itertools import product
from unittest import TestCase

import numpy as np

from control.interval import AffineForm


class AffineFormTests(TestCase):
    """Check the arithmetic used by dependency-preserving loads."""

    def test_core_arithmetic_contains_correlated_samples(self) -> None:
        """Contain addition, products, quotients, cross products, and norms."""

        first = AffineForm(
            (0.3, -0.2, 0.5),
            ((0.2, -0.1), (0.05, 0.12), (-0.08, 0.16)),
            (0.03, 0.02, 0.04),
        )
        second = AffineForm(
            (1.5, 1.2, 1.8),
            ((-0.1, 0.08), (0.06, -0.05), (0.04, 0.09)),
            (0.02, 0.03, 0.02),
            first.basis,
        )
        enclosures = (
            ((first + second).interval_hull(), lambda x, y: x + y),
            ((first * second).interval_hull(), lambda x, y: x * y),
            ((first / second).interval_hull(), lambda x, y: x / y),
            ((first.cross(second)).interval_hull(), np.cross),
            ((first.norm()).interval_hull(), lambda x, _: np.linalg.norm(x)),
        )
        random = np.random.default_rng(5021)
        coefficients = list(product((-1.0, 1.0), repeat=2))
        coefficients.extend(random.uniform(-1.0, 1.0, 2) for _ in range(256))

        for coefficient in coefficients:
            value = np.asarray(coefficient)
            first_error = random.uniform(-first.remainder, first.remainder)
            second_error = random.uniform(-second.remainder, second.remainder)
            first_sample = first.center + first.generators @ value + first_error
            second_sample = second.center + second.generators @ value + second_error
            for enclosure, operation in enclosures:
                self.assertTrue(
                    enclosure.contains(operation(first_sample, second_sample))
                )

    def test_norm_contains_zero_crossing_components(self) -> None:
        """Fall back safely when a squared affine remainder crosses zero."""

        form = AffineForm(
            np.zeros(3),
            ((0.4, -0.2), (-0.1, 0.3), (0.25, 0.15)),
            (0.05, 0.04, 0.03),
        )
        enclosure = form.norm().interval_hull()
        random = np.random.default_rng(211)
        for _ in range(256):
            coefficient = random.uniform(-1.0, 1.0, 2)
            error = random.uniform(-form.remainder, form.remainder)
            sample = form.center + form.generators @ coefficient + error
            self.assertTrue(enclosure.contains(np.linalg.norm(sample)))

    def test_independent_generator_bases_are_rejected(self) -> None:
        """Reject accidental correlation between independently built forms."""

        first = AffineForm((0.0,), ((1.0,),), (0.0,))
        second = AffineForm((0.0,), ((1.0,),), (0.0,))
        with self.assertRaisesRegex(ValueError, "independent generator bases"):
            _ = first + second
