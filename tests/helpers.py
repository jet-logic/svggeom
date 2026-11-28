"""Shared assertions for the svggeom test-suite."""

import unittest


class GeomTestCase(unittest.TestCase):
    """``TestCase`` with helpers for comparing vectors/points component-wise."""

    def assertVecAlmostEqual(self, actual, expected, places=9):
        self.assertEqual(
            len(actual), len(expected), f"length differs: {actual!r} vs {expected!r}"
        )
        for i, (a, e) in enumerate(zip(actual, expected)):
            self.assertAlmostEqual(
                a, e, places=places, msg=f"component {i}: {actual!r} vs {expected!r}"
            )
