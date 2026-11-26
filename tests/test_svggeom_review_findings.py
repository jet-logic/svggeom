"""Regression tests for independently reproduced svggeom issues.

Run from the repository root with:
    python -m unittest -v test_svggeom_review_findings
"""

import importlib
import unittest

from svggeom.bezier import Bezier
from svggeom.vector import Vector
from svggeom.ray import Ray


class ConfirmedBugTests(unittest.TestCase):
    # @unittest.expectedFailure
    def test_pathlerp_module_imports(self):
        """The pathlerp helper should be importable as part of svggeom."""
        module = importlib.import_module("svggeom.help.pathlerp")
        self.assertTrue(callable(module.pathlc_to_beziers_simple))

    # @unittest.expectedFailure
    def test_quad_to_without_move_raises_value_error(self):
        """Like curve_to, quad_to should reject a missing current point clearly."""
        with self.assertRaisesRegex(ValueError, "preceding move_to"):
            Bezier().quad_to((1, 1), (2, 2))

    # @unittest.expectedFailure
    def test_arc_to_without_move_raises_value_error(self):
        """arc_to should reject a missing current point clearly."""
        with self.assertRaisesRegex(ValueError, "preceding move_to"):
            Bezier().arc_to(1, 1, 0, False, True, (2, 2))


class PassingRegressionTests(unittest.TestCase):
    def test_curve_to_without_move_already_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "preceding move_to"):
            Bezier().curve_to((1, 1), (2, 2), (3, 3))

    def test_bezier_transform_preserves_a_straight_segment(self):
        curve = Bezier().move_to((0, 0)).line_to((2, 3))
        curve.transform(__import__("svggeom.matrix", fromlist=["Matrix"]).Matrix())
        self.assertEqual(curve[0][0], (0, 0))
        self.assertEqual(curve[1][0], (2, 3))

    def test_vector_scalar_multiplication_on_both_sides(self):
        v = Vector(2, -3)
        self.assertEqual(v * 4, Vector(8, -12))
        self.assertEqual(4 * v, Vector(8, -12))

    def test_ray_intersection_parameters_are_forward(self):
        first = Ray((0, 0), (1, 0))
        second = Ray((2, -1), (0, 1))
        hit = first.intersection_with_ray(second)
        self.assertEqual(tuple(hit), (2.0, 0.0))
        self.assertTrue(hit.is_forward)
        self.assertEqual((hit.t_self, hit.t_other), (2.0, 1.0))


if __name__ == "__main__":
    unittest.main()
