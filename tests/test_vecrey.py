"""Review tests for the `svggeom` package.
Layout
------
* ``TestVectorRegressions`` / ``TestRayRegressions``
Passing tests for bugs that were already fixed in the source code.
* ``TestVectorBugs`` / ``TestRayBugs``
One test per confirmed remaining bug. Each asserts the DESIRED behaviour
and is marked ``@unittest.expectedFailure``.
"""

import copy
import math
import pickle
import sys
import unittest
from math import inf, nan, pi, tau
from pathlib import Path

# # Allow running from the repo root or from a tests/ subfolder.
# _here = Path(__file__).resolve().parent
# for _p in (_here, _here.parent):
#     if str(_p) not in sys.path:
#         sys.path.insert(0, str(_p))

from svggeom.ray import Ray, VecRay
from svggeom.vector import Vector, vec


class VecTestCase(unittest.TestCase):
    def assertVecAlmostEqual(self, actual, expected, places=9):
        self.assertEqual(
            len(actual), len(expected), f"length differs: {actual!r} vs {expected!r}"
        )
        for i, (a, e) in enumerate(zip(actual, expected)):
            self.assertAlmostEqual(
                a, e, places=places, msg=f"component {i}: {actual!r} vs {expected!r}"
            )


# =============================================================================
# Regressions: Tests for bugs that are ALREADY FIXED in the source code.
# =============================================================================
class TestVectorRegressions(VecTestCase):
    def test_copy_deepcopy_pickle_roundtrip(self):
        v = Vector(1, 2)
        for clone in (copy.copy(v), copy.deepcopy(v), pickle.loads(pickle.dumps(v))):
            self.assertIsInstance(clone, Vector)
            self.assertEqual(clone, (1, 2))

    def test_tuple_on_left_of_plus(self):
        self.assertEqual(tuple((1, 2) + Vector(3, 4)), (4, 6))

    def test_radians_stays_below_tau(self):
        v = Vector(1, -1e-20)
        self.assertGreaterEqual(v.radians, 0)
        self.assertLess(v.radians, tau)

    def test_is_close_false_for_nan(self):
        self.assertFalse(Vector(nan, 0).is_close((1, 0)))

    def test_abs_and_normalize_no_overflow(self):
        self.assertTrue(math.isfinite(Vector(1e200, 1e200).length()))
        self.assertVecAlmostEqual(Vector(1e200, 0).normalized(), (1, 0))

    def test_with_x_pads_empty(self):
        self.assertEqual(Vector().with_x(5), (5,))


class TestRayRegressions(VecTestCase):
    def test_zero_direction_rejected(self):
        with self.assertRaises(ValueError):
            VecRay(aim=[0, 0])

    def test_side_of_large_coordinates(self):
        self.assertEqual(VecRay([1e16, 0], [1, 0]).side_of((1e16, 5)), 1)

    def test_builders_preserve_subclass(self):
        class MyRay(Ray):
            pass

        r = MyRay()
        self.assertIsInstance(r.advanced(1), MyRay)
        self.assertIsInstance(r.clone(), MyRay)


# =============================================================================
# Real Bugs: Desired behaviour, marked as expectedFailure
# =============================================================================


class TestRayBugs(VecTestCase):
    # @unittest.expectedFailure
    def test_accessors_work_for_3d_ray(self):
        """Cause: x/y/dx/dy unpack exactly 2 items, crashing on 3D rays."""
        r = VecRay([1, 2, 3], [4, 5, 6])
        self.assertEqual((r.x, r.y, r.z), (1, 2, 3))
        self.assertEqual((r.dx, r.dy), (4, 5))

    # @unittest.expectedFailure
    def test_constructor_rejects_mismatched_dimensions(self):
        """Cause: __init__ doesn't check if len(pos) == len(dir)."""
        with self.assertRaises(ValueError):
            VecRay([0, 0], [1, 0, 0])

    @unittest.expectedFailure
    def test_intersection_with_line_ignores_hits_behind_origin(self):
        """Cause: Treats ray as infinite line, doesn't check t >= 0."""
        r = VecRay([0, 0], [1, 0])
        self.assertIsNone(r.intersection_with_line((-5, -1), (-5, 1)))

    @unittest.expectedFailure
    def test_intersection_with_ray_ignores_hits_behind_either_origin(self):
        """Cause: Same as above, doesn't check t >= 0 for both rays."""
        a = VecRay([0, 0], [1, 0])
        behind_a = VecRay([-5, -1], [0, 1])
        behind_b = VecRay([5, 1], [0, 1])
        self.assertIsNone(a.intersection_with_ray(behind_a))
        self.assertIsNone(a.intersection_with_ray(behind_b))

    # @unittest.expectedFailure
    def test_point_at_distance_works_for_3d_ray(self):
        """Cause: Uses Vector.polar which always returns 2D, causing dim mismatch."""
        r = VecRay([0, 0, 0], [1, 0, 0])
        self.assertVecAlmostEqual(r.point_at_distance(5), (5, 0, 0))


def load_tests(loader, tests, ignore):
    """Also run the docstring examples of the package modules."""
    import doctest
    import svggeom.ray
    import svggeom.vector

    for module in (svggeom.vector, svggeom.ray):
        tests.addTests(doctest.DocTestSuite(module))
    return tests


if __name__ == "__main__":
    unittest.main(verbosity=2)
