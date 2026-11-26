# tests/geom/test_vector_review.py
"""Review tests for geom/vector.py.

`TestVectorPassingBehavior` guards existing behaviour so fixes can't regress it.
`TestVectorConfirmedBugs` asserts desired behaviour for the confirmed bugs;
they are currently @expectedFailure and will flag "unexpected success" once fixed.
"""

import unittest
from math import pi

from svggeom.vector import Vector, vec
from svggeom.matrix import Matrix
from svggeom.ray import Ray


class TestVectorPassingBehavior(unittest.TestCase):

    def test_len_and_abs_quad(self):
        v = Vector(3, 4)
        self.assertEqual(len(v), 2)
        self.assertEqual(v.length_squared(), 25)
        self.assertAlmostEqual(v.length(), 5.0)

    def test_dot(self):
        self.assertEqual(Vector(1, 2).dot([3, 4]), 11)
        with self.assertRaises(ValueError):
            Vector(1, 2).dot([3, 4, 5])

    def test_add_sub(self):
        self.assertEqual(Vector(1, 2) + Vector(3, 4), Vector(4, 6))
        self.assertEqual(Vector(1, 2) - Vector(3, 4), Vector(-2, -2))

    def test_add_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            Vector(1, 2) + Vector(3, 4, 5)

    def test_mul_vector_on_left(self):
        # The scalar-on-right direction works correctly today.
        self.assertEqual(Vector(1, 2) * 3, Vector(3, 6))

    def test_truediv(self):
        self.assertEqual(Vector(2, 4) / 2, Vector(1, 2))

    def test_neg(self):
        self.assertEqual(-Vector(1, 2), Vector(-1, -2))

    def test_normalize(self):
        v = Vector(3, 4).normalized()
        self.assertAlmostEqual(v[0], 0.6)
        self.assertAlmostEqual(v[1], 0.8)
        self.assertAlmostEqual(v.length(), 1.0)

    def test_normalize_zero_raises(self):
        with self.assertRaises(ValueError):
            Vector(0, 0).normalized()

    def test_lerp(self):
        self.assertEqual(Vector(0, 0).lerp(Vector(10, 20), 0.5), Vector(5, 10))
        self.assertEqual(Vector(0, 0).lerp(Vector(10, 20), 0.0), Vector(0, 0))
        self.assertEqual(Vector(0, 0).lerp(Vector(10, 20), 1.0), Vector(10, 20))

    def test_reflect_at(self):
        self.assertEqual(Vector(1, 1).reflect_through((2, 2)), Vector(3, 3))
        self.assertEqual(Vector(3, 3).reflect_through((2, 2)), Vector(1, 1))
        # A point at the pivot reflects onto itself.
        self.assertEqual(Vector(2, 2).reflect_through((2, 2)), Vector(2, 2))

    def test_close_to(self):
        self.assertTrue(Vector(1, 2).is_close((1, 2)))
        self.assertFalse(Vector(1, 2).is_close((1, 2.5)))
        self.assertFalse(Vector(1, 2).is_close((1, 2, 3)))

    def test_eq_with_tuple(self):
        # Vector inherits __eq__ from tuple -- comparison with tuples works.
        self.assertEqual(Vector(1, 2), (1, 2))

    def test_vec_helper(self):
        self.assertIsInstance(vec((1, 2)), Vector)
        v = Vector(1, 2)
        self.assertIs(vec(v), v)

    def test_transform_matches_matrix_apply(self):
        # Vector.transform (hexad form) and Matrix.apply_to_point (complex form)
        # must agree for any affine matrix.
        m = Matrix().add_translate(5, 3)
        v = Vector(1, 2)
        a = v.apply_svg_matrix(m.to_hexad())
        b = m.apply_to_point(v)
        self.assertAlmostEqual(a[0], b[0])
        self.assertAlmostEqual(a[1], b[1])

    def test_polar_3d_mode_is_3d(self):
        # 3-D mode (theta provided) is correct today.
        v = Vector.from_spherical(5, 0, pi / 2)
        self.assertEqual(len(v), 3)
        self.assertAlmostEqual(v[0], 5)
        self.assertAlmostEqual(v[1], 0)
        self.assertAlmostEqual(v.z, 0)

    def test_nearest_point_of_line_normal_case(self):
        r = Vector(3, 5).closest_point_on_segment((0, 0), (10, 0))
        self.assertAlmostEqual(r[0], 3)
        self.assertAlmostEqual(r[1], 0)
        # Clamped to the segment
        r2 = Vector(20, 5).closest_point_on_segment((0, 0), (10, 0))
        self.assertAlmostEqual(r2[0], 10)
        self.assertAlmostEqual(r2[1], 0)


class TestVectorConfirmedBugs(unittest.TestCase):

    # @unittest.expectedFailure
    def test_polar_2d_returns_2d(self):
        """Bug 1: polar() with theta=None returns a 3-D vector."""
        v = Vector.from_polar(5, pi / 2)
        self.assertEqual(len(v), 2)
        self.assertAlmostEqual(v[0], 0, places=9)
        self.assertAlmostEqual(v[1], 5)

    # @unittest.expectedFailure
    def test_polar_2d_zero_radius_returns_2d(self):
        """Bug 1b: polar(0) falls through to the 3-D zero vector."""
        v = Vector.from_polar(0)
        self.assertEqual(len(v), 2)

    # @unittest.expectedFailure
    def test_rmul_scalar_multiplies(self):
        """Bug 2: 2 * Vector must scalar-multiply, not repeat the tuple."""
        r = 2 * Vector(1, 2)
        self.assertEqual(r, Vector(2, 4))

    # @unittest.expectedFailure
    def test_nearest_point_of_line_degenerate(self):
        """Bug 3: degenerate segment a==b should return a, not raise."""
        r = Vector(5, 5).closest_point_on_segment((1, 1), (1, 1))
        self.assertEqual(r, Vector(1, 1))

    # @unittest.expectedFailure
    def test_with_y_pads_short_vector(self):
        """Bug 4: with_y ignores the new value on a 1-D vector (with_z pads)."""
        v = Vector(1)
        r = v.with_y(9)
        self.assertEqual(len(r), 2)
        self.assertEqual(r, Vector(1, 9))

    # @unittest.expectedFailure
    def test_ray_point_along_works(self):
        """Cross-file evidence that Bug 1 is user-visible via geom.ray."""
        r = Ray((0, 0), (1, 0))
        p = r.point_at_distance(5)
        self.assertAlmostEqual(p[0], 5)
        self.assertAlmostEqual(p[1], 0)
