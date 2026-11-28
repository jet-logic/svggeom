import copy
import math
import pickle
import unittest
from math import nan, pi, tau

from svggeom.matrix import Matrix
from svggeom.vector import Vector, vec

from .helpers import GeomTestCase


class TestVector(GeomTestCase):

    # -- sizes and basic arithmetic ------------------------------------------

    def test_len_is_dimension_and_length_is_euclidean(self):
        v = Vector(3, 4)
        self.assertEqual(len(v), 2)
        self.assertEqual(v.length_squared(), 25)
        self.assertAlmostEqual(v.length(), 5.0)

    def test_length_and_normalize_do_not_overflow(self):
        self.assertTrue(math.isfinite(Vector(1e200, 1e200).length()))
        self.assertVecAlmostEqual(Vector(1e200, 0).normalized(), (1, 0))

    def test_dot(self):
        self.assertEqual(Vector(1, 2).dot([3, 4]), 11)
        with self.assertRaises(ValueError):
            Vector(1, 2).dot([3, 4, 5])

    def test_add_and_sub(self):
        self.assertEqual(Vector(1, 2) + Vector(3, 4), Vector(4, 6))
        self.assertEqual(Vector(1, 2) - Vector(3, 4), Vector(-2, -2))

    def test_add_rejects_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            Vector(1, 2) + Vector(3, 4, 5)

    def test_tuple_on_left_of_plus(self):
        self.assertEqual(tuple((1, 2) + Vector(3, 4)), (4, 6))

    def test_scalar_multiplication_works_on_both_sides(self):
        self.assertEqual(Vector(1, 2) * 3, Vector(3, 6))
        # ``2 * v`` must scale, not repeat the tuple.
        self.assertEqual(2 * Vector(1, 2), Vector(2, 4))
        self.assertEqual(4 * Vector(2, -3), Vector(8, -12))

    def test_truediv(self):
        self.assertEqual(Vector(2, 4) / 2, Vector(1, 2))

    def test_neg(self):
        self.assertEqual(-Vector(1, 2), Vector(-1, -2))

    def test_normalized(self):
        v = Vector(3, 4).normalized()
        self.assertAlmostEqual(v[0], 0.6)
        self.assertAlmostEqual(v[1], 0.8)
        self.assertAlmostEqual(v.length(), 1.0)

    def test_normalized_zero_vector_raises(self):
        with self.assertRaises(ValueError):
            Vector(0, 0).normalized()

    def test_lerp(self):
        a, b = Vector(0, 0), Vector(10, 20)
        self.assertEqual(a.lerp(b, 0.0), Vector(0, 0))
        self.assertEqual(a.lerp(b, 0.5), Vector(5, 10))
        self.assertEqual(a.lerp(b, 1.0), Vector(10, 20))

    def test_reflect_through(self):
        self.assertEqual(Vector(1, 1).reflect_through((2, 2)), Vector(3, 3))
        self.assertEqual(Vector(3, 3).reflect_through((2, 2)), Vector(1, 1))
        # A point at the pivot reflects onto itself.
        self.assertEqual(Vector(2, 2).reflect_through((2, 2)), Vector(2, 2))

    # -- comparison and identity ---------------------------------------------

    def test_equals_tuple(self):
        self.assertEqual(Vector(1, 2), (1, 2))

    def test_is_close(self):
        self.assertTrue(Vector(1, 2).is_close((1, 2)))
        self.assertFalse(Vector(1, 2).is_close((1, 2.5)))
        self.assertFalse(Vector(1, 2).is_close((1, 2, 3)))
        self.assertFalse(Vector(nan, 0).is_close((1, 0)))

    def test_vec_helper(self):
        self.assertIsInstance(vec((1, 2)), Vector)
        v = Vector(1, 2)
        self.assertIs(vec(v), v)

    def test_copy_deepcopy_pickle_roundtrip(self):
        v = Vector(1, 2)
        for clone in (copy.copy(v), copy.deepcopy(v), pickle.loads(pickle.dumps(v))):
            self.assertIsInstance(clone, Vector)
            self.assertEqual(clone, (1, 2))

    # -- angles, polar and spherical constructors ----------------------------

    def test_radians_stays_below_tau(self):
        v = Vector(1, -1e-20)
        self.assertGreaterEqual(v.radians, 0)
        self.assertLess(v.radians, tau)

    def test_from_polar_is_2d(self):
        v = Vector.from_polar(5, pi / 2)
        self.assertEqual(len(v), 2)
        self.assertAlmostEqual(v[0], 0, places=9)
        self.assertAlmostEqual(v[1], 5)

    def test_from_polar_zero_radius_is_2d(self):
        self.assertEqual(len(Vector.from_polar(0)), 2)

    def test_from_spherical_is_3d(self):
        v = Vector.from_spherical(5, 0, pi / 2)
        self.assertEqual(len(v), 3)
        self.assertAlmostEqual(v[0], 5)
        self.assertAlmostEqual(v[1], 0)
        self.assertAlmostEqual(v.z, 0)

    # -- with_x / with_y padding ---------------------------------------------

    def test_with_x_pads_empty_vector(self):
        self.assertEqual(Vector().with_x(5), (5,))

    def test_with_y_pads_short_vector(self):
        r = Vector(1).with_y(9)
        self.assertEqual(len(r), 2)
        self.assertEqual(r, Vector(1, 9))

    # -- geometry helpers ----------------------------------------------------

    def test_closest_point_on_segment(self):
        r = Vector(3, 5).closest_point_on_segment((0, 0), (10, 0))
        self.assertAlmostEqual(r[0], 3)
        self.assertAlmostEqual(r[1], 0)
        # Clamped to the segment end.
        r = Vector(20, 5).closest_point_on_segment((0, 0), (10, 0))
        self.assertAlmostEqual(r[0], 10)
        self.assertAlmostEqual(r[1], 0)

    def test_closest_point_on_degenerate_segment_is_that_point(self):
        r = Vector(5, 5).closest_point_on_segment((1, 1), (1, 1))
        self.assertEqual(r, Vector(1, 1))

    def test_apply_svg_matrix_agrees_with_matrix_apply_to_point(self):
        m = Matrix().add_translate(5, 3)
        v = Vector(1, 2)
        a = v.apply_svg_matrix(m.to_hexad())
        b = m.apply_to_point(v)
        self.assertAlmostEqual(a[0], b[0])
        self.assertAlmostEqual(a[1], b[1])
