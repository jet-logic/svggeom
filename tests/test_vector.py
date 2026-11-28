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


class TestVectorConstruction(GeomTestCase):

    def test_rejects_non_numeric_components(self):
        for args in (("a", 1), (None,), ((1, 2),)):
            with self.subTest(args=args):
                with self.assertRaises(TypeError):
                    Vector(*args)

    def test_coordinate_accessors(self):
        v = Vector(1, 2, 3)
        self.assertEqual((v.x, v.y, v.z), (1, 2, 3))

    def test_z_on_a_2d_vector_raises_attribute_error(self):
        with self.assertRaises(AttributeError):
            _ = Vector(1, 2).z

    def test_empty_vector(self):
        v = Vector()
        self.assertEqual(len(v), 0)
        self.assertTrue(v.is_close(()))

    def test_from_spherical_zero_radius_is_the_3d_origin(self):
        self.assertEqual(Vector.from_spherical(0, 1, 1), Vector(0, 0, 0))

    def test_from_polar_defaults_to_the_unit_x_vector(self):
        self.assertVecAlmostEqual(Vector.from_polar(), (1, 0))


class TestVectorAngles(GeomTestCase):

    def test_radians_degrees_and_gradians_per_quadrant(self):
        cases = {
            (1, 0): (0.0, 0.0, 0.0),
            (0, 1): (pi / 2, 90.0, 100.0),
            (-1, 0): (pi, 180.0, 200.0),
            (0, -1): (3 * pi / 2, 270.0, 300.0),
        }
        for (x, y), (rad, deg, grad) in cases.items():
            with self.subTest(x=x, y=y):
                v = Vector(x, y)
                self.assertAlmostEqual(v.radians, rad)
                self.assertAlmostEqual(v.degrees, deg)
                self.assertAlmostEqual(v.gradians, grad)

    def test_angle_ignores_extra_dimensions(self):
        self.assertAlmostEqual(Vector(0, 2, 99).degrees, 90.0)


class TestVectorReflectedOperators(unittest.TestCase):
    """Plain sequences on the left of ``+`` / ``-``."""

    def test_sequence_minus_vector(self):
        r = (5, 5) - Vector(1, 2)
        self.assertIsInstance(r, Vector)
        self.assertEqual(r, (4, 3))

    def test_sequence_plus_vector_is_a_vector(self):
        self.assertIsInstance([1, 2] + Vector(3, 4), Vector)

    def test_dimension_mismatch_still_raises_value_error(self):
        with self.assertRaises(ValueError):
            Vector(1, 2) - Vector(1, 2, 3)
        with self.assertRaises(ValueError):
            (1, 2, 3) + Vector(1, 2)
        with self.assertRaises(ValueError):
            (1, 2, 3) - Vector(1, 2)

    def test_non_sequences_are_not_supported(self):
        with self.assertRaises(TypeError):
            5 + Vector(1, 2)
        with self.assertRaises(TypeError):
            5 - Vector(1, 2)


class TestVectorAxisHelpers(unittest.TestCase):

    def test_with_axis_replaces_one_component(self):
        v = Vector(1, 2, 3)
        self.assertEqual(v.with_x(9), (9, 2, 3))
        self.assertEqual(v.with_y(9), (1, 9, 3))
        self.assertEqual(v.with_z(9), (1, 2, 9))

    def test_with_z_pads_a_2d_vector(self):
        self.assertEqual(Vector(1, 2).with_z(3), (1, 2, 3))

    def test_shift_axis_adds_to_one_component(self):
        v = Vector(1, 2, 3)
        self.assertEqual(v.shift_x(10), (11, 2, 3))
        self.assertEqual(v.shift_y(10), (1, 12, 3))
        self.assertEqual(v.shift_z(10), (1, 2, 13))

    def test_shift_axis_pads_short_vectors_with_zero(self):
        self.assertEqual(Vector().shift_x(5), (5,))
        self.assertEqual(Vector(1).shift_y(5), (1, 5))
        self.assertEqual(Vector(1, 2).shift_z(5), (1, 2, 5))

    def test_helpers_return_new_vectors(self):
        v = Vector(1, 2)
        self.assertIsNot(v.with_x(1), v)
        self.assertEqual(v, (1, 2))


class TestVectorMisc(GeomTestCase):

    def test_is_close_edge_cases(self):
        inf = math.inf
        self.assertTrue(Vector(inf, 0).is_close((inf, 0)))
        self.assertFalse(Vector(inf, 0).is_close((-inf, 0)))
        self.assertFalse(Vector(1, 2).is_close((1, 2 + 1e-9)))
        self.assertTrue(Vector(1, 2).is_close((1, 2 + 1e-9), abs_tol=1e-6))

    def test_apply_svg_matrix_keeps_extra_dimensions(self):
        r = Vector(1, 2, 7).apply_svg_matrix((2, 0, 0, 3, 10, 20))
        self.assertEqual(r, (12, 26, 7))

    def test_apply_svg_matrix_general_hexad(self):
        # x' = a*x + c*y + e ; y' = b*x + d*y + f
        r = Vector(1, 2).apply_svg_matrix((1, 2, 3, 4, 5, 6))
        self.assertEqual(r, (1 * 1 + 3 * 2 + 5, 2 * 1 + 4 * 2 + 6))

    def test_normalized_honours_epsilon(self):
        with self.assertRaises(ValueError):
            Vector(1e-3, 0).normalized(epsilon=1e-2)
        self.assertVecAlmostEqual(Vector(1e-3, 0).normalized(), (1, 0))


class TestVectorSphericalCoordinates(GeomTestCase):

    def test_general_angles(self):
        # radius 2, azimuth 90 degrees, polar angle 30 degrees from +z
        self.assertVecAlmostEqual(Vector.from_spherical(2, pi / 2, pi / 6), (0, 1, 3**0.5))
        self.assertVecAlmostEqual(Vector.from_spherical(2, 0, pi / 6), (1, 0, 3**0.5))

    def test_result_has_the_requested_radius(self):
        self.assertAlmostEqual(Vector.from_spherical(3, 1.1, 0.7).length(), 3.0)

    def test_poles(self):
        self.assertVecAlmostEqual(Vector.from_spherical(4, 1.0, 0), (0, 0, 4))
        self.assertVecAlmostEqual(Vector.from_spherical(4, 1.0, pi), (0, 0, -4))
