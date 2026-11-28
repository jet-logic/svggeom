import math
import unittest

from svggeom.matrix import Matrix

from .helpers import GeomTestCase


class TestMatrix(unittest.TestCase):

    def test_default_is_identity(self):
        self.assertTrue(Matrix().is_identity())

    def test_translate(self):
        m = Matrix().add_translate(10, 20)
        self.assertTrue(m.is_translate())
        pt = m.apply_to_point((5, 5))
        self.assertAlmostEqual(pt[0], 15)
        self.assertAlmostEqual(pt[1], 25)

    def test_scale(self):
        pt = Matrix().add_scale(2, 3).apply_to_point((4, 5))
        self.assertAlmostEqual(pt[0], 8)
        self.assertAlmostEqual(pt[1], 15)

    def test_hexad_roundtrip(self):
        for hexad in ((1, 0, 0, 1, 10, 20), (2, 3, 5, 7, 11, 13), (1, 0, 0, 0, 0, 0)):
            with self.subTest(hexad=hexad):
                self.assertEqual(Matrix.hexad(*hexad).to_hexad(), hexad)

    def test_rotate_about_center_fixes_the_center(self):
        center = (10.0, 10.0)
        m = Matrix().add_rotate(90, center_x=center[0], center_y=center[1])
        pt = m.apply_to_point(center)
        self.assertAlmostEqual(pt[0], center[0])
        self.assertAlmostEqual(pt[1], center[1])


A = (2, 1, -1, 3, 4, 5)
B = (0.5, -2, 1, 1, -3, 7)
IDENTITY = (1, 0, 0, 1, 0, 0)


class TestMatrixConstruction(GeomTestCase):

    def test_hexad_defaults_to_identity(self):
        self.assertEqual(Matrix.hexad().to_hexad(), IDENTITY)
        self.assertTrue(Matrix.hexad().is_identity())

    def test_is_identity_is_false_for_anything_else(self):
        for m in (Matrix().add_translate(1, 0), Matrix().add_scale(2), Matrix().add_rotate(10)):
            self.assertFalse(m.is_identity())

    def test_capply_to_point_takes_and_returns_complex(self):
        self.assertEqual(Matrix().add_translate(1, 2).capply_to_point(3 + 4j), 4 + 6j)
        self.assertEqual(Matrix().add_scale(2, 3).capply_to_point(1 + 1j), 2 + 3j)

    def test_apply_to_point_matches_the_hexad_formula(self):
        a, b, c, d, e, f = A
        x, y = 3, -4
        self.assertVecAlmostEqual(
            Matrix.hexad(*A).apply_to_point((x, y)),
            (a * x + c * y + e, b * x + d * y + f),
        )


class TestMatrixBuilders(GeomTestCase):

    def test_builders_mutate_and_return_the_same_matrix(self):
        m = Matrix()
        for result in (
            m.add_translate(1, 1),
            m.add_scale(2),
            m.add_rotate(10),
            m.add_skewx(10),
            m.add_skewy(10),
            m.add_matrix(1, 0, 0, 1, 1, 1),
            m.add_skew_from_axis(0.1, 10),
        ):
            self.assertIs(result, m)

    def test_translate_accepts_a_pair(self):
        for pair in ((5, 7), [5, 7]):
            with self.subTest(pair=pair):
                self.assertEqual(Matrix().add_translate(pair).to_hexad(), (1, 0, 0, 1, 5, 7))

    def test_translate_y_defaults_to_zero(self):
        self.assertEqual(Matrix().add_translate(5).to_hexad(), (1, 0, 0, 1, 5, 0))

    def test_uniform_scale(self):
        self.assertEqual(Matrix().add_scale(3).to_hexad(), (3, 0, 0, 3, 0, 0))

    def test_rotate_about_a_center_given_as_a_pair(self):
        for center in ((10, 10), [10, 10]):
            with self.subTest(center=center):
                m = Matrix().add_rotate(90, center)
                self.assertVecAlmostEqual(m.apply_to_point((10, 10)), (10, 10))
                # (20, 10) is 10 to the right of the center -> 10 above it.
                self.assertVecAlmostEqual(m.apply_to_point((20, 10)), (10, 20))

    def test_rotation_is_counter_clockwise_in_degrees(self):
        self.assertVecAlmostEqual(Matrix().add_rotate(90).apply_to_point((1, 0)), (0, 1))
        self.assertVecAlmostEqual(Matrix().add_rotate(180).apply_to_point((1, 2)), (-1, -2))

    def test_skewx_and_skewy(self):
        skew_x = Matrix().add_skewx(45)
        self.assertVecAlmostEqual(skew_x.to_hexad(), (1, 0, 1, 1, 0, 0))
        self.assertVecAlmostEqual(skew_x.apply_to_point((0, 1)), (1, 1))
        skew_y = Matrix().add_skewy(45)
        self.assertVecAlmostEqual(skew_y.to_hexad(), (1, 1, 0, 1, 0, 0))
        self.assertVecAlmostEqual(skew_y.apply_to_point((1, 0)), (1, 1))

    def test_add_matrix(self):
        self.assertEqual(Matrix().add_matrix(2, 0, 0, 3, 4, 5).to_hexad(), (2, 0, 0, 3, 4, 5))
        self.assertTrue(Matrix().add_matrix().is_identity())

    def test_later_operations_are_applied_first(self):
        # SVG semantics: translate(1) scale(2) maps x -> 2x + 1.
        self.assertVecAlmostEqual(
            Matrix().add_translate(1, 0).add_scale(2).apply_to_point((1, 0)), (3, 0)
        )
        self.assertVecAlmostEqual(
            Matrix().add_translate(1, 0).add_matrix(2, 0, 0, 2).apply_to_point((1, 0)), (3, 0)
        )
        self.assertVecAlmostEqual(
            Matrix().add_translate(10, 0).add_rotate(90).apply_to_point((1, 0)), (10, 1)
        )

    def test_skew_from_axis_with_no_skew_is_the_identity(self):
        self.assertVecAlmostEqual(Matrix().add_skew_from_axis(0, 30).to_hexad(), IDENTITY)

    def test_skew_from_axis_along_x_is_skewx(self):
        # The skew angle is in degrees, like add_skewx / add_rotate.
        self.assertVecAlmostEqual(
            Matrix().add_skew_from_axis(45, 0).to_hexad(), (1, 0, 1, 1, 0, 0)
        )
        self.assertVecAlmostEqual(
            Matrix().add_skew_from_axis(30, 0).to_hexad(),
            Matrix().add_skewx(30).to_hexad(),
        )

    def test_skew_from_axis_keeps_the_axis_fixed_and_shears_parallel_to_it(self):
        axis = 30
        ux, uy = math.cos(math.radians(axis)), math.sin(math.radians(axis))
        m = Matrix().add_skew_from_axis(45, axis)
        # Points on the axis do not move ...
        self.assertVecAlmostEqual(m.apply_to_point((3 * ux, 3 * uy)), (3 * ux, 3 * uy))
        # ... a point one unit off the axis slides along it by tan(45 deg) = 1.
        self.assertVecAlmostEqual(m.apply_to_point((-uy, ux)), (-uy + ux, ux + uy))


class TestMatrixComposition(GeomTestCase):

    def test_matmul_applies_the_right_operand_first(self):
        translate, scale = Matrix().add_translate(10, 0), Matrix().add_scale(2)
        self.assertVecAlmostEqual((translate @ scale).apply_to_point((1, 0)), (12, 0))
        self.assertVecAlmostEqual((scale @ translate).apply_to_point((1, 0)), (22, 0))

    def test_matmul_matches_applying_each_matrix_in_turn(self):
        a, b = Matrix.hexad(*A), Matrix.hexad(*B)
        p = (3, -4)
        self.assertVecAlmostEqual(
            (a @ b).apply_to_point(p), a.apply_to_point(b.apply_to_point(p))
        )

    def test_matmul_leaves_its_operands_alone(self):
        a, b = Matrix.hexad(*A), Matrix.hexad(*B)
        a @ b
        self.assertEqual(a.to_hexad(), A)
        self.assertEqual(b.to_hexad(), B)

    def test_imatmul_updates_in_place(self):
        m = Matrix().add_translate(10, 0)
        alias = m
        scale = Matrix().add_scale(2)
        m @= scale
        self.assertIs(m, alias)
        self.assertVecAlmostEqual(m.apply_to_point((1, 0)), (12, 0))
        self.assertEqual(scale.to_hexad(), (2, 0, 0, 2, 0, 0))

    def test_identity_is_neutral(self):
        m = Matrix.hexad(*A)
        self.assertVecAlmostEqual((m @ Matrix()).to_hexad(), A)
        self.assertVecAlmostEqual((Matrix() @ m).to_hexad(), A)

    def test_negation_is_the_inverse(self):
        m = Matrix.hexad(*A)
        self.assertVecAlmostEqual((-m @ m).to_hexad(), IDENTITY)
        self.assertVecAlmostEqual((m @ -m).to_hexad(), IDENTITY)
        self.assertVecAlmostEqual((-m).apply_to_point(m.apply_to_point((7, -2))), (7, -2))


class TestMatrixInversion(unittest.TestCase):

    def test_singular_matrices_cannot_be_inverted(self):
        for name, m in (
            ("collapsed y", Matrix.hexad(1, 0, 0, 0)),
            ("all zero", Matrix.hexad(0, 0, 0, 0)),
            ("rank one", Matrix.hexad(1, 2, 2, 4)),
        ):
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, "singular"):
                    -m


class TestMatrixClassification(unittest.TestCase):

    def test_is_translate(self):
        self.assertTrue(Matrix().add_translate(5, 3).is_translate())
        self.assertTrue(Matrix().is_translate())
        self.assertFalse(Matrix().add_scale(2).is_translate())
        self.assertFalse(Matrix().add_rotate(30).is_translate())

    def test_is_scale(self):
        self.assertTrue(Matrix().add_scale(2, 3).is_scale())
        self.assertFalse(Matrix().add_translate(5, 3).is_scale())
        self.assertFalse(Matrix().add_rotate(90).is_scale())
        self.assertFalse(Matrix().add_skewx(30).is_scale())

    def test_is_rotate(self):
        self.assertTrue(Matrix().add_rotate(30).is_rotate())
        self.assertFalse(Matrix().add_scale(2).is_rotate())  # not unit length
        self.assertFalse(Matrix().add_rotate(30).add_translate(1, 0).is_rotate())
        self.assertFalse(Matrix().add_skewx(45).is_rotate())
        self.assertFalse(Matrix().add_scale(2, 3).is_rotate())  # non-uniform

    def test_tolerance_versus_exactly(self):
        almost_translate = Matrix.hexad(1 + 1e-6, 0, 0, 1, 5, 5)
        self.assertTrue(almost_translate.is_translate())
        self.assertFalse(almost_translate.is_translate(exactly=True))
        almost_scale = Matrix.hexad(2, 1e-7, 0, 3)
        self.assertTrue(almost_scale.is_scale())
        self.assertFalse(almost_scale.is_scale(exactly=True))
        almost_rotate = Matrix.hexad(1 + 1e-6, 0, 0, 1 + 1e-6)
        self.assertTrue(almost_rotate.is_rotate())
        self.assertFalse(almost_rotate.is_rotate(exactly=True))

    def test_rotation_degrees(self):
        for deg in (30, 90, -45, 179):
            with self.subTest(deg=deg):
                self.assertAlmostEqual(Matrix().add_rotate(deg).rotation_degrees(), deg)

    def test_rotation_degrees_ignores_uniform_scale(self):
        self.assertAlmostEqual(Matrix().add_rotate(90).add_scale(2).rotation_degrees(), 90)

    def test_rotation_degrees_is_undefined_for_non_uniform_or_skewed_matrices(self):
        for name, m in (
            ("non-uniform scale", Matrix().add_scale(2, 3)),
            ("skew", Matrix().add_skewx(30)),
        ):
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    m.rotation_degrees()


class TestMatrixStr(unittest.TestCase):

    def test_identity_is_the_empty_string(self):
        self.assertEqual(str(Matrix()), "")
        self.assertEqual(str(Matrix().add_translate(0, 0)), "")
        self.assertEqual(str(Matrix.hexad()), "")

    def test_translate(self):
        self.assertEqual(str(Matrix().add_translate(5, 3)), "translate(5, 3)")

    def test_scale(self):
        self.assertEqual(str(Matrix().add_scale(2, 3)), "scale(2, 3)")

    def test_rotate(self):
        self.assertEqual(str(Matrix().add_rotate(90)), "rotate(90)")

    def test_general_matrix(self):
        self.assertEqual(str(Matrix.hexad(1, 2, 3, 4, 5, 6)), "matrix(1 2 3 4 5 6)")

    def test_rotation_about_a_center_is_a_general_matrix(self):
        self.assertTrue(str(Matrix().add_rotate(90, 10, 10)).startswith("matrix("))

    def test_numbers_use_six_significant_digits(self):
        self.assertEqual(str(Matrix().add_translate(1 / 3, 2 / 3)), "translate(0.333333, 0.666667)")


class TestMatrixToleranceBoundary(unittest.TestCase):

    def test_values_exactly_at_the_tolerance_still_count(self):
        tol = Matrix.absolute_tolerance
        self.assertTrue(Matrix.hexad(1, tol, 0, 1).is_translate())
        self.assertTrue(Matrix.hexad(1, 0, tol, 1).is_translate())
        self.assertFalse(Matrix.hexad(1, 2 * tol, 0, 1).is_translate())
        self.assertFalse(Matrix.hexad(1, 0, 2 * tol, 1).is_translate())
