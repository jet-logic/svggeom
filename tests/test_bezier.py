import unittest
from math import cos, pi, sin

from svggeom.bezier import Bezier
from svggeom.matrix import Matrix


def cubic_arch() -> Bezier:
    """(0,0) -> (10,0) via control points (0,10) and (10,10)."""
    b = Bezier()
    b.move_to((0, 0))
    b.curve_to((0, 10), (10, 10), (10, 0))
    return b


class TestBezierConstruction(unittest.TestCase):

    def test_move_to_creates_first_point(self):
        b = Bezier()
        b.move_to((10, 20))
        self.assertEqual(len(b), 1)
        self.assertEqual(b[0], ((10, 20), (0, 0), (0, 0)))

    def test_second_move_to_replaces_the_point(self):
        b = Bezier()
        b.move_to((0, 0))
        b.move_to((10, 20))
        self.assertEqual(len(b), 1)
        self.assertEqual(b[0][0], (10, 20))

    def test_move_to_after_drawing_raises(self):
        b = Bezier()
        b.move_to((0, 0))
        b.line_to((10, 0))
        with self.assertRaises(RuntimeError):
            b.move_to((0, 0))

    def test_line_to_appends_zero_tangent_point(self):
        b = Bezier()
        b.move_to((0, 0))
        b.line_to((10, 10))
        self.assertEqual(len(b), 2)
        self.assertEqual(tuple(b[1][0]), (10, 10))
        self.assertEqual(tuple(b[1][1]), (0, 0))
        self.assertEqual(tuple(b[1][2]), (0, 0))

    def test_curve_to_stores_tangents_as_vertex_minus_control_point(self):
        b = cubic_arch()
        self.assertEqual(tuple(b[0][0]), (0, 0))
        self.assertEqual(tuple(b[0][2]), (0, -10))  # out = (0,0) - (0,10)
        self.assertEqual(tuple(b[1][0]), (10, 0))
        self.assertEqual(tuple(b[1][1]), (0, -10))  # in  = (10,0) - (10,10)

    def test_quad_to_converts_to_the_equivalent_cubic(self):
        # Quad (0,0) -(1,1)- (2,0) has cubic control points (2/3,2/3) and (4/3,2/3).
        b = Bezier()
        b.move_to((0, 0))
        b.quad_to((1, 1), (2, 0))
        self.assertAlmostEqual(b[0][2][0], -2 / 3, places=9)
        self.assertAlmostEqual(b[0][2][1], -2 / 3, places=9)
        self.assertAlmostEqual(b[1][1][0], 2 / 3, places=9)
        self.assertAlmostEqual(b[1][1][1], -2 / 3, places=9)

    def test_close_appends_a_copy_of_the_first_point(self):
        b = Bezier()
        b.move_to((0, 0))
        b.line_to((10, 0))
        b.line_to((10, 10))
        self.assertFalse(b.is_closed())
        b.close()
        self.assertTrue(b.is_closed())
        self.assertEqual(len(b), 4)
        self.assertEqual(tuple(b[-1][0]), (0, 0))
        self.assertEqual(tuple(b[0][0]), (0, 0))

    def test_drawing_without_a_current_point_raises_value_error(self):
        calls = {
            "curve_to": lambda b: b.curve_to((1, 1), (2, 2), (3, 3)),
            "quad_to": lambda b: b.quad_to((1, 1), (2, 2)),
            "arc_to": lambda b: b.arc_to(1, 1, 0, False, True, (2, 2)),
        }
        for name, call in calls.items():
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, "preceding move_to"):
                    call(Bezier())

    def test_close_on_empty_bezier_raises_value_error(self):
        with self.assertRaises(ValueError):
            Bezier().close()

    def test_reverse_dir_swaps_in_and_out_tangents(self):
        b = cubic_arch()
        original = [tuple(pt) for pt in b]
        b.reverse_dir()
        self.assertEqual(tuple(b[0][0]), tuple(original[-1][0]))
        self.assertEqual(tuple(b[-1][0]), tuple(original[0][0]))
        for (v_o, i_o, o_o), (v_r, i_r, o_r) in zip(reversed(original), b):
            self.assertEqual(tuple(v_r), tuple(v_o))
            self.assertEqual(tuple(i_r), tuple(o_o))  # in_new == out_old
            self.assertEqual(tuple(o_r), tuple(i_o))  # out_new == in_old


class TestBezierBoundingBox(unittest.TestCase):

    def test_single_vertex_is_a_point_box(self):
        b = Bezier()
        b.move_to((4, 7))
        self.assertEqual(b.bounding_box(), ((4, 4), (7, 7)))

    def test_straight_lines(self):
        for end, expected in (((10, 0), ((0, 10), (0, 0))), ((10, 20), ((0, 10), (0, 20)))):
            with self.subTest(end=end):
                b = Bezier()
                b.move_to((0, 0))
                b.line_to(end)
                self.assertEqual(b.bounding_box(), expected)

    def test_curve_bulging_up(self):
        bb = cubic_arch().bounding_box()
        # True curve: x in [0, 10], y in [0, 7.5]
        self.assertAlmostEqual(bb[0][0], 0.0, places=6)
        self.assertAlmostEqual(bb[0][1], 10.0, places=6)
        self.assertAlmostEqual(bb[1][0], 0.0, places=6)
        self.assertAlmostEqual(bb[1][1], 7.5, places=6)

    def test_curve_bulging_left(self):
        b = Bezier()
        b.move_to((0, 0))
        b.curve_to((-10, 0), (-10, 10), (0, 10))
        bb = b.bounding_box()
        # True curve: x in [-7.5, 0], y in [0, 10]
        self.assertAlmostEqual(bb[0][0], -7.5, places=3)
        self.assertAlmostEqual(bb[0][1], 0.0, places=6)


class TestBezierTransform(unittest.TestCase):

    def assertPointsAlmostEqual(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for p_a, p_e in zip(actual, expected):
            for a, e in zip(p_a, p_e):
                self.assertAlmostEqual(a, e, places=9)

    def test_identity_is_a_noop(self):
        b = cubic_arch()
        original = [tuple(pt) for pt in b]
        b.transform(Matrix())
        for old, new in zip(original, b):
            self.assertPointsAlmostEqual(old, new)

    def test_translate_moves_vertices_but_not_tangents_of_a_line(self):
        b = Bezier()
        b.move_to((0, 0))
        b.line_to((10, 20))
        b.transform(Matrix().add_translate(5, 3))
        self.assertEqual(tuple(b[0][0]), (5, 3))
        self.assertEqual(tuple(b[1][0]), (15, 23))
        for point in b:
            self.assertEqual(tuple(point[1]), (0, 0))
            self.assertEqual(tuple(point[2]), (0, 0))

    def test_translate_preserves_curve_tangents(self):
        b = cubic_arch()
        original = [tuple(pt) for pt in b]
        b.transform(Matrix().add_translate(100, 200))
        for (v_o, i_o, o_o), (v_n, i_n, o_n) in zip(original, b):
            self.assertAlmostEqual(v_n[0], v_o[0] + 100)
            self.assertAlmostEqual(v_n[1], v_o[1] + 200)
            self.assertPointsAlmostEqual((i_n, o_n), (i_o, o_o))

    def test_rotation_rotates_tangents(self):
        b = cubic_arch()
        original = [tuple(pt) for pt in b]
        theta = pi / 2
        b.transform(Matrix().add_rotate(theta * 180 / pi))
        ct, st = cos(theta), sin(theta)

        def rot(v):
            return (v[0] * ct - v[1] * st, v[0] * st + v[1] * ct)

        for (_, i_o, o_o), (_, i_n, o_n) in zip(original, b):
            self.assertPointsAlmostEqual((i_n, o_n), (rot(i_o), rot(o_o)))
