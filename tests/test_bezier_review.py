"""
`TestBezierPassingBehavior` guards behaviour that already works.
`TestBezierConfirmedBugs` asserts the desired behaviour for confirmed bugs;
they are currently @expectedFailure and will flip to "unexpected success"
once the corresponding fix lands, so the suite passes today.
"""

import unittest
from math import cos, sin, pi

from svggeom.bezier import Bezier
from svggeom.matrix import Matrix


class TestBezierPassingBehavior(unittest.TestCase):

    def test_move_to_creates_first_point(self):
        b = Bezier()
        b.move_to((10, 20))
        self.assertEqual(len(b), 1)
        self.assertEqual(b[0], ((10, 20), (0, 0), (0, 0)))

    def test_move_to_second_call_replaces_point(self):
        b = Bezier()
        b.move_to((0, 0))
        b.move_to((10, 20))
        self.assertEqual(len(b), 1)
        self.assertEqual(b[0][0], (10, 20))

    def test_move_to_third_call_raises(self):
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

    def test_curve_to_sets_tangents(self):
        # Locks in the curve_to convention: tangent = vertex - control_point.
        b = Bezier()
        b.move_to((0, 0))
        b.curve_to((0, 10), (10, 10), (10, 0))
        self.assertEqual(tuple(b[0][0]), (0, 0))
        # out = (0, 0) - (0, 10) = (0, -10)
        self.assertEqual(tuple(b[0][2]), (0, -10))
        self.assertEqual(tuple(b[1][0]), (10, 0))
        # in = (10, 0) - (10, 10) = (0, -10)
        self.assertEqual(tuple(b[1][1]), (0, -10))

    def test_close_appends_duplicate_endpoint(self):
        b = Bezier()
        b.move_to((0, 0))
        b.line_to((10, 0))
        b.line_to((10, 10))
        b.close()
        self.assertEqual(len(b), 4)
        self.assertEqual(tuple(b[-1][0]), (0, 0))
        self.assertEqual(tuple(b[0][0]), (0, 0))

    def test_is_closed_false_before_close(self):
        b = Bezier()
        b.move_to((0, 0))
        b.line_to((10, 0))
        b.line_to((10, 10))
        self.assertFalse(b.is_closed())

    def test_is_closed_true_after_close(self):
        b = Bezier()
        b.move_to((0, 0))
        b.line_to((10, 0))
        b.line_to((10, 10))
        b.close()
        self.assertTrue(b.is_closed())

    def test_reverse_dir_swaps_in_out(self):
        b = Bezier()
        b.move_to((0, 0))
        b.curve_to((0, 10), (10, 10), (10, 0))
        original = [tuple(pt) for pt in b]
        b.reverse_dir()
        self.assertEqual(tuple(b[0][0]), tuple(original[-1][0]))
        self.assertEqual(tuple(b[-1][0]), tuple(original[0][0]))
        for (v_o, i_o, o_o), (v_r, i_r, o_r) in zip(reversed(original), b):
            self.assertEqual(tuple(v_r), tuple(v_o))
            self.assertEqual(tuple(i_r), tuple(o_o))  # in_new == out_old
            self.assertEqual(tuple(o_r), tuple(i_o))  # out_new == in_old

    def test_quad_to_converts_to_cubic_correctly(self):
        # Quad: (0,0) --cp-- (1,1) --cp-- (2,0)
        # Cubic control points should be (2/3, 2/3) and (4/3, 2/3).
        # Under curve_to convention: out = -c1, in = -c2_relative.
        b = Bezier()
        b.move_to((0, 0))
        b.quad_to((1, 1), (2, 0))
        self.assertAlmostEqual(b[0][2][0], -2 / 3, places=9)
        self.assertAlmostEqual(b[0][2][1], -2 / 3, places=9)
        self.assertAlmostEqual(b[1][1][0], 2 / 3, places=9)
        self.assertAlmostEqual(b[1][1][1], -2 / 3, places=9)

    def test_single_vertex_bezier_has_point_bbox(self):
        """A Bezier containing only a move point has a degenerate point bbox."""
        b = Bezier()
        b.move_to((4, 7))
        self.assertEqual(b.bounding_box(), ((4, 4), (7, 7)))

    def test_bounding_box_straight_line(self):
        # Straight lines have zero tangents, so + and - agree and bbox is
        # correct even in the currently-buggy implementation.
        b = Bezier()
        b.move_to((0, 0))
        b.line_to((10, 20))
        bb = b.bounding_box()
        if bb:
            self.assertAlmostEqual(bb[0][0], 0.0)
            self.assertAlmostEqual(bb[1][0], 0.0)
            self.assertAlmostEqual(bb[0][1], 10.0)
            self.assertAlmostEqual(bb[1][1], 20.0)
        else:
            self.assertIsNotNone(bb)

    def test_transform_translate_moves_vertices(self):
        # Only checks vertices -- passes even with the buggy tangent handling.
        b = Bezier()
        b.move_to((0, 0))
        b.line_to((10, 20))
        b.transform(Matrix().add_translate(5, 3))
        self.assertAlmostEqual(b[0][0][0], 5)
        self.assertAlmostEqual(b[0][0][1], 3)
        self.assertAlmostEqual(b[1][0][0], 15)
        self.assertAlmostEqual(b[1][0][1], 23)

    def test_transform_rotation_preserves_tangents(self):
        # Rotation (pure linear, no translation) is already correct today.
        b = Bezier()
        b.move_to((0, 0))
        b.curve_to((0, 10), (10, 10), (10, 0))
        original = [tuple(pt) for pt in b]
        theta = pi / 2
        b.transform(Matrix().add_rotate(theta * 180 / pi))
        ct, st = cos(theta), sin(theta)
        for (_, i_o, o_o), (_, i_n, o_n) in zip(original, b):
            i_rot = (i_o[0] * ct - i_o[1] * st, i_o[0] * st + i_o[1] * ct)
            o_rot = (o_o[0] * ct - o_o[1] * st, o_o[0] * st + o_o[1] * ct)
            self.assertAlmostEqual(i_n[0], i_rot[0], places=9)
            self.assertAlmostEqual(i_n[1], i_rot[1], places=9)
            self.assertAlmostEqual(o_n[0], o_rot[0], places=9)
            self.assertAlmostEqual(o_n[1], o_rot[1], places=9)

    # in TestBezierConfirmedBugs, replacing the old passing test:

    # @unittest.expectedFailure
    def test_transform_translate_preserves_line_tangents(self):
        """Translation of a straight line must leave tangents at zero."""
        b = Bezier()
        b.move_to((0, 0))
        b.line_to((10, 20))
        b.transform(Matrix().add_translate(5, 3))
        # Vertices translate
        self.assertAlmostEqual(b[0][0][0], 5)
        self.assertAlmostEqual(b[0][0][1], 3)
        self.assertAlmostEqual(b[1][0][0], 15)
        self.assertAlmostEqual(b[1][0][1], 23)
        # Tangents stay zero -- a translation must not bend a straight line
        for in_tan, out_tan in (b[0][1:], b[1][1:]):
            self.assertAlmostEqual(in_tan[0], 0)
            self.assertAlmostEqual(in_tan[1], 0)
            self.assertAlmostEqual(out_tan[0], 0)
            self.assertAlmostEqual(out_tan[1], 0)

    def test_transform_identity_is_noop(self):
        b = Bezier()
        b.move_to((0, 0))
        b.curve_to((0, 10), (10, 10), (10, 0))
        original = [tuple(pt) for pt in b]
        b.transform(Matrix())
        for old, new in zip(original, b):
            self.assertAlmostEqual(tuple(old[0]), tuple(new[0]))
            self.assertAlmostEqual(tuple(old[1]), tuple(new[1]))
            self.assertAlmostEqual(tuple(old[2]), tuple(new[2]))


class TestBezierConfirmedBugs(unittest.TestCase):

    # @unittest.expectedFailure
    def test_bounding_box_curved_upward(self):
        """Bug 1: control points reconstructed with +tangent instead of -tangent."""
        b = Bezier()
        b.move_to((0, 0))
        b.curve_to((0, 10), (10, 10), (10, 0))
        bb = b.bounding_box()
        # True curve: x in [0,10], y in [0, 7.5]
        self.assertIsNotNone(bb)
        if bb:
            self.assertAlmostEqual(bb[0][0], 0.0, places=6)
            self.assertAlmostEqual(bb[0][1], 10.0, places=6)
            self.assertAlmostEqual(bb[1][0], 0.0, places=6)
            self.assertGreater(bb[1][1], 5.0)

    # @unittest.expectedFailure
    def test_bounding_box_curved_leftward(self):
        """Bug 1 alternate axis: a leftward bulge must appear in bb.left."""
        b = Bezier()
        b.move_to((0, 0))
        b.curve_to((-10, 0), (-10, 10), (0, 10))
        bb = b.bounding_box()
        # True curve: x in [-7.5, 0], y in [0, 10]
        self.assertAlmostEqual(bb[0][0], -7.5, places=3)
        self.assertAlmostEqual(bb[0][1], 0.0, places=6)

    # @unittest.expectedFailure
    def test_transform_translation_preserves_tangents(self):
        """Bug 2: translation is applied to tangents, deforming every curve."""
        b = Bezier()
        b.move_to((0, 0))
        b.curve_to((0, 10), (10, 10), (10, 0))
        original = [tuple(pt) for pt in b]
        b.transform(Matrix().add_translate(100, 200))
        for (v_o, i_o, o_o), (v_n, i_n, o_n) in zip(original, b):
            self.assertAlmostEqual(v_n[0], v_o[0] + 100)
            self.assertAlmostEqual(v_n[1], v_o[1] + 200)
            # Tangents are translation-invariant under a linear map.
            self.assertAlmostEqual(i_n[0], i_o[0], places=6)
            self.assertAlmostEqual(i_n[1], i_o[1], places=6)
            self.assertAlmostEqual(o_n[0], o_o[0], places=6)
            self.assertAlmostEqual(o_n[1], o_o[1], places=6)

    # @unittest.expectedFailure
    def test_curve_to_empty_raises_value_error(self):
        """Bug 3: empty Bezier + curve_to raises IndexError, not ValueError."""
        b = Bezier()
        with self.assertRaises(ValueError):
            b.curve_to((0, 0), (1, 1), (2, 2))

    # @unittest.expectedFailure
    def test_close_empty_raises_value_error(self):
        """Bug 4: empty Bezier + close raises IndexError, not ValueError."""
        b = Bezier()
        with self.assertRaises(ValueError):
            b.close()


if __name__ == "__main__":
    unittest.main()
