# tests/geom/test_pathlc_review.py
"""
Review tests for geom/pathlc.

Every @expectedFailure test asserts DESIRED behaviour for a confirmed bug.
When the bug is fixed the test flips to "unexpected success" and flags it,
so the suite passes today and alerts you the day you fix it.
"""

import unittest

from svggeom.pathlc import PathLC
from svggeom.pathlc.command import LineLC
from svggeom.matrix import Matrix


class TestPassingPathLCBehavior(unittest.TestCase):
    """Regression guards for behaviour that works today."""

    def test_parse_and_describe_roundtrip_simple(self):
        d = "M0,0 L10,0 L10,10 Z"
        p = PathLC.parse(d)
        self.assertIn("M0,0", p.describe())
        self.assertTrue("Z" in p.describe())

    def test_square_length(self):
        p = PathLC.parse("M0,0 L10,0 L10,10 L0,10 Z")
        self.assertAlmostEqual(p.length(), 40.0)

    def test_square_bbox(self):
        p = PathLC.parse("M0,0 L10,0 L10,10 L0,10 Z")
        b = p.bbox()
        self.assertAlmostEqual(b[0][0], 0)
        self.assertAlmostEqual(b[1][0], 0)
        self.assertAlmostEqual(b[0][1], 10)
        self.assertAlmostEqual(b[1][1], 10)

    def test_point_at_endpoints(self):
        p = PathLC.parse("M0,0 L10,0 L10,10")
        self.assertAlmostEqual(p.point_at(0)[0], 0)
        self.assertAlmostEqual(p.point_at(0)[1], 0)
        self.assertAlmostEqual(p.point_at(1)[0], 10)
        self.assertAlmostEqual(p.point_at(1)[1], 10)

    def test_point_at_midpoint_of_line(self):
        p = PathLC.parse("M0,0 L10,0")
        pt = p.point_at(0.5)
        self.assertAlmostEqual(pt[0], 5.0)
        self.assertAlmostEqual(pt[1], 0.0)

    def test_segment_at_length_mid(self):
        p = PathLC.parse("M0,0 L10,0 L10,10")
        seg, n, N = p.segment_at_length(5.0)
        self.assertIsInstance(seg, LineLC)
        self.assertAlmostEqual(n, 5.0)
        self.assertAlmostEqual(N, 10.0)

    def test_reversed_open_path(self):
        p = PathLC.parse("M0,0 L10,0 L20,5")
        r = p.reversed()
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r.point_at(0)[0], 20)
        self.assertAlmostEqual(r.point_at(0)[1], 5)
        self.assertAlmostEqual(r.point_at(1)[0], 0)
        self.assertAlmostEqual(r.point_at(1)[1], 0)

    def test_transform_translate(self):
        p = PathLC.parse("M0,0 L10,0")
        m = Matrix().add_translate(5, 3)
        tp = p.transform(m.to_hexad())
        self.assertAlmostEqual(tp.point_at(0)[0], 5)
        self.assertAlmostEqual(tp.point_at(0)[1], 3)
        self.assertAlmostEqual(tp.point_at(1)[0], 15)
        self.assertAlmostEqual(tp.point_at(1)[1], 3)

    def test_bbox_single_line_no_prior_moves(self):
        # Regression: with a *single* leading move, bbox is already correct.
        p = PathLC.parse("M100,100 L110,110")
        b = p.bbox()
        self.assertAlmostEqual(b[0][0], 100)
        self.assertAlmostEqual(b[1][0], 100)
        self.assertAlmostEqual(b[0][1], 110)
        self.assertAlmostEqual(b[1][1], 110)


    def test_svg_arc_rotation_uses_degrees(self):
        """SVG A-command rotation is in degrees, not radians."""
        quarter_turn = PathLC.parse("M0,0 A10,5 90 0 1 20,0").point_at(0.5)
        equivalent_turn = PathLC.parse("M0,0 A10,5 450 0 1 20,0").point_at(0.5)
        self.assertAlmostEqual(quarter_turn[0], equivalent_turn[0], places=6)
        self.assertAlmostEqual(quarter_turn[1], equivalent_turn[1], places=6)


class TestPathLCConfirmedBugs(unittest.TestCase):
    """@expectedFailure: each test asserts the *desired* behaviour."""

    # @unittest.expectedFailure
    def test_move_does_not_expand_bbox(self):
        """
        A MoveLC has zero length and must not contribute area to the bbox
        (bug: it currently contributes the rectangle between prev.to() and to()).
        """
        p = PathLC.parse("M0,0 M100,100 L110,110")
        b = p.bbox()
        self.assertAlmostEqual(b[0][0], 100)
        self.assertAlmostEqual(b[1][0], 100)
        self.assertAlmostEqual(b[0][1], 110)
        self.assertAlmostEqual(b[1][1], 110)

    # @unittest.expectedFailure
    def test_cut_at_one_returns_full_path(self):
        """
        cut_at(1.0) should return the whole path
        (bug: 1.0 % 1.0 == 0.0 -> returns the empty prefix).
        """
        p = PathLC.parse("M0,0 L10,0 L20,0")
        c = p.cut_at(1.0)
        self.assertAlmostEqual(c.length(), 20.0)

    # @unittest.expectedFailure
    def test_cut_at_negative_returns_suffix(self):
        """
        Per the docstring, cut_at(-x) should wrap and return the *suffix*
        from (-x % 1) to the end.
        (bug: it returns the prefix [0, x % 1].)
        """
        p = PathLC.parse("M0,0 L10,0")  # length 10
        c = p.cut_at(-0.5)  # desired: [5, 10]
        self.assertAlmostEqual(c.length(), 5.0)
        self.assertAlmostEqual(c.point_at(0)[0], 5.0)
        self.assertAlmostEqual(c.point_at(1)[0], 10.0)

    # @unittest.expectedFailure
    def test_crop_at_middle_of_line(self):
        """
        crop_at(0.3, 0.7) on a length-100 line should span x=30..70
        (bug: broken by cut_at(<0); current result is x=0..17.14).
        """
        p = PathLC.parse("M0,0 L100,0")
        c = p.crop_at(0.3, 0.7)
        self.assertAlmostEqual(c.point_at(0)[0], 30.0, places=3)
        self.assertAlmostEqual(c.point_at(1)[0], 70.0, places=3)

    # @unittest.expectedFailure
    def test_arc_transform_with_singular_matrix(self):
        """
        A singular transform (det == 0) should degrade gracefully,
        not raise UnboundLocalError from get_degen().
        """
        p = PathLC.parse("M0,0 A10,10 0 0 0 10,10")
        zero = Matrix.hexad(0, 0, 0, 0, 0, 0)
        # Desired: no exception
        _ = p.transform(zero.to_hexad())


if __name__ == "__main__":
    unittest.main()
