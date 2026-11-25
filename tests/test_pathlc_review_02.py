"""
Review tests for geom/pathlc/*.py.

Two groups:
  * TestPathLCPassingBehavior  -- locks in behaviour that already works.
  * TestPathLCConfirmedBugs    -- asserts the *desired* behaviour for confirmed
    bugs, marked @expectedFailure so the suite is green today and every
    @expectedFailure flips to "unexpected success" once the fix lands.
"""

import math
import unittest

from svggeom.pathlc import PathLC


class TestPathLCPassingBehavior(unittest.TestCase):
    """Behaviour that already works; guards against regressions."""

    def test_parse_move_lineto(self):
        self.assertEqual(PathLC.parse("M0,0 L10,0").describe(), "M0,0L10,0")

    def test_point_at_midpoint(self):
        p = PathLC.parse("M0,0 L10,0")
        pt = p.point_at(0.5)
        self.assertAlmostEqual(pt[0], 5.0)
        self.assertAlmostEqual(pt[1], 0.0)

    def test_point_at_endpoints(self):
        p = PathLC.parse("M2,3 L12,13")
        a = p.point_at(0.0)
        b = p.point_at(1.0)
        self.assertAlmostEqual(a[0], 2.0)
        self.assertAlmostEqual(a[1], 3.0)
        self.assertAlmostEqual(b[0], 12.0)
        self.assertAlmostEqual(b[1], 13.0)

    def test_length_simple(self):
        p = PathLC.parse("M0,0 L10,0 L10,10")
        self.assertAlmostEqual(p.length(), 20.0)

    def test_length_ignores_moves(self):
        p = PathLC.parse("M0,0 L10,0 M20,20 L30,20")
        self.assertAlmostEqual(p.length(), 20.0)

    def test_split_at_half(self):
        p = PathLC.parse("M0,0 L10,0 L20,0")
        left, right = p.split_at(0.5)
        self.assertEqual(left.describe(), "M0,0L10,0")
        self.assertEqual(right.describe(), "M10,0L20,0")

    def test_reverse_simple_line(self):
        p = PathLC.parse("M0,0 L10,0")
        self.assertEqual(p.reversed().describe(), "M10,0L0,0")

    def test_reverse_curve_swaps_endpoints(self):
        p = PathLC.parse("M0,0 C0,10 10,10 10,0")
        rev = p.reversed()
        self.assertAlmostEqual(rev.point_at(0.0)[0], 10.0)
        self.assertAlmostEqual(rev.point_at(0.0)[1], 0.0)
        self.assertAlmostEqual(rev.point_at(1.0)[0], 0.0)
        self.assertAlmostEqual(rev.point_at(1.0)[1], 0.0)

    def test_empty_path_point_at_returns_move(self):
        p = PathLC.at((10, 10))
        pt = p.point_at(0.3)
        self.assertAlmostEqual(pt[0], 10.0)
        self.assertAlmostEqual(pt[1], 10.0)

    def test_segment_at_wraps_without_clamp(self):
        p = PathLC.parse("M0,0 L10,0")
        _seg, t = p.segment_at(1.5)  # wraps to t == 0.5
        self.assertAlmostEqual(t, 0.5)

    def test_point_at_length(self):
        p = PathLC.parse("M0,0 L10,0 L10,10")
        pt = p.point_at_length(15.0)
        self.assertAlmostEqual(pt[0], 10.0)
        self.assertAlmostEqual(pt[1], 5.0)

    def test_bbox_covers_segments(self):
        p = PathLC.parse("M0,0 L10,0 L10,10 Z")
        b = p.bbox()
        self.assertAlmostEqual(b[0][0], 0.0)
        self.assertAlmostEqual(b[1][0], 0.0)
        self.assertAlmostEqual(b[0][1], 10.0)
        self.assertAlmostEqual(b[1][1], 10.0)

    def test_arc_no_rotation_midpoint(self):
        # Circle of radius 10 from (0,0) to (20,0): midpoint x is 10.
        p = PathLC.parse("M0,0 A 10 10 0 0 1 20,0")
        mid = p.point_at(0.5)
        self.assertAlmostEqual(mid[0], 10.0)
        self.assertAlmostEqual(abs(mid[1]), 10.0)


class TestPathLCConfirmedBugs(unittest.TestCase):
    """Desired behaviour for confirmed bugs; flips to unexpected-success on fix."""

    # Bug 1 -- arc rotation is degrees
    @unittest.expectedFailure
    def test_arc_rotation_is_degrees(self):
        # SVG spec: the A-command rotation is in degrees.
        # Therefore 90deg must yield exactly the same arc as PI/2 rad.
        a = PathLC.parse("M0,0 A 10 5 90 0 1 20 0").point_at(0.5)
        b = PathLC.parse(f"M0,0 A 10 5 {math.pi / 2} 0 1 20 0").point_at(0.5)
        self.assertAlmostEqual(a[0], b[0], places=6)
        self.assertAlmostEqual(a[1], b[1], places=6)

    # Bug 2 -- segment_at clamp ignored
    # @unittest.expectedFailure
    def test_segment_at_clamp_is_respected(self):
        p = PathLC.parse("M0,0 L10,0")
        seg, t = p.segment_at(1.5, clamp=True)
        self.assertEqual(t, 1.0)
        self.assertAlmostEqual(seg.point_at(t)[0], 10.0)

    # Bug 3 -- ZeroDivisionError on zero-length path
    # @unittest.expectedFailure
    def test_segment_at_length_zero_length_path(self):
        p = PathLC.at((10, 10))
        seg, _n, _N = p.segment_at_length(5.0)
        self.assertIsNone(seg)

    # @unittest.expectedFailure
    def test_point_at_length_zero_length_path(self):
        p = PathLC.at((10, 10))
        pt = p.point_at_length(5.0)
        self.assertIsNotNone(pt)

    # Bug 4 -- reversed() on a single-move path
    # @unittest.expectedFailure
    def test_reverse_single_move_path(self):
        p = PathLC.at((10, 10))
        rev = p.reversed()
        self.assertEqual(rev.describe(), "M10,10")

    # Bug 5 -- reversed() drops trailing Move subpaths
    # @unittest.expectedFailure
    def test_reverse_keeps_trailing_move(self):
        p = PathLC.parse("M0,0 L10,0 M99,99")
        rev = p.reversed()
        # The (99,99) subpath must survive reversal.
        self.assertIn("99", rev.describe())


if __name__ == "__main__":
    unittest.main()
