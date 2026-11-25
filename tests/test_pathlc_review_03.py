"""Tests for geom.pathlc: confirmed bugs (expectedFailure) and working behaviour."""

import unittest
from math import isclose, pi

from svggeom.pathlc import PathLC
from svggeom.pathlc.command import (
    ArcLC,
    CloseLC,
    CubicLC,
    LineLC,
    MoveLC,
    QuadLC,
)
from svggeom.vector import vec
from svggeom.matrix import Matrix


class TestPathLCWorking(unittest.TestCase):
    """Behaviour that currently works; regressions should fail these."""

    def test_parse_line_and_length(self):
        p = PathLC.parse("M0,0 L10,0")
        self.assertEqual(p.describe(), "M0,0L10,0")
        self.assertAlmostEqual(p.length(), 10.0)

    def test_parse_closed_square(self):
        p = PathLC.parse("M0,0 H10 V10 H0 Z")
        self.assertAlmostEqual(p.length(), 40.0)
        self.assertTrue(isinstance(p._tail, CloseLC))

    def test_point_at_midpoint_line(self):
        p = PathLC.parse("M0,0 L20,0")
        pt = p.point_at(0.5)
        self.assertTrue(isclose(pt[0], 10.0) and isclose(pt[1], 0.0))

    def test_split_at_half(self):
        p = PathLC.parse("M0,0 L40,0")
        left, right = p.split_at(0.5)
        self.assertAlmostEqual(left.length(), 20.0)
        self.assertAlmostEqual(right.length(), 20.0)
        self.assertTrue(isclose(left.point_at(1)[0], 20.0))
        self.assertTrue(isclose(right.point_at(0)[0], 20.0))

    def test_reverse_open_path(self):
        p = PathLC.parse("M0,0 L10,0 L10,10")
        r = p.reversed()
        self.assertAlmostEqual(r.length(), 20.0)
        self.assertTrue(
            isclose(r.point_at(0)[0], 10.0) and isclose(r.point_at(0)[1], 10.0)
        )
        self.assertTrue(
            isclose(r.point_at(1)[0], 0.0) and isclose(r.point_at(1)[1], 0.0)
        )

    def test_arc_small_vs_large_length(self):
        small = PathLC.parse("M0,0 A10,10 0 0 1 10,0")
        large = PathLC.parse("M0,0 A10,10 0 1 1 10,0")
        self.assertAlmostEqual(small.length(), 10 * (pi / 3), places=3)
        self.assertAlmostEqual(large.length(), 10 * (5 * pi / 3), places=3)

    def test_cubic_bbox_and_point(self):
        p = PathLC.parse("M0,0 C0,20 40,20 40,0")
        bb = p.bbox()
        self.assertTrue(bb)
        mid = p.point_at(0.5)
        self.assertTrue(isclose(mid[0], 20.0, abs_tol=1e-6))

    def test_transform_translate_scale(self):
        p = PathLC.parse("M0,0 L10,0")
        m = Matrix().add_translate(5, 5).add_scale(2)
        # Note: matrix application order depends on @ implementation
        pt = p.transform(m.to_hexad())
        self.assertTrue(pt.bbox())

    def test_as_curve_line_to_cubic(self):
        p = PathLC.parse("M0,0 L10,10")
        c = p.as_curve()
        self.assertIsInstance(c._tail, CubicLC)
        self.assertAlmostEqual(c.length(), p.length(), places=3)

    def test_relative_commands(self):
        p = PathLC.parse("M10,10 l5,0 l0,5")
        self.assertEqual(p.describe(), "M10,10L15,10L15,15")
        self.assertAlmostEqual(p.length(), 10.0)


class TestPathLCBugs(unittest.TestCase):
    """Confirmed bugs: tests assert DESIRED behaviour; marked expectedFailure."""

    # @unittest.expectedFailure
    def test_arc_same_start_end_does_not_raise(self):
        """SVG: identical endpoints -> omit the arc segment entirely.
        Currently raises ZeroDivisionError in arc_params / ArcLC.__init__.
        """
        # Should not raise; equivalent to M5,5 (or M5,5 L5,5 depending on choice)
        p = PathLC.parse("M5,5 A10,10 0 0 1 5,5")
        # After omit, length should be 0
        self.assertAlmostEqual(p.length(), 0.0)
        # And point queries should succeed
        self.assertIsNotNone(p.point_at(0))

    # @unittest.expectedFailure
    def test_arc_same_start_end_in_middle_omitted(self):
        """Arc with identical endpoints in the middle of a path should be omitted,
        not crash the parser.
        """
        p = PathLC.parse("M0,0 L10,0 A5,5 0 0 1 10,0 L20,0")
        # Desired: behaves like M0,0 L10,0 L20,0  (arc omitted)
        self.assertAlmostEqual(p.length(), 20.0)
        self.assertTrue(isclose(p.point_at(1)[0], 20.0))

    # @unittest.expectedFailure
    def test_point_at_length_zero_length_path(self):
        """point_at_length on a zero-length path should return the single point,
        consistent with point_at. Currently returns None when segment length N==0.
        """
        p = PathLC.parse("M0,0 L0,0")
        self.assertAlmostEqual(p.length(), 0.0)
        pt = p.point_at_length(0)
        self.assertIsNotNone(pt)
        self.assertTrue(isclose(pt[0], 0.0) and isclose(pt[1], 0.0))

    @unittest.expectedFailure
    def test_reverse_move_only_path(self):
        """Reversing a path that is only a Move should succeed and yield an
        equivalent path. Currently raises RuntimeError('Can not reverse').
        """
        p = PathLC.parse("M5,5")
        r = p.reversed()
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r.length(), 0.0)
        self.assertTrue(
            isclose(r.point_at(0)[0], 5.0) and isclose(r.point_at(0)[1], 5.0)
        )

    @unittest.expectedFailure
    def test_reverse_closed_path_preserves_closedness(self):
        """Reversing a path that ends with Z should produce a path that is still
        closed (ends with Z / CloseLC). Currently the Close is turned into a
        plain LineLC, so the result is open.
        """
        p = PathLC.parse("M0,0 L10,0 L10,10 L0,10 Z")
        self.assertIsInstance(p._tail, CloseLC)
        r = p.reversed()
        # Desired: still closed
        self.assertIsInstance(r._tail, CloseLC)
        # And geometry starts at original end and ends at original start
        self.assertTrue(
            isclose(r.point_at(0)[0], 0.0) and isclose(r.point_at(0)[1], 0.0)
        )
        self.assertAlmostEqual(r.length(), p.length())


if __name__ == "__main__":
    unittest.main()
