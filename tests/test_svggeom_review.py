"""Focused review tests for the svggeom project.

Run from the repository root with:
    python -m unittest -v test_svggeom_review

Confirmed bugs are expressed as desired-behaviour tests decorated with
@unittest.expectedFailure. Remove that decorator when the corresponding fix
is implemented. Passing tests protect existing behaviour.
"""

import math
import unittest

from svggeom.bbox import BoundingBox
from svggeom.bezier import Bezier
from svggeom.pathlc import PathLC


class TestWorkingBehavior(unittest.TestCase):
    def test_line_path_length_and_midpoint(self):
        path = PathLC.parse("M0,0 L10,0")
        self.assertAlmostEqual(path.length(), 10.0)
        self.assertAlmostEqual(path.point_at(0.5)[0], 5.0)
        self.assertAlmostEqual(path.point_at(0.5)[1], 0.0)

    def test_rectangle_bbox_and_resize_uniform(self):
        box = BoundingBox.from_rect(0, 0, 10, 20)
        self.assertEqual(box.dump_rect(), (0, 0, 10, 20))
        self.assertEqual(box.resize(2).dump_rect(), (-2, -2, 14, 24))

    def test_nonempty_bezier_curve_bbox(self):
        curve = Bezier()
        curve.move_to((0, 0))
        curve.line_to((10, 0))
        self.assertEqual(curve.bounding_box(), ((0, 10), (0, 0)))


class TestConfirmedBugs(unittest.TestCase):
    # @unittest.expectedFailure
    def test_resize_explicit_zero_vertical_delta(self):
        """resize(delta_x, 0) should leave the vertical bounds unchanged."""
        box = BoundingBox.from_rect(0, 0, 10, 10).resize(2, 0)
        self.assertEqual(box.dump_rect(), (-2, 0, 14, 10))

    # @unittest.expectedFailure
    def test_single_vertex_bezier_has_point_bbox(self):
        """A nonempty one-vertex Bezier has a degenerate point bounding box."""
        curve = Bezier()
        curve.move_to((4, 7))
        self.assertEqual(curve.bounding_box(), ((4, 4), (7, 7)))

    @unittest.expectedFailure
    def test_svg_arc_rotation_uses_degrees(self):
        """SVG A-command rotation is specified in degrees, not radians."""
        degrees = PathLC.parse("M0,0 A10,5 90 0 1 20,0").point_at(0.5)
        radians = PathLC.parse(f"M0,0 A10,5 {math.pi / 2} 0 1 20,0").point_at(0.5)
        self.assertAlmostEqual(degrees[0], radians[0], places=6)
        self.assertAlmostEqual(degrees[1], radians[1], places=6)

    # @unittest.expectedFailure
    def test_segment_at_honours_clamp(self):
        path = PathLC.parse("M0,0 L10,0")
        segment, t = path.segment_at(1.5, clamp=True)
        self.assertEqual(t, 1.0)
        self.assertAlmostEqual(segment.point_at(t)[0], 10.0)

    # @unittest.expectedFailure
    def test_segment_at_length_on_move_only_path(self):
        path = PathLC.at((10, 10))
        segment, _start, _length = path.segment_at_length(5.0)
        self.assertIsNone(segment)

    # @unittest.expectedFailure
    def test_point_at_length_on_move_only_path(self):
        path = PathLC.at((10, 10))
        point = path.point_at_length(5.0)
        self.assertIsNotNone(point)
        self.assertAlmostEqual(point[0], 10.0)
        self.assertAlmostEqual(point[1], 10.0)

    # @unittest.expectedFailure
    def test_reverse_single_move_path(self):
        path = PathLC.at((10, 10))
        reversed_path = path.reversed()
        self.assertEqual(reversed_path.describe(), "M10,10")

    # @unittest.expectedFailure
    def test_reverse_preserves_trailing_move_subpath(self):
        path = PathLC.parse("M0,0 L10,0 M99,99")
        reversed_path = path.reversed()
        self.assertIn("99", reversed_path.describe())

    # @unittest.expectedFailure
    def test_reverse_preserves_closed_path_command(self):
        path = PathLC.parse("M0,0 L10,0 L10,10 L0,10 Z")
        reversed_path = path.reversed()
        self.assertTrue(reversed_path.describe().endswith("Z"))
        self.assertAlmostEqual(reversed_path.length(), path.length())


if __name__ == "__main__":
    unittest.main(verbosity=2)
