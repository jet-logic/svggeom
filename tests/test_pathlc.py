import math
import unittest
from math import isclose, pi

from svggeom.matrix import Matrix
from svggeom.pathlc import PathLC
from svggeom.pathlc.command import CloseLC, CubicLC, LineLC

SQUARE = "M0,0 L10,0 L10,10 L0,10 Z"


class TestParseAndDescribe(unittest.TestCase):

    def test_line(self):
        p = PathLC.parse("M0,0 L10,0")
        self.assertEqual(p.describe(), "M0,0L10,0")
        self.assertAlmostEqual(p.length(), 10.0)

    def test_closed_path_roundtrips(self):
        self.assertEqual(PathLC.parse("M0,0 L10,0 L10,10 Z").describe(), "M0,0L10,0L10,10Z")

    def test_relative_commands_become_absolute(self):
        p = PathLC.parse("M10,10 l5,0 l0,5")
        self.assertEqual(p.describe(), "M10,10L15,10L15,15")
        self.assertAlmostEqual(p.length(), 10.0)

    def test_h_and_v_commands_with_close(self):
        p = PathLC.parse("M0,0 H10 V10 H0 Z")
        self.assertAlmostEqual(p.length(), 40.0)
        self.assertIsInstance(p._tail, CloseLC)

    def test_describe_relative_short(self):
        self.assertEqual(PathLC.parse(SQUARE).describe(relative=True, short=True), "M0,0h10v10h-10z")


class TestMeasure(unittest.TestCase):

    def test_length(self):
        cases = {
            "polyline": ("M0,0 L10,0 L10,10", 20.0),
            "closed square": (SQUARE, 40.0),
            "moves add no length": ("M0,0 L10,0 M20,20 L30,20", 20.0),
        }
        for name, (d, expected) in cases.items():
            with self.subTest(name):
                self.assertAlmostEqual(PathLC.parse(d).length(), expected)

    def test_bbox(self):
        cases = {
            "closed square": (SQUARE, ((0, 10), (0, 10))),
            "closed triangle": ("M0,0 L10,0 L10,10 Z", ((0, 10), (0, 10))),
            "line away from origin": ("M100,100 L110,110", ((100, 110), (100, 110))),
            # A MoveLC has no extent: the bbox must not span the jump to the new subpath.
            "leading moves do not expand it": ("M0,0 M100,100 L110,110", ((100, 110), (100, 110))),
        }
        for name, (d, expected) in cases.items():
            with self.subTest(name):
                self.assertEqual(PathLC.parse(d).bbox(), expected)

    def test_bbox_of_cubic(self):
        # The curve peaks at y = 15 (t = 0.5) for control points at y = 20.
        self.assertEqual(PathLC.parse("M0,0 C0,20 40,20 40,0").bbox(), ((0, 40), (0, 15)))

    def test_bbox_of_move_only_path_falls_back_to_default(self):
        invalid = ((math.inf, -math.inf), (math.inf, -math.inf))
        self.assertIs(PathLC.parse("M10,10").bbox(default=invalid), invalid)


class TestPointsAndSegments(unittest.TestCase):

    def test_point_at_endpoints_and_midpoint(self):
        cases = {
            "line": ("M0,0 L10,0", {0: (0, 0), 0.5: (5, 0), 1: (10, 0)}),
            "offset line": ("M2,3 L12,13", {0.0: (2, 3), 1.0: (12, 13)}),
            "polyline": ("M0,0 L10,0 L10,10", {0: (0, 0), 1: (10, 10)}),
        }
        for name, (d, expected) in cases.items():
            path = PathLC.parse(d)
            for t, (x, y) in expected.items():
                with self.subTest(name, t=t):
                    pt = path.point_at(t)
                    self.assertAlmostEqual(pt[0], x)
                    self.assertAlmostEqual(pt[1], y)

    def test_point_at_on_move_only_path_returns_the_move(self):
        pt = PathLC.at((10, 10)).point_at(0.3)
        self.assertEqual((pt[0], pt[1]), (10.0, 10.0))

    def test_point_at_length(self):
        pt = PathLC.parse("M0,0 L10,0 L10,10").point_at_length(15.0)
        self.assertAlmostEqual(pt[0], 10.0)
        self.assertAlmostEqual(pt[1], 5.0)

    def test_point_at_length_on_move_only_path_returns_the_move(self):
        pt = PathLC.at((10, 10)).point_at_length(5.0)
        self.assertIsNotNone(pt)
        self.assertAlmostEqual(pt[0], 10.0)
        self.assertAlmostEqual(pt[1], 10.0)

    def test_point_at_length_on_zero_length_line(self):
        p = PathLC.parse("M0,0 L0,0")
        self.assertAlmostEqual(p.length(), 0.0)
        pt = p.point_at_length(0)
        self.assertIsNotNone(pt)
        self.assertTrue(isclose(pt[0], 0.0) and isclose(pt[1], 0.0))

    def test_segment_at_length(self):
        p = PathLC.parse("M0,0 L10,0 L10,10")
        for length, expected_n in ((5.0, 5.0), (0.0, 0.0)):
            with self.subTest(length=length):
                seg, n, total = p.segment_at_length(length)
                self.assertIsInstance(seg, LineLC)
                self.assertAlmostEqual(seg.to()[0], 10)
                self.assertAlmostEqual(seg.to()[1], 0)
                self.assertAlmostEqual(n, expected_n)
                self.assertAlmostEqual(total, 10.0)

    def test_segment_at_length_on_move_only_path_has_no_segment(self):
        seg, _n, _total = PathLC.at((10, 10)).segment_at_length(5.0)
        self.assertIsNone(seg)

    def test_segment_at_wraps_without_clamp(self):
        _seg, t = PathLC.parse("M0,0 L10,0").segment_at(1.5)
        self.assertAlmostEqual(t, 0.5)

    def test_segment_at_honours_clamp(self):
        seg, t = PathLC.parse("M0,0 L10,0").segment_at(1.5, clamp=True)
        self.assertEqual(t, 1.0)
        self.assertAlmostEqual(seg.point_at(t)[0], 10.0)


class TestSplitCutCrop(unittest.TestCase):

    def test_split_at_a_segment_boundary(self):
        left, right = PathLC.parse("M0,0 L10,0 L20,0").split_at(0.5)
        self.assertEqual(left.describe(), "M0,0L10,0")
        self.assertEqual(right.describe(), "M10,0L20,0")

    def test_split_inside_a_segment(self):
        left, right = PathLC.parse("M0,0 L40,0").split_at(0.5)
        self.assertAlmostEqual(left.length(), 20.0)
        self.assertAlmostEqual(right.length(), 20.0)
        self.assertAlmostEqual(left.point_at(1)[0], 20.0)
        self.assertAlmostEqual(right.point_at(0)[0], 20.0)

    def test_split_at_one_gives_whole_path_and_an_empty_move(self):
        cases = {
            "open": ("M0,0 L10,0 L10,10", 20.0, (10, 10)),
            "closed": (SQUARE, 40.0, None),
        }
        for name, (d, length, end) in cases.items():
            with self.subTest(name):
                left, right = PathLC.parse(d).split_at(1.0)
                self.assertAlmostEqual(left.length(), length, places=5)
                self.assertAlmostEqual(right.length(), 0.0, places=5)
                if end:
                    self.assertAlmostEqual(right.point_at(0)[0], end[0])
                    self.assertAlmostEqual(right.point_at(0)[1], end[1])

    def test_cut_at_positive_returns_prefix(self):
        prefix = PathLC.parse("M0,0 L100,0").cut_at(0.3)
        self.assertAlmostEqual(prefix.length(), 30.0, places=4)
        self.assertAlmostEqual(prefix.point_at(1)[0], 30.0, places=4)

    def test_cut_at_one_returns_whole_path(self):
        self.assertAlmostEqual(PathLC.parse("M0,0 L10,0 L20,0").cut_at(1.0).length(), 20.0)

    def test_cut_at_negative_returns_suffix(self):
        # cut_at(-0.2) wraps like cut_at(0.8) but keeps the *suffix*.
        suffix = PathLC.parse("M0,0 L100,0").cut_at(-0.2)
        self.assertAlmostEqual(suffix.length(), 20.0, places=4)
        self.assertAlmostEqual(suffix.point_at(0)[0], 80.0, places=4)
        self.assertAlmostEqual(suffix.point_at(1)[0], 100.0, places=4)

    def test_crop_at_interior_interval(self):
        sub = PathLC.parse("M0,0 L100,0").crop_at(0.2, 0.8)
        self.assertAlmostEqual(sub.length(), 60.0, places=4)
        self.assertAlmostEqual(sub.point_at(0)[0], 20.0, places=4)
        self.assertAlmostEqual(sub.point_at(1)[0], 80.0, places=4)


class TestReverse(unittest.TestCase):

    def test_line(self):
        self.assertEqual(PathLC.parse("M0,0 L10,0").reversed().describe(), "M10,0L0,0")

    def test_open_polyline_swaps_endpoints_and_keeps_length(self):
        r = PathLC.parse("M0,0 L10,0 L10,10").reversed()
        self.assertEqual(r.describe(), "M10,10L10,0L0,0")
        self.assertAlmostEqual(r.length(), 20.0)

    def test_cubic_swaps_endpoints(self):
        r = PathLC.parse("M0,0 C0,10 10,10 10,0").reversed()
        self.assertAlmostEqual(r.point_at(0.0)[0], 10.0)
        self.assertAlmostEqual(r.point_at(0.0)[1], 0.0)
        self.assertAlmostEqual(r.point_at(1.0)[0], 0.0)
        self.assertAlmostEqual(r.point_at(1.0)[1], 0.0)

    def test_move_only_path(self):
        for name, path in (("at", PathLC.at((10, 10))), ("parsed", PathLC.parse("M10,10"))):
            with self.subTest(name):
                r = path.reversed()
                self.assertEqual(r.describe(), "M10,10")
                self.assertAlmostEqual(r.length(), 0.0)

    def test_trailing_move_subpath_is_kept(self):
        r = PathLC.parse("M0,0 L10,0 M99,99").reversed()
        self.assertIn("99", r.describe())

    def test_closed_path_stays_closed(self):
        p = PathLC.parse(SQUARE)
        self.assertIsInstance(p._tail, CloseLC)
        r = p.reversed()
        self.assertIsInstance(r._tail, CloseLC)
        self.assertTrue(r.describe().endswith("Z"))
        self.assertAlmostEqual(r.length(), p.length())
        # The reversed path starts where the original ended, i.e. back at (0, 0).
        self.assertAlmostEqual(r.point_at(0)[0], 0.0)
        self.assertAlmostEqual(r.point_at(0)[1], 0.0)


class TestTransform(unittest.TestCase):

    def test_identity_keeps_the_path(self):
        path = PathLC.parse("M10,20 L30,40")
        self.assertEqual(path.transform(Matrix().to_hexad()).describe(), path.describe())

    def test_translate(self):
        tp = PathLC.parse("M0,0 L10,0").transform(Matrix().add_translate(5, 3).to_hexad())
        self.assertAlmostEqual(tp.point_at(0)[0], 5)
        self.assertAlmostEqual(tp.point_at(0)[1], 3)
        self.assertAlmostEqual(tp.point_at(1)[0], 15)
        self.assertAlmostEqual(tp.point_at(1)[1], 3)

    def test_translate_then_scale(self):
        m = Matrix().add_translate(5, 5).add_scale(2)
        # The order in which the two compose is an implementation detail of Matrix.
        self.assertTrue(PathLC.parse("M0,0 L10,0").transform(m.to_hexad()).bbox())

    def test_arc_survives_degenerate_matrices(self):
        # Each matrix is singular (or nearly so); the arc must degrade, not raise.
        cases = {
            "y collapsed": ("M0,0 A5,5 0 0 1 10,0", (1, 0, 0, 0, 0, 0)),
            "almost collapsed": ("M0,0 A5,5 0 0 1 10,0", (1, 0, 0, 1e-20, 0, 0)),
            "all zero": ("M0,0 A10,10 0 0 0 10,10", (0, 0, 0, 0, 0, 0)),
        }
        for name, (d, hexad) in cases.items():
            with self.subTest(name):
                self.assertIsNotNone(PathLC.parse(d).transform(Matrix.hexad(*hexad).to_hexad()))

    def test_arc_endpoint_is_transformed_by_a_singular_matrix(self):
        m = Matrix.hexad(1, 0, 0, 0, 0, 0)
        tp = PathLC.parse("M0,0 A5,5 0 0 1 10,0").transform(m.to_hexad())
        end = tp.point_at(1.0)
        self.assertAlmostEqual(end[0], 10.0)  # x scale 1 keeps 10, y * 0 -> 0
        self.assertAlmostEqual(end[1], 0.0)

    def test_as_curve_converts_to_cubics(self):
        for d in ("M0,0 L10,10", "M0,0 L10,0 Q15,5 20,0"):
            with self.subTest(d):
                self.assertIsInstance(PathLC.parse(d).as_curve()._tail, CubicLC)

    def test_as_curve_preserves_line_length(self):
        p = PathLC.parse("M0,0 L10,10")
        self.assertAlmostEqual(p.as_curve().length(), p.length(), places=3)


class TestArcs(unittest.TestCase):

    def test_unrotated_midpoint(self):
        # Circle of radius 10 from (0,0) to (20,0): the midpoint is at x = 10.
        mid = PathLC.parse("M0,0 A 10 10 0 0 1 20,0").point_at(0.5)
        self.assertAlmostEqual(mid[0], 10.0)
        self.assertAlmostEqual(abs(mid[1]), 10.0)

    def test_small_and_large_arc_flags(self):
        small = PathLC.parse("M0,0 A10,10 0 0 1 10,0")
        large = PathLC.parse("M0,0 A10,10 0 1 1 10,0")
        self.assertAlmostEqual(small.length(), 10 * (pi / 3), places=3)
        self.assertAlmostEqual(large.length(), 10 * (5 * pi / 3), places=3)

    def test_rotation_is_in_degrees(self):
        d = "M0,0 A10,5 {} 0 1 20,0"
        ninety = PathLC.parse(d.format(90)).point_at(0.5)
        # 450 degrees is the same turn plus a full revolution.
        full_turn_later = PathLC.parse(d.format(450)).point_at(0.5)
        self.assertAlmostEqual(ninety[0], full_turn_later[0], places=6)
        self.assertAlmostEqual(ninety[1], full_turn_later[1], places=6)
        # pi/2 is only ~1.57 degrees, so it must not match a 90 degree rotation.
        tiny = PathLC.parse(d.format(pi / 2)).point_at(0.5)
        self.assertGreater(abs(ninety[1] - tiny[1]), 1.0)

    def test_arc_with_identical_endpoints_is_omitted(self):
        p = PathLC.parse("M5,5 A10,10 0 0 1 5,5")
        self.assertAlmostEqual(p.length(), 0.0)
        self.assertIsNotNone(p.point_at(0))

    def test_arc_with_identical_endpoints_mid_path_is_omitted(self):
        p = PathLC.parse("M0,0 L10,0 A5,5 0 0 1 10,0 L20,0")  # == M0,0 L10,0 L20,0
        self.assertAlmostEqual(p.length(), 20.0)
        self.assertAlmostEqual(p.point_at(1)[0], 20.0)
