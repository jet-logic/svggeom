# test_pathlc_bugs.py
"""Additional review tests for pathlc bugs and locking in working behavior."""

import unittest
from svggeom.pathlc import PathLC
from svggeom.pathlc.command import LineLC, CubicLC
from svggeom.matrix import Matrix


class TestPassingPathLCBehaviors(unittest.TestCase):
    """Lock in behaviors that already work correctly."""

    def test_parse_and_describe_roundtrip_simple(self):
        d = "M0,0 L10,0 L10,10 Z"
        p = PathLC.parse(d)
        # describe may reformat numbers/separators but commands preserved
        self.assertIn("M0,0", p.describe())
        self.assertTrue(p.describe().endswith("Z") or "Z" in p.describe())

    def test_path_length_closed_square(self):
        p = PathLC.parse("M0,0 L10,0 L10,10 L0,10 Z")
        self.assertAlmostEqual(p.length(), 40.0)

    def test_bbox_of_square(self):
        p = PathLC.parse("M0,0 L10,0 L10,10 L0,10 Z")
        b = p.bbox()
        self.assertAlmostEqual(b[0][0], 0)
        self.assertAlmostEqual(b[0][1], 10)
        self.assertAlmostEqual(b[1][0], 0)
        self.assertAlmostEqual(b[1][1], 10)

    def test_point_at_endpoints(self):
        p = PathLC.parse("M0,0 L10,0 L10,10")
        self.assertAlmostEqual(p.point_at(0)[0], 0)
        self.assertAlmostEqual(p.point_at(0)[1], 0)
        self.assertAlmostEqual(p.point_at(1)[0], 10)
        self.assertAlmostEqual(p.point_at(1)[1], 10)

    def test_segment_at_length_mid(self):
        p = PathLC.parse("M0,0 L10,0 L10,10")
        seg, n, N = p.segment_at_length(5.0)
        self.assertIsInstance(seg, LineLC)
        self.assertAlmostEqual(seg.to()[0], 10)
        self.assertAlmostEqual(seg.to()[1], 0)
        self.assertAlmostEqual(n, 5.0)
        self.assertAlmostEqual(N, 10.0)

    def test_reversed_open_path(self):
        p = PathLC.parse("M0,0 L10,0 L20,5")
        r = p.reversed()
        self.assertIsNotNone(r)
        # start of reversed should be end of original
        self.assertAlmostEqual(r.point_at(0)[0], 20)
        self.assertAlmostEqual(r.point_at(0)[1], 5)

    def test_transform_line_translate(self):
        p = PathLC.parse("M0,0 L10,0")
        m = Matrix().add_translate(5, 3)
        tp = p.transform(m.to_hexad())
        self.assertAlmostEqual(tp.point_at(0)[0], 5)
        self.assertAlmostEqual(tp.point_at(0)[1], 3)
        self.assertAlmostEqual(tp.point_at(1)[0], 15)
        self.assertAlmostEqual(tp.point_at(1)[1], 3)

    def test_as_curve_produces_cubics(self):
        p = PathLC.parse("M0,0 L10,0 Q15,5 20,0")
        c = p.as_curve()
        # tail should be CubicLC after conversion
        self.assertIsInstance(c._tail, CubicLC)

    def test_describe_relative_short(self):
        p = PathLC.parse("M0,0 L10,0 L10,10 L0,10 Z")
        d = p.describe(relative=True, short=True)
        self.assertIn("h10", d)
        self.assertIn("v10", d)
        self.assertTrue(d.endswith("z") or "z" in d)


class TestArcTransformDegenerateBug(unittest.TestCase):
    """
    ArcLC.transform crashes with NameError on singular (det=0) matrices
    because get_degen() references undefined rx_, ry_, sweep.
    """

    # @unittest.expectedFailure
    def test_arc_transform_singular_matrix_no_nameerror(self):
        p = PathLC.parse("M0,0 A5,5 0 0 1 10,0")
        # y-scale = 0 -> detT == 0
        m = Matrix.hexad(1, 0, 0, 0, 0, 0)
        # Desired: should not raise; return some degenerate representation
        # (e.g. line or arc with zero radius) without NameError
        tp = p.transform(m.to_hexad())
        self.assertIsNotNone(tp)
        # At minimum the endpoint should be transformed
        end = tp.point_at(1.0)
        self.assertAlmostEqual(end[0], 10.0)  # scale x keeps 10, y*0
        self.assertAlmostEqual(end[1], 0.0)

    # @unittest.expectedFailure
    def test_arc_transform_near_singular_no_nameerror(self):
        p = PathLC.parse("M0,0 A5,5 0 0 1 10,0")
        m = Matrix.hexad(1, 0, 0, 1e-20, 0, 0)
        tp = p.transform(m.to_hexad())
        self.assertIsNotNone(tp)


class TestSplitAtTEqualsOneBug(unittest.TestCase):
    """
    PathLC.split_at(1.0) raises TypeError because it calls
    BaseLC.move_to(seg.to()) unbound (missing self).
    Desired: successful split returning (full path, empty/move-only path).
    """

    # @unittest.expectedFailure
    def test_split_at_one_returns_prefix_and_suffix(self):
        p = PathLC.parse("M0,0 L10,0 L10,10")
        left, right = p.split_at(1.0)
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        # left should be the whole path (length ~20)
        self.assertAlmostEqual(left.length(), 20.0, places=5)
        # right should have zero length (just a move at the end point)
        self.assertAlmostEqual(right.length(), 0.0, places=5)
        self.assertAlmostEqual(right.point_at(0)[0], 10)
        self.assertAlmostEqual(right.point_at(0)[1], 10)

    # @unittest.expectedFailure
    def test_split_at_one_closed_path(self):
        p = PathLC.parse("M0,0 L10,0 L10,10 L0,10 Z")
        left, right = p.split_at(1.0)
        self.assertAlmostEqual(left.length(), 40.0, places=5)
        self.assertAlmostEqual(right.length(), 0.0, places=5)


class TestCutCropSemanticsBug(unittest.TestCase):
    """
    cut_at / crop_at do not implement the documented semantics for
    negative T (should yield suffix) and produce incorrect sub-paths
    for interior crop intervals.
    """

    # @unittest.expectedFailure
    def test_crop_at_interior_produces_correct_length(self):
        # Path of length 100
        p = PathLC.parse("M0,0 L100,0")
        # Desired: crop from 20% to 80% -> length 60, starts near x=20 ends near x=80
        sub = p.crop_at(0.2, 0.8)
        self.assertAlmostEqual(sub.length(), 60.0, places=4)
        start = sub.point_at(0)
        end = sub.point_at(1)
        self.assertAlmostEqual(start[0], 20.0, places=4)
        self.assertAlmostEqual(end[0], 80.0, places=4)

    # @unittest.expectedFailure
    def test_cut_at_negative_returns_suffix_per_docstring(self):
        # Docstring: "If T < 0, it wraps: cut_at(-0.2) == cut_at(0.8) but returns the *suffix*."
        p = PathLC.parse("M0,0 L100,0")
        # cut_at(-0.2) should be the suffix corresponding to the last 20% -> from 80 to 100
        suffix = p.cut_at(-0.2)
        self.assertAlmostEqual(suffix.length(), 20.0, places=4)
        self.assertAlmostEqual(suffix.point_at(0)[0], 80.0, places=4)
        self.assertAlmostEqual(suffix.point_at(1)[0], 100.0, places=4)

    def test_cut_at_positive_prefix_works(self):
        # This already works; lock it in
        p = PathLC.parse("M0,0 L100,0")
        prefix = p.cut_at(0.3)
        self.assertAlmostEqual(prefix.length(), 30.0, places=4)
        self.assertAlmostEqual(prefix.point_at(1)[0], 30.0, places=4)


class TestSegmentAtLengthZero(unittest.TestCase):
    """Confirm current (correct) handling of n=0."""

    def test_segment_at_length_zero_returns_first_segment(self):
        p = PathLC.parse("M0,0 L10,0 L10,10")
        seg, n, N = p.segment_at_length(0.0)
        self.assertIsInstance(seg, LineLC)
        self.assertAlmostEqual(seg.to()[0], 10)
        self.assertAlmostEqual(seg.to()[1], 0)
        self.assertAlmostEqual(n, 0.0)
        self.assertAlmostEqual(N, 10.0)


if __name__ == "__main__":
    unittest.main()
