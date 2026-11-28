import math
import unittest

from svggeom.bbox import BoundingBox, Interval

INF = float("inf")


class TestInterval(unittest.TestCase):

    def test_init_empty(self):
        bi = Interval()
        self.assertEqual(bi.minimum, INF)
        self.assertEqual(bi.maximum, -INF)

    def test_init_single_value(self):
        bi = Interval(5.0)
        self.assertEqual((bi.minimum, bi.maximum), (5.0, 5.0))

    def test_init_two_values_sorts_them(self):
        self.assertEqual(Interval(2.0, 8.0), (2.0, 8.0))
        self.assertEqual(Interval(10.0, 1.0), (1.0, 10.0))

    def test_init_tuple(self):
        bi = Interval((3.0, 7.0))
        self.assertEqual((bi.minimum, bi.maximum), (3.0, 7.0))

    def test_init_copy(self):
        bi = Interval(1.0, 5.0)
        self.assertEqual(Interval(bi), bi)

    def test_bool(self):
        self.assertFalse(Interval())
        self.assertTrue(Interval(5.0))
        self.assertTrue(Interval(0, 10))
        self.assertFalse(Interval(float("nan"), 10))

    def test_neg(self):
        self.assertEqual(-Interval(2.0, 5.0), (-5.0, -2.0))

    def test_add_is_union(self):
        self.assertEqual(Interval(1.0, 5.0) + Interval(3.0, 8.0), (1.0, 8.0))

    def test_iadd_is_union(self):
        bi = Interval(1.0, 5.0)
        bi += Interval(3.0, 8.0)
        self.assertEqual(bi, (1.0, 8.0))

    def test_and_is_intersection(self):
        self.assertEqual(Interval(1.0, 5.0) & Interval(3.0, 8.0), (3.0, 5.0))

    def test_iand_is_intersection(self):
        bi = Interval(1.0, 5.0)
        bi &= Interval(3.0, 8.0)
        self.assertEqual(bi, (3.0, 5.0))

    def test_empty_intersection_is_the_union_identity(self):
        cases = {
            "disjoint": Interval(1.0, 2.0) & Interval(4.0, 5.0),
            "with empty": Interval() & Interval(1.0, 2.0),
        }
        for name, empty in cases.items():
            with self.subTest(name):
                self.assertFalse(empty)
                self.assertEqual((empty.minimum, empty.maximum), (INF, -INF))
                self.assertEqual(empty + Interval(10.0, 20.0), Interval(10.0, 20.0))

    def test_mul_scales(self):
        bi = Interval(2.0, 4.0)
        self.assertEqual(bi * 2.0, (4.0, 8.0))
        # A negative factor flips minimum and maximum.
        self.assertEqual(bi * -1.0, (-4.0, -2.0))

    def test_imul_scales(self):
        bi = Interval(2.0, 4.0)
        bi *= 0.5
        self.assertEqual(bi, (1.0, 2.0))

    def test_contains(self):
        bi = Interval(0.0, 10.0)
        self.assertIn(5.0, bi)
        self.assertIn(0.0, bi)
        self.assertNotIn(11.0, bi)

    def test_center_and_size(self):
        bi = Interval(2.0, 8.0)
        self.assertEqual(bi.center, 5.0)
        self.assertEqual(bi.size, 6.0)

    def test_equals_uses_epsilon(self):
        bi = Interval(1.0, 5.0)
        self.assertTrue(bi.equals(Interval(1.0 + 1e-13, 5.0 - 1e-13)))
        self.assertFalse(bi.equals(Interval(1.1, 5.0)))


class TestBoundingBox(unittest.TestCase):

    def test_init_empty(self):
        self.assertFalse(BoundingBox())

    def test_init_from_intervals(self):
        bb = BoundingBox((0, 10), (5, 15))
        self.assertTrue(bb)
        self.assertEqual((bb.min_x, bb.max_x, bb.min_y, bb.max_y), (0, 10, 5, 15))

    def test_from_rect(self):
        bb = BoundingBox.from_rect(10, 20, 100, 50)
        # top is y.minimum
        self.assertEqual((bb.left, bb.top, bb.width, bb.height), (10, 20, 100, 50))

    def test_dump_rect_roundtrip(self):
        self.assertEqual(BoundingBox.from_rect(10, 20, 100, 50).dump_rect(), (10, 20, 100, 50))

    def test_derived_properties(self):
        bb = BoundingBox((0, 10), (0, 20))
        self.assertEqual((bb.width, bb.height, bb.area), (10, 20, 200))
        self.assertEqual(bb.center, (5.0, 10.0))
        self.assertEqual(bb.size, (10.0, 20.0))
        self.assertAlmostEqual(bb.diagonal_length, math.sqrt(100 + 400))

    def test_add_is_union(self):
        union = BoundingBox((0, 10), (0, 10)) + BoundingBox((5, 15), (5, 15))
        self.assertEqual((union.min_x, union.max_x), (0, 15))
        self.assertEqual((union.min_y, union.max_y), (0, 15))

    def test_iadd_is_union(self):
        bb = BoundingBox((0, 10), (0, 10))
        bb += BoundingBox((5, 15), (5, 15))
        self.assertEqual((bb.max_x, bb.max_y), (15, 15))

    def test_and_is_intersection(self):
        bb1 = BoundingBox((0, 10), (0, 10))
        inter = bb1 & BoundingBox((5, 15), (5, 15))
        self.assertEqual((inter.min_x, inter.max_x), (5, 10))
        self.assertEqual((inter.min_y, inter.max_y), (5, 10))
        # Non-overlapping boxes intersect to an empty box.
        self.assertFalse(bb1 & BoundingBox((20, 30), (20, 30)))

    def test_mul_scales(self):
        scaled = BoundingBox((0, 10), (0, 10)) * 2.0
        self.assertEqual((scaled.width, scaled.height), (20, 20))

    def test_resize_uniform_and_asymmetric(self):
        bb = BoundingBox((10, 20), (10, 20))
        r = bb.resize(5)
        self.assertEqual((r.min_x, r.max_x, r.min_y, r.max_y), (5, 25, 5, 25))
        r = bb.resize(2, 3)
        self.assertEqual((r.min_x, r.max_x, r.min_y, r.max_y), (8, 22, 7, 23))

    def test_resize_explicit_zero_delta_leaves_that_axis_unchanged(self):
        bb = BoundingBox.from_rect(0, 0, 10, 10)
        self.assertEqual(bb.resize(2, 0).dump_rect(), (-2, 0, 14, 10))
        self.assertEqual(bb.resize(0, 2).dump_rect(), (0, -2, 10, 14))

    def test_transform_translate(self):
        # matrix is the SVG hexad [a, b, c, d, e, f]
        t = BoundingBox((0, 10), (0, 10)).transform((1, 0, 0, 1, 5, 5))
        self.assertEqual((t.min_x, t.max_x, t.min_y, t.max_y), (5, 15, 5, 15))

    def test_transform_scale(self):
        t = BoundingBox((0, 10), (0, 10)).transform((2, 0, 0, 2, 0, 0))
        self.assertEqual((t.width, t.height), (20, 20))

    def test_transform_rotate(self):
        # A segment on the X axis rotated 90 degrees ends up on the Y axis.
        t = BoundingBox((0, 10), (0, 0)).transform((0, 1, -1, 0, 0, 0))
        self.assertAlmostEqual(t.min_x, 0)
        self.assertAlmostEqual(t.max_x, 0)
        self.assertAlmostEqual(t.min_y, 0)
        self.assertAlmostEqual(t.max_y, 10)
