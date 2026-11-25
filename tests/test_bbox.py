import unittest
import math
from svggeom.bbox import Interval, BoundingBox


class TestBoundingInterval(unittest.TestCase):

    def test_init_empty(self):
        bi = Interval()
        self.assertFalse(bi)
        self.assertEqual(bi.minimum, float("+inf"))
        self.assertEqual(bi.maximum, float("-inf"))

    def test_init_single_value(self):
        bi = Interval(5.0)
        self.assertTrue(bi)
        self.assertEqual(bi.minimum, 5.0)
        self.assertEqual(bi.maximum, 5.0)

    def test_init_two_values(self):
        bi = Interval(2.0, 8.0)
        self.assertEqual(bi.minimum, 2.0)
        self.assertEqual(bi.maximum, 8.0)

        # Test auto-sorting if passed in wrong order
        bi_rev = Interval(10.0, 1.0)
        self.assertEqual(bi_rev.minimum, 1.0)
        self.assertEqual(bi_rev.maximum, 10.0)

    def test_init_tuple(self):
        bi = Interval((3.0, 7.0))
        self.assertEqual(bi.minimum, 3.0)
        self.assertEqual(bi.maximum, 7.0)

    def test_init_copy(self):
        bi1 = Interval(1.0, 5.0)
        bi2 = Interval(bi1)
        self.assertEqual(bi1, bi2)

    def test_bool(self):
        self.assertFalse(Interval())
        self.assertTrue(Interval(0, 10))
        # NaN or Inf should evaluate to False
        self.assertFalse(Interval(float("nan"), 10))

    def test_neg(self):
        bi = Interval(2.0, 5.0)
        neg_bi = -bi
        self.assertEqual(neg_bi.minimum, -5.0)
        self.assertEqual(neg_bi.maximum, -2.0)

    def test_add_union(self):
        bi1 = Interval(1.0, 5.0)
        bi2 = Interval(3.0, 8.0)
        union = bi1 + bi2
        self.assertEqual(union.minimum, 1.0)
        self.assertEqual(union.maximum, 8.0)

    def test_iadd_union(self):
        bi1 = Interval(1.0, 5.0)
        bi2 = Interval(3.0, 8.0)
        bi1 += bi2
        self.assertEqual(bi1.minimum, 1.0)
        self.assertEqual(bi1.maximum, 8.0)

    def test_and_intersection(self):
        bi1 = Interval(1.0, 5.0)
        bi2 = Interval(3.0, 8.0)
        inter = bi1 & bi2
        self.assertEqual(inter.minimum, 3.0)
        self.assertEqual(inter.maximum, 5.0)

        # Non-overlapping intervals should result in an empty/invalid interval
        bi3 = Interval(6.0, 10.0)
        inter_none = bi1 & bi3
        self.assertFalse(inter_none)

    def test_iand_intersection(self):
        bi1 = Interval(1.0, 5.0)
        bi1 &= Interval(3.0, 8.0)
        self.assertEqual(bi1.minimum, 3.0)
        self.assertEqual(bi1.maximum, 5.0)

    def test_mul_scale(self):
        bi = Interval(2.0, 4.0)
        scaled = bi * 2.0
        self.assertEqual(scaled.minimum, 4.0)
        self.assertEqual(scaled.maximum, 8.0)

        # Negative scale should flip min/max
        neg_scaled = bi * -1.0
        self.assertEqual(neg_scaled.minimum, -4.0)
        self.assertEqual(neg_scaled.maximum, -2.0)

    def test_imul_scale(self):
        bi = Interval(2.0, 4.0)
        bi *= 0.5
        self.assertEqual(bi.minimum, 1.0)
        self.assertEqual(bi.maximum, 2.0)

    def test_contains(self):
        bi = Interval(0.0, 10.0)
        self.assertIn(5.0, bi)
        self.assertIn(0.0, bi)
        self.assertNotIn(11.0, bi)

    def test_properties(self):
        bi = Interval(2.0, 8.0)
        self.assertEqual(bi.center, 5.0)
        self.assertEqual(bi.size, 6.0)

    def test_equals_epsilon(self):
        bi1 = Interval(1.0, 5.0)
        bi2 = Interval(1.0 + 1e-13, 5.0 - 1e-13)
        self.assertTrue(bi1.equals(bi2))
        self.assertFalse(bi1.equals(Interval(1.1, 5.0)))


class TestBoundingBox(unittest.TestCase):

    def test_init_empty(self):
        bb = BoundingBox()
        self.assertFalse(bb)

    def test_init_xy(self):
        bb = BoundingBox((0, 10), (5, 15))
        self.assertTrue(bb)
        self.assertEqual(bb.min_x, 0)
        self.assertEqual(bb.max_x, 10)
        self.assertEqual(bb.min_y, 5)
        self.assertEqual(bb.max_y, 15)

    def test_new_xywh(self):
        bb = BoundingBox.from_rect(10, 20, 100, 50)
        self.assertEqual(bb.left, 10)
        self.assertEqual(bb.top, 20)  # Note: top is y.minimum
        self.assertEqual(bb.width, 100)
        self.assertEqual(bb.height, 50)

    def test_properties(self):
        bb = BoundingBox((0, 10), (0, 20))
        self.assertEqual(bb.width, 10)
        self.assertEqual(bb.height, 20)
        self.assertEqual(bb.area, 200)
        self.assertEqual(bb.center, (5.0, 10.0))
        self.assertEqual(bb.size, (10.0, 20.0))
        self.assertAlmostEqual(bb.diagonal_length, math.sqrt(100 + 400))

    def test_add_union(self):
        bb1 = BoundingBox((0, 10), (0, 10))
        bb2 = BoundingBox((5, 15), (5, 15))
        union = bb1 + bb2
        self.assertEqual(union.min_x, 0)
        self.assertEqual(union.max_x, 15)
        self.assertEqual(union.min_y, 0)
        self.assertEqual(union.max_y, 15)

    def test_iadd_union(self):
        bb1 = BoundingBox((0, 10), (0, 10))
        bb1 += BoundingBox((5, 15), (5, 15))
        self.assertEqual(bb1.max_x, 15)
        self.assertEqual(bb1.max_y, 15)

    def test_and_intersection(self):
        bb1 = BoundingBox((0, 10), (0, 10))
        bb2 = BoundingBox((5, 15), (5, 15))
        inter = bb1 & bb2
        self.assertEqual(inter.min_x, 5)
        self.assertEqual(inter.max_x, 10)

        # Non-overlapping
        bb3 = BoundingBox((20, 30), (20, 30))
        self.assertFalse(bb1 & bb3)

    def test_mul_scale(self):
        bb = BoundingBox((0, 10), (0, 10))
        scaled = bb * 2.0
        self.assertEqual(scaled.width, 20)
        self.assertEqual(scaled.height, 20)

    def test_resize(self):
        bb = BoundingBox((10, 20), (10, 20))
        resized = bb.resize(5)
        self.assertEqual(resized.min_x, 5)
        self.assertEqual(resized.max_x, 25)
        self.assertEqual(resized.min_y, 5)
        self.assertEqual(resized.max_y, 25)

        # Asymmetric resize
        resized2 = bb.resize(2, 3)
        self.assertEqual(resized2.min_x, 8)
        self.assertEqual(resized2.max_x, 22)
        self.assertEqual(resized2.min_y, 7)
        self.assertEqual(resized2.max_y, 23)

    def test_transform_translate(self):
        bb = BoundingBox((0, 10), (0, 10))
        # Translate by (5, 5) -> matrix [a, b, c, d, e, f] = [1, 0, 0, 1, 5, 5]
        m = (1, 0, 0, 1, 5, 5)
        t_bb = bb.transform(m)
        self.assertEqual(t_bb.min_x, 5)
        self.assertEqual(t_bb.max_x, 15)
        self.assertEqual(t_bb.min_y, 5)
        self.assertEqual(t_bb.max_y, 15)

    def test_transform_scale(self):
        bb = BoundingBox((0, 10), (0, 10))
        # Scale by 2 -> matrix [2, 0, 0, 2, 0, 0]
        m = (2, 0, 0, 2, 0, 0)
        t_bb = bb.transform(m)
        self.assertEqual(t_bb.width, 20)
        self.assertEqual(t_bb.height, 20)

    def test_transform_rotate(self):
        bb = BoundingBox((0, 10), (0, 0))  # A line on the X axis
        # Rotate 90 degrees -> matrix [0, 1, -1, 0, 0, 0]
        m = (0, 1, -1, 0, 0, 0)
        t_bb = bb.transform(m)
        self.assertAlmostEqual(t_bb.min_x, 0)
        self.assertAlmostEqual(t_bb.max_x, 0)
        self.assertAlmostEqual(t_bb.min_y, 0)
        self.assertAlmostEqual(t_bb.max_y, 10)

    def test_dump_rect(self):
        bb = BoundingBox.from_rect(10, 20, 100, 50)
        x, y, w, h = bb.dump_rect()
        self.assertEqual(x, 10)
        self.assertEqual(y, 20)
        self.assertEqual(w, 100)
        self.assertEqual(h, 50)


if __name__ == "__main__":
    unittest.main()
