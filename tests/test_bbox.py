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

    def test_scalar_multiplication_works_on_both_sides(self):
        bb = BoundingBox((1, 2), (3, 4))
        self.assertEqual(2 * bb, bb * 2)
        self.assertEqual((2 * bb).dump_rect(), (2, 6, 2, 2))

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


class TestIntervalConstruction(unittest.TestCase):

    def test_list_is_accepted_and_sorted(self):
        self.assertEqual(Interval([3, 1]), (1, 3))

    def test_rejects_non_numeric_input(self):
        for args in (("a",), ("a", 1), ((1, 2), 3), ((1, 2, 3),)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    Interval(*args)

    def test_iterates_minimum_then_maximum(self):
        self.assertEqual(list(Interval(8, 2)), [2, 8])
        lo, hi = Interval(8, 2)
        self.assertEqual((lo, hi), (2, 8))

    def test_repr_shows_both_ends(self):
        self.assertEqual(repr(Interval(1.5, 2.5)), "Interval(1.5, 2.5)")


class TestIntervalOperatorEdgeCases(unittest.TestCase):

    def test_equality_with_plain_values(self):
        self.assertEqual(Interval(1, 2), (1, 2))
        self.assertNotEqual(Interval(1, 2), (1, 3))
        # ``==`` (not ``!=``) is what routes through Interval.__eq__.
        self.assertFalse(Interval(1, 2) == "abc")  # not convertible -> unequal
        self.assertFalse(Interval(1, 2) == 5)
        self.assertTrue(Interval(5) == 5)

    def test_reflected_union_and_intersection_with_plain_values(self):
        self.assertEqual((0, 3) + Interval(1, 5), (0, 5))
        self.assertEqual((3, 8) & Interval(1, 5), (3, 5))

    def test_none_is_the_identity_for_union_and_scaling(self):
        bi = Interval(1, 2)
        self.assertEqual(bi + None, bi)
        self.assertEqual(None + bi, bi)
        self.assertEqual(bi * None, bi)

    def test_none_is_the_zero_for_intersection(self):
        # Same as BoundingBox: ``None`` counts as the empty interval.
        bi = Interval(1, 2)
        for name, result in (("right", bi & None), ("left", None & bi)):
            with self.subTest(name):
                self.assertFalse(result)
                self.assertEqual(result, Interval())

    def test_scalar_multiplication_works_on_both_sides(self):
        for result in (Interval(1, 2) * 2, 2 * Interval(1, 2)):
            self.assertIsInstance(result, Interval)
            self.assertEqual(result, (2, 4))
        # A negative factor flips minimum and maximum, on either side.
        self.assertEqual(-1 * Interval(1, 2), (-2, -1))

    def test_inequality_is_consistent_with_equality(self):
        cases = [
            (Interval(5), 5),
            (Interval(1, 2), (1, 2)),
            (Interval(1, 2), (1, 3)),
            (Interval(1, 2), "abc"),
            (Interval(1, 2), Interval(1, 2)),
            (Interval(1, 2), None),
        ]
        for interval, other in cases:
            with self.subTest(interval=interval, other=other):
                self.assertNotEqual(interval == other, interval != other)

    def test_union_with_an_empty_interval_is_the_interval(self):
        self.assertEqual(Interval() + Interval(1, 2), (1, 2))

    def test_intervals_touching_at_a_point_intersect_in_that_point(self):
        self.assertEqual(Interval(1, 2) & Interval(2, 3), (2, 2))

    def test_intervals_are_immutable(self):
        # ``+=`` rebinds the name; other references keep the old value.
        a = Interval(1, 5)
        b = a
        b += Interval(0, 1)
        self.assertEqual(a, (1, 5))
        self.assertEqual(b, (0, 5))
        self.assertIsNot(a & Interval(0, 9), a)

    def test_equals_special_cases(self):
        bi = Interval(1, 5)
        self.assertFalse(bi.equals(Interval()))  # empty never equals
        self.assertTrue(bi.equals(bi))
        self.assertTrue(bi.equals(Interval(1.001, 5), epsilon=0.01))
        self.assertFalse(bi.equals(Interval(1.001, 5), epsilon=1e-6))


class TestBoundingBoxConstruction(unittest.TestCase):

    def test_copy_constructor_is_independent(self):
        original = BoundingBox((0, 10), (0, 10))
        copy = BoundingBox(original)
        copy += BoundingBox((20, 30), (20, 30))
        self.assertEqual(original.max_x, 10)
        self.assertEqual(copy.max_x, 30)

    def test_rejects_a_lone_non_box_argument(self):
        for bad in (5, "x", (0, 1)):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    BoundingBox(bad)

    def test_center_per_axis(self):
        bb = BoundingBox((0, 10), (2, 8))
        self.assertEqual((bb.center_x, bb.center_y), (5, 5))

    def test_edges(self):
        bb = BoundingBox((1, 4), (2, 8))
        self.assertEqual((bb.left, bb.right, bb.top, bb.bottom), (1, 4, 2, 8))
        self.assertEqual((bb.minimum, bb.maximum), ((1, 2), (4, 8)))

    def test_iterates_x_then_y_intervals(self):
        x, y = BoundingBox((0, 10), (5, 15))
        self.assertEqual((x, y), ((0, 10), (5, 15)))

    def test_repr(self):
        text = repr(BoundingBox((0, 10), (5, 15)))
        self.assertTrue(text.startswith("BoundingBox("))
        self.assertIn("10", text)
        self.assertIn("15", text)

    def test_emptiness(self):
        self.assertFalse(BoundingBox())
        self.assertFalse(BoundingBox((0, 10), Interval()))
        self.assertTrue(BoundingBox((0, 0), (0, 0)))  # a point is not empty


class TestBoundingBoxOperators(unittest.TestCase):

    def test_neg_mirrors_through_the_origin(self):
        neg = -BoundingBox((1, 4), (2, 8))
        self.assertEqual((neg.min_x, neg.max_x, neg.min_y, neg.max_y), (-4, -1, -8, -2))

    def test_equality(self):
        bb = BoundingBox((0, 10), (0, 10))
        self.assertEqual(bb, BoundingBox((0, 10), (0, 10)))
        self.assertNotEqual(bb, BoundingBox((0, 10), (0, 11)))
        self.assertNotEqual(bb, ((0, 10), (0, 10)))  # only compares with boxes
        self.assertNotEqual(BoundingBox(), BoundingBox((0, 1), (0, 1)))

    def test_add_with_none_or_empty_is_a_copy(self):
        bb = BoundingBox((0, 10), (0, 10))
        for name, other in (("none", None), ("empty", BoundingBox())):
            with self.subTest(name):
                result = bb + other
                self.assertEqual(result, bb)
                self.assertIsNot(result, bb)

    def test_reflected_add_with_none(self):
        bb = BoundingBox((0, 10), (0, 10))
        self.assertEqual(None + bb, bb)

    def test_sum_accumulates_a_union(self):
        boxes = [BoundingBox((0, 1), (0, 1)), BoundingBox((5, 6), (-2, 0)), BoundingBox((2, 3), (4, 9))]
        total = sum(boxes, BoundingBox())
        self.assertEqual((total.min_x, total.max_x, total.min_y, total.max_y), (0, 6, -2, 9))

    def test_add_does_not_mutate_its_operands(self):
        a = BoundingBox((0, 10), (0, 10))
        b = BoundingBox((5, 15), (5, 15))
        a + b
        self.assertEqual((a.max_x, b.min_x), (10, 5))

    def test_iadd_mutates_in_place_and_returns_self(self):
        a = BoundingBox((0, 10), (0, 10))
        alias = a
        a += BoundingBox((5, 15), (5, 15))
        self.assertIs(a, alias)
        self.assertEqual(alias.max_x, 15)

    def test_and_does_not_mutate_its_operands(self):
        a = BoundingBox((0, 10), (0, 10))
        a & BoundingBox((5, 15), (5, 15))
        self.assertEqual((a.min_x, a.max_x), (0, 10))

    def test_intersection_empty_on_either_axis_empties_both(self):
        # x overlaps (5..10) but y does not.
        inter = BoundingBox((0, 10), (0, 10)) & BoundingBox((5, 15), (20, 30))
        self.assertFalse(inter)
        self.assertFalse(inter.x)
        self.assertFalse(inter.y)

    def test_intersection_with_none_is_empty(self):
        bb = BoundingBox((0, 10), (0, 10))
        self.assertFalse(bb & None)
        self.assertFalse(None & bb)

    def test_imul_scales_in_place(self):
        bb = BoundingBox((1, 2), (3, 4))
        bb *= 2
        self.assertEqual((bb.min_x, bb.max_x, bb.min_y, bb.max_y), (2, 4, 6, 8))

    def test_negative_scale_keeps_min_below_max(self):
        scaled = BoundingBox((1, 2), (3, 4)) * -1
        self.assertEqual((scaled.min_x, scaled.max_x, scaled.min_y, scaled.max_y), (-2, -1, -4, -3))

    def test_mul_does_not_mutate_the_original(self):
        bb = BoundingBox((1, 2), (3, 4))
        bb * 10
        self.assertEqual(bb.max_x, 2)

    def test_resize_can_shrink(self):
        r = BoundingBox((0, 10), (0, 20)).resize(-2)
        self.assertEqual((r.min_x, r.max_x, r.min_y, r.max_y), (2, 8, 2, 18))


class TestBoundingBoxEquals(unittest.TestCase):

    def test_equals_special_cases(self):
        bb = BoundingBox((0, 10), (0, 10))
        self.assertFalse(bb.equals(BoundingBox()))  # an empty box never equals
        self.assertTrue(bb.equals(bb))

    def test_equals_uses_epsilon_on_both_axes(self):
        bb = BoundingBox((0, 10), (0, 10))
        self.assertTrue(bb.equals(BoundingBox((1e-13, 10), (0, 10 - 1e-13))))
        self.assertFalse(bb.equals(BoundingBox((0, 10), (0, 10.1))))
        self.assertFalse(bb.equals(BoundingBox((0.1, 10), (0, 10))))
        self.assertTrue(bb.equals(BoundingBox((0, 10.1), (0, 10)), epsilon=0.5))


class TestBoundingBoxTransformEdgeCases(unittest.TestCase):

    def test_transforming_an_empty_box_gives_an_empty_box(self):
        self.assertFalse(BoundingBox().transform((1, 0, 0, 1, 5, 5)))

    def test_negative_scale_still_yields_ordered_bounds(self):
        t = BoundingBox((1, 3), (2, 4)).transform((-1, 0, 0, -2, 0, 0))
        self.assertEqual((t.min_x, t.max_x, t.min_y, t.max_y), (-3, -1, -8, -4))

    def test_rotation_by_45_degrees_grows_the_box(self):
        c = math.cos(math.pi / 4)
        s = math.sin(math.pi / 4)
        t = BoundingBox((0, 10), (0, 10)).transform((c, s, -s, c, 0, 0))
        self.assertAlmostEqual(t.width, 10 * 2**0.5)
        self.assertAlmostEqual(t.height, 10 * 2**0.5)
        self.assertAlmostEqual(t.min_x, -10 * s)

    def test_transform_does_not_mutate_the_original(self):
        bb = BoundingBox((0, 10), (0, 10))
        bb.transform((2, 0, 0, 2, 5, 5))
        self.assertEqual((bb.min_x, bb.max_x), (0, 10))

    def test_transform_preserves_the_subclass(self):
        class MyBox(BoundingBox):
            pass

        self.assertIsInstance(MyBox((0, 1), (0, 1)).transform((1, 0, 0, 1, 0, 0)), MyBox)


class TestHalfInfiniteIntervals(unittest.TestCase):

    def test_an_interval_with_an_infinite_end_is_falsy(self):
        self.assertFalse(Interval(1, INF))
        self.assertFalse(Interval(-INF, 1))
        self.assertFalse(BoundingBox((0, INF), (0, 1)))
        self.assertFalse(BoundingBox((0, 1), (-INF, 1)))
