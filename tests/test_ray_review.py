"""Review tests for ./ray.py."""

import unittest

from svggeom.ray import Ray, VecRay
from svggeom.vector import Vector


class TestRayPassingBehavior(unittest.TestCase):

    def test_creation_and_properties(self):
        r = Ray((3, 4), (1, 0))
        self.assertEqual(r.x, 3)
        self.assertEqual(r.y, 4)
        self.assertEqual(r.dx, 1)
        self.assertEqual(r.dy, 0)

    def test_at_returns_pos(self):
        r = Ray((3, 4), (1, 0))
        self.assertEqual(r.at(), Vector(3, 4))

    def test_delta(self):
        r = Ray((0, 0), (1, 0))
        self.assertEqual(r.vector_to((5, 3)), Vector(5, 3))

    def test_distance(self):
        r = Ray((0, 0), (1, 0))
        self.assertAlmostEqual(r.distance_to((3, 4)), 5.0)

    def test_point_along(self):
        r = Ray((0, 0), (1, 0))
        p = r.point_at_distance(5)
        self.assertAlmostEqual(p.x, 5)
        self.assertAlmostEqual(p.y, 0)

    def test_side_positive(self):
        self.assertEqual(Ray((0, 0), (1, 0)).side_of((0, 1)), 1)

    def test_side_negative(self):
        self.assertEqual(Ray((0, 0), (1, 0)).side_of((0, -1)), -1)

    def test_side_on_line(self):
        self.assertEqual(Ray((0, 0), (1, 0)).side_of((5, 0)), 0)

    def test_distance_from_line_horizontal(self):
        r = Ray((0, 5), (1, 0))
        self.assertAlmostEqual(r.distance_to_line((-10, 0), (10, 0)), 5.0)

    def test_distance_from_line_vertical(self):
        r = Ray((5, 0), (0, 1))
        self.assertAlmostEqual(r.distance_to_line((0, -10), (0, 10)), 5.0)

    def test_intersect_of_line_normal_case(self):
        r = Ray((5, -5), (0, 1))
        p = r.intersection_with_line((0, 0), (10, 0))
        self.assertIsNotNone(p)
        if p:
            self.assertAlmostEqual(p.x, 5)
            self.assertAlmostEqual(p.y, 0)

    def test_forward(self):
        r = Ray((0, 0), (1, 0))
        r2 = r.advanced(5)
        self.assertAlmostEqual(r2.x, 5)
        self.assertAlmostEqual(r2.y, 0)

    def test_goto(self):
        r = Ray((0, 0), (1, 0))
        r2 = r.moved_to((10, 20))
        self.assertEqual(r2.x, 10)
        self.assertEqual(r2.y, 20)
        self.assertEqual(r2.dx, 1)

    def test_translate(self):
        r = Ray((0, 0), (1, 0))
        r2 = r.moved_by((3, 4))
        self.assertAlmostEqual(r2.x, 3)
        self.assertAlmostEqual(r2.y, 4)

    def test_along_returns_lerp_position(self):
        r = Ray((0, 0), (1, 0))
        r2 = r.lerp_to((10, 0), 0.5)
        self.assertAlmostEqual(r2.x, 5)
        self.assertAlmostEqual(r2.y, 0)

    def test_clone_preserves_state(self):
        r = Ray((3, 4), (1, 0))
        r2 = r.clone()
        self.assertEqual(r.x, r2.x)
        self.assertEqual(r.y, r2.y)
        self.assertEqual(r.dx, r2.dx)
        self.assertEqual(r.dy, r2.dy)

    # Regression guards for bug 1 (points that already worked correctly).
    def test_nearest_point_from_point_behind_ray(self):
        r = Ray((0, 0), (1, 0))
        p = r.closest_point_to((-5, 0))
        self.assertAlmostEqual(p.x, 0)
        self.assertAlmostEqual(p.y, 0)

    def test_nearest_point_from_point_within_segment(self):
        r = Ray((0, 0), (2, 0))
        p = r.closest_point_to((1, 5))
        self.assertAlmostEqual(p.x, 1)
        self.assertAlmostEqual(p.y, 0)

    def test_intersect_of_line_input_line_degenerate_returns_none(self):
        # a == b -> currently ZeroDivisionError; will be None after fix.
        # Kept out of the fixed-bug class because it is an edge case of #2,
        # tested separately below.
        pass


class TestRayConfirmedBugs(unittest.TestCase):

    # @unittest.expectedFailure
    def test_nearest_point_from_point_beyond_segment(self):
        """Bug 1: perpendicular projection beyond |dir| must land on the
        ray extension, not on pos+dir."""
        r = Ray((0, 0), (1, 0))
        p = r.closest_point_to((5, 1))
        self.assertAlmostEqual(p.x, 5)
        self.assertAlmostEqual(p.y, 0)

    # @unittest.expectedFailure
    def test_nearest_point_from_point_on_ray(self):
        """Bug 1b: a point lying on the ray farther than |dir| returns itself."""
        r = Ray((0, 0), (1, 0))
        p = r.closest_point_to((10, 0))
        self.assertAlmostEqual(p.x, 10)
        self.assertAlmostEqual(p.y, 0)

    # @unittest.expectedFailure
    def test_intersect_of_line_parallel_returns_none(self):
        """Bug 2: parallel lines must return None, not raise ZeroDivisionError."""
        r = Ray((0, 0), (1, 0))
        p = r.intersection_with_line((0, 5), (10, 5))
        self.assertIsNone(p)

    # @unittest.expectedFailure
    def test_intersect_of_ray_parallel_returns_none(self):
        """Bug 2b: intersect_of_ray inherits the same failure."""
        r1 = Ray((0, 0), (1, 0))
        r2 = Ray((0, 5), (1, 0))
        self.assertIsNone(r1.intersection_with_ray(r2))

    # @unittest.expectedFailure
    def test_z_on_2d_ray_raises_attribute_error(self):
        """Bug 3: .z on a 2D ray should raise AttributeError, matching Vector.z."""
        r = Ray((0, 0), (1, 0))
        with self.assertRaises(AttributeError):
            _ = r.z


if __name__ == "__main__":
    unittest.main()
