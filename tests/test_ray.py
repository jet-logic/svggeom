import pickle
import unittest

from svggeom.ray import Ray, RayIntersection, VecRay
from svggeom.vector import Vector

from .helpers import GeomTestCase


class TestRayBasics(GeomTestCase):

    def test_accessors(self):
        r = Ray((3, 4), (1, 0))
        self.assertEqual((r.x, r.y, r.dx, r.dy), (3, 4, 1, 0))
        self.assertEqual(r.at(), Vector(3, 4))

    def test_accessors_work_for_3d_ray(self):
        r = VecRay([1, 2, 3], [4, 5, 6])
        self.assertEqual((r.x, r.y, r.z), (1, 2, 3))
        self.assertEqual((r.dx, r.dy), (4, 5))

    def test_z_on_2d_ray_raises_attribute_error(self):
        # Matches Vector.z on a 2D vector.
        with self.assertRaises(AttributeError):
            _ = Ray((0, 0), (1, 0)).z

    def test_constructor_rejects_zero_direction(self):
        with self.assertRaises(ValueError):
            VecRay(aim=[0, 0])

    def test_constructor_rejects_mismatched_dimensions(self):
        with self.assertRaises(ValueError):
            VecRay([0, 0], [1, 0, 0])

    def test_vector_to_and_distance_to(self):
        r = Ray((0, 0), (1, 0))
        self.assertEqual(r.vector_to((5, 3)), Vector(5, 3))
        self.assertAlmostEqual(r.distance_to((3, 4)), 5.0)

    def test_point_at_distance(self):
        p = Ray((0, 0), (1, 0)).point_at_distance(5)
        self.assertAlmostEqual(p.x, 5)
        self.assertAlmostEqual(p.y, 0)

    def test_point_at_distance_works_for_3d_ray(self):
        r = VecRay([0, 0, 0], [1, 0, 0])
        self.assertVecAlmostEqual(r.point_at_distance(5), (5, 0, 0))

    def test_side_of(self):
        r = Ray((0, 0), (1, 0))
        self.assertEqual(r.side_of((0, 1)), 1)
        self.assertEqual(r.side_of((0, -1)), -1)
        self.assertEqual(r.side_of((5, 0)), 0)

    def test_side_of_with_large_coordinates(self):
        self.assertEqual(VecRay([1e16, 0], [1, 0]).side_of((1e16, 5)), 1)

    def test_distance_to_line(self):
        self.assertAlmostEqual(Ray((0, 5), (1, 0)).distance_to_line((-10, 0), (10, 0)), 5.0)
        self.assertAlmostEqual(Ray((5, 0), (0, 1)).distance_to_line((0, -10), (0, 10)), 5.0)

    def test_closest_point_to(self):
        cases = {
            "behind the origin": (Ray((0, 0), (1, 0)), (-5, 0), (0, 0)),
            "within |dir|": (Ray((0, 0), (2, 0)), (1, 5), (1, 0)),
            "projects beyond |dir|": (Ray((0, 0), (1, 0)), (5, 1), (5, 0)),
            "on the ray beyond |dir|": (Ray((0, 0), (1, 0)), (10, 0), (10, 0)),
        }
        for name, (ray, point, expected) in cases.items():
            with self.subTest(name):
                p = ray.closest_point_to(point)
                self.assertAlmostEqual(p.x, expected[0])
                self.assertAlmostEqual(p.y, expected[1])


class TestRayBuilders(GeomTestCase):
    """Methods returning a new ray; the original must stay untouched."""

    def test_clone_preserves_state(self):
        r = Ray((3, 4), (1, 0))
        c = r.clone()
        self.assertIsNot(c, r)
        self.assertEqual((c.x, c.y, c.dx, c.dy), (3, 4, 1, 0))

    def test_advanced(self):
        r2 = Ray((0, 0), (1, 0)).advanced(5)
        self.assertAlmostEqual(r2.x, 5)
        self.assertAlmostEqual(r2.y, 0)

    def test_moved_to_keeps_direction(self):
        r2 = Ray((0, 0), (1, 0)).moved_to((10, 20))
        self.assertEqual((r2.x, r2.y, r2.dx), (10, 20, 1))

    def test_moved_by(self):
        r2 = Ray((0, 0), (1, 0)).moved_by((3, 4))
        self.assertAlmostEqual(r2.x, 3)
        self.assertAlmostEqual(r2.y, 4)

    def test_lerp_to(self):
        r2 = Ray((0, 0), (1, 0)).lerp_to((10, 0), 0.5)
        self.assertAlmostEqual(r2.x, 5)
        self.assertAlmostEqual(r2.y, 0)

    def test_builders_preserve_subclass(self):
        class MyRay(Ray):
            pass

        r = MyRay((1, 2), (3, 4))
        builders = {
            "advanced": lambda: r.advanced(1),
            "clone": r.clone,
            "rotated": lambda: r.rotated(90),
            "shifted": lambda: r.shifted((1, 0), (0, 1)),
        }
        for name, build in builders.items():
            with self.subTest(name):
                self.assertIsInstance(build(), MyRay)


class TestRayRotated(GeomTestCase):

    def test_numeric_argument_is_degrees_counter_clockwise(self):
        rotated = Ray((2, 3), (2, 0)).rotated(90)
        self.assertEqual(rotated.pos, (2, 3))
        self.assertVecAlmostEqual(rotated.dir, (0, 2))

    def test_zero_is_identity(self):
        ray = Ray((1, 2), (3, 4))
        self.assertVecAlmostEqual(ray.rotated(0).dir, ray.dir)

    def test_vector_argument_encodes_the_angle(self):
        ray = Ray((0, 0), (1, 0))
        self.assertVecAlmostEqual(ray.rotated((0, 5)).dir, (0, 1))
        self.assertVecAlmostEqual(ray.rotated((-1, 0)).dir, (-1, 0))

    def test_preserves_direction_magnitude(self):
        self.assertAlmostEqual(Ray((0, 0), (3, 4)).rotated(90).dir.length(), 5)

    def test_does_not_mutate_original(self):
        ray = Ray((4, 5), (1, 0))
        ray.rotated(90)
        self.assertEqual(ray.pos, (4, 5))
        self.assertEqual(ray.dir, (1, 0))

    def test_rejects_zero_or_non_2d_rotation_vector(self):
        for rotation in ((0, 0), (1, 2, 3)):
            with self.subTest(rotation=rotation):
                with self.assertRaises(ValueError):
                    Ray().rotated(rotation)

    def test_rejects_non_2d_ray(self):
        with self.assertRaises(ValueError):
            Ray((0, 0, 0), (1, 0, 0)).rotated(90)


class TestRayShifted(unittest.TestCase):

    def test_shifts_position_and_direction(self):
        shifted = Ray((1, 2), (3, 4)).shifted((5, 6), (7, 8))
        self.assertEqual(shifted.pos, Vector(6, 8))
        self.assertEqual(shifted.dir, Vector(10, 12))

    def test_does_not_mutate_original(self):
        ray = Ray((1, 2), (3, 4))
        ray.shifted((5, 6), (7, 8))
        self.assertEqual(ray.pos, Vector(1, 2))
        self.assertEqual(ray.dir, Vector(3, 4))

    def test_supports_3d(self):
        result = Ray((1, 2, 3), (0, 1, 0)).shifted((1, 2, 3), (2, 0, 1))
        self.assertEqual(result.pos, Vector(2, 4, 6))
        self.assertEqual(result.dir, Vector(2, 1, 1))

    def test_rejects_zero_resulting_direction(self):
        with self.assertRaises(ValueError):
            Ray((0, 0), (1, 0)).shifted((0, 0), (-1, 0))


class TestRayIntersection(unittest.TestCase):

    def test_with_line(self):
        p = Ray((5, -5), (0, 1)).intersection_with_line((0, 0), (10, 0))
        self.assertIsNotNone(p)
        self.assertAlmostEqual(p.x, 5)
        self.assertAlmostEqual(p.y, 0)

    def test_with_parallel_line_is_none(self):
        self.assertIsNone(Ray((0, 0), (1, 0)).intersection_with_line((0, 5), (10, 5)))

    def test_with_ray_is_a_vector_carrying_the_parameters(self):
        a = VecRay((0, 0), (1, 0))
        b = VecRay((5, -2), (0, 1))
        hit = a.intersection_with_ray(b)
        self.assertIsInstance(hit, RayIntersection)
        self.assertIsInstance(hit, Vector)
        self.assertEqual(hit, (5, 0))
        self.assertAlmostEqual(hit.t_self, 5)
        self.assertAlmostEqual(hit.t_other, 2)
        self.assertTrue(hit.is_forward)

    def test_hit_behind_self_is_returned_and_flagged(self):
        hit = VecRay((0, 0), (1, 0)).intersection_with_ray(VecRay((-5, -1), (0, 1)))
        self.assertEqual(hit, (-5, 0))
        self.assertAlmostEqual(hit.t_self, -5)
        self.assertAlmostEqual(hit.t_other, 1)
        self.assertFalse(hit.is_forward)

    def test_hit_behind_other_is_returned_and_flagged(self):
        hit = VecRay((0, 0), (1, 0)).intersection_with_ray(VecRay((5, 1), (0, 1)))
        self.assertEqual(hit, (5, 0))
        self.assertAlmostEqual(hit.t_self, 5)
        self.assertAlmostEqual(hit.t_other, -1)
        self.assertFalse(hit.is_forward)

    def test_hit_at_the_origin_counts_as_forward(self):
        hit = VecRay((0, 0), (1, 0)).intersection_with_ray(VecRay((0, -1), (0, 1)))
        self.assertEqual(hit, (0, 0))
        self.assertEqual((hit.t_self, hit.t_other), (0, 1))
        self.assertTrue(hit.is_forward)

    def test_parallel_and_coincident_rays_return_none(self):
        a = VecRay((0, 0), (1, 0))
        self.assertIsNone(a.intersection_with_ray(VecRay((0, 1), (1, 0))))
        self.assertIsNone(a.intersection_with_ray(VecRay((2, 0), (2, 0.0))))

    def test_result_is_immutable_and_pickleable(self):
        hit = VecRay().intersection_with_ray(VecRay((2, -1), (0, 1)))
        with self.assertRaises(AttributeError):
            hit._t_self = 99
        restored = pickle.loads(pickle.dumps(hit))
        self.assertEqual(restored, hit)
        self.assertEqual((restored.t_self, restored.t_other), (2, 1))
        self.assertTrue(restored.is_forward)
