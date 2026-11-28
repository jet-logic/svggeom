import math
import pickle
import unittest
from math import pi

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


class TestRayIteration(unittest.TestCase):

    def test_iterating_a_ray_yields_its_origin_coordinates(self):
        self.assertEqual(tuple(Ray((3, 4), (1, 0))), (3, 4))
        self.assertEqual(list(Ray((1, 2, 3), (0, 0, 1))), [1, 2, 3])


class TestRayConstructors(GeomTestCase):

    def test_defaults_are_the_origin_pointing_along_x(self):
        r = Ray()
        self.assertEqual((r.pos, r.dir), (Vector(0, 0), Vector(1, 0)))

    def test_new_ray_ignores_the_current_ray(self):
        r = Ray((9, 9), (0, 5)).new_ray((1, 2), (3, 4))
        self.assertEqual((r.pos, r.dir), (Vector(1, 2), Vector(3, 4)))

    def test_new_dir_keeps_the_origin(self):
        r = Ray((1, 2), (1, 0)).new_dir(Vector(0, 3))
        self.assertEqual((r.pos, r.dir), (Vector(1, 2), Vector(0, 3)))

    def test_new_dir_rejects_the_zero_vector(self):
        with self.assertRaises(ValueError):
            Ray().new_dir(Vector(0, 0))

    def test_new_pos_keeps_the_direction(self):
        r = Ray((1, 2), (3, 4)).new_pos(Vector(7, 8))
        self.assertEqual((r.pos, r.dir), (Vector(7, 8), Vector(3, 4)))


class TestRayOrientation(GeomTestCase):

    def test_pointing_with_an_angle_gives_a_unit_direction(self):
        # The old direction's length (5) is discarded.
        r = Ray((1, 1), (3, 4)).pointing(pi / 2)
        self.assertEqual(r.pos, Vector(1, 1))
        self.assertVecAlmostEqual(r.dir, (0, 1))

    def test_pointing_with_a_vector_uses_it_as_given(self):
        self.assertEqual(Ray().pointing((2, 3)).dir, Vector(2, 3))

    def test_oriented_takes_degrees(self):
        self.assertVecAlmostEqual(Ray((1, 1), (3, 4)).oriented(90).dir, (0, 1))
        self.assertVecAlmostEqual(Ray().oriented(180).dir, (-1, 0))

    def test_towards_preserves_length_by_default(self):
        r = Ray((0, 0), (3, 4)).towards((10, 0))
        self.assertVecAlmostEqual(r.dir, (5, 0))

    def test_towards_magnitude_modes(self):
        ray = Ray((0, 0), (3, 4))
        self.assertVecAlmostEqual(ray.towards((10, 0), "distance").dir, (10, 0))
        self.assertVecAlmostEqual(ray.towards((10, 0), 2.5).dir, (2.5, 0))

    def test_towards_is_measured_from_the_origin(self):
        r = Ray((1, 1), (2, 0)).towards((1, 6))
        self.assertEqual(r.pos, Vector(1, 1))
        self.assertVecAlmostEqual(r.dir, (0, 2))

    def test_towards_rejects_bad_arguments(self):
        ray = Ray((0, 0), (1, 0))
        cases = {
            "target equals origin": ((0, 0), "preserve"),
            "unknown mode": ((5, 5), "huge"),
            "zero magnitude": ((5, 5), 0),
            "negative magnitude": ((5, 5), -1),
        }
        for name, (target, magnitude) in cases.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    ray.towards(target, magnitude)


class TestRayQueries(GeomTestCase):

    def test_negative_distance_goes_backwards(self):
        self.assertVecAlmostEqual(Ray((0, 0), (2, 0)).point_at_distance(-3), (-3, 0))
        self.assertVecAlmostEqual(Ray((0, 0), (0, 2)).advanced(-3).pos, (0, -3))

    def test_direction_length_is_ignored_when_advancing(self):
        self.assertVecAlmostEqual(Ray((0, 0), (10, 0)).advanced(2).pos, (2, 0))
        self.assertVecAlmostEqual(Ray((0, 0), (10, 0)).point_at_distance(2), (2, 0))

    def test_distance_to_a_diagonal_line(self):
        # Distance from (0, 10) to the line y = x is 10 / sqrt(2).
        d = Ray((0, 10), (1, 0)).distance_to_line((0, 0), (10, 10))
        self.assertAlmostEqual(d, 10 / 2**0.5)

    def test_distance_to_an_undefined_line_is_nan(self):
        self.assertTrue(math.isnan(Ray().distance_to_line((1, 1), (1, 1))))

    def test_distance_to_line_is_not_clamped_to_the_segment(self):
        self.assertAlmostEqual(Ray((100, 5), (1, 0)).distance_to_line((0, 0), (10, 0)), 5.0)

    def test_closest_point_on_segment(self):
        r = Ray((3, 5), (1, 0))
        self.assertVecAlmostEqual(r.closest_point_on_segment((0, 0), (10, 0)), (3, 0))
        # Clamped to the nearest end.
        self.assertVecAlmostEqual(r.closest_point_on_segment((5, 0), (10, 0)), (5, 0))

    def test_closest_point_to_with_a_vanishing_direction_is_the_origin(self):
        # |dir|**2 underflows to 0 for a (legal) denormal-scale direction.
        p = Ray((1, 1), (1e-200, 0)).closest_point_to((5, 5))
        self.assertEqual(p, Vector(1, 1))

    def test_intersection_with_a_diagonal_line(self):
        p = Ray((0, 5), (1, 0)).intersection_with_line((0, 0), (10, 10))
        self.assertVecAlmostEqual(p, (5, 5))

    def test_intersection_with_ray_is_2d_only(self):
        r3 = VecRay([0, 0, 0], [1, 0, 0])
        with self.assertRaises(ValueError):
            r3.intersection_with_ray(r3)
        with self.assertRaises(ValueError):
            VecRay([0, 0], [1, 0]).intersection_with_ray(r3)


class TestRayWithOffsetOriginsAndNonUnitDirections(GeomTestCase):
    """Cases where the origin is not (0, 0) and directions are not unit length,
    so that sign or scaling mistakes cannot cancel out."""

    def test_vector_to_and_distance_to_are_relative_to_the_origin(self):
        r = Ray((3, 4), (1, 0))
        self.assertEqual(r.vector_to((5, 9)), Vector(2, 5))
        self.assertAlmostEqual(r.distance_to((6, 8)), 5.0)

    def test_side_of_uses_the_origin(self):
        r = Ray((10, 10), (1, 0))
        self.assertEqual(r.side_of((0, 11)), 1)
        self.assertEqual(r.side_of((20, 9)), -1)
        self.assertEqual(r.side_of((-5, 10)), 0)
        # Direction (0, 1): left of travel is -x.
        self.assertEqual(Ray((10, 10), (0, 1)).side_of((9, 0)), 1)

    def test_distance_to_axis_parallel_lines(self):
        horizontal = Ray((0, 5), (1, 0)).distance_to_line((-10, 3), (10, 3))
        self.assertAlmostEqual(horizontal, 2.0)
        vertical = Ray((5, 0), (0, 1)).distance_to_line((3, -10), (3, 10))
        self.assertAlmostEqual(vertical, 2.0)

    def test_closest_point_to_with_an_offset_origin(self):
        r = Ray((10, 10), (2, 0))
        self.assertVecAlmostEqual(r.closest_point_to((14, 15)), (14, 10))
        self.assertVecAlmostEqual(r.closest_point_to((0, 15)), (10, 10))

    def test_intersection_with_line_not_through_the_origin(self):
        # Ray along y = 2 meets the vertical line x = 4.
        p = Ray((1, 2), (1, 0)).intersection_with_line((4, 0), (4, 10))
        self.assertVecAlmostEqual(p, (4, 2))
        # ... and a line that is offset from the origin diagonally.
        p = Ray((0, 0), (0, 1)).intersection_with_line((2, 3), (4, 7))
        self.assertVecAlmostEqual(p, (0, -1))

    def test_intersection_with_ray_between_offset_rays_with_scaled_directions(self):
        a = VecRay((1, 1), (2, 0))
        b = VecRay((5, -4), (0, 3))
        hit = a.intersection_with_ray(b)
        self.assertEqual(hit, (5, 1))
        self.assertAlmostEqual(hit.t_self, 2)  # 1 + 2*2 = 5
        self.assertAlmostEqual(hit.t_other, 5 / 3)  # -4 + 3*(5/3) = 1
        self.assertTrue(hit.is_forward)

    def test_intersection_parameters_on_offset_rays_behind_the_origin(self):
        a = VecRay((10, 10), (0, 1))
        b = VecRay((20, 0), (-1, 0))
        hit = a.intersection_with_ray(b)
        self.assertEqual(hit, (10, 0))
        self.assertAlmostEqual(hit.t_self, -10)
        self.assertAlmostEqual(hit.t_other, 10)
        self.assertFalse(hit.is_forward)
