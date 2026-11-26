import pickle
import unittest

from svggeom.ray import RayIntersection, VecRay
from svggeom.vector import Vector


class TestRayIntersectionResult(unittest.TestCase):
    def test_forward_intersection_is_vector_with_parameters(self):
        a = VecRay((0, 0), (1, 0))
        b = VecRay((5, -2), (0, 1))
        hit = a.intersection_with_ray(b)
        self.assertIsInstance(hit, RayIntersection)
        self.assertIsInstance(hit, Vector)
        self.assertEqual(hit, (5, 0))
        self.assertAlmostEqual(hit.t_self, 5)
        self.assertAlmostEqual(hit.t_other, 2)
        self.assertTrue(hit.is_forward)

    def test_behind_intersection_is_returned_and_flagged(self):
        a = VecRay((0, 0), (1, 0))
        b = VecRay((-5, -1), (0, 1))
        hit = a.intersection_with_ray(b)
        self.assertEqual(hit, (-5, 0))
        self.assertAlmostEqual(hit.t_self, -5)
        self.assertAlmostEqual(hit.t_other, 1)
        self.assertFalse(hit.is_forward)

    def test_intersection_behind_other_ray_is_flagged(self):
        a = VecRay((0, 0), (1, 0))
        b = VecRay((5, 1), (0, 1))
        hit = a.intersection_with_ray(b)
        self.assertEqual(hit, (5, 0))
        self.assertAlmostEqual(hit.t_self, 5)
        self.assertAlmostEqual(hit.t_other, -1)
        self.assertFalse(hit.is_forward)

    def test_origin_intersection_counts_as_forward(self):
        a = VecRay((0, 0), (1, 0))
        b = VecRay((0, -1), (0, 1))
        hit = a.intersection_with_ray(b)
        self.assertEqual(hit, (0, 0))
        self.assertEqual((hit.t_self, hit.t_other), (0, 1))
        self.assertTrue(hit.is_forward)

    def test_parallel_and_coincident_lines_return_none(self):
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


if __name__ == "__main__":
    unittest.main()
