import unittest

from svggeom.ray import Ray
from svggeom.vector import Vector


class TestRayShifted(unittest.TestCase):
    def test_shifts_position_and_direction(self):
        ray = Ray((1, 2), (3, 4))
        shifted = ray.shifted((5, 6), (7, 8))
        self.assertEqual(shifted.pos, Vector(6, 8))
        self.assertEqual(shifted.dir, Vector(10, 12))

    def test_does_not_mutate_original(self):
        ray = Ray((1, 2), (3, 4))
        ray.shifted((5, 6), (7, 8))
        self.assertEqual(ray.pos, Vector(1, 2))
        self.assertEqual(ray.dir, Vector(3, 4))

    def test_preserves_subclass(self):
        class MyRay(Ray):
            pass

        result = MyRay((1, 2), (3, 4)).shifted((1, 0), (0, 1))
        self.assertIsInstance(result, MyRay)

    def test_supports_3d(self):
        result = Ray((1, 2, 3), (0, 1, 0)).shifted((1, 2, 3), (2, 0, 1))
        self.assertEqual(result.pos, Vector(2, 4, 6))
        self.assertEqual(result.dir, Vector(2, 1, 1))

    def test_rejects_zero_resulting_direction(self):
        with self.assertRaises(ValueError):
            Ray((0, 0), (1, 0)).shifted((0, 0), (-1, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
