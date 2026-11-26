import unittest

from svggeom.ray import Ray


class TestRayRotated(unittest.TestCase):
    def assertVectorAlmostEqual(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertAlmostEqual(a, e, places=12)

    def test_numeric_degrees_rotate_counter_clockwise(self):
        ray = Ray((2, 3), (2, 0))
        rotated = ray.rotated(90)
        self.assertEqual(rotated.pos, (2, 3))
        self.assertVectorAlmostEqual(rotated.dir, (0, 2))

    def test_integer_zero_is_identity(self):
        ray = Ray((1, 2), (3, 4))
        self.assertVectorAlmostEqual(ray.rotated(0).dir, ray.dir)

    def test_rotation_vector_encodes_angle(self):
        ray = Ray((0, 0), (1, 0))
        self.assertVectorAlmostEqual(ray.rotated((0, 5)).dir, (0, 1))
        self.assertVectorAlmostEqual(ray.rotated((-1, 0)).dir, (-1, 0))

    def test_original_ray_is_unchanged(self):
        ray = Ray((4, 5), (1, 0))
        ray.rotated(90)
        self.assertEqual(ray.pos, (4, 5))
        self.assertEqual(ray.dir, (1, 0))

    def test_preserves_direction_magnitude_and_subclass(self):
        class MyRay(Ray):
            pass

        ray = MyRay((0, 0), (3, 4))
        result = ray.rotated(90)
        self.assertIsInstance(result, MyRay)
        self.assertAlmostEqual(result.dir.length(), 5)

    def test_rejects_zero_or_non_2d_rotation_vector(self):
        ray = Ray()
        for rotation in ((0, 0), (1, 2, 3)):
            with self.subTest(rotation=rotation):
                with self.assertRaises(ValueError):
                    ray.rotated(rotation)

    def test_rejects_non_2d_ray(self):
        with self.assertRaises(ValueError):
            Ray((0, 0, 0), (1, 0, 0)).rotated(90)


if __name__ == "__main__":
    unittest.main()
