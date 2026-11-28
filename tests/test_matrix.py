import unittest

from svggeom.matrix import Matrix


class TestMatrix(unittest.TestCase):

    def test_default_is_identity(self):
        self.assertTrue(Matrix().is_identity())

    def test_translate(self):
        m = Matrix().add_translate(10, 20)
        self.assertTrue(m.is_translate())
        pt = m.apply_to_point((5, 5))
        self.assertAlmostEqual(pt[0], 15)
        self.assertAlmostEqual(pt[1], 25)

    def test_scale(self):
        pt = Matrix().add_scale(2, 3).apply_to_point((4, 5))
        self.assertAlmostEqual(pt[0], 8)
        self.assertAlmostEqual(pt[1], 15)

    def test_hexad_roundtrip(self):
        for hexad in ((1, 0, 0, 1, 10, 20), (2, 3, 5, 7, 11, 13), (1, 0, 0, 0, 0, 0)):
            with self.subTest(hexad=hexad):
                self.assertEqual(Matrix.hexad(*hexad).to_hexad(), hexad)

    def test_rotate_about_center_fixes_the_center(self):
        center = (10.0, 10.0)
        m = Matrix().add_rotate(90, center_x=center[0], center_y=center[1])
        pt = m.apply_to_point(center)
        self.assertAlmostEqual(pt[0], center[0])
        self.assertAlmostEqual(pt[1], center[1])
