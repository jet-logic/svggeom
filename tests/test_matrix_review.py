# test_matrix_review.py
import unittest
from svggeom.matrix import Matrix


class TestMatrixPassing(unittest.TestCase):
    """Passing tests locking in currently working behavior."""

    def test_identity_matrix(self):
        m = Matrix()
        self.assertTrue(m.is_identity())

    def test_translation_matrix(self):
        m = Matrix().add_translate(10, 20)
        self.assertTrue(m.is_translate())
        pt = m.apply_to_point((5, 5))
        self.assertAlmostEqual(pt[0], 15)
        self.assertAlmostEqual(pt[1], 25)

    def test_scale_matrix(self):
        m = Matrix().add_scale(2, 3)
        pt = m.apply_to_point((4, 5))
        self.assertAlmostEqual(pt[0], 8)
        self.assertAlmostEqual(pt[1], 15)

    def test_hexad_roundtrip(self):
        m = Matrix.hexad(1, 0, 0, 1, 10, 20)
        self.assertEqual(m.to_hexad(), (1, 0, 0, 1, 10, 20))


class TestMatrixBugs(unittest.TestCase):
    """Tests asserting desired behavior for confirmed bugs."""

    # @unittest.expectedFailure
    def test_add_rotate_with_center_bug(self):
        """
        Rotating a point that lies precisely at the center of rotation
        should leave that point invariant (unchanged).
        Due to the rotation center bug, rotation happens around (0,0) instead.
        """
        center = (10.0, 10.0)
        m = Matrix().add_rotate(90, center_x=center[0], center_y=center[1])

        # Point at the center of rotation
        pt = m.apply_to_point(center)

        # Desired behavior: point remains (10, 10)
        self.assertAlmostEqual(pt[0], center[0])
        self.assertAlmostEqual(pt[1], center[1])


if __name__ == "__main__":
    unittest.main()
