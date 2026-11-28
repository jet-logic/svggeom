import unittest
from svggeom.pathlc import PathLC
from svggeom.matrix import Matrix


class TestPathLCIssues(unittest.TestCase):

    def test_empty_move_bbox(self):
        """
        Oddity: Path with only Move returns None for bbox.
        """
        INVALID_BBOX = (
            (float("inf"), float("-inf")),
            (float("inf"), float("-inf")),
        )
        path = PathLC.parse("M10,10")
        bb = path.bbox(default=INVALID_BBOX)
        # Current behavior: None
        # Desired behavior? Arguable. But let's document it.
        self.assertIs(bb, INVALID_BBOX)

    def test_transform_identity(self):
        """
        Passing test: Identity transform should not change coordinates.
        """
        path = PathLC.parse("M10,20 L30,40")
        m = Matrix.hexad(1, 0, 0, 1, 0, 0)  # Identity
        transformed = path.transform(m.to_hexad())
        self.assertEqual(transformed.describe(), path.describe())


if __name__ == "__main__":
    unittest.main()
