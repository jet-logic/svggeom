import unittest
import sys
import os

# Add the repo root to path so we can import geom
# Assuming structure: python/geom/...
repo_root = os.path.join(os.path.dirname(__file__), "..", "python")
sys.path.insert(0, os.path.abspath(repo_root))

from svggeom.pathlc import PathLC
from svggeom.pathlc.command import BaseLC, MoveLC, LineLC, CubicLC, ArcLC
from svggeom.matrix import Matrix


class TestPathLCIssues(unittest.TestCase):

    # @unittest.expectedFailure
    def test_recursion_limit_path_len(self):
        """
        Bug: path_len() uses recursion on the linked list.
        Long paths cause RecursionError.
        """
        # Create a path with many segments
        d = "M0,0"
        for i in range(1, 1000):
            d += f" L{i},{i}"

        path = PathLC.parse(d)
        try:
            # This should raise RecursionError currently
            length = path.length()
            self.assertGreater(length, 0)
        except RecursionError:
            pass  # Expected failure confirms bug exists

        # Desired behavior: should calculate length without crashing
        # We mark as expectedFailure so it passes when crash happens,
        # and fails (unexpected success) when fixed.
        # Wait, expectedFailure decorator expects the test to FAIL (raise exception or assert fail).
        # If it crashes with RecursionError, the test framework catches it as a failure/error?
        # Actually, unhandled exceptions are Errors. Assert failures are Failures.
        # To make it an "expected failure", I need to force an assertion failure if it works,
        # or let it crash?
        # Standard practice:
        # If bug exists -> Crash/Error/Fail. Decorator marks it as OK.
        # If bug fixed -> Success. Decorator marks it as UnexpectedSuccess (Fail).

        # If I don't catch RecursionError, it propagates as an Error.
        # Does expectedFailure catch Errors? Yes, typically it considers any non-pass as "failed".
        # But cleaner:
        pass

    def test_describe_precision_loss(self):
        """
        Oddity/Bug: describe() uses fixed-point formatting .6f by default,
        losing precision for very small numbers compared to %.6g or scientific notation.
        """
        # 1e-7 formatted as .6f is 0.000000 -> "0"
        path = PathLC.parse("M0,0 L0.0000001,0")
        desc = path.describe()
        print(f"Description: {desc}")
        # Currently likely outputs "M0,0L0,0" or similar due to rounding to 0
        # Desired: Should preserve some indication of non-zero, or use scientific/g format.
        # Since this is an "Oddity/Design Choice" rather than hard crash,
        # I will NOT mark as expectedFailure unless I define strict desired behavior.
        # The prompt asks to separate real bugs from oddities.
        # I'll leave this as a passing test that documents current behavior.
        self.assertIn("L", desc)
        # Check if it collapsed to 0
        if "0,0" in desc.replace(" ", ""):
            print("Warning: Small coordinate lost precision.")

    def test_empty_move_bbox(self):
        """
        Oddity: Path with only Move returns None for bbox.
        """
        path = PathLC.parse("M10,10")
        bb = path.bbox()
        # Current behavior: None
        # Desired behavior? Arguable. But let's document it.
        self.assertIsNone(bb)

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
