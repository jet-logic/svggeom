import importlib
import unittest


class TestPathLerpModule(unittest.TestCase):

    def test_is_importable_as_part_of_svggeom(self):
        module = importlib.import_module("svggeom.help.pathlerp")
        self.assertTrue(callable(module.pathlc_to_beziers_simple))
