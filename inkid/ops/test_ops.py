import unittest

import inkid.ops


class AreCoordinatesWithinTestCase(unittest.TestCase):
    def test_when_within(self):
        self.assertTrue(inkid.ops.are_coordinates_within((0, 0), (10, 10), 11))

    def test_when_not_within(self):
        self.assertFalse(inkid.ops.are_coordinates_within((0, 0), (10, 10), 4))


if __name__ == '__main__':
    unittest.main()
