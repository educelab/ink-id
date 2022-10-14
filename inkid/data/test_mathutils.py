import unittest

import inkid


class VectorMathTestCase(unittest.TestCase):
    def test_get_basis_from_square(self):
        square_corners = [[-5, 5, 0], [5, 5, 0], [-5, -5, 0], [5, -5, 0]]
        basis_vectors = inkid.data.get_basis_from_square(square_corners)
        self.assertEqual(
            basis_vectors,
            (
                (1.0, 0.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, -1.0)
            )
        )

    def test_get_component_vectors_from_normal_trivial(self):
        normal = (0.0, 0.0, 1.0)
        component_vectors = inkid.data.get_component_vectors_from_normal(normal)
        self.assertEqual(
            component_vectors,
            (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0)
            )
        )

    def test_get_component_vectors_from_normal(self):
        normal = (10.0, 9.0, 8.0)
        component_vectors = inkid.data.get_component_vectors_from_normal(normal)
        self.assertEqual(
            component_vectors,
            (
                (0.7298901944751115, -0.24309882497239957, -0.63887656499994),
                (-0.24309882497239957, 0.7812110575248404, -0.5749889084999459),
                (0.63887656499994, 0.5749889084999459, 0.5111012519999519)
            )
        )


if __name__ == "__main__":
    unittest.main()
