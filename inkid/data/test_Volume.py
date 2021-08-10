import unittest

import inkid


class VectorMathTestCase(unittest.TestCase):
    def test_get_basis_from_square(self):
        square_corners = [[-5, 5, 0], [5, 5, 0], [-5, -5, 0], [5, -5, 0]]
        basis_vectors = inkid.data.get_basis_from_square(square_corners)
        self.assertEqual(
            basis_vectors,
            {
                'x': {'x': 1.0, 'y': 0.0, 'z': 0.0},
                'y': {'x': 0.0, 'y': -1.0, 'z': 0.0},
                'z': {'x': 0.0, 'y': 0.0, 'z': -1.0}
            }
        )

    def test_get_component_vectors_from_normal_trivial(self):
        normal = {'x': 0, 'y': 0, 'z': 1}
        component_vectors = inkid.data.get_component_vectors_from_normal(normal)
        self.assertEqual(
            component_vectors,
            {
                'x': {'x': 1.0, 'y': 0.0, 'z': 0.0},
                'y': {'x': 0.0, 'y': 1.0, 'z': 0.0},
                'z': {'x': 0.0, 'y': 0.0, 'z': 1.0}
            }
        )

    def test_get_component_vectors_from_normal(self):
        normal = {'x': 10, 'y': 9, 'z': 8}
        component_vectors = inkid.data.get_component_vectors_from_normal(normal)
        self.assertEqual(
            component_vectors,
            {
                'x': {'x': 0.7298901677131653, 'y': -0.24309885501861572, 'z': -0.6388766169548035},
                'y': {'x': -0.24309885501861572, 'y': 0.7812110185623169, 'z': -0.5749889612197876},
                'z': {'x': 0.6388766169548035, 'y': 0.5749889612197876, 'z': 0.5111011862754822}
            }
        )


if __name__ == '__main__':
    unittest.main()
