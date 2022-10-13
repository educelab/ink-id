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
                (0.7298901677131653, -0.24309885501861572, -0.6388766169548035),
                (-0.24309885501861572, 0.7812110185623169, -0.5749889612197876),
                (0.6388766169548035, 0.5749889612197876, 0.5111011862754822)
            )
        )


if __name__ == "__main__":
    unittest.main()
