import unittest
from grus_ch04_code import *

# # Vector operations
#
# height_weight_age = [70,   # inches
#                      170,  # pounds
#                      40]   # years
#
# grades = [95,  # exam1
#           80,  # exam2
#           75,  # exam3
#           62]  # exam4
#
# assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]
# assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]
# assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]
# assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]
# assert vector_mean([[1, 2, 3], [3, 4, 1]]) == [2, 3, 2]
# assert dot([1, 2, 3], [4, 5, 6]) == 32
# assert magnitude([3, 4]) == 5
# assert distance([0, 1, 2, 3, 4, 5], [1, 2, 5, 6, 6, 4]) == 5
#
# # Matrix operations
# a = [[1, 2, 3], [4, 5, 6]]    # 2 rows and 3 columns
# b = [[1, 2], [3, 4], [5, 6]]  # 3 rows and 2 columns
# assert shape(a) == (2, 3)
# assert get_column(b, 0) == [1, 3, 5]
# assert get_column(b, 1) == [2, 4, 6]
# assert identity_matrix(5) == [[1, 0, 0, 0, 0],
#                               [0, 1, 0, 0, 0],
#                               [0, 0, 1, 0, 0],
#                               [0, 0, 0, 1, 0],
#                               [0, 0, 0, 0, 1]]


# The following two classes are intended to replace all the asserts above

# This class needs to have a test for each of the vector functions
class TestVectorFunctions(unittest.TestCase):
    def test_add(self):
        """Test add with two vectors."""
        vectors = [[1, 2, 3], [4, 5, 6]]
        result = add(vectors[0], vectors[1])
        self.assertEqual([5, 7, 9], result)

# This class needs to have a test for each of the matrix functions
class TestMatrixFunctions(unittest.TestCase):
    def test_shape(self):
        """Test vector_sum with a list of valid vectors."""
        a = [[1, 2, 3], [4, 5, 6]]    # 2 rows and 3 columns
        assert shape(a) == (2, 3)
        result = shape(a)
        self.assertEqual((2, 3), result)

if __name__ == '__main__':
    unittest.main()
