from typing import cast

import unittest
import operator

from grus_ch19_code import *

# PDF p. 307

class MyTestCase(unittest.TestCase):
    def test_shape_3(self):
        tensor_3: Tensor = [1, 2, 3]
        self.assertEqual([3], shape(tensor_3))

    def test_shape_3_by_2(self):
        tensor_3_by_2: Tensor = [[1, 2], [3, 4], [5, 6]]
        self.assertEqual([3, 2], shape(tensor_3_by_2))

    def test_bad_shape_2_by_2(self):
        # It is notable that shape() doesn't catch this bad tensor
        bad_tensor: Tensor = [[1.0, 2.0], 3.0]
        self.assertEqual([2, 2], shape(bad_tensor))

    def test_is_1d_true(self):
        self.assertTrue(is_1d([1, 2, 3]))

    def test_is_1d_false(self):
        self.assertFalse(is_1d([[1, 2], [3, 4]]))

    def test_tensor_sum(self):
        self.assertEqual(6, tensor_sum([1, 2, 3]))
        self.assertEqual(10, tensor_sum([[1, 2], [3, 4]]))

    def test_tensor_apply_increment(self):
        expected = [2, 3, 4]
        actual = tensor_apply(lambda x: x + 1, [1, 2, 3])
        self.assertEqual(expected, actual)

    def test_tensor_apply_double(self):
        expected = [[2, 4], [6, 8]]
        actual = tensor_apply(lambda x: 2 * x, [[1, 2], [3, 4]])
        self.assertEqual(expected, actual)

    def test_zeros_like_1d(self):
        self.assertEqual(zeros_like([1, 2, 3]), [0, 0, 0])

    def test_zeros_like_2d(self):
        self.assertEqual(zeros_like([[1, 2, 3], [4, 5, 6]]), [[0, 0, 0], [0, 0, 0]])

    def test_tensor_combine_add1d(self):
        expected = [5, 7, 9]
        casted_add = cast(Callable[[float, float], float], operator.add)
        result = tensor_combine(casted_add, [1, 2, 3], [4, 5, 6])
        self.assertEqual(expected, result)

    def test_tensor_combine_mul2d(self):
        expected = [[5, 12], [21, 32]]
        casted_mul = cast(Callable[[float, float], float], operator.mul)
        result = tensor_combine(casted_mul, [[1, 2], [3, 4]], [[5, 6], [7, 8]])
        self.assertEqual(expected, result)

if __name__ == '__main__':
    unittest.main()
