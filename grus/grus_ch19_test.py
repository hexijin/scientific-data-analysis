import unittest

from grus_ch19_code import Tensor, shape

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

if __name__ == '__main__':
    unittest.main()
