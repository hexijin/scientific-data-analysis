import unittest

from grus_ch14_code import sum_of_sqerrors, demeaned_sum_of_squares

test_alpha = 1.0
test_beta = 2.0
test_xs = [1, 2, 3]
errors = [0.5, -0.5, 1.0]
test_ys = [test_beta * x + test_alpha + error for x, error in zip(test_xs, errors)]

class MyTestCase(unittest.TestCase):
    def test_sum_of_sqerrors(self):
        expected_sum_of_sqerrors = sum(error ** 2 for error in errors)
        result = sum_of_sqerrors(test_alpha, test_beta, test_xs, test_ys)
        self.assertEqual(expected_sum_of_sqerrors, result)

    def test_demeaned_sum_of_squares(self):
        mean_y = sum(test_ys) / len(test_ys)
        expected_total_sum_of_squares = sum((y - mean_y) ** 2 for y in test_ys)
        result = demeaned_sum_of_squares(test_ys)
        self.assertEqual(expected_total_sum_of_squares, result)

if __name__ == '__main__':
    unittest.main()
