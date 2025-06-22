import unittest

from grus_ch18_code import perceptron_output

# PDF p. 293

class MyTestCase(unittest.TestCase):
    def test_and_perceptron(self):
        and_weights = [2.0, 2.0]
        and_bias = -3.0
        self.assertEqual(1, perceptron_output(and_weights, and_bias, [1, 1]))
        self.assertEqual(0, perceptron_output(and_weights, and_bias, [0, 1]))
        self.assertEqual(0, perceptron_output(and_weights, and_bias, [1, 0]))
        self.assertEqual(0, perceptron_output(and_weights, and_bias, [0, 0]))

    def test_or_perceptron(self):
        or_weights = [2.0, 2.0]
        or_bias = -1.0
        self.assertEqual(1, perceptron_output(or_weights, or_bias, [1, 1]))
        self.assertEqual(1, perceptron_output(or_weights, or_bias, [0, 1]))
        self.assertEqual(1, perceptron_output(or_weights, or_bias, [1, 0]))
        self.assertEqual(0, perceptron_output(or_weights, or_bias, [0, 0]))

    def test_not_perceptron(self):
        not_weights = [-2.0]
        not_bias = 1.0
        self.assertEqual(0, perceptron_output(not_weights, not_bias, [1]))
        self.assertEqual(1, perceptron_output(not_weights, not_bias, [0]))
