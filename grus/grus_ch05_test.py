import unittest

from grus_ch05_code import *
from grus_ch05_examples import num_friends

class TestStatisticsFunctions(unittest.TestCase):

    def test_mean(self):
        result = mean(num_friends)
        self.assertAlmostEqual(7.3333, result, 4)

    def test_median_odd(self):
        result = median([1, 10, 2, 9, 5])
        self.assertEqual(5, result)

    def test_median_even(self):
        result = median([1, 9, 2, 10])
        self.assertEqual(5.5, result)

    def test_quantile_ten_percent(self):
        result = quantile(num_friends, 0.10)
        self.assertEqual(1, result)

    def test_quantile_twenty_five_percent(self):
        result = quantile(num_friends, 0.25)
        self.assertEqual(3, result)

    def test_quantile_seventy_five_percent(self):
        result = quantile(num_friends, 0.75)
        self.assertEqual(9, result)

    def test_quantile_ninety_percent(self):
        result = quantile(num_friends, 0.90)
        self.assertEqual(13, result)

    def test_mode(self):
        result = mode(num_friends)
        self.assertEqual({1, 6}, set(result))

    def test_data_range(self):
        result = data_range(num_friends)
        self.assertEqual(99, result)

# PDF p. 98
