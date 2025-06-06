import unittest

from grus_ch05_code import *
from grus_ch05_examples import num_friends, daily_minutes, daily_hours

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

    def test_variance(self):
        result = variance(num_friends)
        self.assertAlmostEqual(81.54, result, 2)

    def test_standard_deviation(self):
        result = standard_deviation(num_friends)
        self.assertAlmostEqual(9.03, result, 2)

    def test_interquartile_range(self):
        result = interquartile_range(num_friends)
        self.assertEqual(6, result)

    # PDF p. 100

    def test_covariance_with_daily_minutes(self):
        result = covariance(daily_minutes, num_friends)
        self.assertAlmostEqual(22.43, result, 2)

    def test_covariance_with_daily_hours(self):
        result = covariance(daily_hours, num_friends)
        self.assertAlmostEqual(22.43 / 60, result, 2)

    def test_correlation_with_daily_minutes(self):
        result = correlation(daily_minutes, num_friends)
        self.assertAlmostEqual(0.25, result, 2)

    def test_correlation_with_daily_hours(self):
        result = correlation(daily_hours, num_friends)
        self.assertAlmostEqual(0.25, result, 2)
