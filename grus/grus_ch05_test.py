import unittest

from grus_ch05_code import *

class TestStatisticsFunctions(unittest.TestCase):

    def test_mean(self):
        some_friends = [100, 49, 41, 40, 25]
        result = mean(some_friends)
        self.assertEqual(51, result)  # add assertion here
