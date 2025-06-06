from collections import Counter
from typing import List
import math

from grus_ch04_code import sum_of_squares, dot

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)

def _median_odd(xs: List[float]) -> float:
    """If len(xs) is odd, the median is the middle element"""
    sorted_xs = sorted(xs)
    midpoint = len(xs) // 2
    return sorted_xs[midpoint]

def _median_even(xs: List[float]) -> float:
    """If len(xs) is even, it's the average of the middle two elements"""
    sorted_xs = sorted(xs)
    hi_midpoint = len(xs) // 2
    return (sorted_xs[hi_midpoint - 1] + sorted_xs[hi_midpoint]) / 2

def median(xs: List[float]) -> float:
    """Finds the middle-most value of v"""
    return _median_even(xs) if len(xs) % 2 == 0 else _median_odd(xs)

def quantile(xs: List[float], p: float) -> float:
    """Returns the pth-percentile value in x"""
    p_index = int(len(xs) * p)
    return sorted(xs)[p_index]

def mode(xs: List[float]) -> List[float]:
    """Returns a list since there might be more than one mode"""
    counts = Counter(xs)
    max_count = max(counts.values())
    return [x_i for x_i, count_i in counts.items() if count_i == max_count]

def data_range(xs: List[float]) -> float:
    return max(xs) - min(xs)

# PDF p. 98

def de_mean(xs: List[float]) -> List[float]:
    """Translate xs by subtracting its mean"""
    xs_mean = mean(xs)
    return [x - xs_mean for x in xs]

def variance(xs: List[float]) -> float:
    """Almost the average squared deviation from the mean"""
    n = len(xs)
    assert n >= 2
    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n - 1)

def standard_deviation(xs: List[float]) -> float:
    """The standard deviation is the square root of the variance"""
    return math.sqrt(variance(xs))

def interquartile_range(xs: List[float]) -> float:
    """Returns the difference between the 75th and 25th percentiles"""
    return quantile(xs, 0.75) - quantile(xs, 0.25)

def covariance(xs: List[float], ys: List[float]) -> float:
    assert len(xs) == len(ys)
    return dot(de_mean(xs), de_mean(ys)) / (len(xs) - 1)

# PDF p. 100

def correlation(xs: List[float], ys: List[float]) -> float:
    """Measures how much xs and ys vary in tandem about their means"""
    stdev_x = standard_deviation(xs)
    stdev_y = standard_deviation(ys)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(xs, ys) / (stdev_x * stdev_y)
    else:
        return 0
