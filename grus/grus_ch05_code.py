from collections import Counter

from grus_ch04_code import *

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
