from random import random
from typing import List, Dict
from collections import Counter
import math
import matplotlib.pyplot as plt
from grus_ch04_code import Matrix, Vector, make_matrix
from grus_ch05_code import correlation
from grus_ch06_code import inverse_normal_cdf

# PDF p. 174

def bucketize(point: float, bucket_size: float) -> float:
    """Floor the point to the next lower multiple of bucket_size"""
    return bucket_size * math.floor(point / bucket_size)

def make_histogram(points: List[float], bucket_size: float) -> Dict[float, int]:
    """Buckets the points and counts how many in each bucket"""
    return Counter(bucketize(point, bucket_size) for point in points)

def plot_histogram(points: List[float], bucket_size: float, title: str = "") -> None:
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
    plt.title(title)
    plt.show()

# PDF p. 176

def random_normal() -> float:
    """Returns a random draw from a standard normal distribution"""
    return inverse_normal_cdf(random())

# PDF p. 178

def correlation_matrix(data: List[Vector]) -> Matrix:
    """
    Returns the len(data) x len(data) matrix whose (i, j)-th entry
    is the correlation between data[i] and data[j]
    """
    return make_matrix(len(data), len(data), lambda i, j: correlation(data[i], data[j]))

# PDF. p. 184
