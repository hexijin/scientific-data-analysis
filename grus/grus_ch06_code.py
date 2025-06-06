import math
import random

SQRT_TWO_PI = math.sqrt(2 * math.pi)

def uniform_pdf(x: float) -> float:
    """A uniform probability distribution function"""
    return 1 if 0 <= x < 1 else 0

def uniform_cdf(x: float) -> float:
    """Returns the probability that a uniform random variable is <= x"""
    if x < 0:
        return 0
    if x < 1:
        return x
    else:
        return 1

def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return math.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (SQRT_TWO_PI * sigma)

def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return (1 + math.erf((x - mu) / (math.sqrt(2) * sigma))) / 2

def inverse_normal_cdf(p: float,
                       mu: float = 0,
                       sigma: float = 1,
                       tolerance: float = 0.00001) -> float:
    """Find appropriate inverse using binary search"""
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)

    low_z = -10.0
    hi_z = 10.0

    mid_z = (low_z + hi_z) / 2

    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2
        mid_p = normal_pdf(mid_z)
        if mid_p < p:
            low_z = mid_z
        else:
            hi_z = mid_z

    return mid_z

def bernoulli_trial(p: float) -> float:
    """Returns 1 with probability p and 0 with probability 1-p"""
    return 1 if random.random() < p else 0

def binomial(n: int, p: float) -> int:
    """Returns the sum of n Bernoulli(p) trials"""
    return sum(bernoulli_trial(p) for _ in range(n))
