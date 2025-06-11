from typing import Tuple
import math
import numpy as np
import matplotlib.pyplot as plt
from grus_ch06_code import normal_cdf, inverse_normal_cdf

# PDF p. 124

def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
    """
    Returns mu (the mean) and sigma (the standard deviation) for
    a binomial with n tosses with probability of heads p and tails 1 - p
    """
    mu = n * p
    sigma = math.sqrt(n * p * (1 - p))
    return mu, sigma

def normal_probability_below(hi: float, mu: float = 0, sigma: float = 1)\
        -> float:
    return normal_cdf(hi, mu, sigma)

def normal_probability_above(lo: float,
                             mu: float = 0,
                             sigma: float = 1) -> float:
    """The probability that an N(mu, sigma) is greater than lo."""
    return 1 - normal_cdf(lo, mu, sigma)

def normal_probability_between(lo: float,
                               hi: float,
                               mu: float = 0,
                               sigma: float = 1) -> float:
    """The probability that an N(mu, sigma) is between lo and hi."""
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

def normal_probability_outside(lo: float,
                               hi: float,
                               mu: float = 0,
                               sigma: float = 1) -> float:
    """The probability that an N(mu, sigma is not between lo and hi)."""
    return 1 - normal_probability_between(lo, hi, mu, sigma)

# I don't see much need for this function;
# it is a trivial cover of inverse_normal_cdf.
def normal_upper_bound(probability: float,
                       mu: float = 0,
                       sigma: float = 1) -> float:
    """Returns the z for which P(Z >= z) == probability."""
    return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability: float,
                       mu: float = 0,
                       sigma: float = 1) -> float:
    """Returns the z for which P(Z <= z) == probability."""
    return inverse_normal_cdf(1 - probability, mu, sigma)

# I need to draw this to see what it is doing
def normal_two_sided_bounds(probability: float,
                            mu: float = 0,
                            sigma: float = 1) -> Tuple[float, float]:
    """
    Returns the symmetric (about the mean) bounds that contain the
    specified probability
    """
    tail_probability = (1 - probability) / 2
    # upper bound should have tail_probability above it
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)
    # lower bound should have tail_probability below it
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)
    return lower_bound, upper_bound

# PDF pp. 127-128

def two_sided_p_value(x: float, mu: float = 0, sigma: float = 1) -> float:
    """
    How likely are we to see a value at least as extreme as x (in either
    direction) if our values are from an N(mu, sigma) distribution?
    """
    if x >= mu:
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        return 2 * normal_probability_below(x, mu, sigma)

# PDF p. 133

def estimated_parameters(N: int, n: int) -> Tuple[float, float]:
    p = n / N
    sigma = math.sqrt(p * (1 - p) / N)
    return p, sigma

def a_b_test_statistic(N_a: int, n_a: int, N_b: int, n_b: int) -> float:
    p_a, sigma_a = estimated_parameters(N_a, n_a)
    p_b, sigma_b = estimated_parameters(N_b, n_b)
    return (p_b - p_a) / math.sqrt(sigma_a ** 2 + sigma_b ** 2)

# PDF p. 134

def beta_normalization(a_param: float, b_param: float) -> float:
    return (math.gamma(a_param) * math.gamma(b_param) /
            math.gamma(a_param + b_param))

def beta_pdf(x: float, a_param: float, b_param: float) -> float:
    if x <= 0 or x >= 1:
        return 0
    else:
        return (x ** (a_param - 1) * (1 - x) ** (b_param - 1) /
                beta_normalization(a_param, b_param))
