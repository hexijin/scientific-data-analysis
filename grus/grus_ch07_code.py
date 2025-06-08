from typing import Tuple
import math
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
