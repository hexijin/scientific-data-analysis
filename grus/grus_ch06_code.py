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
