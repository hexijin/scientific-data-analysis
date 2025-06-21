from typing import Tuple

from grus_ch04_code import Vector
from grus_ch05_code import correlation, standard_deviation, mean, de_mean

# PDF p. 240

def predict(alpha: float, beta: float, x_i: float) -> float:
    return beta * x_i + alpha

def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    return predict(alpha, beta, x_i) - y_i

def sum_of_sqerrors(alpha: float, beta: float, xs: Vector, ys: Vector) -> float:
    return sum(error(alpha, beta, x, y) ** 2 for x, y in zip(xs, ys))

def least_squares_fit(xs: Vector, ys: Vector) -> Tuple[float, float]:
    beta = correlation(xs, ys) * standard_deviation(ys) / standard_deviation(xs)
    alpha = mean(ys) - beta * mean(xs)
    return alpha, beta

# PDF p. 243

def demeaned_sum_of_squares(y: Vector) -> float:
    return sum(v ** 2 for v in de_mean(y))

def r_squared(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    sqerrors = sum_of_sqerrors(alpha, beta, x, y)
    tso_squares = demeaned_sum_of_squares(y)
    return 1.0 - sqerrors / tso_squares
