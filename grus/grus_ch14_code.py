from typing import Tuple

from grus_ch04_code import Vector
from grus_ch05_code import correlation, standard_deviation, mean


# PDF p. 240

def predict(alpha: float, beta: float, x_i: float) -> float:
    return alpha * x_i + beta

def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    return predict(alpha, beta, x_i) - y_i

def sum_of_sqerrors(alpha: float, beta: float, xs: Vector, ys: Vector) -> float:
    return sum(error(alpha, beta, x, y) ** 2 for x, y in zip(xs, ys))

def least_squares_fit(xs: Vector, ys: Vector) -> Tuple[float, float]:
    beta = correlation(xs, ys) * standard_deviation(ys) / standard_deviation(xs)
    alpha = mean(ys) - beta * mean(xs)
    return alpha, beta

# PDF p. 243
