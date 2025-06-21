import random
from typing import List

from grus_ch04_code import dot, Vector, vector_mean
from grus_ch08_code import gradient_step
from grus_ch14_code import demeaned_sum_of_squares

# PDF p. 247

def predict(x: Vector, beta: Vector) -> float:
    return dot(x, beta)

# PDF. p. 249

def error(x: Vector, y: float, beta: Vector) -> float:
    return predict(x, beta) - y

def squared_error(x: Vector, y: float, beta: Vector) -> float:
    return error(x, y, beta) ** 2

def minus_sqerror_gradient(x: Vector, y: float, beta: Vector) -> Vector:
    err = error(x, y, beta)
    return [-2.0 * err * x_i for x_i in x]

# PDF p. 250

def least_squares_fit(xs: List[Vector],
                      ys: List[float],
                      learning_rate: float = 0.001,
                      num_steps: int = 1000,
                      batch_size: int = 1) -> Vector:
    guess = [random.random() for _ in xs[0]]
    for _ in range(num_steps):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start:start + batch_size]
            batch_ys = ys[start:start + batch_size]

            minus_gradient = vector_mean([minus_sqerror_gradient(x, y, guess)
                                          for x, y in zip(batch_xs, batch_ys)])
            guess = gradient_step(guess, minus_gradient, learning_rate)
    return guess

def multiple_r_squared(xs: List[Vector], ys: Vector, beta: Vector) -> float:
    sum_of_squared_errors = sum(error(x, y, beta) ** 2 for x, y in zip(xs, ys))
    return 1.0 - sum_of_squared_errors / demeaned_sum_of_squares(ys)

# PDF p. 253 - Digression: The Bootstrap

