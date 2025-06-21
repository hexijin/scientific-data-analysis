import random
from typing import List, TypeVar, Callable

from grus_ch04_code import dot, Vector, vector_mean, subtract
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

X = TypeVar('X')
Stat = TypeVar('Stat')

def bootstrap_sample(data: List[X]) -> List[X]:
    return [random.choice(data) for _ in data]

def bootstrap_statistic(data: List[X],
                        stats_fn: Callable[[List[X]], Stat],
                        num_samples: int) -> List[Stat]:
    return [stats_fn(bootstrap_sample(data)) for _ in range(num_samples)]

# PDF p. 257 - Regularization

def ridge_penalty(beta: Vector, alpha: float) -> float:
    return alpha * dot(beta[1:], beta[1:])

def squared_error_ridge(x: Vector,
                        y: float,
                        beta: Vector,
                        alpha: float) -> float:
    return error(x, y, beta) ** 2 + ridge_penalty(beta, alpha)

def ridge_penalty_gradient(beta: Vector,
                           alpha: float) -> Vector:
    """gradient of just the ridge penalty"""
    return [0.] + [2 * alpha * beta_j for beta_j in beta[1:]]

def minus_sqerror_ridge_gradient(x: Vector,
                                 y: float,
                                 beta: Vector,
                                 alpha: float) -> Vector:
    return subtract(minus_sqerror_gradient(x, y, beta), ridge_penalty_gradient(beta, alpha))

def least_squares_fit_ridge(xs: List[Vector],
                            ys: List[float],
                            alpha: float,
                            learning_rate: float,
                            num_steps: int,
                            batch_size: int = 1) -> Vector:
    # Start guess with mean
    guess = [random.random() for _ in xs[0]]

    for i in range(num_steps):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start:start+batch_size]
            batch_ys = ys[start:start+batch_size]

            minus_gradient = vector_mean([minus_sqerror_ridge_gradient(x, y, guess, alpha)
                                          for x, y in zip(batch_xs, batch_ys)])
            guess = gradient_step(guess, minus_gradient, learning_rate)

    return guess

def lasso_penalty(beta, alpha):
    return alpha * sum(abs(beta_i) for beta_i in beta[1:])
