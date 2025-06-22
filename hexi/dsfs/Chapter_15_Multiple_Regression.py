from typing import List

inputs: List[List[float]] = [[1.,49,4,0],[1,41,9,0],[1,40,8,0],[1,25,6,0],[1,21,1,0],[1,21,0,0],[1,19,3,0],[1,19,0,0],[1,18,9,0],[1,18,8,0],[1,16,4,0],[1,15,3,0],[1,15,0,0],[1,15,2,0],[1,15,7,0],[1,14,0,0],[1,14,1,0],[1,13,1,0],[1,13,7,0],[1,13,4,0],[1,13,2,0],[1,12,5,0],[1,12,0,0],[1,11,9,0],[1,10,9,0],[1,10,1,0],[1,10,1,0],[1,10,7,0],[1,10,9,0],[1,10,1,0],[1,10,6,0],[1,10,6,0],[1,10,8,0],[1,10,10,0],[1,10,6,0],[1,10,0,0],[1,10,5,0],[1,10,3,0],[1,10,4,0],[1,9,9,0],[1,9,9,0],[1,9,0,0],[1,9,0,0],[1,9,6,0],[1,9,10,0],[1,9,8,0],[1,9,5,0],[1,9,2,0],[1,9,9,0],[1,9,10,0],[1,9,7,0],[1,9,2,0],[1,9,0,0],[1,9,4,0],[1,9,6,0],[1,9,4,0],[1,9,7,0],[1,8,3,0],[1,8,2,0],[1,8,4,0],[1,8,9,0],[1,8,2,0],[1,8,3,0],[1,8,5,0],[1,8,8,0],[1,8,0,0],[1,8,9,0],[1,8,10,0],[1,8,5,0],[1,8,5,0],[1,7,5,0],[1,7,5,0],[1,7,0,0],[1,7,2,0],[1,7,8,0],[1,7,10,0],[1,7,5,0],[1,7,3,0],[1,7,3,0],[1,7,6,0],[1,7,7,0],[1,7,7,0],[1,7,9,0],[1,7,3,0],[1,7,8,0],[1,6,4,0],[1,6,6,0],[1,6,4,0],[1,6,9,0],[1,6,0,0],[1,6,1,0],[1,6,4,0],[1,6,1,0],[1,6,0,0],[1,6,7,0],[1,6,0,0],[1,6,8,0],[1,6,4,0],[1,6,2,1],[1,6,1,1],[1,6,3,1],[1,6,6,1],[1,6,4,1],[1,6,4,1],[1,6,1,1],[1,6,3,1],[1,6,4,1],[1,5,1,1],[1,5,9,1],[1,5,4,1],[1,5,6,1],[1,5,4,1],[1,5,4,1],[1,5,10,1],[1,5,5,1],[1,5,2,1],[1,5,4,1],[1,5,4,1],[1,5,9,1],[1,5,3,1],[1,5,10,1],[1,5,2,1],[1,5,2,1],[1,5,9,1],[1,4,8,1],[1,4,6,1],[1,4,0,1],[1,4,10,1],[1,4,5,1],[1,4,10,1],[1,4,9,1],[1,4,1,1],[1,4,4,1],[1,4,4,1],[1,4,0,1],[1,4,3,1],[1,4,1,1],[1,4,3,1],[1,4,2,1],[1,4,4,1],[1,4,4,1],[1,4,8,1],[1,4,2,1],[1,4,4,1],[1,3,2,1],[1,3,6,1],[1,3,4,1],[1,3,7,1],[1,3,4,1],[1,3,1,1],[1,3,10,1],[1,3,3,1],[1,3,4,1],[1,3,7,1],[1,3,5,1],[1,3,6,1],[1,3,1,1],[1,3,6,1],[1,3,10,1],[1,3,2,1],[1,3,4,1],[1,3,2,1],[1,3,1,1],[1,3,5,1],[1,2,4,1],[1,2,2,1],[1,2,8,1],[1,2,3,1],[1,2,1,1],[1,2,9,1],[1,2,10,1],[1,2,9,1],[1,2,4,1],[1,2,5,1],[1,2,0,1],[1,2,9,1],[1,2,9,1],[1,2,0,1],[1,2,1,1],[1,2,1,1],[1,2,4,1],[1,1,0,1],[1,1,2,1],[1,1,2,1],[1,1,5,1],[1,1,3,1],[1,1,10,1],[1,1,6,1],[1,1,0,1],[1,1,8,1],[1,1,6,1],[1,1,4,1],[1,1,9,1],[1,1,9,1],[1,1,4,1],[1,1,2,1],[1,1,9,1],[1,1,0,1],[1,1,8,1],[1,1,6,1],[1,1,1,1],[1,1,1,1],[1,1,5,1]]

from Chapter_4_Linear_Algebra import dot, Vector

def predict(x_p: Vector, beta_p: Vector) -> float:
    return dot(x_p, beta_p)


from typing import List

def error(x_er: Vector, y_er: Vector, beta_er: Vector) -> float:
    return predict(x_er, beta_er) - y_er

def squared_error(x_sqer: Vector, y_sqer: Vector, beta_er: Vector) -> float:
    return error(x_sqer, y_sqer, beta_er) ** 2

x = [1, 2, 3]
y = 30
beta = [4, 4, 4]

assert error(x, y, beta) == -6
assert squared_error(x, y, beta) == 36

def sqerror_gradient(x_sg: Vector, y_sg: Vector, beta_sg: Vector) -> Vector:
    err = error(x_sg, y_sg, beta_sg)
    return [2 * err * x_ for x_ in x_sg]

assert sqerror_gradient(x, y, beta) == [-12, -24, -36]


import random
import tqdm
from Chapter_4_Linear_Algebra import vector_mean
from Chapter_8_Gradient_Descent import inputs


def least_squares_fit(xs: List[Vector], ys: List[float], learning_rate_lsqf: float = 0.001, num_steps: int = 1000, batch_size: int = 1) -> Vector:
    guess = [random.random() for _ in xs[0]]

    for _ in tqdm.trange(num_steps, desc="least squares fit"):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start:start + batch_size]
            batch_ys = ys[start:start + batch_size]

            gradient = vector_mean([sqerror_gradient(x_lsqf, y_lsqf, guess) for x_lsqf, y_lsqf in zip(batch_xs, batch_ys)])
            guess = gradient_step(guess, gradient, -learning_rate_lsqf)

    return guess


from Chapter_5_Statistics import daily_minutes_good
from Chapter_8_Gradient_Descent import gradient_step

random.seed(0)
learning_rate = 0.001
beta_ = least_squares_fit(inputs, daily_minutes_good, learning_rate, 5000, 25)

assert 30.5 < beta[0] < 30.70
assert 0.96 < beta[1] < 1.00
assert -1.89 < beta[2] < -1.85
assert 0.91 < beta[3] < 0.94



# Goodness of Fit
from Chapter_14_Simple_Linear_Regression import total_sum_of_squares

def multiple_r_squared(xs: List[Vector], ys: List[float], beta_m: Vector) -> float:
    sum_of_squared_errors = sum(error(x_mrsq, y_mrsq, beta_m) ** 2
                                for x_mrsq, y_mrsq in zip(xs, ys))
    return 1.0 - sum_of_squared_errors / (total_sum_of_squares(ys))

assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta) < 0.68



# Digression: The Bootstrap

from typing import TypeVar, Callable

X = TypeVar('X')
Stat = TypeVar('Stat')

def bootstrap_sample(data_bootstrap_sample: List[X]) -> List[X]:
    return [random.choice(data_bootstrap_sample) for _ in data_bootstrap_sample]

def bootstrap_statistics(data_bootstrap_statistics: List[X], stats_fn: Callable[[List[X]], Stat], num_samples: int) -> List[Stat]:
    return [stats_fn(bootstrap_sample(data_bootstrap_statistics)) for _ in range(num_samples)]


close_to_100 = [99.5 + random.random() for _ in range(101)]
far_from_100 = ([99.5 + random.random()] + [random.random() for _ in range(50)] + [200 + random.random() for _ in range(50)])

from Chapter_5_Statistics import median, standard_deviation

medians_close = bootstrap_statistics(close_to_100, median, 100)
medians_far = bootstrap_statistics(far_from_100, median, 100)

assert standard_deviation(medians_close) < 1
assert standard_deviation(medians_far) > 90



# Standard Errors of Regression Coefficients

from typing import Tuple

def estimate_sample_beta(pairs: List[Tuple[Vector, float]]):
    x_sample = [x_estimate_sample for x_estimate_sample, _ in pairs]
    y_sample = [y_estimate_sample for _, y_estimate_sample in pairs]
    beta_sample = least_squares_fit(x_sample, y_sample, learning_rate, 5000, 25)
    print("bootstrap sample")
    return beta_sample

random.seed(0)

bootstrap_betas = bootstrap_statistics(list(zip(inputs, daily_minutes_good)), estimate_sample_beta, 100)

bootstrap_standard_errors = [standard_deviation([beta[i] for beta in bootstrap_betas]) for i in range(4)]

print(bootstrap_standard_errors)


from Chapter_6_Probability import normal_cdf

def p_value(beta_hat_j: float, sigma_hat_j: float) -> float:
    if beta_hat_j > 0:
        return 2 * (1 - normal_cdf(beta_hat_j, sigma_hat_j))
    else:
        return 2 * normal_cdf(beta_hat_j, sigma_hat_j)

assert p_value(30.58, 1.27) < 0.001
assert p_value(0.972, 0.103) < 0.001
assert p_value(-1.865, 0.155) < 0.001
assert p_value(0.923, 1.249) < 0.4



# Regularization
"""add to the error term a penalty that gets larger as beta gets larger. Then minimize the combined error and penalty"""

def ridge_penalty(beta_rp: Vector, alpha: float) -> float:
    return alpha * dot(beta_rp[1:], beta_rp[1:])

def squared_error_ridge(x_sqer_ridge: Vector, y_sqer_ridge: float, beta_sqer_ridge: Vector, alpha: float) -> float:
    return error(x_sqer_ridge, y_sqer_ridge, beta_sqer_ridge) ** 2 + ridge_penalty(beta_sqer_ridge, alpha)


from Chapter_4_Linear_Algebra import add

def ridge_penality_gradient(beta_rpg: Vector, alpha: float) -> Vector:
    return [0.] +[2 * alpha * beta_j for beta_j in beta_rpg[1:]]

def sqerror_ridge_gradient(x_sqer_rg, y_sqer_rg: float, beta_sqer_rg: Vector, alpha: float) -> Vector:
    return add(sqerror_gradient(x_sqer_rg, y_sqer_rg, beta_sqer_rg), ridge_penality_gradient(beta_sqer_rg, alpha))

def least_squares_fit_ridge(xs: List[Vector],
                            ys: List[float],
                            alpha: float,
                            learning_rate_ridge: float,
                            num_steps: int,
                            batch_size: int = 1) -> Vector:

    guess = [random.random() for _ in range(xs[0])]

    for i in range(num_steps):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start:start + batch_size]
            batch_ys = ys[start:start + batch_size]
            gradient = vector_mean([sqerror_ridge_gradient(x_lsqf_ridge, y_lsqf_ridge, guess, alpha)
                                    for x_lsqf_ridge, y_lsqf_ridge in zip(batch_xs, batch_ys)])
            guess = gradient_step(guess, gradient, -learning_rate_ridge)

    return guess

random.seed(0)
beta_0 = least_squares_fit_ridge(inputs, daily_minutes_good, 0.0, learning_rate, 5000, 25)

assert 5 < dot(beta_0[1:], beta_0[1:]) < 6
assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta_0) < 0.69


beta_0_1 = least_squares_fit_ridge(inputs, daily_minutes_good, 0.1, learning_rate, 5000, 25)

assert 4 < dot(beta_0_1[1:], beta_0_1[1:]) < 5
assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta_0_1) < 0.69

beta_1 = least_squares_fit_ridge(inputs, daily_minutes_good, 1, learning_rate, 5000, 25)

assert 3 < dot(beta_1[1:], beta_1[1:]) < 4
assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta_1) < 0.69

beta_10 = least_squares_fit_ridge(inputs, daily_minutes_good, 10, learning_rate, 5000, 25)

assert 1 < dot(beta_10[1:], beta_10[1:]) < 2
assert 0.5 < multiple_r_squared(inputs, daily_minutes_good, beta_10) < 0.6


def lasso_penalty(beta_lp, alpha):
    return alpha * sum(abs(beta_i) for beta_i in beta_lp[1:])
