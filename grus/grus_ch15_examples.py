import random
from typing import List, Tuple

from grus_ch04_code import Vector
from grus_ch05_code import median, standard_deviation
from grus_ch06_code import normal_cdf
from grus_ch15_code import error, squared_error, least_squares_fit, multiple_r_squared, bootstrap_statistic

from grus_ch05_examples import daily_minutes_good

# PDF p. 249

test_x = [1, 2, 3]
test_y = 30
test_beta = [4, 4, 4]

assert error(test_x, test_y, test_beta) == -6
assert squared_error(test_x, test_y, test_beta) == 36

# PDF p. 250

random.seed(0)

learning_rate = 0.001

# This is what Grus is using for his inputs:

inputs: List[List[float]] = [[1., 49, 4, 0], [1, 41, 9, 0], [1, 40, 8, 0], [1, 25, 6, 0], [1, 21, 1, 0], [1, 21, 0, 0],
                             [1, 19, 3, 0], [1, 19, 0, 0], [1, 18, 9, 0], [1, 18, 8, 0], [1, 16, 4, 0], [1, 15, 3, 0],
                             [1, 15, 0, 0], [1, 15, 2, 0], [1, 15, 7, 0], [1, 14, 0, 0], [1, 14, 1, 0], [1, 13, 1, 0],
                             [1, 13, 7, 0], [1, 13, 4, 0], [1, 13, 2, 0], [1, 12, 5, 0], [1, 12, 0, 0], [1, 11, 9, 0],
                             [1, 10, 9, 0], [1, 10, 1, 0], [1, 10, 1, 0], [1, 10, 7, 0], [1, 10, 9, 0], [1, 10, 1, 0],
                             [1, 10, 6, 0], [1, 10, 6, 0], [1, 10, 8, 0], [1, 10, 10, 0], [1, 10, 6, 0], [1, 10, 0, 0],
                             [1, 10, 5, 0], [1, 10, 3, 0], [1, 10, 4, 0], [1, 9, 9, 0], [1, 9, 9, 0], [1, 9, 0, 0],
                             [1, 9, 0, 0], [1, 9, 6, 0], [1, 9, 10, 0], [1, 9, 8, 0], [1, 9, 5, 0], [1, 9, 2, 0],
                             [1, 9, 9, 0], [1, 9, 10, 0], [1, 9, 7, 0], [1, 9, 2, 0], [1, 9, 0, 0], [1, 9, 4, 0],
                             [1, 9, 6, 0], [1, 9, 4, 0], [1, 9, 7, 0], [1, 8, 3, 0], [1, 8, 2, 0], [1, 8, 4, 0],
                             [1, 8, 9, 0], [1, 8, 2, 0], [1, 8, 3, 0], [1, 8, 5, 0], [1, 8, 8, 0], [1, 8, 0, 0],
                             [1, 8, 9, 0], [1, 8, 10, 0], [1, 8, 5, 0], [1, 8, 5, 0], [1, 7, 5, 0], [1, 7, 5, 0],
                             [1, 7, 0, 0], [1, 7, 2, 0], [1, 7, 8, 0], [1, 7, 10, 0], [1, 7, 5, 0], [1, 7, 3, 0],
                             [1, 7, 3, 0], [1, 7, 6, 0], [1, 7, 7, 0], [1, 7, 7, 0], [1, 7, 9, 0], [1, 7, 3, 0],
                             [1, 7, 8, 0], [1, 6, 4, 0], [1, 6, 6, 0], [1, 6, 4, 0], [1, 6, 9, 0], [1, 6, 0, 0],
                             [1, 6, 1, 0], [1, 6, 4, 0], [1, 6, 1, 0], [1, 6, 0, 0], [1, 6, 7, 0], [1, 6, 0, 0],
                             [1, 6, 8, 0], [1, 6, 4, 0], [1, 6, 2, 1], [1, 6, 1, 1], [1, 6, 3, 1], [1, 6, 6, 1],
                             [1, 6, 4, 1], [1, 6, 4, 1], [1, 6, 1, 1], [1, 6, 3, 1], [1, 6, 4, 1], [1, 5, 1, 1],
                             [1, 5, 9, 1], [1, 5, 4, 1], [1, 5, 6, 1], [1, 5, 4, 1], [1, 5, 4, 1], [1, 5, 10, 1],
                             [1, 5, 5, 1], [1, 5, 2, 1], [1, 5, 4, 1], [1, 5, 4, 1], [1, 5, 9, 1], [1, 5, 3, 1],
                             [1, 5, 10, 1], [1, 5, 2, 1], [1, 5, 2, 1], [1, 5, 9, 1], [1, 4, 8, 1], [1, 4, 6, 1],
                             [1, 4, 0, 1], [1, 4, 10, 1], [1, 4, 5, 1], [1, 4, 10, 1], [1, 4, 9, 1], [1, 4, 1, 1],
                             [1, 4, 4, 1], [1, 4, 4, 1], [1, 4, 0, 1], [1, 4, 3, 1], [1, 4, 1, 1], [1, 4, 3, 1],
                             [1, 4, 2, 1], [1, 4, 4, 1], [1, 4, 4, 1], [1, 4, 8, 1], [1, 4, 2, 1], [1, 4, 4, 1],
                             [1, 3, 2, 1], [1, 3, 6, 1], [1, 3, 4, 1], [1, 3, 7, 1], [1, 3, 4, 1], [1, 3, 1, 1],
                             [1, 3, 10, 1], [1, 3, 3, 1], [1, 3, 4, 1], [1, 3, 7, 1], [1, 3, 5, 1], [1, 3, 6, 1],
                             [1, 3, 1, 1], [1, 3, 6, 1], [1, 3, 10, 1], [1, 3, 2, 1], [1, 3, 4, 1], [1, 3, 2, 1],
                             [1, 3, 1, 1], [1, 3, 5, 1], [1, 2, 4, 1], [1, 2, 2, 1], [1, 2, 8, 1], [1, 2, 3, 1],
                             [1, 2, 1, 1], [1, 2, 9, 1], [1, 2, 10, 1], [1, 2, 9, 1], [1, 2, 4, 1], [1, 2, 5, 1],
                             [1, 2, 0, 1], [1, 2, 9, 1], [1, 2, 9, 1], [1, 2, 0, 1], [1, 2, 1, 1], [1, 2, 1, 1],
                             [1, 2, 4, 1], [1, 1, 0, 1], [1, 1, 2, 1], [1, 1, 2, 1], [1, 1, 5, 1], [1, 1, 3, 1],
                             [1, 1, 10, 1], [1, 1, 6, 1], [1, 1, 0, 1], [1, 1, 8, 1], [1, 1, 6, 1], [1, 1, 4, 1],
                             [1, 1, 9, 1], [1, 1, 9, 1], [1, 1, 4, 1], [1, 1, 2, 1], [1, 1, 9, 1], [1, 1, 0, 1],
                             [1, 1, 8, 1], [1, 1, 6, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 5, 1]]

test_beta = least_squares_fit(inputs, daily_minutes_good, learning_rate, 5000, 25)

assert 30.50 < test_beta[0] < 30.70  # constant
assert 0.96 < test_beta[1] < 1.00    # num friends
assert -1.89 < test_beta[2] < -1.85  # work hours per day
assert 0.91 < test_beta[3] < 0.94    # has PhD

assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, test_beta) < 0.68

# PDF p. 254 - Digression: The Bootstrap

# 101 points all very close to 100
close_to_100 = [99.5 + random.random() for _ in range(101)]

# 101 points, 1 near 100, 50 near 0, 50 near 200
far_from_100 = ([99.5 + random.random()] +
                [random.random() for _ in range(50)] +
                [200 + random.random() for _ in range(50)])

medians_close = bootstrap_statistic(close_to_100, median, 100)
medians_far = bootstrap_statistic(far_from_100, median, 100)

assert standard_deviation(medians_close) < 1.0
assert standard_deviation(medians_far) > 90

# PDF p. 255

def estimate_sample_beta(pairs: List[Tuple[Vector, float]]) -> List[float]:
    x_sample = [x for x, _ in pairs]
    y_sample = [y for _, y in pairs]
    beta = least_squares_fit(x_sample, y_sample, learning_rate, 5000, 25)
    return beta

random.seed(0)

bootstrap_betas = bootstrap_statistic(list(zip(inputs, daily_minutes_good)),
                                      estimate_sample_beta,
                                      100)

# I commented out the time-consuming calculations

# # This takes a couple of minutes
# bootstrap_standard_errors = [
#     standard_deviation([beta[i] for beta in bootstrap_betas])
#     for i in range(4)
# ]
#
# assert 1.271 < bootstrap_standard_errors[0] < 1.272  # coef of 1
# assert 0.103 < bootstrap_standard_errors[1] < 0.104  # coef of num_friends
# assert 0.155 < bootstrap_standard_errors[2] < 0.156  # coef of work_hours
# assert 1.249 < bootstrap_standard_errors[3] < 1.250  # coef of PhD

def p_value(beta_hat_j: float, sigma_hat_j: float) -> float:
    t_j = beta_hat_j / sigma_hat_j
    if beta_hat_j > 0:
        return 2 * (1 - normal_cdf(t_j))
    else:
        return 2 * normal_cdf(t_j)

assert p_value(30.58, 1.27) < 0.001
assert p_value(0.972, 0.103) < 0.001
assert p_value(-1.865, 0.155) < 0.001
assert p_value(0.923, 1.249) > 0.4

# PDF p. 258 - Regularization
