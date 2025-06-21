def predict(alpha_p: float, beta_p: float, x_i: float) -> float:
    return beta_p * x_i + alpha_p

def error(alpha_er: float, beta_er: float, x_i: float, y_i: float) -> float:
    return predict(alpha_er, beta_er, x_i) - y_i

from Chapter_4_Linear_Algebra import Vector

def sum_of_sqerrors(alpha_sum_of_sqerrors: float, beta_sum_of_sqerrors: float, x_sum_of_sqerrors: Vector, y_sum_of_sqerrors: Vector) -> float:
    return sum(error(alpha_sum_of_sqerrors, beta_sum_of_sqerrors, x_i, y_i) ** 2
               for x_i, y_i in zip(x_sum_of_sqerrors, y_sum_of_sqerrors))


from typing import Tuple
from Chapter_4_Linear_Algebra import Vector
from Chapter_5_Statistics import correlation, standard_deviation, mean

def least_squares_fit(x_lsqf: Vector, y_lsqf: Vector) -> Tuple[float, float]:
    beta_lsqf = correlation(x_lsqf, y_lsqf) * standard_deviation(y_lsqf) / standard_deviation(x_lsqf)
    alpha_lsqf = mean(y_lsqf) - beta_lsqf * mean(x_lsqf)
    return alpha_lsqf, beta_lsqf


x = [i for i in range(-100, 110, 10)]
y = [3 * i - 5 for i in x]

assert least_squares_fit(x, y) == (-5, 3)


from Chapter_5_Statistics import num_friends_good, daily_minutes_good

alpha, beta = least_squares_fit(num_friends_good, daily_minutes_good)

assert 22.9 < alpha < 23.0
assert 0.9 < beta < 0.905


from Chapter_5_Statistics import de_mean

def total_sum_of_squares(y_total_sum_of_squares: Vector) -> float:
    return sum(v ** 2 for v in de_mean(y_total_sum_of_squares))

def r_squared(alpha_rsq: float, beta_rsq: float, x_rsq: Vector, y_rsq: Vector) -> float:
    """
    the fraction of variation in y captured by the model, which equals 1 - the fraction of variation in y not captured by the model
    """
    return 1.0 - (sum_of_sqerrors(alpha_rsq, beta_rsq, x_rsq, y_rsq) / total_sum_of_squares(y_rsq))

rsq = r_squared(alpha, beta, num_friends_good, daily_minutes_good)

assert 0.328 < rsq < 0.330



#Using Gradient Descent

import random
import tqdm
from Chapter_8_Gradient_Descent import gradient_step

num_epochs = 10000
random.seed(0)

guess = [random.random(), random.random()]

learning_rate = 0.00001

with tqdm.trange(num_epochs) as t:
    for _ in t:
        alpha, beta = guess

        grad_a = sum(2 * error(alpha, beta, x_i, y_i)
                     for x_i, y_i in zip(num_friends_good, daily_minutes_good))

        grad_b = sum(2 * error(alpha, beta, x_i, y_i) * x_i
                     for x_i, y_i in zip(num_friends_good, daily_minutes_good))

        loss = sum_of_sqerrors(alpha, beta, num_friends_good, daily_minutes_good)
        t.set_description(f"loss: {loss:.3f}")

        guess = gradient_step(guess, [grad_a, grad_b], -learning_rate)

alpha, beta = guess
assert 22.9 < alpha < 23.0
assert 0.9 < beta < 0.905


