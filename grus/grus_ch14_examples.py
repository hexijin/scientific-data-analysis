import random

from grus_ch05_examples import num_friends_good, daily_minutes_good
from grus_ch08_code import gradient_step
from grus_ch14_code import least_squares_fit, r_squared, error

# PDF p. 241

sample_xs = [i for i in range(-100, 110, 10)]
sample_ys = [3 * x - 5 for x in sample_xs]

sample_fit = least_squares_fit(sample_xs, sample_ys)

assert sample_fit[0] == -5
assert sample_fit[1] == 3

alpha_good, beta_good = least_squares_fit(num_friends_good, daily_minutes_good)

assert 22.9 < alpha_good < 23.0
assert 0.9 < beta_good < 0.905

rsq = r_squared(alpha_good, beta_good, num_friends_good, daily_minutes_good)

assert 0.328 < rsq < 0.331

# PDF p. 244

num_epochs = 10000
random.seed(0)

guess = [random.random(), random.random()]

learning_rate = 0.00001

for _ in range(num_epochs):
    alpha, beta = guess
    minus_grad_a = -1.0 * sum(2 * error(alpha, beta, x_i, y_i)
                              for x_i, y_i in zip(num_friends_good, daily_minutes_good))
    minus_grad_b = -1.0 * sum(2 * error(alpha, beta, x_i, y_i) * x_i
                              for x_i, y_i in zip(num_friends_good, daily_minutes_good))

    guess = gradient_step(guess, [minus_grad_a, minus_grad_b], learning_rate)

alpha_gradient_descent, beta_gradient_descent = guess

assert 22.9 < alpha_gradient_descent < 23.0
assert 0.9 < beta_gradient_descent < 0.905
