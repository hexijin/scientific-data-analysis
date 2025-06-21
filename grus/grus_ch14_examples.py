import random

from grus_ch05_examples import num_friends_good, daily_minutes_good
from grus_ch14_code import least_squares_fit, r_squared

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


