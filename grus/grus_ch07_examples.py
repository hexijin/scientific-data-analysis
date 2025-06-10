import random
from typing import List
from grus_ch07_code import *

# PDF p. 126

mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)

# I am getting 463, 658, but Grus says we should get 469, 531
lower_bound, upper_bound = normal_two_sided_bounds(0.95, mu_0, sigma_0)

# PDF bottom of p. 126

lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)

print(f'Lower bound: {lower_bound}, upper bound: {upper_bound}')

# The mu and sigma based on p = 0.55
mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)

print(f'Mean for p=0.55: {mu_1}, sigma: {sigma_1}')

# A type 2 error means we fail to reject the null hypothesis,
# which will happen when X is still in our original interval
type_2_probability = normal_probability_between(lo, hi, mu_1, sigma_1)
power = 1 - type_2_probability    # 0.887

print(f'Power: {power}')

hi = normal_upper_bound(0.95, mu_0, sigma_0)

type_2_probability = normal_probability_below(hi, mu_1, sigma_1)
power = 1 - type_2_probability    # 0.936

# PDF pp. 127-128

two_sided_p_value(529.5, mu_0, sigma_0)   # 0.062

extreme_value_count = 0
for _ in range(1000):
    num_heads = sum(1 if random.random() < 0.5 else 0 for _ in range(1000))
    if num_heads >= 530 or num_heads <= 470:
        extreme_value_count += 1

print(f'Extreme value count: {extreme_value_count}')

upper_p_value = normal_probability_above
lower_p_value = normal_probability_below

upv525 = upper_p_value(524.5, mu_0, sigma_0)  # 0.061
print(f'upper p value for 525: {upv525}')
upv527 = upper_p_value(526.5, mu_0, sigma_0)  # 0.047
print(f'upper p value for 527: {upv527}')

# PDF p. 130

p525 = 525 / 1000
mu525 = p525
sigma525 = math.sqrt(p525 * (1 - p525) / 1000)   # 0.0158
print(f'mu for 525: {mu525}, sigma for 525: {sigma525}')
n525 = normal_two_sided_bounds(0.95, mu525, sigma525)
print(f'normal_two_sided_bounds for 525: {n525}')

p540 = 540 / 1000
mu540 = p540
sigma540 = math.sqrt(p540 * (1 - p540) / 1000)   # 0.0158
print(f'mu for 540: {mu540}, sigma for 540: {sigma540}')
n540 = normal_two_sided_bounds(0.95, mu540, sigma540)
print(f'normal_two_sided_bounds for 540: {n540}')


def run_experiment():
    """Flips a fair coin 1000 times, True = heads, False = tails"""
    return [random.random() < 0.5 for _ in range(1000)]

def reject_fairness(experiment: List[bool]) -> bool:
    count_heads = len([flip for flip in experiment if flip])
    return count_heads < 469 or count_heads > 531

random.seed(0)
# 1000 experiments, each consisting of 1000 flips
experiments = [run_experiment() for _ in range(1000)]
num_rejections = len([experiment for experiment in experiments
                      if reject_fairness(experiment)])

print(f'Number of rejections: {num_rejections} was expected to be 46')

# PDF p. 134

z = a_b_test_statistic(1000, 200, 1000, 180)   # -1.14
print(f'z = {z}')
print(f'two_sided_p_value(z)', two_sided_p_value(z))   # 0.254

z = a_b_test_statistic(1000, 200, 1000, 150)   # -2.94
print(f'z = {z}')
print(f'two_sided_p_value(z)', two_sided_p_value(z))   # 0.003
