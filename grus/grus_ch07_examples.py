from grus_ch07_code import (normal_approximation_to_binomial,
                            normal_two_sided_bounds, normal_probability_between)

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
