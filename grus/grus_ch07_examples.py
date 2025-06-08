
# PDF p. 124

from matplotlib import pyplot as plt
import random
from grus_ch07_code import difference_quotient
from grus_ch04_code import distance, add, scalar_multiply

# PDF p. 142

def square(x: float) -> float:
    return x ** 2

def derivative(x: float) -> float:
    return 2 * x

xs = range(-10, 11)
actuals = [derivative(x) for x in xs]
estimates = [difference_quotient(square, x, h=0.001) for x in xs]

# plot to show they're basically the same

plt.title("Actual derivatives vs. Estimates")
plt.plot(xs, actuals, 'rx', label='Actual')      # red  x
plt.plot(xs, estimates, 'b+', label='Estimate')  # blue +
plt.legend(loc=9)
plt.show()

# PDF p. 145

def sum_of_squares_gradient(v: Vector) -> Vector:
    return [2 * v_i for v_i in v]

# pick a random starting point
v = [random.uniform(-10, 10) for i in range(3)]

for epoch in range(1000):
    grad = sum_of_squares_gradient(v)
    v = gradient_step(-0.01)
    print(epoch, v)

assert distance(v, [0, 0, 0]) < 0.001  # v should be close to 0