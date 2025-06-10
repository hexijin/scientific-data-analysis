import math

from matplotlib import pyplot as plt
import random
from typing import TypeVar, List, Iterator
from grus_ch08_code import difference_quotient, gradient_step
from grus_ch04_code import Vector, distance, scalar_multiply, vector_mean

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

def sum_of_squares_gradient(vec: Vector) -> Vector:
    return [2 * v_i for v_i in vec]

# pick a random starting point
v = [random.uniform(-10, 10) for i in range(3)]

for epoch in range(1000):
    grad_value = sum_of_squares_gradient(v)
    v = gradient_step(v, grad_value, -0.01)
    print(epoch, v)

assert distance(v, [0, 0, 0]) < 0.001  # v should be close to 0

# PDF p. 146

xs = range(-50, 50)
ys = [20 * x + 5 for x in xs]
inputs = zip(xs, ys)

def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta
    predicted = slope * x + intercept
    error = predicted - y
    # squared_error = error ** 2
    grad = [2 * error * x, 2 * error]
    return grad

# First version without the minibatches:
#
# theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
#
# learning_rate = 0.001
#
# # We call a pass through the dataset an epoch
# for epoch in range(5001):
#     # Compute the mean of the gradients
#     grad = vector_mean([linear_gradient(x, y, theta) for x, y in inputs])
#     # Take a step in that direction
#     theta = gradient_step(theta, grad, -learning_rate)
#     if epoch % 100 == 0:
#         print(epoch, theta)

T = TypeVar('T')  # This allows us to type "generic" functions

def minibatches(dataset: List[T],
                batch_size: int,
                shuffle: bool = True) -> Iterator[List[T]]:
    """Generates `batch_size`-sized minibatches from the dataset"""
    # start indexes 0, batch_size, 2 * batch_size, ...
    batch_starts = [start for start in range(0, len(dataset), batch_size)]

    if shuffle: random.shuffle(batch_starts)  # shuffle the batches

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]

theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

for epoch in range(1000):
    for batch in minibatches(minibatches(inputs, batch_size=20):
        grad = vector_mean([linear_gradient(x, y, theta) for x, y in batch])
        theta = gradient_step(theta, grad, -learning_rate)
        if epoch % 100 == 0:
            print(epoch, theta)
