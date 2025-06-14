import numpy as np
import matplotlib.pyplot as plt

from quadratic_fit import *

test_data_xs = [-2, -1, 1, 2]
test_data_ys = [2, 4, 6, 4]
test_dataset = list(zip(test_data_xs, test_data_ys))

params = (0.0, 0.0, 0.0)
learning_rate = 0.01
for epoch in range(101):
    gradients = ssr_gradients_quadratic(test_dataset, params)
    params = new_params_from_gradients(params, gradients, learning_rate)
    print(f'epoch: {epoch}')
    print(f'gradients are {gradients}')
    print(f'params are {params}')

xs = np.linspace(-3, 3, 61)
ys = [quadratic_function(x, params) for x in xs]

# noinspection DuplicatedCode
plt.plot(xs, ys)
plt.xlim(-4, 4)
plt.ylim(0, 8)
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(test_data_xs, test_data_ys)
plt.title(f'Quadratic function with a: {params[0]:.2f}, ' +
          f'b: {params[1]:.2f}, ' +
          f'c: {params[2]:.2f}')
plt.grid(True)
plt.show()
