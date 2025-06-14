from typing import Tuple, List

def quadratic_function(x: float, params: Tuple[float, float, float]) -> float:
    a, b, c = params
    return a * x**2 + b * x + c

def quadratic_gradient(x: float) -> Tuple[float, float, float]:
    partial_wrt_a = x ** 2
    partial_wrt_b = x
    partial_wrt_c = 1
    return partial_wrt_a, partial_wrt_b, partial_wrt_c

def ssr_gradients_quadratic(dataset: List, params: Tuple[float, float, float]) \
        -> Tuple[float, float, float]:
    """The partial derivatives of the sum of squared residuals (SSR)"""
    dataset = list(dataset)  # because we are going to use it twice
    residuals = [y - quadratic_function(x, params) for x, y in dataset]
    gradients = [quadratic_gradient(x) for x, _ in dataset]
    # noinspection DuplicatedCode
    running_sum = (0.0, 0.0, 0.0)
    for residual, gradient in zip(residuals, gradients):
        running_sum = (
            running_sum[0] - 2.0 * residual * gradient[0],
            running_sum[1] - 2.0 * residual * gradient[1],
            running_sum[2] - 2.0 * residual * gradient[2]
        )
    return running_sum

def new_params_from_gradients(
        old_params: Tuple[float, float, float],
        gradients: Tuple[float, float, float],
        learning_rate: float) \
        -> Tuple[float, float, float]:
    zipped = list(zip(old_params, gradients))
    # Trying to go lower, so we move from the old parameters
    # in the opposite direction of the gradient, scaled by the learning rate.
    new_params = [-1.0 * learning_rate * gradient + old_param
                  for old_param, gradient in zipped]
    return new_params[0], new_params[1], new_params[2]
