"""
Our neural nets will be made up of layers.
"""
from typing import Dict, Callable

import numpy as np

from joelnet.tensor import Tensor

class Layer:
    def __init__(self):
        self.params: Dict[str, Tensor] = dict()
        self.grads: Dict[str, Tensor] = dict()
        # noinspection PyTypeChecker
        self.inputs: Tensor = None

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError

class Linear(Layer):
    """
    computes output = inputs @ w + b
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        # inputs will be (batch_size, input_size)
        # outputs will be (batch_size, output_size)
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = a * b + c
        then
        dy/da = f'(x) * b
        dy/db = f'(x) * a
        dy/dc = f'(x)
        """
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T

F = Callable[[Tensor], Tensor]

class Activation(Layer):
    """
    An activation layer just applies a function element-wise to its inputs
    """
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        """
        return self.f_prime(self.inputs) * grad

def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y ** 2

class Tanh(Activation):
    def __init__(self) -> None:
        super().__init__(tanh, tanh_prime)
