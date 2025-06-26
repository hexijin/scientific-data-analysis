from typing import List, Callable, Iterable
from math import exp, log

import random

from grus_ch04_code import dot
from grus_ch06_code import inverse_normal_cdf
from grus_ch18_code import sigmoid

Tensor = list

# PDF p. 307

def shape(tensor: Tensor) -> List[int]:
    sizes: List[int] = []
    while isinstance(tensor, Tensor):
        sizes.append(len(tensor))
        tensor = tensor[0]
    return sizes

def is_1d(tensor: Tensor) -> bool:
    return not isinstance(tensor[0], list)

def tensor_sum(tensor: Tensor) -> float:
    if is_1d(tensor):
        return sum(tensor)
    else:
        return sum(tensor_sum(tensor_i) for tensor_i in tensor)

def tensor_apply(f: Callable[[float], float], tensor: Tensor) -> Tensor:
    if is_1d(tensor):
        return [f(x) for x in tensor]
    else:
        return [tensor_apply(f, tensor_i) for tensor_i in tensor]

def zeros_like(tensor: Tensor) -> Tensor:
    return tensor_apply(lambda _: 0.0, tensor)

def tensor_combine(f: Callable[[float, float], float],
                   t1: Tensor,
                   t2: Tensor) -> Tensor:
    if is_1d(t1):
        return [f(x, y) for x, y in zip(t1, t2)]
    else:
        return [tensor_combine(f, t1_i, t2_i) for t1_i, t2_i in zip(t1, t2)]

# PDF p. 310 - The Layer Abstraction

class Layer:
    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, gradient: Tensor) -> Tensor:
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def params(self) -> Iterable[Tensor]:
        return list()

    # noinspection PyMethodMayBeStatic
    def grads(self) -> Iterable[Tensor]:
        return list()

class Sigmoid(Layer):

    def __init__(self) -> None:
        self.sigmoids = list()  # to quiet linter

    def forward(self, inputs: Tensor) -> Tensor:
        self.sigmoids = tensor_apply(sigmoid, inputs)
        return self.sigmoids

    def backward(self, gradient: Tensor) -> Tensor:
        return tensor_combine(lambda sig, grad: sig * (1 - sig) * grad, self.sigmoids, gradient)

def random_uniform(*dims: int) -> Tensor:
    if len(dims) == 1:
        return [random.random() for _ in range(dims[0])]
    else:
        return [random_uniform(*dims[1:]) for _ in range(dims[0])]

def random_normal(*dims: int, mean: float = 0.0, variance: float = 1.0) -> Tensor:
    if len(dims) == 1:
        return [mean + variance * inverse_normal_cdf(random.random()) for _ in range(dims[0])]
    else:
        return [random_normal(*dims[1:], mean=mean, variance=variance) for _ in range(dims[0])]

def random_tensor(*dims: int, init: str = 'normal') -> Tensor:
    if init == 'normal':
        return random_normal(*dims)
    elif init == 'uniform':
        return random_uniform(*dims)
    elif init == 'xavier':
        variance = len(dims) / sum(dims)
        return random_normal(*dims, variance=variance)
    else:
        raise ValueError(f'unknown init: {init}')

class Linear(Layer):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 init: str = 'xavier') -> None:
        self.inputs = list()  # to quiet linter
        self.b_grad = list()  # to quiet linter
        self.w_grad = list()  # to quiet linter
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w = random_tensor(output_dim, input_dim, init=init)
        self.b = random_tensor(output_dim, init=init)

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return [dot(inputs, self.w[o]) + self.b[o] for o in range(self.output_dim)]

    def backward(self, gradient: Tensor) -> Tensor:
        self.b_grad = gradient
        self.w_grad = [[self.inputs[i] * gradient[o]
                        for i in range(self.input_dim)]
                       for o in range(self.output_dim)]
        return [sum(self.w[o][i] * gradient[o] for o in range(self.output_dim))
                for i in range(self.input_dim)]

    def params(self) -> Iterable[Tensor]:
        return [self.w, self.b]

    def grads(self) -> Iterable[Tensor]:
        return [self.w_grad, self.b_grad]

# PDF p. 315 - Neural Networks as a Sequence of Layers

class Sequential(Layer):
    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, gradient: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def params(self) -> Iterable[Tensor]:
        return (param for layer in self.layers for param in layer.params())

    def grads(self) -> Iterable[Tensor]:
        return (grad for layer in self.layers for grad in layer.grads())

# PDF p. 317 - Loss and Optimization

class Loss:
    def loss(self, predicted, actual: Tensor) -> float:
        raise NotImplementedError

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError

class SSE(Loss):
    def loss(self, predicted, actual: Tensor) -> float:
        squared_errors = tensor_combine(
            lambda predicted_i, actual_i: (predicted_i - actual_i) ** 2,
            predicted, actual)
        return tensor_sum(squared_errors)

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return tensor_combine(
            lambda predicted_i, actual_i: 2 * (predicted_i - actual_i),
            predicted, actual
        )

class Optimizer:
    def step(self, layer: Layer) -> None:
        raise NotImplementedError

class GradientDescent(Optimizer):
    def __init__(self, learning_rate: float = 0.1) -> None:
        self.lr = learning_rate

    def step(self, layer: Layer) -> None:
        for param, grad in zip(layer.params(), layer.grads()):
            param[:] = tensor_combine(lambda p, g: p - g * self.lr, param, grad)

class Momentum(Optimizer):
    def __init__(self, learning_rate: float, momentum: float = 0.9) -> None:
        self.lr = learning_rate
        self.mo = momentum
        self.updates: List[Tensor] = []  # running average

    def step(self, layer: Layer) -> None:
        if not self.updates:
            self.updates = [zeros_like(grad) for grad in layer.grads()]

        for update, param, grad in zip(self.updates, layer.params(), layer.grads()):
            update[:] = tensor_combine(lambda u, g: self.mo * u + (1 - self.mo) * g, update, grad)
            param[:] = tensor_combine(lambda p, u: p - self.lr * u, param, update)

# PDF p. 322

def tanh(x: float) -> float:
    if x < -100:
        return -1
    elif x > 100:
        return 1
    else:
        em2x = exp(-2 * x)
        return (1 - em2x) / (1 + em2x)

class Tanh(Layer):
    def __init__(self) -> None:
        self.tanh = list()  # to quiet linter

    def forward(self, inputs: Tensor) -> Tensor:
        self.tanh = tensor_apply(tanh, inputs)
        return self.tanh

    def backward(self, gradient: Tensor) -> Tensor:
        return tensor_combine(
            lambda t, g: (1 - t ** 2) * g,
            self.tanh,
            gradient
        )

class Relu(Layer):
    def __init__(self) -> None:
        self.inputs = list()  # to quiet linter

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return tensor_apply(lambda x: max(x, 0), inputs)

    def backward(self, gradient: Tensor) -> Tensor:
        return tensor_combine(lambda x, grad: grad if x > 0 else 0, self.inputs, gradient)

# PDF p. 325

def softmax(tensor: Tensor) -> Tensor:
    if is_1d(tensor):
        # Subtract largest value for numerical stability
        largest = max(tensor)
        exps = [exp(x - largest) for x in tensor]
        sum_of_exps = sum(exps)
        return [exp_i / sum_of_exps for exp_i in exps]
    else:
        return [softmax(tensor_i) for tensor_i in tensor]

class SoftmaxCrossEntropy(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        # Apply softmax to get probabilities
        probabilities = softmax(predicted)
        likelihoods = tensor_combine(lambda p, act: log(p + 1e-30) * act,
                                     probabilities, actual)

        return -tensor_sum(likelihoods)

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        probabilities = softmax(predicted)

        return tensor_combine(lambda p, act: p - act, probabilities, actual)

# PDF p. 328
