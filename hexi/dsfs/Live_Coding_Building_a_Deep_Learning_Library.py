"""
1.
"""

# Tensors
"""
A tensor is just a n-dimensional array
"""

from numpy import ndarray as tensor



# Loss function
"""
A loss function measures how good our predictions are,
we can use this to adjust the parameters of our network
"""

class Loss:
    def loss(self, predicted_l: tensor, actual: tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted_l: tensor, actual: tensor) -> tensor:
        raise NotImplementedError

class MSE(Loss):
    """
    MSE is mean squared error, although we're just going to do total squared error
    """

    def loss(self, predicted_mse: tensor, actual: tensor) -> float:
        return np.sum((predicted_mse - actual) ** 2)

    def grad(self, predicted_mse: tensor, actual: tensor) -> tensor:
        return 2  * (predicted_mse - actual)



# Layers
"""
Our neural nets will be made up of layers.
Each layer needs to pass its inputs forward
and propagate gradients backward. For example, a neural net might look like
inputs -> Linear -> Tanh -> Linear -> outputs"""

from typing import Dict

class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, tensor] = {}
        self.grads: Dict[str, tensor] = {}

    def forward(self, inputs_f: tensor) -> tensor:
        """
        Produce the outputs corresponding to the given inputs
        """
        raise NotImplementedError

    def backward(self, grad: tensor) -> tensor:
        """
        Backpropagate this gradient through the layer
        """
        raise NotImplementedError


class Linear(Layer):
    """
    Computes output = inputs @ w + b
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        """
        inputs will be (batch_size, input_size)
        outputs will be (batch_size, output_size)
        """
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.zeros(output_size)
        self.inputs_lf = None

    def forward(self, inputs_lf: tensor) -> tensor:
        """
        outputs = inputs @ w + b
        """
        self.inputs_lf = inputs_lf
        return inputs_lf @ self.params["w"] + self.params["b"]

    def backward(self, grad: tensor) -> tensor:
        """
        if y = f(x) and x = a * b + c
        then dy/da = f'(x) * b
        and dy/dn = f'(x) * a
        and dy/dc = f'(x)

        if y = f(x) and x = a @ b + c
        then dy/da = f'(x) @ b.T
        and dy/dn = a.T @ f'(x)
        and dy/dc = f'(x)
        """
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs_lf.T @ grad
        return grad @ self.params["w"].T

from typing import Callable

F = Callable[[tensor], tensor]

class Activation(Layer):
    """
    An activation layer just applies a function
    elementwise to its inputs
    """
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime
        self.inputs_al = None

    def forward(self, inputs_al: tensor) -> tensor:
        self.inputs_al = inputs_al
        return self.f(inputs_al)

    def backward(self, grad: tensor) -> tensor:
        """
        if y = f(x) and x = g(z)
        then dy/dc = f'(x) * g'(z)
        """
        return self.f_prime(self.inputs_al) * grad

def tanh(x_t: tensor) -> tensor:
    return np.tanh(x_t)

def tanh_prime(x_tp: tensor) -> tensor:
    y_tp = tanh(x_tp)
    return 1 - y_tp ** 2


class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)



# Neural Net
"""
A NeuralNet is just a collection of layers.
It behaves a lot like a layer itself, 
although we're not going to make it one
"""

from typing import Sequence, Iterator, Tuple

class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs_nn: tensor) -> tensor:
        for layer in self.layers:
            inputs_nn = layer.forward(inputs_nn)
        return inputs_nn

    def backward(self, grad: tensor) -> tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params_and_grads(self) -> Iterator[Tuple[tensor, tensor]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad


# Optimizers
"""
We use an optimizer to adjust the parameters
of our network based on the gradients computed 
during propagation
"""

class Optimizer:
    def step(self, net_opt: NeuralNet) -> None:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, net_sgd: NeuralNet) -> None:
        for param, grad in net_sgd.params_and_grads():
            param -= self.lr * grad



# Data
"""
We'll feed inputs into our network in batches.
So here are some tools for iterating over data in batches.
"""

from typing import Iterator, NamedTuple

batch = NamedTuple("batch", [("inputs", tensor), ("targets", tensor)])

class DataIterator:
    def __call__(self, inputs_di: tensor, targets_di: tensor) -> Iterator:
        raise NotImplementedError

class BatchIterator:
    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs_bi: tensor, targets_bi: tensor) -> Iterator:
        starts = np.arange(0, len(inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)

        for start in starts:
            end =  start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            yield batch(batch_inputs, batch_targets)



# Training
"""
Here's a function that can train a neural net.
"""

def train(net_t: NeuralNet,
          inputs_t: tensor,
          targets_t: tensor,
          num_epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD()) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_ in iterator(inputs_t, targets_t):
            predicted_t = net_t.forward(batch_.inputs)
            epoch_loss += loss.loss(predicted_t, batch_.targets)
            grad = loss.grad(predicted_t, batch_.targets)
            net.backward(grad)
            optimizer.step(net)
        print(epoch, epoch_loss)



# XOR Example
"""
The canonical example of a function that can't be 
learned with a simple linear model is XOR
"""

import numpy as np

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

net = NeuralNet([
    Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2),
])

train(net, inputs, targets)

for x,y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)



# Fizz Buzz
"""
FizzBuzz is the following problem:

For each of the numbers 1 to 100:
* if the number is divisible by 3, print "fizz"
* if the number is divisible by 5, print "buzz",
* if the number is divisible by 15, print "fizzbuzz"
* otherwise, just print the number
"""

from typing import List

def fizz_buzz_encode(x_fb: int) -> List[int]:
    if x_fb % 15 == 0:
        return [0, 0, 0, 1]
    elif x_fb % 5 == 0:
        return [0, 0, 1, 0]
    elif x_fb % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]

def binary_encode(x_be: int) -> List[int]:
    """
    10 digit binary encoding of x
    """
    return [x_be >> i & 1 for i in range(10)]

inputs = np.array([
    binary_encode(x)
    for x in range(101, 1024)
])

targets = np.array([
    fizz_buzz_encode(x)
    for x in range(101, 1024)
])

net = NeuralNet([
    Linear(input_size=10, output_size=50),
    Tanh(),
    Linear(input_size=50, output_size=4)
])

train(net,
      inputs,
      targets,
      num_epochs=50)

for x in range(1, 101):
    predicted_fb = net.forward(binary_encode(x))
    predicted_idx = np.argmax(predicted_fb)
    actual_idx = np.argmax(fizz_buzz_encode(x))
    labels = [str(x), "fizz", "buzz", "fizzbuzz"]
    print(x, labels[predicted_idx], labels[actual_idx])