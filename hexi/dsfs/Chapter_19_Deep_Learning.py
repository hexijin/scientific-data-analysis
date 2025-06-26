# The Tensor
"""tensors: n-dimensional arrays"""

from typing import List

Tensor = List

def shape(tensor_s: Tensor) -> List[int]:
    sizes: List[int] = []
    while isinstance(tensor_s, list):
        sizes.append(len(tensor_s))
        tensor_s = tensor_s[0]
    return sizes

assert shape([1, 2, 3]) == [3]
assert shape([[1, 2], [3, 4], [5, 6]]) == [3,2]


def is_1d(tensor_is_1d: Tensor) -> bool:
    """
   If tensor[0] is a list, it's a higher-order tensor.
   Otherwise, tensor is 1-dimensional (that is, a vector).
    """
    return not isinstance(tensor_is_1d[0], list)

assert is_1d([1, 2, 3])
assert not is_1d([[1, 2], [3, 4]])


def tensor_sum(tensor_s: Tensor) -> float:
    """
    Sums up all the values in the tensor
    """
    if is_1d(tensor_s):
        return sum(tensor_s)
    else:
        return sum(tensor_sum(tensor_i)
                   for tensor_i in tensor_s)

assert tensor_sum([1, 2, 3]) == 6
assert tensor_sum([[1, 2], [3, 4]]) == 10


from typing import Callable

def tensor_apply(f: Callable[[float], float], tensor_a: Tensor) -> Tensor:
    """Applies f element-wise"""
    if is_1d(tensor_a):
        return [f(x_ta) for x_ta in tensor_a]
    else:
        return [tensor_apply(f, tensor_i) for tensor_i in tensor_a]

assert tensor_apply(lambda m: m + 1, [1, 2, 3]) == [2, 3, 4]
assert tensor_apply(lambda m: 2 * m, [[1, 2], [3, 4]]) == [[2, 4], [6, 8]]


def zeros_like(tensor_zl: Tensor) -> Tensor:
    return tensor_apply(lambda _: 0.0, tensor_zl)

assert zeros_like([1, 2, 3]) == [0, 0, 0]
assert zeros_like([[1, 2], [3, 4]]) == [[0, 0], [0, 0]]


def tensor_combine(f: Callable[[float, float], float],
                   t1: Tensor,
                   t2: Tensor) -> Tensor:
    """applies f to corresponding elements of t1 and t2"""
    if is_1d(t1):
        return [f(x_t1, y_t1) for x_t1, y_t1 in zip(t1, t2)]
    else:
        return [tensor_combine(f, t1_i, t2_i)
                for t1_i, t2_i in zip(t1, t2)]

import operator
assert tensor_combine(operator.add, [1, 2, 3], [4, 5, 6]) == [5, 7, 9]
assert tensor_combine(operator.mul, [1, 2, 3], [4, 5, 6]) == [4, 10, 18]



# The Layer Abstraction

from typing import Iterable

class Layer:
    """
    Our neural networks will be composed of Layers, each of which
    knows how to do some computation on its inputs in the "forward"
    direction and propagate gradients in the "backward" direction.
    """
    def forward(self, inputs):
        """Note the lack of types. We're not going to be prescriptive
        about what kinds of inputs layers can take and what kinds
        of outputs they can return"""
        raise NotImplementedError

    def backward(self, gradient_layer):
        """
        Similarly, we're not going to be prescriptive about what the
        gradient looks like. It's up to you the user to make sure
        that you're doing things sensibly
        """
        raise NotImplementedError

    @staticmethod
    def params() -> Iterable[Tensor]:
        """
        Returns the parameters of this layer. The default implementation
        returns nothing, so that if you have a layer with no parameters
        you don't have to implement this.
        """
        return()

    @staticmethod
    def grads() -> Iterable[Tensor]:
        """
        Returns the gradients, in the same order as params().
        """
        return ()


from Chapter_18_Neural_Networks import sigmoid

class Sigmoid(Layer):
    def __init__(self):
        self.sigmoids = None

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Apply sigmoid to each element of the input tensor,
        and save the results to use in backpropagation.
        """
        self.sigmoids = tensor_apply(sigmoid, inputs)
        return self.sigmoids

    def backward(self, gradient_sl: Tensor) -> Tensor:
        return tensor_combine(lambda sig, grad: sig * (1 - sig) * grad,
                              self.sigmoids,
                              gradient_sl)



# The Linear Layer

import random
from Chapter_6_Probability import inverse_normal_cdf

def random_uniform(*dims: int) -> Tensor:
    if len(dims) == 1:
        return [random.random() for _ in range(dims[0])]
    else:
        return [random_uniform(*dims[1:]) for _ in range(dims[0])]

def random_normal(*dims: int,
                  mean: float = 0.0,
                  variance: float = 1.0) -> Tensor:
    if len(dims) == 1:
        return [mean + variance * inverse_normal_cdf(random.random()) for _ in range(dims[0])]
    else:
        return [random_normal(*dims[1:], mean=mean, variance=variance) for _ in range(dims[0])]

assert shape(random_uniform(2, 3, 4)) == [2, 3, 4]
assert shape(random_normal(5, 6, mean=10)) == [5, 6]

def random_tensor(*dims: int, init: str = 'normal') -> Tensor:
    if init == 'normal':
        return random_normal(*dims)
    elif init == 'uniform':
        return random_uniform(*dims)
    elif init == 'xavier':
        variance = len(dims) / sum(dims)
        return random_normal(*dims, variance=variance)
    else:
        raise ValueError(f"unknow init: {init}")


from Chapter_4_Linear_Algebra import dot

class Linear(Layer):
    def __init__(self, input_dim: int, output_dim: int, init: str = 'xavier') -> None:
        """
        A layer of output_dim neurons, each with input_dim weights (and a bias)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w = random_tensor(output_dim, input_dim, init=init)
        self.b = random_tensor(output_dim, init=init)
        self.inputs = None
        self.b_grad = None
        self.w_grad = None

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return [dot(inputs, self.w[o]) + self.b[o]
                for o in range(self.output_dim)]

    def backward(self, gradient_ll: Tensor) -> Tensor:
        self.b_grad = gradient_ll
        self.w_grad = [[self.inputs[ip] * gradient_ll[o]
                        for ip in range(self.input_dim)]
                       for o in range(self.output_dim)]
        return [sum(self.w[o][ip] * gradient_ll[o] for o in range(self.output_dim))
                for ip in range(self.input_dim)]

    def params(self) -> Iterable[Tensor]:
        return [self.w, self.b]

    def grads(self) -> Iterable[Tensor]:
        return [self.w_grad, self.b_grad]



# Neural Networks as a Sequence of Layers
from typing import List

class Sequential(Layer):
    """
    A layer consisting of a sequence of other layers.
    It's up to you to make sure that the output of each layer
    makes sense as the input to the next layer.
    """
    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs):
        """Just forward the input through the layers in order."""
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, gradient_sl):
        """Just backpropagate the gradients through layers in reverse"""
        for layer in self.layers:
            gradient_sl = layer.backward(gradient_sl)
        return gradient_sl

    def params(self)-> Iterable[Tensor]:
        """Just return the params for each layer."""
        return (param for layer in self.layers for param in layer.params())

    def grads(self) -> Iterable[Tensor]:
        """Just return the gradients for each layer."""
        return (grad for layer in self.layers for grad in layer.grads())


xor_net = Sequential([
    Linear(input_dim=2, output_dim=2),
    Sigmoid(),
    Linear(input_dim=2, output_dim=2),
    Sigmoid()
])



# Loss and Optimization

class Loss:
    def loss(self, predicted_loss: Tensor, actual: Tensor) -> float:
        """How good are our predictions? (Large numbers are worse.)"""
        raise NotImplementedError()

    def gradient(self, predicted_loss: Tensor, actual: Tensor) -> float:
        """How does the loss change as the predictions change"""
        raise NotImplementedError()


class SSE(Loss):
    """Loss function that computes the sum of the squared errors."""
    def loss(self, predicted_sse: Tensor, actual: Tensor) -> float:
        squared_errors = tensor_combine(lambda predicted_sqer, actual_sqer: (predicted_sqer - actual_sqer) ** 2,
                                       predicted_sse, actual)
        return tensor_sum(squared_errors)

    def gradient(self, predicted_sse: Tensor, actual: Tensor) -> Tensor:
        return tensor_combine(
            lambda predicted_g, actual_g: (predicted_g - actual_g) * 2,
            predicted_sse, actual
        )


class Optimizer:
    """
    An optimizer updates the weights of a layer (in place)
    using information known by either the layer or the optimizer.
    """
    def step(self, layer: Layer) -> None:
        raise NotImplementedError


class GradientDescent(Optimizer):
    def __init__(self, learning_rate: float = 0.1) -> None:
        self.lr = learning_rate

    def step(self, layer: Layer) -> None:
        for param, grad in zip(layer.params(), layer.grads()):
            param[:] = tensor_combine(
                lambda param_gd, grad_gd: param_gd -grad_gd * self.lr,
                param, grad
            )


tensor = [[1, 2], [3, 4]]

for row in tensor:
    row = [0, 0]
assert tensor == [[1, 2], [3, 4]], "assignment doesn't update a list"

for row in tensor:
    row[:] = [0, 0]
assert tensor == [[0, 0], [0, 0]], "but slice assignment does"


class Momentum(Optimizer):
    def __init__(self,
                 learning_rate: float,
                 momentum: float = 0.9) -> None:
        self.lr = learning_rate
        self.mo = momentum
        self.updates: List[Tensor] = []

    def step(self, layer: Layer) -> None:
        if not self.updates:
            self.updates = [zeros_like(grad) for grad in layer.grads()]
        for update, param, grad in zip(self.updates, layer.params(), layer.grads()):
            update[:] = tensor_combine(
                lambda u, g: self.mo * u + (1 - self.mo) * g,
                update,
                grad
            )
            param[:] = tensor_combine(
            lambda p, u: p - self.lr * u,
            param,
            update
            )



# Other activation functions

import math

def tanh(x_tanh: float) -> float:
    if x_tanh < -100: return -1,
    elif x_tanh > 100: return 1

    em2x = math.exp(-2 * x_tanh)
    return (1 - em2x) / (1 + em2x)

class Tanh(Layer):
    def __init__(self):
        self.tanh = None

    def forward(self, inputs: Tensor) -> Tensor:
        self.tanh = tensor_apply(tanh, inputs)
        return self.tanh

    def backward(self, gradient_tl: Tensor) -> Tensor:
        return tensor_combine(
            lambda tanh_, grad: (1 - tanh_ ** 2) * grad,
            self.tanh, gradient_tl
        )


class Relu(Layer):
    def __init__(self):
        self.inputs = None
        
    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return tensor_apply(lambda x_rl: max(x_rl, 0), inputs)

    def backward(self, gradient_rl: Tensor) -> Tensor:
        return tensor_combine(lambda x_rl, grad: grad if x_rl > 0 else 0,
                              self.inputs, gradient_rl)


# #Example: FizzBuzz Revisited
# import tqdm
#
# from Chapter_18_Neural_Networks import binary_encode, fizz_buzz_encode, argmax
#
# xs = [binary_encode(n) for n in range(101, 1024)]
# ys = [fizz_buzz_encode(n) for n in range(101, 1024)]
#
# num_hidden = 25
#
# random.seed(0)
#
# net = Sequential([
#     Linear(input_dim=10, output_dim=num_hidden, init='uniform'),
#     Tanh(),
#     Linear(input_dim=num_hidden, output_dim=4, init='uniform'),
#     Sigmoid()
# ])
#
# def fizzbuzz_accuracy(low: int, hi: int, net_fba: Layer) -> float:
#     num_correct = 0
#     for n in range(low, hi):
#         x_fba = binary_encode(n)
#         predicted_fba = argmax(net_fba.forward(x_fba))
#         actual = argmax(fizz_buzz_encode(x_fba))
#         if predicted_fba == actual:
#             num_correct += 1
#
# optimizer = Momentum(learning_rate=0.1, momentum=0.9)
# loss = SSE()
#
# with tqdm.trange(1000) as t:
#     for epoch in t:
#         epoch_loss = 0.0
#     for x, y in zip(xs, ys):
#         predicted = net.forward(x)
#         epoch_loss += loss.loss(predicted, y)
#         gradient = loss.gradient(predicted, y)
#         net.backward(gradient)
#
#         optimizer.step(net)
#
#     accuracy = fizzbuzz_accuracy(101, 1024, net)
#     t.set_description(f"fb loss: {epoch_loss:.2f} acc:{accuracy:.2f}")
#
# print("test results", fizzbuzz_accuracy(1, 101, net))
#
#

#Softmaxes and Cross-Entropy

num_hidden = 25

def softmax(tensor_smax: Tensor) -> Tensor:
    if is_1d(tensor_smax):
        largest = max(tensor_smax)
        exps = [math.exp(x_smax - largest) for x_smax in tensor_smax]

        sum_of_exps = sum(exps)
        return [exp_i / sum_of_exps for exp_i in exps]
    else:
        return [softmax(tensor_i) for tensor_i in tensor_smax]


class SoftmaxCrossEntropy(Loss):
    """
    This is the negative-log-likelihood of the observed values, given the neural net model. So if we choose weights to minimize it,
    our model will be maximizing the likelihood of the observed data.
    """
    def loss(self, predicted_smax: Tensor, actual: Tensor) -> float:
        probabilities = softmax(predicted_smax)

        likelihoods = tensor_combine(lambda p, act: math.log(p + 1e-30) * act,
                                     probabilities, actual)

        return -tensor_sum(likelihoods)

    def gradient(self, predicted_smax: Tensor, actual: Tensor) -> Tensor:
        probabilities = softmax(predicted_smax)
        return tensor_combine(lambda p, actual_sce: p - actual_sce,
                              probabilities, actual)


# random.seed(0)
#
# net = Sequential([
#         Linear(input_dim=10, output_dim=num_hidden, init='uniform'),
#         Tanh(),
#         Linear(input_dim=num_hidden, output_dim=4, init='uniform'),
#     ])
#
# optimizer = Momentum(learning_rate=0.1, momentum=0.9)
# loss = SoftmaxCrossEntropy()
#
# with tqdm.trange(1000) as t:
#     for epoch in t:
#         epoch_loss = 0.0
#
#         for x, y in zip(xs, ys):
#             predicted = net.forward(x)
#             epoch_loss += loss.loss(predicted, y)
#             gradient = loss.gradient(predicted, y)
#             net.backward(gradient)
#
#             optimizer.step(net)
#
#         accuracy = fizzbuzz_accuracy(101, 1024, net)
#         t.set_description(f"fb loss: {epoch_loss:.2f} acc:{accuracy:.2f}")
#
# print("test results", fizzbuzz_accuracy(1, 101, net))



# Dropout

class Dropout(Layer):
    def __init__(self, p: float) -> None:
        self.p = p
        self.train = True
        self.mask = None

    def forward(self, inputs: Tensor) -> Tensor:
        if self.train:
            self.mask= tensor_apply(lambda _: 0 if random.random() < self.p else 1,
                                    inputs)
            return tensor_combine(operator.mul, inputs, self.mask)
        else:
            return tensor_apply(lambda x_dl: x_dl * (1 - self.p),
                                input)

    def backward(self, gradient_dl: Tensor) -> Tensor:
        if self.train:
            return tensor_combine(operator.mul, gradient_dl, self.mask)
        else:
            raise RuntimeError("don't call backward when not in train mode")



# Example: MNIST

import mnist

mnist.temporary_dir = lambda: '/tmp'

train_images = mnist.train_images()
train_labels = mnist.train_labels()

assert shape(train_images) == [60000, 28, 28]
assert shape(train_labels) == [60000]


import matplotlib.pyplot as plt

fig, ax = plt.subplots(10, 10)

for i in range(10):
    for j in range(10):
        ax[i][j].imshow(train_images[10 * i + j], cmap='Greys')
        ax[i][j].xaxis.set_visible(False)
        ax[i][j].yaxis.set_visible(False)

plt.show()

test_images = mnist.test_images().tolist()
test_labels = mnist.test_labels().tolist()

assert shape(test_images) == [10000, 28, 28]
assert shape(test_labels) == [10000]


avg = tensor_sum(train_images) / 60000 / 28 / 28

train_images = [[(pixel - avg) / 256 for row in image for pixel in row]
                for image in train_images]
test_images = [[(pixel - avg) / 256 for row in image for pixel in row]
               for image in test_images]

assert shape(train_images) == [60000, 784], "images should be flattened"
assert shape(test_images) == [10000, 784], "images should be flattened"

assert -0.0001 < tensor_sum(train_images) < 0.0001


def one_hot_encode(i_: int, num_labels: int = 10) -> List[float]:
    return [1.0 if j_ == i_ else 0.0 for j_  in range(num_labels)]


train_labels = [one_hot_encode(label) for label in train_labels]
test_labels = [one_hot_encode(label) for label in test_labels]

assert shape(train_labels) == [60000, 10]
assert shape(test_labels) == [10000, 10]


import tqdm

from Chapter_18_Neural_Networks import argmax

def loop(model_l: Layer,
         images: List[Tensor],
         labels: List[Tensor],
         loss_l: Loss,
         optimizer_l: Optimizer = None) -> None:
    correct = 0
    total_loss = 0.0

    with tqdm.trange(len(images)) as trange:
        for item in trange:
            predicted_trange = model_l.forward(images[item])
            if argmax(predicted_trange) == argmax(labels[item]):
                correct += 1
            total_loss += loss_l.loss(predicted_trange, labels[item])

            if optimizer_l is not None:
                gradient_t = loss_l.gradient(predicted_trange, labels[item])
                model_l.backward(gradient_t)
                optimizer_l.step(model_l)

            avg_loss = total_loss / (i + 1)
            acc = correct / (i + 1)
            trange.set_description(f"mnist loss: {avg_loss:.3f} acc:{acc:.3f}")


random.seed(0)

model = Linear(784, 10)
loss = SoftmaxCrossEntropy()

optimizer = Momentum(learning_rate=0.01, momentum=0.99)

loop(model, train_images, train_labels, loss)

loop(model, test_images, test_labels, loss)


random.seed(0)

dropout1 = Dropout(0.1)
dropout2 = Dropout(0.1)

model = Sequential([
    Linear(784, 30),
    dropout1,
    Tanh(),
    Linear(30, 10),
    dropout2,
    Tanh(),
    Linear(10, 10)
])

optimizer = Momentum(learning_rate=0.01, momentum=0.99)
loss = SoftmaxCrossEntropy()

dropout1.train = dropout2.train = True
loop(model, train_images, train_labels, loss, optimizer)

dropout1.train = dropout2.train = False
loop(model, test_images, test_labels, loss, optimizer)



# Saving and Loading Models

import json

def save_weights(model_sw: Layer, filename: str) -> None:
    weights = list(model_sw.params())
    with open(filename, 'w') as f:
        json.dump(weights, f)

def load_weights(model_lw: Layer, filename: str) -> None:
    with open(filename) as f:
        weights = json.load(f)

    assert all(shape(param) == shape(weight) for param, weight in zip(model_lw.params(), weights))

    for param, weight in zip(model_lw.params(), weights):
        param[:] = weight