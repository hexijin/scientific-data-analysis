"""a predictive model motivated by the way the brain works, which is a collection of neurons wired together. Each neuron looks at the outputs of the other neurons that feed into it, does a calculation, and then either fires or doesn't."""

# Perceptron
"""computes a weighted sum of its inputs and fires if that weighted sum is 0 or greater"""

from Chapter_4_Linear_Algebra import Vector, dot

def step_function(x_sf: float) -> float:
    return 1.0 if x_sf >= 0 else 0.0

def perceptron_output(weights: Vector, bias: float, x_po: Vector) -> float:
    calculation = dot(weights, x_po) + bias
    return step_function(calculation)


and_weights = [2., 2]
and_bias = -3.

assert perceptron_output(and_weights, and_bias, [1, 1]) == 1
assert perceptron_output(and_weights, and_bias, [0, 0]) == 0
assert perceptron_output(and_weights, and_bias, [1, 0]) == 0
assert perceptron_output(and_weights, and_bias, [0, 1]) == 0


or_weights = [2., 2]
or_bias = -1.

assert perceptron_output(or_weights, or_bias, [1, 1]) == 1
assert perceptron_output(or_weights, or_bias, [0, 0]) == 0
assert perceptron_output(or_weights, or_bias, [1, 0]) == 1
assert perceptron_output(or_weights, or_bias, [0, 1]) == 1


not_weights = [-2.]
not_bias = 1.

assert perceptron_output(not_weights, not_bias, [0]) == 1
assert perceptron_output(not_weights, not_bias, [1]) == 0


and_gate = min
or_gate = max
xor_gate = lambda x_, y_: 0 if x_ == y_ else 1



# Feed-Forward Neural Networks
"""
feed-forward neural networks: consist of discrete layers of neurons, each connected to the next.
1) an input layer: receives inputs and feeds them forwards unchanged
2) one or more hidden layers: consists of neurons that take the outputs of the previous layer, performs some calculation, and passes the result to the next layer
3) an output layer: produces the final results
"""

"""use a smoother function to replace the step function, in order to use calculus"""
import math

def sigmoid(t_s: float) -> float:
    return 1 / (1 + math.exp(-t_s))

def neuron_output(weights: Vector, inputs: Vector) -> float:
    # weights includes the bias term, inputs includes a 1
    return sigmoid(dot(weights, inputs))


"""Represent a neural network as a list (layers) of lists (neurons) of vectors (weights)"""

from typing import List

def feed_forward(neural_network: List[List[Vector]],
                 input_vector: Vector) -> List[Vector]:
    """
    Feeds the input vector through the neural network.
    Returns the outputs of all layers (not just the last one).
    """
    outputs: List[Vector] = []
    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias)
                  for neuron in layer]
        outputs.append(output)

        input_vector = output

    return outputs


xor_network = [[[20., 20, -30],
                [20., 20, -10]],
               [[-60., 60, -30]]]

assert 0.000 < feed_forward(xor_network, [0,0])[-1][0] < 0.001
assert 0.999 < feed_forward(xor_network, [1,0])[-1][0] < 1.000
assert 0.999 < feed_forward(xor_network, [0,1])[-1][0] < 1.000
assert 0.000 < feed_forward(xor_network, [1,1])[-1][0] < 0.001



# Backpropagation

def sqerror_gradients(network_: List[List[Vector]],
                      input_vector: Vector,
                      target_vector: Vector) -> List[List[Vector]]:
    """Given a neural network, an input vector, and a target vector,
    make a prediction and compute the gradient of the squared error loss
    with respect to the neuron weights"""
    # forward pass
    hidden_outputs, outputs = feed_forward(network_, input_vector)

    # gradients with respect to output neuron pre-activation outputs
    output_deltas = [output * (1 - output) * (output - target)
                     for output, target in zip(outputs, target_vector)]

    # gradients with respect to output neuron weights
    output_grads = [[output_deltas[i] * hidden_output
                     for hidden_output in hidden_outputs + [1]]
                    for i, output_neuron in enumerate(network[-1])]

    # gradients with respect to hidden neuron preactivation outputs
    hidden_deltas = [hidden_output * (1 - hidden_output) *
                     dot(output_deltas, [n[i] for n in network_[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]

    # gradients with respect to hidden neuron weights
    hidden_grads = [[hidden_deltas[i] * input_ for input_ in input_vector + [1]]
                    for i, hidden_neuron in enumerate(network_[0])]

    return [hidden_grads, output_grads]


import random
random.seed(0)

xs = [[0., 0], [0., 1], [1., 0], [1., 1]]
ys = [[0.], [1.], [1.], [0.]]

network = [
             [[random.random() for _ in range (2 + 1)],
              [random.random() for _ in range (2 + 1)]],
             [[random.random() for _ in range (2 + 1)]]
          ]


from Chapter_8_Gradient_Descent import gradient_step
import tqdm

learning_rate = 1.0

for epoch in tqdm.trange(20000, desc="neural net for xor"):
    for x, y in zip(xs, ys):
        gradients = sqerror_gradients(network, x, y)
        network = [[gradient_step(neuron, grad, -learning_rate)
                    for neuron, grad in zip(layer, layer_grad)]
                   for layer, layer_grad in zip(network, gradients)]

assert feed_forward(network, [0, 0])[-1][0] < 0.01
assert feed_forward(network, [0, 1])[-1][0] > 0.99
assert feed_forward(network, [1, 0])[-1][0] > 0.99
assert feed_forward(network, [1, 1])[-1][0] < 0.01



# # Example: Fizz Buzz
#
# def fizz_buzz_encode(x_fb: int) -> Vector:
#     if x_fb % 15 == 0:
#         return [0, 0, 0, 1]
#     elif x_fb % 5 == 0:
#         return [0, 0, 1, 0]
#     elif x_fb % 3 == 0:
#         return [0, 1, 0, 0]
#     else:
#         return [1, 0, 0, 0]
#
# assert fizz_buzz_encode(2) == [1, 0, 0, 0]
# assert fizz_buzz_encode(6) == [0, 1, 0, 0]
# assert fizz_buzz_encode(10) == [0, 0, 1, 0]
# assert fizz_buzz_encode(30) == [0, 0, 0, 1]
#
#
# def binary_encode(x_b: int) -> Vector:
#     binary: List[float] = []
#
#     for i in range(10):
#         binary.append(x_b % 2)
#         x_b = x_b // 2
#
#     return binary
#
# assert binary_encode(0) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# assert binary_encode(1) == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# assert binary_encode(10) == [0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
# assert binary_encode(101) == [1, 0, 1, 0, 0, 1, 1, 0, 0, 0]
# assert binary_encode(999) == [1, 1, 1, 0, 0, 1, 1, 1, 1, 1]
#
#
# xs = [binary_encode(n) for n in range(101, 1024)]
# ys = [fizz_buzz_encode(n) for n in range(101, 1024)]
#
# num_hidden = 25
#
# network = [
#     [[random.random() for _ in range(10 + 1)] for _ in range(num_hidden)],
#     [[random.random() for _ in range(num_hidden + 1)] for _ in range(4)]
# ]
#
# from Chapter_4_Linear_Algebra import squared_distance
#
# learning_rate = 1.0
#
# with tqdm.trange(500) as t:
#     for epoch in t:
#         epoch_loss = 0.0
#
#         for x, y in zip(xs, ys):
#             predicted = feed_forward(network, x)[-1]
#             epoch_loss += squared_distance(predicted, y)
#             gradients = sqerror_gradients(network, x, y)
#
#             network = [[gradient_step(neuron, grad, -learning_rate)
#                         for neuron, grad, in zip(layer, layer_grad)]
#                        for layer, layer_grad in zip(network, gradients)]
#
#         t.set_description(f"fizz buzz (loss: {epoch_loss:.2f})")
#
#
def argmax(xs_argmax: list) -> int:
    return max(range(len(xs_argmax)), key=lambda i: xs_argmax[i])
#
# assert argmax([0, -1]) == 0
# assert argmax([-1, 1]) == 1
# assert argmax([-1, 10, 5, 20, -3]) == 3
#
#
# num_correct = 0
#
# for n in range(1, 101):
#     x = binary_encode(n)
#     predicted = argmax(feed_forward(network, x)[-1])
#     actual = argmax(fizz_buzz_encode(n))
#     labels = [str(n), "fizz", "buzz", "fizzbuzz"]
#     print(n,  labels[predicted], labels[actual])
#
#     if predicted == actual:
#         num_correct += 1
#
# print(num_correct, "/", 100)
#
