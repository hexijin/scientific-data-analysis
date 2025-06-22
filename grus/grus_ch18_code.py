from typing import List
import math

from grus_ch04_code import Vector, dot

# PDF p. 292

def step_function(x: float) -> float:
    return 1.0 if x >= 0 else 0.0

def perceptron_output(weights: Vector, bias: float, inputs: Vector) -> float:
    output = dot(weights, inputs) + bias
    return step_function(output)

# PDF p. 295

def sigmoid(t: float) -> float:
    return 1 / (1 + math.exp(-t))

def neuron_output(weights: Vector, inputs: Vector) -> float:
    return sigmoid(dot(weights, inputs))

# PDF p. 297

def feed_forward(neural_network: List[List[Vector]], input_vector: Vector) -> List[Vector]:
    """
    Perhaps for debugging reasons(?), Grus is assembling the outputs of all
    the layers, not just the last one.
    """
    outputs: List[Vector] = []

    for layer in neural_network:
        input_with_bias = input_vector + [1]
        layer_output = []
        for neuron in layer:
            single_neuron_output = neuron_output(neuron, input_with_bias)
            layer_output.append(single_neuron_output)
        outputs.append(layer_output)
        input_vector = layer_output

    return outputs

# PDF p. 299

FIRST_LAYER = 0
LAST_LAYER = -1

def minus_sqerror_gradients(network: List[List[Vector]],
                            input_vector: Vector,
                            target_vector) -> List[List[Vector]]:
    hidden_outputs, outputs = feed_forward(network, input_vector)
    output_deltas = [output_i * (1 - output_i) * (output_i - target)
                     for output_i, target in zip(outputs, target_vector)]
    minus_output_grads = [[-1.0 * output_deltas[i] * hidden_output_i
                           for hidden_output_i in hidden_outputs + [1]]
                          for i, _ in enumerate(network[LAST_LAYER])]
    hidden_deltas = [hidden_output_i * (1 - hidden_output_i) *
                     dot(output_deltas, [n[i] for n in network[LAST_LAYER]])
                     for i, hidden_output_i in enumerate(hidden_outputs)]
    minus_hidden_grads = [[-1.0 * hidden_deltas[i] * input_i
                           for input_i in input_vector + [1]]
                          for i, _ in enumerate(network[FIRST_LAYER])]
    return [minus_hidden_grads, minus_output_grads]

# PDF p. 302 - Fizz Buzz

NUMBER = [1, 0, 0, 0]
FIZZ = [0, 1, 0, 0]
BUZZ = [0, 0, 1, 0]
# I wonder what would happen if we had one less output and made FIZZBUZZ = [0, 1, 1]
FIZZBUZZ = [0, 0, 0, 1]

def fizz_buzz_encode(x: int) -> Vector:
    if x % 15 == 0:
        return FIZZBUZZ
    elif x % 5 == 0:
        return BUZZ
    elif x % 3 == 0:
        return FIZZ
    else:
        return NUMBER

BINARY_ENCODING_LENGTH = 10
FIZZBUZZ_ENCODING_LENGTH = 4

def binary_encode(x: int) -> Vector:
    binary: Vector = []
    for i in range(BINARY_ENCODING_LENGTH):
        binary.append(x % 2)
        x = x // 2
    return binary
