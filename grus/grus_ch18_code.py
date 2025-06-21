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
