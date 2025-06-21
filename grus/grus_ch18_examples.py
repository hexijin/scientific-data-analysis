from grus_ch18_code import perceptron_output

# PDF p. 293

and_weights = [2.0, 2.0]
and_bias = -3.0

assert perceptron_output(and_weights, and_bias, [1, 1]) == 1
assert perceptron_output(and_weights, and_bias, [0, 1]) == 0
assert perceptron_output(and_weights, and_bias, [1, 0]) == 0
assert perceptron_output(and_weights, and_bias, [0, 0]) == 0

or_weights = [2.0, 2.0]
or_bias = -1.0

assert perceptron_output(or_weights, or_bias, [1, 1]) == 1
assert perceptron_output(or_weights, or_bias, [0, 1]) == 1
assert perceptron_output(or_weights, or_bias, [1, 0]) == 1
assert perceptron_output(or_weights, or_bias, [0, 0]) == 0

not_weights = [-2.0]
not_bias = 1.0

assert perceptron_output(not_weights, not_bias, [1]) == 0
assert perceptron_output(not_weights, not_bias, [0]) == 1

and_gate = min
or_gate = max
def xor_gate(x: float, y: float) -> int:
    return 0 if x == y else 1

# PDF p. 297
