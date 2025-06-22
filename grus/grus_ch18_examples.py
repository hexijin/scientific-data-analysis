import random
import tqdm

from grus_ch04_code import squared_distance
from grus_ch08_code import gradient_step
from grus_ch18_code import perceptron_output, feed_forward, LAST_LAYER, minus_sqerror_gradients, binary_encode
from grus_ch18_code import fizz_buzz_encode, NUMBER, FIZZ, BUZZ, FIZZBUZZ
from grus_ch18_code import BINARY_ENCODING_LENGTH, FIZZBUZZ_ENCODING_LENGTH

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

# and_gate = min
# or_gate = max
# def xor_gate(x: float, y: float) -> int:
#     return 0 if x == y else 1

# PDF p. 297

xor_network = [
    [[20.0, 20.0, -30.0], [20.0, 20.0, -10.0]],  # The first (input) layer
    [[-60.0, 60.0, -30.0]]                       # The last (output) layer
]

ONLY_OUTPUT = 0

assert 0.000 < feed_forward(xor_network, [0, 0])[LAST_LAYER][ONLY_OUTPUT] < 0.001
assert 0.999 < feed_forward(xor_network, [1, 0])[LAST_LAYER][ONLY_OUTPUT] < 1.000
assert 0.999 < feed_forward(xor_network, [0, 1])[LAST_LAYER][ONLY_OUTPUT] < 1.000
assert 0.000 < feed_forward(xor_network, [1, 1])[LAST_LAYER][ONLY_OUTPUT] < 0.001

# PDF p. 300

random.seed(0)

fb_xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
fb_ys = [[0.0], [1.0], [1.0], [0.0]]

# start with random weights
test_network = [
    # The first (input) layer
    [[random.random() for _ in range(2 + 1)], [random.random() for _ in range(2 + 1)]],
    # The last (output) layer
    [[random.random() for _ in range(2 + 1)]]
]

xor_learning_rate = 1.0

for _ in range(20000):
    for training_x, training_y in zip(fb_xs, fb_ys):
        minus_gradients = minus_sqerror_gradients(test_network, training_x, training_y)
        test_network = [
            [gradient_step(neuron, minus_gradient, xor_learning_rate)
             for neuron, minus_gradient in zip(layer, minus_layer_grad)]
            for layer, minus_layer_grad in zip(test_network, minus_gradients)
        ]

assert feed_forward(test_network, [0, 0])[LAST_LAYER][ONLY_OUTPUT] < 0.01
assert feed_forward(test_network, [0, 1])[LAST_LAYER][ONLY_OUTPUT] > 0.99
assert feed_forward(test_network, [1, 0])[LAST_LAYER][ONLY_OUTPUT] > 0.99
assert feed_forward(test_network, [1, 1])[LAST_LAYER][ONLY_OUTPUT] < 0.01

# PDF p. 302 - Fizz Buzz

assert fizz_buzz_encode(2) == NUMBER
assert fizz_buzz_encode(6) == FIZZ
assert fizz_buzz_encode(10) == BUZZ
assert fizz_buzz_encode(30) == FIZZBUZZ

assert binary_encode(0) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
assert binary_encode(1) == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
assert binary_encode(10) == [0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
assert binary_encode(101) == [1, 0, 1, 0, 0, 1, 1, 0, 0, 0]
assert binary_encode(999) == [1, 1, 1, 0, 0, 1, 1, 1, 1, 1]

fb_range = range(101, 1024)
fb_xs = [binary_encode(n) for n in fb_range]
fb_ys = [fizz_buzz_encode(n) for n in fb_range]

NUM_HIDDEN = 25

fb_network = [
    [[random.random() for _ in range(BINARY_ENCODING_LENGTH + 1)] for _ in range(NUM_HIDDEN)],
    [[random.random() for _ in range(NUM_HIDDEN + 1)] for _ in range(FIZZBUZZ_ENCODING_LENGTH)]
]

fb_learning_rate = 1.0

with tqdm.trange(500) as t:
    for epoch in t:
        epoch_loss = 0.0
        for fb_x, fb_y in zip(fb_xs, fb_ys):
            predicted = feed_forward(fb_network, fb_x)[LAST_LAYER]
            epoch_loss += squared_distance(predicted, fb_y)
            minus_gradients = minus_sqerror_gradients(fb_network, fb_x, fb_y)
            # Take a gradient step for each neuron in each layer
            fb_network = [
                [gradient_step(neuron, minus_grad, fb_learning_rate)
                 for neuron, minus_grad in zip(layer, minus_layer_grad)]
                for layer, minus_layer_grad in zip(fb_network, minus_gradients)
            ]
        t.set_description(f"fizz buzz (loss: {epoch_loss:.2f})")

def argmax(xs: list) -> int:
    return max(range(len(xs)), key=lambda i: xs[i])

assert argmax([0, -1]) == 0
assert argmax([-1, 0]) == 1
assert argmax([-1, 10, 5, 20, -3]) == 3

num_correct = 0
num_tests = 100
for n in range(1, num_tests + 1):
    x = binary_encode(n)
    predicted = argmax(feed_forward(fb_network, x)[LAST_LAYER])
    actual = argmax(fizz_buzz_encode(n))
    labels = [str(n), "fizz", "buzz", "fizzbuzz"]
    print(n, labels[predicted], labels[actual])
    if predicted == actual:
        num_correct += 1

print(num_correct, "/", num_tests)
