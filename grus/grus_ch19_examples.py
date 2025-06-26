import tqdm

from grus_ch18_examples import binary_encode, argmax
from grus_ch18_code import fizz_buzz_encode
from grus_ch18_code import BINARY_ENCODING_LENGTH, FIZZBUZZ_ENCODING_LENGTH

from grus_ch19_code import *

# PDF p. 307

# The following is unfortunately possible and has shape 2x2
bad_tensor: Tensor = [[1.0, 2.0], 3.0]

# PDF p. 320

training_xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
training_ys = [[0.0], [1.0], [1.0], [0.0]]

random.seed(0)

xor_net = Sequential([
    Linear(input_dim=2, output_dim=2),
    Sigmoid(),
    Linear(input_dim=2, output_dim=1),
    Sigmoid()
])

xor_optimizer = GradientDescent(learning_rate=0.01)
xor_loss = SSE()

with tqdm.trange(3000) as t:
    for epoch in t:
        epoch_loss = 0.0
        for x, y in zip(training_xs, training_ys):
            predicted = xor_net.forward(x)
            epoch_loss += xor_loss.loss(predicted, y)
            gradient = xor_loss.gradient(predicted, y)
            xor_net.backward(gradient)
            xor_optimizer.step(xor_net)
        t.set_description(f"xor loss: {epoch_loss:.3f}")

for param in xor_net.params():
    print(param)

fizz_buzz_xs = [binary_encode(n) for n in range(101, 1024)]
fizz_buzz_ys = [fizz_buzz_encode(n) for n in range(101, 1024)]

NUM_HIDDEN = 25
random.seed(0)

fizz_buzz_net = Sequential([
    Linear(input_dim=BINARY_ENCODING_LENGTH, output_dim=NUM_HIDDEN, init='uniform'),
    Tanh(),
    Linear(input_dim=NUM_HIDDEN, output_dim=FIZZBUZZ_ENCODING_LENGTH, init='uniform'),
    Sigmoid()
])

def fizz_buzz_accuracy(low: int, hi: int, net: Layer) -> float:
    num_correct = 0
    for n in range(low, hi):
        encoded_n = binary_encode(n)
        predicted_n = argmax(net.forward(encoded_n))
        actual_n = argmax(fizz_buzz_encode(n))
        if predicted_n == actual_n:
            num_correct += 1

    return num_correct / (hi - low)

fizz_buzz_optimizer = Momentum(learning_rate=0.1, momentum=0.9)
fizz_buzz_loss = SSE()

with tqdm.trange(1000) as t:
    for epoch in t:
        epoch_loss = 0.0

    for x, y in zip(fizz_buzz_xs, fizz_buzz_ys):
        predicted = fizz_buzz_net.forward(x)
        epoch_loss += fizz_buzz_loss.loss(predicted, y)
        gradient = fizz_buzz_loss.gradient(predicted, y)
        fizz_buzz_net.backward(gradient)
        fizz_buzz_optimizer.step(fizz_buzz_net)

    accuracy = fizz_buzz_accuracy(101, 1024, fizz_buzz_net)
    t.set_description(f"fb_loss: {epoch_loss:.2f} acc: {accuracy:.2f}")

print("test results", fizz_buzz_accuracy(1, 101, fizz_buzz_net))

# PDF p. 327
