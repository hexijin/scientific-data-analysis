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
