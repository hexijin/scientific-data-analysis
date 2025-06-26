import tqdm


from grus_ch18_code import binary_encode, fizz_buzz_encode, argmax
from grus_ch19_code import Momentum, SSE, Layer, Sequential, Tensor, Loss, Optimizer

fizz_buzz_xs = [binary_encode(n) for n in range(101, 1024)]
fizz_buzz_ys = [fizz_buzz_encode(n) for n in range(101, 1024)]

NUM_HIDDEN = 25

fizz_buzz_optimizer = Momentum(learning_rate=0.1, momentum=0.9)
fizz_buzz_loss = SSE()

def fizz_buzz_accuracy(low: int, hi: int, net: Layer) -> float:
    num_correct = 0
    for n in range(low, hi):
        encoded_n = binary_encode(n)
        predicted_n = argmax(net.forward(encoded_n))
        actual_n = argmax(fizz_buzz_encode(n))
        if predicted_n == actual_n:
            num_correct += 1

    return num_correct / (hi - low)


def training_loop(iterations: int, net: Sequential, xs: Tensor, ys: Tensor,
                  loss: Loss, optimizer: Optimizer) -> None:
    with tqdm.trange(iterations) as t:
        for _ in t:
            epoch_loss = 0.0

        for x, y in zip(xs, ys):
            predicted = net.forward(x)
            epoch_loss += loss.loss(predicted, y)
            gradient = loss.gradient(predicted, y)
            net.backward(gradient)
            optimizer.step(net)

        accuracy = fizz_buzz_accuracy(101, 1024, net)
        t.set_description(f"fb_loss: {epoch_loss:.2f} acc: {accuracy:.2f}")
