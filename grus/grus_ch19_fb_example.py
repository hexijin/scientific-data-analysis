from grus_ch18_code import BINARY_ENCODING_LENGTH, FIZZBUZZ_ENCODING_LENGTH

from grus_ch19_code import *
from grus_ch19_fb_common import fizz_buzz_accuracy, fizz_buzz_xs, fizz_buzz_ys
from grus_ch19_fb_common import training_loop, fizz_buzz_loss, fizz_buzz_optimizer

NUM_HIDDEN = 25
random.seed(0)

fizz_buzz_net = Sequential([
    Linear(input_dim=BINARY_ENCODING_LENGTH, output_dim=NUM_HIDDEN, init='uniform'),
    Tanh(),
    Linear(input_dim=NUM_HIDDEN, output_dim=FIZZBUZZ_ENCODING_LENGTH, init='uniform'),
    Sigmoid()
])

training_loop(1000, fizz_buzz_net, fizz_buzz_xs, fizz_buzz_ys,
              fizz_buzz_loss, fizz_buzz_optimizer)

print("test results", fizz_buzz_accuracy(1, 101, fizz_buzz_net))
