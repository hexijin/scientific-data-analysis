from grus_ch18_code import BINARY_ENCODING_LENGTH, FIZZBUZZ_ENCODING_LENGTH

from grus_ch19_code import *
from grus_ch19_fb_common import NUM_HIDDEN, fizz_buzz_accuracy, training_loop
from grus_ch19_fb_common import fizz_buzz_xs, fizz_buzz_ys, fizz_buzz_loss, fizz_buzz_optimizer

random.seed(0)

fb_sm_net = Sequential([
    Linear(input_dim=BINARY_ENCODING_LENGTH, output_dim=NUM_HIDDEN, init='uniform'),
    Tanh(),
    Linear(input_dim=NUM_HIDDEN, output_dim=FIZZBUZZ_ENCODING_LENGTH, init='uniform')
])

fb_sm_loss = SoftmaxCrossEntropy()

training_loop(500, fb_sm_net, fizz_buzz_xs, fizz_buzz_ys,
              fizz_buzz_loss, fizz_buzz_optimizer)

print("test results", fizz_buzz_accuracy(1, 101, fb_sm_net))
