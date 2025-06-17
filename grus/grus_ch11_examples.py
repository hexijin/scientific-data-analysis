from grus_ch11_code import *

# PDF p. 207

data = [n for n in range(1000)]
train, test = split_data(data, 0.75)

assert len(train) == 750
assert len(test) == 250
assert sorted(train + test) == data

xs = [x for x in range(1000)]
ys = [2 * x for x in xs]

x_train, x_test, y_train, y_test = train_test_split(xs, ys, 0.25)

assert all(y == 2 * x for x, y, in zip(x_train, y_train))
assert all(y == 2 * x for x, y, in zip(x_test, y_test))

# Then some code like this would use the split to train and evaluate
# model = SomeKindOfModel()
# x_train, x_test, y_train, y_test = train_test_split(xs, ys, 0.33)
# model.train(x_train, y_train)
# evaluation = model.evaluate(x_test, y_test)

assert accuracy(70, 4930, 13930, 981070) == 0.98114

# PDF p. 211
