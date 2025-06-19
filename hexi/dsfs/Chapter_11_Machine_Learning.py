"""Machine Learning: using existing data to develop models that we can use to predict various outcomes for new data"""

# Overfitting and underfitting
"""Overfitting: a model that performs well on the data you train it on but generalizes poorly on any new data
Underfitting: a model that performs poorly even on the data you train it on"""

"""Approach 1: split the dataset--some used to train the model, and some to test the model"""

import random
from typing import TypeVar, List, Tuple
X = TypeVar('X')

def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    data = data[:]
    random.shuffle(data)
    cut = int(len(data) * prob)
    return data[:cut], data[cut:]

data_ = [n for n in range(1000)]
train, test = split_data(data_, 0.75)

assert len(train) == 750
assert len(test) == 250

assert sorted(train + test) == data_


Y = TypeVar('Y')

def train_test_split(xs: List[X],
                     ys: List[Y],
                     test_pct: float)-> Tuple[List[X], List[X], List[Y], List[Y]]:
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)

    return ([xs[i] for i in train_idxs],
            [xs[i] for i in test_idxs],
            [ys[i] for i in train_idxs],
            [ys[i] for i in test_idxs],)

xs = [x for x in range(1000)]
ys = [2 * x for x in xs]
x_train, x_test, y_train, y_test = train_test_split(xs, ys, 0.25)

assert len(x_train) == len(y_train) == 750
assert len(x_test) == len(y_test) == 250

assert all(y == 2 * x for x, y in zip(x_test, y_test))
assert all(y == 2 * x for x, y in zip(x_train, y_train))



# Correctness
"""False positive (type 1 error): when predicted positive, but in fact negative
False negative (type 2 error): when predicted negative, but in fact positive"""

def accuracy(tp: int, fp: int, tn: int, fn: int) -> float:
    correct = tp + tn
    total = tp + fp + tn + fn
    return correct / total

def precision(tp: int, fp: int, tn: int, fn: int) -> float:
    return tp / (tp + fp)

def recall(tp: int, fp: int, tn: int, fn: int) -> float:
    return tp / (tp + fn)

def f1_score(tp: int, fp: int, tn: int, fn: int) -> float:
    p = precision(tp, fp, tn, fn)
    r = recall(tp, fp, tn, fn)
    return 2 * p * r / (p + r)
"""harmonic mean"""


# The Bias-Variance Tradeoff
"""
Underfitting--high bias (how well it fits the training set) and low variance (when given two randomly chose training sets)--add more feautres
Overfitting--low bias and high variance--remove features or add more data
"""

# Feature Extraction and Selection
"""
1) yes-or-no features: naive Bayes classifier
2) numeric features: regression models
3) numeric or categorial data: decision tress
"""
