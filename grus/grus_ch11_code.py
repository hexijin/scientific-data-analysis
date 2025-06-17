import random
from typing import TypeVar, List, Tuple

X = TypeVar('X')
Y = TypeVar('Y')

# PDF p. 207

def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    """Split data into fractions (prob, 1 - prob)"""
    data = data[:]          # Make a shallow copy because shuffle is going to modify the list
    random.shuffle(data)
    cut_point = int(len(data) * prob)
    return data[:cut_point], data[cut_point:]

def train_test_split(xs: List[X],
                     ys: List[Y],
                     test_pct: float) -> Tuple[List[X], List[X], List[Y], List[Y]]:
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)

    return ([xs[i] for i in train_idxs],
            [xs[i] for i in test_idxs],
            [ys[i] for i in train_idxs],
            [ys[i] for i in test_idxs])

def accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct / total

# PDF p. 211

def precision(tp: int, fp: int, _fn: int, _tn: int) -> float:
    return tp / (tp + fp)

def recall(tp: int, _fp: int, fn: int, _tn: int) -> float:
    return tp / (tp + fn)

def harmonic_mean(r1: float, r2: float) -> float:
    # The harmonic mean of r1 and r2 is by definition 1 / ((1/r1 + 1/r2) / 2)
    return 1 / ((1 / r1 + 1 / r2) / 2)

def f1_score(tp: int, fp: int, fn: int, tn: int) -> float:
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)
    return harmonic_mean(p, r)
