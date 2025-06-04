import math
from typing import List

# PDF p. 82

Vector = List[float]

height_weight_age = [70,   # inches
                     170,  # pounds
                     40]   # years

grades = [95,  # exam1
          80,  # exam2
          75,  # exam3
          62]  # exam4

def add(v: Vector, w: Vector) -> Vector:
    """Adds components"""
    assert len(v) == len(w), "vectors must have same length"
    return [v_i + w_i for v_i, w_i in zip(v, w)]

assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]

def subtract(v: Vector, w: Vector) -> Vector:
    """Subtracts components"""
    assert len(v) == len(w), "vectors must have same length"
    return [v_i - w_i for v_i, w_i in zip(v, w)]

assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]

def vector_sum(vectors: List[Vector]) -> Vector:
    """Sums all the vectors in a list of vectors"""
    # Check that vectors is not empty
    assert vectors, "no vectors provided!"

    # Check the vectors are all the same size
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"

    # the ith element of the result is the sum of every vector[i]
    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]

assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]

def scalar_multiply(c: float, v: Vector) -> Vector:
    """Multiplies every element by c"""
    return [c * v_i for v_i in v]

assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]

def vector_mean(vectors: List[Vector]) -> Vector:
    """Computes the element-wise average of vectors"""
    n = len(vectors)
    return scalar_multiply(1 / n, vector_sum(vectors))

assert vector_mean([[1, 2, 3], [3, 4, 1]]) == [2, 3, 2]

def dot(v: Vector, w: Vector) -> float:
    """Computes the dot product of v and w"""
    assert len(v) == len(w), "vectors must have same length"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))

assert dot([1, 2, 3], [4, 5, 6]) == 32

def sum_of_squares(v: Vector) -> float:
    """Returns v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)

assert sum_of_squares([1, 2, 3]) == 14

def magnitude(v: Vector) -> float:
    """Returns the magnitude (or length) of v"""
    return math.sqrt(sum_of_squares(v))

assert magnitude([3, 4]) == 5

def squared_distance(v: Vector, w: Vector) -> float:
    return sum_of_squares(subtract(v, w))

def distance(v: Vector, w: Vector) -> float:
    return magnitude(subtract(v, w))

assert distance([0, 1, 2, 3, 4, 5], [1, 2, 5, 6, 6, 4]) == 5

# PDF p. 87

# Another type alias
Matrix = List[List[float]]

