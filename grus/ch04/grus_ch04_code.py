import math
from typing import List, Tuple, Callable

# PDF p. 82

# A type alias
Vector = List[float]

def add(v: Vector, w: Vector) -> Vector:
    """Adds components"""
    assert len(v) == len(w), "vectors must have same length"
    return [v_i + w_i for v_i, w_i in zip(v, w)]

def subtract(v: Vector, w: Vector) -> Vector:
    """Subtracts components"""
    assert len(v) == len(w), "vectors must have same length"
    return [v_i - w_i for v_i, w_i in zip(v, w)]

def vector_sum(vectors: List[Vector]) -> Vector:
    """Sums all the vectors in a list of vectors"""
    # Check that vectors is not empty
    assert vectors, "no vectors provided!"

    # Check the vectors are all the same size
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"
    # the ith element of the result is the sum of every vector[i]
    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]

def scalar_multiply(c: float, v: Vector) -> Vector:
    """Multiplies every element by c"""
    return [c * v_i for v_i in v]

def vector_mean(vectors: List[Vector]) -> Vector:
    """Computes the element-wise average of vectors"""
    n = len(vectors)
    return scalar_multiply(1 / n, vector_sum(vectors))

def dot(v: Vector, w: Vector) -> float:
    """Computes the dot product of v and w"""
    assert len(v) == len(w), "vectors must have same length"
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def sum_of_squares(v: Vector) -> float:
    """Returns v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)

assert sum_of_squares([1, 2, 3]) == 14

def magnitude(v: Vector) -> float:
    """Returns the magnitude (or length) of v"""
    return math.sqrt(sum_of_squares(v))

def squared_distance(v: Vector, w: Vector) -> float:
    return sum_of_squares(subtract(v, w))

def distance(v: Vector, w: Vector) -> float:
    return magnitude(subtract(v, w))

# PDF p. 87

# Another type alias
Matrix = List[List[float]]

def shape(m: Matrix) -> Tuple[int, int]:
    """Returns (# of rows, # of columns)"""
    num_rows = len(m)
    num_cols = len(m[0] if m else 0)   # of elements in first row
    return num_rows, num_cols

def get_row(m: Matrix, i: int) -> Vector:
    """Returns the row at index i"""
    return m[i]

def get_column(m: Matrix, j: int) -> Vector:
    """Returns the column at index j"""
    return [m_i[j] for m_i in m]

def make_matrix(num_rows: int,
                num_cols: int,
                entry_fn: Callable[[int, int], float]) -> Matrix:
    """
    Returns a matrix of size (num_rows, num_cols)
    whose (i, j)-th entry is entry_fn(i, j)
    """
    return [[entry_fn(i, j)
             for j in range(num_cols)]
            for i in range(num_rows)]

# PDF p. 89

def identity_matrix(n: int) -> Matrix:
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)
