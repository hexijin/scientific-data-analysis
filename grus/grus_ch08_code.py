from typing import Callable, List
from grus_ch04_code import Vector, dot, scalar_multiply, add

def sum_of_squares(v: Vector) -> float:
    """Computes the sum of squared elements in v"""
    return dot(v, v)

def gradient_step(v: Vector, gradient: Vector, scalar: float) -> Vector:
    """Moves step_size in the gradient direction from v"""
    assert len(v) == len(gradient)
    step = scalar_multiply(scalar, gradient)
    return add(v, step)

# PDF p. 141

def difference_quotient(f: Callable[[float], float],
                        x: float,
                        h: float) -> float:
    return (f(x + h) - f(x)) / h

def partial_difference_quotient(f: Callable[[Vector], float],
                                v: Vector,
                                i: int,
                                h: float) -> float:
    """Returns the ith partial difference quotient of f at v"""
    w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h

def estimate_gradient(f: Callable[[Vector], float],
                      v: Vector,
                      h: float) -> List[float]:
    return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]
