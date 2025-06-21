from grus_ch04_code import dot, Vector

# PDF p. 247

def predict(x: Vector, beta: Vector) -> float:
    return dot(x, beta)

# PDF. p. 249

def error(x: Vector, y: float, beta: Vector) -> float:
    return predict(x, beta) - y

def squared_error(x: Vector, y: float, beta: Vector) -> float:
    return error(x, y, beta) ** 2

def sqerror_gradient(x: Vector, y: float, beta: Vector) -> Vector:
    err = error(x, y, beta)
    return [2 * err * x_i for x_i in x]

# PDF p. 250
