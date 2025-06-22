from typing import List

Tensor = list

# PDF p. 307

def shape(tensor: Tensor) -> List[int]:
    sizes: List[int] = []
    while isinstance(tensor, Tensor):
        sizes.append(len(tensor))
        tensor = tensor[0]
    return sizes
