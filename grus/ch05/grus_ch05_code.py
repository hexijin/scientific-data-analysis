# To get the following import to work, you have to add a
# "Content Root" to the ch05 project that has the directory
# where the ch04 project code is
from grus_ch04_code import *

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)
