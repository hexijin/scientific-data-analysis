from grus_ch15_code import error, squared_error

# PDF p. 249

test_x = [1, 2, 3]
test_y = 30
test_beta = [4, 4, 4]

assert error(test_x, test_y, test_beta) == -6
assert squared_error(test_x, test_y, test_beta) == 36

# PDF p. 250
