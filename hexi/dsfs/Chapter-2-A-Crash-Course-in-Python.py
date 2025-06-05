# Whitespace Formatting
from sphinx.addnodes import document

for i in [1, 2, 3, 4, 5]:
    print(i)
    for j in [1, 2, 3, 4, 5]:
        print(j)
        print(i+j)
    print(i)
print("done looping")

# Module
import re
my_regex = re.compile("[0-9]+", re.I)
import re as regex
my_regex = regex.compile("[0-9]+", regex.I)
from collections import defaultdict, Counter
lookup = defaultdict(int)
my_counter = Counter()

# Functions
def double(x):
    return x * 2
def apply_to_one(f):
    return f(1)
my_double = double
x = apply_to_one(my_double)
y = apply_to_one(lambda x: x + 4)
another_double = lambda x: 2 * x
def another_double(x):
    return 2 * x
def my_print(message = "my default message"):
    print(message)
my_print("hello")
my_print()
def full_name(first = "What's-his-name", last = "Something"):
    return first + " " + last
full_name("Hexi")
full_name("last=Jin")
full_name("Hexi", "Jin")

#Strings
single_quoted_string = 'data science'
double_quoted_string = "data science"
tab_string = "\t"
len(tab_string)
not_tab_string = r"\t"
len(not_tab_string)
multi_line_string = """This is the first line.
and this is the second line
and this is the third line"""
first_name = "Hexi"
last_name = "Jin"
full_name1 = first_name + " " + last_name
full_name2= "{0} {1}".format(first_name, last_name)
full_name3 = f"{first_name} {last_name}"

# Exceptions
try:
    print(0/0)
except ZeroDivisionError:
    print("cannot divide by 0")

# Lists
integer_list = [1, 2, 3]
heterogeneous_list = ["string", "0.1", True]
list_of_lists = [integer_list, heterogeneous_list, []]
list_length = len(integer_list)
list_sum = sum(integer_list)
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
zero = x[0]
one = x[1]
nine = x[-1]
eight = x[-2]
x[0] = -1
first_three = x[:3]
three_to_end = x[3:]
last_three = x[-3:]
without_first_and_last = x[1, -1]
copy_of_x = x[:]
every_third = x[::3]
five_to_three = x[5:2:-1]
1 in [1, 2, 3]
0 in [1, 2, 3]
x = [1, 2, 3]
x.extend([4, 5, 6])
x =[1, 2, 3]
y = x + [4, 5, 6]
x = [1, 2, 3]
x.append(0)
y = x[-1]
z = len(x)
x, y = [1, 2]
_, y = [1, 2]

# Tuples
my_list = [1, 2]
my_tuple = (1, 2)
other_tuple = 3, 4
my_list[1] = 3
try:
    my_tuple[1] = 3
except TypeError:
    print("Cannot modify a tuple")
def sum_and_product(x, y):
    return (x+y), (x * y)
sp = sum_and_product(2, 3)
s, p = sum_and_product(5, 10)
x, y = 1, 2
x, y = y, x

# Dictionaries
empty_dict = {}
empty_dict2 = dict()
grades = {"Joel": 80, "Tim": 95}
joels_grade = grades["Joel"]
try:
    kates_grade = grades["Kate"]
except KeyError:
    print("no grade for Kate")
joel_has_grade = "Joel" in grades
kate_has_grade = "Kate" in grades
joels_grade = grades.get("joel", 0)
kates_grade = grades.get("kate", 0)
no_ones_grade = grades.get("No One")
grades["Tim"] = 99
grades["Joel"] = 90
num_students = len(grades)
tweet = {"user" : "joelgrus",
         "text" : "data science is awesome",
         "retweet_count" : 100,
         "hashtags" : ["#data", "#science", "#datascience", "#awesome", "#yolo"]
}
tweet_keys = tweet.keys()
tweet_values = tweet.values()
tweet_items = tweet.items()
"user" in tweet_keys
"user" in tweet
"joelgrus" in tweet_values

# default dict
word_counts = {}
for word in document:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1
word_counts = {}
for word in document:
    try:
        word_counts[word] += 1
    except: KeyError
    word_counts[word] = 1
word_counts = {}
for word in document:
    previous_count = word_counts.get(word, 0)
    word_counts[word] = previous_count + 1
from collections import defaultdict
word_counts = defaultdict(int)
for word in document:
    word_counts[word] += 1
dd_list = defaultdict(list)
dd_list[2].append(1)
dd_dict = defaultdict(dict)
dd_dict["Joel"]["City"] = "Seattle"
dd_pair = defaultdict(lambda: [0, 0])
dd_pair[2][1] = 1

# Counter
from collections import Counter
c = Counter([0, 1, 2, 0])
word_counts = Counter(document)
for word, count in word_counts.most_common(10):
    print(word, count)

# Sets
primes_below_10 = {2, 3, 5, 7}
s = set()
s.add(1)
s.add(2)
s.add(2)
x = len(s)
y = 2 in s
z = 3 in s
stopwords_list = ["a", "an", "at"] + hundreds_of_other_words + ["yet", "you"]
"zip" in stopwords_list
stopwords_set = set(stopwords_list)
"zip" in stopwords_set
item_list= [1, 2, 3, 1, 2, 3]
num_items = len(item_list)
item_set = set(item_list)
num_distinct_items = len(item_set)
distinct_item_list = list(item_set)

# Control Flow
if 1 > 2:
    message = "if only 1 were greater than 2"
elif 1 > 3:
    message = "elif stands for 'else if'"
else:
    message = "when all else fails use else (if you want to)"
parity = "even" if x % 2 == 0 else "odd"

x = 0
while x < 10:
    print(f"{x} is less than 10")
    x += 1

for x in range (10):
    print (f"{x} is less than 10")

for x in range (10):
    if x == 3:
        continue
    if x == 5:
        break
    print(x)


# Truthiness
one_is_less_than_two = 1 < 2
true_equals_false = True == False

x = None
assert x == None
assert x is None

s = some_function_that_returns_a_string()
if s:
    first_char = s[0]
else:
    first_char = ""

first_char = s and s[0]

safe_x = x or 0
safe_x = x if x is not None else 0

all([True, 1, {3}])
all([True, 1, {}])
any([True, 1, {}])
all([])
any([])


# Sorting
x = [4, 1, 2, 3]
y = sorted(x)
x.sort()

x = sorted([-4, 1, -2, 3], key=abs, reverse=True)
wc_sorted(word_counts.items(), key=lambda word_and_count:word_and_count[1],
          reverse=True)

#List Comprehensions
even_numbers = [x for x in range(5) if x % 2 == 0]
squares = [x * x for x in range (5)]
even_squares = [x * x for x in even_numbers]

square_dict = {x: x * x for x in range(5)}
square_set = {x * x for x in [1, -1]}

zeros = [0 for _ in even_numbers]

pairs = [(x, y)
         for x in range(10)
         for y in range(10)]

increasing_pairs = [(x, y)
                    for x in range(10)
                    for y in range(x+1, 10)]


# Automated Testing and assert
assert 1 + 1 == 2
assert 1 + 1 == 2, "1 + 1 should equal 2 but didn't"

def smallest_item(xs):
    return min(xs)
assert smallest_item([10, 20, 5, 40]) == 5
assert smallest_item([1, 0, -1, 2]) == -1

def smallest_item(xs):
    assert xs, "empty list has no smallest item"
    return min(xs)

# Object-oriented programming
class CountingClicker:
    """A class can/should have a docstring, just like a function"""
    def __init__(self, count = 0):
        self.count = count
    clicker1 = CountingClicker()
    clicker2 = CountingClicker(100)
    clicker3 = CountingClicker(count=100)

    def __repr__(self):
        return f"CountingClicker(count={self.count})"

    def click(self, num_times = 1):
        self.count += num_times

    def read(self):
        return self.count

    def reset(self):
        self.count = 0

    clicker = CountingClicker()
    assert clicker.read() == 0, "clicker should start with 0"
    clicker.click()
    clicker.click()
    assert clicker.read() == 2, "after two clicks, clicker should have count 2"
    clicker.reset()
    assert clicker.read() == 0, "after reset, clicker should be back to 0"

    class NoResetClicker(CountingClicker):
        def reset(self):
            pass

    clicker2 = NoResetClicker()
    assert clicker2.read() == 0
    clicker2.click()
    assert clicker2.read() == 1
    clicker2.reset()
    assert clicker2.read() == 1, "reset shouldn't do anything"


# Iterables and Generators
def generate_range(n):
    i = 0,
    while i < n:
        yield i
        i += 1

for i in generate_range(10):
    print(f"i: {i}")

def natural_numbers():
    n = 1
    while True:
        yield n
        n += 1

even_below_20 = (i for i in generate_range(20) if i % 2 == 0)

data = natural_numbers()
evens = (x for x in data if x % 2 == 0)
even_sum = (x ** 2 for x in evens)
even_squares_ending_in_six = (x for x in even_squares if x % 10 == 6)

names = ["Alice", "Bob", "Charlie", "Debbie"]

#not Pythonic
for i in range(len(names))
    print(f"name {i} is {names[i]}")

#also not Pythonic
i = 0
for name in names:
    print(f"name {i} is {name}")
    i += 1

# Pythonic
for i, name in enumerate(names):
    print(f"name {i} is {name}")


# Randomness
import random
random.seed(10)

four_uniform_randoms = [random.random() for _ in range(4)]

random.randrange(10)
random.randrange(3,6)

up_to_ten = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
random.shuffle(up_to_ten)
print(up_to_ten)

my_number = random.choice(up_to_ten)

lottery_numbers = range(60)
winning_numbers = random.sample(lottery_numbers, 6)

four_with_replacement = [random.choice(range(10)) for _ in range(4)]
print(four_with_replacement)

# Regular Expressions
import re
re_examples= [
    not re.match("a", "cat"),
    re.search("a", "cat"),
    not re.search("c", "dog")
    3 == len(re.split("[ab]", "carbs")),
    "R-D-" == re.sub("[0-9]", "-", "R2D2")
]

assert all(re_examples), "all the regex examples should be True"


# zip and Argument Unpacking
list1 = ['a', 'b', 'c']
list2 = [1, 2, 3]
[pair for pair in zip(list1, list2)]

pairs = [('a', 1), ('b', 2), ('c', 3)]
letters, numbers = zip(*pairs)

def add(a, b): return a + b

add(1, 2)
try:
    add([1, 2])
except TypeError:
    print("add expects two inputs")
add(*[1, 2])

# args and kwargs
def doubler(f):
    def g(x):
        return 2 * f(x)
    return g

def f1(x):
    reurn x + 1
g = doubler(f1)
assert g(3) == 8, "(3+1) * 2 should equal 8"
assert g(-1) == 0, "(-1+1) * 2 should equal 0"

def f2(x, y):
    return x + y

g = doubler(f2)
try:
    g(1, 2)
    except TypeError:(
        print("as defined, g only takes one argument"))

def magic (*args, **kwargs):
    print("unnamed args:", args)
    print("keyword args:", kwargs)

magic(1, 2, key="word", key2="word2")

def other_way_magic(x, y, z):
    return x + y + z

x_y_list = [1, 2]
z_dict = {"z": 3}
assert other_way_magic(*x_y_list, **z_dict) == 6, "1 + 2 + 3 should be 6"

def doubler_correct(f):
    """works no matter what kind of inputs f expects"""
    def g(*args, **kwargs):
        """whatever arguments g is supplied, pass them through f"""
        return 2 * f(*args, **kwargs)
    return g

g = doubler_correct(f2)
assert g(1, 2) == 6, "doubler should work now"

# Type Annotations
def add(a, b):
    return a + b

assert add(10, 5) == 15, "+ is valid for numbers"
assert add([1, 2], [3]) == [1, 2, 3], "+ is valid for lists"
assert add("hi", "there") == "hi there", "+ is valid for strings"

try:
    add(10, "five")
except TypeError:
    print("cannot add an int to a string")

def add(a: int, b: int) -> int:
    return a + b
add (10, 5)
add ("hi", "there")

def total(xs: list) -> float:
    return sum(total)

from typing import List

def total(xs: List[float]) -> float:
    return sum(total)

from typing import Optional

values: List[int] = []
best_so_far: Optional[float] = None

from typing import Dict, Iterable, Tuple

counts: Dict[str, int]

lazy = False
if lazy:
    evens: Iterable[int] = (x for x in range(10) if x % 2 == 0)
else:
    evens = [0, 2, 4, 6, 8]

triple: Tuple[int, float, int] = (10, 2.3, 5)

from typing import Callable

def twice(repeater: Callable[[str, int], str], s: str) -> str:
    return repeater(s, 2)

def comma_repeater(s: str, n: int) -> str:
    n_copies = [s for  _ in range(n)]
    return ','.join(n_copies)

assert twice (comma_repeater, "type hints") == "type hints, type hints"

Number = int
Numbers  = List[Number]

def total(xs: Numbers) -> Number:
    return sum(xs)