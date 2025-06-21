# Explore One-Dimensional Data

from typing import List, Dict
from collections import Counter
import math

import matplotlib.pyplot as plt

def bucketize(point: float, bucket_size: float) -> float:
    return bucket_size * math.floor(point / bucket_size)

def make_histogram(points: List[float], bucket_size: float) -> Dict[str, float]:
    return Counter(bucketize(point, bucket_size) for point in points)

def plot_histogram(points: List[float], bucket_size: float, title: str = ""):
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
    plt.title(title)


import random
from Chapter_6_Probability import inverse_normal_cdf

random.seed(0)

uniform = [200 * random.random() - 100 for _ in range(10000)]

normal = [57 * inverse_normal_cdf(random.random()) for _ in range(10000)]

plot_histogram(uniform, 10, "Uniform Histogram")
plot_histogram(normal, 10, "Normal Histogram")


# Two Dimensions
def random_normal() -> float:
    return inverse_normal_cdf(random.random())

xs = [random_normal() for _ in range(10000)]
ys1 = [ x + random_normal() / 2 for x in xs]
ys2 = [-x + random_normal() / 2 for x in xs]

plt.scatter(xs, ys1, marker='.', color='black', label='ys1')
plt.scatter(xs, ys2, marker='.', color='gray', label='ys2')
plt.xlabel('xs')
plt.ylabel('ys')
plt.legend(loc=9)
plt.title("Very Different Joint Distributions")
plt.show()


from Chapter_5_Statistics import correlation

print(correlation(xs, ys1))
print(correlation(xs, ys2))

from Chapter_4_Linear_Algebra import Matrix, Vector, make_matrix

def correlation_matrix(data_: List[Vector]) -> Matrix:
    def correlation_mn(m: int, n: int) -> float:
        return correlation(data_[m], data_[n])

    return make_matrix(len(data_), len(data_), correlation_mn)

import random

data = []

for _ in range(4):
    vector = [random.random() for _ in range(100)]
    data.append(vector)

corr_data = correlation_matrix(data)

num_vectors = len(corr_data)
fig, ax = plt.subplots(num_vectors, num_vectors)

for i in range(num_vectors):
    for j in range(num_vectors):

        if i != j: ax[i][j].scatter(corr_data[j], corr_data[i])

        else: ax[i][j].annotate("series" + str(i), (0.5, 0.5),
                                xycoords='axes fraction',
                                ha='center', va='center')
        if i < num_vectors - 1: ax[i][j].xaix.set_visible(False)
        if j > 0: ax[i][j].yaxis.set_visible(False)

ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
ax[0][0].set_ylim(ax[0][1].get_ylim())

plt.show()



# Using NamedTuples

import datetime
stock_price = {'closing_price': 102.06, 'date': datetime.date(2014, 8, 29), 'symbol': 'AAPL'}

prices: Dict[datetime.date, float] = {}

from collections import namedtuple

StockPrice = namedtuple('StockPrice', ['symbol', 'date', 'closing_price'])
price = StockPrice('MSFT', datetime.date(2018, 12, 14), 106.03)

assert price.symbol == 'MSFT'
assert price.closing_price == 106.03

from typing import NamedTuple

class StockPrice(NamedTuple):
    symbol: str
    date: datetime.date
    closing_price: float

    def is_high_tech(self) -> bool:
        return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMZN', 'AAPL']

price = StockPrice('MSFT', datetime.date(2018, 12, 14), 106.03)

assert price.symbol == 'MSFT'
assert price.closing_price == 106.03
assert price.is_high_tech()
assert price.date == datetime.date(2018, 12, 14)



# Dataclasses

from dataclasses import dataclass

@dataclass
class StockPrice2:
    symbol: str
    date: datetime.date
    closing_price: float

    def is_high_tech(self) -> bool:
        return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMZN', 'AAPL']

price2 = StockPrice2('MSFT', datetime.date(2018, 12, 14), 106.03)

assert price2.symbol == 'MSFT'
assert price2.closing_price == 106.03
assert price2.is_high_tech()

price2.closing_price /= 2
assert price2.closing_price == 51.03

price2.cosing_price = 75



# Cleaning and Munging

from dateutil.parser import parse

def parse_row(row_: List[str]) -> StockPrice:
   symbol_, date_, closing_price_ = row_
   return StockPrice(symbol=symbol,
                      date=parse(date_).date(),
                      closing_price=float(closing_price_))

stock = parse_row(["MSFT", "2018-12-14", "106.03"])

assert stock.symbol == 'MSFT'
assert stock.date == datetime.date(2018, 12, 14)
assert stock.closing_price == 106.03


from typing import Optional
import re

def try_parse_row(row_t: List[str]) -> Optional[StockPrice]:
    symbol_t, date_t, closing_price_t = row_t

    if not re.match(r"^[A-Z]+$", symbol):
        return None

    try:
        date_t  = parse(date_t).date()
    except ValueError:
        return None

    try:
        closing_price_t = float(closing_price_t)
    except ValueError:
        return None

    return StockPrice(symbol_t, date_t, closing_price_t)

assert try_parse_row(["MSFT0", "2018-12-14", "106.03"]) is None
assert try_parse_row(["MSFT", "2018-12--14", "106.03"]) is None
assert try_parse_row(["MSFT", "2018-12-14", "x"]) is None

assert try_parse_row(["MSFT", "2018-12-14", "106.03"]) == stock


import csv

data: List[StockPrice] = []

with open("comma_delimited_stock_prices.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        maybe_stock = try_parse_row(row)
        if maybe_stock is None:
            print(f"skipping invalid row: {row}")
        else:
            data.append(maybe_stock)



# Manipulating Data

data = [
    StockPrice(symbol='MSFT',
               date=datetime.date(2018, 12, 14),
               closing_price=106.03),
]

max_aapl_price = max(stock_price.closing_price
                     for stock_price in data
                     if stock_price.symbol == 'AAPL')


from collections import defaultdict

max_prices: Dict[str, float] = defaultdict(lambda: float('-inf'))

for sp in data:
    symbol, closing_price = sp.symbol, sp.closing_price
    if closing_price > max_prices[symbol]:
        max_prices[symbol] = closing_price


from typing import List
from collections import defaultdict

prices: Dict[str, List[StockPrice]] = defaultdict(list)

for sp in data:
    prices[sp.symbol].append(sp)

prices = {symbol:sorted(symbol_prices)
          for symbol, symbol_prices in prices.items()}

def pct_change(yesterday: StockPrice, today: StockPrice) -> float:
    return today.closing_price - yesterday.closing_price - 1

class DailyChange(NamedTuple):
    symbol: str
    date: datetime.date
    pct_change: float

def day_over_day_changes(prices_: List[StockPrice]) -> List[DailyChange]:
    changes = []
    for yesterday, today in zip(prices_, prices_[1:]):
        changes.append(
            DailyChange(
                symbol=today.symbol,
                date=today.date,
                pct_change=pct_change(yesterday, today)
            )
        )
    return changes

all_changes = [
    change
    for symbol_prices in prices.values()
    for change in day_over_day_changes(symbol_prices)
]

max_change = max(all_changes, key=lambda change: change.pct_change)
assert max_change.symbol == 'AAPL'
assert max_change.date == datetime.date(1997, 8, 6)
assert 0.33 < max_change.pct_change < 0.34

min_change = min(all_changes, key=lambda change: change.pct_change)
assert min_change.symbol == 'AAPL'
assert min_change.date == datetime.date(2000, 9, 29)
assert -0.52 < min_change.pct_change < -0.51

changes_by_month: Dict[int, List[DailyChange]] = defaultdict(list)
for month in range(1, 13):
    changes_by_month[month] = []

for change in all_changes:
    changes_by_month[change.date.month].append(change)

avg_daily_change = {
    month: sum(change.pct_change for change in changes) / len(changes)
    for month, changes in changes_by_month.items()
}

assert avg_daily_change[10] == max(avg_daily_change.values())

# Rescaling

from Chapter_4_Linear_Algebra import distance

a_to_b = distance([63, 150], [67, 160])
a_to_c = distance([63, 150], [70, 171])
b_to_c = distance([67, 160], [70, 171])

from typing import Tuple

from Chapter_4_Linear_Algebra import vector_mean
from Chapter_5_Statistics import standard_deviation

def scale(data_s: List[Vector]) -> Tuple[Vector, Vector]:
    dim = (data_s[0])

    means_s = vector_mean(data_s)
    stdevs_s = [standard_deviation([vector_s[items] for vector_s in data_s]) for items in range(dim)]

    return means_s, stdevs_s

vectors = [[-3, -1, 1], [-1, 0, 1], [1, 1, 1]]
means, stdevs = scale(vectors)
assert means == [-1, 0, 1]
assert stdevs == [2, 1, 0]

def rescale(data_r: List[Vector]) -> Tuple[Vector, Vector]:
    dim = len(data_r[0])
    means_r, stdevs_r = scale(data_r)

    rescaled = [v[:] for v in data_r]

    for v in rescaled:
        for n in range(dim):
            if stdevs_r[n] > 0:
                v[n] = (v[n] - means_r[n]) / stdevs_r[n]

means, stdevs = scale(rescale(vectors))
assert means == [0, 0, 1]
assert stdevs == [1, 1, 0]



# An Aside: tqdm

import tqdm

for i in tqdm.tqdm(range(100)):
    _ = [random.random() for _ in range(1000000)]


from typing import List

def primes_up_to(n: int) -> List[int]:
    primes = [2]

    with tqdm.trange(3, n) as t:
        for i_t in t:
            i_is_prime = not any(i_t % p == 0 for p in primes)
            if i_is_prime:
                primes.append(i_t)

                t.set_description(f"{len(primes)} primes")

    return primes

my_primes = primes_up_to(100_000)



# Dimensionality Reduction
"""principal component analysis"""

from Chapter_4_Linear_Algebra import subtract

def de_mean(data_dm: List[Vector]) -> List[Vector]:
    mean = vector_mean(data_dm)
    result = []
    for vector_dm in data_dm:
        result.append(subtract(vector_dm, mean))
    return result

_ = subtract

from Chapter_4_Linear_Algebra import magnitude

def direction(w: Vector) -> Vector:
    mag = magnitude(w)
    return [w_i / mag for w_i in w]


from Chapter_4_Linear_Algebra import dot

def directional_variance(data_dv: List[Vector], w: Vector) -> float:
    w_dir = direction(w)
    return sum(dot(v, w_dir) ** 2 for v in data_dv)

def directional_variance_gradient(data_dvg: List[Vector], w: Vector) -> Vector:
    w_dir = direction(w)
    return [sum(2 * dot(v, w_dir) * v[i_w] for v in data_dvg)
            for i_w in range(len(w))]


from Chapter_8_Gradient_Descent import gradient_step

def first_principal_component(data_fpc: List[Vector],
                              n: int = 100,
                              step_size: float = 0.1) -> Vector:
    guess = [1.0 for _ in data_fpc[0]]

    with tqdm.trange(n) as t:
        for _ in t:
            dv = directional_variance(data, guess)
            gradient = directional_variance_gradient(data, guess)
            guess = gradient_step(guess, gradient, step_size)
            t.set_description(f"dv: {dv:.3f}")
    return direction(guess)


from Chapter_4_Linear_Algebra import scalar_multiply

def project(v: Vector, w: Vector) -> Vector:
    projection_length = dot(v, w)
    return scalar_multiply(projection_length, w)


from Chapter_4_Linear_Algebra import subtract

def remove_projection_from_vector(v: Vector, w: Vector) -> Vector:
    return subtract(v, project(v, w))

def remove_projection(data_rp: List[Vector], w: Vector) -> List[Vector]:
    return [remove_projection_from_vector(v, w) for v in data_rp]


def pca(data_pca: List[Vector], num_components: int) -> List[Vector]:
    components: List[Vector] = []
    for _ in range(num_components):
        component = first_principal_component(data_pca)
        components.append(component)
        data_pca = remove_projection(data_pca, component)

    return components

def transform_vector(v: Vector, components: List[Vector]) -> Vector:
    return [dot(v, w) for w in components]

def transform(data_tf: List[Vector], components: List[Vector]) -> List[Vector]:
    return [transform_vector(v, components) for v in data_tf]




