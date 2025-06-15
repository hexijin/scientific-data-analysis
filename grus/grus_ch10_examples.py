from typing import List, NamedTuple  # , Dict (not being used except in commented out code)
# from collections import namedtuple (not being used except in commented out code)
import random
import datetime
from matplotlib import pyplot as plt
from dataclasses import dataclass
from dateutil.parser import parse
from typing import Optional
import re
import csv

from grus_ch04_code import distance
from grus_ch05_code import correlation
from grus_ch06_code import inverse_normal_cdf
from grus_ch10_code import plot_histogram, random_normal, scale, rescale

# PDF p. 175

random.seed(0)

# uniform between -100 and 100
uniform = [200 * random.random() - 100 for _ in range(10000)]

# normal distribution with mean 0, standard deviation 57
normal = [57 * inverse_normal_cdf(random.random()) for _ in range(10000)]

plot_histogram(uniform, 10, "Uniform Histogram")
plot_histogram(normal, 10, "Normal Histogram")

# PDF p. 177

random.seed(0)
xs = [random_normal() for _ in range(1000)]
ys1 = [x + random_normal() / 2 for x in xs]
ys2 = [-x + random_normal() / 2 for x in xs]

plot_histogram(ys1, 0.01, "Normal Histogram for ys1 with bucket_size=0.01")
plot_histogram(ys2, 0.01, "Normal Histogram for ys2 with bucket_size=0.01")

plt.scatter(xs, ys1, marker='.', color='black', label='ys1')
plt.scatter(xs, ys2, marker='.', color='gray', label='ys2')
plt.xlabel('xs')
plt.ylabel('ys')
plt.legend(loc=9)
plt.title('Very Different Joint Distributions')
plt.show()

print(correlation(xs, ys1))    # about 0.9
print(correlation(xs, ys2))    # about -0.9

# PDF p. 179

# Grus did not put this next chunk of code into the text, so I had to raid it from
# his GitHub version of the book code:

num_points = 100

def random_row() -> List[float]:
    row = [0.0, 0, 0, 0]
    row[0] = random_normal()
    row[1] = -5 * row[0] + random_normal()
    row[2] = row[0] + row[1] + 5 * random_normal()
    row[3] = 6 if row[2] > -2 else 0
    return row

random.seed(0)
# each row has 4 points, but really we want the columns
corr_rows = [random_row() for _ in range(num_points)]
corr_data = [list(col) for col in zip(*corr_rows)]

# Now we are back to the code that is reproduced in the book:

# corr_data is a list of four 100-d vectors
num_vectors = len(corr_data)

fig, ax = plt.subplots(num_vectors, num_vectors)

for i in range(num_vectors):
    for j in range(num_vectors):
        # Scatter column_i on the y-axis vs. column_j on the x-axis
        if i != j:
            ax[i][j].scatter(corr_data[i], corr_data[j])
        else:
            ax[i][j].annotate("series " + str(i), (0.5, 0.5), xycoords='axes fraction', ha='center', va='center')
        # Hide the axis labels except the left and bottom plots
        if i < num_vectors - 1:
            ax[i][j].xaxis.set_visible(False)
            if j > 0:
                ax[i][j].yaxis.set_visible(False)

ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
ax[0][0].set_ylim(ax[0][1].get_ylim())

plt.show()

# PDF p. 180

# UNDESIRABLE APPROACH 1 -- use a dict

# dictionary with a typo
# stock_price = {
#     'closing_price': 102.06,
#     'date': datetime.date(2014, 8, 29),
#     'symbol': 'AAPL', 'cosing_price': 103.06}
#
# prices: Dict[datetime.date, float] = {}

# BETTER BUT NOT THE MOST DESIRABLE APPROACH 2 -- use a namedtuple

# StockPrice = namedtuple('StockPrice', ['symbol', 'date', 'closing_price'])
# price = StockPrice('MSFT', datetime.date(2018, 12, 14), 106.03)
#
# assert price.symbol == 'MSFT'
# assert price.date == datetime.date(2018, 12, 14)
# assert price.closing_price == 106.03

# MOST DESIRABLE APPROACH 3 -- use NamedTuple

class StockPrice(NamedTuple):
    symbol: str
    date: datetime.date
    closing_price: float

    def is_high_tech(self) -> bool:
        return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMZN', 'AAPL']

stock_price = StockPrice('MSFT', datetime.date(2018, 12, 14), 106.03)

assert stock_price.symbol == 'MSFT'
assert stock_price.date == datetime.date(2018, 12, 14)
assert stock_price.closing_price == 106.03

# PDF p. 183

# YET ANOTHER APPROACH TO BE AWARE OF AND GRUS WILL NOT BE USING -- used dataclasses

@dataclass
class StockPrice2:
    symbol: str
    date: datetime.date
    closing_price: float

    def is_high_tech(self) -> bool:
        return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMZN', 'AAPL']

price2 = StockPrice2('MSFT', datetime.date(2018, 12, 14), 106.04)

assert price2.symbol == 'MSFT'
assert price2.date == datetime.date(2018, 12, 14)
assert price2.closing_price == 106.04
assert price2.is_high_tech()

# stock split

price2.closing_price /= 2
assert price2.closing_price == 53.02

# typo
price2.cosing_Price = 75

def parse_row(row: List[str]) -> StockPrice:
    symbol, date, closing_price = row
    return StockPrice(symbol=symbol,
                      date=parse(date).date(),
                      closing_price=float(closing_price))

# Now test our function

stock = parse_row(["MSFT", "2018-12-14", "106.03"])

assert stock.symbol == "MSFT"
assert stock.date == datetime.date(2018, 12, 14)
assert stock.closing_price == 106.03

def try_parse_row(row: List[str]) -> Optional[StockPrice]:
    symbol, date_, closing_price_ = row

    # Stock symbol should be all capital letters
    if not re.match(r"^[A-Z]+$", symbol):
        return None

    try:
        date = parse(date_).date()
    except ValueError:
        return None

    try:
        closing_price = float(closing_price_)
    except ValueError:
        return None

    return StockPrice(symbol, date, closing_price)

# Should return None due to errors
assert try_parse_row(["MSFT0", "2018-12-14", "106.03"]) is None
assert try_parse_row(["MSFT", "2018-12--14", "106.03"]) is None
assert try_parse_row(["MSFT", "2018-12-14", "x"]) is None

# But should return same as before if data is good
assert try_parse_row(["MSFT", "2018-12-14", "106.03"]) == stock

# For example, if we have comma-delimited stock prices with bad data:

data: List[StockPrice] = []

with open("../grus_resources/brief_comma_delimited_stock_prices.csv") as f:
    reader = csv.reader(f)
    for r in reader:
        maybe_stock = try_parse_row(r)
        if maybe_stock is None:
            print(f"skipping invalid row: {r}")
        else:
            data.append(maybe_stock)

assert len(data) == 5

# I SKIPPED THE MANIPULATING DATA SECTION

# PDF p. 190

# a_to_b = distance([63, 150], [67, 160])      # 10.77
# a_to_c = distance([63, 150], [70, 171])      # 22.14
# b_to_c = distance([67, 160], [70, 171])      # 11.40

a_to_b = distance([160, 150], [170.2, 160])      # 14.28
a_to_c = distance([160, 150], [177.8, 171])      # 27.53
b_to_c = distance([170.2, 160], [177.8, 171])    # 13.37

# PDF p. 191

vectors = [[-3, -1, 1], [-1, 0, 1], [1, 1, 1]]
means, stdevs = scale(vectors)

assert means == [-1, 0, 1]
assert stdevs == [2, 1, 0]

means, stdevs = scale(rescale(vectors))
assert means == [0, 0, 1]
assert stdevs == [1, 1, 0]

# PDF p. 196
