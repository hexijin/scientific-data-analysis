from typing import List, Dict, NamedTuple
from collections import namedtuple
import random
import datetime
from matplotlib import pyplot as plt

from grus_ch05_code import correlation
from grus_ch06_code import inverse_normal_cdf
from grus_ch10_code import plot_histogram, random_normal

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
