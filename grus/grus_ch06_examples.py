import enum
import math
import random
import matplotlib.pyplot as plt
from collections import Counter

from grus_ch06_code import uniform_cdf, normal_pdf, normal_cdf, binomial

class Kid(enum.Enum):
    BOY = 0
    GIRL = 1

def random_kid() -> Kid:
    return random.choice([Kid.BOY, Kid.GIRL])

both_girls = 0
older_girl = 0
either_girl = 0

random.seed(0)

for _ in range(10000):
    younger = random_kid()
    older = random_kid()
    if older == Kid.GIRL:
        older_girl += 1
    if older == Kid.GIRL and younger == Kid.GIRL:
        both_girls += 1
    if older == Kid.GIRL or younger == Kid.GIRL:
        either_girl += 1

print("P(both | older):", both_girls / older_girl)
print("P(both | either):", both_girls / either_girl)

# PDF p. 114

xx = [x / 100 for x in range(-100, 201)]  # Every 0.01 from -1 to 2
yy = [uniform_cdf(x) for x in xx]

plt.plot(xx, yy)
plt.title('The Uniform CDF')
plt.xlim(-1, 2)  # Set horizontal axis limits
plt.show()

# PDF p. 116

xs = [x / 10 for x in range(-50, 51)]  # Every 0.1 from -5 to 5
plt.plot(xs, [normal_pdf(x, sigma=1) for x in xs], '-',
         label='mu=0, sigma=1')
plt.plot(xs, [normal_pdf(x, sigma=2) for x in xs], '--',
         label='mu=0, sigma=2')
plt.plot(xs, [normal_pdf(x, sigma=0.5) for x in xs], ':',
         label='mu=0, sigma=0.5')
plt.plot(xs, [normal_pdf(x, mu=-1) for x in xs], '-.',
         label='mu=-1, sigma=1')
plt.legend()
plt.title("Various Normal PDFs")
plt.show()

plt.plot(xs, [normal_cdf(x, sigma=1) for x in xs], '-',
         label='mu=0, sigma=1')
plt.plot(xs, [normal_cdf(x, sigma=2) for x in xs], '--',
         label='mu=0, sigma=2')
plt.plot(xs, [normal_cdf(x, sigma=0.5) for x in xs], ':',
         label='mu=0, sigma=0.5')
plt.plot(xs, [normal_cdf(x, mu=-1) for x in xs], '-.',
         label='mu=-1, sigma=1')
plt.legend()
plt.title("Various Normal CDFs")
plt.show()

# PDF p. 120

def binomial_histogram(p: float, n: int, num_points: int) -> None:
    """Picks points from binomial(n, p) and plots their histogram"""
    data = [binomial(n, p) for _ in range(num_points)]
    # Use a bar chart to show the actual binomial samples
    histogram = Counter(data)
    plt.bar([x-0.4 for x in histogram.keys()],
            [v / num_points for v in histogram.values()],
            0.8,
            color='0.75')
    mu = p * n
    sigma = math.sqrt(n * p * (1 - p))
    xxx = range(min(data), max(data) + 1)
    yyy = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma)
           for i in xxx]
    plt.plot(xxx, yyy)
    plt.title("Binomial Distribution vs. Normal Approximation")
    plt.show()

binomial_histogram(0.75, 100, 1000)
