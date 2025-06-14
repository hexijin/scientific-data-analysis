import random

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
