import enum
import random
import matplotlib.pyplot as plt

from grus_ch06_code import uniform_cdf

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
