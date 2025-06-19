"""
Naive Bayes: assuming that each probability is independent of each other
"""

# Implementation

from typing import Set


def tokenize(text_t: str) -> Set[str]:
    text_t = text_t.lower()
    all_words = re.findall("[a-z0-9]+", text_t)
    return set(all_words)

assert tokenize("Data Science is science") == {"data", "science", "is"}


from typing import NamedTuple

class Message(NamedTuple):
    text: str
    is_spam: bool

from typing import Tuple, Dict, Iterable
import math
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self, k: float = 0.5) -> None:
        self.k = k

        self.tokens: Set[str] = set()
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)
        self.spam_messages = self.ham_messages = 0

    def train(self, messages_t: Iterable[Message]) -> None:
        for message in messages_t:
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1

            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1

    def probabilities(self, token: str) -> Tuple[float, float]:
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]

        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)

        return p_token_spam, p_token_ham

    def predict(self, text_p: str) -> float:
        text_tokens = tokenize(text_p)
        log_prob_if_spam = log_prob_if_ham = 0,0

        for token in self.tokens:
            prob_if_spam_, prob_if_ham_ = self.probabilities(token)
            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam_)
                log_prob_if_ham += math.log(prob_if_ham_)
            else:
                log_prob_if_spam += math.log(1 - prob_if_spam_)
                log_prob_if_ham += math.log(1 - prob_if_ham_)

        prob_if_spam_ = math.exp(log_prob_if_spam)
        prob_if_ham_ = math.exp(log_prob_if_ham)
        return prob_if_spam_ / (prob_if_spam_ + prob_if_ham_)



# Testing our model

messages = [Message("spam rules", is_spam=True),
            Message("ham rules", is_spam=False),
            Message("hello ham", is_spam=False)]

model = NaiveBayesClassifier(k=0.5)
model.train(messages)

assert model.tokens == {"spam", "ham", "rules", "hello"}
assert model.spam_messages == 1
assert model.ham_messages == 2
assert model.token_spam_counts == {"spam": 1, "rules": 1}
assert model.token_ham_counts == {"ham": 2, "rules": 1, "hello": 1}


text = "hello spam"

prob_if_spam = [
    (1 + 0.5) / (1 + 2 * 0.5),
    1 - (0 + 0.5) / (1 + 2 * 0.5),
    1 - (1 + 0.5) / (1 + 2 * 0.5),
    (0 + 0.5) / (1 + 2 * 0.5)
]

prob_if_ham = [
    (0 + 0.5) / (2 + 2 * 0.5),
    1 - (2 + 0.5) / (2 + 2 * 0.5),
    1 - (1 + 0.5) / (2 + 2 * 0.5),
    (1 + 0.5) / (2 + 2 * 0.5)
]

p_if_spam = math.exp(sum(math.log(p) for p in prob_if_spam))
p_if_ham = math.exp(sum(math.log(p) for p in prob_if_ham))

assert model.predict(text) == p_if_spam / (p_if_spam + p_if_ham)



# Using Our Model

from io import BytesIO
import requests
import tarfile

BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"
FILES = ["20021010_easy_ham.tar.bz2",
         "20021010_hard_ham.tar.bz2"
         "20021010_ham.tar.bz2"]

OUTPUT_DIR = 'spam_Data'

for filename in FILES:
    content = requests.get(f"{BASE_URL}/{filename}").content
    fin = BytesIO(content)

    with tarfile.open(fileobj=fin, mode='r:bz2') as tf:
        tf.extractall(OUTPUT_DIR)


import glob, re
from typing import List

path = 'spam_data/*/*'

data: List[Message] = []

for filename in glob.glob(path):
    is_spam = "ham" not in filename

    with open(filename, errors='ignore') as email_file:
        for line in email_file:
            if line.startswith("Subject: "):
                subject = line.lstrip("Subject: ")
                data.append(Message(subject, is_spam))
                break


import random
from Chapter_11_Machine_Learning import split_data

random.seed(0)
train_messages, test_messages = split_data(data, 0.75)

model = NaiveBayesClassifier()
model.train(train_messages)


from collections import Counter

predictions = [(message, model.predict(message.text)) for message in test_messages]

confusion_matrix = Counter((message.is_spam, spam_probability > 0.5) for message, spam_probability in predictions)

print(confusion_matrix)

def p_spam_given_token(token: str, model_: NaiveBayesClassifier) -> float:
    prob_if_spam_p, prob_if_ham_p = model_.probabilities(token)

    return prob_if_spam_p / (prob_if_spam_p + prob_if_ham_p)

words = sorted(model.tokens, key=lambda t: p_spam_given_token(t, model))

print("spammiest_words", words[-10:])
print("hammiest_words", words[:10])


def drop_final_s(word):
    return re.sub("s$", "", word)

