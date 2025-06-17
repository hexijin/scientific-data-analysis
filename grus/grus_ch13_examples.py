from io import BytesIO
import requests
import tarfile

from grus_ch13_code import *

# PDF p. 232

assert tokenize("Data Science is science.") == {"data", "science", "is"}

# PDF p. 234

# TEST, TEST, TEST, TEST

test_messages = [Message("spam rules", is_spam=True),
                 Message("ham rules", is_spam=False),
                 Message("hello ham", is_spam=False)]

model = NaiveBayesClassifier(k=0.5)
model.train(test_messages)

assert model.tokens == {"spam", "ham", "rules", "hello"}
assert model.spam_messages == 1
assert model.ham_messages == 2
assert model.token_spam_counts == {"spam": 1, "rules": 1}
assert model.token_ham_counts == {"ham": 2, "rules": 1, "hello": 1}

test_text = "hello spam"

spam_bayesian_denominator = 1 + 2 * 0.5

probs_if_spam = [
    (1 + 0.5) / spam_bayesian_denominator,
    1 - (0 + 0.5) / spam_bayesian_denominator,
    1 - (1 + 0.5) / spam_bayesian_denominator,
    (0 + 0.5) / spam_bayesian_denominator
]

ham_bayesian_denominator = 2 + 2 * 0.5

probs_if_ham = [
    (0 + 0.5) / ham_bayesian_denominator,
    1 - (2 + 0.5) / ham_bayesian_denominator,
    1 - (1 + 0.5) / ham_bayesian_denominator,
    (1 + 0.5) / ham_bayesian_denominator
]

BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"
FILES = [
    "20021010_easy_ham.tar.bz2",
    "20021010_hard_ham.tar.bz2",
    "20021010_spam.tar.bz2"
]

OUTPUT_DIR = 'spam_data'

# PDF pp. 235-6

for filename in FILES:
    # Use request to get the file contents at each URL.
    content = requests.get(f'{BASE_URL}/{filename}').content
    assert "won't work" == content
