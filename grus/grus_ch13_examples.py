import glob
import random
from typing import List
from collections import Counter

# Commented out imports are for retrieving the SpamAssassin corpus
# from io import BytesIO
# import requests
# import tarfile

from grus_ch13_code import *
from grus_ch11_code import split_data

# PDF p. 232

assert tokenize("Data Science is science.") == {"data", "science", "is"}

# PDF p. 234

# TEST, TEST, TEST, TEST

test_messages = [Message("spam rules", is_spam=True),
                 Message("ham rules", is_spam=False),
                 Message("hello ham", is_spam=False)]

test_model = NaiveBayesClassifier(k=0.5)
test_model.train(test_messages)

assert test_model.tokens == {"spam", "ham", "rules", "hello"}
assert test_model.spam_messages == 1
assert test_model.ham_messages == 2
assert test_model.token_spam_counts == {"spam": 1, "rules": 1}
assert test_model.token_ham_counts == {"ham": 2, "rules": 1, "hello": 1}

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

# Commented out code is for retrieving the SpamAssassin corpus
# BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"
# FILES = [
#     "20021010_easy_ham.tar.bz2",
#     "20021010_hard_ham.tar.bz2",
#     "20021010_spam.tar.bz2"
# ]
#
# OUTPUT_DIR = '../grus_resources/spam_data'
#
# # PDF pp. 235-6
#
# for filename in FILES:
#     # Use request to get the file contents at each URL.
#     content = requests.get(f'{BASE_URL}/{filename}').content
#     # Wrap the in-memory bytes so we can use them as a "file."
#     fin = BytesIO(content)
#     # And extract all the files to teh specified output dir.
#     with tarfile.open(fileobj=fin, mode='r:bz2') as tf:
#         tf.extractall(OUTPUT_DIR)

# PDF p. 236

path = "../grus_resources/spam_data/*/*"

data: List[Message] = []

for filename in glob.glob(path):
    is_spam = "ham" not in filename
    with open(filename, errors="ignore") as email_file:
        for line in email_file:
            if line.startswith("Subject:"):
                subject = line.strip("Subject: ")
                data.append(Message(subject, is_spam))
                break


random.seed(0)

train_messages, test_messages = split_data(data, 0.75)

test_model = NaiveBayesClassifier()
test_model.train(train_messages)

predictions = [(message, test_model.predict(message.text)) for message in test_messages]

confusion_matrix = Counter((message.is_spam, spam_probability > 0.5)
                           for message, spam_probability in predictions)

print(confusion_matrix)

def p_spam_given_token(token: str, model: NaiveBayesClassifier) -> float:
    # We probably shouldn't call private methods, but it's for a good cause.
    # noinspection PyProtectedMember
    prob_if_spam, prob_if_ham = model._probabilities(token)  # type: ignore
    return prob_if_spam / (prob_if_spam + prob_if_ham)

words = sorted(test_model.tokens, key=lambda t: p_spam_given_token(t, test_model))

print("spammiest words", words[-10:])
print("hammiest words", words[:10])
