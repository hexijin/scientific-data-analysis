from typing import NamedTuple, Set, Tuple, Dict, Iterable
import math
from collections import defaultdict
import re

# PDF p. 232

def tokenize(text: str) -> Set[str]:
    text = text.lower()
    all_words = re.findall(r'[a-z0-9]+', text)
    return set(all_words)

# SPAM, SPAM, SPAM, SPAM

class Message(NamedTuple):
    text: str
    is_spam: bool

class NaiveBayesClassifier(object):

    def __init__(self, k: float = 0.5) -> None:
        self.k = k  # smoothing factor
        self.tokens: Set[str] = set()
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)
        self.spam_messages = 0
        self.ham_messages = 0

    def train(self, messages: Iterable[Message]) -> None:
        for message in messages:
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1
            # Increment word counts
            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1

    def _probabilities(self, token: str) -> Tuple[float, float]:
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]

        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)

        return p_token_spam, p_token_ham

    def predict(self, text: str) -> float:
        text_tokens = tokenize(text)
        log_prob_if_spam = 0.0
        log_prob_if_ham = 0.0

        # Iterate through each word in our vocabulary
        for token in text_tokens:
            prob_if_spam, prob_if_ham = self._probabilities(token)

            # If *token* appears in the message,
            # add the log probability of seeing it
            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham += math.log(prob_if_ham)
            # Otherwise add the log probability of _not_ seeing it
            # which is log(1 - probability of seeing it)
            else:
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_ham += math.log(1.0 - prob_if_ham)

            prob_if_spam = math.exp(log_prob_if_spam)
            prob_if_ham = math.exp(log_prob_if_ham)

            return prob_if_spam / (prob_if_spam + prob_if_ham)
        return 1.0
