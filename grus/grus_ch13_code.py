from typing import Set
import re

# PDF p. 232

def tokenize(text: str) -> Set[str]:
    text = text.lower()
    all_words = re.findall(r'[a-z0-9]+', text)
    return set(all_words)
