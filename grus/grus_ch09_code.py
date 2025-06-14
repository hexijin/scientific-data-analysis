from bs4 import BeautifulSoup

def paragraph_mentions(text: str, keyword: str) -> bool:
    """Returns True if a <p> inside the text mentions {keyword}"""
    soup = BeautifulSoup(text, "html5lib")
    paragraphs = [p.get_text() for p in soup('p')]

    return any(keyword.lower() in paragraph.lower() for paragraph in paragraphs
               for paragraph in paragraphs)
