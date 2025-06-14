
from bs4 import BeautifulSoup
import requests
import re
from typing import Dict, Set
from grus_ch09_code import paragraph_mentions

# PDF p. 159

url = "https://raw.githubusercontent.com/joelgrus/data/master/getting-data.html"

html = requests.get(url).text
soup = BeautifulSoup(html, "html5lib")

first_paragraph = soup.find('p')
first_paragraph_words = soup.p.text.split()

first_paragraph_id = soup.p['id']
first_paragraph_id2 = soup.p.get('id')

all_paragraphs = soup.find_all('p')
paragraphs_with_ids = [p for p in soup('p') if p.get('id')]

important_paragraphs = soup('p', {'class': 'important'})
important_paragraphs2 = soup('p', 'important')
important_paragraphs3 = [p for p in soup('p') if 'important' in p.get('class', [])]

# Warning: will return the same <span> multiple times
# if it sits inside multiple divs.
# Be more clever if that is the case.
spans_inside_divs = [span for div in soup('div') for span in div('span')]

url = "https://www.house.gov/representatives"
text = requests.get(url).text
soup = BeautifulSoup(text, "html5lib")

all_urls = [a['href'] for a in soup('a') if a.has_attr('href')]

print(len(all_urls))  # Currently 967, way too many

# Must start with http:// or https://
# Must end with .house.gov or .house.gov/
regex = r"^https?://.*\.house\.gov/?$"

# Let's write some tests
assert re.match(regex, "http://joel.house.gov")
assert re.match(regex, "https://joel.house.gov")
assert re.match(regex, "http://joel.house.gov/")
assert re.match(regex, "https://joel.house.gov/")
assert not re.match(regex, "http://joel.house.com")
assert not re.match(regex, "http://joel.house.gov/biography")

# And now apply
good_urls = [url for url in all_urls if re.match(regex, url)]

print(len(good_urls))  # Currently 862

good_urls = list(set(good_urls))

print(len(good_urls))  # 437

text = requests.get('https://jayapal.house.gov').text
soup = BeautifulSoup(text, "html5lib")

links = {a['href'] for a in soup('a') if 'press releases' in a.text.lower()}

print(links)  # https://jayapal.house.gov/category/news/ and https://jayapal.house.gov/category/press-releases/

press_releases: Dict[str, Set[str]] = {}

limitation = 40
for house_url in good_urls[:limitation]:
    html = requests.get(house_url).text
    soup = BeautifulSoup(html, "html5lib")
    pr_links = {a['href'] for a in soup('a') if 'press releases' in a.text.lower()}
    press_releases[house_url] = pr_links

# PDF p. 164

text = """<body><h1>Facebook</h1><p>Twitter</p>"""
assert paragraph_mentions(text, "twitter")
assert not paragraph_mentions(text, "facebook")

for house_url, pr_links in press_releases.items():
    for pr_link in pr_links:
        url = f"{house_url}/{pr_link}"
        print(f'Checking {url} for mentions of "data"')
        text = requests.get(url).text
        if paragraph_mentions(text, "data"):
            print(f'Found "data" at {url}')
            break

# PDF p. 165 JSON and XML

# SKIPPED! But it is good to know how to get JSON back from an API. It is a common
# way for computers to share data with each other.
