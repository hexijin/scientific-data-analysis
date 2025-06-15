# stdin and stdout
"""Connect a python script to files, pipelines, etc. in the terminal"""
import sys, re

regex = sys.argv[1]

for line in sys.stdin:
    if re.search(regex, line):
        sys.stdout.write(line)


import sys

count = 0
for line in sys.stdin:
    count += 1

print(count)


import sys
from collections import Counter

try:
    num_words = int(sys.argv[1])
except ValueError:
    print("usage: most_common_words.py num_words")
    sys.exit(1)

counter = Counter(word.lower() for line in sys.stdin
                  for word in line.strip().split()
                  if word)

for word, count in counter.most_common(num_words):
    sys.stdout.write(str(count))
    sys.stdout.write("\t")
    sys.stdout.write(word)
    sys.stdout.write("\n")

"""In terminal: type SomeFile.txt | python most_common_words.py 10"""



## Reading files

# The basics of text files

file_for_reading = open('reading_file.txt', 'r')
file_for_reading2 = open('reading_file.txt')

file_for_writing = open('writing_file.txt', 'w')

file_for_appending = open('appending_file.txt', 'a')

file_for_writing.close()

#with open(filename) as f:
# data = function_that_gets_data_from(f)

starts_with_hash = 0

with open('input.txt') as f:
    for line in f:
        if re.match("^#", line):
            starts_with_hash += 1


def get_domain(email_address: str) -> str:
    return email_address.lower().split('@')[-1]

assert get_domain('hexi.jinkx@deepsprings.edu') == 'deepsprings.edu'
assert get_domain('hexi.jinkx@gmail.com') == 'gmail.com'

from collections import Counter

with open('email_addresses.txt', 'r') as f:
    domain_counts = Counter(get_domain(line.strip()) for line in f if "@" in line)

# Delimited Files

import csv

def process(date, symbol, closing_price):
    print(f"{symbol} closed at ${closing_price} on {date}")

with open('tab_delimited_stock_prices.txt') as f:
    tab_reader = csv.reader(f, delimiter='\t')
    for row in tab_reader:
        date0 = row[0]
        symbol0 = row[1]
        closing_price0 = float(row[2])
        process(date0, symbol0, closing_price0)

with open('colon_delimited_stock_prices.txt') as f:
    colon_reader = csv.reader(f, delimiter=':')
    for dict_row in colon_reader:
        date1 = dict_row["date"]
        symbol1 = dict_row["symbol"]
        closing_price1 = float(dict_row["closing_price"])
        process(date1, symbol1, closing_price1)

todays_prices = {'AAPL': 90.91, "MSFT": 41.68, "FB": 64.5}
with open('comma_delimited_stock_prices.txt', 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    for stock, price in todays_prices.items():
        csv_writer.writerow([stock, price])



## Scraping the Web

# HTML and the parsing Thereof

from bs4 import BeautifulSoup
import requests

url = ("http://raw/githubusercontent.com/"
       "joelgrus/data/master/getting-data.html")
html = requests.get(url).text
soup = BeautifulSoup(html, 'html5lib')

first_paragraph = soup.find('p')

first_paragraph_text = soup.p.text
first_paragraph_words = soup.p.text.split()

first_paragraph_id = soup.p.get('id')
first_paragraph_id2 = soup.p['id']

all_paragraphs = soup.find_all('p')
paragraphs_with_ids = [p for p in soup('p') if p.get('id')]

important_paragraphs = soup('p', {'class': 'important'})
important_paragraphs2 = soup('p', 'important')
important_paragraphs3 = [p for p in soup('p')
                         if 'important' in p.get('class', [])]



# Example: Keeping Tabs on Congress

from bs4 import BeautifulSoup
import requests

url = "http://www.house.gov/representatives"
text = requests.get(url).text
soup = BeautifulSoup(text, "html5lib")

all_urls = [a['href']
            for a in soup('a')
            if a.has_attr('href')]

print(len(all_urls))

import re

regex = r"^http?://.*\house\.gov/?$"

assert re.match(regex, "http://hexi.house.gov")
assert re.match(regex, "https://hexi.house.gov/")
assert re.match(regex, "http://joel.house.gov/")
assert not re.match(regex, "joel.house.gov")
assert not re.match(regex, "http://joel.house.com")

good_urls = [url for url in all_urls if re.match(regex, url)]

print(len(good_urls))

good_urls = list(set(good_urls))

print(len(good_urls))

html = requests.get('http://jayapal.house.gov').text
soup = BeautifulSoup(html, 'html5lib')
links = {a['href'] for a in soup('a') if 'press releases' in a.text.lower()}

print(links)


from typing import Dict, Set

press_releases: Dict[str, Set[str]] = {}

for house_url in good_urls:
    html = requests.get(house_url).text
    soup = BeautifulSoup(html, 'html5lib')
    pr_links = {a['href'] for a in soup('a') if 'press releases' in a.text.lower()}

    print(f"{house_url}: {pr_links}")
    press_releases[house_url] = pr_links


def paragraph_mentions(text: str, keyword: str)-> bool:
    soup = BeautifulSoup(text, 'html5lib')
    paragraphs = [p.get_text() for p in soup('p')]

    return any(keyword.lower() in paragraph.lower()
               for paragraph in paragraphs)

text = """<body><h1>Facebook</h1><p>Twitter<p/>"""
assert paragraph_mentions(text, "twitter")
assert not paragraph_mentions(text, "facebook")

for house_url, pr_links in press_releases.items():
    for pr_link in pr_links:
        url = f"{house_url}/{pr_link}"
        text = requests.get(url).text

        if paragraph_mentions(text, "data"):
            print(f"{house_url}")
            break



##Using APIs (application programming interfaces)
#using an Unauthenticated API

import requests, json

github_user = "hexijin"
endpoint = f"https://api.github.com/users/{github_user}/repos"

repos = json.loads(requests.get(endpoint).text)

from collections import Counter
from dateutil.parser import parse

dates = [parse(repo["created_at"]) for repo in repos]
month_counts = Counter(date.moth for date in dates)
weekday_counts = Counter(data.weekday for date in dates)

last_5_repositories = sorted(repos, key=lambda r: r["pushed_at"], reverse=True)[:5]
last_5_languages = [repo["language"] for repo in last_5_repositories]