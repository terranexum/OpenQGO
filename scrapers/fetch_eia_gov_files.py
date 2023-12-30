#!/usr/bin/env python3

"""
Downloads all ZIP files from EIA.gov site.

Requires BeautifulSoup and requests

pip install beautifulsoup4
pip install requests


How to use:

python scrapers/fetch_eia_gov_files.py

"""

from pathlib import Path
import os

from urllib.request import urlretrieve

import requests
from bs4 import BeautifulSoup

eia_data_dir = Path(__file__).resolve().parent / "eai_data"

try:
    os.mkdir(eia_data_dir)
except OSError:
    pass

# fetch the HTML from the site and parse the hrefs out of the links to ZIP files on the page
url = "https://www.eia.gov/opendata/index.php"

response = requests.get(url)

html = str(response.content)

soup = BeautifulSoup(html)

zip_links = soup.find_all('a', attrs={'class': 'ico zip'})
for zip_link in zip_links:
    # download each ZIP file to a local data directory.
    href = zip_link['href']
    filename = zip_link.contents[0].strip()
    filename = f"{filename}.zip"
    download_file_path = eia_data_dir / filename
    print(download_file_path)
    urlretrieve(href, download_file_path)
    

