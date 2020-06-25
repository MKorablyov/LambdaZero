#!/usr/bin/env python3
"""
This script downloads the chimera dmg from the UCSF website. We need to use
a somewhat sophisticated approach here because the website requires to "accept" some
conditions and then there's a redirect on the download page. This redirect contains an actual download
link which seems to contain a token which is only valid for a finite amount of time.

Simply using wget on the download url does not work, and the download url eventually expires. Thus,
we have to go through the accept page to get a download url with a valid token, and then download the actual dmg.
"""
from pathlib import Path
import re

import requests


def extract_download_relative_url(redirect_page_html: str) -> str:
    """
    We use regex to parse the html in order to avoid
    introducing a dependency on an html parsing library.

    The left and right anchor below we obtained by inspection of the redirect page html.

    """
    left_anchor = '<a href="'
    right_anchor = '">'
    pattern = f'{left_anchor}.*{right_anchor}'
    matched_string = re.search(pattern, redirect_page_html).group()

    relative_url = matched_string.replace(left_anchor, "").replace(right_anchor, "")
    return relative_url


def download_file(remote_url: str, local_file_path: str):
    with requests.get(remote_url, stream=True) as r:
        r.raise_for_status()
        with open(local_file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


base_url = "https://www.cgl.ucsf.edu"
filename = "chimera-1.14-mac64.dmg"
chimera_conditions_page_url = f"{base_url}/chimera/cgi-bin/secure/chimera-get.py?file=mac64/{filename}"

if __name__ == '__main__':

    # accept conditions
    payload = {'choice': 'Accept'}
    r = requests.post(chimera_conditions_page_url, data=payload)

    # get the html for the redirect as a string
    redirect_page_html = r.content.decode('utf-8')

    # extract the redirected url for download
    download_relative_url = extract_download_relative_url(redirect_page_html)
    download_url = f"{base_url}/{download_relative_url}"

    local_file_path = str(Path(__file__).parent.joinpath(filename))
    print(f"Downoading file {filename} from {download_url}...")
    download_file(download_url, local_file_path)
    print(f"Finished downloading.")

