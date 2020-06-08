#!/usr/bin/env python3
"""
This script downloads the chimera dmg from the UCSF website. We need to use
a somewhat sophisticated approach here because the website requires to "accept" some
conditions and then there's more hoops to go through.

wget on the remote_url below doesn't seem to work. Using requests works.
"""
from pathlib import Path
import re

import requests


def extract_download_sub_url(redirect_page_html: str) -> str:
    """
    We use regex to parse the html in order to avoid
    introducing a dependency on beautiful soup or some other
    html parsing library.
    """
    left_achor = '<a href="'
    right_achor = '">'
    pattern = f'{left_achor}.*{right_achor}'
    matched_string = re.search(pattern, redirect_page_html).group()

    sub_url = matched_string.replace(left_achor, "").replace(right_achor, "")
    return sub_url


def download_file(remote_url: str, local_file_path: str):
    with requests.get(remote_url, stream=True) as r:
        r.raise_for_status()
        with open(local_file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


base_url = "https://www.cgl.ucsf.edu"
chimera_conditions_page_url = f"{base_url}/chimera/cgi-bin/secure/chimera-get.py?file=mac64/chimera-1.14-mac64.dmg"

# This remote url was obtained by inspection, playing around with the download page
remote_url = "https://www.cgl.ucsf.edu/chimera/cgi-bin/secure/chimera-get.py?ident=OHeQer2WTadn%2F%2BVltnhB5%2BBgukNERNvx1hxz3g3ghO4rrg%3D%3D&file=mac64%2Fchimera-1.14-mac64.dmg&choice=Notified"

filename = "chimera-1.14-mac64.dmg"

if __name__ == '__main__':

    # accept conditions
    payload = {'choice': 'Accept'}
    r = requests.post(chimera_conditions_page_url, data=payload)

    # get the html for the redirect as a string
    redirect_page_html = r.content.decode('utf-8')

    # extract the redirected url for download
    download_sub_url = extract_download_sub_url(redirect_page_html)
    download_url = f"{base_url}/{download_sub_url}"

    dir = Path(__file__).parent
    local_file_path = str(dir.joinpath(filename))
    print(f"Downoading file {filename} from {remote_url}...")
    download_file(download_url, local_file_path)
    print(f"Finished downloading.")

