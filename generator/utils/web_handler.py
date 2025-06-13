import logging
import os
from urllib.parse import urljoin

import aiohttp
from aiohttp import ClientSession
from bs4 import BeautifulSoup

import generator.utils.constants as const


async def fetch_html(url: str, timeout: int = 10) -> str | None:
    user_agent = os.getenv(const.USER_AGENT, const.DEFAULT_USER_AGENT)
    headers = {"User-Agent": user_agent}
    timeout = aiohttp.ClientTimeout(total=timeout)

    try:
        async with ClientSession(headers=headers, timeout=timeout) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.text()
    except aiohttp.InvalidURL as e:
        logging.exception(f"The url: {url} is incorrect. {e}")
        return None
    except aiohttp.ClientConnectorError as e:
        logging.exception(f"Cannot connect to the url: {url}. {e}")
        return None
    except aiohttp.client_exceptions.ClientResponseError as e:
        if e.status == 403:
            logging.exception(f"Access to the url: {url} is forbidden. {e}")
        else:
            error_msg = f"Client response error occurred for url: {url}. {e}"
            logging.exception(error_msg)
        return None


def conv_html_to_md(
    html_content: str,
    base_url: str | None,
) -> tuple[str, str]:
    soup = BeautifulSoup(html_content, "html.parser")

    for i in range(1, 7):
        headers = soup.find_all(f"h{i}")
        for header in headers:
            header_text = header.get_text(strip=True)
            new_header = soup.new_tag("p")
            new_header.string = f'{"#" * i} {header_text}\n'
            header.replace_with(new_header)

    links = soup.find_all("a", href=True)
    for link in links:
        link_text = link.get_text(strip=True)
        link_url = urljoin(base_url, link["href"])
        new_link = soup.new_tag("p")
        new_link.string = f"[{link_text}]({link_url})"
        link.replace_with(new_link)

    unordered_lists = soup.find_all("ul")
    for ul in unordered_lists:
        ul_elements = ul.find_all("li")
        for li in ul_elements:
            new_li = soup.new_tag("p")
            new_li.string = f"- {li.get_text(strip=True)}"
            li.replace_with(new_li)
        ul.unwrap()

    ordered_lists = soup.find_all("ol")
    for ol in ordered_lists:
        ol_elements = ol.find_all("li")
        for index, li in enumerate(ol_elements, start=1):
            new_li = soup.new_tag("p")
            new_li.string = f"{index}. {li.get_text(strip=True)}"
            li.replace_with(new_li)
        ol.unwrap()

    markdown_text = "\n".join(soup.stripped_strings)
    title = soup.title.string if soup.title else "Untitled Page"
    return markdown_text, title
