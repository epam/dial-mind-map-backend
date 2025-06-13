from aiohttp import ClientConnectorError, InvalidUrlClientError

from general_mindmap.v2.dial.client import DialClient
from .logger import logging


async def read_file_bytes_from_dial(
    dial_client: DialClient, doc_url: str
) -> bytes | None:
    """
    Fetch file bytes from a document URL using the DIAL client.

    Args:
        dial_client: Initialized DIAL client to use for the request
        doc_url: URL of the document to fetch

    Returns:
        Bytes content of the file or None if an error occurs
    """
    try:
        return await dial_client.read_raw_file_by_url(doc_url)
    except InvalidUrlClientError as e:
        logging.info(f"The file is incorrect: {doc_url}. Error: {e}")
        return None
    except ClientConnectorError as e:
        logging.info(f"Cannot download the file: {doc_url}. Error: {e}")
        return None
    except Exception as e:
        logging.error(
            f"Unexpected error when downloading file: {doc_url}. Error: {e}"
        )
        return None
