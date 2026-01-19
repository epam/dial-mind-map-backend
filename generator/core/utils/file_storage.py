from aiohttp import ClientConnectorError, InvalidUrlClientError

from common_utils.logger_config import logger
from generator.common.interfaces import FileStorage


async def read_file_bytes(
    file_storage: FileStorage, doc_url: str
) -> bytes | None:
    """
    Fetch file bytes from a document URL using the File Storage.

    Args:
        file_storage: Initialized File Storage to use for the request
        doc_url: URL of the document to fetch

    Returns:
        Bytes content of the file or None if an error occurs
    """
    try:
        return await file_storage.read_raw_file_by_url(doc_url)
    except InvalidUrlClientError as e:
        logger.info(f"The file is incorrect: {doc_url}. Error: {e}")
        return None
    except ClientConnectorError as e:
        logger.info(f"Cannot download the file: {doc_url}. Error: {e}")
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error when downloading file: {doc_url}. Error: {e}"
        )
        return None
