import asyncio as aio
import logging

from generator.common.interfaces import FileStorage
from generator.common.structs import Document
from generator.core.stages.doc_handler.constants import DocContentType, DocType
from generator.core.stages.doc_handler.structs import DocAndContent
from generator.core.utils.file_storage import read_file_bytes
from generator.core.utils.web_handler import fetch_html


async def fetch_all_docs_content(
    docs: list[Document], file_storage: FileStorage
) -> list[DocAndContent]:
    tasks = [_fetch_document_content(doc, file_storage) for doc in docs]
    docs_with_content = await aio.gather(*tasks)

    successful_fetches = [item for item in docs_with_content if item.content]

    failed_docs = [
        item.doc.url for item in docs_with_content if not item.content
    ]
    if failed_docs:
        logging.warning(f"Failed to fetch content for: {failed_docs}")

    return successful_fetches


async def _fetch_document_content(
    doc: Document, file_storage: FileStorage
) -> DocAndContent:
    """
    Fetches the content for a single document, ensuring the correct
    data type.

    - Link content is returned as a string.
    - Uploaded HTML file content is decoded and returned as a string.
    - Other file types (PDF, PPTX) are returned as bytes.
    """
    doc_type = doc.type
    doc_url = doc.url

    # 1. Handle Links: They are always fetched as strings.
    if doc_type == DocType.LINK:
        html_content = await fetch_html(doc_url)
        return DocAndContent(doc=doc, content=html_content or "")

    # 2. Handle Files: This requires checking the content_type.
    if doc_type == DocType.FILE:
        try:
            byte_content = await read_file_bytes(file_storage, doc_url)

            # If the uploaded file is HTML, it must be decoded to a
            # string.
            if doc.content_type == DocContentType.HTML:
                return DocAndContent(
                    doc=doc, content=byte_content.decode("utf-8")
                )
            else:
                # For all other files (PDF, PPTX), keep them as bytes.
                return DocAndContent(doc=doc, content=byte_content)

        except Exception as e:
            logging.error(
                f"Failed to read file from DIAL for doc {doc.id}: {e}"
            )
            return DocAndContent(doc=doc, content=b"")

    # 3. Fallback for unsupported cases
    logging.warning(f"Could not fetch content for doc {doc.id}: {doc_url}")
    return DocAndContent(doc=doc, content=b"")
