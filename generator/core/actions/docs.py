import asyncio as aio
from typing import List

from langchain_community.document_loaders import AsyncHtmlLoader

from common_utils.logger_config import logger
from generator.common.interfaces import FileStorage
from generator.common.structs import Document
from generator.core.stages.doc_handler.constants import DocContentType, DocType
from generator.core.stages.doc_handler.structs import DocAndContent
from generator.core.utils.file_storage import read_file_bytes


async def fetch_all_docs_content(
    docs: list[Document], file_storage: FileStorage
) -> list[DocAndContent]:
    """
    Fetches content for all provided documents concurrently.
    - Web Links: Uses standard AsyncHtmlLoader (Static fetch).
    - Files: Processed via internal storage.
    """
    if not docs:
        return []

    link_docs = [d for d in docs if d.type == DocType.LINK]
    file_docs = [d for d in docs if d.type == DocType.FILE]

    link_task = _fetch_batch_links(link_docs)
    file_tasks = [_fetch_file_content(doc, file_storage) for doc in file_docs]
    all_tasks = [link_task] + file_tasks

    results = await aio.gather(*all_tasks)
    link_results = results[0]
    file_results = results[1:]
    all_results = link_results + list(file_results)

    successful_fetches = [item for item in all_results if item.content]
    failed_docs = [item.doc.url for item in all_results if not item.content]

    if successful_fetches:
        successful_docs_names = [item.doc.name for item in successful_fetches]
        logger.info(f"Successfully fetched content for: {successful_docs_names}")

    if failed_docs:
        logger.warning(f"Failed to fetch content for: {failed_docs}")

    return successful_fetches


async def _fetch_batch_links(docs: list[Document]) -> list[DocAndContent]:
    """
    Fetches web links using only AsyncHtmlLoader (requests/aiohttp).
    Does NOT support dynamic JS-heavy sites (SPA).
    """
    if not docs:
        return []

    urls = [doc.url for doc in docs]
    results: List[DocAndContent] = []

    loader = AsyncHtmlLoader(
        urls, requests_per_second=2, ignore_load_errors=True
    )
    logger.info(f"Batch fetching {len(urls)} web links...")

    try:
        html_contents = await loader.fetch_all(urls)

        for doc, html_content in zip(docs, html_contents):
            content_str = html_content if html_content else ""

            results.append(DocAndContent(doc=doc, content=content_str))

    except Exception as e:
        logger.exception(f"Critical error during batch link fetching: {e}")
        results = [DocAndContent(doc=doc, content="") for doc in docs]

    return results


async def _fetch_file_content(
    doc: Document, file_storage: FileStorage
) -> DocAndContent:
    """
    Helper specifically for internal file storage
    (PDF, Uploaded HTML, etc.).
    """
    try:
        byte_content = await read_file_bytes(file_storage, doc.url)

        if doc.content_type == DocContentType.HTML:
            return DocAndContent(doc=doc, content=byte_content.decode("utf-8"))
        return DocAndContent(doc=doc, content=byte_content)

    except Exception as e:
        logger.error(f"Failed to read file from DIAL for doc {doc.id}: {e}")

        if doc.content_type == DocContentType.HTML:
            return DocAndContent(doc=doc, content="")
        return DocAndContent(doc=doc, content=b"")
