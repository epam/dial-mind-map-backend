import logging
from io import BytesIO

import aiohttp
from typing import List
from langchain_community.document_loaders.unstructured import UnstructuredFileIOLoader
from langchain.schema import Document
from unstructured.file_utils.model import FileType
from aidial_sdk import HTTPException
from pdf2image.exceptions import PDFInfoNotInstalledError
from unstructured_pytesseract.pytesseract import TesseractNotFoundError

from dial_rag.attachment_link import AttachmentLink
from dial_rag.content_stream import SupportsWriteStr
from dial_rag.errors import InvalidDocumentError
from dial_rag.image_processor.extract_pages import extract_number_of_pages, are_image_pages_supported
from dial_rag.print_stats import print_documents_stats
from dial_rag.request_context import RequestContext
from dial_rag.utils import int_env_var, size_env_var, format_size, get_bytes_length, timed_block
from dial_rag.resources.cpu_pools import run_in_indexing_cpu_pool


UNSTRUCTURED_CHUNK_SIZE = 1000

MAX_DOCUMENT_TEXT_SIZE: int = size_env_var("MAX_DOCUMENT_TEXT_SIZE", "5MiB")

WEB_LOADER_TIMEOUT_SECONDS = int_env_var("WEB_LOADER_TIMEOUT_SECONDS", 30)


async def download_attachment(url, headers) -> tuple[str, bytes]:
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, timeout=WEB_LOADER_TIMEOUT_SECONDS) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")

            content = await response.read()  # Await the coroutine
            logging.debug(f"Downloaded {url}: {len(content)} bytes")
            return content_type, content


def add_source_metadata(pages: List[Document], attachment_link: AttachmentLink) -> List[Document]:
    for page in pages:
        page.metadata['source'] = attachment_link.dial_link
        page.metadata['source_display_name'] = attachment_link.display_name
    return pages


def add_pdf_source_metadata(pages: List[Document], attachment_link: AttachmentLink) -> List[Document]:
    assert len(pages)
    pages = add_source_metadata(pages, attachment_link)

    for page in pages:
        if 'page_number' in page.metadata:
            page.metadata['source'] += f"#page={page.metadata['page_number']}"
    return pages


async def load_dial_document_metadata(
    request_context: RequestContext,
    attachment_link: AttachmentLink,
) -> dict:
    if not attachment_link.is_dial_document:
        raise ValueError("Not a Dial document")

    metadata_url = attachment_link.dial_metadata_url
    assert metadata_url is not None

    headers = request_context.get_file_access_headers(metadata_url)
    async with aiohttp.ClientSession() as session:
        async with session.get(metadata_url, headers=headers) as response:
            if not response.ok:
                error_message = f"{response.status} {response.reason}"
                raise InvalidDocumentError(error_message)
            return await response.json()


async def load_attachment(
    attachment_link: AttachmentLink,
    headers: dict
) -> tuple[str, str, bytes]:
    absolute_url = attachment_link.absolute_url
    file_name = attachment_link.display_name
    content_type, attachment_bytes = await download_attachment(absolute_url, headers)
    if attachment_bytes:
        return file_name, content_type, attachment_bytes
    raise InvalidDocumentError(f"Attachment {file_name}, can't be read properly")


def get_image_only_chunks(
    document_bytes: bytes,
    mime_type: str,
) -> List[Document]:
    page_num = extract_number_of_pages(mime_type, document_bytes)
    return [
        Document(page_content="", metadata={
            "filetype": mime_type,
            "page_number": i + 1,
        })
        for i in range(page_num)
    ]


def get_document_chunks(
    document_bytes: bytes,
    mime_type: str,
    attachment_link: AttachmentLink
) -> List[Document]:
    try:
        chunks = UnstructuredFileIOLoader(
            file=BytesIO(document_bytes),
            # Current version of unstructured library expect mime type instead of the full content type with encoding, etc.
            content_type=mime_type,
            mode="elements",
            strategy="fast",
            chunking_strategy="by_title",
            multipage_sections=False,
            # Disable combining text chunks, because it does not respect multipage_sections=False
            # TODO: Update unstructured library to the version with chunking/title.py refactoring
            combine_text_under_n_chars=0,
            new_after_n_chars=UNSTRUCTURED_CHUNK_SIZE,
            max_characters=UNSTRUCTURED_CHUNK_SIZE
        ).load()
    except ValueError as e:
        raise HTTPException(
            "Unable to load document content. Try another document format.",
        ) from e
    except (PDFInfoNotInstalledError, TesseractNotFoundError) as e:
        # TODO: Update unstructured library to avoid attempts to use ocr
        logging.warning('PDF file without text. Trying to extract images.')
        chunks = None

    if not chunks:
        if are_image_pages_supported(mime_type):
            chunks = get_image_only_chunks(document_bytes, mime_type)
        else:
            raise InvalidDocumentError("The document is empty")

    filetype = FileType.from_mime_type(mime_type)

    if filetype == FileType.PDF:
        chunks = add_pdf_source_metadata(chunks, attachment_link)
    else:
        chunks = add_source_metadata(chunks, attachment_link)
    return chunks


async def parse_document(
        stageio: SupportsWriteStr,
        document_bytes: bytes,
        mime_type: str,
        attachment_link: AttachmentLink
) -> List[Document]:
    async with timed_block("Parsing document", stageio):
        stageio.write("Loader: Unstructured\n")
        chunks = await run_in_indexing_cpu_pool(
            get_document_chunks,
            document_bytes,
            mime_type,
            attachment_link
        )

        stageio.write(f"File type: {chunks[0].metadata['filetype']}\n")
        print_documents_stats(stageio, chunks)

        total_text_size = sum(get_bytes_length(chunk.page_content) for chunk in chunks)
        if total_text_size > MAX_DOCUMENT_TEXT_SIZE:
            raise InvalidDocumentError(
                f"Document text is too large: {format_size(total_text_size)} > "
                f"{format_size(MAX_DOCUMENT_TEXT_SIZE)}"
            )

        return chunks
