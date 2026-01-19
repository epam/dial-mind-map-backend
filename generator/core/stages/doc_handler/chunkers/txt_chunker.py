from langchain_core.documents import Document as LCDoc
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common_utils.logger_config import logger
from generator.common.constants import DataFrameCols as Col

from ..aggregation import aggregate_text_chunks
from ..constants import DEFAULT_DOC_DESC, MAX_CHUNK_SIZE
from ..constants import DocCategories as DocCat
from ..registry import register_handler
from ..structs import Chunk, DocAndContent
from .base_chunker import BaseDocChunker


@register_handler
class TXTChunker(BaseDocChunker):
    """Handler for chunking plain text (TXT) documents."""

    @property
    def supported_categories(self) -> list[str]:
        return [DocCat.TXT]

    async def chunk(self, item: DocAndContent) -> list[Chunk]:
        """Splits a single plain text file into chunks."""
        if not isinstance(item.content, bytes):
            logger.warning(f"Expected bytes for TXT {item.doc.id}, got non-bytes.")
            return []

        try:
            text_content = item.content.decode("utf-8")
            doc_title = item.doc.name
        except UnicodeDecodeError:
            logger.error(f"Could not decode TXT file {item.doc.id} as UTF-8.")
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_CHUNK_SIZE,
            chunk_overlap=int(MAX_CHUNK_SIZE * 0.1),
        )
        initial_chunks = text_splitter.create_documents([text_content])
        if not initial_chunks:
            return []

        merged_lc_docs = aggregate_text_chunks(initial_chunks, MAX_CHUNK_SIZE)

        return [
            Chunk(
                lc_doc,
                item.doc.id,
                item.doc.url,
                doc_title,
                DocCat.TXT,
                getattr(item.doc, Col.DOC_DESC, DEFAULT_DOC_DESC),
            )
            for lc_doc in merged_lc_docs
        ]

    async def to_markdown(self, item: DocAndContent) -> str:
        """Reads the content of a TXT file as a string."""
        if not isinstance(item.content, bytes):
            return ""
        try:
            return item.content.decode("utf-8")
        except UnicodeDecodeError:
            logger.error(f"Could not decode TXT file {item.doc.id} as UTF-8.")
            return ""


@register_handler
class WholeTXTChunker(BaseDocChunker):
    """
    Handler for treating an entire plain text (TXT) document as a single chunk.
    """

    @property
    def supported_categories(self) -> list[str]:
        return [DocCat.TXT_AS_A_WHOLE]

    async def chunk(self, item: DocAndContent) -> list[Chunk]:
        """Reads the entire text file into a single chunk."""
        if not isinstance(item.content, bytes):
            logger.warning(f"Expected bytes for TXT {item.doc.id}, got non-bytes.")
            return []

        try:
            text_content = item.content.decode("utf-8")
            doc_title = item.doc.name
        except UnicodeDecodeError:
            logger.error(f"Could not decode TXT file {item.doc.id} as UTF-8.")
            return []

        if not text_content:
            return []

        # Create a single chunk with the entire file content
        return [
            Chunk(
                lc_doc=LCDoc(page_content=text_content),
                doc_id=item.doc.id,
                doc_url=item.doc.url,
                doc_title=doc_title,
                doc_cat=DocCat.TXT_AS_A_WHOLE,
                doc_desc=getattr(item.doc, Col.DOC_DESC, DEFAULT_DOC_DESC),
            )
        ]

    async def to_markdown(self, item: DocAndContent) -> str:
        """Reads the content of a TXT file as a string."""
        # Logic is identical to the standard TXTChunker
        if not isinstance(item.content, bytes):
            return ""
        try:
            return item.content.decode("utf-8")
        except UnicodeDecodeError:
            logger.error(f"Could not decode TXT file {item.doc.id} as UTF-8.")
            return ""
