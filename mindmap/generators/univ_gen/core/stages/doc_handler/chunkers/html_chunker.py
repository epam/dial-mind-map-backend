import asyncio as aio

from langchain_core.documents import Document as LCDoc
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from mindmap.generators.univ_gen.common.constants import DataFrameCols as Col
from mindmap.generators.univ_gen.core.utils.web_handler import conv_html_to_md
from mindmap.utils.logger_config import logger

from ..aggregation import aggregate_text_chunks
from ..constants import DEFAULT_DOC_DESC, MAX_CHUNK_SIZE
from ..constants import DocCategories as DocCat
from ..constants import DocType
from ..registry import register_handler
from ..structs import Chunk, DocAndContent
from .base_chunker import BaseDocChunker

# --- Constants ---
HEADER_TO_SPLIT_ON = "Header 1"


@register_handler
class HTMLChunker(BaseDocChunker):
    """Handler for chunking HTML documents and web links."""

    @property
    def supported_categories(self) -> list[str]:
        return [DocCat.HTML, DocCat.LINK]

    async def chunk(self, doc_with_content: DocAndContent) -> list[Chunk]:
        """
        Converts HTML to Markdown and splits it into optimized chunks.
        """
        if not isinstance(doc_with_content.content, str):
            logger.warning(
                f"Expected string for HTML {doc_with_content.doc.id}, "
                "got non-string."
            )
            return []

        md_content, doc_title = conv_html_to_md(
            doc_with_content.content, doc_with_content.doc.base_url
        )

        text_splitter = RecursiveCharacterTextSplitter()
        md_header_splits = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", HEADER_TO_SPLIT_ON)]
        ).split_text(md_content)
        initial_chunks = text_splitter.split_documents(md_header_splits)
        if not initial_chunks:
            return []

        merged_lc_docs = aggregate_text_chunks(
            initial_chunks, MAX_CHUNK_SIZE, HEADER_TO_SPLIT_ON
        )

        doc_cat = (
            DocCat.LINK
            if doc_with_content.doc.type == DocType.LINK
            else DocCat.HTML
        )

        return [
            Chunk(
                lc_doc,
                doc_with_content.doc.id,
                doc_with_content.doc.url,
                doc_title,
                doc_cat,
                getattr(doc_with_content.doc, Col.DOC_DESC, DEFAULT_DOC_DESC),
            )
            for lc_doc in merged_lc_docs
        ]

    async def to_markdown(self, doc_with_content: DocAndContent) -> str:
        """Converts an HTML-like document's content to Markdown."""
        if not isinstance(doc_with_content.content, str):
            logger.warning(
                f"Expected string for {doc_with_content.doc.id}, "
                "got non-string."
            )
            return ""
        md_content, _ = await aio.to_thread(
            conv_html_to_md,
            doc_with_content.content,
            doc_with_content.doc.base_url,
        )
        return md_content


@register_handler
class WholeHTMLChunker(BaseDocChunker):
    """
    Handler for treating an entire HTML/Link document as a single chunk.
    """

    @property
    def supported_categories(self) -> list[str]:
        return [DocCat.HTML_AS_A_WHOLE, DocCat.LINK_AS_A_WHOLE]

    async def chunk(self, doc_with_content: DocAndContent) -> list[Chunk]:
        """
        Converts the entire HTML to Markdown and returns it as a single
        chunk.
        """
        if not isinstance(doc_with_content.content, str):
            logger.warning(
                f"Expected string for HTML {doc_with_content.doc.id}, "
                "got non-string."
            )
            return []

        # 1. Convert the entire HTML to Markdown
        md_content, doc_title = await aio.to_thread(
            conv_html_to_md,
            doc_with_content.content,
            doc_with_content.doc.base_url,
        )

        if not md_content:
            return []

        # 2. Determine the document category
        doc_cat = (
            DocCat.LINK_AS_A_WHOLE
            if doc_with_content.doc.type == DocType.LINK
            else DocCat.HTML_AS_A_WHOLE
        )

        # 3. Create a single chunk containing the full Markdown content
        return [
            Chunk(
                lc_doc=LCDoc(page_content=md_content),
                doc_id=doc_with_content.doc.id,
                doc_url=doc_with_content.doc.url,
                doc_title=doc_title,
                doc_cat=doc_cat,
                doc_desc=getattr(
                    doc_with_content.doc, Col.DOC_DESC, DEFAULT_DOC_DESC
                ),
            )
        ]

    async def to_markdown(self, doc_with_content: DocAndContent) -> str:
        """Converts an HTML-like document's content to Markdown."""
        # Logic is identical to the standard HTMLChunker
        if not isinstance(doc_with_content.content, str):
            logger.warning(
                f"Expected string for {doc_with_content.doc.id}, "
                "got non-string."
            )
            return ""
        md_content, _ = await aio.to_thread(
            conv_html_to_md,
            doc_with_content.content,
            doc_with_content.doc.base_url,
        )
        return md_content
