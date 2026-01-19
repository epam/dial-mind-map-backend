import asyncio as aio
import base64
import io
from typing import List

import PIL.Image
from langchain_core.documents import Document as LCDoc
from unstructured.documents.elements import Text
from unstructured.partition.pptx import partition_pptx

from common_utils.logger_config import logger
from generator.chainer.model_handler import LLMUtils
from generator.common.constants import DataFrameCols as Col

from ..aggregation import aggregate_page_content
from ..constants import DEFAULT_DOC_DESC
from ..constants import DocCategories as DocCat
from ..registry import register_handler
from ..structs import Chunk, DocAndContent, Document, PageContent
from ..utils import (
    _convert_pptx_to_pdf,
    calculate_image_tokens,
    convert_pptx_to_images,
)
from .base_chunker import BaseDocChunker
from .pdf_chunker import PDFChunker, WholePDFChunker

# --- Constants ---
MAX_CHUNK_SIZE = 2048
MAX_CHUNK_IMG_NUM = 50
ENCODER = LLMUtils.get_encoding_for_model()


@register_handler
class PPTXChunker(BaseDocChunker):
    """
    Handler for chunking PPTX documents by converting them to PDF and
    delegating to the PDFChunker.
    """

    @property
    def supported_categories(self) -> list[str]:
        return [
            DocCat.PPTX,
        ]

    async def chunk(self, item: DocAndContent) -> list[Chunk]:
        """
        Converts the PPTX to PDF and uses PDFChunker for processing.
        """
        if not isinstance(item.content, bytes):
            return []

        # 1. Convert PPTX to PDF bytes
        pdf_bytes = await _convert_pptx_to_pdf(item.content)
        if not pdf_bytes:
            return []

        # 2. Create a new DocAndContent item with the PDF content
        pdf_item = DocAndContent(doc=item.doc, content=pdf_bytes)

        # 3. Delegate chunking to an instance of PDFChunker
        pdf_chunker = PDFChunker()
        chunks = await pdf_chunker.chunk(pdf_item)

        # 4. Correct the document category on the resulting chunks
        for chunk in chunks:
            chunk.doc_category = DocCat.PPTX

        return chunks

    async def to_markdown(self, item: DocAndContent) -> str:
        """
        Converts PPTX to PDF and uses PDFChunker to extract markdown.
        """
        if not isinstance(item.content, bytes):
            return ""

        pdf_bytes = await _convert_pptx_to_pdf(item.content)
        if not pdf_bytes:
            return ""

        pdf_item = DocAndContent(doc=item.doc, content=pdf_bytes)

        # Delegate markdown extraction to an instance of PDFChunker
        pdf_chunker = PDFChunker()
        return await pdf_chunker.to_markdown(pdf_item)


@register_handler
class WholePPTXChunker(BaseDocChunker):
    """
    Handler for treating an entire PPTX as a single chunk by converting
    it to PDF and delegating to WholePDFChunker.
    """

    def __init__(self, include_page_numbers: bool = False):
        super().__init__()
        self.include_page_numbers = include_page_numbers

    @property
    def supported_categories(self) -> list[str]:
        return [DocCat.PPTX_AS_A_WHOLE]

    async def chunk(self, item: DocAndContent) -> list[Chunk]:
        """
        Processes a single PPTX as one chunk using WholePDFChunker.
        """
        if not isinstance(item.content, bytes):
            return []

        # 1. Convert PPTX to PDF bytes
        pdf_bytes = await _convert_pptx_to_pdf(item.content)
        if not pdf_bytes:
            return []

        # 2. Create a new DocAndContent item with the PDF content
        pdf_item = DocAndContent(doc=item.doc, content=pdf_bytes)

        # 3. Delegate chunking to an instance of WholePDFChunker
        whole_pdf_chunker = WholePDFChunker(
            include_page_numbers=self.include_page_numbers
        )
        chunks = await whole_pdf_chunker.chunk(pdf_item)

        # 4. Correct the document category on the resulting chunk
        for chunk in chunks:
            chunk.doc_category = DocCat.PPTX_AS_A_WHOLE

        return chunks

    async def to_markdown(self, item: DocAndContent) -> str:
        """
        Converts PPTX to PDF and uses WholePDFChunker for markdown extraction.
        """
        if not isinstance(item.content, bytes):
            return ""

        pdf_bytes = await _convert_pptx_to_pdf(item.content)
        if not pdf_bytes:
            return ""

        pdf_item = DocAndContent(doc=item.doc, content=pdf_bytes)

        whole_pdf_chunker = WholePDFChunker()
        return await whole_pdf_chunker.to_markdown(pdf_item)


@register_handler
class UnstructuredPPTXChunker(BaseDocChunker):
    """
    Handler for chunking PPTX documents using 'unstructured' for text
    and a custom converter for whole slide images.
    """

    @property
    def supported_categories(self) -> list[str]:
        return [
            # DocCat.PPTX,
        ]

    async def chunk(self, item: DocAndContent) -> list[Chunk]:
        """
        Chunks a PPTX file by extracting text with unstructured and
        converting each slide to an image.
        """
        if not isinstance(item.content, bytes):
            logger.warning(
                f"Expected bytes for PPTX {item.doc.id}, got non-bytes."
            )
            return []

        try:
            # 1. Asynchronously convert PPTX slides to PIL images
            slide_images = await convert_pptx_to_images(item.content)
        except Exception as e:
            logger.error(
                f"Failed to convert PPTX to images for doc {item.doc.id}: {e}",
                exc_info=True,
            )
            return []

        # 2. Pass content and images to the synchronous processing function in a thread
        pages = await aio.to_thread(
            self._extract_and_combine, item.content, slide_images
        )

        if not pages:
            logger.warning(f"Chunking resulted in 0 pages for doc {item.doc.id}.")
            return []

        # 3. Aggregate pages into chunks
        merged_slide_groups = aggregate_page_content(
            pages, ENCODER, MAX_CHUNK_SIZE, MAX_CHUNK_IMG_NUM
        )
        return self._create_chunks_from_page_groups(
            merged_slide_groups, item.doc
        )

    @staticmethod
    def _extract_and_combine(
        content: bytes, slide_images: List[PIL.Image.Image]
    ) -> list[PageContent]:
        """
        Extracts text using unstructured and combines it with the pre-converted
        slide images.
        """
        # 1. Extract text elements using unstructured
        try:
            elements = partition_pptx(
                file=io.BytesIO(content), include_page_breaks=False
            )
        except Exception as e:
            logger.error(
                f"Unstructured failed to partition PPTX: {e}", exc_info=True
            )
            return []

        page_texts: dict[int, list[str]] = {}
        for el in elements:
            if isinstance(el, Text) and el.text.strip():
                page_num = getattr(el.metadata, "page_number", 1)
                if page_num not in page_texts:
                    page_texts[page_num] = []
                page_texts[page_num].append(el.text.strip())

        # 2. Combine text with corresponding slide images
        pages = []
        for i, pil_image in enumerate(slide_images):
            page_num = i + 1

            # Convert PIL image to raw base64 string
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            texts = page_texts.get(page_num, [])
            images = [img_str]

            text_tokens = sum(len(ENCODER.encode(t)) for t in texts)
            image_tokens = calculate_image_tokens(img_str)
            total_tokens = text_tokens + image_tokens

            pages.append(
                PageContent(
                    page_id=page_num,
                    texts=texts,
                    images=images,
                    tokens=total_tokens,
                )
            )
        return pages

    @staticmethod
    def _create_chunks_from_page_groups(
        groups: list[list[PageContent]], doc: Document
    ) -> list[Chunk]:
        """Converts grouped slide content into final Chunk objects."""
        doc_title = doc.name
        page_chunks = []
        for group in groups:
            texts = [
                f"Slide {content.page_id}: {txt}"
                for content in group
                for txt in content.texts
            ]
            page_ids = sorted(list({content.page_id for content in group}))
            page_chunks.append(
                Chunk(
                    LCDoc("\n".join(texts)),
                    doc.id,
                    doc.url,
                    doc_title,
                    DocCat.PPTX,
                    getattr(doc, Col.DOC_DESC, DEFAULT_DOC_DESC),
                    group,
                    page_ids,
                )
            )
        return page_chunks

    async def to_markdown(self, item: DocAndContent) -> str:
        """Extracts all text from a PPTX using unstructured."""
        if not isinstance(item.content, bytes):
            return ""

        def sync_process_pptx():
            try:
                elements = partition_pptx(file=io.BytesIO(item.content))
                return "\n\n".join(
                    [el.text for el in elements if hasattr(el, "text")]
                )
            except Exception as e:
                logger.error(
                    f"Failed to convert PPTX to markdown: {e}", exc_info=True
                )
                return ""

        return await aio.to_thread(sync_process_pptx)
