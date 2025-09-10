import asyncio as aio
import base64
import io
from typing import List

import PIL.Image
from langchain_core.documents import Document as LCDoc
from unstructured.documents.elements import Text
from unstructured.partition.pptx import partition_pptx

from generator.chainer.model_handler import LLMUtils
from generator.common.constants import DataFrameCols as Col
from generator.common.logger import logging

from ..aggregation import aggregate_page_content
from ..constants import DEFAULT_DOC_DESC
from ..constants import DocCategories as DocCat
from ..registry import register_handler
from ..structs import Chunk, DocAndContent, Document, PageContent
from ..utils import (
    calculate_image_tokens,
    convert_pptx_to_images,
    get_visible_slides,
    load_presentation_from_bytes,
    process_image,
)
from .base_chunker import BaseDocChunker

# --- Constants ---
MAX_CHUNK_SIZE = 2048
MAX_CHUNK_IMG_NUM = 50
ENCODER = LLMUtils.get_encoding_for_model()


@register_handler
class PPTXChunker(BaseDocChunker):
    """Handler for chunking PPTX documents."""

    @property
    def supported_categories(self) -> list[str]:
        return [
            DocCat.PPTX,
        ]

    async def chunk(self, item: DocAndContent) -> list[Chunk]:
        """Splits a single PPTX file into chunks based on slides."""
        if not isinstance(item.content, bytes):
            logging.warning(
                f"Expected bytes for PPTX {item.doc.id}, got non-bytes."
            )
            return []

        presentation = load_presentation_from_bytes(item.content)
        slides = [
            self.process_slide(slide, i)
            for i, slide in enumerate(get_visible_slides(presentation), 1)
        ]

        merged_slide_groups = aggregate_page_content(
            slides, ENCODER, MAX_CHUNK_SIZE, MAX_CHUNK_IMG_NUM
        )
        return self.create_chunks_from_page_groups(
            merged_slide_groups, item.doc
        )

    async def to_markdown(self, item: DocAndContent) -> str:
        """
        Extracts all text from a PPTX document into a single string.
        """
        if not isinstance(item.content, bytes):
            return ""

        def sync_process_pptx():
            presentation = load_presentation_from_bytes(item.content)
            all_texts = []
            for i, slide in enumerate(get_visible_slides(presentation), 1):
                slide_content = self.process_slide(slide, i)
                all_texts.extend(slide_content.texts)
            return "\n\n".join(all_texts)

        return await aio.to_thread(sync_process_pptx)

    @staticmethod
    def process_slide(slide, slide_number: int) -> PageContent:
        """
        Extracts text, images, and token counts from a single slide.
        """
        texts, images, total_tokens = [], [], 0
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                text = shape.text.strip()
                total_tokens += len(ENCODER.encode(text))
                texts.append(text)
            elif shape.shape_type == 13:  # Picture
                image_data = process_image(shape.image)
                if image_data:
                    total_tokens += calculate_image_tokens(image_data)
                    images.append(image_data)
        return PageContent(slide_number, texts, images, total_tokens)

    @staticmethod
    def create_chunks_from_page_groups(
        groups: list[list[PageContent]], doc: Document, is_whole: bool = False
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
                    DocCat.PPTX if not is_whole else DocCat.PPTX_AS_A_WHOLE,
                    getattr(doc, Col.DOC_DESC, DEFAULT_DOC_DESC),
                    group,
                    page_ids,
                )
            )
        return page_chunks


@register_handler
class WholePPTXChunker(BaseDocChunker):
    """
    Handler for treating an entire PPTX document as a single chunk.
    It extracts all slides and content but does not split them.
    """

    @property
    def supported_categories(self) -> list[str]:
        return [DocCat.PPTX_AS_A_WHOLE]

    async def chunk(self, item: DocAndContent) -> list[Chunk]:
        """Processes a single PPTX file, treating the entire document as one chunk."""
        if not isinstance(item.content, bytes):
            logging.warning(
                f"Expected bytes for PPTX {item.doc.id}, got non-bytes."
            )
            return []

        # 1. Extract all slides and their content, same as the standard chunker
        presentation = load_presentation_from_bytes(item.content)
        slides = [
            PPTXChunker.process_slide(slide, i)
            for i, slide in enumerate(get_visible_slides(presentation), 1)
        ]

        # 2. Treat all slides as a single group to create one chunk
        merged_slide_groups = [slides] if slides else []

        # 3. Create the single chunk from this group
        return PPTXChunker.create_chunks_from_page_groups(
            merged_slide_groups, item.doc, is_whole=True
        )

    async def to_markdown(self, item: DocAndContent) -> str:
        """Extracts all text from a PPTX document into a single string."""
        # This logic is identical to the standard PPTXChunker
        if not isinstance(item.content, bytes):
            return ""

        def sync_process_pptx():
            presentation = load_presentation_from_bytes(item.content)
            all_texts = []
            for i, slide in enumerate(get_visible_slides(presentation), 1):
                slide_content = PPTXChunker.process_slide(slide, i)
                all_texts.extend(slide_content.texts)
            return "\n\n".join(all_texts)

        return await aio.to_thread(sync_process_pptx)


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
            logging.warning(
                f"Expected bytes for PPTX {item.doc.id}, got non-bytes."
            )
            return []

        try:
            # 1. Asynchronously convert PPTX slides to PIL images
            slide_images = await convert_pptx_to_images(item.content)
        except Exception as e:
            logging.error(
                f"Failed to convert PPTX to images for doc {item.doc.id}: {e}",
                exc_info=True,
            )
            return []

        # 2. Pass content and images to the synchronous processing function in a thread
        pages = await aio.to_thread(
            self._extract_and_combine, item.content, slide_images
        )

        if not pages:
            logging.warning(
                f"Chunking resulted in 0 pages for doc {item.doc.id}."
            )
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
            logging.error(
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
                logging.error(
                    f"Failed to convert PPTX to markdown: {e}", exc_info=True
                )
                return ""

        return await aio.to_thread(sync_process_pptx)
