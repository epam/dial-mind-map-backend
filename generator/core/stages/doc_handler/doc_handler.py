import asyncio as aio
import logging
from itertools import chain as iter_chain

import numpy as np
import pandas as pd

from common_utils.logger_config import logger
from generator.common.constants import DataFrameCols as Col
from generator.core.utils.constants import FrontEndStatuses as Fes
from generator.core.utils.frontend_handler import put_status

from .constants import DocCategories as DocCat
from .constants import DocContentType, DocType
from .registry import get_handler
from .structs import DocAndContent, Document


class DocHandler:
    """
    A class responsible for orchestrating the splitting of various
    document types into smaller, semantically meaningful chunks.
    Can operate in 'chunk' mode (default) or 'whole' mode (one chunk
    per doc).
    """

    _WHOLE_CATEGORY_MAP = {
        DocCat.PDF: DocCat.PDF_AS_A_WHOLE,
        DocCat.PPTX: DocCat.PPTX_AS_A_WHOLE,
        DocCat.HTML: DocCat.HTML_AS_A_WHOLE,
        DocCat.LINK: DocCat.LINK_AS_A_WHOLE,
        DocCat.TXT: DocCat.TXT_AS_A_WHOLE,
    }

    def __init__(self, queue: aio.Queue | None = None, strategy: str = "chunk"):
        """
        Initializes the DocHandler.
        Args:
            queue: An optional asyncio queue for status updates.
            strategy: The chunking strategy.
                      'chunk' (default): Splits documents into smaller
                      chunks.
                      'whole': Treats each document as a single chunk.
        """
        self.queue = queue
        self.strategy = strategy
        logger.info(f"DocHandler initialized with strategy: '{self.strategy}'")

    async def chunk_docs(
        self,
        docs_with_content: list[DocAndContent],
        start_part_id: int = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        """Unified entry point for chunking a set of documents."""
        logger.info(f"Document chunking: Start (Mode: {Fes.LOAD_DOCS})")
        if self.queue:
            await put_status(self.queue, Fes.LOAD_DOCS)

        # 1. Create chunking tasks for all documents
        tasks = []
        for item in docs_with_content:
            doc_cat = self._get_doc_cat(item.doc)
            handler = get_handler(doc_cat)
            if handler:
                tasks.append(aio.create_task(handler.chunk(item)))
            else:
                logger.warning(
                    f"No handler found for doc category: {doc_cat}. Skipping."
                )

        if not tasks:
            return await self._wrap_up()

        # 2. Run tasks and consolidate results
        chunk_groups = await aio.gather(*tasks)
        all_chunks = list(iter_chain.from_iterable(chunk_groups))

        if not all_chunks:
            return await self._wrap_up()

        # 3. Format results into DataFrames
        chunk_df = pd.DataFrame([chunk.as_dict() for chunk in all_chunks])
        chunk_df[Col.CHUNK_ID] = chunk_df.groupby(Col.DOC_ID).cumcount() + 1
        chunk_df.reset_index(names=Col.FLAT_CHUNK_ID, inplace=True)
        chunk_df[Col.FLAT_CHUNK_ID] += 1
        chunk_df[Col.PAGE_ID] = chunk_df[Col.PAGE_ID].apply(tuple)

        flat_part_df = self._form_flat_part_df(chunk_df)
        if start_part_id > 0:
            flat_part_df[Col.FLAT_PART_ID] += start_part_id

        await self._put_chunk_result_status(
            len(docs_with_content), len(chunk_df)
        )
        return await self._wrap_up((chunk_df, flat_part_df))

    async def get_markdown_for_docs(
        self, docs_with_content: list[DocAndContent]
    ) -> list[str]:
        """
        Converts a list of documents into their markdown representation.
        """
        logger.info("Document to markdown conversion: Start")
        tasks = []
        for item in docs_with_content:
            doc_cat = self._get_doc_cat(item.doc)
            handler = get_handler(doc_cat)
            if handler:
                tasks.append(aio.create_task(handler.to_markdown(item)))
            else:
                logger.warning(
                    f"No handler for markdown conversion: {doc_cat}. Skipping."
                )
                tasks.append(aio.create_task(self._empty_markdown()))

        markdown_docs = await aio.gather(*tasks)
        logger.info("Document to markdown conversion: End")
        return markdown_docs

    @staticmethod
    async def _empty_markdown() -> str:
        return ""

    def _get_doc_cat(self, doc: Document) -> str:
        """
        Categorizes a document based on its type, content type, and the
        handler's strategy using a scalable mapping.
        """
        # Step 1: Determine the base category from document properties.
        base_cat = DocCat.UNSUPPORTED
        if doc.type == DocType.LINK:
            base_cat = DocCat.LINK
        elif doc.type == DocType.FILE:
            match doc.content_type:
                case DocContentType.PRESENTATION:
                    base_cat = DocCat.PPTX
                case DocContentType.HTML:
                    base_cat = DocCat.HTML
                case DocContentType.PDF:
                    base_cat = DocCat.PDF
                case DocContentType.TEXT:
                    base_cat = DocCat.TXT

        # Step 2: If the strategy is "whole", try to find a "whole"
        # category from our map.
        if self.strategy == "whole":
            # Use .get() to safely look up the category. If the base
            # category isn't in the map, it returns the base category
            # itself as a fallback.
            return self._WHOLE_CATEGORY_MAP.get(base_cat, base_cat)

        # Step 3: If the strategy is not "whole", just return the base category.
        return base_cat

    @staticmethod
    def _form_flat_part_df(chunk_df: pd.DataFrame) -> pd.DataFrame:
        temp_df = chunk_df.copy()
        temp_df[Col.PAGE_ID] = temp_df[Col.PAGE_ID].apply(
            lambda p: p if p else [np.nan]
        )
        exploded_df = temp_df.explode(Col.PAGE_ID, ignore_index=True)
        exploded_df[Col.PAGE_ID] = [
            (int(pid),) if pd.notna(pid) else ()
            for pid in exploded_df[Col.PAGE_ID]
        ]
        exploded_df[Col.CITATION] = (
            exploded_df[Col.DOC_ID].astype(str)
            + "."
            + np.where(
                exploded_df[Col.PAGE_ID].str.len() > 0,
                # Convert to nullable Int, then to string to remove ".0"
                exploded_df[Col.PAGE_ID].str[0].astype("Int64").astype(str),
                exploded_df[Col.CHUNK_ID].astype(str),
            )
        )

        exploded_df[Col.FLAT_PART_ID] = exploded_df.index + 1

        return exploded_df[
            [
                Col.DOC_ID,
                Col.CHUNK_ID,
                Col.FLAT_CHUNK_ID,
                Col.PAGE_ID,
                Col.FLAT_PART_ID,
                Col.CITATION,
            ]
        ]

    async def _wrap_up(self, result=None):
        if self.queue:
            await self.queue.put(None)
        return result

    @staticmethod
    async def _put_chunk_result_status(num_docs: int, num_chunks: int):
        doc_str = "document" if num_docs == 1 else "documents"
        chunk_str = "chunk" if num_chunks == 1 else "chunks"
        verb_str = "was" if num_docs == 1 else "were"
        logger.info(
            f"{num_docs} {doc_str} {verb_str} split into {num_chunks} "
            f"{chunk_str}"
        )
