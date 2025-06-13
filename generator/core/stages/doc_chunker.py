import asyncio as aio
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from itertools import chain as iter_chain

import numpy as np
import pandas as pd
import tiktoken
from langchain_core.documents import Document as LCDoc
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pptx.shapes.picture import Picture
from pptx.slide import Slide

from general_mindmap.models.request import Document as MindMapDoc
from general_mindmap.v2.dial.client import DialClient

from ...utils.constants import DEFAULT_CHAT_MODEL_NAME
from ...utils.constants import DataFrameCols as Col
from ...utils.constants import DocCategories as DocCat
from ...utils.constants import DocContentTypes, DocTypes
from ...utils.constants import FrontEndStatuses as Fes
from ...utils.dial import read_file_bytes_from_dial
from ...utils.frontend_handler import put_status
from ...utils.logger import logging
from ...utils.pdf_handler import get_text_pages, page_to_base64
from ...utils.pptx_handler import PPTXHandler as Ph
from ...utils.web_handler import conv_html_to_md, fetch_html
from ..actions.docs import extract_doc_title


@dataclass
class CatChunkInputs:
    """
    Input components for a document category.

    Attributes:
        chunk_method: A function that defines
            how to chunk the input documents.
        doc_cat: Document category
        docs: A list of Mind Map documents to be chunked.
        client: Dial Client if applicable
    """

    doc_cat: str | None
    chunk_method: Callable | None
    docs: list[MindMapDoc] = field(default_factory=list)
    client: DialClient | None = None

    @property
    def inputs(
        self,
    ) -> (
        tuple[list[MindMapDoc], str] | tuple[list[MindMapDoc], str, DialClient]
    ):
        if self.client:
            return self.docs, self.doc_cat, self.client
        return self.docs, self.doc_cat


@dataclass
class PageContent:
    page_id: int
    texts: list[str]
    images: list[str]  # base64 encoded images
    tokens: int = 0


@dataclass
class Chunk:
    """
    Attributes:
        lc_doc: Chunk content in a form of a LangChain document.
        doc_id: Chunk source document id.
        doc_url: Chunk source document url.
        doc_title: Chunk source document title.
        doc_cat: Chunk source document category.
        page_contents: List of chunk page contents.
        page_ids: List of page ids in the chunk if applicable.
    """

    lc_doc: LCDoc
    doc_id: str
    doc_url: str
    doc_title: str
    doc_cat: str
    page_contents: list[PageContent] = field(
        default_factory=list, metadata={"pop_in_as_dict": True}
    )
    page_ids: list[int] = field(default_factory=list)

    @property
    def _content(self) -> str | list[dict]:
        if self.page_contents:
            # noinspection PyTypeChecker
            return [asdict(content) for content in self.page_contents]
        header = self.lc_doc.metadata.get(DocChunker.HEADER_TO_SPLIT_ON)
        if header is not None:
            return "# " + header + "\n" + self.lc_doc.page_content
        return self.lc_doc.page_content

    def as_dict(self) -> dict[str, int | LCDoc | str]:
        # noinspection PyTypeChecker
        data = asdict(self)
        # noinspection PyTypeChecker
        for f in fields(self):
            if f.metadata.get("pop_in_as_dict", False):
                data.pop(f.name)
        data[Col.CONTENT] = self._content
        return data


class DocChunker:
    encoder = tiktoken.encoding_for_model(DEFAULT_CHAT_MODEL_NAME)

    CHUNK_SIZE_LIMIT = 2048
    MAX_CHUNK_IMG_NUM = 50  # OpenAI requirement
    HEADER_TO_SPLIT_ON = "Header 1"

    def __init__(self, queue: aio.Queue):
        self.queue = queue

    @staticmethod
    def _get_doc_cat(doc: MindMapDoc) -> str:
        if doc.type == DocTypes.LINK:
            return DocCat.LINK

        if doc.type == DocTypes.FILE:
            match doc.content_type:
                case DocContentTypes.PRESENTATION:
                    return DocCat.PPTX
                case DocContentTypes.HTML:
                    return DocCat.HTML
                case DocContentTypes.PDF:
                    return DocCat.PDF

        return DocCat.UNSUPPORTED

    @staticmethod
    def _get_doc_meta(
        doc_id: str, doc_url: str, doc_cat: str
    ) -> dict[str, str]:
        return {
            Col.DOC_ID: doc_id,
            Col.DOC_URL: doc_url,
            Col.DOC_CAT: doc_cat,
        }

    @staticmethod
    async def _conv_html_to_md_w_meta(
        html: str, metadata: dict[str, str]
    ) -> tuple[str, dict[str, str]] | None:
        doc_url = None
        if metadata[Col.DOC_CAT] == "LINK":
            doc_url = metadata[Col.DOC_URL]
        md, doc_title = conv_html_to_md(html, doc_url)
        metadata[Col.DOC_TITLE] = doc_title
        return md, metadata

    @staticmethod
    def _create_page_chunks(
        merged_chunks: list[list[PageContent]], doc: MindMapDoc, doc_cat: str
    ) -> list[Chunk]:
        page_chunks = []
        doc_title = extract_doc_title(doc.url)
        for chunk in merged_chunks:
            chunk_page_texts = []
            chunk_page_ids = []

            for content in chunk:
                chunk_page_text = f"{content.page_id}: {content.texts}"
                chunk_page_texts.append(chunk_page_text)
                chunk_page_ids.append(content.page_id)

            lc_doc = LCDoc("\n".join(chunk_page_texts))
            page_chunk = Chunk(
                lc_doc,
                doc.id,
                doc.url,
                doc_title,
                doc_cat,
                chunk,
                chunk_page_ids,
            )
            page_chunks.append(page_chunk)

        return page_chunks

    @staticmethod
    async def _form_chunk_df(chunk_groups: list[list[Chunk]]) -> pd.DataFrame:
        chunks: iter_chain[Chunk] = iter_chain.from_iterable(chunk_groups)
        return pd.DataFrame([chunk.as_dict() for chunk in chunks])

    @staticmethod
    def _form_flat_part_df(chunk_df: pd.DataFrame) -> pd.DataFrame:
        chunk_df = chunk_df.copy()
        page_part_col = "page_part"
        chunk_df[page_part_col] = chunk_df.apply(
            lambda row: (
                [None] + row[Col.PAGE_ID]
                if Col.PAGE_ID in row and isinstance(row[Col.PAGE_ID], list)
                else [None]
            ),
            axis=1,
        )

        exploded_df = chunk_df.explode(
            page_part_col, ignore_index=False
        ).reset_index(drop=True)

        exploded_df[Col.PAGE_ID] = [
            () if x is None else (x,) for x in exploded_df[page_part_col]
        ]

        exploded_df[Col.CITATION] = (
            exploded_df[Col.DOC_ID].astype(str)
            + "."
            + np.where(
                exploded_df[page_part_col].isnull(),
                exploded_df[Col.CHUNK_ID].astype(str),
                exploded_df[page_part_col].astype(str),
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

    @classmethod
    async def _fetch_html_w_meta(
        cls, doc: MindMapDoc, doc_cat: str, dial_client: DialClient | None
    ) -> tuple[str | None, dict[str, str]]:
        doc_id, doc_url = doc.id, doc.url

        if dial_client is not None:
            html_bytes = await read_file_bytes_from_dial(dial_client, doc_url)
            html = html_bytes.decode()
        else:
            html = await fetch_html(doc_url)

        return html, cls._get_doc_meta(doc_id, doc_url, doc_cat)

    @classmethod
    async def gather_fetch_html_w_meta(
        cls,
        docs: list[MindMapDoc],
        doc_cat: str,
        dial_client: DialClient | None = None,
    ) -> list[tuple[str | None, dict[str, str]]]:
        tasks = (
            aio.create_task(cls._fetch_html_w_meta(doc, doc_cat, dial_client))
            for doc in docs
        )
        return await aio.gather(*tasks)

    @classmethod
    async def _gather_conv_html_to_md_w_meta(
        cls, htmls_w_meta: list[tuple[str, dict]]
    ) -> list[tuple[str, dict]]:
        tasks = []
        for html, meta in htmls_w_meta:
            if html is not None:
                task = aio.create_task(cls._conv_html_to_md_w_meta(html, meta))
                tasks.append(task)
            else:
                logging.warning(f"{meta[Col.DOC_URL]} processing failed.")
        return await aio.gather(*tasks)

    @classmethod
    def _merge_chunk_data(cls, chunk1: LCDoc, chunk2: LCDoc) -> LCDoc:
        for key, value in chunk2.metadata.items():
            if (
                key == cls.HEADER_TO_SPLIT_ON
                and key in chunk1.metadata
                and chunk1.metadata[key] != chunk2.metadata[key]
            ):
                chunk1.metadata[key] += f" and {value}"
            else:
                chunk1.metadata[key] = value

        merged_content = chunk1.page_content + chunk2.page_content

        return LCDoc(
            page_content=merged_content,
            metadata=chunk1.metadata,
        )

    @classmethod
    async def _split_md(
        cls,
        md_w_meta: tuple[str, dict[str, str]],
        md_splitter: MarkdownHeaderTextSplitter,
        text_splitter: RecursiveCharacterTextSplitter,
    ) -> list[Chunk]:
        md, metadata = md_w_meta

        md_header_splits = md_splitter.split_text(md)
        initial_chunks = text_splitter.split_documents(md_header_splits)

        # Merge chunks that together do not exceed the chunk size limit
        merged_chunks = []
        cur_chunk = None
        current_size = 0

        for initial_chunk in initial_chunks:
            chunk_size = len(cls.encoder.encode(str(initial_chunk)))
            if cur_chunk is None:
                cur_chunk = initial_chunk
                current_size = chunk_size
            elif current_size + chunk_size <= cls.CHUNK_SIZE_LIMIT:
                cur_chunk = cls._merge_chunk_data(cur_chunk, initial_chunk)
                current_size += chunk_size
            else:
                merged_chunks.append(cur_chunk)
                cur_chunk = initial_chunk
                current_size = chunk_size

        if cur_chunk is not None:
            merged_chunks.append(cur_chunk)

        doc_id = metadata[Col.DOC_ID]
        doc_url = metadata[Col.DOC_URL]
        doc_title = metadata[Col.DOC_TITLE]
        doc_cat = metadata[Col.DOC_CAT]

        return [
            Chunk(lc_doc, doc_id, doc_url, doc_title, doc_cat)
            for chunk_id, lc_doc in enumerate(merged_chunks)
        ]

    @classmethod
    async def _split_mds(
        cls, mds_w_meta: list[tuple[str, dict[str, str]]]
    ) -> list[list[Chunk]]:
        headers_to_split_on = [("#", cls.HEADER_TO_SPLIT_ON)]
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
        text_splitter = RecursiveCharacterTextSplitter()

        tasks = (
            cls._split_md(md_w_meta, md_splitter, text_splitter)
            for md_w_meta in mds_w_meta
        )
        return await aio.gather(*tasks)

    @classmethod
    async def _chunk_html_like_docs(
        cls,
        docs: list[MindMapDoc],
        doc_cat: str,
        dial_client: DialClient | None = None,
    ) -> pd.DataFrame:
        htmls_w_meta = await cls.gather_fetch_html_w_meta(
            docs, doc_cat, dial_client
        )
        mds_w_meta = await cls._gather_conv_html_to_md_w_meta(htmls_w_meta)

        if mds_w_meta:
            chunk_groups = await cls._split_mds(mds_w_meta)
            return await cls._form_chunk_df(chunk_groups)
        else:
            return pd.DataFrame()

    @classmethod
    def _process_slide(cls, slide: Slide, slide_number: int) -> PageContent:
        texts = []
        images = []
        total_tokens = 0

        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text = shape.text.strip()
                if text:
                    text_tokens = len(cls.encoder.encode(text))
                    total_tokens += text_tokens
                    texts.append(text)
            elif shape.shape_type == 13:
                shape: Picture
                image_data = Ph.process_image(shape.image)
                if image_data:
                    img_tokens = Ph.calculate_image_tokens(image_data)
                    total_tokens += img_tokens
                    images.append(image_data)

        return PageContent(
            page_id=slide_number,
            texts=texts,
            images=images,
            tokens=total_tokens,
        )

    @classmethod
    def _split_page_by_images(
        cls, page_content: PageContent, max_images: int
    ) -> list[list[PageContent]]:
        chunks = []
        images = page_content.images

        for i in range(0, len(images), max_images):
            chunk_images = images[i : i + max_images]

            # Recalculate tokens for the partial content
            text_tokens = sum(
                len(cls.encoder.encode(text)) for text in page_content.texts
            )
            image_tokens = sum(
                Ph.calculate_image_tokens(img) for img in chunk_images
            )
            total_tokens = image_tokens + text_tokens

            partial_content = PageContent(
                page_id=page_content.page_id,
                texts=page_content.texts,
                images=chunk_images,
                tokens=total_tokens,
            )
            chunks.append([partial_content])

        return chunks

    @classmethod
    def _should_start_new_chunk(
        cls,
        current_size: int,
        page_content: PageContent,
        current_items: int,
        max_images: int,
    ) -> bool:
        return (
            current_size + page_content.tokens > cls.CHUNK_SIZE_LIMIT
            or current_items + len(page_content.images) > max_images
        )

    @classmethod
    async def _split_pptx(
        cls, doc: MindMapDoc, doc_cat: str, dial_client: DialClient
    ) -> list[Chunk]:
        presentation = await Ph.load_presentation(dial_client, doc.url)

        merged_chunks = []
        current_chunk: list[PageContent] = []
        current_size = 0

        for slide_number, slide in enumerate(
            Ph.get_visible_slides(presentation), 1
        ):
            slide_content = cls._process_slide(slide, slide_number)

            # Split slide if it contains too many images
            if len(slide_content.images) > cls.MAX_CHUNK_IMG_NUM:
                split_chunks = cls._split_page_by_images(
                    slide_content, cls.MAX_CHUNK_IMG_NUM
                )

                for split_chunk in split_chunks:
                    split_content = split_chunk[0]

                    if split_content.tokens > cls.CHUNK_SIZE_LIMIT:
                        merged_chunks.append([split_content])
                    else:
                        if cls._should_start_new_chunk(
                            current_size,
                            split_content,
                            len(current_chunk),
                            cls.MAX_CHUNK_IMG_NUM,
                        ):
                            if current_chunk:
                                merged_chunks.append(current_chunk)
                            current_chunk = [split_content]
                            current_size = split_content.tokens
                        else:
                            current_chunk.append(split_content)
                            current_size += split_content.tokens
                continue

            # Check if adding this slide would exceed limits
            if cls._should_start_new_chunk(
                current_size,
                slide_content,
                len(current_chunk),
                cls.MAX_CHUNK_IMG_NUM,
            ):
                if current_chunk:
                    merged_chunks.append(current_chunk)
                current_chunk = [slide_content]
                current_size = slide_content.tokens
            else:
                current_chunk.append(slide_content)
                current_size += slide_content.tokens

        if current_chunk:
            merged_chunks.append(current_chunk)

        return cls._create_page_chunks(merged_chunks, doc, doc_cat)

    @classmethod
    async def _chunk_pptx_docs(
        cls, docs: list[MindMapDoc], doc_cat: str, dial_client: DialClient
    ) -> pd.DataFrame:
        tasks = (cls._split_pptx(doc, doc_cat, dial_client) for doc in docs)
        chunk_groups = await aio.gather(*tasks)
        return await cls._form_chunk_df(chunk_groups)

    @classmethod
    async def _split_pdf(
        cls, doc: MindMapDoc, doc_cat: str, dial_client: DialClient
    ) -> list[Chunk]:
        merged_chunks = []
        current_chunk: list[PageContent] = []
        current_size = 0

        pdf_bytes = await read_file_bytes_from_dial(dial_client, doc.url)

        text_pages = await get_text_pages(pdf_bytes)
        for page_number, page in enumerate(text_pages, 1):
            total_tokens = 0

            text = page.page_content
            text_tokens = len(cls.encoder.encode(text))
            total_tokens += text_tokens

            image_data = page_to_base64(pdf_bytes, page_number)
            img_tokens = Ph.calculate_image_tokens(image_data)
            total_tokens += img_tokens

            texts = [text]
            images = [image_data]

            page_content = PageContent(
                page_id=page_number,
                texts=texts,
                images=images,
                tokens=total_tokens,
            )

            # Split page if it contains too many images
            if len(page_content.images) > cls.MAX_CHUNK_IMG_NUM:
                split_chunks = cls._split_page_by_images(
                    page_content, cls.MAX_CHUNK_IMG_NUM
                )

                for split_chunk in split_chunks:
                    split_content = split_chunk[0]

                    if split_content.tokens > cls.CHUNK_SIZE_LIMIT:
                        merged_chunks.append([split_content])
                    else:
                        if cls._should_start_new_chunk(
                            current_size,
                            split_content,
                            len(current_chunk),
                            cls.MAX_CHUNK_IMG_NUM,
                        ):
                            if current_chunk:
                                merged_chunks.append(current_chunk)
                            current_chunk = [split_content]
                            current_size = split_content.tokens
                        else:
                            current_chunk.append(split_content)
                            current_size += split_content.tokens
                continue

            # Check if adding this page would exceed limits
            if cls._should_start_new_chunk(
                current_size,
                page_content,
                len(current_chunk),
                cls.MAX_CHUNK_IMG_NUM,
            ):
                if current_chunk:
                    merged_chunks.append(current_chunk)
                current_chunk = [page_content]
                current_size = page_content.tokens
            else:
                current_chunk.append(page_content)
                current_size += page_content.tokens

        if current_chunk:
            merged_chunks.append(current_chunk)

        return cls._create_page_chunks(merged_chunks, doc, doc_cat)

    @classmethod
    async def _chunk_pdf_docs(
        cls, docs: list[MindMapDoc], doc_cat: str, dial_client: DialClient
    ) -> pd.DataFrame:
        tasks = (cls._split_pdf(doc, doc_cat, dial_client) for doc in docs)
        chunk_groups = await aio.gather(*tasks)
        return await cls._form_chunk_df(chunk_groups)

    @classmethod
    def _get_chunk_inputs(
        cls, dial_client: DialClient
    ) -> dict[str, CatChunkInputs]:
        return {
            DocCat.LINK: CatChunkInputs(DocCat.LINK, cls._chunk_html_like_docs),
            DocCat.HTML: CatChunkInputs(
                DocCat.HTML, cls._chunk_html_like_docs, client=dial_client
            ),
            DocCat.PPTX: CatChunkInputs(
                DocCat.PPTX, cls._chunk_pptx_docs, client=dial_client
            ),
            DocCat.PDF: CatChunkInputs(
                DocCat.PDF, cls._chunk_pdf_docs, client=dial_client
            ),
            DocCat.UNSUPPORTED: CatChunkInputs(None, None),
        }

    async def _wrap_up(self, result: tuple[pd.DataFrame, ...] | None = None):
        await self.queue.put(None)
        logging.info("Input Preparation end")
        return result

    @classmethod
    async def _put_chunk_result_status(
        cls,
        num_docs: int,
        num_chunk: int,
    ):
        document_str = "document" if num_docs == 1 else "documents"
        status_chunk_str = "chunk" if num_chunk == 1 else "chunks"
        status_verb_str = "was" if num_docs == 1 else "were"

        logging.info(
            (
                f"{num_docs} {document_str} {status_verb_str} "
                f"split into \n{num_chunk} {status_chunk_str}"
            ),
        )

    async def chunk_docs(
        self, docs: list[MindMapDoc], dial_client: DialClient
    ) -> tuple[pd.DataFrame | None, ...]:
        """
        1. Split the initial docs list into lists by document category.
        Document category is determined by a combination of
        Mind Map document type and content type.
        2. Run chunking for documents in each category.
        3. Concat chunking results into a single chunk_df
        and do additional processing on it.
        """
        logging.info("Input Preparation start")
        await put_status(self.queue, Fes.LOAD_DOCS)

        # 1. Split docs by categories
        chunk_inputs = self._get_chunk_inputs(dial_client)
        for doc in docs:
            doc_cat = self._get_doc_cat(doc)
            if doc.type == DocTypes.FILE:
                logging.info(
                    f"Doc {doc.id} {doc_cat}: {extract_doc_title(doc.url)}"
                )
            else:
                logging.info(f"Doc {doc.id} {doc_cat}: {doc.url}")
            chunk_inputs[doc_cat].docs.append(doc)

        # 2. Run chunking for each document category
        tasks = []
        for doc_cat, chunk_input in chunk_inputs.items():
            if input_docs := chunk_input.docs:
                chunk_method = chunk_input.chunk_method
                if chunk_method is not None:
                    inputs = chunk_input.inputs
                    task = aio.create_task(chunk_method(*inputs))
                    tasks.append(task)
                else:
                    for input_doc in input_docs:
                        log_msg = f"Document {input_doc.id} is not supported."
                        logging.warning(log_msg)

        init_chunk_dfs = await aio.gather(*tasks)

        # 3. Prepare output chunk_df
        if not init_chunk_dfs:
            return await self._wrap_up()
        chunk_df = pd.concat(init_chunk_dfs, ignore_index=True)
        if chunk_df.empty:
            return await self._wrap_up()

        chunk_df[Col.CHUNK_ID] = chunk_df.groupby(Col.DOC_ID).cumcount()
        chunk_df[Col.CHUNK_ID] += 1

        chunk_df.reset_index(names=Col.FLAT_CHUNK_ID, inplace=True)
        chunk_df[Col.FLAT_CHUNK_ID] += 1

        flat_part_df = self._form_flat_part_df(chunk_df)
        await self._put_chunk_result_status(len(docs), len(chunk_df))
        return await self._wrap_up((chunk_df, flat_part_df))

    async def chunk_add_docs(
        self,
        docs: list[MindMapDoc],
        dial_client: DialClient,
        start_part_id: int,
    ) -> tuple[pd.DataFrame | None, ...] | None:
        """
        1. Split the initial docs list into lists by document category.
        Document category is determined by a combination of
        Mind Map document type and content type.
        2. Run chunking for documents in each category.
        3. Concat chunking results into a single chunk_df
        and do additional processing on it.
        """
        logging.info("Input Preparation start")
        await put_status(self.queue, Fes.LOAD_DOCS)

        # 1. Split docs by categories
        chunk_inputs = self._get_chunk_inputs(dial_client)
        for doc in docs:
            doc_cat = self._get_doc_cat(doc)
            if doc.type == DocTypes.FILE:
                logging.info(
                    f"Doc {doc.id} {doc_cat}: {extract_doc_title(doc.url)}"
                )
            else:
                logging.info(f"Doc {doc.id} {doc_cat}: {doc.url}")
            chunk_inputs[doc_cat].docs.append(doc)

        # 2. Run chunking for each document category
        tasks = []
        for doc_cat, chunk_input in chunk_inputs.items():
            if input_docs := chunk_input.docs:
                chunk_method = chunk_input.chunk_method
                if chunk_method is not None:
                    inputs = chunk_input.inputs
                    task = aio.create_task(chunk_method(*inputs))
                    tasks.append(task)
                else:
                    for input_doc in input_docs:
                        log_msg = f"Document {input_doc.id} is not supported."
                        logging.warning(log_msg)

        init_chunk_dfs = await aio.gather(*tasks)

        # 3. Prepare output chunk_df
        chunk_df = pd.concat(init_chunk_dfs, ignore_index=True)
        if chunk_df.empty:
            return await self._wrap_up()

        chunk_df[Col.CHUNK_ID] = chunk_df.groupby(Col.DOC_ID).cumcount()
        chunk_df[Col.CHUNK_ID] += 1

        chunk_df.reset_index(names=Col.FLAT_CHUNK_ID, inplace=True)
        chunk_df[Col.FLAT_CHUNK_ID] += 1

        flat_part_df = self._form_flat_part_df(chunk_df)
        flat_part_df[Col.FLAT_PART_ID] += start_part_id

        await self._put_chunk_result_status(len(docs), len(chunk_df))
        return await self._wrap_up((chunk_df, flat_part_df))
