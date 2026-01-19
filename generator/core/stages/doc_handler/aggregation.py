from langchain_core.documents import Document as LCDoc
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tiktoken import Encoding

from common_utils.logger_config import logger
from generator.chainer import LLMUtils

from .structs import PageContent
from .utils import calculate_image_tokens


def aggregate_text_chunks(
    chunks: list[LCDoc],
    max_chunk_size: int,
    header_to_split_on: str = "Header 1",
    encoder: Encoding = LLMUtils.get_encoding_for_model(),
) -> list[LCDoc]:
    """
    Merges small text chunks into larger ones respecting the token
    limit.
    """
    if not chunks:
        return []

    def _merge_md_chunks(chunk1: LCDoc, chunk2: LCDoc) -> LCDoc:
        merged_meta = chunk1.metadata.copy()
        if (key := header_to_split_on) in chunk2.metadata:
            if key in merged_meta and merged_meta[key] != chunk2.metadata[key]:
                merged_meta[key] += f" and {chunk2.metadata[key]}"
            else:
                merged_meta[key] = chunk2.metadata[key]
        return LCDoc(
            chunk1.page_content + "\n\n" + chunk2.page_content,
            metadata=merged_meta,
        )

    merged_chunks = []
    current_chunk = chunks[0]
    current_size = len(encoder.encode(str(current_chunk)))

    for next_chunk in chunks[1:]:
        next_size = len(encoder.encode(str(next_chunk)))
        if current_size + next_size <= max_chunk_size:
            current_chunk = _merge_md_chunks(current_chunk, next_chunk)
            current_size += next_size
        else:
            merged_chunks.append(current_chunk)
            current_chunk = next_chunk
            current_size = next_size

    merged_chunks.append(current_chunk)
    return merged_chunks


def aggregate_page_content(
    pages: list[PageContent],
    encoder,
    max_chunk_size: int,
    max_chunk_img_num: int,
) -> list[list[PageContent]]:
    """
    Aggregates page-like content (PDF pages, PPTX slides) into chunks,
    ensuring no chunk exceeds token or image limits.
    """
    if not pages:
        return []

    processed_pages = _preprocess_oversized_pages(
        pages, encoder, max_chunk_size
    )

    merged_chunks: list[list[PageContent]] = []
    current_chunk: list[PageContent] = []
    current_size, current_images = 0, 0

    for page in processed_pages:
        num_page_images = len(page.images)
        will_overflow = (
            current_size + page.tokens > max_chunk_size
            or current_images + num_page_images > max_chunk_img_num
        )

        if current_chunk and will_overflow:
            merged_chunks.append(current_chunk)
            current_chunk, current_size, current_images = [], 0, 0

        current_chunk.append(page)
        current_size += page.tokens
        current_images += num_page_images

    if current_chunk:
        merged_chunks.append(current_chunk)
    return merged_chunks


def _preprocess_oversized_pages(
    pages: list[PageContent], encoder, max_chunk_size: int
) -> list[PageContent]:
    """Pre-splits pages that are individually too large."""
    processed_pages: list[PageContent] = []
    for page in pages:
        if page.tokens > max_chunk_size:
            processed_pages.extend(
                _split_oversized_page(page, encoder, max_chunk_size)
            )
        else:
            processed_pages.append(page)
    return processed_pages


def _split_oversized_page(
    page: PageContent, encoder, max_chunk_size: int
) -> list[PageContent]:
    """
    Splits a single oversized PageContent object into smaller ones.
    """
    logger.warning(
        f"Page/Slide {page.page_id} is oversized ({page.tokens} tokens). "
        "Splitting."
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size, chunk_overlap=int(max_chunk_size * 0.1)
    )
    full_text = "\n".join(page.texts)
    split_texts = text_splitter.split_text(full_text)

    sub_pages = []
    for i, text_part in enumerate(split_texts):
        text_tokens = len(encoder.encode(text_part))
        images, img_tokens = [], 0

        if i == 0 and page.images:
            potential_img_tokens = sum(
                calculate_image_tokens(img) for img in page.images
            )
            if (
                potential_img_tokens < max_chunk_size
                and text_tokens + potential_img_tokens <= max_chunk_size
            ):
                images = page.images
                img_tokens = potential_img_tokens
            else:
                logger.warning(
                    f"Image(s) on page {page.page_id} were skipped because "
                    "they would create an oversized chunk."
                )

        sub_pages.append(
            PageContent(
                page_id=page.page_id,
                texts=[text_part],
                images=images,
                tokens=text_tokens + img_tokens,
            )
        )
    return sub_pages
