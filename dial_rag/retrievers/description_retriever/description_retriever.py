import asyncio
import logging
import sys
import time
from typing import Any, AsyncGenerator, List, Sequence, Tuple

import numpy as np
from langchain_community.callbacks import OpenAICallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain_openai.chat_models.base import BaseChatOpenAI

from dial_rag.content_stream import SupportsWriteStr
from dial_rag.dial_config import DialConfig
from dial_rag.document_record import DocumentRecord, MultiEmbeddings
from dial_rag.embeddings.embeddings import bge_embedding, build_embeddings
from dial_rag.index_record import RetrievalType
from dial_rag.llm import LlmConfig, create_llm
from dial_rag.resources.dial_limited_resources import (
    AsyncGeneratorWithTotal,
    DialLimitedResources,
    map_with_resource_limits,
)
from dial_rag.retrievers.description_retriever.page_description import (
    PageDescription,
)
from dial_rag.retrievers.description_retriever.prompts import (
    PAGE_DESCRIPTION_PROMPT_TEMPLATE,
)
from dial_rag.retrievers.embeddings_index import (
    EmbeddingsIndex,
    create_index_by_page,
    pack_multi_embeddings,
)
from dial_rag.retrievers.page_image_retriever_utils import extract_page_images
from dial_rag.utils import timed_block

MAX_PNG_SIZE_FOR_DESCRIPTION = 800
EXTRACT_PAGES_KWARGS = {"scaled_size": MAX_PNG_SIZE_FOR_DESCRIPTION}

# Error message in the openai library tells to use math.inf, but the type for the max_retries is int
MAX_RETRIES = 1_000_000_000  # One billion retries should be enough


logger = logging.getLogger(__name__)


class DescriptionIndexConfig(BaseModel):
    llm_config: LlmConfig
    estimated_task_tokens: int
    time_limit_multiplier: float = 1.5
    min_time_limit_sec: float = 5 * 60


class DescriptionRetriever(BaseRetriever):
    index: EmbeddingsIndex

    def __init__(self, document_records: List[DocumentRecord], k: int = 4):
        # description_embeddings_index is just a list of lists of embeddings by the page number
        # we need to convert it to a list of EmbeddingIndexItem
        indexes = [
            create_index_by_page(doc.chunks, doc.description_embeddings_index)
            for doc in document_records
        ]

        super().__init__(
            index=EmbeddingsIndex(
                retrieval_type=RetrievalType.IMAGE,
                indexes=indexes,
                limit=k,
            )
        )

    def _find_relevant_documents(self, query_emb: np.ndarray) -> List[Document]:
        return self.index.find(query=query_emb)

    def _get_relevant_documents(
        self, query: str, *args, **kwargs
    ) -> List[Document]:
        query_emb = np.array(bge_embedding.embed_query(query))
        return self._find_relevant_documents(query_emb)

    async def _aget_relevant_documents(self, query: str, *args, **kwargs):
        query_emb = np.array(await bge_embedding.aembed_query(query))
        return await asyncio.get_running_loop().run_in_executor(
            None, self._find_relevant_documents, query_emb
        )

    @staticmethod
    def has_index(document_records: List[DocumentRecord]) -> bool:
        return any(
            doc.description_embeddings_index is not None
            for doc in document_records
        )

    @staticmethod
    async def build_index(
        dial_config: DialConfig,
        dial_limited_resources: DialLimitedResources,
        index_config: DescriptionIndexConfig,
        doc_bytes: bytes,
        mime_type: str,
        stageio: SupportsWriteStr = sys.stderr,
    ) -> MultiEmbeddings | None:
        async with timed_block("Building Description indexes", stageio):
            logger.debug(f"Building Description indexes.")

            model_name = index_config.llm_config.model_deployment_name
            llm = create_llm(
                dial_config,
                LlmConfig(
                    model_deployment_name=model_name,
                    max_prompt_tokens=0,
                    max_retries=MAX_RETRIES,
                ),
            )

            extracted_images = await extract_page_images(
                mime_type=mime_type,
                original_document=doc_bytes,
                extract_pages_kwargs=EXTRACT_PAGES_KWARGS,
                stageio=stageio,
            )
            if extracted_images is None:
                return None

            pages_embeddings = await _calculate_embeddings(
                llm,
                dial_limited_resources,
                index_config,
                extracted_images,
                stageio,
            )

            return pages_embeddings


async def _calculate_embeddings(
    llm,
    dial_limited_resources: DialLimitedResources,
    index_config: DescriptionIndexConfig,
    extracted_images: AsyncGeneratorWithTotal,
    stageio: SupportsWriteStr,
) -> MultiEmbeddings:
    stageio.write("Collect pages descriptions\n")

    page_descriptions = await map_with_resource_limits(
        dial_limited_resources,
        extracted_images,
        lambda image: _get_page_description(llm, image),
        index_config.estimated_task_tokens,
        llm.model_name,
        stageio,
        index_config.time_limit_multiplier,
        index_config.min_time_limit_sec,
    )

    stageio.write("Calculate page embeddings\n")
    page_indexes, description_chunks = _extract_chunks(page_descriptions)
    embeddings = await build_embeddings(description_chunks, stageio=stageio)
    return pack_multi_embeddings(
        page_indexes, embeddings, extracted_images.total
    )


def _extract_chunks(
    page_descriptions: Sequence[PageDescription],
) -> Tuple[List[int], List[str]]:
    page_indexes: List[int] = []
    description_chunks: List[str] = []
    for page_index, description in enumerate(page_descriptions):
        for chunk in description.to_chunks():
            page_indexes.append(page_index)
            description_chunks.append(chunk)
    return page_indexes, description_chunks


def _build_prompt(image_base64, image_details, image_max_size):
    prompt_template = PAGE_DESCRIPTION_PROMPT_TEMPLATE
    content = [{"type": "text", "text": prompt_template}]
    image_content = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{image_base64}",
            "detail": image_details,  # 'low', 'high', or 'auto'
            "resize": image_max_size,
        },
    }
    content.append(image_content)
    return content


async def _invoke_image_list_prompt(llm: BaseChatOpenAI, prompt: Any) -> str:
    start_time = time.perf_counter()

    message = HumanMessage(prompt)

    cb = OpenAICallbackHandler()
    response = await llm.ainvoke(input=[message], config={"callbacks": [cb]})

    end_time = time.perf_counter()
    logger.debug(f"LLM Time ({llm}): {end_time - start_time:.2f}s")
    logger.debug(f"LLM Response: {response.content}")
    logger.debug(
        f"{cb.total_tokens=} ({cb.prompt_tokens=}, {cb.completion_tokens=})"
    )

    return _get_fixed_json(response.content)


async def _get_page_description(
    llm, page_bitmap_base64: str
) -> PageDescription:
    logger.debug("Generated description for the page.")
    prompt = _build_prompt(
        page_bitmap_base64, "low", MAX_PNG_SIZE_FOR_DESCRIPTION
    )
    page_description_str = await _invoke_image_list_prompt(llm, prompt)

    return PageDescription.from_json_str(page_description_str)


def _get_fixed_json(text: str) -> str:
    """
    Extract JSON from text
    """
    text = text.replace(", ]", "]").replace(",]", "]").replace(",\n]", "]")

    # check if JSON is in code block
    if "```json" in text:
        open_bracket = text.find("```json")
        close_bracket = text.rfind("```")
        if open_bracket != -1 and close_bracket != -1:
            return text[open_bracket + 7 : close_bracket].strip()

    # check if JSON is in brackets
    tmp_text = text.replace("{", "[").replace("}", "]")
    open_bracket = tmp_text.find("[")
    if open_bracket == -1:
        return text

    close_bracket = tmp_text.rfind("]")
    if close_bracket == -1:
        return text

    return text[open_bracket : close_bracket + 1]


def split_text_by_separators(
    text: str, separators: list[str], min_word_count: int
) -> list[str]:
    """
    Split text by separators
    """
    result = []
    current_chunk = ""
    for char in text:
        if char in separators:  # it's a separator
            # we need to have at least min_word_count words in the chunk
            if current_chunk and len(current_chunk.split()) >= min_word_count:
                result.append(current_chunk)
                current_chunk = ""
            else:
                current_chunk += char
        else:
            current_chunk += char
    if current_chunk:
        result.append(current_chunk)
    return result
