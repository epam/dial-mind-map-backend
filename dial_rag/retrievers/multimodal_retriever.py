import asyncio
import logging
import sys
from typing import List

import numpy as np
from langchain.schema import BaseRetriever, Document
from langchain_core.pydantic_v1 import BaseModel

from dial_rag.content_stream import SupportsWriteStr
from dial_rag.dial_config import DialConfig
from dial_rag.document_record import DocumentRecord, MultiEmbeddings
from dial_rag.embeddings.multimodal_embeddings import MultimodalEmbeddings
from dial_rag.index_record import RetrievalType
from dial_rag.resources.dial_limited_resources import (
    DialLimitedResources,
    map_with_resource_limits,
)
from dial_rag.retrievers.embeddings_index import (
    EmbeddingsIndex,
    Metric,
    create_index_by_page,
    pack_simple_embeddings,
)
from dial_rag.retrievers.page_image_retriever_utils import extract_page_images
from dial_rag.utils import timed_block

# Error message in the openai library tells to use math.inf, but the type for the max_retries is int
MAX_RETRIES = 1_000_000_000  # One billion retries should be enough


logger = logging.getLogger(__name__)


class MultimodalIndexConfig(BaseModel):
    embeddings_model: str
    metric: Metric = Metric.SQEUCLIDEAN_DIST
    image_size: int = 1536
    estimated_task_tokens: int = 1
    time_limit_multiplier: float = 1.5
    min_time_limit_sec: float = 5 * 60


class MultimodalRetriever(BaseRetriever):
    index: EmbeddingsIndex
    dial_config: DialConfig
    index_config: MultimodalIndexConfig

    @staticmethod
    def has_index(document_records: List[DocumentRecord]) -> bool:
        return any(
            doc.multimodal_embeddings_index is not None
            for doc in document_records
        )

    def __init__(
        self,
        dial_config: DialConfig,
        index_config: MultimodalIndexConfig,
        document_records: List[DocumentRecord],
        k: int = 1,
    ):
        # multimodal_embeddings_index is just a list of embeddings by the page number
        # we need to convert it to a list of EmbeddingIndexItem
        indexes = [
            create_index_by_page(doc.chunks, doc.multimodal_embeddings_index)
            for doc in document_records
        ]

        super().__init__(
            index=EmbeddingsIndex(
                retrieval_type=RetrievalType.IMAGE,
                indexes=indexes,
                metric=index_config.metric,
                limit=k,
            ),
            dial_config=dial_config,
            index_config=index_config,
        )

    def _find_relevant_documents(self, query_emb: np.ndarray) -> List[Document]:
        return self.index.find(query=query_emb)

    def _get_relevant_documents(
        self, query: str, *args, **kwargs
    ) -> List[Document]:
        multimodal_embeddings = MultimodalEmbeddings(
            self.dial_config, self.index_config.embeddings_model
        )
        query_emb = np.array(multimodal_embeddings.embed_query(query))
        return self._find_relevant_documents(query_emb)

    async def _aget_relevant_documents(self, query: str, *args, **kwargs):
        multimodal_embeddings = MultimodalEmbeddings(
            self.dial_config, self.index_config.embeddings_model
        )
        query_emb = np.array(await multimodal_embeddings.aembed_query(query))
        return await asyncio.get_running_loop().run_in_executor(
            None, self._find_relevant_documents, query_emb
        )

    @staticmethod
    async def build_index(
        dial_config: DialConfig,
        dial_limited_resources: DialLimitedResources,
        index_config: MultimodalIndexConfig,
        mime_type: str,
        original_document: bytes,
        stageio: SupportsWriteStr = sys.stderr,
    ) -> MultiEmbeddings | None:
        async with timed_block("Building Multimodal indexes", stageio):
            logger.debug(f"Building Multimodal indexes.")

            multimodal_embeddings = MultimodalEmbeddings(
                dial_config,
                index_config.embeddings_model,
                max_retries=MAX_RETRIES,
            )

            extract_pages_kwargs = {"scaled_size": index_config.image_size}

            extracted_images = await extract_page_images(
                mime_type,
                original_document,
                extract_pages_kwargs,
                stageio,
            )
            if extracted_images is None:
                return

            stageio.write("Building image embeddings\n")
            embeddings = await map_with_resource_limits(
                dial_limited_resources,
                extracted_images,
                multimodal_embeddings.aembed_image,
                index_config.estimated_task_tokens,
                index_config.embeddings_model,
                stageio,
                index_config.time_limit_multiplier,
                index_config.min_time_limit_sec,
            )

            return pack_simple_embeddings(embeddings)
