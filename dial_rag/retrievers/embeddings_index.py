from enum import StrEnum
from typing import Iterable, List

import numpy as np
from docarray import BaseDoc, DocList
from docarray.typing import ID, NdArray
from docarray.utils.find import find as docarray_find
from langchain.schema import Document

from dial_rag.document_record import Chunk, ItemEmbeddings, MultiEmbeddings
from dial_rag.index_record import (
    EmbeddingIndexItem,
    RetrievalType,
    to_metadata_doc,
)


class Metric(StrEnum):
    COSINE_SIM = "cosine_sim"
    EUCLIDEAN_DIST = "euclidean_dist"
    SQEUCLIDEAN_DIST = "sqeuclidean_dist"


class SearchItem(BaseDoc):
    id: ID | None = None  # Disable random ID generation for performance reasons
    doc_index: int
    item: EmbeddingIndexItem


class EmbeddingsIndex:
    retrieval_type: RetrievalType
    search_items: DocList[SearchItem]
    metric: str
    limit: int

    def __init__(
        self,
        retrieval_type: RetrievalType,
        indexes: Iterable[DocList[EmbeddingIndexItem] | None],
        metric: Metric = Metric.SQEUCLIDEAN_DIST,
        limit: int = 1,
    ):
        self.retrieval_type = retrieval_type
        self.metric = metric
        self.limit = limit
        self.search_items = DocList[SearchItem]()
        for i, index in enumerate(indexes):
            if index is None:
                continue
            self.search_items.extend(
                SearchItem(doc_index=i, item=item) for item in index
            )

    def find(self, query: np.ndarray) -> List[Document]:
        find_result = docarray_find(
            index=self.search_items,
            query=query,
            search_field="item__embedding",
            metric=self.metric,
            limit=self.limit,
        )

        return [
            to_metadata_doc(
                result.doc_index,
                result.item.chunk_index,
                retrieval_type=self.retrieval_type,
            )
            for result in find_result.documents
        ]


def _get_page_index(chunk: Chunk) -> int:
    # Page numbers are 1-based
    return chunk.metadata["page_number"] - 1


def to_ndarray(arr: np.ndarray):
    return NdArray(shape=arr.shape, buffer=arr, dtype=arr.dtype)


def create_index_by_page(
    chunks: DocList[Chunk],
    pages_embeddings: MultiEmbeddings | None,
):
    if pages_embeddings is None:
        return DocList([])

    return DocList(
        [
            EmbeddingIndexItem(
                chunk_index=i,
                embedding=to_ndarray(embedding),
            )
            for i, chunk in enumerate(chunks)
            for embedding in pages_embeddings[_get_page_index(chunk)].embeddings
        ]
    )


def create_index_by_chunk(
    chunks_embeddings: MultiEmbeddings,
):
    return DocList(
        [
            EmbeddingIndexItem(
                chunk_index=i,
                embedding=to_ndarray(embedding),
            )
            for i, chunk_embeddings in enumerate(chunks_embeddings)
            for embedding in chunk_embeddings.embeddings
        ]
    )


def pack_multi_embeddings(
    indexes: List[int], embeddings: Iterable[np.ndarray], number_of_pages: int
) -> MultiEmbeddings:
    page_embeddings = [[] for _ in range(number_of_pages)]
    for page_index, embedding in zip(indexes, embeddings):
        page_embeddings[page_index].append(embedding)

    return MultiEmbeddings(
        [
            ItemEmbeddings(
                embeddings=to_ndarray(np.array(embeddings, dtype=np.float32))
            )
            for embeddings in page_embeddings
        ]
    )


def pack_simple_embeddings(embeddings: Iterable[np.ndarray]) -> MultiEmbeddings:
    return MultiEmbeddings(
        [
            ItemEmbeddings(
                embeddings=to_ndarray(np.array([embedding], dtype=np.float32))
            )
            for embedding in embeddings
        ]
    )
