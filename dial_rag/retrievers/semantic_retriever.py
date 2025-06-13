import asyncio
import logging

import numpy as np
import sys
from typing import List
from docarray import DocList
from langchain.schema import BaseRetriever, Document

from dial_rag.content_stream import SupportsWriteStr
from dial_rag.document_record import DocumentRecord, Chunk, MultiEmbeddings
from dial_rag.embeddings.embeddings import bge_embedding, build_embeddings
from dial_rag.index_record import RetrievalType
from dial_rag.retrievers.embeddings_index import EmbeddingsIndex, pack_simple_embeddings, create_index_by_chunk

from dial_rag.utils import timed_block

logger = logging.getLogger(__name__)


class SemanticRetriever(BaseRetriever):
    index: EmbeddingsIndex

    def __init__(self, document_records: List[DocumentRecord], k: int = 1):
        indexes = [
            create_index_by_chunk(doc.embeddings_index)
            for doc in document_records if doc.embeddings_index is not None
        ]

        super().__init__(index=EmbeddingsIndex(
            retrieval_type=RetrievalType.TEXT,
            indexes=indexes,
            limit=k,
        ))

    def _find_relevant_documents(self, query_emb: np.ndarray) -> List[Document]:
        return self.index.find(query=query_emb)

    def _get_relevant_documents(self, query: str, *args, **kwargs) -> List[Document]:
        query_emb = np.array(bge_embedding.embed_query(query))
        return self._find_relevant_documents(query_emb)

    async def _aget_relevant_documents(self, query: str, *args, **kwargs):
        query_emb = np.array(await bge_embedding.aembed_query(query))
        return await asyncio.get_running_loop().run_in_executor(None, self._find_relevant_documents, query_emb)

    @staticmethod
    async def build_index(chunks: DocList[Chunk], stageio: SupportsWriteStr=sys.stderr) -> MultiEmbeddings:
        async with timed_block("Building Semantic indexes", stageio):
            logger.debug(f"Building Semantic indexes.")
            texts_gen = (chunk.text for chunk in chunks)
            embeddings = await build_embeddings(texts_gen, stageio)
            return pack_simple_embeddings(embeddings)
