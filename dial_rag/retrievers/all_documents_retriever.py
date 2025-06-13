from typing import List

from langchain.schema import BaseRetriever, Document

from dial_rag.document_record import Chunk, DocumentRecord
from dial_rag.index_record import RetrievalType, to_metadata_doc
from dial_rag.qa_chain import format_attributes


class AllDocumentsRetriever(BaseRetriever):
    metadata_chunks: List[Document]

    @staticmethod
    def _estimated_size(i: int, chunk: Chunk) -> int:
        # For small chunks the overhead for the document name or file path is significant
        # So we have to include the size of the metadata in the size estimation
        return len(chunk.text) + len(format_attributes(i, chunk.metadata)) + 30

    @staticmethod
    def is_within_limit(document_records: List[DocumentRecord]) -> bool:
        total_length = sum(
            AllDocumentsRetriever._estimated_size(i, chunk)
            for i, chunk in enumerate(
                chunk for doc in document_records for chunk in doc.chunks
            )
        )
        return total_length <= 12_000

    def __init__(self, document_records: List[DocumentRecord] = None):
        if document_records is None:
            document_records = []

        metadata_chunks = [
            to_metadata_doc(i, j, RetrievalType.TEXT)
            for i, doc in enumerate(document_records)
            for j in range(len(doc.chunks))
        ]

        super().__init__(metadata_chunks=metadata_chunks)

    def _get_relevant_documents(self, query: str, *args, **kwargs):
        return self.metadata_chunks

    async def _aget_relevant_documents(self, query: str, *args, **kwargs):
        return self.metadata_chunks
