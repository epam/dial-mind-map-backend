import logging
from typing import List

from docarray import BaseDoc, DocList
from docarray.typing import ID, NdArray
from langchain.schema import Document

from dial_rag.index_record import TextIndexItem

FORMAT_VERSION: int = 11

logger = logging.getLogger(__name__)


class Chunk(BaseDoc):
    id: ID | None = None  # Disable random ID generation for performance reasons
    text: str
    metadata: dict

    def to_langchain_doc(self) -> Document:
        return Document(
            page_content=self.text,
            metadata=self.metadata,
        )


class MultimodalIndexSettings(BaseDoc):
    id: ID | None = None  # Disable random ID generation for performance reasons
    embeddings_model: str


class IndexSettings(BaseDoc):
    id: ID | None = None  # Disable random ID generation for performance reasons
    multimodal_index: MultimodalIndexSettings | None
    use_description_index: bool


class ItemEmbeddings(BaseDoc):
    """Represents a list of embeddings associated with a single item (chunk or page)."""

    id: ID | None = None  # Disable random ID generation for performance reasons
    embeddings: NdArray["i", "x"]


MultiEmbeddings = DocList[ItemEmbeddings]


class DocumentRecord(BaseDoc):
    format_version: int | None
    index_settings: IndexSettings
    id: ID | None = None  # Disable random ID generation for performance reasons
    chunks: DocList[Chunk]
    text_index: DocList[TextIndexItem] | None
    embeddings_index: MultiEmbeddings | None
    multimodal_embeddings_index: MultiEmbeddings | None
    description_embeddings_index: MultiEmbeddings | None
    mime_type: str
    original_file_name: str
    original_document: bytes


async def build_chunks_list(chunk_docs: List[Document]) -> DocList:
    logger.debug(f"Cutting text chunks.")
    chunks = DocList(
        [
            Chunk(
                text=chunk.page_content,
                metadata=chunk.metadata,
            )
            for chunk in chunk_docs
        ]
    )

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks
