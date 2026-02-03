from typing import Protocol


class FileStorage(Protocol):
    async def read_raw_file_by_url(self, url: str) -> bytes:
        raise NotImplementedError


class EmbeddingModel(Protocol):
    """An interface for any model that can create embeddings."""

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError
