from typing import Any, Protocol


class FileStorage(Protocol):
    async def read_raw_file_by_url(self, url: str) -> bytes: ...


class EmbeddingModel(Protocol):
    """An interface for any model that can create embeddings."""

    def embed_query(self, text: str, **kwargs: Any) -> list[float]: ...

    def embed_documents(
        self, texts: list[str], chunk_size: int | None = None, **kwargs: Any
    ) -> list[list[float]]: ...
