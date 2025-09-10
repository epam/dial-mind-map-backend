from abc import ABC, abstractmethod

from ..structs import Chunk, DocAndContent


class BaseDocChunker(ABC):
    """
    Abstract base class for a document chunking strategy. Each handler
    for a specific document type must implement this interface.
    """

    @property
    @abstractmethod
    def supported_categories(self) -> list[str]:
        """A list of DocCat categories this handler supports."""
        pass

    @abstractmethod
    async def chunk(self, doc_with_content: DocAndContent) -> list[Chunk]:
        """
        Chunks a single document.

        Args:
            doc_with_content: The document and its raw content.

        Returns:
            A list of Chunk objects derived from the document.
        """
        pass

    @abstractmethod
    async def to_markdown(self, doc_with_content: DocAndContent) -> str:
        """
        Converts a single document's content to a markdown string.

        Args:
            doc_with_content: The document and its raw content.

        Returns:
            A string containing the markdown representation of the
            document.
        """
        pass
