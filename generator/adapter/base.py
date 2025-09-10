from abc import ABC, abstractmethod
from typing import Any

from generator.common import structs
from generator.common.interfaces import EmbeddingModel


class AbstractAdapter(ABC):
    """
    Defines the contract that all backend adapters must implement.
    The rest of the application will interact with this interface.
    """

    gm_contract: Any

    graph_files: Any

    @abstractmethod
    def get_embeddings_model(self) -> EmbeddingModel:
        """Returns an instance of the embeddings model."""
        pass

    @abstractmethod
    def translate_graph_files(self, raw_data: dict) -> structs.GraphFiles:
        """Translates raw backend data into clean internal structs."""
        pass
