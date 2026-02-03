from abc import ABC, abstractmethod
from typing import Any

from mindmap.generators import base
from mindmap.generators.univ_gen.common.interfaces import EmbeddingModel


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

    @abstractmethod
    def translate_graph_files(self, raw_data: dict) -> base.GraphFiles:
        """Translates raw backend data into clean internal structs."""
