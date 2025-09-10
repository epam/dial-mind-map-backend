import logging

from pydantic import ConfigDict, Field, ValidationError

from generator.common import structs
from generator.common.interfaces import EmbeddingModel

from .base import AbstractAdapter


# ======================================================================
# 1. DATA CONTRACT
# ======================================================================
# This class defines the string keys used by the external
# 'general_mindmap' package. It is the single source of truth for all
# hardcoded strings.
class GMContract:
    """Defines the data contract keys for general_mindmap objects."""

    # General
    DATA_KEY = "data"

    # File names
    NODES_FILE = "nodes_file"
    EDGES_FILE = "edges_file"

    # Keys within files
    NODES_KEY = "nodes"
    ROOT_KEY = "root"
    EDGE_KEY = "edges"

    # Node data keys
    NODE_ID = "id"
    NAME = "label"
    QUESTION = "question"
    ANSWER_STR = "details"
    METADATA = "metadata"

    # Node metadata keys
    ANSWER = "answer"
    LVL = "level"
    CLUSTER_ID = "cluster_id"

    # Edge data keys
    EDGE_ID = "id"
    SOURCE = "source"
    TARGET = "target"
    TYPE = "type"
    WEIGHT = "weight"

    # Custom Prompting keys
    FILTER_SECTION = "filter_section"
    STYLE_SECTION = "style_section"
    FILTER_PROMPT_FIELD = "filter_prompt"
    STYLE_PROMPT_FIELD = "style_prompt"
    IS_FINAL_FIELD = "is_final"

    # Edge types
    GENERATED = "Generated"
    INIT = "Init"
    MANUAL = "Manual"


# 2. PARSING-ONLY MODELS (Private to this adapter)
# These models inherit from the base structs and add the parsing logic
# (aliases).


class _GMNode(structs.Node):
    data: structs.NodeData = Field(alias=GMContract.DATA_KEY)

    model_config = ConfigDict(populate_by_name=True, from_attributes=True)


class _GMEdge(structs.Edge):
    data: structs.EdgeData = Field(alias=GMContract.DATA_KEY)

    model_config = ConfigDict(populate_by_name=True, from_attributes=True)


class _GMNodesFile(structs.NodesFile):
    nodes: list[_GMNode] = Field(alias=GMContract.NODES_KEY)
    root_id: str = Field(alias=GMContract.ROOT_KEY)

    model_config = ConfigDict(populate_by_name=True, from_attributes=True)


class _GMEdgesFile(structs.EdgesFile):
    edges: list[_GMEdge] = Field(alias=GMContract.EDGE_KEY)

    model_config = ConfigDict(populate_by_name=True, from_attributes=True)


class GMGraphFiles(structs.GraphFiles):
    nodes_file: _GMNodesFile = Field(alias=GMContract.NODES_FILE)
    edges_file: _GMEdgesFile = Field(alias=GMContract.EDGES_FILE)

    # This config is important for Pydantic to use the aliases in nested
    # models
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)


# --- GM-Specific Runtime Dependencies and Fallbacks ---
try:
    from general_mindmap.utils.graph_patch import embeddings_model

    GM_AVAILABLE = True
except ImportError:
    _ERROR_MESSAGE = "The 'general_mindmap' package is required..."

    class _MissingClass:
        def __init__(self, *args, **kwargs):
            raise ImportError(_ERROR_MESSAGE)

    class _MissingInstance:
        def __call__(self, *args, **kwargs):
            raise ImportError(_ERROR_MESSAGE)

        def __getattr__(self, name):
            raise ImportError(_ERROR_MESSAGE)

    # Define placeholders for ALL runtime dependencies
    embeddings_model = _MissingInstance()

    GM_AVAILABLE = False


# --- The Adapter Implementation ---
class GeneralMindmapAdapter(AbstractAdapter):
    """The concrete adapter for the 'general_mindmap' backend."""

    gm_contract = GMContract

    graph_files = GMGraphFiles

    def __init__(self):
        if not GM_AVAILABLE:
            raise ImportError(
                "Cannot instantiate GeneralMindmapAdapter: 'general_mindmap' "
                "package not found."
            )

    def get_embeddings_model(self) -> EmbeddingModel:
        return embeddings_model

    def translate_graph_files(self, raw_data: dict) -> structs.GraphFiles:
        """
        Translates raw backend data by parsing it with the
        adapter-specific Pydantic models.
        """
        try:
            # Use the private parsing model to validate and create an
            # instance.
            parsed_object = GMGraphFiles.model_validate(raw_data)

            # The return type hint is `structs.GraphFiles`, which is
            # correct because `_GMGraphFiles` is a subclass of it.
            return parsed_object
        except ValidationError as e:
            logging.error(f"GM Adapter validation failed: {e}")
            raise
