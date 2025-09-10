from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, List, Union

from pydantic import BaseModel, ConfigDict


# Output Structures
class StatusChunk(BaseModel):
    title: str
    details: str | None = None


class NodeMetadata(BaseModel):
    answer: list[dict[str, str | list[str]]]
    level: int
    cluster_id: int

    model_config = ConfigDict(extra="forbid")


class NodeData(BaseModel):
    id: str | None = None
    label: str
    details: str
    question: str | None = None
    questions: List[str] | None = None
    metadata: NodeMetadata | None = None
    link: str | None = None
    icon: str | None = None
    status: str | None = None
    neon: bool = False

    model_config = ConfigDict(extra="forbid")


class RootNodeChunk(BaseModel):
    root_id: str


class EdgeData(BaseModel):
    id: str | None = None
    source: str
    target: str
    type: str | None = "Manual"
    weight: str | None = None

    model_config = ConfigDict(extra="forbid")


GeneratorStream = Union[StatusChunk, NodeData, RootNodeChunk, EdgeData]


# Input structures
class Document(BaseModel):
    id: str
    url: str
    type: str = "LINK"
    content_type: str | None = None
    base_url: str = None
    name: str = None

    model_config = ConfigDict(extra="forbid")


class InitMindmapRequest(BaseModel):
    documents: List[Document]

    model_config = ConfigDict(extra="forbid")


class ApplyMindmapRequest(BaseModel):
    documents: List[Document]
    del_documents: List[Document]
    add_documents: List[Document]
    graph_files: dict[str, Any]

    model_config = ConfigDict(extra="forbid")


MMRequest = InitMindmapRequest | ApplyMindmapRequest


class PatchNodeData(BaseModel):
    id: str
    label: str | None = None
    details: str | None = None
    question: str | None = None
    questions: List[str] | None = None
    link: str | None = None
    icon: str | None = None
    status: str | None = None
    neon: bool = False

    model_config = ConfigDict(extra="forbid")


class NodePosition(BaseModel):
    x: float | None = None
    y: float | None = None

    model_config = ConfigDict(extra="forbid")


class PatchNode(BaseModel):
    data: PatchNodeData
    position: NodePosition

    model_config = ConfigDict(extra="forbid")


class PatchEdgeData(BaseModel):
    id: str | None = None
    source: str | None = None
    target: str | None = None
    type: str | None = "Manual"

    model_config = ConfigDict(extra="forbid")


class PatchEdge(BaseModel):
    data: PatchEdgeData

    model_config = ConfigDict(extra="forbid")


class PatchGraphRequest(BaseModel):
    root: str | None = None
    nodes: List[PatchNode] | None = None
    edges: List[PatchEdge] | None = None
    edges_to_delete: List[str] | None = None
    history_skip: bool = False

    model_config = ConfigDict(extra="forbid")


class AddDocumentRequest(BaseModel):
    id: str | None = None
    url: str
    type: str
    content_type: str | None = None

    model_config = ConfigDict(extra="forbid")


class AddEdgeRequest(BaseModel):
    data: EdgeData

    model_config = ConfigDict(extra="forbid")


class AddNodeRequest(BaseModel):
    data: NodeData
    position: NodePosition

    model_config = ConfigDict(extra="forbid")


class DocStatus(str, Enum):
    FAILED = "FAILED"
    INDEXED = "INDEXED"


class DocStatusChunk(BaseModel):
    id: str
    status: DocStatus
    status_description: str | None = None
    chunks: List[Document] | None = None


class Generator(ABC):
    dial_url: str
    api_key: str

    def __init__(self, dial_url: str, api_key: str):
        self.dial_url = dial_url
        self.api_key = api_key

    @abstractmethod
    async def generate(
        self,
        request: InitMindmapRequest,
    ) -> AsyncGenerator[
        StatusChunk | NodeData | EdgeData | DocStatusChunk | RootNodeChunk, None
    ]:
        pass


# --- Wrapper structures ---
class Node(BaseModel):
    """A container for node data."""

    data: NodeData


class Edge(BaseModel):
    """A container for edge data."""

    data: EdgeData


class NodesFile(BaseModel):
    """Represents the contents of the nodes file."""

    nodes: List[Node]
    root_id: str


class EdgesFile(BaseModel):
    """Represents the contents of the edges file."""

    edges: List[Edge]


class GraphFiles(BaseModel):
    """The complete, clean representation of the graph data."""

    nodes_file: NodesFile
    edges_file: EdgesFile


# --- Stream Chunks ---


@dataclass
class StreamRootNode:
    """Represents the root node being streamed."""

    node: Node


@dataclass
class StreamDocStatus:
    """Represents the status of a document being processed."""

    doc_id: str
    status: str


@dataclass
class StreamStatus:
    """A generic status update."""

    message: str
    is_final: bool = False


StreamChunk = Union[StreamRootNode, StreamDocStatus, StreamStatus]
