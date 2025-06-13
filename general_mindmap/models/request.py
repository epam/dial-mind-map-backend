from typing import Any, List

from pydantic import BaseModel, ConfigDict


class Document(BaseModel):
    id: str
    url: str
    type: str = "LINK"
    content_type: str | None = None

    model_config = ConfigDict(extra="forbid")


class InitMindmapRequest(BaseModel):
    documents: List[Document]

    model_config = ConfigDict(extra="forbid")


class ApplyMindmapRequest(BaseModel):
    del_documents: List[Document]
    add_documents: List[Document]
    graph_files: dict[str, Any]

    model_config = ConfigDict(extra="forbid")


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

    model_config = ConfigDict(extra="forbid")


class AddDocumentRequest(BaseModel):
    id: str | None = None
    url: str
    type: str
    content_type: str | None = None

    model_config = ConfigDict(extra="forbid")


class EdgeData(BaseModel):
    id: str | None = None
    source: str
    target: str
    type: str | None = "Manual"
    weight: str | None = None

    model_config = ConfigDict(extra="forbid")


class AddEdgeRequest(BaseModel):
    data: EdgeData

    model_config = ConfigDict(extra="forbid")


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


class AddNodeRequest(BaseModel):
    data: NodeData
    position: NodePosition

    model_config = ConfigDict(extra="forbid")
