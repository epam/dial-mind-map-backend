from enum import StrEnum
from typing import List

from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field

# --- Common Models for Generation ---


class AnswerPart(BaseModel):
    """A single, discrete factual statement with its source."""

    statement: str = Field(
        ...,
        description="A single, discrete factual statement extracted directly from the text.",
    )
    doc_id: str = Field(
        ..., description="The document ID from the source <document> tag."
    )
    chunk_id: int = Field(
        ...,
        description="The chunk ID from the source <chunk> tag where the fact was found.",
    )


class Edge(BaseModel):
    """Represents a directed edge between two nodes in the mind map."""

    source: str = Field(..., description="The label of the parent node.")
    target: str = Field(..., description="The label of the child node.")


# --- Models for Simple Generation ---


class Node(BaseModel):
    """Represents a node in the mind map."""

    label: str = Field(..., description="Unique label for the node.")
    question: str = Field(
        ..., description="A question that this node's content answers."
    )
    answer_parts: List[AnswerPart] = Field(
        default_factory=list,
        description="List of factual statements that form the answer.",
    )


class Mindmap(BaseModel):
    """The complete mind map structure."""

    root: str = Field(..., description="The label of the root node.")
    nodes: List[Node] = Field(
        ..., description="A list of all nodes in the graph."
    )
    edges: List[Edge] = Field(
        ..., description="A list of all edges connecting the nodes."
    )


# --- Models for Two-Stage Generation ---


class NodeStructure(BaseModel):
    label: str = Field(..., description="Unique label for the node.")
    question: str = Field(
        ..., description="A question that this node's content will answer."
    )


class MindmapStructure(BaseModel):
    root: str = Field(..., description="The label of the root node.")
    nodes: List[NodeStructure] = Field(
        ..., description="A list of nodes with their questions."
    )
    edges: List[Edge] = Field(
        ..., description="A list of all edges connecting the nodes."
    )


class Answer(BaseModel):
    answer_parts: List[AnswerPart] = Field(
        ...,
        description="A list of factual statements answering a specific question.",
    )


class AnswerList(BaseModel):
    answers: List[Answer] = Field(
        ...,
        description="A list of answer objects, in the exact same order as the input questions.",
    )


class RootLabel(BaseModel):
    root_label: str = Field(
        ...,
        description="The label of the single most logical root node from the provided graph structure.",
    )


# --- Models for Node Analysis ---


class DuplicateNode(BaseModel):
    label: str = Field(
        ..., description="The unique label of the duplicate node."
    )
    duplicate_of: str = Field(
        ..., description="The label of the node that it is a duplicate of."
    )


class LowSubstanceNode(BaseModel):
    label: str = Field(
        ..., description="The unique label of the low-substance node."
    )
    reason: str = Field(
        ...,
        description="Explanation of why the node is low-substance (e.g., 'Answer just repeats the label').",
    )


class NodeAnalysisResult(BaseModel):
    duplicate_nodes: List[DuplicateNode] = Field(default_factory=list)
    low_substance_nodes: List[LowSubstanceNode] = Field(default_factory=list)


class ToolResponseStatus(StrEnum):
    # same as in langchain
    SUCCESS = "success"
    ERROR = "error"


class PreFilterResponse(BaseModel):
    user_friendly_error: str | None = Field(default=None)
    detailed_error: str | None = Field(default=None)
    llm_output: None = Field(default=None)
    # NOTE: Currently None means no filter applied; but RagFilterDial can also contain an empty list;
    # NOTE: Should we replace all the None logic with just an empty list?
    rag_filter: None = Field(default=None)


class StatGPTToolResponse(ToolMessage):
    tool_call_id: str
    custom_content: dict | None = None

    @property
    def is_success(self) -> bool:
        return self.status == ToolResponseStatus.SUCCESS

    @property
    def is_failed(self) -> bool:
        return self.status == ToolResponseStatus.ERROR

    @property
    def state_dict(self) -> dict | None:
        if not self.custom_content:
            return None
        return self.custom_content.get("state", None)

    @property
    def response_content(self) -> str | None:
        if self.state_dict is None:
            return None
        return self.state_dict.get("response", "")

    @property
    def tool_type(self) -> str | None:
        if self.state_dict is None:
            return None
        return self.state_dict.get("type")
