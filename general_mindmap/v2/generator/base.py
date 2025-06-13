from abc import ABC, abstractmethod
from enum import Enum
from typing import AsyncGenerator, List

from langchain_core.documents import Document
from pydantic import BaseModel

from general_mindmap.models.request import (
    EdgeData,
    InitMindmapRequest,
    NodeData,
)


class StatusChunk(BaseModel):
    title: str
    details: str | None = None


class DocStatus(str, Enum):
    FAILED = "FAILED"
    INDEXED = "INDEXED"


class DocStatusChunk(BaseModel):
    id: str
    status: DocStatus
    status_description: str | None = None
    chunks: List[Document] | None = None


class RootNodeChunk(BaseModel):
    root_id: str


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
