import json
from enum import Enum
from time import time
from typing import Dict, List

from pydantic import BaseModel

HISTORY_LIMIT = 10
DELETE_AFTER = 60


class ExtraForbidModel(BaseModel):
    class Config:
        extra = "forbid"


class File(ExtraForbidModel):
    file: str
    expired: float


class FileType(str, Enum):
    NODES = "nodes"
    EDGES = "edges"
    DOCUMENTS = "documents"
    SINGLE_NODE = "single_node"
    SINGLE_DOCUMENT = "single_document"


class HistoryFile(ExtraForbidModel):
    file: str
    new_file: str | None = None
    type: FileType
    id: str | None = None


class HistoryItemType(str, Enum):
    NODES = "nodes"
    EDGES = "edges"
    SOURCES = "sources"
    SINGLE_NODE = "single_node"
    SOURCE_STATE = "source_state"
    APPEARANCES = "appearances"
    PARAMS = "params"


class HistoryItem(ExtraForbidModel):
    new_value: str
    old_value: str
    type: HistoryItemType
    id: str | None = None
    version: int | None = None


class HistoryStep(ExtraForbidModel):
    user: str
    changed_files: List[HistoryFile] | None = None
    changes: List[HistoryItem] | None = None
    skip: bool = False


class History(ExtraForbidModel):
    current_step: int
    steps: List[HistoryStep]

    def append(self, metadata: "Metadata", step: HistoryStep):
        deleted_steps_pref = []
        deleted_steps_suff = []

        if len(self.steps) and self.current_step != len(self.steps) - 1:
            deleted_steps_suff = self.cut_steps(0, self.current_step + 1)

        self.steps.append(step)

        deleted_steps_pref = self.cut_steps(
            max(0, len(self.steps) - HISTORY_LIMIT), len(self.steps)
        )

        self.delete_steps(metadata, deleted_steps_pref, deleted_steps_suff)

        self.current_step = len(self.steps) - 1

    def cut_steps(self, left: int, right: int) -> List[HistoryStep]:
        to_delete = self.steps[: max(0, left - 1)] + self.steps[right:]
        self.steps = self.steps[left:right]
        return to_delete

    def delete_steps(
        self,
        metadata: "Metadata",
        deleted_steps_pref: List[HistoryStep],
        deleted_steps_suff: List[HistoryStep],
    ):
        for step in deleted_steps_pref:
            for file in step.changes or []:
                if file.old_value and file.type not in [
                    HistoryItemType.SOURCE_STATE,
                    HistoryItemType.PARAMS,
                ]:
                    metadata.append_to_delete(file.old_value)
        for step in deleted_steps_suff:
            for file in step.changes or []:
                if file.new_value and file.type not in [
                    HistoryItemType.SOURCE_STATE,
                    HistoryItemType.PARAMS,
                ]:
                    metadata.append_to_delete(file.new_value)


class Metadata(ExtraForbidModel):
    version: int = 3
    last_change: str
    graph_history: History = History(steps=[], current_step=0)
    docs_history: History = History(steps=[], current_step=0)
    history: History = History(steps=[], current_step=0)
    nodes_file: str | None = None
    edges_file: str | None = None
    documents_file: str | None = None
    documents: Dict[str, str] = {}
    nodes: Dict[str, str] = {}
    source_names: Dict[str, str] = {}
    to_delete: List[File] = []
    last_doc_id: int = 1
    appearances_file: str = ""
    params: dict[str, str] = {}

    def append_to_delete(self, file_name: str):
        if any(file.file == file_name for file in self.to_delete):
            return

        self.to_delete.append(
            File(file=file_name, expired=time() + DELETE_AFTER)
        )
