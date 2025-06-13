from copy import deepcopy
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from general_mindmap.v2.config import DIAL_URL
from general_mindmap.v2.dial.client import DialClient
from general_mindmap.v2.models.metadata import History, HistoryItemType
from general_mindmap.v2.routers.sources import migrate_sources_from_old_format
from general_mindmap.v2.utils.batch_file_reader import BatchFileReader

router = APIRouter()


async def read_graph(client: DialClient):
    file_reader = BatchFileReader(client)

    if client._metadata.nodes_file:
        file_reader.add_file(client._metadata.nodes_file)
    if client._metadata.edges_file:
        file_reader.add_file(client._metadata.edges_file)

    graph = {}
    for result in await file_reader.read():
        graph = graph | result[1]

    if not graph:
        graph = {"nodes": [], "edges": []}

    return graph


def is_possible_undo(history: History, user: str) -> bool:
    step = history.current_step

    return (
        step >= 0
        and step < len(history.steps)
        and user == history.steps[step].user
    )


def is_possible_redo(history: History, user: str) -> bool:
    step = history.current_step + 1

    return (
        step >= 0
        and step < len(history.steps)
        and user == history.steps[step].user
    )


async def undo(client: DialClient, history: History, user: str) -> int:
    if history.current_step < 0 or history.current_step >= len(history.steps):
        raise HTTPException(400, "Impossible to undo. History is empty")
    if history.steps[history.current_step].user != user:
        raise HTTPException(400, "Impossible to undo another user's change")

    sources_related = False
    graph_related = False
    for changed_file in history.steps[history.current_step].changes or []:
        sources_related = sources_related or changed_file.type in [
            HistoryItemType.SOURCES,
            HistoryItemType.SOURCE_STATE,
        ]
        graph_related = graph_related or changed_file.type in [
            HistoryItemType.NODES,
            HistoryItemType.SINGLE_NODE,
            HistoryItemType.EDGES,
        ]

    for changed_file in history.steps[history.current_step].changes or []:
        match changed_file.type:
            case HistoryItemType.NODES:
                client._metadata.nodes_file = changed_file.old_value
            case HistoryItemType.EDGES:
                client._metadata.edges_file = changed_file.old_value
            case HistoryItemType.SOURCES:
                client._metadata.documents_file = changed_file.old_value
            case HistoryItemType.SOURCE_STATE:
                assert changed_file.id
                if changed_file.old_value:
                    client._metadata.source_names[changed_file.id] = (
                        changed_file.old_value
                    )
                else:
                    del client._metadata.source_names[changed_file.id]
            case HistoryItemType.SINGLE_NODE:
                assert changed_file.id is not None

                if not changed_file.old_value:
                    del client._metadata.nodes[changed_file.id]
                else:
                    client._metadata.nodes[changed_file.id] = (
                        changed_file.old_value
                    )

    return (1 if sources_related else 0) | (2 if graph_related else 0)


async def redo(client: DialClient, history: History, user: str) -> int:
    if history.current_step < 0 or history.current_step >= len(history.steps):
        raise HTTPException(400, "There is no action to redo")
    if history.steps[history.current_step].user != user:
        raise HTTPException(400, "Impossible to redo another user's change")

    sources_related = False
    graph_related = False
    for changed_file in history.steps[history.current_step].changes or []:
        sources_related = sources_related or changed_file.type in [
            HistoryItemType.SOURCES,
            HistoryItemType.SOURCE_STATE,
        ]
        graph_related = graph_related or changed_file.type in [
            HistoryItemType.NODES,
            HistoryItemType.SINGLE_NODE,
            HistoryItemType.EDGES,
        ]

    for changed_file in history.steps[history.current_step].changes or []:
        match changed_file.type:
            case HistoryItemType.NODES:
                client._metadata.nodes_file = changed_file.new_value
            case HistoryItemType.EDGES:
                client._metadata.edges_file = changed_file.new_value
            case HistoryItemType.SOURCES:
                client._metadata.documents_file = changed_file.new_value
            case HistoryItemType.SOURCE_STATE:
                assert changed_file.id
                client._metadata.source_names[changed_file.id] = (
                    changed_file.new_value
                )
            case HistoryItemType.SINGLE_NODE:
                assert changed_file.id

                if not changed_file.new_value:
                    del client._metadata.nodes[changed_file.id]
                else:
                    client._metadata.nodes[changed_file.id] = (
                        changed_file.new_value
                    )

    return (1 if sources_related else 0) | (2 if graph_related else 0)


@router.post("/mindmaps/{mindmap:path}/history")
async def undo_redo(request: Request, action: str):
    async with await DialClient.create_with_folder(
        DIAL_URL,
        "auto",
        request.headers["x-mindmap"],
        request.headers.get("etag", ""),
    ) as client:
        if action == "undo":
            related_to = await undo(client, client._metadata.history, "USER")

            client._metadata.history.current_step = (
                client._metadata.history.current_step - 1
            )
        else:
            client._metadata.history.current_step = (
                client._metadata.history.current_step + 1
            )

            related_to = await redo(client, client._metadata.history, "USER")

        response: Dict[str, Any] = {
            "undo": is_possible_undo(client._metadata.history, "USER"),
            "redo": is_possible_redo(client._metadata.history, "USER"),
        }

        if related_to & 1:
            if client._metadata.documents_file:
                docs, _ = await client.read_file_by_name_and_etag(
                    client._metadata.documents_file
                )
            else:
                docs = {
                    "documents": [],
                    "generation_status": "NOT_STARTED",
                    "generated": False,
                }

            if "generated" not in docs:
                docs["generated"] = docs["generation_status"] not in [
                    "NOT_STARTED",
                    "IN_PROGRESS",
                ]

            file_reader = BatchFileReader(client)

            storage_url_to_id = {}
            for i, source in enumerate(docs["documents"]):
                if "storage_url" in source:
                    storage_url_to_id[source["storage_url"]] = i
                    file_reader.add_file(source["storage_url"])

            for result in await file_reader.read():
                docs["documents"][storage_url_to_id[result[0]]] = result[1]

            await migrate_sources_from_old_format(client, docs)

            response["sources"] = deepcopy(docs)
            response["sources"]["sources"] = response["sources"]["documents"]
            del response["sources"]["documents"]

            response["sources"]["names"] = {
                source: name
                for source, name in client._metadata.source_names.items()
                if name
            }
        else:
            response["sources"] = None

        if related_to & 2:
            response["graph"] = await read_graph(client)
        else:
            response["graph"] = None

        etag = await client.close()
        return JSONResponse(response, headers={"ETag": etag})


@router.get("/mindmaps/{mindmap:path}/history")
async def history(request: Request):
    client = await DialClient.create_with_folder(
        DIAL_URL,
        "auto",
        request.headers["x-mindmap"],
        request.headers.get("etag", ""),
    )

    await client.read_metadata()

    return {
        "undo": is_possible_undo(client._metadata.history, "USER"),
        "redo": is_possible_redo(client._metadata.history, "USER"),
    }
