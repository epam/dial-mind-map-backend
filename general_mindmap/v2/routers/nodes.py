import json
from copy import deepcopy
from time import time
from typing import Any, List
from uuid import uuid4

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, ValidationError

from general_mindmap.models.graph import Node
from general_mindmap.models.request import AddNodeRequest, PatchGraphRequest
from general_mindmap.utils.docstore import (
    embeddings_model,
    encode_docstore,
    node_to_documents,
)
from general_mindmap.utils.graph_patch import (
    embeddings_model as patch_embeddings_model,
)
from general_mindmap.utils.graph_patch import (
    node_to_document as patch_node_to_document,
)
from general_mindmap.v2.config import DIAL_URL
from general_mindmap.v2.dial.client import DialClient
from general_mindmap.v2.models.metadata import (
    HistoryItem,
    HistoryItemType,
    HistoryStep,
)
from general_mindmap.v2.routers.utils.errors import (
    INCORRECT_JSON_RESPONSE_ERROR,
    timeout_after,
)
from general_mindmap.v2.utils.batch_file_reader import BatchFileReader

router = APIRouter()


# TODO: batch write
@router.post("/mindmaps/{mindmap:path}/graph/nodes")
@timeout_after()
async def add_node(request: Request):
    start_time = str(time())

    try:
        req = AddNodeRequest.model_validate(await request.json())
    except json.JSONDecodeError:
        return INCORRECT_JSON_RESPONSE_ERROR
    except ValidationError as e:
        return Response(status_code=400, content=str(e))

    if req.data.id is None:
        req.data.id = str(uuid4())

    if req.data.question is None and req.data.questions:
        req.data.question = req.data.questions[0]

    async with await DialClient.create_with_folder(
        DIAL_URL or "",
        request.headers["authorization"],
        request.headers["x-mindmap"],
        request.headers.get("etag", ""),
    ) as client:
        assert client._metadata.nodes_file
        current_nodes_file, _ = await client.read_file_by_name_and_etag(
            client._metadata.nodes_file
        )
        nodes = current_nodes_file["nodes"]

        if any(req.data.id == node["data"]["id"] for node in nodes):
            return Response(f'Node "{req.data.id}" already exists', 400)

        node = req.model_dump(exclude_none=True)
        nodes.append(node)

        nodes_file = f"{start_time}_nodes"
        await client.write_file(nodes_file, current_nodes_file)

        node_obj = Node.model_validate(node["data"])

        docs = node_to_documents(node_obj)
        for doc in docs:
            doc.metadata["id"] = req.data.id
        docstore = FAISS.from_documents(docs, embeddings_model)

        patch_docs = [patch_node_to_document(node_obj)]
        patch_docstore = FAISS.from_documents(
            patch_docs, patch_embeddings_model
        )

        node_file = f"nodes/{start_time}_{req.data.id}"
        await client.write_file(
            node_file,
            {
                "docstore": encode_docstore(docstore),
                "patcher_docstore": encode_docstore(patch_docstore),
            },
        )

        client._metadata.history.append(
            client._metadata,
            HistoryStep(
                user="USER",
                changes=[
                    HistoryItem(
                        old_value=client._metadata.nodes_file,
                        new_value=nodes_file,
                        type=HistoryItemType.NODES,
                    ),
                    HistoryItem(
                        old_value="",
                        new_value=node_file,
                        type=HistoryItemType.SINGLE_NODE,
                        id=req.data.id,
                    ),
                ],
            ),
        )

        client._metadata.nodes_file = nodes_file
        client._metadata.nodes[req.data.id] = node_file

        etag = await client.close()
        return JSONResponse(content=node, headers={"ETag": etag})


@router.put("/mindmaps/{mindmap:path}/graph/nodes/{node_id}")
@timeout_after()
async def change_node(request: Request, node_id: str):
    start_time = str(time())

    try:
        req = AddNodeRequest.model_validate(await request.json())
    except json.JSONDecodeError:
        return INCORRECT_JSON_RESPONSE_ERROR
    except ValidationError as e:
        return Response(status_code=400, content=str(e))

    if req.data.id:
        if req.data.id != node_id:
            return Response(
                status_code=400,
                content="The node id in the body and in the path don't match",
            )
    else:
        req.data.id = node_id

    if req.data.question is None and req.data.questions:
        req.data.question = req.data.questions[0]

    async with await DialClient.create_with_folder(
        DIAL_URL,
        request.headers["authorization"],
        request.headers["x-mindmap"],
        request.headers.get("etag", ""),
    ) as client:
        assert client._metadata.nodes_file
        current_nodes_file, _ = await client.read_file_by_name_and_etag(
            client._metadata.nodes_file
        )
        nodes = current_nodes_file["nodes"]

        if not any(req.data.id == node["data"]["id"] for node in nodes):
            return Response(404)

        node = req.model_dump(exclude_none=True)

        for i in range(len(nodes)):
            if nodes[i]["data"]["id"] == req.data.id:
                nodes[i] = node

        nodes_file = f"{start_time}_nodes"
        await client.write_file(nodes_file, current_nodes_file)

        node_obj = Node.model_validate(node["data"])
        docs = node_to_documents(node_obj)
        for doc in docs:
            doc.metadata["id"] = req.data.id
        docstore = FAISS.from_documents(docs, embeddings_model)

        patch_docs = [patch_node_to_document(node_obj)]
        patch_docstore = FAISS.from_documents(
            patch_docs, patch_embeddings_model
        )

        node_file = f"nodes/{start_time}_{req.data.id}"
        await client.write_file(
            node_file,
            {
                "docstore": encode_docstore(docstore),
                "patcher_docstore": encode_docstore(patch_docstore),
            },
        )

        client._metadata.history.append(
            client._metadata,
            HistoryStep(
                user="USER",
                changes=[
                    HistoryItem(
                        old_value=client._metadata.nodes_file,
                        new_value=nodes_file,
                        type=HistoryItemType.NODES,
                    ),
                    HistoryItem(
                        old_value="",
                        new_value=node_file,
                        type=HistoryItemType.SINGLE_NODE,
                        id=req.data.id,
                    ),
                ],
            ),
        )

        client._metadata.nodes_file = nodes_file
        client._metadata.nodes[req.data.id] = node_file

        etag = await client.close()
        return JSONResponse(content=node, headers={"ETag": etag})


@router.delete("/mindmaps/{mindmap:path}/graph/nodes/{node_id}")
@timeout_after()
async def delete_node(request: Request, node_id: str):
    start_time = str(time())

    async with await DialClient.create_with_folder(
        DIAL_URL,
        "auto",
        request.headers["x-mindmap"],
        request.headers.get("etag", ""),
    ) as client:
        file_reader = BatchFileReader(client)

        assert client._metadata.nodes_file and client._metadata.edges_file
        file_reader.add_file(client._metadata.nodes_file)
        file_reader.add_file(client._metadata.edges_file)

        nodes_file, edges_file = None, None
        for result in await file_reader.read():
            if result[0] == client._metadata.nodes_file:
                nodes_file = result[1]
            else:
                edges_file = result[1]

        assert nodes_file != None and edges_file != None

        if nodes_file.get("root", None) == node_id:
            nodes_file["root"] = None

        nodes_file["nodes"] = [
            node
            for node in nodes_file["nodes"]
            if node["data"]["id"] != node_id
        ]

        edges_file["edges"] = [
            edge
            for edge in edges_file["edges"]
            if edge["data"]["source"] != node_id
            and edge["data"]["target"] != node_id
        ]

        nodes_file_name = f"{start_time}_nodes"
        await client.write_file(nodes_file_name, nodes_file)

        edges_file_name = f"{start_time}_edges"
        await client.write_file(edges_file_name, edges_file)

        client._metadata.history.append(
            client._metadata,
            HistoryStep(
                user="USER",
                changes=[
                    HistoryItem(
                        old_value=client._metadata.nodes_file,
                        new_value=nodes_file_name,
                        type=HistoryItemType.NODES,
                    ),
                    HistoryItem(
                        old_value=client._metadata.edges_file,
                        new_value=edges_file_name,
                        type=HistoryItemType.EDGES,
                    ),
                    HistoryItem(
                        old_value=client._metadata.nodes[node_id],
                        new_value="",
                        type=HistoryItemType.SINGLE_NODE,
                        id=node_id,
                    ),
                ],
            ),
        )
        del client._metadata.nodes[node_id]

        client._metadata.nodes_file = nodes_file_name
        client._metadata.edges_file = edges_file_name

        etag = await client.close()
        return Response(headers={"ETag": etag})


class NodeToChange(BaseModel):
    node: str
    file: str
    content: Any


def is_same_nodes(a, b) -> bool:
    a = a["data"]
    b = b["data"]

    return (
        a["label"] == b.get("label", None)
        and a["details"] == b.get("details", None)
        and a["question"] == b.get("question", None)
    )


@router.patch("/mindmaps/{mindmap:path}/graph")
@timeout_after()
async def change_graph(request: Request):
    start_time = str(time())

    try:
        req = PatchGraphRequest.model_validate(await request.json())
    except json.JSONDecodeError:
        return INCORRECT_JSON_RESPONSE_ERROR
    except ValidationError as e:
        return Response(status_code=400, content=str(e))

    async with await DialClient.create_with_folder(
        DIAL_URL,
        request.headers["authorization"],
        request.headers["x-mindmap"],
        request.headers.get("etag", ""),
    ) as client:
        assert client._metadata.nodes_file
        current_nodes_file, _ = await client.read_file_by_name_and_etag(
            client._metadata.nodes_file
        )

        changes = []
        if "root" in req.model_fields_set:
            if req.root:
                found = False
                for node in current_nodes_file["nodes"]:
                    node_id = node["data"]["id"]

                    if node_id == req.root:
                        found = True
                        break

                if not found:
                    return Response(
                        status_code=400,
                        content=f'There is no "{req.root}" node in the graph',
                    )

            current_nodes_file["root"] = req.root

            nodes_file = f"{start_time}_nodes"
            await client.write_file(nodes_file, current_nodes_file)

            changes.append(
                HistoryItem(
                    old_value=client._metadata.nodes_file,
                    new_value=nodes_file,
                    type=HistoryItemType.NODES,
                )
            )
            client._metadata.nodes_file = nodes_file

        if req.nodes:
            nodes_map = dict(
                (node["data"]["id"], node)
                for node in current_nodes_file["nodes"]
            )

            nodes_to_change: List[NodeToChange] = []
            for patch_node in req.nodes:
                node = nodes_map.get(patch_node.data.id, None)

                if node is None:
                    node = {"data": {}, "position": {}}
                    current_nodes_file["nodes"].append(node)

                if "question" not in node and "questions" in node:
                    node["question"] = node["questions"][0]
                if patch_node.data.questions:
                    patch_node.data.question = patch_node.data.questions[0]

                original_node = deepcopy(node)

                for field, value in patch_node.data.model_dump(
                    exclude_unset=True
                ).items():
                    if value is None:
                        del node["data"][field]
                    else:
                        node["data"][field] = value

                for field, value in patch_node.position.model_dump(
                    exclude_unset=True
                ).items():
                    if "position" not in node:
                        node["position"] = {}

                    if value is None:
                        del node["position"][field]
                    else:
                        node["position"][field] = value

                if is_same_nodes(node, original_node):
                    continue

                node_obj = Node.model_validate(node["data"])
                docs = node_to_documents(node_obj)
                for doc in docs:
                    doc.metadata["id"] = node["data"]["id"]
                docstore = FAISS.from_documents(docs, embeddings_model)

                patch_docs = [patch_node_to_document(node_obj)]
                patch_docstore = FAISS.from_documents(
                    patch_docs, patch_embeddings_model
                )

                docstore = FAISS.from_documents(docs, embeddings_model)

                nodes_to_change.append(
                    NodeToChange(
                        node=node["data"]["id"],
                        file=f"nodes/{start_time}_{node['data']['id']}",
                        content={
                            "docstore": encode_docstore(docstore),
                            "patcher_docstore": encode_docstore(patch_docstore),
                        },
                    )
                )

            for node_to_change in nodes_to_change:
                await client.write_file(
                    node_to_change.file, node_to_change.content
                )

            for node_to_change in nodes_to_change:
                changes.append(
                    HistoryItem(
                        old_value=client._metadata.nodes.get(
                            node_to_change.node, ""
                        ),
                        new_value=node_to_change.file,
                        type=HistoryItemType.SINGLE_NODE,
                        id=node_to_change.node,
                    ),
                )

            for node_to_change in nodes_to_change:
                client._metadata.nodes[node_to_change.node] = (
                    node_to_change.file
                )

            nodes_file = f"{start_time}_nodes"
            await client.write_file(nodes_file, current_nodes_file)

            if "root" not in req.model_fields_set:
                changes.append(
                    HistoryItem(
                        old_value=client._metadata.nodes_file,
                        new_value=nodes_file,
                        type=HistoryItemType.NODES,
                    )
                )
                client._metadata.nodes_file = nodes_file

        if req.edges:
            assert client._metadata.edges_file
            current_edges_file, _ = await client.read_file_by_name_and_etag(
                client._metadata.edges_file
            )

            edges_map = dict(
                (edge["data"]["id"], edge)
                for edge in current_edges_file["edges"]
            )

            for patch_edge in req.edges:
                if patch_edge.data.id is None:
                    patch_edge.data.id = str(uuid4())

                edge = edges_map.get(patch_edge.data.id, None)

                if edge is None:
                    edge = {"data": {}}
                    current_edges_file["edges"].append(edge)

                for field, value in patch_edge.data.model_dump(
                    exclude_unset=True
                ).items():
                    if value is None:
                        del edge["data"][field]
                    else:
                        edge["data"][field] = value

            edges_file = f"{start_time}_edges"
            await client.write_file(edges_file, current_edges_file)

            changes.append(
                HistoryItem(
                    old_value=client._metadata.edges_file,
                    new_value=edges_file,
                    type=HistoryItemType.EDGES,
                )
            )
            client._metadata.edges_file = edges_file

        client._metadata.history.append(
            client._metadata,
            HistoryStep(
                user="USER",
                changes=changes,
            ),
        )

        etag = await client.close()
        return Response(headers={"ETag": etag})
