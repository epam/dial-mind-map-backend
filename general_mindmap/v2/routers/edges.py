import json
from time import time
from typing import Any, List
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
from pydantic import ValidationError

from general_mindmap.models.request import AddEdgeRequest
from general_mindmap.utils.graph_patch import embeddings_model
from general_mindmap.v2.config import DIAL_URL
from general_mindmap.v2.dial.client import DialClient
from general_mindmap.v2.models.edge_type import EdgeType
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
from generator.core.stages import EdgeProcessor
from generator.utils.constants import DataFrameCols as Col

TARGET_NUMBER_OF_EDGES = 3

router = APIRouter()


@router.post("/mindmaps/{mindmap:path}/graph/edges")
@timeout_after()
async def add_edge(request: Request):
    start_time = str(time())
    try:
        req = AddEdgeRequest.model_validate(await request.json())
    except json.JSONDecodeError:
        return INCORRECT_JSON_RESPONSE_ERROR
    except ValidationError as e:
        return Response(status_code=400, content=str(e))

    if req.data.id is None:
        req.data.id = str(uuid4())
    if req.data.type and req.data.type != EdgeType.MANUAL:
        return Response("The type can be only manual", 400)

    async with await DialClient.create_with_folder(
        DIAL_URL,
        request.headers["authorization"],
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
        nodes = nodes_file["nodes"]
        edges = edges_file["edges"]

        node_ids = [node["data"]["id"] for node in nodes]

        if req.data.source not in node_ids or req.data.target not in node_ids:
            return Response(
                status_code=400,
                content="The source or the target doesn't exist in the graph",
            )

        if any(edge["data"]["id"] == req.data.id for edge in edges):
            return Response(
                status_code=400,
                content=f'Edge "{req.data.id}" already exists',
            )

        if any(
            req.data.source == edge["data"]["source"]
            and req.data.target == edge["data"]["target"]
            for edge in edges
        ):
            return Response(
                status_code=400,
                content="The edge between the nodes is already exists",
            )

        edge = req.model_dump()
        edge["data"]["type"] = EdgeType.MANUAL
        edges.append(edge)

        edges_file_name = f"{start_time}_edges"
        await client.write_file(edges_file_name, edges_file)

        client._metadata.history.append(
            client._metadata,
            HistoryStep(
                user="USER",
                changes=[
                    HistoryItem(
                        old_value=client._metadata.edges_file,
                        new_value=edges_file_name,
                        type=HistoryItemType.EDGES,
                    ),
                ],
            ),
        )

        client._metadata.edges_file = edges_file_name

        etag = await client.close()
        return JSONResponse(content=edge, headers={"ETag": etag})


def node_to_document(node: Any) -> Document:
    return Document(
        page_content=f"{node['data']['label']}\n{node['data']['details']}",
        metadata={
            "id": node["data"]["id"],
            "title": node["data"]["label"],
            "question": node["data"]["question"],
        },
    )


def get_node_by_id(nodes: List[Any], id: str):
    for node in nodes:
        if id == node["data"]["id"]:
            return node

    raise ValueError("The node is not found")


def get_closest_nodes(
    docstore: FAISS, nodes: List[Any], node: Any, k: int
) -> List[Any]:
    node_text = node_to_document(node).page_content
    closest_docs = docstore.similarity_search(node_text, k=(k + 1))

    closest_nodes = [
        get_node_by_id(nodes, doc.metadata["id"])
        for doc in closest_docs
        if doc.metadata["id"] != node["data"]["id"]
    ][:k]

    return closest_nodes


def get_node_pairs_with_sim(
    docstore: FAISS,
    node_by_id: dict[int, dict],
    nodes: List[Any],
    pairs: List[tuple[Any, Any]],
) -> List[tuple[Any, Any, float]]:
    """
    Create a list of node pairs with their similarity score.

    Returns:
        List of tuples (node_1, node_2, similarity_score)
    """
    pairs_with_sim = []

    for node_1_id, node_2_id in pairs:
        node_1 = node_by_id[node_1_id]
        node_2 = node_by_id[node_2_id]

        node_1_text = node_to_document(node_1).page_content
        search_results = docstore.similarity_search_with_score(
            node_1_text, k=len(nodes)
        )

        node_2_id = node_2["data"]["id"]
        similarity_score = None

        for doc, score in search_results:
            if doc.metadata["id"] == node_2_id:
                similarity_score = score
                break

        if similarity_score is None:
            similarity_score = 0.0

        pairs_with_sim.append((node_1, node_2, similarity_score))

    return pairs_with_sim


@router.post("/mindmaps/{mindmap:path}/graph/edges/auto")
@timeout_after()
async def generate_edges(request: Request):
    start_time = str(time())

    async with await DialClient.create_with_folder(
        DIAL_URL,
        request.headers["authorization"],
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
        nodes = nodes_file["nodes"]
        edges = edges_file["edges"]

        edges = [
            edge
            for edge in edges
            if edge["data"].get("type") != EdgeType.GENERATED
        ]

        node_by_id = {node["data"]["id"]: node for node in nodes}

        node_to = {node["data"]["id"]: set() for node in nodes}
        node_from = {node["data"]["id"]: set() for node in nodes}

        for edge in edges:
            source = edge["data"]["source"]
            target = edge["data"]["target"]

            node_to[source].add(target)
            node_from[target].add(source)

        documents = [node_to_document(node) for node in nodes]
        docstore = FAISS.from_documents(documents, embeddings_model)

        for id, to in node_to.items():
            if len(to) >= TARGET_NUMBER_OF_EDGES:
                continue

            node = node_by_id[id]

            need_to_add = TARGET_NUMBER_OF_EDGES - len(to)
            if need_to_add > 0:
                closest_nodes = get_closest_nodes(
                    docstore, nodes, node, k=TARGET_NUMBER_OF_EDGES
                )

                closest_nodes = [
                    closest_node
                    for closest_node in closest_nodes
                    if closest_node["data"]["id"] not in to
                ][:need_to_add]

                for closest_node in closest_nodes:
                    edges.append(
                        {
                            "data": {
                                "id": str(uuid4()),
                                "source": id,
                                "target": closest_node["data"]["id"],
                                "type": EdgeType.GENERATED,
                            }
                        }
                    )

                    node_to[id].add(closest_node["data"]["id"])
                    node_from[closest_node["data"]["id"]].add(id)

        for id, from_set in node_from.items():
            if len(from_set) > 0:
                continue

            node = node_by_id[id]

            closest_nodes = get_closest_nodes(
                docstore, nodes, node, k=TARGET_NUMBER_OF_EDGES
            )

            closest_nodes = [
                closest_node
                for closest_node in closest_nodes
                if closest_node["data"]["id"] != id  # Avoid self-loops
                and id not in node_to[closest_node["data"]["id"]]
            ]

            if closest_nodes:
                source_node = closest_nodes[0]
                edges.append(
                    {
                        "data": {
                            "id": str(uuid4()),
                            "source": source_node["data"]["id"],
                            "target": id,
                            "type": EdgeType.GENERATED,
                        }
                    }
                )
                node_to[source_node["data"]["id"]].add(id)
                node_from[id].add(source_node["data"]["id"])

        edge_id_map = {}
        for edge in edges:
            source = edge["data"]["source"]
            target = edge["data"]["target"]
            if "id" in edge["data"]:
                edge_id_map[(source, target)] = edge["data"]["id"]

        edge_df = pd.DataFrame(
            {
                Col.ORIGIN_CONCEPT_ID: edge["data"]["source"],
                Col.TARGET_CONCEPT_ID: edge["data"]["target"],
                Col.TYPE: edge["data"].get("type", "Init"),
                Col.WEIGHT: None,
                "edge_id": edge["data"].get("id", None),
            }
            for edge in edges
        )

        edge_df = await EdgeProcessor.strong_con_graph(
            edge_df, get_node_pairs_with_sim, None, docstore, nodes, node_by_id
        )

        edges = [
            {
                "data": {
                    "source": row[Col.ORIGIN_CONCEPT_ID],
                    "target": row[Col.TARGET_CONCEPT_ID],
                    "type": row[Col.TYPE],
                    "id": edge_id_map.get(
                        (
                            row[Col.ORIGIN_CONCEPT_ID],
                            row[Col.TARGET_CONCEPT_ID],
                        ),
                        str(uuid4()),
                    ),
                }
            }
            for _, row in edge_df.iterrows()
        ]

        edges_file["edges"] = edges
        edges_file_name = f"{start_time}_edges"
        await client.write_file(edges_file_name, edges_file)

        client._metadata.history.append(
            client._metadata,
            HistoryStep(
                user="USER",
                changes=[
                    HistoryItem(
                        old_value=client._metadata.edges_file,
                        new_value=edges_file_name,
                        type=HistoryItemType.EDGES,
                    ),
                ],
            ),
        )

        client._metadata.edges_file = edges_file_name

        etag = await client.close()
        return JSONResponse(content=edges, headers={"ETag": etag})


@router.delete("/mindmaps/{mindmap:path}/graph/edges/auto")
@timeout_after()
async def delete_generate_edges(request: Request):
    start_time = str(time())

    async with await DialClient.create_with_folder(
        DIAL_URL,
        request.headers["authorization"],
        request.headers["x-mindmap"],
        request.headers.get("etag", ""),
    ) as client:
        file_reader = BatchFileReader(client)

        assert client._metadata.edges_file
        file_reader.add_file(client._metadata.edges_file)

        edges_file = None
        for result in await file_reader.read():
            edges_file = result[1]
        assert edges_file != None

        edges_file["edges"] = [
            edge
            for edge in edges_file["edges"]
            if edge["data"]["type"] != EdgeType.GENERATED
        ]
        edges_file_name = f"{start_time}_edges"
        await client.write_file(edges_file_name, edges_file)

        client._metadata.history.append(
            client._metadata,
            HistoryStep(
                user="USER",
                changes=[
                    HistoryItem(
                        old_value=client._metadata.edges_file,
                        new_value=edges_file_name,
                        type=HistoryItemType.EDGES,
                    ),
                ],
            ),
        )

        client._metadata.edges_file = edges_file_name

        etag = await client.close()
        return Response(headers={"ETag": etag})


@router.delete("/mindmaps/{mindmap:path}/graph/edges/{edge_id}")
@timeout_after()
async def delete_edge(request: Request, edge_id: str):
    start_time = str(time())

    async with await DialClient.create_with_folder(
        DIAL_URL,
        request.headers["authorization"],
        request.headers["x-mindmap"],
        request.headers.get("etag", ""),
    ) as client:
        file_reader = BatchFileReader(client)

        assert client._metadata.edges_file
        file_reader.add_file(client._metadata.edges_file)

        edges_file = None
        for result in await file_reader.read():
            edges_file = result[1]
        assert edges_file != None

        removed_edge = next(
            (
                edge
                for edge in edges_file["edges"]
                if edge["data"]["id"] == edge_id
            ),
            None,
        )
        if removed_edge is None:
            return Response(status_code=404)

        edges_file["edges"] = [
            edge
            for edge in edges_file["edges"]
            if edge["data"]["id"] != edge_id
        ]
        edges_file_name = f"{start_time}_edges"
        await client.write_file(edges_file_name, edges_file)

        client._metadata.history.append(
            client._metadata,
            HistoryStep(
                user="USER",
                changes=[
                    HistoryItem(
                        old_value=client._metadata.edges_file,
                        new_value=edges_file_name,
                        type=HistoryItemType.EDGES,
                    ),
                ],
            ),
        )

        client._metadata.edges_file = edges_file_name

        etag = await client.close()
        return Response(headers={"ETag": etag})


@router.put("/mindmaps/{mindmap:path}/graph/edges/{edge_id}")
@timeout_after()
async def change_edge(request: Request, edge_id: str):
    start_time = str(time())

    try:
        req = AddEdgeRequest.model_validate(await request.json())
    except json.JSONDecodeError:
        return INCORRECT_JSON_RESPONSE_ERROR
    except ValidationError as e:
        return Response(status_code=400, content=str(e))

    if req.data.id:
        if req.data.id != edge_id:
            return Response(
                "The edge id in the body and in the path don't match", 400
            )
    else:
        req.data.id = edge_id

    if req.data.type and req.data.type != EdgeType.MANUAL:
        return Response("The type can be only manual", 400)

    async with await DialClient.create_with_folder(
        DIAL_URL,
        request.headers["authorization"],
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
        nodes = nodes_file["nodes"]
        edges = edges_file["edges"]

        target_edge = next(
            (
                edge
                for edge in edges_file["edges"]
                if edge["data"]["id"] == edge_id
            ),
            None,
        )
        if target_edge is None:
            return Response(status_code=404)

        node_ids = [node["data"]["id"] for node in nodes]

        if req.data.source not in node_ids or req.data.target not in node_ids:
            return Response(
                status_code=400,
                content="The source or the target doesn't exist in the graph",
            )

        if any(
            edge["data"]["id"] != req.data.id
            and req.data.source == edge["data"]["source"]
            and req.data.target == edge["data"]["target"]
            for edge in edges
        ):
            return Response(
                status_code=400,
                content="The edge between the nodes is already exists",
            )

        for i in range(len(edges)):
            current_edge = edges[i]

            if current_edge["data"]["id"] == edge_id:
                edge = req.model_dump()
                edge["data"]["type"] = EdgeType.MANUAL
                edges[i] = edge

                break

        edges_file_name = f"{start_time}_edges"
        await client.write_file(edges_file_name, edges_file)

        client._metadata.history.append(
            client._metadata,
            HistoryStep(
                user="USER",
                changes=[
                    HistoryItem(
                        old_value=client._metadata.edges_file,
                        new_value=edges_file_name,
                        type=HistoryItemType.EDGES,
                    ),
                ],
            ),
        )

        client._metadata.edges_file = edges_file_name

        etag = await client.close()
        return Response(headers={"ETag": etag})
