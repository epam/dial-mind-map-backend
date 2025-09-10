import json
from copy import deepcopy
from time import time
from typing import Any, Dict, List

from aidial_sdk.utils.streaming import add_heartbeat
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from langchain_community.vectorstores import FAISS
from openai import RateLimitError

from general_mindmap.models.graph import Node
from general_mindmap.utils.docstore import (
    embeddings_model,
    encode_docstore,
    node_to_documents,
)
from general_mindmap.utils.errors import pretify_rate_limit
from general_mindmap.utils.graph_patch import (
    embeddings_model as patch_embeddings_model,
)
from general_mindmap.utils.graph_patch import (
    node_to_document as patch_node_to_document,
)
from general_mindmap.v2.config import DIAL_URL
from general_mindmap.v2.dial.client import DialClient
from general_mindmap.v2.generator.simple import (
    GeneratorConfig,
    TwoStageGenerator,
)
from general_mindmap.v2.models.metadata import (
    HistoryItem,
    HistoryItemType,
    HistoryStep,
)
from general_mindmap.v2.routers.sources import add_generated_to_docs
from general_mindmap.v2.utils.batch_file_reader import BatchFileReader
from general_mindmap.v2.utils.batch_file_writer import BatchFileWriter
from generator import MindMapGenerator
from generator.common.context import cur_run_id
from generator.common.misc import ContextPreservingAsyncIterator
from generator.common.structs import (
    ApplyMindmapRequest,
    Document,
    EdgeData,
    Generator,
    InitMindmapRequest,
    NodeData,
    RootNodeChunk,
    StatusChunk,
)

router = APIRouter()


def parse_llm_json(res: Any):
    content = str(res)

    content = content.removeprefix("```json")
    content = content.removesuffix("```")

    return json.loads(content)


async def generate(
    docs_file: Dict[str, Any],
    client: DialClient,
    old_documents_file: str,
    generator: Generator,
):
    start_time = str(time())

    builded_docs = []
    for doc in docs_file["documents"]:
        if not doc.get("active", True):
            continue

        if doc["type"] == "FILE" or "copy_url" in doc:
            builded_docs.append(
                Document(
                    id=f"{doc['id']}.{doc.get('version', 1)}",
                    url=(
                        doc["url"] if doc["type"] == "FILE" else doc["copy_url"]
                    ),
                    type="FILE",
                    content_type=doc.get("content_type", "text/html")
                    or "text/html",
                    base_url=doc.get("url"),
                    name=doc.get("name"),
                )
            )
        else:
            builded_docs.append(
                Document(
                    id=f"{doc['id']}.{doc.get('version', 1)}",
                    url=doc["url"],
                    type="LINK",
                    content_type="",
                    base_url=doc.get("url"),
                    name=doc.get("name"),
                )
            )

    req = InitMindmapRequest(documents=builded_docs)

    async with client:
        batchFileWriter = BatchFileWriter(client)

        changes: List[HistoryItem] = []
        old_nodes_map = client._metadata.nodes

        for document in req.documents:
            if document.type == "FILE":
                if not document.url.startswith(client._folder):
                    document.url = f"{client._folder}{document.url}"

        edges = []
        nodes_map = {}
        nodes = []
        root_id = None

        context_vars = {cur_run_id: cur_run_id.get()}
        wrapped_generator = ContextPreservingAsyncIterator(
            generator.generate(req), context_vars
        )

        stream_with_heartbeat = add_heartbeat(
            wrapped_generator,
            heartbeat_interval=30,
            heartbeat_object=StatusChunk(title="heartbeat", details=""),
        )

        async for chunk in stream_with_heartbeat:
            if isinstance(chunk, StatusChunk):
                if (
                    isinstance(chunk, StatusChunk)
                    and chunk.title == "heartbeat"
                ):
                    await client.update_lock()
                else:
                    if isinstance(chunk, StatusChunk):
                        chunk_data = f"data: {json.dumps({'title': chunk.title, 'details': chunk.details})}\n\n"
                        yield chunk_data
            elif isinstance(chunk, EdgeData):
                edges.append({"data": chunk.model_dump()})
            elif isinstance(chunk, NodeData):
                node_id = chunk.id

                node_obj = Node.model_validate(chunk.model_dump())
                docs = node_to_documents(node_obj)
                node_docstore = FAISS.from_documents(docs, embeddings_model)

                patch_docs = [patch_node_to_document(node_obj)]
                patch_docstore = FAISS.from_documents(
                    patch_docs, patch_embeddings_model
                )

                file_name = f"nodes/{start_time}_{node_id}"
                batchFileWriter.add_file(
                    file_name,
                    {
                        "docstore": encode_docstore(node_docstore),
                        "patcher_docstore": encode_docstore(patch_docstore),
                    },
                )

                if node_id in old_nodes_map:
                    changes.append(
                        HistoryItem(
                            old_value=old_nodes_map[node_id],
                            new_value=file_name,
                            type=HistoryItemType.SINGLE_NODE,
                            id=node_id,
                        )
                    )
                    del old_nodes_map[node_id]
                else:
                    changes.append(
                        HistoryItem(
                            old_value="",
                            new_value=file_name,
                            type=HistoryItemType.SINGLE_NODE,
                            id=node_id,
                        )
                    )

                nodes_map[node_id] = file_name

                nodes.append({"data": chunk.model_dump()})
            elif isinstance(chunk, RootNodeChunk):
                root_id = chunk.root_id

        for node_id, file in old_nodes_map.items():
            changes.append(
                HistoryItem(
                    old_value=file,
                    new_value="",
                    type=HistoryItemType.SINGLE_NODE,
                    id=node_id,
                )
            )

        new_edges_file = f"{start_time}_edges"
        batchFileWriter.add_file(new_edges_file, {"edges": edges})
        changes.append(
            HistoryItem(
                old_value=(
                    client._metadata.edges_file
                    if client._metadata.edges_file
                    else ""
                ),
                new_value=new_edges_file,
                type=HistoryItemType.EDGES,
            )
        )
        client._metadata.edges_file = new_edges_file

        new_documents_file = f"{start_time}_documents"
        client._metadata.documents_file = new_documents_file
        changes.append(
            HistoryItem(
                old_value=old_documents_file,
                new_value=new_documents_file,
                type=HistoryItemType.SOURCES,
            )
        )

        docs_file["generation_status"] = "FINISHED"
        docs_file["generated"] = True

        for source in docs_file["documents"]:
            if source.get("active", True):
                source["in_graph"] = True
            else:
                source["in_graph"] = False

            if "storage_url" in source:
                new_active_storage_url = f"sources/{source['id']}/versions/{source.get('version', 1)}/{start_time}_state"
                source["storage_url"] = new_active_storage_url
                batchFileWriter.add_file(new_active_storage_url, source)

        batchFileWriter.add_file(new_documents_file, docs_file)

        new_nodes_file = f"{start_time}_nodes"
        batchFileWriter.add_file(
            new_nodes_file,
            {"root": root_id, "nodes": nodes},
        )
        changes.append(
            HistoryItem(
                old_value=client._metadata.nodes_file or "",
                new_value=new_nodes_file,
                type=HistoryItemType.NODES,
            )
        )
        client._metadata.nodes_file = new_nodes_file

        await batchFileWriter.write()

        client._metadata.nodes = nodes_map
        client._metadata.last_change = start_time
        client._metadata.history.append(
            client._metadata,
            HistoryStep(user="USER", changes=changes),
        )

        etag = await client.close()

        yield f"data: {json.dumps({'title': 'Graph generated', 'etag': etag})}\n\n"
        yield f"[DONE]\n"


def _process_doc_changes(
    sources: List[Any], changed_source_ids: List[str]
) -> Dict[str, List[Any]]:
    """
    Using provided source ids identify changes made to the sources.

    Args:
        sources: The original list of Documents
        changed_source_ids: IDs of the changed sources

    Returns:
        Dictionary with lists of Documents to 'add' and 'del'
    """
    removed = []
    added = []

    versions_by_id = {}
    for source in sources:
        source_id = source["id"]
        if source_id not in versions_by_id:
            versions_by_id[source_id] = []
        versions_by_id[source_id].append(source)

    for _id in changed_source_ids:
        versions = versions_by_id.get(_id)

        if not versions:
            continue

        active = None
        in_graph = None
        for version in versions:
            if version.get("active", True):
                active = version
            else:
                if version.get("in_graph", True):
                    in_graph = version

        assert active

        if active["status"] == "REMOVED":
            removed.append(active)
        elif not active["in_graph"]:
            if in_graph:
                removed.append(in_graph)
                added.append(active)
            else:
                added.append(active)

    return {"add": added, "del": removed}


def process_apply_docs(docs_file: Dict[str, Any], target_sources: List[str]):
    sources = docs_file["documents"]

    doc_changes = _process_doc_changes(sources, target_sources)
    del_docs = doc_changes["del"]
    add_docs = doc_changes["add"]

    add_documents = [
        Document(
            id=f"{doc['id']}.{doc.get('version', 1)}",
            url=(doc["url"] if doc["type"] == "FILE" else doc["copy_url"]),
            type="FILE",
            content_type=doc.get("content_type", "text/html") or "text/html",
            base_url=doc.get("url"),
            name=doc.get("name"),
        )
        for doc in add_docs
    ]

    del_documents = [
        Document(
            id=f"{doc['id']}.{doc.get('version', 1)}",
            url=(doc["url"] if doc["type"] == "FILE" else doc["copy_url"]),
            type="FILE",
            content_type=doc.get("content_type", "text/html") or "text/html",
            base_url=doc.get("url"),
            name=doc.get("name"),
        )
        for doc in del_docs
    ]
    return add_documents, del_documents


async def apply(
    docs_file: Dict[str, Any],
    client: DialClient,
    target_sources: List[str],
    old_documents_file: str,
):
    start_time = str(time())

    builded_docs = []
    for doc in docs_file["documents"]:
        if not doc.get("active", True):
            continue

        if doc["type"] == "FILE" or "copy_url" in doc:
            builded_docs.append(
                Document(
                    id=f"{doc['id']}.{doc.get('version', 1)}",
                    url=(
                        doc["url"] if doc["type"] == "FILE" else doc["copy_url"]
                    ),
                    type="FILE",
                    content_type=doc.get("content_type", "text/html")
                    or "text/html",
                    base_url=doc.get("url"),
                    name=doc.get("name"),
                )
            )
        else:
            builded_docs.append(
                Document(
                    id=f"{doc['id']}.{doc.get('version', 1)}",
                    url=doc["url"],
                    type="LINK",
                    content_type="",
                    base_url=doc.get("url"),
                    name=doc.get("name"),
                )
            )

    generator = MindMapGenerator(DIAL_URL, api_key="auto", file_storage=client)

    add_documents, del_documents = process_apply_docs(docs_file, target_sources)

    async with client:
        assert client._metadata.nodes_file and client._metadata.edges_file
        nodes_file, _ = await client.read_file_by_name_and_etag(
            client._metadata.nodes_file
        )
        edges_file, _ = await client.read_file_by_name_and_etag(
            client._metadata.edges_file
        )

        graph_files = {
            "nodes_file": nodes_file,
            "edges_file": edges_file,
        }

        req = ApplyMindmapRequest(
            documents=builded_docs,
            add_documents=add_documents,
            del_documents=del_documents,
            graph_files=graph_files,
        )

        for document in req.documents:
            if document.type == "FILE":
                if not document.url.startswith(client._folder):
                    document.url = f"{client._folder}{document.url}"

        batchFileWriter = BatchFileWriter(client)

        changes: List[HistoryItem] = []

        new_documents_file = f"{start_time}_documents"
        client._metadata.documents_file = new_documents_file

        docs_file["generation_status"] = "FINISHED"
        docs_file["generated"] = True

        for document in req.add_documents + req.del_documents:
            if document.type == "FILE":
                if not document.url.startswith(client._folder):
                    document.url = f"{client._folder}{document.url}"

        deleted_sources = set()
        for source in docs_file["documents"]:
            if source["id"] not in target_sources:
                continue

            if source.get("active", True):
                source["in_graph"] = True

                if source["status"] == "REMOVED":
                    deleted_sources.add(source["id"])
                    if source["id"] in client._metadata.source_names:
                        del client._metadata.source_names[source["id"]]
            else:
                source["in_graph"] = False

        docs_file["documents"] = [
            source
            for source in docs_file["documents"]
            if source["id"] not in deleted_sources
        ]

        for source in docs_file["documents"]:
            if source["id"] not in target_sources:
                continue

            if "storage_url" in source:
                new_active_storage_url = f"sources/{source['id']}/versions/{source.get('version', 1)}/{start_time}_state"
                source["storage_url"] = new_active_storage_url
                batchFileWriter.add_file(new_active_storage_url, source)

        edges = []
        nodes_map = {}
        nodes = []
        root_id = None

        context_vars = {cur_run_id: cur_run_id.get()}
        wrapped_generator = ContextPreservingAsyncIterator(
            generator.apply(req), context_vars
        )

        stream_with_heartbeat = add_heartbeat(
            wrapped_generator,
            heartbeat_interval=30,
            heartbeat_object=StatusChunk(title="heartbeat", details=""),
        )

        old_nodes_map = client._metadata.nodes

        async for chunk in stream_with_heartbeat:
            if isinstance(chunk, StatusChunk):
                if (
                    isinstance(chunk, StatusChunk)
                    and chunk.title == "heartbeat"
                ):
                    await client.update_lock()
                else:
                    if isinstance(chunk, StatusChunk):
                        chunk_data = f"data: {json.dumps({'title': chunk.title, 'details': chunk.details})}\n\n"
                        yield chunk_data
            elif isinstance(chunk, EdgeData):
                edges.append({"data": chunk.model_dump()})
            elif isinstance(chunk, NodeData):
                node_id = chunk.id

                node_obj = Node.model_validate(chunk.model_dump())
                docs = node_to_documents(node_obj)
                node_docstore = FAISS.from_documents(docs, embeddings_model)

                patch_docs = [patch_node_to_document(node_obj)]
                patch_docstore = FAISS.from_documents(
                    patch_docs, patch_embeddings_model
                )

                file_name = f"nodes/{start_time}_{node_id}"
                batchFileWriter.add_file(
                    file_name,
                    {
                        "docstore": encode_docstore(node_docstore),
                        "patcher_docstore": encode_docstore(patch_docstore),
                    },
                )

                if node_id in old_nodes_map:
                    changes.append(
                        HistoryItem(
                            old_value=old_nodes_map[node_id],
                            new_value=file_name,
                            type=HistoryItemType.SINGLE_NODE,
                            id=node_id,
                        )
                    )
                    del old_nodes_map[node_id]
                else:
                    changes.append(
                        HistoryItem(
                            old_value="",
                            new_value=file_name,
                            type=HistoryItemType.SINGLE_NODE,
                            id=node_id,
                        )
                    )

                nodes_map[node_id] = file_name

                nodes.append({"data": chunk.model_dump()})
            elif isinstance(chunk, RootNodeChunk):
                root_id = chunk.root_id

        for node_id, file in old_nodes_map.items():
            changes.append(
                HistoryItem(
                    old_value=file,
                    new_value="",
                    type=HistoryItemType.SINGLE_NODE,
                    id=node_id,
                )
            )

        new_edges_file = f"{start_time}_edges"
        batchFileWriter.add_file(new_edges_file, {"edges": edges})
        changes.append(
            HistoryItem(
                old_value=(
                    client._metadata.edges_file
                    if client._metadata.edges_file
                    else ""
                ),
                new_value=new_edges_file,
                type=HistoryItemType.EDGES,
            )
        )
        client._metadata.edges_file = new_edges_file

        batchFileWriter.add_file(new_documents_file, docs_file)
        changes.append(
            HistoryItem(
                old_value=old_documents_file,
                new_value=new_documents_file,
                type=HistoryItemType.SOURCES,
            )
        )

        new_nodes_file = f"{start_time}_nodes"
        batchFileWriter.add_file(
            new_nodes_file,
            {"root": root_id, "nodes": nodes},
        )
        changes.append(
            HistoryItem(
                old_value=client._metadata.nodes_file or "",
                new_value=new_nodes_file,
                type=HistoryItemType.NODES,
            )
        )
        client._metadata.nodes_file = new_nodes_file

        await batchFileWriter.write()

        client._metadata.nodes = nodes_map
        client._metadata.last_change = start_time
        client._metadata.history.append(
            client._metadata,
            HistoryStep(user="USER", changes=changes),
        )

        etag = await client.close()

        yield f"data: {json.dumps({'title': 'Graph generated', 'etag': etag})}\n\n"
        yield f"[DONE]\n"


async def apply_wrapper(
    client: DialClient,
    docs_file: Dict[str, Any],
    target_sources: List[str],
    old_documents_file: str,
):
    try:
        async for message in apply(
            docs_file, client, target_sources, old_documents_file
        ):
            if not message.startswith("data: "):
                continue

            message = json.loads(message[6 : len(message) - 2])
            message["time"] = time()

            await client.write_file("generate", message)
    except RateLimitError as e:
        await client.write_file(
            "generate",
            {
                "error": pretify_rate_limit(e),
                "user_friendly": True,
                "time": time(),
            },
        )
    except Exception as e:
        await client.write_file(
            "generate", {"error": "Something went wrong", "time": time()}
        )

        raise e


async def generate_wrapper(
    client: DialClient,
    docs_file: Dict[str, Any],
    old_documents_file: str,
    generator: Generator,
):
    try:
        async for message in generate(
            docs_file, client, old_documents_file, generator
        ):
            if not message.startswith("data: "):
                continue

            message = json.loads(message[6 : len(message) - 2])
            message["time"] = time()

            await client.write_file("generate", message)
    except RateLimitError as e:
        await client.write_file(
            "generate",
            {
                "error": pretify_rate_limit(e),
                "user_friendly": True,
                "time": time(),
            },
        )
    except Exception as e:
        await client.write_file(
            "generate", {"error": "Something went wrong", "time": time()}
        )

        raise e


@router.post("/mindmaps/{mindmap:path}/generate")
async def generate_mindmap(request: Request, background_tasks: BackgroundTasks):
    request_body = None
    if (await request.body()).strip():
        request_body = await request.json()
    else:
        request_body = {}

    client = await DialClient.create_with_folder(
        DIAL_URL or "",
        "auto",
        request.headers["x-mindmap"],
        request.headers.get("etag", ""),
    )

    await client.open()
    await client.write_file(
        "generate", {"title": "Starting generation", "time": time()}
    )

    old_documents_file = client._metadata.documents_file or ""
    docs_file, _ = await client.read_file_and_etag(
        client._metadata.documents_file
    )

    add_generated_to_docs(docs_file)
    docs_file["generation_status"] = "IN_PROGRESS"

    current_time = str(time())
    client._metadata.documents_file = f"{current_time}_documents"

    await client.write_file(
        client._metadata.documents_file,
        docs_file,
    )

    client._metadata.last_change = current_time

    etag = await client.write_file(
        "metadata", client._metadata.model_dump(exclude_none=True), client._etag
    )
    client._etag = etag

    file_reader = BatchFileReader(client)

    storage_url_to_id = {}
    for i, source in enumerate(docs_file["documents"]):
        if "storage_url" in source:
            storage_url_to_id[source["storage_url"]] = i
            file_reader.add_file(source["storage_url"])

    for result in await file_reader.read():
        docs_file["documents"][storage_url_to_id[result[0]]] = result[1]

    if "sources" in request_body:
        background_tasks.add_task(
            apply_wrapper,
            client,
            docs_file,
            request_body["sources"],
            old_documents_file,
        )
    else:
        if client._metadata.params.get("type", "universal") == "universal":
            generator = MindMapGenerator(
                DIAL_URL, api_key="auto", file_storage=client
            )
        else:
            _config = GeneratorConfig(
                model=client._metadata.params.get("model", ""),
                prompt=client._metadata.params.get("prompt", ""),
                dial_url=DIAL_URL,
                api_key="auto",
            )
            generator = TwoStageGenerator(_config)

        background_tasks.add_task(
            generate_wrapper, client, docs_file, old_documents_file, generator
        )

    return Response()


@router.post("/mindmaps/{mindmap:path}/generation_status")
async def generation_status(request: Request):
    client = await DialClient.create_with_folder(
        DIAL_URL,
        "auto",
        request.headers["x-mindmap"],
        "",
    )

    try:
        data, _ = await client.read_file_and_etag("generate")
    except HTTPException:
        return HTTPException(404, "Generation is not started")

    return StreamingResponse(
        client.subscribe_to_generate(request, data),
        media_type="text/event-stream",
    )


@router.post("/mindmaps/{mindmap:path}/generate/params")
async def edit_params(request: Request):
    start_time = time()

    async with await DialClient.create_with_folder(
        DIAL_URL,
        "auto",
        request.headers["x-mindmap"],
        request.headers.get("etag", ""),
    ) as client:
        if "last_change" not in client._metadata.model_fields_set:
            client._metadata.last_change = str(start_time)

        old_params = deepcopy(client._metadata.params)

        client._metadata.params |= await request.json()

        client._metadata.history.append(
            client._metadata,
            HistoryStep(
                user="USER",
                changes=[
                    HistoryItem(
                        old_value=json.dumps(old_params),
                        new_value=json.dumps(client._metadata.params),
                        type=HistoryItemType.PARAMS,
                    ),
                ],
            ),
        )

        return Response(headers={"ETag": await client.close()})
