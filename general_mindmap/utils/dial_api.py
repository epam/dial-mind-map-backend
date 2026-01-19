import base64
import json
import os
import re
from typing import Any, Dict, Iterable, List

from aidial_sdk.chat_completion import Choice
from langchain_core.runnables import Runnable, RunnableConfig

from dial_rag.document_record import DocumentRecord
from dial_rag.index_record import ChunkMetadata
from dial_rag.index_storage import SERIALIZATION_CONFIG
from general_mindmap.models.graph import GraphData
from general_mindmap.utils.log_config import logger
from general_mindmap.utils.rag_eval_utils import (
    TokenUsageByModelsCallback,
    calc_token_usage_costs,
    token_usage_context,
)
from general_mindmap.utils.tokens_queue import TokensQueue
from general_mindmap.v2.dial.client import DialClient
from general_mindmap.v2.utils.batch_file_reader import BatchFileReader


async def build_references(
    ans: str,
    docs: List[Any],
    nodes: List[GraphData],
    client: DialClient,
) -> Dict[str, Any]:
    docs_map = {f"{doc['id']}.{doc.get('version', 1)}": doc for doc in docs}

    left = ans.find("^")

    batch_file_reader = BatchFileReader(client)

    references = {"docs": [], "nodes": []}
    used = set()
    doc_url_to_ref = {}
    while left != -1:
        right = ans.find("^", left + 1)

        if right == -1:
            break

        if (
            re.match(r"\^\[\d+\.\d+\]\^$", ans[left : right + 1])
            or re.match(r"\^\[\d+\.\d+\.\d+\]\^$", ans[left : right + 1])
            or re.match(r"\^\[[a-fA-F0-9-]+\]\^$", ans[left : right + 1])
        ):
            doc_chunk = ans[left + 2 : right - 1]

            if doc_chunk in used:
                left = right
                continue
            used.add(doc_chunk)

            if "." in doc_chunk:
                parts = doc_chunk.split(".")

                if len(parts) == 2:
                    doc_id, chunk_id = f"{parts[0]}.1", parts[1]
                else:
                    doc_id = f"{parts[0]}.{parts[1]}"
                    chunk_id = parts[2]

                doc = docs_map[doc_id]

                parsed_doc_id = doc_id.split(".")[0]
                parsed_version = int(doc_id.split(".")[1])

                source_name = client._metadata.source_names.get(
                    parsed_doc_id, None
                )
                if not source_name:
                    source_name = None

                if (
                    doc["content_type"]
                    == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    or doc["content_type"] == "application/pdf"
                ):
                    if int(chunk_id) - 1 >= len(doc["pages"]):
                        left = right
                        continue

                    references["docs"].append(
                        {
                            "doc_id": parsed_doc_id,
                            "version": parsed_version,
                            "chunk_id": chunk_id,
                            "doc_name": (
                                doc["url"][8:]
                                if "name" not in doc
                                else doc["name"]
                            ),
                            "source_name": source_name,
                            "doc_type": doc["type"],
                            "doc_content_type": doc["content_type"],
                            "doc_url": doc["url"],
                            "content": doc["pages"][int(chunk_id) - 1],
                            "content_type": "image/jpeg",
                        }
                    )
                else:
                    if "index" in doc:
                        url_to_file = doc["index"]
                    else:
                        url_to_file = client._metadata.documents[doc["id"]]

                    if url_to_file not in doc_url_to_ref:
                        doc_url_to_ref[url_to_file] = []
                        batch_file_reader.add_file(url_to_file)

                    doc_url_to_ref[url_to_file].append(
                        {
                            "ref_id": len(references["docs"]),
                            "chunk_id": int(chunk_id) - 1,
                            "doc_display_name": doc["url"],
                        }
                    )

                    references["docs"].append(
                        {
                            "doc_id": parsed_doc_id,
                            "version": parsed_version,
                            "chunk_id": chunk_id,
                            "doc_name": (
                                doc["url"][8:]
                                if "name" not in doc
                                else doc["name"]
                            ),
                            "doc_type": doc["type"],
                            "doc_content_type": doc["content_type"],
                            "doc_url": doc["url"],
                            "content_type": "text/markdown",
                            "source_name": source_name,
                        }
                    )
            else:
                nodes_map = {node.data.id: node for node in nodes}

                node = nodes_map.get(doc_chunk, None)
                if node:
                    references["nodes"].append(node.data.model_dump())
                else:
                    logger.warning(f"There is no node {doc_chunk} in the graph")

        left = right

    to_delete = []
    for result in await batch_file_reader.read():
        file_name = result[0]

        for i in doc_url_to_ref[file_name]:
            if i["chunk_id"] >= len(result[1]["chunks"]):
                record = None
                if "rag_record" in result[1]:
                    record = DocumentRecord.from_bytes(
                        base64.b64decode(result[1]["rag_record"]),
                        **SERIALIZATION_CONFIG,
                    )

                chunk_id = i["chunk_id"] - len(result[1]["chunks"])

                if record and len(record.chunks) <= chunk_id:
                    logger.warning(
                        f'There is no chunk {i["chunk_id"]} in doc {file_name}'
                    )
                    to_delete.append(i["ref_id"])
                    continue

                chunk = record.chunks[chunk_id]
                references["docs"][i["ref_id"]]["content"] = chunk.text

                continue

            references["docs"][i["ref_id"]]["content"] = json.loads(
                result[1]["chunks"][i["chunk_id"]]
            )["page_content"]

    references["docs"] = [
        ref for i, ref in enumerate(references["docs"]) if i not in to_delete
    ]

    return references


def build_citation_maps(
    content_data: dict,
    active_docs: List[Any],
    records: List[DocumentRecord],
    chunks_by_doc: Dict[str, List[Any]],
) -> tuple[Dict[str, str], Dict[str, str]]:
    """
    Extracts nodes and chunks from the attachment content and builds
    the maps required for TokensQueue.
    """
    cards = content_data.get("nodes", [])
    chunks = content_data.get("chunks", [])

    chunks_metadatas = [
        ChunkMetadata(**index_item.metadata) for index_item in chunks
    ]

    chunks_map = {}
    doc_url_to_doc = {doc["url"]: doc for doc in active_docs}

    for i, chunk_metadata in enumerate(chunks_metadatas, start=len(cards) + 1):
        doc_record = records[chunk_metadata["doc_id"]]
        chunk = doc_record.chunks[chunk_metadata["chunk_id"]]

        doc = doc_url_to_doc[chunk.metadata["source_display_name"]]

        if "page_number" in chunk.metadata:
            suffix = chunk.metadata["page_number"]
        else:
            offset = len(chunks_by_doc[doc["id"]]) + 1
            suffix = chunk.metadata["chunk_id"] + offset

        chunks_map[str(i)] = f'{doc["id"]}.{doc.get("version", 1)}.{suffix}'

    nodes_map = {
        str(i): node.metadata["id"] for i, node in enumerate(cards, start=1)
    }

    return nodes_map, chunks_map


async def run_chain(
    choice: Choice,
    chain: Runnable,
    input: dict,
    docs: List[Any],
    nodes: List[GraphData],
    client: DialClient,
    content_key: str = "content",
    attachment_keys: Iterable[str] = tuple("attachment"),
    records: List[DocumentRecord] = [],
    chunks_by_doc: Dict[str, List[Any]] = {},
) -> None:
    active_docs = [doc for doc in docs if doc.get("active", True)]

    if os.getenv("IS_EVAL_TEST_RUN", "0") == "1":
        item = await chain.ainvoke(input)
        with token_usage_context() as token_usage_manager:
            cb = TokenUsageByModelsCallback()
            run_config: RunnableConfig = {"callbacks": [cb]}
            item = await chain.ainvoke(input, config=run_config)
            priced_usage = await calc_token_usage_costs(token_usage_manager)

        logger.debug(item)

        full_content = ""
        attachment_graph = None

        nodes_map = {}
        chunks_map = {}

        for key in attachment_keys:
            if key in item:
                if key == "attachment_graph":
                    attachment_graph = item[key]
                elif item[key] and "content" in item[key]:
                    nodes_map, chunks_map = build_citation_maps(
                        item[key]["content"],
                        active_docs,
                        records,
                        chunks_by_doc,
                    )

        if content_key in item:
            full_content = item[content_key]

            if nodes_map or chunks_map:
                tq = TokensQueue(transofmation_map=nodes_map | chunks_map)
                processed_text = tq.add(full_content.response_text)
                processed_text += tq.tokens

                full_content.response_text = processed_text

            choice.append_content(full_content.response_text)

        # Partially adapted to StatGPT cost eval
        priced_usage_dict = [
            model_priced_usage.model_dump()
            for model_priced_usage in priced_usage
        ]
        choice.set_state(
            {**full_content.model_dump(), "priced_usage": priced_usage_dict}
        )

        # Build and add references attachment
        refs = json.dumps(
            await build_references(
                full_content.response_text, docs, nodes, client
            )
        )
        choice.add_attachment(
            type="application/vnd.dial.mindmap.references.v1+json",
            title="Used references",
            data=refs,
        )

        # Ensure the attachment graph exists and update it with the full content
        if attachment_graph:
            data = json.loads(attachment_graph["data"])
            # Assuming the structure is a list with at least one element
            if data and isinstance(data, list) and "data" in data[0]:
                data[0]["data"]["details"] = full_content.response_text
            attachment_graph["data"] = json.dumps(data)

            choice.add_attachment(**attachment_graph)
        else:
            logger.warning(
                "Expected 'attachment_graph' not found in the chain's output."
            )

    else:
        full_content = ""
        tokens_queue = None
        attachment_graph = None
        async for item in chain.astream(input):
            logger.debug(item)
            if content_key in item:
                if tokens_queue:
                    result = tokens_queue.add(item[content_key])

                    full_content += result
                    choice.append_content(result)
                else:
                    full_content = item[content_key]
                    choice.append_content(full_content)

                    refs = json.dumps(
                        await build_references(
                            full_content, docs, nodes, client
                        )
                    )
            for key in attachment_keys:
                if key in item:
                    if key == "attachment_graph":
                        attachment_graph = item[key]
                    else:
                        nodes_map, chunks_map = build_citation_maps(
                            item[key]["content"],
                            active_docs,
                            records,
                            chunks_by_doc,
                        )

                        tokens_queue = TokensQueue(
                            transofmation_map=nodes_map | chunks_map
                        )

        if tokens_queue:
            full_content += tokens_queue.tokens
            choice.append_content(tokens_queue.tokens)

        refs = json.dumps(
            await build_references(full_content, docs, nodes, client)
        )
        choice.add_attachment(
            type="application/vnd.dial.mindmap.references.v1+json",
            title="Used references",
            data=refs,
        )

        assert attachment_graph

        data = json.loads(attachment_graph["data"])
        data[0]["data"]["details"] = full_content
        attachment_graph["data"] = json.dumps(data)

        choice.add_attachment(**attachment_graph)
