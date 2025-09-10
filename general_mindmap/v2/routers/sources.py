import base64
import io
import json
import zipfile
from io import BytesIO
from time import time
from typing import Any, Dict

import tiktoken
from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import SecretStr

from dial_rag.document_loaders import download_attachment
from dial_rag.document_record import DocumentRecord
from dial_rag.errors import NotEnoughDailyTokensError
from dial_rag.index_storage import SERIALIZATION_CONFIG
from general_mindmap.pptx.converter import (
    convert_pdf_to_images,
    convert_pptx_to_images,
)
from general_mindmap.v2.completion.dial_rag_integration import build_index
from general_mindmap.v2.config import DIAL_URL
from general_mindmap.v2.dial.client import DialClient, Lock
from general_mindmap.v2.models.metadata import (
    HistoryItem,
    HistoryItemType,
    HistoryStep,
)
from general_mindmap.v2.routers.utils.errors import timeout_after
from general_mindmap.v2.utils.batch_file_reader import (
    BatchFileReader,
    BatchRawFileReader,
)
from general_mindmap.v2.utils.batch_file_writer import BatchFileWriter
from generator.common.structs import Document
from generator.core.actions.docs import fetch_all_docs_content
from generator.core.stages.doc_handler import DocHandler

PPTX_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.presentationml.presentation"
)
PDF_CONTENT_TYPE = "application/pdf"

CONTENT_TYPE_TO_EXTENSION = {
    PDF_CONTENT_TYPE: "pdf",
    PPTX_CONTENT_TYPE: "pptx",
    "text/html": "html",
}

router = APIRouter()


def status_to_json(status: str):
    return {
        "status": (status if status == "INDEXED" else "FAILED"),
        "status_description": status if status != "INDEXED" else None,
    }


def add_generated_to_docs(docs: Dict[str, Any]):
    if "generated" not in docs:
        docs["generated"] = docs["generation_status"] not in [
            "NOT_STARTED",
            "IN_PROGRESS",
        ]


async def migrate_sources_from_old_format(
    client: DialClient, docs: Dict[str, Any]
):
    if "generation_status" not in docs:
        docs["generation_status"] = "FINISHED"

    add_generated_to_docs(docs)

    changed: bool = False
    file_reader: BatchFileReader = BatchFileReader(client)
    doc_path_to_doc_index = {}
    for i, doc in enumerate(docs["documents"]):
        if "active" not in doc:
            doc["active"] = True
        if "version" not in doc:
            doc["version"] = 1
        if "in_graph" not in doc:
            doc["in_graph"] = True
        if "type" not in doc:
            doc["type"] = "LINK"
            changed = True
        if "created" not in doc:
            doc_path = client._metadata.documents[doc["id"]]
            doc_path_to_doc_index[doc_path] = i

            file_reader.add_file(doc_path)
            changed = True

    start_time = time()
    for result in await file_reader.read():
        doc_data = result[1]

        if "rag_record" in doc_data:
            record = DocumentRecord.from_bytes(
                base64.b64decode(doc_data["rag_record"]),
                **SERIALIZATION_CONFIG,
            )

            encoding = tiktoken.encoding_for_model("gpt-4o")
            tokens = 0
            for chunk in record.chunks:
                tokens += len(encoding.encode(chunk.text))

            docs["documents"][doc_path_to_doc_index[result[0]]][
                "tokens"
            ] = tokens
        docs["documents"][doc_path_to_doc_index[result[0]]][
            "created"
        ] = start_time

    try:
        if changed:
            async with client:
                documents_file_name = f"{start_time}_documents"
                await client.write_file(documents_file_name, docs)
                client._metadata.documents_file = documents_file_name

                await client.close()
    except Exception:
        pass


@router.get("/v1/sources")
@timeout_after()
async def get_sources(request: Request):
    client = await DialClient.create_with_folder(
        DIAL_URL,
        "auto",
        json.loads(request.headers["x-dial-application-properties"])[
            "mindmap_folder"
        ],
        request.headers.get("etag", ""),
    )

    await client.read_metadata()

    if "documents_file" not in client._metadata.model_fields_set:
        return JSONResponse(
            {
                "sources": [],
                "generation_status": "NOT_STARTED",
                "generated": False,
            },
            headers={"ETag": client._etag},
        )

    docs, _ = await client.read_file_by_name_and_etag(
        client._metadata.documents_file
    )

    file_reader = BatchFileReader(client)

    storage_url_to_id = {}
    for i, source in enumerate(docs["documents"]):
        if "storage_url" in source:
            storage_url_to_id[source["storage_url"]] = i
            file_reader.add_file(source["storage_url"])

    for result in await file_reader.read():
        docs["documents"][storage_url_to_id[result[0]]] = result[1]

    await migrate_sources_from_old_format(client, docs)

    docs["sources"] = docs["documents"]
    del docs["documents"]

    docs["names"] = {
        source: name
        for source, name in client._metadata.source_names.items()
        if name
    }
    docs["params"] = client._metadata.params

    return JSONResponse(docs, headers={"ETag": client._etag})


def get_file_extension(content_type: str | None):
    if content_type and content_type in CONTENT_TYPE_TO_EXTENSION:
        return CONTENT_TYPE_TO_EXTENSION[content_type]
    else:
        raise HTTPException(400, "Unsupported content type")


async def add_file_source_background(
    client: DialClient,
    new_document: Any,
    file_data: bytes,
    content_type: str | None,
    source_lock: Lock,
):
    id = new_document["id"]
    version = new_document["version"]

    batchFileWriter = BatchFileWriter(client)

    if content_type in [PDF_CONTENT_TYPE, PPTX_CONTENT_TYPE]:
        pages = (
            await convert_pptx_to_images(file_data)
            if content_type == PPTX_CONTENT_TYPE
            else convert_pdf_to_images(file_data)
        )

        new_document["pages"] = []
        for i, page in enumerate(pages):
            buf = BytesIO()
            page.save(buf, format="JPEG")
            bytes = buf.getvalue()

            page_path = f"sources/{id}/versions/{version}/pages/{i + 1}.jpeg"
            new_document["pages"].append(page_path)
            batchFileWriter.add_raw_file(
                page_path,
                bytes,
            )

    status = "INDEXED"
    try:
        record = await build_index(
            client._dial_url,
            SecretStr(client._api_key),
            client.make_url_without_extension(new_document["url"]),
            new_document["url"],
        )

        if record:
            encoding = tiktoken.encoding_for_model("gpt-4o")
            tokens = 0
            for chunk in record.chunks:
                tokens += len(encoding.encode(chunk.text))

            new_document["tokens"] = tokens
        else:
            status = "Unable to index the source"
    except NotEnoughDailyTokensError:
        status = f"Not enough tokens for indexing"
        record = None

    doc_chunker = DocHandler(None)
    docs_and_their_content = await fetch_all_docs_content(
        [
            Document(
                id="TEMP",
                url=f"{client._folder}{new_document['url']}",
                type="FILE",
                content_type=new_document["content_type"],
            )
        ],
        client,
    )
    chunk_df, _ = await doc_chunker.chunk_docs(docs_and_their_content)

    if chunk_df is None:
        status = "Unable to index the source"
        record = None

    new_document = new_document | status_to_json(status)

    index_file = {
        "docstore": "",
        "chunks": (
            chunk_df["lc_doc"].apply(lambda x: x.model_dump_json()).tolist()
            if chunk_df is not None
            else []
        ),
    } | (
        {
            "rag_record": base64.b64encode(
                record.to_bytes(**SERIALIZATION_CONFIG)
            ).decode("utf-8")
        }
        if record
        else {}
    )

    index_file_name = f"sources/{id}/versions/{version}/index"
    batchFileWriter.add_file(index_file_name, index_file)
    new_document["index"] = index_file_name

    batchFileWriter.add_file(new_document["storage_url"], new_document)

    await batchFileWriter.write()

    await source_lock.release()


async def add_link_source_background(
    client: DialClient, new_document: Any, source_lock: Lock
):
    id = new_document["id"]
    version = new_document["version"]

    batchFileWriter = BatchFileWriter(client)

    status = "INDEXED"
    try:
        record = await build_index(
            client._dial_url,
            SecretStr(client._api_key),
            client.make_url_without_extension(new_document["copy_url"]),
            new_document["url"],
        )

        if record:
            encoding = tiktoken.encoding_for_model("gpt-4o")
            tokens = 0
            for chunk in record.chunks:
                tokens += len(encoding.encode(chunk.text))

            new_document["tokens"] = tokens
        else:
            status = "Unable to index the source"
    except NotEnoughDailyTokensError:
        status = f"Not enough tokens for indexing"
        record = None

    doc_chunker = DocHandler(None)
    docs_and_their_content = await fetch_all_docs_content(
        [
            Document(
                id="TEMP",
                url=f"{client._folder}{new_document['copy_url']}",
                type="FILE",
                content_type="text/html",
            )
        ],
        client,
    )
    chunk_df, _ = await doc_chunker.chunk_docs(docs_and_their_content)

    if chunk_df is None:
        status = "Unable to index the source"
        record = None

    new_document = new_document | status_to_json(status)

    index_file = {
        "docstore": "",
        "chunks": (
            chunk_df["lc_doc"].apply(lambda x: x.model_dump_json()).tolist()
            if chunk_df is not None
            else []
        ),
    } | (
        {
            "rag_record": base64.b64encode(
                record.to_bytes(**SERIALIZATION_CONFIG)
            ).decode("utf-8")
        }
        if record
        else {}
    )

    index_file_name = f"sources/{id}/versions/{version}/index"
    batchFileWriter.add_file(index_file_name, index_file)
    new_document["index"] = index_file_name

    batchFileWriter.add_file(new_document["storage_url"], new_document)

    await batchFileWriter.write()

    await source_lock.release()


@router.post("/v1/sources/{source_id}/versions/{version_id}/events")
async def source_status(request: Request, source_id: str, version_id: int):
    client = await DialClient.create_with_folder(
        DIAL_URL,
        "auto",
        json.loads(request.headers["x-dial-application-properties"])[
            "mindmap_folder"
        ],
        "",
    )

    await client.read_metadata()

    docs, _ = await client.read_file_by_name_and_etag(
        client._metadata.documents_file
    )

    for source in docs["documents"]:
        if source["id"] != source_id or source.get("version", 1) != version_id:
            continue

        if "storage_url" in source:
            return StreamingResponse(
                client.subscribe_to_source(
                    request,
                    (await client.read_file_and_etag(source["storage_url"]))[0],
                    source["storage_url"],
                ),
                media_type="text/event-stream",
            )
        else:
            return StreamingResponse(
                client.subscribe_to_source(
                    request,
                    source,
                    None,
                ),
                media_type="text/event-stream",
            )

    raise HTTPException(404, "Source doesn't exist")


@router.post("/v1/sources")
@timeout_after()
async def add_source(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile | None = File(None),
    link: str = Form(None),
    name: str = Form(""),
):
    start_time = time()

    if file and link:
        raise HTTPException(400, "Only file or link should be specified")

    client = await DialClient.create_with_folder(
        DIAL_URL,
        "auto",
        json.loads(request.headers["x-dial-application-properties"])[
            "mindmap_folder"
        ],
        request.headers.get("etag", ""),
    )
    await client.open()

    id = str(client._metadata.last_doc_id)
    client._metadata.last_doc_id += 1

    source_lock = client.create_lock(f"sources/{id}")
    await source_lock.acquire()

    new_document = {
        "id": id,
        "version": 1,
        "in_graph": False,
        "created": start_time,
        "updated": start_time,
        "status": "IN_PROGRESS",
        "status_description": None,
        "active": True,
    }

    try:
        last_docs_file = ""
        if "documents_file" not in client._metadata.model_fields_set:
            client._metadata.last_change = str(start_time)
            client._metadata.version = 3

            docs, docs_etag = {
                "documents": [],
                "generation_status": "NOT_STARTED",
            }, ""
        else:
            last_docs_file = client._metadata.documents_file
            docs, docs_etag = await client.read_file_by_name_and_etag(
                client._metadata.documents_file
            )

        documents_file_name = f"{start_time}_documents"
        new_document["storage_url"] = (
            f"sources/{id}/versions/1/{start_time}_state"
        )
        docs["documents"].append(new_document)

        if file:
            new_document["name"] = file.filename
            new_document["type"] = "FILE"
            new_document["content_type"] = file.content_type

            file_data = await file.read()

            file_path = f"sources/{id}/versions/1/source.{get_file_extension(file.content_type)}"
            await client.write_raw_file(
                file_path, file_data, content_type=file.content_type
            )
            new_document["url"] = file_path
        else:
            new_document["name"] = link
            new_document["type"] = "LINK"
            new_document["url"] = link
            new_document["content_type"] = "text/html"

            status = "INDEXED"
            try:
                _, file_data = await download_attachment(link, {})
            except:
                file_data = None
                status = "Unable to download the source"

            if file_data:
                file_path = f"sources/{id}/versions/1/source.html"
                await client.write_raw_file(
                    file_path, file_data, content_type="text/html"
                )
                new_document["copy_url"] = file_path

            if status != "INDEXED":
                new_document = new_document | status_to_json(status)

        batch_file_writer = BatchFileWriter(client)

        client._metadata.source_names[id] = name

        batch_file_writer.add_file(
            documents_file_name, filter_sources(docs), docs_etag
        )
        batch_file_writer.add_file(new_document["storage_url"], new_document)

        await batch_file_writer.write()

        client._metadata.documents_file = documents_file_name

        client._metadata.history.append(
            client._metadata,
            HistoryStep(
                user="USER",
                changes=[
                    HistoryItem(
                        old_value=last_docs_file,
                        new_value=documents_file_name,
                        type=HistoryItemType.SOURCES,
                    ),
                    HistoryItem(
                        old_value="",
                        new_value=name,
                        id=id,
                        type=HistoryItemType.SOURCE_STATE,
                    ),
                ],
            ),
        )

        if new_document["status"] == "IN_PROGRESS":
            if file:
                assert file_data
                background_tasks.add_task(
                    add_file_source_background,
                    client,
                    new_document,
                    file_data,
                    file.content_type,
                    source_lock,
                )
            else:
                background_tasks.add_task(
                    add_link_source_background,
                    client,
                    new_document,
                    source_lock,
                )
        else:
            await source_lock.release()
    except Exception as e:
        await client.release_lock()
        await source_lock.release()

        raise e

    etag = await client.close()

    return JSONResponse(content=new_document, headers={"ETag": etag})


@router.get("/v1/sources/{source_id}/versions/{version_id}/file")
async def download_source(request: Request, source_id: str, version_id: int):
    client = await DialClient.create_with_folder(
        DIAL_URL,
        "auto",
        json.loads(request.headers["x-dial-application-properties"])[
            "mindmap_folder"
        ],
        request.headers.get("etag", ""),
    )

    await client.read_metadata()

    docs, _ = await client.read_file_by_name_and_etag(
        client._metadata.documents_file
    )

    sources = docs["documents"]

    source = next(
        (
            source
            for source in sources
            if source["id"] == source_id
            and source.get("version", 1) == version_id
        ),
        None,
    )

    if source is None:
        raise HTTPException(404, "Not found")

    if "storage_url" in source:
        source, _ = await client.read_file_by_name_and_etag(
            source["storage_url"]
        )

    if "copy_url" in source:
        file_url = source["copy_url"]
    else:
        file_url = source["url"]

    return StreamingResponse(
        BytesIO(
            await client.read_raw_file_by_url(
                client.make_url_without_extension(file_url)
            )
        ),
        media_type=(
            source["content_type"] if source["content_type"] else "text/html"
        ),
    )


def filter_sources(sources: Any) -> Any:
    documents = sources["documents"]

    for i in range(len(documents)):
        if "storage_url" in documents[i]:
            documents[i] = {
                key: documents[i][key]
                for key in ["storage_url", "id", "version"]
            }

    return sources


@router.delete("/v1/sources/{source_id}")
async def delete_source(request: Request, source_id: str):
    start_time = str(time())

    async with await DialClient.create_with_folder(
        DIAL_URL,
        "auto",
        json.loads(request.headers["x-dial-application-properties"])[
            "mindmap_folder"
        ],
        request.headers.get("etag", ""),
    ) as client, client.create_lock(f"sources/{source_id}"):
        docs, _ = await client.read_file_by_name_and_etag(
            client._metadata.documents_file
        )

        file_reader = BatchFileReader(client)

        storage_url_to_id = {}
        for i, source in enumerate(docs["documents"]):
            if "storage_url" in source and source["id"] == source_id:
                storage_url_to_id[source["storage_url"]] = i
                file_reader.add_file(source["storage_url"])

        for result in await file_reader.read():
            docs["documents"][storage_url_to_id[result[0]]] = result[1]

        active_version = None
        in_graph = False
        for doc in docs["documents"]:
            if doc["id"] != source_id:
                continue

            in_graph = in_graph or doc.get("in_graph", True)

            if doc.get("active", True):
                active_version = doc

        if active_version is None:
            raise HTTPException(404, "Not found")

        history_step = HistoryStep(
            user="USER",
            changes=[],
        )
        assert history_step.changes is not None

        new_documents_file_name = f"{start_time}_documents"
        old_documents_file_name = client._metadata.documents_file

        if active_version["status"] == "FAILED":
            batch_file_writer = BatchFileWriter(client)

            docs["documents"] = [
                doc
                for doc in docs["documents"]
                if doc["id"] != source_id
                or doc.get("version", 1) != active_version.get("version", 1)
            ]

            versions = [
                doc for doc in docs["documents"] if doc["id"] == source_id
            ]
            if len(versions):
                max_version = max(
                    [version.get("version", 1) for version in versions]
                )
                new_active = next(
                    version
                    for version in versions
                    if version.get("version", 1) == max_version
                )
                new_active["active"] = True

                if "storage_url" in new_active:
                    new_active_storage_url = f"sources/{source_id}/versions/{max_version}/{start_time}_state"
                    new_active["storage_url"] = new_active_storage_url

                    batch_file_writer.add_file(
                        new_active_storage_url, new_active
                    )
            else:
                if "source_id" in client._metadata.source_names:
                    history_step.changes.append(
                        HistoryItem(
                            old_value=client._metadata.source_names[source_id],
                            new_value="",
                            type=HistoryItemType.SOURCE_STATE,
                            id=source_id,
                        ),
                    )

                    del client._metadata.source_names[source_id]

            history_step.changes.append(
                HistoryItem(
                    old_value=old_documents_file_name,
                    new_value=new_documents_file_name,
                    type=HistoryItemType.SOURCES,
                )
            )

            batch_file_writer.add_file(
                new_documents_file_name, filter_sources(docs)
            )

            await batch_file_writer.write()

            client._metadata.documents_file = new_documents_file_name
        elif in_graph:
            old_storage_url = f"{start_time}_{active_version['storage_url']}"

            await client.write_file(old_storage_url, active_version)

            active_version["status"] = "REMOVED"

            if "storage_url" in active_version:
                new_active_storage_url = f"sources/{source_id}/versions/{active_version.get('version', 1)}/{start_time}_state"
                active_version["storage_url"] = new_active_storage_url

                await client.write_file(new_active_storage_url, active_version)

            history_step.changes.append(
                HistoryItem(
                    old_value=old_documents_file_name,
                    new_value=new_documents_file_name,
                    type=HistoryItemType.SOURCES,
                )
            )

            await client.write_file(
                new_documents_file_name, filter_sources(docs)
            )
            client._metadata.documents_file = new_documents_file_name
        else:
            docs["documents"] = [
                doc for doc in docs["documents"] if doc["id"] != source_id
            ]

            await client.write_file(
                new_documents_file_name, filter_sources(docs)
            )
            client._metadata.documents_file = new_documents_file_name

            history_step.changes.append(
                HistoryItem(
                    old_value=old_documents_file_name,
                    new_value=new_documents_file_name,
                    type=HistoryItemType.SOURCES,
                ),
            )

            if "source_id" in client._metadata.source_names:
                history_step.changes.append(
                    HistoryItem(
                        old_value=client._metadata.source_names[source_id],
                        new_value="",
                        type=HistoryItemType.SOURCE_STATE,
                        id=source_id,
                    ),
                )

                del client._metadata.source_names[source_id]

        client._metadata.history.append(client._metadata, history_step)

        return Response(headers={"ETag": await client.close()})


@router.post("/v1/sources/{source_id}/versions")
@timeout_after()
async def add_version(
    request: Request,
    background_tasks: BackgroundTasks,
    source_id: str,
    file: UploadFile | None = File(None),
    link: str = Form(None),
):
    start_time = time()

    if file and link:
        raise HTTPException(400, "Only file or link should be specified")

    client = await DialClient.create_with_folder(
        DIAL_URL,
        "auto",
        json.loads(request.headers["x-dial-application-properties"])[
            "mindmap_folder"
        ],
        request.headers.get("etag", ""),
    )
    await client.open()
    source_lock = client.create_lock(f"sources/{source_id}")
    try:
        await source_lock.acquire()
    except Exception:
        await client.release_lock()

    history_step = HistoryStep(
        user="USER",
        changes=[],
    )
    assert history_step.changes is not None

    try:
        last_docs_file = client._metadata.documents_file
        docs, docs_etag = await client.read_file_by_name_and_etag(
            client._metadata.documents_file
        )
        sources = docs["documents"]

        file_reader = BatchFileReader(client)

        storage_url_to_id = {}
        for i, source in enumerate(sources):
            if "storage_url" in source and source["id"] == source_id:
                storage_url_to_id[source["storage_url"]] = i
                file_reader.add_file(source["storage_url"])

        for result in await file_reader.read():
            sources[storage_url_to_id[result[0]]] = result[1]

        batch_file_writer = BatchFileWriter(client)

        max_version = 1
        for source in sources:
            if source["id"] != source_id:
                continue

            if source.get("active", True):
                source["active"] = False

                if "storage_url" in source:
                    old_storage_url = source["storage_url"]
                    new_storage_url = f"sources/{source_id}/versions/{source.get('version', 1)}/{start_time}_state"
                    source["storage_url"] = new_storage_url

                    batch_file_writer.add_file(source["storage_url"], source)

                    history_step.changes.append(
                        HistoryItem(
                            old_value=old_storage_url,
                            new_value=new_storage_url,
                            type=HistoryItemType.SOURCES,
                            id=source_id,
                            version=source.get("version", 1),
                        ),
                    )

            max_version = max(max_version, source.get("version", 1))

        version = max_version + 1

        new_document = {
            "id": source_id,
            "version": version,
            "in_graph": False,
            "created": start_time,
            "updated": start_time,
            "status": "IN_PROGRESS",
            "status_description": None,
            "active": True,
        }

        documents_file_name = f"{start_time}_documents"
        new_document["storage_url"] = (
            f"sources/{source_id}/versions/{version}/{start_time}_state"
        )
        docs["documents"].append(new_document)

        if file:
            new_document["name"] = file.filename
            new_document["type"] = "FILE"
            new_document["content_type"] = file.content_type

            file_data = await file.read()

            file_path = f"sources/{source_id}/versions/{version}/source.{get_file_extension(file.content_type)}"
            await client.write_raw_file(
                file_path, file_data, content_type=file.content_type
            )
            new_document["url"] = file_path
        else:
            new_document["name"] = link
            new_document["type"] = "LINK"
            new_document["url"] = link
            new_document["content_type"] = "text/html"

            status = "INDEXED"
            try:
                _, file_data = await download_attachment(link, {})
            except:
                file_data = None
                status = "Unable to download the source"

            if file_data:
                file_path = (
                    f"sources/{source_id}/versions/{version}/source.html"
                )
                await client.write_raw_file(
                    file_path, file_data, content_type="text/html"
                )
                new_document["copy_url"] = file_path

            if status != "INDEXED":
                new_document = new_document | status_to_json(status)

        batch_file_writer.add_file(
            documents_file_name, filter_sources(docs), docs_etag
        )
        batch_file_writer.add_file(new_document["storage_url"], new_document)

        await batch_file_writer.write()

        client._metadata.documents_file = documents_file_name

        history_step.changes.append(
            HistoryItem(
                old_value=last_docs_file,
                new_value=documents_file_name,
                type=HistoryItemType.SOURCES,
            ),
        )

        client._metadata.history.append(client._metadata, history_step)

        if new_document["status"] == "IN_PROGRESS":
            if file:
                assert file_data
                background_tasks.add_task(
                    add_file_source_background,
                    client,
                    new_document,
                    file_data,
                    file.content_type,
                    source_lock,
                )
            else:
                background_tasks.add_task(
                    add_link_source_background,
                    client,
                    new_document,
                    source_lock,
                )
        else:
            await source_lock.release()
    except Exception as e:
        await source_lock.release()
        await client.release_lock()

        raise e

    etag = await client.close()

    return JSONResponse(content=new_document, headers={"ETag": etag})


@router.post("/v1/sources/{source_id}/versions/{version}")
@timeout_after()
async def try_process_again(
    request: Request,
    background_tasks: BackgroundTasks,
    source_id: str,
    version: int,
    file: UploadFile | None = File(None),
    link: str = Form(None),
):
    start_time = time()

    if file and link:
        raise HTTPException(400, "Only file or link should be specified")

    client = await DialClient.create_with_folder(
        DIAL_URL,
        "auto",
        json.loads(request.headers["x-dial-application-properties"])[
            "mindmap_folder"
        ],
        request.headers.get("etag", ""),
    )
    await client.open()
    source_lock = client.create_lock(f"sources/{source_id}")
    try:
        await source_lock.acquire()
    except Exception:
        await client.release_lock()

    try:
        docs, _ = await client.read_file_by_name_and_etag(
            client._metadata.documents_file
        )
        sources = docs["documents"]

        file_reader = BatchFileReader(client)

        storage_url_to_id = {}
        new_document = None
        for i, source in enumerate(sources):
            if source["id"] != source_id or source.get("version", 1) != version:
                continue

            new_document = source

            if "storage_url" in source:
                storage_url_to_id[source["storage_url"]] = i
                file_reader.add_file(source["storage_url"])

        for result in await file_reader.read():
            sources[storage_url_to_id[result[0]]] = result[1]
            new_document = result[1]

        assert new_document
        new_document["updated"] = start_time
        new_document["status"] = "IN_PROGRESS"
        new_document["status_description"] = None

        if file:
            new_document["name"] = file.filename
            new_document["type"] = "FILE"
            new_document["content_type"] = file.content_type

            file_data = await file.read()

            file_path = f"sources/{source_id}/versions/{version}/source.{get_file_extension(file.content_type)}"
            await client.write_raw_file(
                file_path, file_data, content_type=file.content_type
            )
            new_document["url"] = file_path
        else:
            new_document["name"] = link
            new_document["type"] = "LINK"
            new_document["url"] = link
            new_document["content_type"] = "text/html"

            status = "INDEXED"
            try:
                _, file_data = await download_attachment(link, {})
            except:
                file_data = None
                status = "Unable to download the source"

            if file_data:
                file_path = (
                    f"sources/{source_id}/versions/{version}/source.html"
                )
                await client.write_raw_file(
                    file_path, file_data, content_type="text/html"
                )
                new_document["copy_url"] = file_path

            if status != "INDEXED":
                new_document = new_document | status_to_json(status)

        await client.write_file(new_document["storage_url"], new_document)

        if new_document["status"] == "IN_PROGRESS":
            if file:
                assert file_data
                background_tasks.add_task(
                    add_file_source_background,
                    client,
                    new_document,
                    file_data,
                    file.content_type,
                    source_lock,
                )
            else:
                background_tasks.add_task(
                    add_link_source_background,
                    client,
                    new_document,
                    source_lock,
                )
        else:
            await source_lock.release()
    except Exception as e:
        await source_lock.release()
        await client.release_lock()

        raise e

    etag = await client.close()

    return JSONResponse(content=new_document, headers={"ETag": etag})


@router.post("/v1/sources/{source_id}/versions/{version}/active")
@timeout_after()
async def change_active_version(
    request: Request,
    source_id: str,
    version: int,
    file: UploadFile | None = File(None),
    link: str = Form(None),
):
    start_time = time()

    if file and link:
        raise HTTPException(400, "Only file or link should be specified")

    client = await DialClient.create_with_folder(
        DIAL_URL,
        "auto",
        json.loads(request.headers["x-dial-application-properties"])[
            "mindmap_folder"
        ],
        request.headers.get("etag", ""),
    )
    await client.open()
    source_lock = client.create_lock(f"sources/{source_id}")
    try:
        await source_lock.acquire()
    except Exception:
        await client.release_lock()

    history_step = HistoryStep(
        user="USER",
        changes=[],
    )
    assert history_step.changes is not None

    try:
        docs, docs_etag = await client.read_file_by_name_and_etag(
            client._metadata.documents_file
        )
        sources = docs["documents"]

        file_reader = BatchFileReader(client)

        storage_url_to_id = {}
        for i, source in enumerate(sources):
            if "storage_url" in source and source["id"] == source_id:
                storage_url_to_id[source["storage_url"]] = i
                file_reader.add_file(source["storage_url"])

        for result in await file_reader.read():
            sources[storage_url_to_id[result[0]]] = result[1]

        batch_file_writer = BatchFileWriter(client)

        for source in sources:
            if source["id"] != source_id:
                continue

            new_active_value = source.get("version", 1) == version

            if source.get("active", True) != new_active_value:
                source["active"] = new_active_value

                if "storage_url" in source:
                    new_storage_url = f"sources/{source_id}/versions/{source.get('version', 1)}/{start_time}_state"
                    source["storage_url"] = new_storage_url

                    batch_file_writer.add_file(new_storage_url, source)

        new_documents_file_name = f"{start_time}_documents"
        old_documents_file_name = client._metadata.documents_file
        client._metadata.documents_file = new_documents_file_name

        batch_file_writer.add_file(
            new_documents_file_name, filter_sources(docs), docs_etag
        )
        history_step.changes.append(
            HistoryItem(
                new_value=new_documents_file_name,
                old_value=old_documents_file_name,
                type=HistoryItemType.SOURCES,
            )
        )

        await batch_file_writer.write()

        client._metadata.history.append(client._metadata, history_step)
    except Exception as e:
        await client.release_lock()
        await source_lock.release()

        raise e

    await source_lock.release()
    etag = await client.close()

    return Response(status_code=200, headers={"ETag": etag})


@router.post("/v1/sources/{source_id}")
@timeout_after()
async def change_name(
    request: Request,
    source_id: str,
):
    async with await DialClient.create_with_folder(
        DIAL_URL,
        "auto",
        json.loads(request.headers["x-dial-application-properties"])[
            "mindmap_folder"
        ],
        request.headers.get("etag", ""),
    ) as client, client.create_lock(f"sources/{source_id}"):
        old_name = client._metadata.source_names.get(source_id, "")
        new_name = (await request.json())["name"] or ""

        client._metadata.source_names[source_id] = new_name

        client._metadata.history.append(
            client._metadata,
            HistoryStep(
                user="USER",
                changes=[
                    HistoryItem(
                        old_value=old_name,
                        new_value=new_name,
                        id=source_id,
                        type=HistoryItemType.SOURCE_STATE,
                    )
                ],
            ),
        )

        return Response(headers={"ETag": await client.close()})


@router.get("/v1/sources/export")
@timeout_after()
async def export(request: Request):
    client = await DialClient.create_with_folder(
        DIAL_URL,
        "auto",
        json.loads(request.headers["x-dial-application-properties"])[
            "mindmap_folder"
        ],
        request.headers.get("etag", ""),
    )

    await client.read_metadata()

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        files, next_token = await client.get_files_list(
            "/", "limit=1000&recursive=true"
        )

        while next_token != None:
            new_files, next_token = await client.get_files_list(
                "/", f"limit=1000&recursive=true&token={next_token}"
            )

            files += new_files

        file_reader = BatchRawFileReader(client)
        file_url_to_path = {}

        for file in files:
            full_url_to_file = f"{DIAL_URL}/v1/{file['url']}"

            file_reader.add_file(full_url_to_file)
            file_url_to_path[full_url_to_file] = full_url_to_file[
                len(client._folder) :
            ]

        for result in await file_reader.read():
            zipf.writestr(file_url_to_path[result[0]], result[1])

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=mind_map.zip"},
    )


@router.post("/v1/import")
@timeout_after()
async def import_file(
    request: Request,
    file: UploadFile,
):
    async with await DialClient.create_with_folder(
        DIAL_URL,
        "auto",
        json.loads(request.headers["x-dial-application-properties"])[
            "mindmap_folder"
        ],
        request.headers.get("etag", ""),
    ) as client:
        zip_buffer = io.BytesIO(await file.read())

        file_writer = BatchFileWriter(client)

        with zipfile.ZipFile(zip_buffer, "r") as zipf:
            for info in zipf.infolist():
                if not info.is_dir():
                    file_writer.add_raw_file(
                        info.filename, zipf.read(info.filename)
                    )

        await file_writer.write()

        return Response()
