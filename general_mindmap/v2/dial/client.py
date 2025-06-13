import base64
import io
import json
from time import time
from typing import Any, List, Tuple

import aiohttp
from fastapi import HTTPException, Request
from langchain_community.vectorstores import FAISS
from matplotlib.pyplot import flag
from pydantic import SecretStr, ValidationError

from dial_rag.index_storage import SERIALIZATION_CONFIG
from general_mindmap.models.graph import Node
from general_mindmap.utils.docstore import encode_docstore
from general_mindmap.utils.graph_patch import embeddings_model, node_to_document
from general_mindmap.utils.log_config import logger
from general_mindmap.v2.completion.dial_rag_integration import build_index
from general_mindmap.v2.models.metadata import Metadata
from general_mindmap.v2.utils.batch_file_deleter import BatchFileDeleter

LOCK_EXPIRATION_SEC = 60
PROCESSING_LIMIT = 45


class Lock:
    _client: "DialClient"
    _path: str | None
    _lock_etag: str | None = None

    def __init__(self, client: "DialClient", path: str | None = None):
        self._client = client
        self._path = path

    def _full_path(self):
        if self._path:
            return f"{self._path}/lock"
        else:
            return "lock"

    async def _update(self):
        if self._lock_etag:
            self._lock_etag = await self._client.write_file(
                self._full_path(),
                {"expired": time() + LOCK_EXPIRATION_SEC},
                self._lock_etag,
            )
        else:
            self._lock_etag = await self._client.write_file(
                self._full_path(),
                {"expired": time() + LOCK_EXPIRATION_SEC},
                "-",
            )

    async def acquire(self):
        try:
            await self._update()
        except HTTPException as e:
            if e.status_code == 412:
                lock_etag = e.detail.split()[-1]

                lock, _ = await self._client.read_file_by_name_and_etag(
                    self._full_path(), lock_etag
                )

                if lock["expired"] <= time():
                    try:
                        await self._client.delete_file(
                            self._full_path(), lock_etag
                        )

                        await self._update()
                    except HTTPException as e:
                        if e.status_code == 404:
                            await self._update()
                        else:
                            raise e
                else:
                    raise e
            else:
                raise e

    async def release(self):
        await self._client.delete_file(self._full_path(), self._lock_etag)

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()
        return self


class DialClient:
    _api_key: str
    _dial_url: str
    _folder: str
    _metadata: Metadata

    _etag: str
    _lock_etag: str | None = None
    _opened: bool = False
    _closed: bool = False

    @classmethod
    async def create_with_folder(
        cls, dial_url: str, api_key: str, folder: str, etag: str
    ) -> "DialClient":
        self = cls(dial_url, api_key, etag)
        self._folder = f"{self._dial_url}/v1/{folder}"
        return self

    def __init__(self, dial_url: str, api_key: str, etag: str):
        self._dial_url = dial_url
        self._api_key = api_key
        self._etag = etag

    def create_lock(self, path: str) -> Lock:
        return Lock(self, path)

    async def __aenter__(self):
        if self._opened:
            return self

        return await self.open()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._closed:
            return

        await self.release_lock()

    async def migrate_1_to_2(self):
        from general_mindmap.v2.utils.batch_file_reader import BatchFileReader
        from general_mindmap.v2.utils.batch_file_writer import BatchFileWriter

        start_time = str(time())

        logger.info(f"Migration started for {self._folder}")

        async with self:
            batch_file_reader = BatchFileReader(self)
            batch_file_writer = BatchFileWriter(self)

            batch_file_reader.add_file(self._metadata.documents_file)

            file_name_to_id = {}
            for id, file in self._metadata.documents.items():
                batch_file_reader.add_file(file)
                file_name_to_id[file] = id

            docs = {}
            docs_file = None
            for result in await batch_file_reader.read():
                if result[0] == self._metadata.documents_file:
                    docs_file = result[1]["documents"]
                else:
                    id = file_name_to_id[result[0]]

                    docs[id] = result[1]

            assert docs_file
            docs_metainfo = {doc["id"]: doc for doc in docs_file}

            for id, doc in docs.items():
                if "rag_record" not in doc:
                    record = await build_index(
                        self._dial_url,
                        SecretStr(self._api_key),
                        (
                            docs_metainfo[id]["url"]
                            if docs_metainfo[id]["type"] == "LINK"
                            else self.make_url_without_extension(
                                docs_metainfo[id]["url"]
                            )
                        ),
                        docs_metainfo[id]["url"],
                    )

                    doc = doc | (
                        {
                            "rag_record": base64.b64encode(
                                record.to_bytes(**SERIALIZATION_CONFIG)
                            ).decode("utf-8")
                        }
                        if record
                        else {}
                    )

                    new_file_name = f"documents/{start_time}_{id}"
                    batch_file_writer.add_file(new_file_name, doc)
                    self._metadata.documents[id] = new_file_name

                    logger.info(f"Migrated document #{id}")

            await batch_file_writer.write()

            self._metadata.version = 2
            self._metadata.docs_history.steps = []
            self._metadata.docs_history.current_step = 0

            await self.close()

        logger.info(f"Migration finished for {self._folder}")

    async def migrate_2_to_3(self):
        from general_mindmap.v2.utils.batch_file_reader import BatchFileReader
        from general_mindmap.v2.utils.batch_file_writer import BatchFileWriter

        start_time = str(time())

        logger.info(f"Migration 2 to 3 started for {self._folder}")

        async with self:
            batch_file_reader = BatchFileReader(self)
            batch_file_writer = BatchFileWriter(self)

            assert self._metadata.nodes_file
            batch_file_reader.add_file(self._metadata.nodes_file)

            file_name_to_id = {}
            for id, file in self._metadata.nodes.items():
                batch_file_reader.add_file(file)
                file_name_to_id[file] = id

            nodes = {}
            nodes_file = None
            for result in await batch_file_reader.read():
                if result[0] == self._metadata.nodes_file:
                    nodes_file = result[1]["nodes"]
                else:
                    id = file_name_to_id[result[0]]

                    nodes[id] = result[1]

            assert nodes_file
            nodes_by_id = {
                node["data"]["id"]: Node.model_validate(node["data"])
                for node in nodes_file
            }

            for id, node in nodes.items():
                if "patcher_docstore" not in node:
                    docs = [node_to_document(nodes_by_id[id])]
                    docstore = FAISS.from_documents(docs, embeddings_model)

                    node = node | {
                        "patcher_docstore": encode_docstore(docstore)
                    }

                    new_file_name = f"nodes/{start_time}_{id}"
                    batch_file_writer.add_file(new_file_name, node)
                    self._metadata.nodes[id] = new_file_name

                    logger.info(f"Migrated node #{id}")

            await batch_file_writer.write()

            self._metadata.version = 3
            self._metadata.graph_history.steps = []
            self._metadata.graph_history.current_step = 0

            await self.close()

        logger.info(f"Migration finished for {self._folder}")

    async def read_metadata(self):
        try:
            raw_metadata, self._etag = await self.read_file_by_name_and_etag(
                "metadata", self._etag
            )
            self._metadata = Metadata.model_validate(raw_metadata)

            if self._opened == False:
                if self._metadata.version == 1:
                    await self.migrate_1_to_2()

                if self._metadata.version == 2:
                    await self.migrate_2_to_3()

        except ValidationError:
            self._metadata = Metadata.model_construct()
        except HTTPException as e:
            if e.status_code == 404:
                self._metadata = Metadata.model_construct()
            else:
                raise e

    async def update_lock(self):
        if self._lock_etag:
            self._lock_etag = await self.write_file(
                "lock",
                {"expired": time() + LOCK_EXPIRATION_SEC},
                self._lock_etag,
            )
        else:
            self._lock_etag = await self.write_file(
                "lock", {"expired": time() + LOCK_EXPIRATION_SEC}, "-"
            )

    async def open(self):
        try:
            await self.update_lock()
        except HTTPException as e:
            if e.status_code == 412:
                lock_etag = e.detail.split()[-1]

                try:
                    lock, _ = await self.read_file_by_name_and_etag(
                        "lock", lock_etag
                    )

                    if lock["expired"] <= time():
                        await self.delete_file("lock", lock_etag)

                        await self.update_lock()
                    else:
                        raise e
                except HTTPException as e:
                    if e.status_code == 404:
                        await self.update_lock()
                    else:
                        raise e
            else:
                raise e

        self._opened = True

        await self.read_metadata()

        return self

    async def release_lock(self):
        await self.delete_file("lock", self._lock_etag)

    async def close(self) -> str:
        current_time = time()

        batch_deleter = BatchFileDeleter(self)

        for file in self._metadata.to_delete:
            if current_time >= file.expired:
                batch_deleter.add_file(file.file)

        deleted = set()
        for result in await batch_deleter.delete():
            deleted.add(result[0])

        self._metadata.to_delete = [
            file
            for file in self._metadata.to_delete
            if file.file not in deleted
        ]

        etag = await self.write_file(
            "metadata", self._metadata.model_dump(exclude_none=True), self._etag
        )

        await self.release_lock()

        self._closed = True

        return etag

    async def read_raw_file_by_url(self, url: str) -> bytes:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers={"Authorization": self._api_key},
            ) as response:
                if response.status != 200:
                    raise HTTPException(status_code=response.status)

                return await response.read()

    async def read_file_by_url(self, url: str) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers={"Authorization": self._api_key},
            ) as response:
                if response.status != 200:
                    raise HTTPException(status_code=response.status)

                return json.loads(await response.text())

    def make_url(self, file_name: str):
        return f"{self._folder}{file_name}.json"

    def make_url_without_extension(self, file_with_extension: str):
        return f"{self._folder}{file_with_extension}"

    async def read_file_by_name_and_etag(
        self,
        file_name: str,
        etag: str = "",
        session: aiohttp.ClientSession | None = None,
    ) -> Tuple[Any, str]:
        if session is None:
            async with aiohttp.ClientSession() as session:
                return await self.read_file(file_name, etag, session)
        else:
            return await self.read_file(file_name, etag, session)

    async def read_file(
        self, file_name: str, etag: str, session: aiohttp.ClientSession
    ) -> Tuple[Any, str]:
        logger.info(f"Read {file_name}.json")
        async with session.get(
            self.make_url(file_name),
            headers={"Authorization": self._api_key, "If-Match": etag},
        ) as response:
            if response.status != 200:
                raise HTTPException(status_code=response.status)
            return (
                json.loads(await response.text()),
                response.headers["ETag"],
            )

    async def read_file_and_etag(self, file_name: str) -> Tuple[Any, str]:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.make_url(file_name),
                headers={"Authorization": self._api_key},
            ) as response:
                if response.status != 200:
                    raise HTTPException(status_code=response.status)

                return (
                    json.loads(await response.text()),
                    response.headers["ETag"],
                )

    async def write_file(
        self,
        file_name: str,
        content: dict,
        etag: str | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> str:
        logger.info(f"Write to {file_name}.json")
        if session is None:
            async with aiohttp.ClientSession() as session:
                async with session.put(
                    self.make_url(file_name),
                    headers=(
                        {"Authorization": self._api_key} | {"If-Match": etag}
                        if etag
                        else {}
                    ),
                    data={"key": io.StringIO(json.dumps(content))},
                ) as response:
                    if response.status != 200:
                        raise HTTPException(
                            detail=await response.text(),
                            status_code=response.status,
                        )
                    return response.headers["ETag"]
        else:
            async with session.put(
                self.make_url(file_name),
                headers=(
                    {"Authorization": self._api_key} | {"If-Match": etag}
                    if etag
                    else {}
                ),
                data={"key": io.StringIO(json.dumps(content))},
            ) as response:
                if response.status != 200:
                    raise HTTPException(
                        detail=await response.text(),
                        status_code=response.status,
                    )
                return response.headers["ETag"]

    async def write_raw_file(
        self,
        file_name: str,
        content: bytes,
        etag: str | None = None,
        session: aiohttp.ClientSession | None = None,
        content_type: str | None = None,
    ) -> str:
        logger.info(f"Write to {file_name}")
        if session is None:
            async with aiohttp.ClientSession() as session:
                data = aiohttp.FormData()
                data.add_field(
                    "key", io.BytesIO(content), content_type=content_type
                )

                async with session.put(
                    self.make_url_without_extension(file_name),
                    headers=(
                        {"Authorization": self._api_key}
                        | ({"If-Match": etag} if etag else {})
                    ),
                    data=data,
                ) as response:
                    if response.status != 200:
                        raise HTTPException(
                            detail=await response.text(),
                            status_code=response.status,
                        )
                    return response.headers["ETag"]
        else:
            async with session.put(
                self.make_url_without_extension(file_name),
                headers=(
                    {"Authorization": self._api_key} | {"If-Match": etag}
                    if etag
                    else {}
                ),
                data={"key": io.BytesIO(content)},
            ) as response:
                if response.status != 200:
                    raise HTTPException(
                        detail=await response.text(),
                        status_code=response.status,
                    )
                return response.headers["ETag"]

    async def delete_file(
        self,
        file_name: str,
        etag: str | None,
        session: aiohttp.ClientSession | None = None,
    ):
        logger.info(f"Delete {file_name}.json")

        if session is None:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    self.make_url(file_name),
                    headers=(
                        {"Authorization": self._api_key} | {"If-Match": etag}
                        if etag
                        else {}
                    ),
                ) as response:
                    if response.status != 200:
                        raise HTTPException(status_code=response.status)
        else:
            async with session.delete(
                self.make_url(file_name),
                headers=(
                    {"Authorization": self._api_key} | {"If-Match": etag}
                    if etag
                    else {}
                ),
            ) as response:
                if response.status != 200:
                    raise HTTPException(status_code=response.status)

    async def subscribe(self, request: Request):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._dial_url}/v1/ops/resource/subscribe",
                json={
                    "resources": [
                        {
                            "url": f"{self._folder}metadata.json".removeprefix(
                                f"{self._dial_url}/v1/"
                            )
                        }
                    ]
                },
                headers={"Authorization": self._api_key},
                timeout=aiohttp.ClientTimeout(),
            ) as response:
                async for line in response.content:
                    if await request.is_disconnected():
                        return

                    line = line.decode("utf-8").strip()

                    if line.startswith("data:"):
                        data = json.loads(line[len("data:") :])

                        del data["url"]
                        data["action"] = "UPDATE"

                        yield f"data: {json.dumps(data, separators=(',',':'))}\n\n"

                    if line.startswith(": heartbeat"):
                        yield f"{line}\n\n"

    async def subscribe_to_generate(
        self, request: Request, current_status: Any
    ):
        yield f"data: {json.dumps(current_status, separators=(',',':'))}\n\n"

        if (
            current_status.get("title") == "Graph generated"
            or "error" in current_status
        ):
            return

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._dial_url}/v1/ops/resource/subscribe",
                json={
                    "resources": [
                        {
                            "url": f"{self._folder}generate.json".removeprefix(
                                f"{self._dial_url}/v1/"
                            )
                        }
                    ]
                },
                headers={"Authorization": self._api_key},
                timeout=aiohttp.ClientTimeout(),
            ) as response:
                async for line in response.content:
                    if await request.is_disconnected():
                        return

                    line = line.decode("utf-8").strip()

                    if line.startswith("data:"):
                        data, _ = await self.read_file_and_etag("generate")
                        yield f"data: {json.dumps(data, separators=(',',':'))}\n\n"

                        if (
                            data.get("title") == "Graph generated"
                            or "error" in data
                        ):
                            return

                    if line.startswith(": heartbeat"):
                        yield f"{line}\n\n"

    async def subscribe_to_source(
        self, request: Request, current_state: Any, url: str | None
    ):
        yield f"data: {json.dumps(current_state, separators=(',',':'))}\n\n"

        if current_state["status"] != "IN_PROGRESS" or url is None:
            return

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._dial_url}/v1/ops/resource/subscribe",
                json={
                    "resources": [
                        {
                            "url": f"{self._folder}{url}.json".removeprefix(
                                f"{self._dial_url}/v1/"
                            )
                        }
                    ]
                },
                headers={"Authorization": self._api_key},
                timeout=aiohttp.ClientTimeout(),
            ) as response:
                async for line in response.content:
                    if await request.is_disconnected():
                        return

                    line = line.decode("utf-8").strip()

                    if line.startswith("data:"):
                        data, _ = await self.read_file_and_etag(url)
                        yield f"data: {json.dumps(data, separators=(',',':'))}\n\n"

                        if data["status"] != "IN_PROGRESS":
                            return

                    if line.startswith(": heartbeat"):
                        yield f"{line}\n\n"
