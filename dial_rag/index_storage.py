import aiohttp
import hashlib
import io
import logging
from abc import ABC, abstractmethod
from cachetools import LRUCache
from typing import cast

from dial_rag.attachment_link import AttachmentLink
from dial_rag.document_record import DocumentRecord, FORMAT_VERSION, IndexSettings
from dial_rag.request_context import RequestContext
from dial_rag.utils import bool_env_var, size_env_var


USE_DIAL_FILE_STORAGE: bool = bool_env_var('USE_DIAL_FILE_STORAGE', default=False)
IN_MEMORY_CACHE_CAPACITY_IN_BYTES: int = size_env_var('IN_MEMORY_CACHE_CAPACITY', '128MiB')


SERIALIZATION_CONFIG = dict(
    protocol="pickle",
    compress="gzip"
)

INDEX_MIME_TYPE = "application/x.aidial-rag-index.v0"


def link_to_key(attachment_link: AttachmentLink) -> str:
    return hashlib.sha256(attachment_link.dial_link.encode()).hexdigest()


class IndexStorageBackend(ABC):
    @abstractmethod
    async def load(self, key: str, request_context: RequestContext) -> bytes | None:
        pass

    @abstractmethod
    async def store(self, key: str, data: bytes, request_context: RequestContext) -> dict:
        pass


class LRUCacheStorage(IndexStorageBackend):
    def __init__(self, capacity: int = IN_MEMORY_CACHE_CAPACITY_IN_BYTES):
        self._cache = LRUCache(maxsize=capacity, getsizeof=len)

    async def load(self, key: str, request_context: RequestContext) -> bytes | None:
        return cast(bytes | None, self._cache.get(key))

    async def store(self, key, data: bytes, request_context: RequestContext) -> dict:
        self._cache[key] = data
        return {}


class DialFileStorage(IndexStorageBackend):
    """Dial File API storage for index files
    Implements API from https://gitlab.deltixhub.com/Deltix/openai-apps/documentation/-/issues/12
    """

    def __init__(self, dial_url: str):
        self._dial_url = dial_url

    async def get_bucket_id(self, headers: dict) -> str:
        # Cannot initialize bucket_id in __init__ because we need api-key headers to access the bucket
        bucket_url = f"{self._dial_url}/v1/bucket"
        async with aiohttp.ClientSession() as session:
            async with session.get(bucket_url, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                return data['bucket']


    async def index_url(self, key: str, headers: dict, dir_part_size: int = 8) -> str:
        bucket_id = await self.get_bucket_id(headers)

        # split the key into parts to avoid too many files in one directory
        dir_path = "/".join(
            key[i:i + dir_part_size]
            for i in range(0, len(key), dir_part_size)
        )

        return f"{self._dial_url}/v1/files/{bucket_id}/dial-rag-index/{dir_path}/index.bin"


    def to_form_data(self, key: str, data: bytes) -> aiohttp.FormData:
        form_data = aiohttp.FormData()
        form_data.add_field(
            'file',
            io.BytesIO(data),
            filename=key,
            content_type=INDEX_MIME_TYPE)
        return form_data


    async def load(self, key: str, request_context: RequestContext) -> bytes | None:
        async with aiohttp.ClientSession() as session:
            headers = request_context.get_api_key_headers()
            url = await self.index_url(key, headers)
            async with session.get(url, headers=headers) as response:
                if not response.ok:
                    logging.warning(f"Failed to load index from {url}: {response.status}, {response.reason}")
                    return None
                return await response.read()


    async def store(self, key, data: bytes, request_context: RequestContext) -> dict:
        async with aiohttp.ClientSession() as session:
            headers = request_context.get_api_key_headers()
            url = await self.index_url(key, headers)
            form_data = self.to_form_data(key, data)
            async with session.put(url, data=form_data, headers=headers) as response:
                response.raise_for_status()
                return await response.json()


class CachedStorage(IndexStorageBackend):
    def __init__(self, storage: IndexStorageBackend, capacity: int = IN_MEMORY_CACHE_CAPACITY_IN_BYTES):
        self._storage = storage
        self._cache = LRUCacheStorage(capacity)

    async def load(self, key: str, request_context: RequestContext) -> bytes | None:
        data = await self._cache.load(key, request_context)
        if data is not None:
            return data

        data = await self._storage.load(key, request_context)
        if data is not None:
            await self._cache.store(key, data, request_context)
        return data

    async def store(self, key, data: bytes, request_context: RequestContext) -> dict:
        await self._cache.store(key, data, request_context)
        return await self._storage.store(key, data, request_context)


class IndexStorage:
    def __init__(self, dial_url):
        if USE_DIAL_FILE_STORAGE:
            self._storage = CachedStorage(DialFileStorage(dial_url))
        else:
            self._storage = LRUCacheStorage()

    async def load(
        self,
        attachment_link: AttachmentLink,
        index_settings: IndexSettings,
        request_context: RequestContext,
    ) -> DocumentRecord | None:
        doc_record_bytes = await self._storage.load(link_to_key(attachment_link), request_context)
        if doc_record_bytes is None:
            return None
        try:
            doc_record = DocumentRecord.from_bytes(doc_record_bytes, **SERIALIZATION_CONFIG)
            if doc_record.format_version != FORMAT_VERSION:
                logging.warning(f"Index format version mismatch for {attachment_link}: {doc_record.format_version}")
                return None
            if doc_record.index_settings != index_settings:
                logging.warning(f"Index settings mismatch for {attachment_link}: {doc_record.index_settings}")
                return None
            return doc_record
        except Exception as e:
            logging.warning(f"Failed to deserialize index for {attachment_link}: {e}")
            return None

    async def store(
        self,
        attachment_link: AttachmentLink,
        doc_record: DocumentRecord,
        request_context: RequestContext
    ) -> dict:
        doc_record_bytes = doc_record.to_bytes(**SERIALIZATION_CONFIG)
        key = link_to_key(attachment_link)
        logging.debug(f"Stored document {attachment_link} with key: {key}")
        return await self._storage.store(key, doc_record_bytes, request_context)
