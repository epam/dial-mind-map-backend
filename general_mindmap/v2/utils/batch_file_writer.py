import asyncio
from typing import Any, Dict

import aiohttp
from pydantic import BaseModel

from general_mindmap.v2.dial.client import DialClient


class File(BaseModel):
    path: str
    etag: str | None = None


class JsonFile(File):
    content: Dict[str, Any]


class RawFile(File):
    content: bytes


class BatchFileWriter:
    files: list[File]
    client: DialClient

    def __init__(self, client: DialClient):
        self.files = []
        self.client = client

    def add_file(
        self, file: str, content: Dict[str, Any], etag: str | None = None
    ):
        self.files.append(JsonFile(path=file, content=content, etag=etag))

    def add_raw_file(self, file: str, content: bytes, etag: str | None = None):
        self.files.append(RawFile(path=file, content=content, etag=etag))

    async def write_file(self, file: File, session: aiohttp.ClientSession):
        if isinstance(file, JsonFile):
            return (
                file,
                (
                    await self.client.write_file(
                        file.path, file.content, session=session, etag=file.etag
                    )
                )[0],
            )
        elif isinstance(file, RawFile):
            return (
                file,
                (
                    await self.client.write_raw_file(
                        file.path, file.content, session=session, etag=file.etag
                    )
                )[0],
            )

    async def write(self):
        async with aiohttp.ClientSession() as session:
            tasks = [
                asyncio.ensure_future(self.write_file(file, session))
                for file in self.files
            ]
            return await asyncio.gather(*tasks)
