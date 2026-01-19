import asyncio
import os
from typing import Any, Dict

import aiohttp
from pydantic import BaseModel

from general_mindmap.utils.log_config import logger
from general_mindmap.v2.dial.client import DialClient

BATCH_WRITE_REQUESTS_LIMIT = int(os.getenv("BATCH_WRITE_REQUESTS_LIMIT", 300))


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

    async def write_file(
        self, sem: asyncio.Semaphore, file: File, session: aiohttp.ClientSession
    ):
        async with sem:
            if isinstance(file, JsonFile):
                return (
                    file,
                    (
                        await self.client.write_file(
                            file.path,
                            file.content,
                            session=session,
                            etag=file.etag,
                            batch=True,
                        )
                    )[0],
                )
            elif isinstance(file, RawFile):
                return (
                    file,
                    (
                        await self.client.write_raw_file(
                            file.path,
                            file.content,
                            session=session,
                            etag=file.etag,
                            batch=True,
                        )
                    )[0],
                )

    async def write(self):
        logger.info(f"Starting batch file write (0/{len(self.files)})")

        sem = asyncio.Semaphore(BATCH_WRITE_REQUESTS_LIMIT)

        async with aiohttp.ClientSession() as session:
            tasks = [
                asyncio.ensure_future(self.write_file(sem, file, session))
                for file in self.files
            ]

            result = await asyncio.gather(*tasks)

        logger.info(
            f"Finished batch file write ({len(self.files)}/{len(self.files)})"
        )

        return result
