import asyncio

import aiohttp
from theine import Cache

from general_mindmap.v2.dial.client import DialClient


class BatchFileReader:
    files: list[str]
    client: DialClient
    cache: Cache | None

    def __init__(self, client: DialClient, cache: Cache | None = None):
        self.files = []
        self.client = client
        self.cache = cache

    def add_file(self, file: str):
        self.files.append(file)

    async def read_file(self, file: str, session: aiohttp.ClientSession):
        file_name = self.client.make_url(file)

        cache_value = (
            self.cache.get(file_name) if self.cache is not None else None
        )

        if cache_value:
            return (file, cache_value)
        else:
            value = (
                await self.client.read_file_by_name_and_etag(
                    file, session=session
                )
            )[0]

            if self.cache is not None:
                self.cache.set(file_name, value)

            return (file, value)

    async def read(self):
        async with aiohttp.ClientSession() as session:
            tasks = [
                asyncio.ensure_future(self.read_file(file, session))
                for file in self.files
            ]
            return await asyncio.gather(*tasks)


class BatchRawFileReader:
    files: list[str]
    client: DialClient

    def __init__(self, client: DialClient):
        self.files = []
        self.client = client

    def add_file(self, file: str):
        self.files.append(file)

    async def read_file(self, file: str):
        return (file, await self.client.read_raw_file_by_url(file))

    async def read(self):
        tasks = [
            asyncio.ensure_future(self.read_file(file)) for file in self.files
        ]
        return await asyncio.gather(*tasks)
