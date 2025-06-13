import asyncio

import aiohttp
from fastapi import HTTPException


class BatchFileDeleter:
    files: list[str]
    client: "DialClient"

    def __init__(self, client: "DialClient"):
        self.files = []
        self.client = client

    def add_file(self, file: str):
        self.files.append(file)

    async def delete_file(self, file: str, session: aiohttp.ClientSession):
        try:
            res = await self.client.delete_file(file, "", session)
        except HTTPException as e:
            if e.status_code == 404:
                return (file, True)
            else:
                print(
                    f"Can't delete a file {file} with status code {e.status_code}"
                )
                raise e
        except Exception as e:
            print(f"Can't delete a file {file}")
            raise e

        return (file, True)

    async def delete(self):
        if not len(self.files):
            return []

        async with aiohttp.ClientSession() as session:
            tasks = [
                asyncio.ensure_future(self.delete_file(file, session))
                for file in self.files
            ]
            return await asyncio.gather(*tasks)
