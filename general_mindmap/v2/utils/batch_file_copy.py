import asyncio
import os

from general_mindmap.utils.log_config import logger

BATCH_COPY_REQUESTS_LIMIT = int(os.getenv("BATCH_COPY_REQUESTS_LIMIT", 300))


class BatchFileCopy:
    items: list[list[str]]
    client: "DialClient"

    def __init__(self, client: "DialClient"):
        self.items = []
        self.client = client

    def add_copy(self, source: str, target: str):
        self.items.append([source, target])

    async def copy_file(self, sem: asyncio.Semaphore, source: str, target: str):
        async with sem:
            return (
                [source, target],
                await self.client.copy(source, target, True),
            )

    async def copy(self):
        logger.info(f"Starting batch file copy (0/{len(self.items)})")

        sem = asyncio.Semaphore(BATCH_COPY_REQUESTS_LIMIT)

        tasks = [
            asyncio.ensure_future(self.copy_file(sem, item[0], item[1]))
            for item in self.items
        ]

        result = await asyncio.gather(*tasks)

        logger.info(
            f"Finished batch file copy ({len(self.items)}/{len(self.items)})"
        )

        return result
