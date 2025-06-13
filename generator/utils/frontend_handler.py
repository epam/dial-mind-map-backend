import asyncio as aio

from general_mindmap.v2.generator.base import StatusChunk
from .logger import logging


async def put_status(
    queue: aio.Queue, msg: str, details: str | None = None
) -> None:
    if details:
        await queue.put(StatusChunk(title=msg, details=details))
        logging.info(f"{msg}. Details: {details}")
        return None
    await queue.put(StatusChunk(title=msg))
    logging.info(msg)
    return None
