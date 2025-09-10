import asyncio as aio
import logging

from generator.common.structs import StatusChunk


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
