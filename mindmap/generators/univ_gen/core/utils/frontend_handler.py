import asyncio as aio

from mindmap.generators.base import StatusChunk
from mindmap.utils.logger_config import logger


async def put_status(
    queue: aio.Queue, msg: str, details: str | None = None
) -> None:
    if details:
        await queue.put(StatusChunk(title=msg, details=details))
        logger.info(f"{msg}. Details: {details}")
        return None
    await queue.put(StatusChunk(title=msg))
    logger.info(msg)
    return None
