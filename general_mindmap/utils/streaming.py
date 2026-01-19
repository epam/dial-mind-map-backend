import asyncio
from typing import AsyncIterable


async def stream_and_wait(
    stream: AsyncIterable[str], task: asyncio.Task | None
) -> AsyncIterable[str]:
    async for message in stream:
        yield message

    if task:
        await task
