import asyncio
from typing import Any, AsyncGenerator

from general_mindmap.v2.generator.base import StatusChunk


def cancel_tasks(tasks: list[asyncio.Task]) -> None:
    """
    Cancels all tasks in the provided list that are not yet done.

    Args:
        tasks: A list of asyncio.Task objects to potentially cancel.
    """
    for task in tasks:
        if not task.done():
            task.cancel()


async def process_task_w_queue(
    status_queue: asyncio.Queue, task: asyncio.Task
) -> AsyncGenerator[StatusChunk | Any, None]:
    """
    Monitors an asyncio task and yields status updates from a queue.

    Args:
        status_queue: An asyncio.Queue
            from which to receive status updates
            (e.g., StatusChunk objects) related to the task's progress.
        task: The asyncio.Task object to monitor
            for completion and retrieve the final result from.

    Yields
        - StatusChunk objects during task execution (progress updates)
        - The final task result when complete (could be any type)
    """
    while not task.done():
        try:
            status_chunk = await asyncio.wait_for(
                status_queue.get(),
                # Timeout prevents indefinite wait on queue,
                # allowing periodic check for task completion.
                timeout=5,
            )
            if status_chunk:
                yield status_chunk
        except asyncio.TimeoutError:
            # If timeout occurs,
            # check if the task is done and break if so.
            if task.done():
                break
    yield await task
