import asyncio as aio
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel


@dataclass
class ChainComponents:
    """
    Represents the components necessary
    to create a specific type of chain.

    Attributes:
        prompt_gen: Template or callable for generating prompts.
        response_format: The data structure format expected as output.
    """

    prompt_gen: Callable | ChatPromptTemplate
    response_format: type[BaseModel] | None


@dataclass
class ProgressTracker:
    """Helper to track progress of an asynchronous batch operation."""

    total: int
    status_update_func: Callable
    status_args: dict[str, Any]
    _lock: aio.Lock = aio.Lock()
    _counter: int = 0

    async def increment(self) -> None:
        """
        Increment the counter and trigger the status update callback.
        """
        async with self._lock:
            self._counter += 1
            await self.status_update_func(
                self._counter, self.total, **self.status_args
            )
