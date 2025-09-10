import logging
import threading
import time
from functools import wraps
from typing import TYPE_CHECKING, AsyncGenerator, Callable, ClassVar

from generator.common import exceptions as exc
from generator.common.context import cur_llm_cost_handler, cur_run_id
from generator.common.llm import LLMCostHandler
from generator.common.structs import GeneratorStream, MMRequest

from .constants import LoggingMessages as Lm


class GeneratorConfigurator:
    @classmethod
    def configure(cls) -> None:
        """Configure the LLM generator environment.

        Sets up logging, LLM caching,
        and Langchain debugging options based on
        environment variables."""

        # Init LLM Cost Tracker
        llm_cost_handler = LLMCostHandler()
        cur_llm_cost_handler.set(llm_cost_handler)

        # Run id context variable
        run_id = HybridIDGenerator.get_next_id()
        cur_run_id.set(run_id)


if TYPE_CHECKING:
    # noinspection PyUnusedImports
    from generator.mind_map_generator import MindMapGenerator


GeneratorFuncType = Callable[
    ["MindMapGenerator", MMRequest], AsyncGenerator[GeneratorStream, None]
]


def handle_exceptions_and_logs(
    generator_func: GeneratorFuncType,
) -> GeneratorFuncType:
    """
    A decorator to handle common exception logging and final cost
    reporting for generator methods.
    """

    @wraps(generator_func)
    async def wrapper(
        self: "MindMapGenerator",
        request: MMRequest,
    ) -> AsyncGenerator[GeneratorStream, None]:
        func_name = generator_func.__name__.capitalize()
        logging.info(f"{func_name}: Start")
        try:
            async for item in generator_func(self, request):
                yield item
        except (
            exc.GenPipeException,
            exc.AddPipeException,
            exc.DelPipeException,
            exc.ApplyException,
            exc.GenerationException,
        ) as e:
            logging.exception(e.msg)
            raise
        finally:
            llm_cost_handler = cur_llm_cost_handler.get()
            logging.info(Lm.LLM_COST.format(llm_cost_handler))
            logging.info(f"{func_name}: End")

    return wrapper


class HybridIDGenerator:
    """
    A thread-safe hybrid ID generator
    that combines timestamp and a counter.

    Generates unique IDs by using microsecond-precision timestamps
    and a counter that increments when multiple IDs are requested
    within the same microsecond.
    """

    _prev_time: ClassVar[int] = 0
    _counter: ClassVar[int] = 0
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def get_next_id(cls) -> str:
        """
        Generate a unique ID by combining current time and a counter.

        Returns:
            str: A unique ID in the format "timestamp-counter"
        """
        with cls._lock:
            # Use time_ns() for better precision, convert to microseconds
            current_time = time.time_ns() // 1000

            if current_time == cls._prev_time:
                cls._counter += 1
            else:
                cls._counter = 0
                cls._prev_time = current_time

            return f"{current_time}-{cls._counter}"
