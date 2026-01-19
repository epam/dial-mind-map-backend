from functools import wraps
from typing import TYPE_CHECKING, AsyncGenerator, Callable

from common_utils.logger_config import logger
from generator.common import exceptions as exc
from generator.common.context import cur_llm_cost_handler
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
        logger.info(f"{func_name}: Start")
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
            logger.exception(e.msg)
            raise
        finally:
            llm_cost_handler = cur_llm_cost_handler.get()
            logger.info(Lm.LLM_COST.format(llm_cost_handler))
            logger.info(f"{func_name}: End")

    return wrapper
