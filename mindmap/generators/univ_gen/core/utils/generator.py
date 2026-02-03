from functools import wraps
from typing import TYPE_CHECKING, AsyncGenerator, Callable, ParamSpec, Protocol

from mindmap.generators.base import GeneratorStream, MMRequest
from mindmap.generators.univ_gen.common.context import cur_llm_cost_handler
from mindmap.generators.univ_gen.common.llm import LLMCostHandler
from mindmap.utils.logger_config import logger

from ...common import exceptions as exc
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
    from mindmap.generators.univ_gen.mind_map_generator import MindMapGenerator


class GeneratorFuncType(Protocol):
    def __call__(
        self, generator: "MindMapGenerator", request: MMRequest
    ) -> AsyncGenerator[GeneratorStream, None]:
        raise NotImplementedError


P = ParamSpec("P")


def handle_exceptions_and_logs(
    # 2. Accept ANY function that returns an AsyncGenerator, regardless of input args
    generator_func: Callable[P, AsyncGenerator[GeneratorStream, None]],
) -> Callable[P, AsyncGenerator[GeneratorStream, None]]:
    """
    A decorator to handle common exception logging and final cost
    reporting for generator methods.
    """

    # 3. Apply the parameters (P) to the wrapper
    @wraps(generator_func)
    async def wrapper(
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> AsyncGenerator[GeneratorStream, None]:
        func_name = generator_func.__name__.capitalize()
        logger.info(f"{func_name}: Start")
        try:
            # Pass args directly. PyCharm now knows 'generator_func' returns an AsyncGenerator
            async for item in generator_func(*args, **kwargs):
                yield item
        except (
            exc.GenPipeException,
            exc.AddPipeException,
            exc.DelPipeException,
            exc.ApplyException,
            exc.GenerationException,
        ) as e:
            logger.exception(e)
            raise
        finally:
            llm_cost_handler = cur_llm_cost_handler.get()
            logger.info(Lm.LLM_COST.format(llm_cost_handler))
            logger.info(f"{func_name}: End")

    return wrapper
