from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_debug, set_llm_cache, set_verbose

from ..chainer.model_handler import LLMCostHandler
from . import constants as const
from .context import cur_llm_cost_handler, cur_run_id
from .logger import setup_logging
from .misc import HybridIDGenerator, env_to_bool


class Configurator:
    configured = False

    @classmethod
    def configure_generator(cls) -> None:
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

        if cls.configured is False:
            # Log Flag
            if env_to_bool(const.IS_LOG):
                setup_logging()

            # LLM Cache Flag
            if env_to_bool(const.IS_LLM_CACHE):
                sqlite_cache = SQLiteCache(const.LLM_CACHE_PATH)
                set_llm_cache(sqlite_cache)

            # Langchain Debug options
            set_debug(env_to_bool(const.IS_LANGCHAIN_DEBUG))
            set_verbose(env_to_bool(const.IS_LANGCHAIN_VERBOSE))

            cls.configured = True
