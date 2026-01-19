import os

from langchain_core.globals import set_debug, set_llm_cache, set_verbose

from ..cache_handler.db import TimestampedSQLiteCache
from .constants import EnvConsts as Env


class PackageConfigurator:
    """
    Provides a centralized, idempotent mechanism for package-wide
    configuration.

    This class is responsible for setting up environment variables,
    logging, caching, and other global states required for the package
    to operate correctly.

    The `configure` method is designed to be safe to call multiple
    times.
    """

    configured = False

    @classmethod
    def configure(cls) -> None:
        """
        Performs the actual package configuration.

        This method is idempotent and will have no effect on subsequent
        calls after the first successful execution. It performs the
        following actions:
        - Sets required environment variables for Azure compatibility.
        - Loads all application-specific settings from environment
            variables.
        - Initializes logging, LLM caching, and LangChain debug flags
            based on the loaded settings.
        """
        # --- Env variables ---
        # Populate the Env class with values from environment variables.
        Env.get_consts_from_env()

        # --- LangChain Azure OpenAI ---
        os.environ["AZURE_OPENAI_ENDPOINT"] = Env.DIAL_URL
        # The real API key is propagated through request headers by the
        # DIAL. However, it is required for this environment variable to
        # be set, so a mock value is used.
        os.environ["AZURE_OPENAI_API_KEY"] = "dial_api_key"
        os.environ["OPENAI_API_VERSION"] = "2024-10-21"

        # The following configurations are applied only once per
        # process.
        if not cls.configured:
            # --- Optional Feature Initialization ---

            # Configure LLM response caching if enabled.
            if Env.IS_LLM_CACHE:
                cache_dir = os.path.dirname(Env.LLM_CACHE_PATH)
                # Ensure the cache directory exists before initializing
                # the cache.
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)

                sqlite_cache = TimestampedSQLiteCache(Env.LLM_CACHE_PATH)
                set_llm_cache(sqlite_cache)

            # Apply LangChain-specific debug and verbosity settings.
            set_debug(Env.IS_LANGCHAIN_DEBUG)
            set_verbose(Env.IS_LANGCHAIN_VERBOSE)

            cls.configured = True
