import os
from functools import lru_cache
from math import ceil
from typing import Optional

import httpx
import tiktoken
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from tiktoken import Encoding

from generator.common.constants import EnvConsts

from ..adapter import embeddings_model
from .utils import constants as const


class ModelCreator:
    """
    Single source for creating and retrieving LLM and embedding models.
    All creation methods are cached to prevent re-instantiating objects.
    """

    _timeout = httpx.Timeout(150, connect=5.0)
    _rate_limiter = InMemoryRateLimiter()

    _INNER_EMBEDDING_MODELS = {"text-embedding-3-large": AzureOpenAIEmbeddings}

    @classmethod
    @lru_cache(maxsize=4)
    def get_chat_model(
        cls,
        model_name: str = const.DEFAULT_CHAT_MODEL_NAME,
        total_timeout: float = 150.0,
        read_timeout: Optional[float] = None,
        connect_timeout: float = 5.0,
    ) -> AzureChatOpenAI:
        """
        Get a cached instance of a chat model.

        Args:
            model_name: The name of the model deployment.
            total_timeout: The total time for the entire request in
                seconds.
            read_timeout: The time to wait between receiving data
                chunks. Defaults to None (no read timeout).
            connect_timeout: The time to wait for establishing a
                connection.
        """
        timeout_obj = httpx.Timeout(
            total_timeout, read=read_timeout, connect=connect_timeout
        )

        model_kwargs = {
            "azure_deployment": model_name,
            "model": model_name,
            "temperature": 0.0,
            "timeout": timeout_obj,
            "rate_limiter": cls._rate_limiter,
        }

        if not "reasoning" in model_name.lower():
            seed_str = os.getenv(const.CHAT_MODEL_SEED)
            if seed_str is not None:
                model_kwargs["seed"] = (
                    int(seed_str) if seed_str is not None else None
                )

        if model_name.lower().startswith("gpt-5"):
            model_kwargs["reasoning_effort"] = "low"

        return AzureChatOpenAI(**model_kwargs)

    @classmethod
    @lru_cache(maxsize=2)
    def get_embedding_model(
        cls,
        model_name: str = const.DEFAULT_EMBEDDING_MODEL_NAME,
    ) -> AzureOpenAIEmbeddings | CacheBackedEmbeddings:
        """
        Get a cached embedding model, with optional file-based caching
        for the embeddings themselves.
        """
        # Handle the globally injected model first.
        if (
            model_name == const.DEFAULT_EMBEDDING_MODEL_NAME
            and embeddings_model is not None
        ):
            embedding_model = embeddings_model
        elif model_constructor := cls._INNER_EMBEDDING_MODELS.get(model_name):
            embedding_model = model_constructor(
                deployment=model_name, timeout=cls._timeout
            )
        else:
            raise ValueError(f"Unsupported embedding model: {model_name}")

        if EnvConsts.IS_EMBED_CACHE:
            store = LocalFileStore(EnvConsts.EMBEDDING_CACHE_DIR_PATH)
            return CacheBackedEmbeddings.from_bytes_store(
                embedding_model,
                store,
                namespace=model_name,
                query_embedding_cache=True,
            )
        return embedding_model


class LLMUtils:
    """
    A collection of stateless utility functions for working with LLMs.
    """

    _MODEL_ENCODING_ALIASES = {
        "gpt-4.1-2025-04-14": "gpt-4.1-2025-04-14",
    }

    IMG_BASE_TOKENS = 85
    IMG_TOKENS_PER_TILE = 170
    IMG_TILE_SIZE = 512
    IMG_REDUCTION_THRESHOLD = 768
    IMG_MAX_DIMENSION = 2048

    @classmethod
    def get_encoding_for_model(cls) -> Encoding:
        """
        Get the tiktoken encoding for the default chat model.
        Handles custom model name aliases.
        """
        encoding_name = cls._MODEL_ENCODING_ALIASES.get(
            const.DEFAULT_CHAT_MODEL_NAME, const.DEFAULT_CHAT_MODEL_NAME
        )
        return tiktoken.encoding_for_model(encoding_name)

    @classmethod
    def calculate_img_tokens(cls, width: int, height: int) -> int:
        """
        Calculate the number of tokens for an image based on its
        dimensions, following the pricing model for high-detail images.
        """
        # 1. Scale down if the image exceeds the max dimension.
        if width > cls.IMG_MAX_DIMENSION or height > cls.IMG_MAX_DIMENSION:
            aspect_ratio = width / height
            if aspect_ratio > 1:
                width = cls.IMG_MAX_DIMENSION
                height = int(cls.IMG_MAX_DIMENSION / aspect_ratio)
            else:
                width = int(cls.IMG_MAX_DIMENSION * aspect_ratio)
                height = cls.IMG_MAX_DIMENSION

        # 2. Scale down to fit within a 768px square if needed.
        if (
            width > cls.IMG_REDUCTION_THRESHOLD
            and height > cls.IMG_REDUCTION_THRESHOLD
        ):
            if width > height:
                scale_factor = cls.IMG_REDUCTION_THRESHOLD / height
                height = cls.IMG_REDUCTION_THRESHOLD
                width = int(width * scale_factor)
            else:
                scale_factor = cls.IMG_REDUCTION_THRESHOLD / width
                width = cls.IMG_REDUCTION_THRESHOLD
                height = int(height * scale_factor)

        # 3. Calculate tiles and total tokens.
        tiles = ceil(width / cls.IMG_TILE_SIZE) * ceil(
            height / cls.IMG_TILE_SIZE
        )
        return cls.IMG_BASE_TOKENS + (cls.IMG_TOKENS_PER_TILE * tiles)


class Embedder:
    """
    Provides a simple, high-level interface for embedding text.
    """

    @classmethod
    def embed(cls, inputs: str | list[str]) -> list[float] | list[list[float]]:
        """
        Embed text inputs using the configured embedding model.

        Args:
            inputs: A string or a list of strings to embed.

        Returns:
            An embedded representation (or a list of them).

        Raises:
            TypeError: If the input is not a string or a list of
                strings.
            ValueError: If the embedding operation fails.
        """
        model = ModelCreator.get_embedding_model()

        try:
            if isinstance(inputs, str):
                return model.embed_query(inputs)
            if isinstance(inputs, list):
                if not inputs:
                    return []
                if not all(isinstance(item, str) for item in inputs):
                    raise TypeError("All items in the list must be strings.")
                return model.embed_documents(inputs)

            # noinspection PyUnreachableCode
            raise TypeError("Input must be a string or a list of strings.")
        except Exception as e:
            raise ValueError(f"Embedding failed: {e}") from e
