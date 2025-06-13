import os
from math import ceil

import httpx
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.callbacks import OpenAICallbackHandler
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

import generator.utils.code_analysis as ca
import generator.utils.constants as const
from general_mindmap.utils.graph_patch import embeddings_model
from generator.utils.constants import DocCategories as DocCat
from generator.utils.constants import Pi
from generator.utils.misc import env_to_bool


class ModelCreator:
    timeout = httpx.Timeout(150, connect=5.0)
    rate_limiter = InMemoryRateLimiter()

    @classmethod
    def get_chat_model(
        cls,
        model_name: str = const.DEFAULT_CHAT_MODEL_NAME,
    ) -> AzureChatOpenAI:
        seed_str = os.getenv(const.CHAT_MODEL_SEED)
        seed = int(seed_str) if seed_str is not None else None

        return AzureChatOpenAI(
            azure_deployment=model_name,
            model=model_name,
            temperature=0.0,
            seed=seed,
            timeout=cls.timeout,
            rate_limiter=cls.rate_limiter,
        )

    @classmethod
    def get_embedding_model(
        cls,
        model_name: str = const.DEFAULT_EMBEDDING_MODEL_NAME,
    ) -> AzureOpenAIEmbeddings | CacheBackedEmbeddings:
        """
        Get an embedding model based on the specified model name
        with optional caching.

        Args:
            model_name: Name of the embedding model to use.
                Defaults to the constant DEFAULT_EMBEDDING_MODEL_NAME.

        Returns:
            The requested embedding model,
            potentially wrapped with a cache
            if enabled by environment variable.

        Raises:
            ValueError: If an unsupported model name is provided.
        """
        if model_name == "BAAI/bge-small-en-v1.5":
            # Model used for graph patching is used for consistency
            embedding_model = embeddings_model
        elif model_name == "text-embedding-3-large":
            embedding_model = AzureOpenAIEmbeddings(
                deployment=model_name, timeout=cls.timeout
            )
        else:
            raise ValueError(f"Unsupported embedding model: {model_name}")
        if env_to_bool(const.IS_EMBED_CACHE):
            store = LocalFileStore(const.EMBEDDING_CACHE_DIR_PATH)
            return CacheBackedEmbeddings.from_bytes_store(
                embedding_model,
                store,
                namespace=model_name,
                query_embedding_cache=True,
            )
        return embedding_model


class LLMCostHandler:
    __repr__ = OpenAICallbackHandler.__repr__
    cost_types = ca.get_attr_names_from_callable(__repr__)

    def __init__(self):
        for cost_type in self.cost_types:
            setattr(self, cost_type, 0)

    def update_costs(self, cb_handler: OpenAICallbackHandler) -> None:
        for cost_type in self.cost_types:
            setattr(
                self,
                cost_type,
                (getattr(self, cost_type) + getattr(cb_handler, cost_type)),
            )


class ModelUtils:
    @staticmethod
    def calculate_img_tokens(
        width: int,
        height: int,
        max_dimension: int = 2048,
        tile_size: int = 512,
        base_tokens: int = 85,
        tokens_per_tile: int = 170,
        reduction_threshold: int = 768,
    ) -> int:
        """
        Calculate the number of tokens required for an image
        based on its dimensions.

        Args:
            width: The width of the image in pixels.
            height: The height of the image in pixels.
            max_dimension: Maximum dimension to scale down to
                if necessary.
            tile_size: Size of each tile for token calculation.
            base_tokens: Base number of tokens added to the total.
            tokens_per_tile: Number of tokens added per tile.
            reduction_threshold: Dimension threshold
                above which the image is scaled down proportionally.

        Returns:
            The calculated number of tokens.
        """
        # Constrain the maximum dimensions of the image
        if width > max_dimension or height > max_dimension:
            aspect_ratio = width / height
            if aspect_ratio > 1:
                width, height = max_dimension, int(max_dimension / aspect_ratio)
            else:
                width, height = int(max_dimension * aspect_ratio), max_dimension

        # Reduce dimensions proportionally
        # if either dimension exceeds the reduction threshold
        if width >= height > reduction_threshold:
            width, height = (
                int((reduction_threshold / height) * width),
                reduction_threshold,
            )
        elif height > width > reduction_threshold:
            width, height = reduction_threshold, int(
                (reduction_threshold / width) * height
            )

        tiles = ceil(width / tile_size) * ceil(height / tile_size)
        return base_tokens + tokens_per_tile * tiles

    @staticmethod
    def form_text_inputs(text_contents) -> list:
        return [{Pi.TEXTUAL_EXTRACTION: content} for content in text_contents]

    @staticmethod
    def form_multimodal_inputs(file_contents: list[dict], cat: str) -> list:
        if cat == DocCat.PPTX:
            page_pref = "Slide"
        else:
            page_pref = "Page"
        multimodal_inputs = []
        for file_content in file_contents:
            multimodal_content = []
            for slide in file_content:
                slide_text = "\n".join(slide.get("texts"))
                slide_text_part = {
                    "type": "text",
                    "text": f"{page_pref} {slide.get('page_id')}: {slide_text}",
                }
                multimodal_content.append(slide_text_part)
                slide_img_part = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        },
                    }
                    for image_data in slide.get("images")
                ]
                multimodal_content.extend(slide_img_part)
            multimodal_inputs.append(
                {Pi.MULTIMODAL_CONTENT: multimodal_content}
            )
        return multimodal_inputs
