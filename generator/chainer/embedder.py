from functools import lru_cache

from langchain.embeddings import CacheBackedEmbeddings
from langchain_openai import AzureOpenAIEmbeddings

from .model_handler import ModelCreator


@lru_cache(maxsize=1)
def get_embedding_model() -> AzureOpenAIEmbeddings | CacheBackedEmbeddings:
    """Cache the embedding model to avoid recreating it on each call"""
    return ModelCreator.get_embedding_model()


def embed(inputs: str | list[str]) -> list[float] | list[list[float]]:
    """
    Embed text inputs using the embedding model.

    Args:
        inputs: A string or list of strings to embed

    Returns:
        Embedded representation as lists

    Raises:
        TypeError: If input is not a string or list of strings
        ValueError: If embedding operation fails
    """
    model = get_embedding_model()

    try:
        if isinstance(inputs, list):
            if not all(isinstance(item, str) for item in inputs):
                raise TypeError("All items in the list must be strings")
            if not inputs:
                return []
            return model.embed_documents(inputs)
        elif isinstance(inputs, str):
            return model.embed_query(inputs)
        else:
            raise TypeError("Input must be a string or a list of strings")
    except Exception as e:
        raise ValueError(f"Embedding failed: {str(e)}") from e
