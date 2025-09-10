from typing import Type

from .chunkers.base_chunker import BaseDocChunker

HANDLER_REGISTRY: dict[str, BaseDocChunker] = {}


def register_handler(handler_class: Type[BaseDocChunker]):
    """
    Registers a handler class for the document categories it supports.
    This is intended to be used as a class decorator.
    """
    handler_instance = handler_class()
    for category in handler_instance.supported_categories:
        if category in HANDLER_REGISTRY:
            raise ValueError(f"Duplicate handler for category '{category}'")
        HANDLER_REGISTRY[category] = handler_instance

    return handler_class


def get_handler(doc_cat: str) -> BaseDocChunker | None:
    """Retrieves a handler instance for a given document category."""
    return HANDLER_REGISTRY.get(doc_cat)
