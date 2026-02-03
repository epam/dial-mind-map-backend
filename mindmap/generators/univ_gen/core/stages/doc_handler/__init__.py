# Import all handler modules. This is critical because the act of
# importing them will execute the @register_handler decorator and
# populate our HANDLER_REGISTRY.
from . import chunkers
from .chunkers import html_chunker, pdf_chunker, pptx_chunker, txt_chunker
from .doc_handler import DocHandler

__all__ = ["DocHandler"]
