import base64
from io import BytesIO
from typing import Iterator

# noinspection PyPackageRequirements
import fitz
from langchain.document_loaders.base import BaseLoader
from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_core.documents.base import Blob, Document
from PIL import Image


class BytesPDFLoader(BaseLoader):
    def __init__(self, pdf_bytes: bytes):
        self.pdf_bytes = pdf_bytes
        self.parser = PyPDFParser()

    def lazy_load(self) -> Iterator[Document]:
        blob = Blob.from_data(self.pdf_bytes)
        yield from self.parser.lazy_parse(blob)


async def get_text_pages(pdf_bytes: bytes) -> list[Document]:
    try:
        loader = BytesPDFLoader(pdf_bytes)
        pages = []
        async for page in loader.alazy_load():
            pages.append(page)
        return pages
    except AttributeError as e:
        print(f"Error processing PDF: {e}")
        return []


def _page_to_base64(page: "fitz.Page") -> str:
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    buffer = BytesIO()
    img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode()


def get_pages_as_base64(pdf_bytes: bytes) -> list[str]:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf_document:
        return [_page_to_base64(page) for page in pdf_document]
