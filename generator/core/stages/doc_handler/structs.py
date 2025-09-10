from dataclasses import asdict, dataclass, field, fields

from langchain_core.documents import Document as LCDoc

from generator.common.constants import DataFrameCols as Col
from generator.common.structs import Document
from generator.core.stages.doc_handler.constants import HEADER_TO_SPLIT_ON


@dataclass
class DocAndContent:
    doc: Document
    content: str | bytes


@dataclass
class PageContent:
    page_id: int
    texts: list[str]
    images: list[str] = field(default_factory=list)
    tokens: int = 0


@dataclass
class Chunk:
    """
    Attributes:
        lc_doc: Chunk content in a form of a LangChain document.
        doc_id: Chunk source document id.
        doc_url: Chunk source document url.
        doc_title: Chunk source document title.
        doc_cat: Chunk source document category.
        doc_desc: Document description provided by the user.
        page_contents: List of chunk page contents.
        page_ids: List of page ids in the chunk if applicable.
    """

    lc_doc: LCDoc
    doc_id: str
    doc_url: str
    doc_title: str
    doc_cat: str
    doc_desc: str
    page_contents: list[PageContent] = field(
        default_factory=list, metadata={"pop_in_as_dict": True}
    )
    page_ids: list[int] = field(default_factory=list)

    @property
    def _content(self) -> str | list[dict]:
        if self.page_contents:
            # noinspection PyTypeChecker
            return [asdict(content) for content in self.page_contents]
        header = self.lc_doc.metadata.get(HEADER_TO_SPLIT_ON)
        if header is not None:
            return "# " + header + "\n" + self.lc_doc.page_content
        return self.lc_doc.page_content

    def as_dict(self) -> dict[str, int | LCDoc | str]:
        # noinspection PyTypeChecker
        data = asdict(self)
        # noinspection PyTypeChecker
        for f in fields(self):
            if f.metadata.get("pop_in_as_dict", False):
                data.pop(f.name)
        data[Col.CONTENT] = self._content
        return data
