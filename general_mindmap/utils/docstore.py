import base64
from typing import Any, List, Sequence

from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dial_rag.embeddings.embeddings import BGE_EMBEDDINGS_MODEL_NAME_OR_PATH
from general_mindmap.models.graph import Node

embeddings_model = HuggingFaceBgeEmbeddings(
    model_name=BGE_EMBEDDINGS_MODEL_NAME_OR_PATH,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
    show_progress=True,
)


def calculate_cache(documents: Sequence[Document]) -> str:
    ids = [doc.metadata["id"] for doc in documents]
    docstore = FAISS.from_documents(list(documents), embeddings_model, ids=ids)
    return encode_docstore(docstore)


def encode_docstore(docstore: FAISS) -> str:
    return base64.b64encode(docstore.serialize_to_bytes()).decode("utf-8")


def node_to_card_doc(node: Node) -> Document:
    return Document(
        page_content=f"{node.details}",
        metadata={
            "id": node.id,
            "title": node.label,
            "question": node.question,
            "questions": node.questions,
            "source": node.link,
        },
    )


def decode_docstore(encoded: str) -> FAISS:
    return FAISS.deserialize_from_bytes(
        base64.b64decode(encoded),
        embeddings_model,
        allow_dangerous_deserialization=True,
    )


def node_to_documents(node: Node) -> List[Document]:
    card_doc = node_to_card_doc(node)

    search_chunks = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=200
    ).split_documents([card_doc])

    search_chunks.append(
        Document(
            page_content=card_doc.metadata["title"], metadata=card_doc.metadata
        )
    )
    search_chunks.append(
        Document(
            page_content=card_doc.metadata["question"],
            metadata=card_doc.metadata,
        )
    )

    return search_chunks
