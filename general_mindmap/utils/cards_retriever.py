from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

from dial_rag.embeddings.embeddings import BGE_EMBEDDINGS_MODEL_NAME_OR_PATH
from general_mindmap.models.graph import Graph
from general_mindmap.utils.docstore import node_to_card_doc

embeddings_model = HuggingFaceBgeEmbeddings(
    model_name=BGE_EMBEDDINGS_MODEL_NAME_OR_PATH,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
    show_progress=True,
)


def create_cards_retriever(graph_data: Graph, vectorstore: FAISS | None):
    if vectorstore is None:
        return None

    card_docs = [node_to_card_doc(node) for node in graph_data.get_nodes()]

    docstore = InMemoryStore()
    docstore.mset([(d.metadata["id"], d) for d in card_docs])

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key="id",
        search_kwargs={"k": 5},
    )

    return retriever
