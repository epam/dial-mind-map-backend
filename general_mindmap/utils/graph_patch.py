import base64

from langchain.schema.document import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import chain

from dial_rag.embeddings.embeddings import BGE_EMBEDDINGS_MODEL_NAME_OR_PATH
from general_mindmap.models.attachment import graph_to_attach
from general_mindmap.models.graph import Edge, Graph, Node

INSTRUCTION = "Represent this passage for clusterization: "


embeddings_model = HuggingFaceBgeEmbeddings(
    model_name=BGE_EMBEDDINGS_MODEL_NAME_OR_PATH,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
    query_instruction=INSTRUCTION,
    embed_instruction=INSTRUCTION,
    show_progress=True,
)


def decode_docstore(encoded: str) -> FAISS:
    return FAISS.deserialize_from_bytes(
        base64.b64decode(encoded),
        embeddings_model,
        allow_dangerous_deserialization=True,
    )


def node_to_document(node: Node) -> Document:
    return Document(
        page_content=f"{node.label}\n{node.details}",
        metadata={
            "id": node.id,
            "title": node.label,
            "question": node.question,
            "source": node.link,
        },
    )


class GraphPatcher:
    def __init__(self, graph: Graph, docstore: FAISS | None) -> None:
        self.graph = graph
        self.docstore = docstore

    def make_graph_patch(
        self, label: str, details: str, node_id: str | None = None
    ) -> Graph:
        if not node_id:
            node_id = f"GEN{hash(label)}"

        new_node = Node(
            id=node_id,
            label=label,
            details=details,
        )

        new_node_text = node_to_document(new_node).page_content
        closest_docs = (
            self.docstore.similarity_search(new_node_text, k=4)
            if self.docstore
            else []
        )

        closest_nodes = [
            self.graph.get_node_by_id(doc.metadata["id"])
            for doc in closest_docs
            if doc.metadata["id"] != new_node.id
        ]

        if not closest_nodes:
            new_edges = []
        elif len(closest_nodes) == 1:
            new_edges = [Edge.make_edge(new_node.id, closest_nodes[0].id)]
        else:
            in_edges = [Edge.make_edge(closest_nodes.pop().id, new_node.id)]
            out_edges = [
                Edge.make_edge(new_node.id, node.id) for node in closest_nodes
            ]
            new_edges = out_edges + in_edges

        sub_graph = Graph.make_graph([new_node], new_edges)
        return sub_graph

    def create_chain(self):
        @chain
        def make_graph_attachment(input: dict) -> dict:
            sub_graph = self.make_graph_patch(
                input["label"], input["answer"], input["node_id"]
            )
            return graph_to_attach(sub_graph)

        return make_graph_attachment
