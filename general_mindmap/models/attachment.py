from langchain.schema import Document

from general_mindmap.models.graph import Graph

GRAPH_CONTENT_TYPE = "application/vnd.dial.mindmap.graph.v1+json"


def doc_to_attach(document: Document, index=None) -> dict:
    if index is None:
        index = document.metadata.get("chunk_id", 0)

    return dict(
        type="text/markdown",
        title="[{index}] '{doc_title}'".format(
            **document.metadata, index=index
        ),
        data=f"{document.page_content}",
        # reference_url=document.metadata["doc_url"],
    )


def graph_to_attach(graph: Graph) -> dict:
    return dict(
        type=GRAPH_CONTENT_TYPE,
        title="Generated graph node",
        data=graph.model_dump_json(indent=2, exclude_unset=True),
    )
