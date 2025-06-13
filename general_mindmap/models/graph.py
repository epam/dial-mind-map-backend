from typing import List, Union

from pydantic import BaseModel, RootModel


class Node(BaseModel):
    id: str
    label: str
    details: str | None = None  # TODO: can't be none
    question: str | None = None
    link: str | None = None
    icon: str | None = None
    status: str | None = None
    neon: bool = False


class Edge(BaseModel):
    id: str
    source: str
    target: str
    type: str | None = "Manual"
    weight: str | None = None

    @staticmethod
    def make_edge(source: str, target: str) -> "Edge":
        return Edge(
            id=f"E{source}_{target}", source=source, target=target, type="Init"
        )


class Position(BaseModel):
    x: float | None = None
    y: float | None = None


class GraphData(BaseModel):
    data: Union[Node, Edge]
    position: Position | None = None


class Graph(RootModel):
    root: List[GraphData]

    def get_nodes_iter(self):
        return (d.data for d in self.root if isinstance(d.data, Node))

    def get_nodes(self) -> List[Node]:
        return list(self.get_nodes_iter())

    def get_edges(self) -> List[Edge]:
        return [d.data for d in self.root if isinstance(d.data, Edge)]

    def get_node_by_id(self, node_id: str) -> Node:
        for node in self.get_nodes_iter():
            if node.id == node_id:
                return node
        raise ValueError(f"No node found with id {node_id}")

    @staticmethod
    def make_graph(nodes: List[Node], edges: List[Edge]) -> "Graph":
        return Graph(
            root=[GraphData(data=node) for node in nodes]
            + [GraphData(data=edge) for edge in edges]
        )
