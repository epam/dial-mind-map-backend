from queue import Queue
from typing import List, Optional, Tuple

from general_mindmap.models.graph import Edge, Graph, GraphData, Node
from general_mindmap.v2.models.edge_type import EdgeType

MAX_FIRST_DEPTH_CONNECTIONS = 6


def is_added(queue: Queue, id: str):
    temp = []

    while not queue.empty():
        item = queue.get()
        temp.append(item)

    found = False
    for item in temp:
        if item["id"] == id:
            found = True
        queue.put(item)

    return found


def get_subgraph(
    graph: Graph,
    root: str,
    maxDepth: int,
    maxNodes: int,
    previous: Optional[str],
) -> Tuple[List[GraphData], List[GraphData]]:
    nodes = [item for item in graph.root if isinstance(item.data, Node)]
    edges = [item for item in graph.root if isinstance(item.data, Edge)]

    nodeMap = dict((node.data.id, node) for node in nodes)

    adjacencyListByType = dict()
    reverseAdjacencyList = dict()
    for edge in edges:
        assert isinstance(edge.data, Edge)

        source = edge.data.source
        target = edge.data.target
        type = edge.data.type

        if source not in adjacencyListByType:
            adjacencyListByType[source] = {}
        if type not in adjacencyListByType[source]:
            adjacencyListByType[source][type] = []
        adjacencyListByType[source][type].append(target)

        if target not in reverseAdjacencyList:
            reverseAdjacencyList[target] = []
        reverseAdjacencyList[target].append(source)

    adjacencyList = dict()
    for source, edgesByType in adjacencyListByType.items():
        adjacencyList[source] = (
            edgesByType.get(EdgeType.MANUAL, [])
            + edgesByType.get(EdgeType.INIT, [])
            + edgesByType.get(EdgeType.GENERATED, [])
        )

    currentLevelQueue = Queue()
    nextLevelQueue = Queue()
    visitedNodes = set()
    resultNodes: List[GraphData] = []
    resultEdges: List[GraphData] = []
    nodeDepths = dict()

    requiredEdges = []

    nextNeigborIndex = dict()

    visitedNodes.add(root)
    resultNodes.append(nodeMap[root])
    nodeDepths[root] = 0

    first_depth_connections = adjacencyList.get(root, [])
    for target in first_depth_connections[
        0 : min(MAX_FIRST_DEPTH_CONNECTIONS, len(first_depth_connections))
    ]:
        if target not in visitedNodes:
            currentLevelQueue.put({"id": target, "depth": 1})
            requiredEdges.append({"source": root, "target": target})

    foundPathToPreviousNode = []
    if previous:
        reverseQueue = Queue()
        reverseVisitedNodes = set()

        reverseQueue.put({"id": root, "path": [root], "depth": 0})
        reverseVisitedNodes.add(root)

        while not reverseQueue.empty():
            top = reverseQueue.get()
            id, path, depth = top["id"], top["path"], top["depth"]

            if depth > maxDepth:
                continue

            if previous in path:
                foundPathToPreviousNode = path

            if id in reverseAdjacencyList:
                for source in reverseAdjacencyList[id]:
                    if source not in reverseVisitedNodes:
                        reverseVisitedNodes.add(source)
                        reverseQueue.put(
                            {
                                "id": source,
                                "path": path + [source],
                                "depth": depth + 1,
                            }
                        )

            if id in adjacencyList:
                for target in adjacencyList[id]:
                    if target not in reverseVisitedNodes:
                        reverseVisitedNodes.add(target)
                        reverseQueue.put(
                            {
                                "id": target,
                                "path": path + [target],
                                "depth": depth + 1,
                            }
                        )

    if foundPathToPreviousNode:
        for index, nodeId in enumerate(foundPathToPreviousNode):
            if nodeId in nodeMap and not any(
                node.data.id == nodeId for node in resultNodes
            ):
                node = nodeMap[nodeId]

                resultNodes.append(node)
                visitedNodes.add(nodeId)

                if nodeId not in nodeDepths:
                    nodeDepths[nodeId] = index

            if index > 0:
                source = foundPathToPreviousNode[index - 1]
                target = nodeId

                foundEdges = [
                    edge
                    for edge in edges
                    if edge.data.source == target and edge.data.target == source
                ]
                if not foundEdges:
                    foundEdges = [
                        edge
                        for edge in edges
                        if edge.data.source == source
                        and edge.data.target == target
                    ]
                if len(foundEdges) == 1:
                    resultEdges.append(foundEdges[0])

    while not currentLevelQueue.empty() and len(resultNodes) < maxNodes:
        while not currentLevelQueue.empty():
            top = currentLevelQueue.get()
            id = top["id"]
            depth = top["depth"]

            if depth > maxDepth:
                continue

            if id not in visitedNodes:
                visitedNodes.add(id)
                if id in nodeMap and id not in resultNodes:
                    resultNodes.append(nodeMap[id])
                nodeDepths[id] = depth

                if len(resultNodes) >= maxNodes:
                    break

            if id not in nextNeigborIndex:
                nextNeigborIndex[id] = 0

            neighborIndex = nextNeigborIndex[id]

            if id in adjacencyList and neighborIndex < len(adjacencyList[id]):
                target = adjacencyList[id][neighborIndex]
                if (
                    target not in visitedNodes
                    and not is_added(currentLevelQueue, target)
                    and not is_added(nextLevelQueue, target)
                ):
                    nextLevelQueue.put({"id": target, "depth": depth + 1})
                    requiredEdges.append({"source": id, "target": target})

                nextNeigborIndex[id] = neighborIndex + 1

            if id not in adjacencyList or nextNeigborIndex[id] >= len(
                adjacencyList[id]
            ):
                del nextNeigborIndex[id]
            else:
                currentLevelQueue.put({"id": id, "depth": depth})

        while not nextLevelQueue.empty():
            currentLevelQueue.put(nextLevelQueue.get())

    if foundPathToPreviousNode:
        for nodeDepth, nodeId in enumerate(foundPathToPreviousNode):
            if nodeDepth >= maxDepth or nodeId == root:
                continue

            queue = Queue()
            queue.put({"id": nodeId, "depth": nodeDepth})

            while not queue.empty():
                top = queue.get()
                id, depth = top["id"], top["depth"]

                if (
                    depth >= maxDepth
                    or nodeId == root
                    or len(resultNodes) >= maxNodes
                ):
                    continue

                if id in adjacencyList:
                    for target in adjacencyList[id]:
                        if (
                            len(resultNodes) < maxNodes
                            and target not in visitedNodes
                        ):
                            visitedNodes.add(target)

                            if target in nodeMap and not any(
                                node.data.id == target for node in resultNodes
                            ):
                                resultNodes.append(nodeMap[target])

                                if target not in nodeDepths:
                                    nodeDepths[target] = depth + 1

                                relatedEdges = [
                                    edge
                                    for edge in edges
                                    if edge.data.source == id
                                    and edge.data.target == target
                                ]

                                if len(relatedEdges) == 1:
                                    resultEdges.append(relatedEdges[0])
                                queue.put({"id": target, "depth": depth + 1})

    for edge in resultEdges:
        requiredEdges.append(
            {"source": edge.data.source, "target": edge.data.target}
        )
    resultEdges = []

    for edge in edges:
        sourceDepth = nodeDepths.get(edge.data.source, -1)
        targetDepth = nodeDepths.get(edge.data.target, -1)

        if (
            sourceDepth != -1
            and targetDepth != -1
            and sourceDepth <= maxDepth
            and targetDepth <= maxDepth
            and sourceDepth != targetDepth
            and any(
                edge.data.source == requiredEdge["source"]
                and edge.data.target == requiredEdge["target"]
                for requiredEdge in requiredEdges
            )
            and not any(
                (
                    addedEdge.data.source == edge.data.source
                    and addedEdge.data.target == edge.data.target
                )
                or (
                    addedEdge.data.source == edge.data.target
                    and addedEdge.data.target == edge.data.source
                )
                for addedEdge in resultEdges
            )
        ):
            resultEdges.append(edge)

    return (resultNodes, resultEdges)
