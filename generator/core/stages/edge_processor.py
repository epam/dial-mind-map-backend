import asyncio as aio
from itertools import product
from typing import Callable, Optional

import faiss
import networkx as nx
import numpy as np
import pandas as pd
from langchain_community.vectorstores import FAISS
from sklearn.metrics.pairwise import cosine_similarity

from common_utils.logger_config import logger
from generator.common.constants import ColVals
from generator.common.constants import DataFrameCols as Col
from generator.common.context import cur_llm_cost_handler
from generator.core.utils.constants import FrontEndStatuses as Fes
from generator.core.utils.frontend_handler import put_status


class EdgeProcessor:
    """
    Transforms a set of relations into a high-quality, navigable graph.

    This class takes the concepts and relations from the orchestrator
    and enhances the graph's structure. Its primary goal is to ensure
    the final graph is **strongly connected**, meaning it's possible to
    navigate from any concept to any other. It achieves this through a
    three-stage process: adding edges for density, guaranteeing strong
    connectivity, and then pruning redundant edges for clarity.
    """

    def __init__(self, queue: aio.Queue):
        """Initializes the processor with a queue for status updates."""
        self.queue = queue

    @staticmethod
    def _get_similarity_df(
        concept_df: pd.DataFrame, cos: bool = False
    ) -> pd.DataFrame:
        """
        Builds a similarity matrix from concept embeddings.

        This matrix serves as a lookup table to find the most similar
        concepts when new edges need to be added to the graph. It can
        use either FAISS-based L2 (Euclidean) distance or standard
        cosine similarity.

        Args:
            concept_df: The DataFrame of concepts with their embeddings.
            cos: If True, use cosine similarity. If False (default),
                use inverted Euclidean distance.

        Returns:
            A DataFrame where both index and columns are concept IDs,
            and values are their similarity scores.
        """
        embeddings = np.stack(concept_df[Col.EMBEDDING])
        if not cos:
            embeddings = embeddings.astype(np.float32)
            n = len(embeddings)
            dimension = embeddings.shape[1]

            index = faiss.IndexFlatL2(dimension)
            # noinspection PyArgumentList
            index.add(embeddings)

            distance_matrix = np.zeros((n, n), dtype=np.float32)

            for i in range(n):
                # noinspection PyArgumentList
                distances_i, indices_i = index.search(embeddings[i : i + 1], n)

                for j, idx in enumerate(indices_i[0]):
                    distance_matrix[i, idx] = distances_i[0, j]

            max_distance = np.max(distance_matrix)
            if max_distance > 0:
                similarity_matrix = 1 - (distance_matrix / max_distance)
            else:
                similarity_matrix = np.ones((n, n))

        else:
            similarity_matrix = cosine_similarity(embeddings)
        return pd.DataFrame(
            similarity_matrix,
            index=concept_df.index,
            columns=concept_df.index,
        )

    @staticmethod
    def _adj_root_out_edges(
        edge_df: pd.DataFrame,
        root_id: int,
        similarity_df: pd.DataFrame,
        max_num_root_out_edges: int = 4,
        root_edge_weight: float = 5.0,
        rejected_edge_weight: float = 0.1,
    ) -> pd.DataFrame:
        """
        Adjusts edges from the root to ensure good graph entry points.

        This method ensures the main root concept has a fixed number of
        high-quality outgoing edges.
        - If there are too many, it prunes the weakest ones,
          prioritizing connections to other "hub" nodes.
        - If there are too few, it adds new ones, first by connecting
          to other hubs, then by finding the most similar concepts.

        Args:
            edge_df: The current DataFrame of edges.
            root_id: The ID of the main root concept.
            similarity_df: The concept similarity matrix for lookups.
            max_num_root_out_edges: The target number of outgoing edges.
            root_edge_weight: The high weight to assign to root edges.
            rejected_edge_weight: The low weight for pruned root edges.

        Returns:
            The edge DataFrame with adjusted root edges.
        """
        origins = edge_df[Col.ORIGIN_CONCEPT_ID]
        root_out_mask = (origins == root_id) & (
            edge_df[Col.TARGET_CONCEPT_ID] != root_id
        )
        root_out_edges: pd.DataFrame = edge_df.loc[root_out_mask]
        num_root_out_edges = len(root_out_edges)

        num_needed = max_num_root_out_edges - num_root_out_edges
        if num_needed < 0:
            target_out_col = "target_outgoing"

            high_weight_mask = root_out_edges[Col.WEIGHT] > 3.5
            high_weight_edges: pd.DataFrame
            high_weight_edges = root_out_edges.loc[high_weight_mask]
            num_high_weight_edges = len(high_weight_edges)

            num_high_weight_edges_needed = (
                max_num_root_out_edges - num_high_weight_edges
            )
            if num_high_weight_edges_needed < 0:
                high_weight_edges_copy = high_weight_edges.copy()
                high_weight_targets = high_weight_edges[Col.TARGET_CONCEPT_ID]

                bi_dir_mask = origins.isin(high_weight_targets.unique())
                bi_dir_edges = edge_df[bi_dir_mask]

                origin_groups = bi_dir_edges.groupby(Col.ORIGIN_CONCEPT_ID)
                target_counts = origin_groups.size().to_dict()

                high_weight_edges_copy[target_out_col] = (
                    high_weight_targets.map(lambda x: target_counts.get(x, 0))
                )

                sorted_edges = high_weight_edges_copy.sort_values(
                    by=[target_out_col, Col.TARGET_CONCEPT_ID],
                    ascending=[False, False],
                )
                selected_edges = sorted_edges.head(4)
            elif num_high_weight_edges_needed > 0:
                low_weight_edges = root_out_edges[~high_weight_mask].copy()
                low_weight_targets = low_weight_edges[Col.TARGET_CONCEPT_ID]

                bi_dir_mask = origins.isin(low_weight_targets.unique())
                bi_dir_edges = edge_df[bi_dir_mask]

                origin_groups = bi_dir_edges.groupby(Col.ORIGIN_CONCEPT_ID)
                target_counts = origin_groups.size().to_dict()

                low_weight_edges[target_out_col] = low_weight_targets.map(
                    lambda x: target_counts.get(x, 0)
                )

                sorted_edges = low_weight_edges.sort_values(
                    by=[target_out_col, Col.TARGET_CONCEPT_ID],
                    ascending=[False, False],
                )
                selected_low_weight_edges = sorted_edges.head(
                    num_high_weight_edges_needed
                )
                selected_edges = pd.concat(
                    [high_weight_edges, selected_low_weight_edges]
                )
            else:
                selected_edges = high_weight_edges

            selected_indices = selected_edges.index
            edge_df.loc[selected_indices, Col.WEIGHT] = root_edge_weight
            edge_df.loc[selected_indices, Col.TYPE] = ColVals.RELATED

            root_out_indices = root_out_edges.index
            rejected_indices = root_out_indices.difference(selected_indices)
            edge_df.loc[rejected_indices, Col.WEIGHT] = rejected_edge_weight

        elif num_needed > 0:
            root_out_indices = root_out_edges.index
            edge_df.loc[root_out_indices, Col.WEIGHT] = root_edge_weight
            edge_df.loc[root_out_indices, Col.TYPE] = ColVals.RELATED

            existing_targets = (
                root_out_edges[Col.TARGET_CONCEPT_ID].unique().tolist()
            )
            all_targets = edge_df[Col.TARGET_CONCEPT_ID].unique()
            candidate_targets = [
                t
                for t in all_targets
                if t not in existing_targets and t != root_id
            ]

            target_outgoing_counts = (
                edge_df.groupby(Col.ORIGIN_CONCEPT_ID).size().to_dict()
            )
            sorted_candidates = sorted(
                candidate_targets,
                key=lambda x: (-target_outgoing_counts.get(x, 0), x),
            )
            selected_targets = sorted_candidates[:num_needed]
            existing_targets = existing_targets + selected_targets

            num_more_needed = num_needed - len(selected_targets)

            candidate_targets = []
            if num_more_needed > 0:
                root_similarities = similarity_df.loc[root_id]
                sorted_similarities = root_similarities.sort_values(
                    ascending=False
                )
                candidate_targets = [
                    target
                    for target in sorted_similarities.index
                    if target != root_id and target not in existing_targets
                ]
            selected_nodes = (
                selected_targets + candidate_targets[:num_more_needed]
            )

            new_relations = []
            for target in selected_nodes:
                new_row = {
                    Col.ORIGIN_CONCEPT_ID: root_id,
                    Col.TARGET_CONCEPT_ID: target,
                    Col.WEIGHT: root_edge_weight,
                    Col.TYPE: ColVals.RELATED,
                }
                new_relations.append(new_row)

            if new_relations:
                new_edges_df = pd.DataFrame(new_relations)
                edge_df = pd.concat([edge_df, new_edges_df], ignore_index=True)

        else:
            edge_df.loc[root_out_edges.index, Col.WEIGHT] = root_edge_weight
            edge_df.loc[root_out_edges.index, Col.TYPE] = ColVals.RELATED

        return edge_df

    @staticmethod
    def _get_neighbours(
        edge_df: pd.DataFrame, concept_id: int
    ) -> tuple[set[int], set[int]]:
        """
        Gets the incoming and outgoing neighbors for a given concept.
        """
        all_origins = edge_df[Col.ORIGIN_CONCEPT_ID]
        out_relations = edge_df[all_origins == concept_id]
        targets = set(out_relations[Col.TARGET_CONCEPT_ID])

        all_targets = edge_df[Col.TARGET_CONCEPT_ID]
        inc_relations = edge_df[all_targets == concept_id]
        origins = set(inc_relations[Col.ORIGIN_CONCEPT_ID])

        return targets, origins

    @staticmethod
    def _is_digraph_strong_con(edge_df: pd.DataFrame) -> tuple[list[set], bool]:
        """
        Checks if a directed graph is strongly connected.

        A graph is strongly connected if for every pair of nodes (A, B),
        there is a path from A to B and a path from B to A. This is
        checked by finding the number of strongly connected components
        (SCCs); if there is only one, the graph is strongly connected.

        Args:
            edge_df: The DataFrame of edges representing the graph.

        Returns:
            A tuple containing the list of SCCs (as sets of nodes) and
            a boolean indicating if the graph is strongly connected.
        """
        graph = nx.from_pandas_edgelist(
            edge_df,
            source=Col.ORIGIN_CONCEPT_ID,
            target=Col.TARGET_CONCEPT_ID,
            create_using=nx.DiGraph,
        )

        # SCC = Strongly Connected Components
        scc = list(nx.strongly_connected_components(graph))
        scc_num = len(scc)
        if is_strong_con := scc_num == 1:
            logger.info(f"Is the graph strongly connected? {is_strong_con}")
            logger.info(f"Number of Strongly Connected Components: {scc_num}")
            logger.info(f"SCC sizes: {[len(component) for component in scc]}")
        else:
            is_weak_con = is_strong_con or nx.is_weakly_connected(graph)
            logger.info(f"Is the graph weakly connected? {is_weak_con}")

        return scc, is_strong_con

    @staticmethod
    def _get_node_pairs_with_sim(
        pairs: product, similarity_df: pd.DataFrame
    ) -> list:
        """Looks up the similarity score for a list of node pairs."""
        return [
            (node_1, node_2, similarity_df.loc[node_1, node_2])
            for node_1, node_2 in pairs
        ]

    @staticmethod
    def _get_weight_threshold(num_nodes: int) -> float:
        """Gets a heuristic weight threshold for edge pruning."""
        match num_nodes:
            case value if value > 100:
                return 2.882
            case _:
                return 2.971

    @classmethod
    def _add_edges(
        cls,
        edge_df: pd.DataFrame,
        concept_df: pd.DataFrame,
        similarity_df: pd.DataFrame,
        num_min_out: int = 3,
        num_min_inc: int = 1,
    ):
        """
        Ensures all nodes have a minimum number of in/out edges.

        This prevents "orphan" or "dead-end" nodes in the graph. It
        iterates through each concept and, if the concept has too few
        incoming or outgoing edges, it adds new ones by connecting to
        the most similar non-neighboring concepts.

        Args:
            edge_df: The current DataFrame of edges.
            concept_df: The DataFrame of all concepts.
            similarity_df: The concept similarity matrix.
            num_min_out: The minimum required number of outgoing edges.
            num_min_inc: The minimum required number of incoming edges.

        Returns:
            The edge DataFrame with new edges added to meet minimums.
        """
        new_edges = []
        for concept_id in concept_df.index:
            targets, origins = cls._get_neighbours(edge_df, concept_id)

            neighbours = targets | origins
            neighbours.add(concept_id)  # Ensure no self-loops

            num_need_out = max(num_min_out - len(targets), 0)
            num_need_inc = max(num_min_inc - len(origins), 0)

            if (num_need_edges := num_need_out + num_need_inc) > 0:
                candidates: pd.Series = similarity_df.loc[concept_id]
                new_candidates = candidates.drop(list(neighbours))
                top_candidates = new_candidates.nlargest(num_need_edges)
                top_candidate_ids = top_candidates.index

                if num_need_out > 0:
                    for target_id in top_candidate_ids[:num_need_out]:
                        weight = top_candidates[target_id]
                        new_edges.append(
                            {
                                Col.ORIGIN_CONCEPT_ID: concept_id,
                                Col.TARGET_CONCEPT_ID: target_id,
                                Col.WEIGHT: weight,
                            }
                        )

                if num_need_inc > 0:
                    for source_id in top_candidate_ids[-num_need_inc:]:
                        weight = top_candidates[source_id]
                        new_edges.append(
                            {
                                Col.ORIGIN_CONCEPT_ID: source_id,
                                Col.TARGET_CONCEPT_ID: concept_id,
                                Col.WEIGHT: weight,
                            }
                        )

        if new_edges:
            edge_df = pd.concat(
                [edge_df, pd.DataFrame(new_edges)],
                ignore_index=True,
            )

        return edge_df

    @classmethod
    def _ensure_edges(
        cls,
        edge_df: pd.DataFrame,
        concept_df: pd.DataFrame,
        similarity_df: pd.DataFrame,
        root_id: int,
    ):
        """Orchestrates the initial edge addition phase."""
        edge_df = cls._adj_root_out_edges(edge_df, root_id, similarity_df)
        edge_df = cls._add_edges(edge_df, concept_df, similarity_df)

        all_origins = edge_df[Col.ORIGIN_CONCEPT_ID]
        root_out_edge_mask = all_origins == root_id
        not_root_main_edge_weight_mask = edge_df[Col.WEIGHT] != 5
        root_out_other_edge_mask = (
            root_out_edge_mask & not_root_main_edge_weight_mask
        )
        edge_df.loc[root_out_other_edge_mask, Col.WEIGHT] = 0.1

        return edge_df

    @classmethod
    def _extend_edges(
        cls,
        relation_df: pd.DataFrame,
        concept_df: pd.DataFrame,
        similarity_df: pd.DataFrame,
        root_index: int,
    ) -> pd.DataFrame:
        """Extends the graph by adding edges to ensure density."""
        edge_df = relation_df.copy()

        edge_df = cls._ensure_edges(
            edge_df, concept_df, similarity_df, root_index
        )

        # Remove potential duplicates that could have been introduced
        edge_df = edge_df.sort_values(Col.WEIGHT, ascending=False)
        edge_df.drop_duplicates(
            subset=[Col.ORIGIN_CONCEPT_ID, Col.TARGET_CONCEPT_ID],
            keep="first",
            inplace=True,
        )

        return edge_df.fillna({Col.TYPE: ColVals.ARTIFICIAL})

    @classmethod
    async def strong_con_graph(
        cls,
        edge_df: pd.DataFrame,
        get_node_pairs_w_sim_func: Callable,
        similarity_df: Optional[pd.DataFrame],
        docstore: FAISS | None = None,
        node_by_id: dict[int, dict] | None = None,
    ) -> pd.DataFrame:
        """Forces the graph to be strongly connected.

        This method implements the core connectivity guarantee. It finds
        all strongly connected components (SCCs) and, if there is more
        than one, it systematically adds "bridge" edges. For each small
        SCC, it creates a bidirectional link to the largest SCC by
        finding the most similar pair of nodes between them.

        Args:
            edge_df: The current DataFrame of edges.
            get_node_pairs_w_sim_func: A function to get similarity
                scores.
            similarity_df: The similarity matrix.
            docstore: (Optional) A FAISS docstore for similarity
                lookups.
            nodes: (Optional) A list of nodes for the docstore.
            node_by_id: (Optional) A dict mapping node IDs.

        Returns:
            The edge DataFrame with new bridge edges added to ensure it
            represents a single strongly connected component.
        """
        scc, is_strong_con = cls._is_digraph_strong_con(edge_df)
        if is_strong_con:
            return edge_df

        node_cols = [Col.ORIGIN_CONCEPT_ID, Col.TARGET_CONCEPT_ID]
        existing_edges = edge_df[node_cols].itertuples(index=False, name=None)

        largest_scc, *smaller_sccs = sorted(scc, key=len, reverse=True)

        new_edges = []
        for smaller_scc in smaller_sccs:
            pairs = product(largest_scc, smaller_scc)
            if docstore is None:
                pairs_with_sim = get_node_pairs_w_sim_func(pairs, similarity_df)
            else:
                pairs_with_sim = get_node_pairs_w_sim_func(
                    docstore, node_by_id, pairs
                )
            two_top_sorted_pairs = sorted(
                pairs_with_sim,
                key=lambda pair_with_sim: pair_with_sim[2],
                reverse=True,
            )[:2]

            for i, (src, dst, _) in enumerate(two_top_sorted_pairs):
                if i == 1:
                    src, dst = dst, src
                edge = (src, dst)
                if edge not in existing_edges:
                    new_edges.append(
                        {
                            Col.ORIGIN_CONCEPT_ID: src,
                            Col.TARGET_CONCEPT_ID: dst,
                            Col.TYPE: ColVals.ARTIFICIAL,
                            Col.WEIGHT: 0.5,
                        }
                    )

        new_edge_df = pd.DataFrame(new_edges)
        max_idx = edge_df.index.max() if not edge_df.empty else -1
        new_edge_df.index = range(max_idx + 1, max_idx + 1 + len(new_edge_df))
        return pd.concat([edge_df, new_edge_df])

    @classmethod
    async def _connect_graph(
        cls,
        edge_df: pd.DataFrame,
        concept_df: pd.DataFrame,
        similarity_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Orchestrates the graph connection and cleanup process."""
        # Main root edges must not change direction
        edge_df = await cls.strong_con_graph(
            edge_df, cls._get_node_pairs_with_sim, similarity_df
        )

        origins_as_int = edge_df[Col.ORIGIN_CONCEPT_ID].astype(int)
        edge_df[Col.ORIGIN_CONCEPT_ID] = origins_as_int
        targets_as_int = edge_df[Col.TARGET_CONCEPT_ID].astype(int)
        edge_df[Col.TARGET_CONCEPT_ID] = targets_as_int

        origins = edge_df[Col.ORIGIN_CONCEPT_ID]
        targets = edge_df[Col.TARGET_CONCEPT_ID]
        condition = origins == targets
        self_edge_ids = edge_df[condition].index
        edge_df.drop(self_edge_ids, inplace=True)
        edge_df.sort_values(Col.WEIGHT, ascending=False, inplace=True)
        node_cols = [Col.ORIGIN_CONCEPT_ID, Col.TARGET_CONCEPT_ID]
        edge_df.drop_duplicates(subset=node_cols, keep="first", inplace=True)

        concept_ids = concept_df.index
        valid_source_ids = edge_df[Col.ORIGIN_CONCEPT_ID].isin(concept_ids)
        valid_target_ids = edge_df[Col.TARGET_CONCEPT_ID].isin(concept_ids)
        valid_rows = valid_source_ids & valid_target_ids
        return edge_df[valid_rows]

    @classmethod
    def _remove_redundant_edges(
        cls, edge_df: pd.DataFrame, num_nodes: int
    ) -> pd.DataFrame:
        """
        Prunes unnecessary edges while preserving strong connectivity.

        After adding many edges to ensure connectivity, this method
        cleans up the graph. It iterates through the weakest edges and
        removes each one, but only if the removal does not break the
        graph's strongly connected property. This results in a minimal,
        cleaner graph that is still fully navigable.

        Args:
            edge_df: The DataFrame of the dense, connected graph.
            num_nodes: The total number of nodes in the graph.

        Returns:
            A pruned DataFrame representing a minimal strongly
            connected graph.
        """
        weight_threshold = cls._get_weight_threshold(num_nodes)
        weight_mask = edge_df[Col.WEIGHT] < weight_threshold
        edge_df.loc[weight_mask, Col.TYPE] = ColVals.ARTIFICIAL

        di_graph = nx.from_pandas_edgelist(
            edge_df,
            source=Col.ORIGIN_CONCEPT_ID,
            target=Col.TARGET_CONCEPT_ID,
            edge_attr=[Col.TYPE, Col.WEIGHT],
            create_using=nx.DiGraph,
        )

        di_graph_copy = di_graph.copy()

        # Filter edges to include only those with Col.ART_EDGE_TYPE_VAL
        sorted_edges = sorted(
            (
                edge
                for edge in di_graph.edges(data=True)
                if edge[2][Col.WEIGHT] < 5
            ),
            key=lambda edge_w_data: edge_w_data[2][Col.WEIGHT],
        )

        for edge in sorted_edges:
            di_graph_copy.remove_edge(edge[0], edge[1])

            if not nx.is_strongly_connected(di_graph_copy):
                di_graph_copy.add_edge(
                    edge[0],
                    edge[1],
                    type=edge[2][Col.TYPE],
                    weight=edge[2][Col.WEIGHT],
                )
                origin_mask = edge_df[Col.ORIGIN_CONCEPT_ID] == edge[0]
                target_mask = edge_df[Col.TARGET_CONCEPT_ID] == edge[1]
                needed_mask = origin_mask & target_mask
                edge_df.loc[needed_mask, Col.TYPE] = ColVals.RELATED

        return edge_df

    async def _wrap_up(
        self,
        edge_df: pd.DataFrame,
        root_index: int,
    ) -> tuple[pd.DataFrame, int]:
        """Performs final sorting, logging, and returns results."""
        edge_df = edge_df.sort_values(
            by=[Col.TYPE, Col.WEIGHT],
            ascending=[True, False],
            key=lambda col: (
                col
                if col.name != Col.TYPE
                else col.map({ColVals.RELATED: 0, ColVals.ARTIFICIAL: 1})
            ),
        )
        llm_cost_handler = cur_llm_cost_handler.get()
        logger.info(f"=Postprocessing LLM costs:\n{llm_cost_handler}")
        await self.queue.put(None)
        logger.info("Edge Enhancing end")
        return edge_df, root_index

    async def enhance_edges(
        self,
        concept_df: pd.DataFrame,
        relation_df: pd.DataFrame,
        root_index: int,
    ) -> tuple[pd.DataFrame, int]:
        """
        Enhances a set of relations to form a minimal, connected graph.

        This is the main entry point for the class. It orchestrates a
        multi-stage process to ensure the final graph is navigable and
        clean:
        1.  **Extend**: Adds new edges based on concept similarity to
            ensure all nodes have a minimum number of connections.
        2.  **Connect**: Guarantees the graph is strongly connected by
            adding "bridge" edges between any disconnected components.
        3.  **Prune**: Removes redundant "artificial" edges that are not
            essential for maintaining strong connectivity.

        Args:
            concept_df: The final DataFrame of concepts.
            relation_df: The DataFrame of relations to be enhanced.
            root_index: The ID of the main root concept.

        Returns:
            A tuple containing the final, enhanced edge DataFrame and
            the root index.
        """
        if len(concept_df) == 1:
            return relation_df, root_index

        logger.info("Enhancing edges")

        num_nodes = len(concept_df)
        similarity_df = self._get_similarity_df(concept_df)

        await put_status(self.queue, Fes.CREATE_CONNECTIONS)
        edge_df = self._extend_edges(
            relation_df, concept_df, similarity_df, root_index
        )
        edge_df = await self._connect_graph(edge_df, concept_df, similarity_df)
        minimal_edge_df = self._remove_redundant_edges(edge_df, num_nodes)
        return await self._wrap_up(minimal_edge_df, root_index)
