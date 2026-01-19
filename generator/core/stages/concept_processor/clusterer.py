import asyncio as aio
import math

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering, KMeans

from common_utils.logger_config import logger
from generator.chainer import ChainCreator, ChainRunner
from generator.chainer.utils.constants import ChainTypes as Ct
from generator.common.constants import ColVals
from generator.common.constants import DataFrameCols as Col
from generator.common.constants import EnvConsts
from generator.common.constants import FieldNames as Fn
from generator.core.structs import RawMindMapData
from generator.core.utils.constants import ClusteringMethods as Cm
from generator.core.utils.constants import FrontEndStatuses as Fes
from generator.core.utils.constants import Pi
from generator.core.utils.frontend_handler import put_status

from . import utils


class ConceptClusterer:
    """
    Handles the hierarchical clustering and synthesis of concepts.

    This class builds a multi-level hierarchy from a flat set of
    concepts. It first clusters concepts by embedding similarity. Then,
    for each cluster, it uses an LLM to perform **synthesis**: a
    process of creating a new, abstract parent concept (a `RootConcept`)
    that summarizes the cluster's contents, along with other synthesized
    child concepts and their relationships.
    """

    def __init__(self, chain_creator: ChainCreator, queue: aio.Queue):
        """Initializes the clusterer with a queue for status updates."""
        self.chain_creator = chain_creator
        self.queue = queue
        self.chain_runner = ChainRunner()

    async def create_hierarchy(self, data: RawMindMapData) -> RawMindMapData:
        """
        Defines the skeleton for the concept clustering process.

        This template method orchestrates the creation of a concept
        hierarchy. It first checks if the graph is too small to
        cluster meaningfully. If not, it determines a dynamic max
        cluster size and executes the core clustering strategy.

        Args:
            data: The input data containing the flat list of concepts.

        Returns:
            The data structure now containing the full concept
            hierarchy, including new parent concepts and updated
            relations.
        """
        concept_df = data.concept_df
        relation_df = data.relation_df
        chunk_df = data.chunk_df

        is_graph_trivial = len(chunk_df) <= 1 or len(concept_df) < 4
        if is_graph_trivial:
            concept_df[Col.LVL] = 1
            return RawMindMapData(
                concept_df=concept_df,
                relation_df=relation_df,
                root_df=None,
                chunk_df=data.chunk_df,
                flat_part_df=data.flat_part_df,
            )

        logger.info("Creating concept cluster hierarchy.")
        max_cluster_size = self._det_max_cluster_size(chunk_df)

        concept_df, relation_df, root_df = (
            await self._execute_clustering_strategy(
                concept_df, relation_df, min(max_cluster_size, 12)
            )
        )

        logger.info(f"Created hierarchy with {len(concept_df)} concepts...")
        return RawMindMapData(
            concept_df=concept_df,
            relation_df=relation_df,
            root_df=root_df,
            chunk_df=data.chunk_df,
            flat_part_df=data.flat_part_df,
        )

    async def _execute_clustering_strategy(
        self,
        concept_df: pd.DataFrame,
        relation_df: pd.DataFrame,
        max_chunk_size: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Synthesizes hierarchical clusters of concepts from scratch.

        This method iteratively builds a hierarchy. In each loop, it:
        1. Clusters the active concepts of the current level.
        2. Synthesizes new, higher-level concepts from each cluster
           using an LLM.
        3. Processes the results, updating concepts and relations.
        4. Promotes the new parent concepts to the next level.

        The process continues until a root level with a manageable
        number of concepts is reached.

        Args:
            concept_df: DataFrame containing the initial flat concepts.
            relation_df: DataFrame containing relationships.
            max_chunk_size: The maximum number of concepts per cluster.

        Returns:
            A tuple containing the final concept DataFrame, the final
            relation DataFrame, and the root concepts DataFrame.
        """
        lvl_qapair_df = concept_df.copy()
        lvl = 1

        while True:
            if len(lvl_qapair_df) < 4:
                logger.info(
                    f"Stopping hierarchy creation: only {len(lvl_qapair_df)} "
                    "concepts remain, which will form the root level."
                )
                return concept_df, relation_df, lvl_qapair_df

            lvl_qapair_df[Col.LVL] = lvl
            lvl_qapair_df = utils.embed_active_concepts(lvl_qapair_df)
            embedding_matrix = np.stack(lvl_qapair_df[Col.EMBEDDING])
            lvl_qapair_indices = lvl_qapair_df.index.to_numpy()

            logger.info(f"Creating clusters for Level {int(lvl)}")
            lvl_mult, clusters = self._create_lvl_clusters(
                embedding_matrix, lvl_qapair_indices, max_chunk_size
            )

            # Assign cluster IDs back to the DataFrame for context.
            qapair_to_cluster = {
                idx: cid
                for cid, cluster in enumerate(clusters)
                for idx in cluster
            }
            lvl_qapair_df[Col.CLUSTER_ID] = lvl_qapair_df.index.map(
                qapair_to_cluster
            )

            logger.info(f"Synthesizing clusters for Level {int(lvl)}")
            synthesized_clusters = await self._synth_clusters(
                lvl_qapair_df, clusters, int(lvl)
            )

            (
                processed_lvl_df,
                next_lvl_df,
                relation_df,
            ) = await self._process_synth_clusters(
                synthesized_clusters,
                lvl_qapair_df,
                relation_df,
                lvl,
                concept_df.index.max(),
            )

            logger.info(f"Hierarchy level {int(lvl)} created.")

            # Accumulate the processed concepts from the current level.
            if lvl > 1:
                concept_df = pd.concat([concept_df, processed_lvl_df])
            else:
                concept_df = processed_lvl_df

            # Exit condition: The process stops if the hierarchy
            # multiplier is at its base (0), or if the number of
            # clusters created is small enough to be considered the
            # root level.
            is_root_level = len(clusters) <= min(max_chunk_size, 12)
            if lvl_mult == 0 or is_root_level:
                return concept_df, relation_df, next_lvl_df
            else:
                lvl += 1
                lvl_qapair_df = next_lvl_df

    @classmethod
    def _cluster_until_max_size(
        cls,
        feature_matrix: NDArray,
        qapair_indices: NDArray,
        max_cluster_size: int,
        num_clusters: int,
        cluster_method_name: str,
    ) -> list[NDArray]:
        """
        Iteratively clusters until all clusters are <= max_cluster_size.

        This method takes a set of concepts and an initial number of
        clusters to create. It then recursively splits any resulting
        cluster that is larger than `max_cluster_size` until all final
        clusters are compliant. This ensures that no single cluster is
        too large for the subsequent synthesis step.

        Args:
            feature_matrix: Matrix of features (embeddings) to cluster.
            qapair_indices: Original indices of the concepts.
            max_cluster_size: The maximum allowed size for any cluster.
            num_clusters: The initial number of clusters to create.
            cluster_method_name: The name of the clustering algorithm
                to use ('kmeans' or 'agglomerative').

        Returns:
            A list of arrays, where each array contains the original
            indices for one final, size-compliant cluster.
        """
        cluster_methods = {
            Cm.KMEANS: cls._do_kmeans_clustering,
            Cm.AGGLOMERATIVE: cls._do_agglomerative_clustering,
        }
        cluster_method = cluster_methods.get(cluster_method_name)
        if not cluster_method:
            raise KeyError(
                f"Unsupported clustering method: {cluster_method_name}"
            )

        cluster_labels = cluster_method(feature_matrix, num_clusters)

        # Group indices by their assigned cluster label.
        clusters_to_process = []
        for label in range(num_clusters):
            mask = cluster_labels == label
            if np.any(mask):
                # Add (features, original_indices) tuple for the cluster.
                clusters_to_process.append(
                    (feature_matrix[mask], qapair_indices[mask])
                )

        final_clusters = []
        # Use a queue-like approach to process oversized clusters.
        while clusters_to_process:
            features, indices = clusters_to_process.pop(0)
            if len(features) > max_cluster_size:
                # This cluster is too big; split it further.
                num_sub_clusters = math.ceil(len(features) / max_cluster_size)
                sub_labels = cluster_method(features, num_sub_clusters)

                # Add the new sub-clusters back to the queue for processing.
                for sub_label in range(num_sub_clusters):
                    mask = sub_labels == sub_label
                    if np.any(mask):
                        clusters_to_process.append(
                            (features[mask], indices[mask])
                        )
            else:
                # This cluster is compliant; add its indices to the final list.
                final_clusters.append(indices)

        return final_clusters

    @staticmethod
    def _combine_small_clusters(
        clusters: list[NDArray], target_cluster_size: int = 12
    ) -> list[list[int]]:
        """
        Combines small clusters into larger ones using a greedy
        approach.

        After splitting, there may be many small, fragmented clusters.
        This method sorts clusters by size and iteratively merges the
        smallest ones together until the combined size approaches the
        target size. This helps create more substantial and meaningful
        higher-level concepts.

        Args:
            clusters: A list of cluster arrays (lists of indices).
            target_cluster_size: The desired maximum size for combined
                clusters.

        Returns:
            A list of the new, combined cluster arrays.
        """
        # The current implementation is procedural but very clear and
        # easy to understand.
        clusters = sorted(clusters, key=len)
        combined_clusters = []
        cur_combined_cluster = []
        cur_combined_cluster_size = 0

        for cluster in clusters:
            cluster_size = len(cluster)
            if cur_combined_cluster_size + cluster_size <= target_cluster_size:
                cur_combined_cluster.extend(cluster)
                cur_combined_cluster_size += cluster_size
            else:
                combined_clusters.append(cur_combined_cluster)
                cur_combined_cluster = list(cluster)
                cur_combined_cluster_size = cluster_size

        if cur_combined_cluster:
            combined_clusters.append(cur_combined_cluster)

        return combined_clusters

    @classmethod
    def _create_lvl_clusters(
        cls,
        embedding_matrix: NDArray,
        concept_indices: NDArray,
        max_cluster_size: int,
    ) -> tuple[int, list[list[int]]]:
        """
        Creates hierarchical level clusters from embeddings.

        This method uses a heuristic to determine the optimal number of
        clusters for creating a balanced hierarchy. It aims for a
        branching factor of ~3 at each level. If the initial heuristic
        results in too many final clusters after combining small ones,
        it increases the initial cluster count and tries again.

        Args:
            embedding_matrix: Matrix of embeddings for clustering.
            concept_indices: Indices of concepts for the embeddings.
            max_cluster_size: Maximum allowed size for any cluster.

        Returns:
            A tuple containing the calculated level multiplier and the
            final list of combined clusters for this level.
        """
        num_concepts = len(concept_indices)
        estim_num_clusters = num_concepts / max_cluster_size
        # The formula aims for a base number of clusters that
        # grows by a factor of 3 at each level of the hierarchy.
        quarter_estim = estim_num_clusters / 4
        lvl_mult_float = math.log(quarter_estim, 3) if quarter_estim > 1 else 0
        lvl_mult = max(math.ceil(lvl_mult_float), 0)

        # This loop ensures that if the initial heuristic results in too
        # many final clusters, we increase the multiplier and try again.
        while True:
            # The target number of clusters grows exponentially.
            max_num_lvl_clusters = 4 * (3**lvl_mult)
            split_clusters = cls._cluster_until_max_size(
                embedding_matrix,
                concept_indices,
                max_cluster_size,
                max_num_lvl_clusters,
                Cm.AGGLOMERATIVE,
            )

            combined_clusters = cls._combine_small_clusters(split_clusters)

            if len(combined_clusters) <= max_num_lvl_clusters:
                return lvl_mult, combined_clusters
            else:
                # If we still have too many clusters, increase the level
                # multiplier, which increases the initial number of
                # clusters on the next attempt, leading to smaller clusters.
                lvl_mult += 1

    @staticmethod
    def _det_max_cluster_size(chunk_df: pd.DataFrame) -> int:
        """
        Determines max cluster size based on concept density in chunks.

        This heuristic sets the `max_cluster_size` based on the maximum
        number of concepts found within any single source document
        chunk.
        This helps adapt the clustering process to the density of the
        source material.

        Args:
            chunk_df: DataFrame of document chunks, which may contain a
                list of concept IDs associated with each chunk.

        Returns:
            The calculated maximum cluster size.
        """
        if chunk_df.empty or Col.CONCEPT_IDS not in chunk_df.columns:
            return 5
        chunk_concept_ids = chunk_df[Col.CONCEPT_IDS].dropna()
        if chunk_concept_ids.empty:
            return 5
        max_num_chunk_concepts = int(chunk_concept_ids.apply(len).max())
        return 5 if max_num_chunk_concepts <= 10 else max_num_chunk_concepts

    @staticmethod
    def _do_agglomerative_clustering(
        feature_matrix: np.ndarray, num_clusters: int
    ) -> NDArray:
        """
        Performs agglomerative clustering on a feature matrix.

        Includes an option for deterministic output, which is crucial
        for reproducibility and testing. When enabled, it pre-sorts the
        data before clustering and restores the original label order
        afterward.

        Args:
            feature_matrix: The matrix of features to cluster.
            num_clusters: The desired number of clusters.

        Returns:
            An array of cluster labels for each feature vector.
        """
        agglomerative = AgglomerativeClustering(n_clusters=num_clusters)
        if EnvConsts.IS_STABLE_AGGLOMERATIVE:
            # This block ensures deterministic clustering, which is
            # crucial for reproducibility. It works by pre-sorting data
            # before clustering and then restoring the original order
            # of the labels afterward.
            original_indices = np.arange(feature_matrix.shape[0])

            # Create a stable sort order by sorting by original index,
            # then by each feature value in reverse.
            reversed_features = feature_matrix.T[::-1]
            keys = [original_indices] + list(reversed_features)
            sorted_indices = np.lexsort(keys)

            sorted_features = feature_matrix[sorted_indices]
            labels_sorted = agglomerative.fit_predict(sorted_features)

            # Invert the permutation to restore the original order.
            inv_permutation = np.argsort(sorted_indices)
            return labels_sorted[inv_permutation]

        return agglomerative.fit_predict(feature_matrix)

    @staticmethod
    def _do_kmeans_clustering(
        feature_matrix: np.ndarray, num_clusters: int
    ) -> NDArray:
        """Performs K-means clustering on a feature matrix.

        Args:
            feature_matrix: The matrix of features to cluster.
            num_clusters: The desired number of clusters.

        Returns:
            An array of cluster labels for each feature vector.
        """
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto")
        return kmeans.fit_predict(feature_matrix)

    @classmethod
    def _get_concepts_from_clusters(
        cls,
        cluster_synth_results: list,
        lvl: int,
        concept_name_to_id: dict[str, int],
    ) -> list[dict]:
        """
        Extracts and formats concepts from the LLM synthesis output.

        This function parses the structured result from the synthesis
        chain. It extracts the `RootConcept` and any other
        `SynthesizedConcept` objects returned for each cluster, formats
        them, and prepares them to be added to the main DataFrame.

        Args:
            cluster_synth_results: Raw structured results from the LLM.
            lvl: The current hierarchical level.
            concept_name_to_id: Mapping from concept names to their IDs
                for resolving source concept dependencies.

        Returns:
            A list of new concept dictionaries.
        """
        concepts = []
        for cluster_id, result in enumerate(cluster_synth_results):
            # Safely extract the root concept from the result object.
            root = getattr(result, Fn.ROOT_CONCEPT, None)
            if not root:
                continue  # Skip if the result is malformed.

            root_dict = utils.rename_concept_fields(
                root.model_dump(), concept_name_to_id
            )
            concepts.append(
                {**root_dict, Col.CLUSTER_ID: cluster_id, Col.LVL: lvl + 1}
            )

            # Safely extract the other synthesized concepts.
            synth_concepts = getattr(result, Fn.SYNTH_CONCEPTS, [])
            for synth_qapair in synth_concepts:
                qapair_dict = utils.rename_concept_fields(
                    synth_qapair.model_dump(), concept_name_to_id
                )
                concepts.append(
                    {**qapair_dict, Col.CLUSTER_ID: cluster_id, Col.LVL: lvl}
                )

        return concepts

    @classmethod
    async def _process_synth_clusters(
        cls,
        synth_clusters: list,
        lvl_qapair_df: pd.DataFrame,
        rel_df: pd.DataFrame,
        lvl: int,
        max_id: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Processes LLM synthesis results to update concepts and relations.

        This is a critical step that integrates the LLM's output back
        into the system's data structures. It adds the new/modified
        concepts, deactivates their sources, updates relations based on
        the LLM's output, and prepares the data for the next level.

        Args:
            synth_clusters: Structured results from the synthesis chain.
            lvl_qapair_df: Concepts DataFrame for the current level.
            rel_df: The current relations DataFrame.
            lvl: The current hierarchical level being processed.
            max_id: The maximum concept ID seen so far.

        Returns:
            A tuple containing:
            - DataFrame of concepts finalized at this level.
            - DataFrame of new concepts promoted to the next level.
            - The fully updated relationships DataFrame.
        """
        # Create a name-to-ID map for concepts at the current level.
        qapair_name_to_id = pd.Series(
            lvl_qapair_df.index, index=lvl_qapair_df[Col.NAME]
        ).to_dict()

        # Extract and add the new concepts from the synthesis results.
        qapairs = cls._get_concepts_from_clusters(
            synth_clusters, lvl, qapair_name_to_id
        )
        lvl_qapair_df = utils.add_concepts(lvl_qapair_df, qapairs, max_id)

        # Update derived fields for the newly added concepts.
        answer_str = lvl_qapair_df.apply(utils.repr_answer, axis=1)
        lvl_qapair_df[Col.ANSWER_STR] = answer_str
        lvl_qapair_df = utils.make_unique_names(lvl_qapair_df)

        # Prepare to process new relationships from the synthesis results.
        active_mask = lvl_qapair_df[Col.IS_ACTIVE_CONCEPT] == ColVals.TRUE_INT
        cur_lvls_mask = lvl_qapair_df[Col.LVL].isin([lvl, lvl + 1])
        lvl_active_df = lvl_qapair_df.loc[cur_lvls_mask & active_mask].copy()

        # A composite key is needed to uniquely identify a concept within
        # its cluster, as names might not be unique across clusters.
        composite_key = lvl_active_df[Col.NAME].str.replace(
            r"_.*", "", regex=True
        ) + lvl_active_df[Col.CLUSTER_ID].astype(str)
        rel_name_to_id_map = pd.Series(
            lvl_active_df.index, index=composite_key
        ).to_dict()

        # Extract and resolve new relationships.
        new_rels = []
        raw_rels = [
            {**rel.model_dump(), Col.CLUSTER_ID: cid}
            for cid, res in enumerate(synth_clusters)
            for rel in getattr(res, "relations", [])
        ]

        for rel in raw_rels:
            try:
                cid = rel.pop(Col.CLUSTER_ID)
                origin_key = f"{rel[Fn.ORIGIN_CONCEPT_NAME]}{cid}"
                target_key = f"{rel[Fn.TARGET_CONCEPT_NAME]}{cid}"
                new_rels.append(
                    {
                        Col.ORIGIN_CONCEPT_ID: rel_name_to_id_map[origin_key],
                        Col.TARGET_CONCEPT_ID: rel_name_to_id_map[target_key],
                        Col.WEIGHT: 2.0,  # Default weight for new relations.
                    }
                )
            except KeyError as e:
                logger.warning(f"Skipping relation due to missing key: {e}")

        # Update the main relations DataFrame with the new ones.
        new_rel_df = pd.DataFrame(new_rels)
        rel_df = utils.update_rel_df(rel_df, new_rel_df, lvl_qapair_df)

        # Split concepts into those remaining at this level and those
        # promoted to the next.
        next_lvl_mask = lvl_qapair_df[Col.LVL] == lvl + 1
        return (
            lvl_qapair_df[~next_lvl_mask],
            lvl_qapair_df[next_lvl_mask],
            rel_df,
        )

    async def _put_cluster_synth_status(
        self,
        completed: int,
        total: int,
        level: int,
    ):
        """
        Updates and logs the status for cluster synthesis progress.

        This callback sends progress updates to a frontend via a queue.
        It rounds the percentage to the nearest 5% to avoid sending an
        excessive number of status updates for small changes.

        Args:
            completed: Number of completed clusters.
            total: Total number of clusters to process.
            level: The current hierarchical level being processed.
        """
        cluster_text = "cluster" if completed == 1 else "clusters"
        logger.info(
            f"Level {level}: {completed} {cluster_text} out of {total} processed"
        )

        # This logic rounds the percentage to the nearest 5% (e.g., 12%
        # becomes 10%, 13% becomes 15%), with a floor of 5%. This
        # prevents sending too many granular status updates.
        raw_percentage = (completed / total) * 100
        rounded_percentage = max(5, 5 * round(raw_percentage / 5))

        status = Fes.DEFINE_CONCEPTS.format(int(level))
        status_details = Fes.PROGRESS.format(rounded_percentage)
        await put_status(self.queue, status, status_details)

    async def _synth_clusters(
        self, concept_df: pd.DataFrame, clusters: list[list[int]], lvl: int
    ) -> list:
        """
        Performs synthesis on concept clusters using an LLM chain.

        For each cluster, this method calls an LLM instructed to:
        1. Create a new parent `RootConcept` summarizing the cluster.
           This root can be newly inferred, or an existing concept
           promoted to a higher level if it's sufficiently broad.
        2. Create other `SynthesizedConcept` objects by merging
           redundant facts while preserving distinct ones.
        3. Define relationships between these new concepts.

        Args:
            concept_df: DataFrame containing all concepts.
            clusters: A list of clusters, where each cluster is a list
                of concept indices.
            lvl: The current hierarchical level.

        Returns:
            A list of structured synthesis results from the LLM chain.
        """
        inputs = [
            {Pi.CLUSTER_SYNTH: utils.repr_concepts(concept_df, cluster)}
            for cluster in clusters
        ]
        cluster_synth_chain = self.chain_creator.choose_chain(Ct.CLUSTER_SYNTH)

        return await ChainRunner().run_chains_w_status_updates(
            [(cluster_synth_chain, inputs)],
            self._put_cluster_synth_status,
            level=lvl,
        )


class AddConceptClusterer(ConceptClusterer):
    """
    Extends ConceptClusterer to merge new concepts into an existing
    hierarchy.

    Instead of building a hierarchy from scratch, this class takes a
    set of new concepts and an existing multi-level hierarchy. It then
    intelligently rebuilds the hierarchy from the bottom up, merging
    the new concepts with existing ones at each level and re-running
    the synthesis process.
    """

    async def create_hierarchy(self, data: RawMindMapData) -> RawMindMapData:
        """
        Creates a hierarchy, merging new concepts into an existing
        one.

        This overridden method checks if there are enough *new*
        concepts to warrant a full re-clustering. If so, it executes
        the specialized level-by-level merging strategy.

        Args:
            data: The input data containing both old and new concepts.

        Returns:
            The data structure with the newly rebuilt hierarchy.
        """
        concept_df = data.concept_df
        relation_df = data.relation_df
        chunk_df = data.chunk_df

        num_new_concepts = pd.Series(concept_df["new"] == 1).sum()
        if num_new_concepts < 4:
            logger.info(
                "Skipping hierarchy creation for small number of new concepts "
                f"({num_new_concepts})"
            )
            concept_df[Col.LVL] = 1
            return RawMindMapData(
                concept_df=concept_df,
                relation_df=relation_df,
                root_df=None,
                chunk_df=data.chunk_df,
                flat_part_df=data.flat_part_df,
                old_root_concept=data.old_root_concept,
            )

        max_cluster_size = self._det_max_cluster_size(chunk_df)
        concept_df, relation_df, root_df = (
            await self._execute_clustering_strategy(
                concept_df, relation_df, min(max_cluster_size, 12)
            )
        )

        return RawMindMapData(
            concept_df=concept_df,
            relation_df=relation_df,
            root_df=root_df,
            chunk_df=data.chunk_df,
            flat_part_df=data.flat_part_df,
            old_root_concept=data.old_root_concept,
        )

    # Corrected version of _execute_clustering_strategy
    # The other two methods you provided are correct and do not need changes.

    async def _execute_clustering_strategy(
        self,
        concept_df: pd.DataFrame,
        relation_df: pd.DataFrame,
        max_cluster_size: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Synthesizes a hierarchy by merging new and old concepts.

        This strategy rebuilds the hierarchy level by level. For each
        existing level, it combines the old concepts at that level with
        the new concepts promoted from the level below. This combined
        group is then re-clustered and re-synthesized, producing a new
        set of concepts for the next level up.

        Args:
            concept_df: DataFrame with old and new concepts, marked by a
                'new' column. Old concepts must have a `LVL` column.
            relation_df: DataFrame containing existing relationships.
            max_cluster_size: Maximum number of concepts in a cluster.

        Returns:
            A tuple of (processed_concepts, updated_relations,
                root_concepts).
        """
        new_concepts_df = concept_df[concept_df["new"] == 1].copy()
        old_concepts_df = concept_df[concept_df["new"] == 0].copy()

        concepts_to_process_next = new_concepts_df
        processed_concepts_df = pd.DataFrame()

        max_concept_id = concept_df.index.max()

        # Iterate through existing levels in ascending order.
        # A custom sort key ensures the root level (-1) is processed last.
        sorted_levels = sorted(
            old_concepts_df[Col.LVL].unique(), key=lambda x: (x == -1, x)
        )
        for lvl in sorted_levels:
            level_group = old_concepts_df[old_concepts_df[Col.LVL] == lvl]
            current_processing_df = pd.concat(
                [level_group, concepts_to_process_next]
            )

            if len(current_processing_df) < 4:
                logger.info(
                    f"Stopping hierarchy rebuild at level {lvl}: only "
                    f"{len(current_processing_df)} concepts to process. "
                    "They will form the new root level."
                )
                concepts_to_process_next = current_processing_df
                break

            current_processing_df[Col.LVL] = lvl

            # Embed, cluster, and synthesize the combined group.
            current_processing_df = utils.embed_active_concepts(
                current_processing_df
            )
            embedding_matrix = np.stack(current_processing_df[Col.EMBEDDING])
            indices_arr = current_processing_df.index.to_numpy()

            logger.info(f"Re-clustering concepts for Level {lvl}")
            _, clusters = self._create_lvl_clusters(
                embedding_matrix, indices_arr, max(max_cluster_size, 10)
            )
            current_processing_df[Col.CLUSTER_ID] = (
                current_processing_df.index.map(
                    {idx: cid for cid, c in enumerate(clusters) for idx in c}
                )
            )

            synthesized_clusters = await self._synth_clusters(
                current_processing_df, clusters, lvl
            )
            (
                finished_level_df,
                concepts_to_process_next,
                relation_df,
            ) = await self._process_synth_clusters(
                synthesized_clusters,
                current_processing_df,
                relation_df,
                lvl,
                max_concept_id,
            )

            processed_concepts_df = pd.concat(
                [processed_concepts_df, finished_level_df]
            )

            if not finished_level_df.empty:
                max_concept_id = max(
                    max_concept_id, finished_level_df.index.max()
                )
            if not concepts_to_process_next.empty:
                max_concept_id = max(
                    max_concept_id, concepts_to_process_next.index.max()
                )

        # The remaining concepts are the new roots of the hierarchy.
        return processed_concepts_df, relation_df, concepts_to_process_next
