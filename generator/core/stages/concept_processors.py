import asyncio as aio
import math
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.runnables import RunnableSerializable
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering, KMeans

from ...chainer.chain_factory import ChainCreator, ChainRunner
from ...chainer.embedder import embed
from ...chainer.response_formats import (
    RootClusterSynthesisResult,
    SynthesizedAnswerPart,
)
from ...utils.constants import FACT, SOURCE_IDS, STABLE_AGGLOMERATIVE
from ...utils.constants import ChainTypes as Ct
from ...utils.constants import ClusteringMethods as Cm
from ...utils.constants import DataFrameCols as Col
from ...utils.constants import Fn
from ...utils.constants import FrontEndStatuses as Fes
from ...utils.constants import Pi
from ...utils.context import cur_llm_cost_handler
from ...utils.frontend_handler import put_status
from ...utils.logger import logging
from ...utils.misc import env_to_bool


class ConceptProcessor:
    """
    Handles concept processing operations for mindmap.
    Includes clustering, deduplication,
    and formatting data for visualization.
    """

    def __init__(self, queue: aio.Queue):
        """
        Initialize the Postprocessor.

        Args:
            queue: Asynchronous queue for handling processing tasks
        """
        self.queue = queue

    @staticmethod
    def _format_source_ids(row: pd.Series) -> list[dict]:
        """
        Format source IDs for a row of data.

        Args:
            row: DataFrame row containing answer and part IDs

        Returns:
            List of dictionaries mapping facts to their source IDs
        """
        return [
            {
                FACT: answer_part[Fn.FACT],
                SOURCE_IDS: sorted(row[Col.FLAT_PART_ID]),
            }
            for answer_part in row[Col.ANSWER]
        ]

    @staticmethod
    def _repr_answer(row: pd.Series) -> str:
        """
        Create a string representation of an answer
        with source citations.

        Args:
            row: DataFrame row containing answer data

        Returns:
            Formatted string with facts and their sources
        """
        repr_facts = []
        answer = row[Col.ANSWER]
        for answer_part in answer:
            if isinstance(answer_part, dict):
                fact = answer_part[FACT]
                sources = answer_part[SOURCE_IDS]
            else:
                fact = getattr(answer_part, Fn.SYNTH_FACT)
                sources = getattr(answer_part, Fn.CITATIONS)
            sources_str = "[" + "][".join(map(str, sources)) + "]"
            repr_facts.append(fact + sources_str)

        return " ".join(repr_facts)

    @staticmethod
    def embed_active_concepts(qapair_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate embeddings for active concepts that don't have them.

        Args:
            qapair_df: DataFrame containing concepts

        Returns:
            DataFrame with updated embeddings
        """
        qapair_embeddings = qapair_df[Col.EMBEDDING]
        qapairs_wo_embedding_mask = qapair_embeddings.isna()

        qapair_activity = qapair_df[Col.IS_ACTIVE_CONCEPT]
        active_qapairs_mask = qapair_activity == Col.ACTIVE_CONCEPT_TRUE_VAL

        qapairs_to_embed_mask = qapairs_wo_embedding_mask & active_qapairs_mask
        questions_to_embed = qapair_df.loc[qapairs_to_embed_mask, Col.QUESTION]
        questions_to_embed_list = questions_to_embed.tolist()

        embeddings_list = embed(questions_to_embed_list)
        embeddings = pd.Series(embeddings_list, index=questions_to_embed.index)
        qapair_df.loc[qapairs_to_embed_mask, Col.EMBEDDING] = embeddings

        return qapair_df

    @staticmethod
    def _do_kmeans_clustering(
        feature_matrix: np.array, num_clusters: int
    ) -> NDArray:
        """
        Perform K-means clustering on feature matrix.

        Args:
            feature_matrix: Matrix of features to cluster
            num_clusters: Number of clusters to create

        Returns:
            Array of cluster labels
        """
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        return kmeans.fit_predict(feature_matrix)

    @staticmethod
    def _do_agglomerative_clustering(
        feature_matrix: np.array, num_clusters: int
    ) -> NDArray:
        """
        Perform agglomerative hierarchical clustering on feature matrix.

        Args:
            feature_matrix: Matrix of features to cluster
            num_clusters: Number of clusters to create

        Returns:
            Array of cluster labels
        """
        agglomerative = AgglomerativeClustering(n_clusters=num_clusters)
        det = env_to_bool(STABLE_AGGLOMERATIVE)
        if det:
            original_indices = np.arange(feature_matrix.shape[0])
            reversed_features = feature_matrix.T[::-1]
            keys = [original_indices] + list(reversed_features)
            sorted_indices = np.lexsort(keys)
            sorted_features = feature_matrix[sorted_indices]
            labels_sorted = agglomerative.fit_predict(sorted_features)
            inv_permutation = np.argsort(sorted_indices)
            return labels_sorted[inv_permutation]
        return agglomerative.fit_predict(feature_matrix)

    @staticmethod
    def _combine_small_clusters(
        clusters: list[NDArray], target_cluster_size: int = 12
    ) -> list[list[int]]:
        """
        Combine small clusters to achieve a target cluster size.

        Args:
            clusters: List of cluster arrays
            target_cluster_size: Maximum size of combined clusters

        Returns:
            List of combined cluster arrays
        """
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

    @staticmethod
    def _repr_concepts(
        concept_df: pd.DataFrame, concept_ids: list[int] | None = None
    ) -> list[str]:
        """
        Create string representations of question-answer pairs.

        Args:
            concept_df: DataFrame containing QA pairs
            concept_ids: Optional list of QA pair IDs to filter by

        Returns:
            List of string representations of QA pairs
        """
        target_concept_df = (
            concept_df if concept_ids is None else concept_df.loc[concept_ids]
        )
        target_concepts = target_concept_df[
            [Col.NAME, Col.QUESTION, Col.ANSWER]
        ]

        return [
            (
                f"Name: {row[Col.NAME]}: Question: {row[Col.QUESTION]} "
                f"- Answer: {row[Col.ANSWER]}"
            )
            for idx, row in target_concepts.iterrows()
        ]

    @staticmethod
    def _rename_concept_fields(
        concept: dict[str, Any],
        qapair_name_to_id: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """
        Rename fields in a concept dictionary to standard column names.

        Args:
            concept: Dictionary containing concept data
            qapair_name_to_id: Optional mapping from concept names
                to IDs

        Returns:
            Dictionary with renamed fields
        """
        concept[Col.NAME] = concept.pop(Fn.SYNTH_CONCEPT_NAME)
        concept[Col.QUESTION] = concept.pop(Fn.SYNTH_QUESTION)
        concept[Col.ANSWER] = concept.pop(Fn.SYNTH_ANSWER)
        if qapair_name_to_id:
            concept[Col.SOURCE_IDS] = []
            for source_name in concept[Fn.SOURCE_CONCEPT_NAMES]:
                if source_name in qapair_name_to_id:
                    concept[Col.SOURCE_IDS].append(
                        qapair_name_to_id[source_name]
                    )
                else:
                    logging.warning(
                        f"Source concept name '{source_name}' "
                        f"is invalid and will be omitted from sources."
                    )
        else:
            concept[Col.SOURCE_IDS] = concept.pop(Fn.SOURCE_CONCEPT_IDS)
        return concept

    @staticmethod
    def _make_unique_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all names in the DataFrame are unique by adding suffixes.

        Args:
            df: DataFrame containing names that may have duplicates

        Returns:
            DataFrame with unique names
        """
        name_col = df[Col.NAME]
        counts = name_col.value_counts()
        duplicates = counts[counts > 1]
        dup_postfix = "_"

        for name in duplicates.index:
            name_mask = name_col == name
            duplicate_df = df[name_mask]

            active_col = duplicate_df[Col.IS_ACTIVE_CONCEPT]
            active_mask = pd.Series(active_col == Col.ACTIVE_CONCEPT_TRUE_VAL)
            inactive_mask = pd.Series(active_col == Col.ACTIVE_CONCEPT_TRUE_VAL)
            inactive_dup_mask = inactive_mask & name_mask
            num_active = active_mask.sum()

            if num_active == 1 and inactive_mask.sum() >= 1:
                df.loc[inactive_dup_mask, Col.NAME] += dup_postfix
            elif num_active > 1:
                active_indices = duplicate_df[active_mask].index
                for i, idx in enumerate(active_indices, 1):
                    df.at[idx, Col.NAME] = f"{name}_{i}"
                df.loc[inactive_dup_mask, Col.NAME] += dup_postfix
            else:
                df.loc[name_mask, Col.NAME] += dup_postfix

        return df

    @staticmethod
    def _get_concept_id_changes(
        concept_df: pd.DataFrame,
    ) -> dict[int, list[int]]:
        """
        Get mapping from source IDs to new QA pair IDs.

        Args:
            concept_df: DataFrame containing QA pairs

        Returns:
            Dictionary mapping source IDs to lists of new QA pair IDs
        """
        exploded_df = concept_df.explode(Col.SOURCE_IDS)
        grouped = exploded_df.groupby(Col.SOURCE_IDS)
        return grouped.apply(lambda row: list(row.index)).to_dict()

    @staticmethod
    def _update_rels(
        rel_df: pd.DataFrame, source_id_dict: dict[int, list[int]]
    ) -> pd.DataFrame:
        """
        Update relationships using source ID mapping.

        Args:
            rel_df: DataFrame containing relationships
            source_id_dict: Dictionary mapping source IDs to new IDs

        Returns:
            Updated relationships DataFrame
        """
        cols_to_update = [Col.ORIGIN_CONCEPT_ID, Col.TARGET_CONCEPT_ID]
        new_rows = []

        for idx, row in rel_df.iterrows():
            for col in cols_to_update:
                current_id = row[col]
                if current_id in source_id_dict:
                    new_ids = source_id_dict[current_id]
                    for new_id in new_ids:
                        new_row = row.copy()
                        new_row[col] = new_id
                        new_row[Col.WEIGHT] *= 0.99
                        new_rows.append(new_row)
                else:
                    new_rows.append(row.copy())

        updated_df = pd.DataFrame(new_rows, columns=rel_df.columns)
        return updated_df.reset_index(drop=True)

    @staticmethod
    def _repr_dup_group_as_list(group: pd.DataFrame) -> list[str]:
        """
        Create string representations of duplicate group entries.

        Args:
            group: DataFrame containing duplicate entries

        Returns:
            List of string representations
        """
        return [
            (
                f"ID: {index}"
                f', Name: "{row[Col.NAME]}"'
                f", Question: {row[Col.QUESTION]}"
                f" - Answer: {row[Col.ANSWER]}"
            )
            for index, row in group.iterrows()
        ]

    @staticmethod
    def get_max_value_or_negative_one(series: pd.Series | np.int64) -> int:
        """
        Get maximum value from series or -1 if series contains -1.

        Args:
            series: Series of values or single value

        Returns:
            Maximum value or -1
        """
        if isinstance(series, (int, float, np.number)):
            return -1 if series == -1 else series

        if hasattr(series, "isin"):
            return -1 if series.isin([-1]).any() else series.max()

        series_array = np.asarray(series)
        return -1 if np.any(series_array == -1) else series_array.max()

    @staticmethod
    def _replace_index_with_chunk_id(
        row: pd.Series,
        citation_series: pd.Series,
    ) -> list[dict[str, Any]]:
        """
        Replace indices in answer citations with chunk IDs.

        Args:
            row: DataFrame row containing answers
            citation_series: Series mapping indices to chunk IDs

        Returns:
            List of answer parts with updated citations
        """
        new_annotated_answer = []
        for answer in row[Col.ANSWER]:
            if isinstance(answer, dict):
                text = answer[FACT]
                if len(answer) > 1:
                    chunk_ids = set()
                    indices = answer[SOURCE_IDS]
                    if indices is not None and isinstance(indices, int):
                        # Try to get from slide_id_series, fallback to chunk_id_series if NaN
                        id_value = citation_series[indices - 1]
                        if isinstance(id_value, list):
                            chunk_ids.update(id_value)
                        else:
                            chunk_ids.add(id_value)
                    elif indices is not None and isinstance(indices, list):
                        for index in indices:
                            index = int(index)
                            # Try to get from slide_id_series, fallback to chunk_id_series if NaN
                            id_value = citation_series[index - 1]
                            if isinstance(id_value, list):
                                chunk_ids.update(id_value)
                            else:
                                chunk_ids.add(id_value)
                    new_annotated_answer.append(
                        {
                            FACT: text,
                            SOURCE_IDS: sorted(tuple(chunk_ids)),
                        }
                    )
                else:
                    new_annotated_answer.append([text, []])
            elif isinstance(answer, SynthesizedAnswerPart):
                text = getattr(answer, Fn.SYNTH_FACT)
                chunk_ids = set()
                indices = getattr(answer, Fn.CITATIONS)
                if indices is not None:
                    if indices is not None and isinstance(indices, int):
                        # Try to get from slide_id_series, fallback to chunk_id_series if NaN
                        id_value = citation_series[indices - 1]
                        if isinstance(id_value, list):
                            chunk_ids.update(id_value)
                        else:
                            chunk_ids.add(id_value)
                    elif indices is not None and isinstance(indices, list):
                        for index in indices:
                            index = int(index)
                            # Try to get from slide_id_series, fallback to chunk_id_series if NaN
                            id_value = citation_series[index - 1]
                            if isinstance(id_value, list):
                                chunk_ids.update(id_value)
                            else:
                                chunk_ids.add(id_value)
                    new_annotated_answer.append(
                        {
                            FACT: text,
                            SOURCE_IDS: sorted(tuple(chunk_ids)),
                        }
                    )
                else:
                    new_annotated_answer.append([text, []])
        return new_annotated_answer

    @staticmethod
    def _repr_question_w_answer(row: pd.Series) -> str:
        """
        Create a formatted representation of a question with its answer.

        Args:
            row: DataFrame row containing question and answer

        Returns:
            Formatted string with question and answer
        """
        repr_facts = []
        current_fact = ""
        current_sources = None
        answer = row[Col.ANSWER]
        for answer_part in answer:
            if isinstance(answer_part, dict):
                fact = answer_part[FACT]
                sources = answer_part[SOURCE_IDS]
            else:
                fact = answer_part.synth_fact
                sources = answer_part.citations

            if sources == current_sources:
                current_fact += " " + fact
            else:
                if current_fact:
                    sources_str = (
                        "^[" + "]^^[".join(map(str, current_sources)) + "]^"
                    )
                    repr_facts.append(current_fact + sources_str)
                current_fact = fact
                current_sources = sources

        if current_fact:
            sources_str = "^[" + "]^^[".join(map(str, current_sources)) + "]^"
            repr_facts.append(current_fact + sources_str)
        return (
            f"Question: {row[Col.QUESTION]} "
            f'- Answer: {" ".join(repr_facts)}'
        )

    @staticmethod
    def _split_list_into_batches(
        answers: list[str], batch_size: int = 10
    ) -> list[list[str]]:
        """
        Split a list into batches of specified size.

        Args:
            answers: List to split
            batch_size: Maximum size of each batch

        Returns:
            List of batches
        """
        batches = []
        for i in range(0, len(answers), batch_size):
            batches.append(answers[i : i + batch_size])
        return batches

    @staticmethod
    def _get_edge_idx(edge_df: pd.DataFrame, edge: tuple) -> int:
        """
        Find the index of an edge in an edge DataFrame.

        Args:
            edge_df: DataFrame containing edges.
            edge: Tuple of (origin_id, target_id).

        Returns:
            Index of the edge
        """
        return edge_df[
            (edge_df[Col.ORIGIN_CONCEPT_ID] == edge[0])
            & (edge_df[Col.TARGET_CONCEPT_ID] == edge[1])
        ].index.tolist()[0]

    @staticmethod
    def _reverse_edge(
        edge_df: pd.DataFrame, edge_index: int, edge: tuple
    ) -> pd.DataFrame:
        """
        Reverse the direction of an edge.

        Args:
            edge_df: DataFrame containing edges
            edge_index: Index of the edge to reverse
            edge: Tuple of (origin_id, target_id)

        Returns:
            Updated edge DataFrame
        """
        edge_df.loc[
            edge_index, [Col.ORIGIN_CONCEPT_ID, Col.TARGET_CONCEPT_ID]
        ] = (
            edge[1],
            edge[0],
        )
        return edge_df

    @classmethod
    def _prep_concept_df(cls, concept_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare concept DataFrame by initializing required columns
        and formatting answers.

        Args:
            concept_df: DataFrame containing concepts to prepare.

        Returns:
            DataFrame with initialized columns and formatted answers.
        """
        concept_df[Col.IS_ACTIVE_CONCEPT] = Col.ACTIVE_CONCEPT_TRUE_VAL
        concept_df[Col.EMBEDDING] = None
        concept_df[Col.ANSWER_STR] = np.nan
        answers_w_flat_part_ids = concept_df[[Col.ANSWER, Col.FLAT_PART_ID]]
        answers = answers_w_flat_part_ids.apply(cls._format_source_ids, axis=1)
        concept_df[Col.ANSWER] = answers
        str_answers = concept_df.apply(cls._repr_answer, axis=1)
        concept_df[Col.ANSWER_STR] = str_answers
        return concept_df

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
        Perform iterative clustering until all clusters
        have size <= max_cluster_size.

        This method recursively divides clusters
        that exceed the maximum size until
        all resulting clusters satisfy the size constraint.

        Args:
            feature_matrix: Matrix of features to cluster
            qapair_indices: Indices of concepts
                corresponding to feature matrix rows
            max_cluster_size: Maximum allowed size for any cluster
            num_clusters: Initial number of clusters to create
            cluster_method_name: Name of clustering method to use
                (kmeans or agglomerative)

        Returns:
            List of arrays, where each array contains indices
            of data points in one cluster

        Raises:
            KeyError: If the specified clustering method is not supported
        """
        cluster_methods = {
            Cm.KMEANS: cls._do_kmeans_clustering,
            Cm.AGGLOMERATIVE: cls._do_agglomerative_clustering,
        }

        if cluster_method_name not in cluster_methods:
            error_msg = f"Unsupported clustering method: {cluster_method_name}"
            raise KeyError(error_msg)

        cluster_method = cluster_methods[cluster_method_name]
        cluster_labels = cluster_method(feature_matrix, num_clusters)

        qapair_idx_col = "qapair_indices"
        cluster_labels_col = "cluster_labels"

        feature_df = pd.DataFrame(feature_matrix)
        feature_df[qapair_idx_col] = qapair_indices
        feature_df[cluster_labels_col] = cluster_labels

        cluster_label_groups = feature_df.groupby(cluster_labels_col)
        clusters = [
            (group.iloc[:, :-2].values, group[qapair_idx_col].values)
            for _, group in cluster_label_groups
        ]

        final_clusters = []
        while clusters:
            cluster_feature_matrix, cluster_indices = clusters.pop(0)
            cluster_size = len(cluster_feature_matrix)
            if cluster_size > max_cluster_size:
                num_sub_clusters = cluster_size // max_cluster_size + 1
                sub_labels = cluster_method(
                    cluster_feature_matrix, num_sub_clusters
                )
                sub_clusters = [
                    (
                        cluster_feature_matrix[sub_labels == sub_idx],
                        cluster_indices[sub_labels == sub_idx],
                    )
                    for sub_idx in range(max(sub_labels) + 1)
                ]
                clusters.extend(sub_clusters)
            else:
                final_clusters.append(cluster_indices)

        return final_clusters

    @classmethod
    def _create_lvl_clusters(
        cls,
        embedding_matrix: NDArray,
        concept_indices: NDArray,
        max_cluster_size: int,
    ) -> tuple[int, list[list[int]]]:
        """
        Create hierarchical level clusters based on embedding matrix.

        Args:
            embedding_matrix: Matrix of embeddings for clustering
            concept_indices: Indices of concepts
                corresponding to embeddings
            max_cluster_size: Maximum allowed size for any cluster

        Returns:
            Tuple of (level_multiplier, combined_clusters)
            where combined_clusters
            is a list of lists containing indices grouped into clusters
        """
        num_lvl_qapair = len(concept_indices)
        estim_num_clusters = num_lvl_qapair / max_cluster_size
        quarter_estim_num_clusters = estim_num_clusters / 4
        lvl_mult_float = math.log(quarter_estim_num_clusters, 3)
        lvl_mult = max(math.ceil(lvl_mult_float), 0)

        while True:
            max_num_lvl_clusters = 4 * 3**lvl_mult
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
                lvl_mult += 1

    async def _put_cluster_synth_status(
        self,
        completed: int,
        total: int,
        level: int,
    ):
        """
        Update status for cluster synthesis progress.

        Args:
            completed: Number of completed clusters
            total: Total number of clusters
            level: Current hierarchical level
        """
        cluster_text = "cluster" if completed == 1 else "clusters"
        logging.info(
            f"Level {level}:\n{completed} "
            f"{cluster_text} out of {total} processed"
        )

        raw_percentage = (completed / total) * 100
        rounded_percentage = max(5, 5 * round(raw_percentage / 5))

        status = Fes.DEFINE_CONCEPTS.format(level)
        status_details = Fes.PROGRESS.format(rounded_percentage)
        await put_status(self.queue, status, status_details)

    async def _synth_clusters(
        self, concept_df: pd.DataFrame, clusters: list[list[int]], lvl: int
    ) -> list:
        """
        Synthesize clusters by generating concept hierarchies.

        Args:
            concept_df: DataFrame containing concepts
            clusters: List of clusters,
                where each cluster is a list of concept indices
            lvl: Current hierarchical level

        Returns:
            List of cluster synthesis results
        """
        inputs = [
            {Pi.CLUSTER_SYNTH: self._repr_concepts(concept_df, cluster)}
            for cluster in clusters
        ]
        cluster_synth_chain = ChainCreator.choose_chain(Ct.CLUSTER_SYNTH)
        return await ChainRunner().run_chains_w_status_updates(
            [(cluster_synth_chain, inputs)],
            self._put_cluster_synth_status,
            level=lvl,
        )

    @classmethod
    def _get_concepts_from_clusters(
        cls,
        cluster_synth_results: list,
        lvl: int,
        concept_name_to_id: dict[str, int],
    ) -> list[dict]:
        """
        Extract QA pairs from cluster synthesis results.

        Args:
            cluster_synth_results: Results from cluster synthesis
            lvl: Current hierarchical level
            concept_name_to_id: Mapping from concept names to their IDs

        Returns:
            List of QA pair dictionaries with proper fields
        """
        concepts = []
        for cluster_id, result in enumerate(cluster_synth_results):
            root = dict(getattr(result, Fn.ROOT_CONCEPT))
            root = cls._rename_concept_fields(root, concept_name_to_id)
            concepts.append(
                {**root, Col.CLUSTER_ID: cluster_id, Col.LVL: lvl + 1}
            )

            for synth_qapair in getattr(result, Fn.SYNTH_CONCEPTS):
                qapair = dict(synth_qapair)
                qapair = cls._rename_concept_fields(qapair, concept_name_to_id)
                concepts.append(
                    {**qapair, Col.CLUSTER_ID: cluster_id, Col.LVL: lvl}
                )

        return concepts

    @classmethod
    def _add_concepts(
        cls,
        concept_df: pd.DataFrame,
        concepts: list[dict],
        max_id: int | None = None,
    ) -> pd.DataFrame:
        """
        Add new concepts to the existing DataFrame.

        Args:
            concept_df: Existing DataFrame of concepts.
            concepts: List of new concepts to add.
            max_id: Optional maximum ID to start indexing from.

        Returns:
            Updated DataFrame with new concepts added
        """
        counter = 0
        max_idx = concept_df.index.max()
        new_qapairs = []
        for qapair in concepts:
            if isinstance(qapair, dict):
                source_ids = []
                for source_id in qapair[Col.SOURCE_IDS]:
                    if source_id in concept_df.index:
                        source_ids.append(source_id)
                    else:
                        logging.warning(f"Nonexistent source_id: {source_id}")

                # chunk_ids = set()
                # slide_ids = set()
                flat_part_ids = set()
                embedding = None
                for source_id in source_ids:
                    source_qapair = concept_df.loc[source_id]
                    source_flat_part_ids = source_qapair[Col.FLAT_PART_ID]
                    flat_part_ids.update(source_flat_part_ids)

                    if len(source_ids) == 1:
                        source_question = source_qapair[Col.QUESTION]
                        question = qapair[Col.QUESTION]
                        is_question_diff = source_question != question

                        source_embedding = source_qapair[Col.EMBEDDING]
                        embedding = (
                            None if is_question_diff else source_embedding
                        )

                    concept_df.loc[source_id, Col.IS_ACTIVE_CONCEPT] = (
                        Col.ACTIVE_CONCEPT_FALSE_VAL
                    )

                qapair[Col.FLAT_PART_ID] = list(flat_part_ids)
                qapair[Col.EMBEDDING] = embedding
                qapair[Col.IS_ACTIVE_CONCEPT] = Col.ACTIVE_CONCEPT_TRUE_VAL

            else:
                source_name = qapair
                active_col = concept_df[Col.IS_ACTIVE_CONCEPT]
                active_mask = active_col == Col.ACTIVE_CONCEPT_TRUE_VAL
                name_mask = concept_df[Col.NAME] == source_name
                source_id = concept_df.loc[name_mask & active_mask].index[0]

                qapair = concept_df.loc[source_id].copy()
                qapair[Col.LVL] += 1
                qapair[Col.SOURCE_IDS] = [source_id]
                concept_df.loc[source_id, Col.IS_ACTIVE_CONCEPT] = (
                    Col.ACTIVE_CONCEPT_FALSE_VAL
                )

            if counter == 0:
                if max_id is not None and max_id > max_idx:
                    new_row_id = max_id + 1
                else:
                    new_row_id = max_idx + 1
                counter += 1
            else:
                new_row_id = new_qapairs[-1].index[-1] + 1
            new_qapairs.append(pd.DataFrame([qapair], index=[new_row_id]))

        return pd.concat([concept_df] + new_qapairs)

    @classmethod
    def _update_rel_df(
        cls,
        rel_df: pd.DataFrame,
        new_rel_df: pd.DataFrame,
        concept_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Update relations DataFrame
        with new relationships and propagate changes.

        Args:
            rel_df: Existing relationships DataFrame
            new_rel_df: New relationships to add
            concept_df: QA pairs DataFrame for resolving ID changes

        Returns:
            Updated relationships DataFrame
        """
        rel_df = pd.concat([rel_df, new_rel_df])

        qapair_id_changes = cls._get_concept_id_changes(concept_df)
        rel_df = cls._update_rels(rel_df, qapair_id_changes)

        rel_df.sort_values(Col.WEIGHT, ascending=False, inplace=True)
        return rel_df.drop_duplicates(
            subset=[Col.ORIGIN_CONCEPT_ID, Col.TARGET_CONCEPT_ID],
            keep="first",
            ignore_index=True,
        )

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
        Process synthesized clusters to update concepts and relations.

        Args:
            synth_clusters: Results from cluster synthesis
            lvl_qapair_df: Concepts DataFrame for current level
            rel_df: Relations DataFrame
            lvl: Current hierarchical level
            max_id: Maximum ID to use for new entries

        Returns:
            Tuple of (current_level_qapairs, next_level_qapairs,
                updated_relationships)
        """
        qapair_ids = lvl_qapair_df.index
        names = lvl_qapair_df[Col.NAME]
        qapair_name_to_id = pd.Series(qapair_ids, index=names).to_dict()

        qapairs = cls._get_concepts_from_clusters(
            synth_clusters, lvl, qapair_name_to_id
        )
        lvl_qapair_df = cls._add_concepts(lvl_qapair_df, qapairs, max_id)
        answer_str = lvl_qapair_df.apply(cls._repr_answer, axis=1)
        lvl_qapair_df[Col.ANSWER_STR] = answer_str
        lvl_qapair_df = cls._make_unique_names(lvl_qapair_df)

        active_col = lvl_qapair_df[Col.IS_ACTIVE_CONCEPT]
        active_mask = active_col == Col.ACTIVE_CONCEPT_TRUE_VAL
        cur_lvls = lvl_qapair_df[Col.LVL].isin([lvl, lvl + 1])
        lvl_active_qapair_df = lvl_qapair_df.loc[cur_lvls & active_mask].copy()
        str_names = lvl_active_qapair_df[Col.NAME].str
        clean_names = str_names.replace(r"_.*", "", regex=True)
        lvl_active_qapair_df[Col.NAME] = clean_names

        rels = [
            {**dict(rel), Col.CLUSTER_ID: cluster_id}
            for cluster_id, cluster_synth_result in enumerate(synth_clusters)
            for rel in cluster_synth_result.relations
        ]
        qapair_name_to_id = pd.Series(
            lvl_active_qapair_df.index,
            index=(
                lvl_active_qapair_df[Col.NAME]
                + lvl_active_qapair_df[Col.CLUSTER_ID].astype(str)
            ),
        ).to_dict()

        new_rels = []
        for rel in rels:
            try:
                cluster_id = rel.pop(Col.CLUSTER_ID)
                for partner, qapair_name in rel.items():
                    cluster_name = f"{qapair_name}{cluster_id}"
                    rel[partner] = qapair_name_to_id[cluster_name]
                new_rels.append(rel)
            except KeyError:
                warn_msg = "Skipping relation due to missing key: "
                logging.warning(warn_msg + str(rel))

        new_rel_df = pd.DataFrame(new_rels)
        new_rel_df[Col.WEIGHT] = 2.0
        new_rel_df.rename(
            columns={
                Fn.ORIGIN_CONCEPT_NAME: Col.ORIGIN_CONCEPT_ID,
                Fn.TARGET_CONCEPT_NAME: Col.TARGET_CONCEPT_ID,
            },
            inplace=True,
        )
        rel_df = cls._update_rel_df(rel_df, new_rel_df, lvl_qapair_df)

        next_lvl_mask = lvl_qapair_df[Col.LVL] == lvl + 1
        return (
            lvl_qapair_df[~next_lvl_mask],
            lvl_qapair_df[next_lvl_mask],
            rel_df,
        )

    async def _cluster_and_synthesize(
        self,
        concept_df: pd.DataFrame,
        relation_df: pd.DataFrame,
        max_chunk_size: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Synthesize hierarchical clusters of concepts.

        Iteratively creates levels of clusters until a root level with
        manageable number of concepts is reached.

        Args:
            concept_df: DataFrame containing concepts
            relation_df: DataFrame containing relationships between concepts
            max_chunk_size: Maximum number of QA pairs in a cluster

        Returns:
            Tuple of (concept_df, relation_df, root_df)
        """
        lvl_qapair_df = concept_df.copy()
        lvl = 1

        while True:
            lvl_qapair_df[Col.LVL] = lvl
            lvl_qapair_ids = lvl_qapair_df.index

            lvl_qapair_df = self.embed_active_concepts(lvl_qapair_df)
            embedding_matrix = np.stack(lvl_qapair_df[Col.EMBEDDING])
            lvl_qapair_indices_arr = lvl_qapair_ids.to_numpy()

            logging.info(f"Clusters creation. Details: Level {lvl}")
            lvl_mult, clusters = self._create_lvl_clusters(
                embedding_matrix,
                lvl_qapair_indices_arr,
                max_chunk_size,
            )
            qapair_to_cluster = {
                qapair_id: cluster_id
                for cluster_id, cluster in enumerate(clusters)
                for qapair_id in cluster
            }
            cluster_ids = lvl_qapair_ids.map(qapair_to_cluster)
            lvl_qapair_df[Col.CLUSTER_ID] = cluster_ids

            status_msg = f"Synthesizing clusters"
            logging.info(f"{status_msg}. Details: Level {lvl}")
            synthesized_clusters = await self._synth_clusters(
                lvl_qapair_df, clusters, lvl
            )

            (
                lvl_qapair_df,
                next_lvl_qapair_df,
                relation_df,
            ) = await self._process_synth_clusters(
                synthesized_clusters,
                lvl_qapair_df,
                relation_df,
                lvl,
                concept_df.index.max(),
            )

            level_text = "level was" if lvl == 1 else "levels were"
            status_msg = f"{lvl} hierarchy {level_text} created"
            logging.info(status_msg)

            if lvl > 1:
                concept_df = pd.concat([concept_df, lvl_qapair_df])
            else:
                concept_df = lvl_qapair_df

            # Root is created for each cluster
            is_num_root_fit = len(clusters) <= max_chunk_size
            if not (lvl_mult == 0 or is_num_root_fit):
                lvl += 1
                lvl_qapair_df = next_lvl_qapair_df
            else:
                return concept_df, relation_df, next_lvl_qapair_df

    @classmethod
    async def _synthesize_roots(
        cls, root_df: pd.DataFrame
    ) -> RootClusterSynthesisResult:
        """
        Synthesize a root concept from root-level concepts.

        Args:
            root_df: DataFrame containing root-level concepts

        Returns:
            Synthesis result containing root concept and sub-concepts
        """
        inputs = {Pi.CLUSTER_SYNTH: cls._repr_concepts(root_df)}
        chain = ChainCreator.choose_chain(Ct.ROOT_CLUSTER_SYNTH)
        return await ChainRunner().run_chain(chain, inputs)

    @classmethod
    async def _prep_roots(
        cls,
        root_df: pd.DataFrame,
        root_synth_results: RootClusterSynthesisResult,
        concept_name_to_id: dict[str, int],
    ) -> list[dict]:
        """
        Prepare root concepts from synthesis results.

        Args:
            root_df: DataFrame containing root-level concepts
            root_synth_results: Results from root synthesis
            concept_name_to_id: Mapping from concept names to IDs

        Returns:
            List of prepared root concept dictionaries
        """
        main_root = [getattr(root_synth_results, Fn.ROOT_CONCEPT)]
        roots = []
        for sub_root in (
            getattr(root_synth_results, Fn.SYNTH_CONCEPTS) + main_root
        ):
            qapair = dict(sub_root)
            qapair = cls._rename_concept_fields(qapair, concept_name_to_id)
            roots.append(
                {
                    **qapair,
                    Col.CLUSTER_ID: -1,
                    Col.LVL: root_df[Col.LVL].max(),
                }
            )

        roots[-1][Col.LVL] = -1
        return roots

    @classmethod
    async def define_root(
        cls,
        other_concept_df: pd.DataFrame,
        rel_df: pd.DataFrame,
        root_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, int]:
        """
        Define the root concept and its relationships to other concepts.

        Args:
            other_concept_df: DataFrame containing non-root concepts
            rel_df: DataFrame containing relationships
            root_df: Optional DataFrame containing root-level QA pairs

        Returns:
            Tuple of (updated_concept_df, updated_rel_df, root_index)
        """
        if root_df is not None and not root_df.empty:
            concept_df = pd.concat([other_concept_df, root_df])
        else:
            root_df = concept_df = other_concept_df

        root_synth_results = await cls._synthesize_roots(root_df)

        root_ids = root_df.index
        root_names = root_df[Col.NAME]
        qapair_name_to_id = pd.Series(root_ids, index=root_names).to_dict()
        sub_roots = await cls._prep_roots(
            root_df, root_synth_results, qapair_name_to_id
        )
        concept_df = cls._add_concepts(concept_df, sub_roots)
        concept_df = cls._make_unique_names(concept_df)

        root_index = concept_df.index[-1]
        synth_answers = concept_df.apply(cls._repr_answer, axis=1)
        concept_df[Col.ANSWER_STR] = synth_answers

        root_rels = []
        cluster_id_mask = concept_df[Col.CLUSTER_ID] == -1
        root_cluster_index = concept_df.loc[cluster_id_mask].index
        num_rels = min(len(root_cluster_index), 4)
        for idx in root_cluster_index[:num_rels]:
            root_rels.append(
                {
                    Col.ORIGIN_CONCEPT_ID: root_index,
                    Col.TARGET_CONCEPT_ID: idx,
                    Col.WEIGHT: 4.0,
                }
            )
        root_rel_df = pd.DataFrame(root_rels)

        rel_df = cls._update_rel_df(rel_df, root_rel_df, concept_df)

        rel_df[Col.TYPE] = Col.RELATED_TYPE_VAL

        return concept_df, rel_df, root_index

    @classmethod
    def _get_dup_groups(
        cls, concept_df: pd.DataFrame, root_id: int
    ) -> (
        tuple[list[list[str]], list[list[int]], list[int], int | None]
        | tuple[None, None, None, None]
    ):
        """
        Identify groups of duplicate concepts.

        Args:
            concept_df: DataFrame containing concepts
            root_id: ID of the root concept

        Returns:
            Tuple of (group_representations, group_indices,
                duplicate_indices, root_group_idx)
            Returns (None, None, None, None) if no duplicates are found
        """
        grouped_by_name = concept_df.groupby(Col.NAME)
        dup_df = grouped_by_name.filter(lambda group: len(group) > 1)
        if dup_df.empty:
            return None, None, None, None

        dup_mask = dup_df.duplicated(Col.NAME, keep=False)
        dup_indices = dup_df.index[dup_mask].tolist()

        grouped_dup = dup_df.groupby(Col.NAME)
        concept_group_repr = grouped_dup.apply(cls._repr_dup_group_as_list)

        group_dup_indices = [group.index.tolist() for _, group in grouped_dup]

        root_group_idx = next(
            (
                i
                for i, indices in enumerate(group_dup_indices)
                if root_id in indices
            ),
            None,
        )

        return (
            concept_group_repr.tolist(),
            group_dup_indices,
            dup_indices,
            root_group_idx,
        )

    @classmethod
    def _get_dedup_qapairs(
        cls,
        dedup_results: list,
        max_cluster_ids: list[int],
        max_lvls: list[int],
    ) -> list[dict]:
        """
        Extract concepts from deduplication results.

        Args:
            dedup_results: List of deduplication results
            max_cluster_ids: List of cluster IDs to assign
            max_lvls: List of levels to assign

        Returns:
            List of deduplicated QA pair dictionaries
        """
        concepts = []
        for dedup_result, cluster_id, lvl in zip(
            dedup_results, max_cluster_ids, max_lvls
        ):
            for concept in getattr(dedup_result, Fn.SYNTH_CONCEPTS):
                concept = dict(concept)
                concept = cls._rename_concept_fields(concept)
                concepts.append(
                    {**concept, Col.CLUSTER_ID: cluster_id, Col.LVL: lvl}
                )

        return concepts

    @classmethod
    async def dedup_concepts(
        cls, concept_df: pd.DataFrame, relation_df: pd.DataFrame, root_id: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, int]:
        """
        Deduplicate concepts by merging similar ones.

        Args:
            concept_df: DataFrame containing concepts
            relation_df: DataFrame containing relationships
            root_id: ID of the root concept

        Returns:
            Tuple of (deduplicated_concept_df, filtered_relation_df,
                new_root_id)
        """
        # Remove postfixes that artificially makes names unique
        str_names = concept_df[Col.NAME].str
        concept_df[Col.NAME] = str_names.replace(r"_.*", "", regex=True)

        # If LLM hallucinates or in some very unfortunate conditions
        # even if there are still concepts with duplicate names
        # we will just ignore them after max number of tries.
        num_max_tries = 50

        for _ in range(num_max_tries):
            active_col = concept_df[Col.IS_ACTIVE_CONCEPT]
            active_mask = active_col == Col.ACTIVE_CONCEPT_TRUE_VAL
            active_concept_df = concept_df[active_mask]
            (
                groups_of_dupes,
                groups_of_dup_indices,
                duplicate_indices,
                root_group_id,
            ) = cls._get_dup_groups(active_concept_df, root_id)

            if duplicate_indices is None:
                break

            max_cluster_ids_per_group = []
            max_lvls_per_group = []
            for group in groups_of_dup_indices:
                group_max_cluster_ids = []
                group_max_lvls = []
                for dup_id in group:
                    if dup_id == root_id:
                        continue
                    cluster_ids = concept_df.loc[dup_id, Col.CLUSTER_ID]
                    group_max_cluster_ids.append(
                        cls.get_max_value_or_negative_one(cluster_ids)
                    )
                    lvls = concept_df.loc[dup_id, Col.LVL]
                    group_max_lvls.append(
                        cls.get_max_value_or_negative_one(lvls)
                    )

                max_lvls_per_group.append(
                    cls.get_max_value_or_negative_one(pd.Series(group_max_lvls))
                )
                max_cluster_ids_per_group.append(
                    cls.get_max_value_or_negative_one(
                        pd.Series(group_max_cluster_ids)
                    )
                )

            root_ded_results = None
            if root_group_id is not None:
                root_group = groups_of_dupes.pop(root_group_id)
                root_chain = ChainCreator.choose_chain(Ct.DEDUP_ROOT)
                root_inputs = {Pi.CLUSTER_SYNTH: root_group}
                root_ded_results = await ChainRunner().run_chain(
                    root_chain, root_inputs
                )
            chain = ChainCreator.choose_chain(Ct.DEDUP)
            inputs = [{Pi.CLUSTER_SYNTH: group} for group in groups_of_dupes]
            ded_results = await ChainRunner().run_chain_on_batch(chain, inputs)
            root = None
            if root_ded_results is not None:
                ded_results.insert(root_group_id, root_ded_results)
                root = getattr(root_ded_results, Fn.ROOT_CONCEPT)
                root = dict(root)
                root = cls._rename_concept_fields(root)
                root = {**root, Col.CLUSTER_ID: -1, Col.LVL: -1}

            qapairs = cls._get_dedup_qapairs(
                ded_results,
                max_cluster_ids_per_group,
                max_lvls_per_group,
            )

            if root is not None:
                qapairs += [root]

            next_id = concept_df.index.max()
            concept_df = cls._add_concepts(concept_df, qapairs)
            synth_answers = concept_df.apply(cls._repr_answer, axis=1)
            concept_df[Col.ANSWER_STR] = synth_answers
            next_mask = concept_df.index > next_id

            names = concept_df[Col.NAME]
            new_rel_df_rows = []
            for result in ded_results:
                rels = result.relations
                for rel in rels:
                    src_qapair_mask = names == getattr(
                        rel, Fn.ORIGIN_CONCEPT_NAME
                    )
                    target_qapair_mask = names == getattr(
                        rel, Fn.TARGET_CONCEPT_NAME
                    )
                    new_rel_df_row = {
                        Col.ORIGIN_CONCEPT_ID: concept_df.index[
                            src_qapair_mask & next_mask
                        ][0],
                        Col.TARGET_CONCEPT_ID: concept_df.index[
                            target_qapair_mask & next_mask
                        ][0],
                        Col.TYPE: Col.RELATED_TYPE_VAL,
                        Col.WEIGHT: 3.0,
                    }
                    new_rel_df_rows.append(new_rel_df_row)

            new_rel_df = pd.DataFrame(new_rel_df_rows)
            relation_df = cls._update_rel_df(
                relation_df, new_rel_df, concept_df
            )

        lvls = concept_df[Col.LVL]
        cluster_ids = concept_df[Col.CLUSTER_ID]
        active_col = concept_df[Col.IS_ACTIVE_CONCEPT]
        new_root_index = concept_df.loc[
            (lvls == -1) & (cluster_ids == -1) & (active_col == 1)
        ].index[0]

        invalid_qapairs = concept_df.loc[
            (concept_df[Col.IS_ACTIVE_CONCEPT] == Col.ACTIVE_CONCEPT_FALSE_VAL)
        ].index.tolist()
        mask = (
            relation_df[[Col.ORIGIN_CONCEPT_ID, Col.TARGET_CONCEPT_ID]]
            .isin(invalid_qapairs)
            .any(axis=1)
        ) | (
            relation_df[Col.ORIGIN_CONCEPT_ID]
            == relation_df[Col.TARGET_CONCEPT_ID]
        )

        return concept_df, relation_df[~mask], new_root_index

    async def _put_prettification_status(
        self, completed: int, total: int
    ) -> None:
        """
        Update status for prettification progress.

        Args:
            completed: Number of completed items
            total: Total number of items
        """
        raw_percentage = (completed / total) * 100
        rounded_percentage = max(5, 5 * round(raw_percentage / 5))

        status_details = Fes.PROGRESS.format(rounded_percentage)
        await put_status(self.queue, Fes.PRETTIFY, status_details)

    async def process_answers_with_retry(
        self, batched_answers: list[list[str]], chain: RunnableSerializable
    ) -> list[str]:
        """
        Process all answer batches
        with automatic retry for failed batches.

        Args:
            batched_answers: List of batches,
                where each batch is a list of answers
            chain: Chain to use for processing

        Returns:
            List of processed answers
        """
        batch_info = [(i, batch) for i, batch in enumerate(batched_answers)]
        result_map = {}

        inputs = [
            {Pi.QAPAIRS: answer_batch, Pi.NUM_ANSWERS: len(answer_batch)}
            for _, answer_batch in batch_info
        ]
        chains_w_inputs = [(chain, inputs)]

        batch_results = await ChainRunner().run_chains_w_status_updates(
            chains_w_inputs, self._put_prettification_status
        )

        retry_batches = []
        for i, ((original_idx, answer_batch), batch_result) in enumerate(
            zip(batch_info, batch_results)
        ):
            processed_answers = (
                batch_result.answers
                if hasattr(batch_result, Fn.ANSWERS)
                else []
            )

            if len(processed_answers) == len(answer_batch):
                result_map[original_idx] = processed_answers
            else:
                retry_batches.append((original_idx, answer_batch))

        if retry_batches:
            for original_idx, answer_batch in retry_batches:
                processed_batch = await self._recursive_process_batch(
                    answer_batch, chain
                )
                result_map[original_idx] = processed_batch

        all_pretty_answers = []
        for i in range(len(batched_answers)):
            all_pretty_answers.extend(result_map[i])

        return all_pretty_answers

    async def _recursive_process_batch(
        self, answer_batch: list[str], chain: RunnableSerializable
    ) -> list[str]:
        """
        Recursively process a batch by splitting when necessary.

        Args:
            answer_batch: Batch of answers to process
            chain: Chain to use for processing

        Returns:
            List of processed answers
        """
        # Base case: single answer
        if len(answer_batch) == 1:
            return await self._process_single_answer(answer_batch[0], chain)

        inputs = [
            {
                Pi.QAPAIRS: answer_batch,
                Pi.NUM_ANSWERS: len(answer_batch),
            }
        ]
        chains_w_inputs = [(chain, inputs)]

        results = await ChainRunner().run_chains_w_status_updates(
            chains_w_inputs, self._put_prettification_status
        )
        result = results[0]
        processed_answers = (
            result.answers if hasattr(result, Fn.ANSWERS) else []
        )

        if len(processed_answers) == len(answer_batch):
            return processed_answers

        mid = len(answer_batch) // 2
        first_half = answer_batch[:mid]
        second_half = answer_batch[mid:]

        first_task = self._recursive_process_batch(first_half, chain)
        second_task = self._recursive_process_batch(second_half, chain)

        first_results, second_results = await aio.gather(
            first_task, second_task
        )

        return first_results + second_results

    async def _process_single_answer(
        self, answer: str, chain: RunnableSerializable
    ) -> list[str]:
        """
        Process a single answer through the prettification chain.

        Args:
            answer: The raw answer text to process
            chain: The processing chain to apply

        Returns:
            List containing the processed answer string,
                or original if processing fails
        """
        inputs = [{Pi.QAPAIRS: [answer], Pi.NUM_ANSWERS: 1}]
        chains_w_inputs = [(chain, inputs)]

        results = await ChainRunner().run_chains_w_status_updates(
            chains_w_inputs, self._put_prettification_status
        )
        if results:
            result = results[0]
            if hasattr(result, Fn.ANSWERS) and (new_answer := result.answers):
                return new_answer

        logging.warning(f"Was not able to prettify node: {answer}")
        return [answer]

    async def _augment_wth_cit(
        self,
        concept_df: pd.DataFrame,
        flat_part_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Augment concept answers with citations from flat part DataFrame.

        This method:
        1. Explodes the FLAT_PART_ID column
            to handle multiple citations per node
        2. Merges with citation data from flat_part_df
        3. Aggregates citations back by node
        4. Updates answers with proper citation references
        5. Applies prettification to the answers

        Args:
            concept_df: DataFrame containing nodes to augment
            flat_part_df: DataFrame containing citation information

        Returns:
            DataFrame with augmented citations and prettified answers
        """
        if Col.CITATION in concept_df.columns:
            if (
                concept_df[Col.CITATION].isna().any()
                and concept_df[Col.FLAT_PART_ID].notna().all()
            ):
                concept_df.drop(Col.CITATION, axis=1, inplace=True)
                # Assume that it only happens for document deletion
                flat_part_df["new_index"] = flat_part_df[Col.FLAT_PART_ID] - 1
                flat_part_df.set_index("new_index", inplace=True)

        node_df_exploded = concept_df.explode(Col.FLAT_PART_ID).reset_index()
        merged_df = pd.merge(
            node_df_exploded,
            flat_part_df[[Col.FLAT_PART_ID, Col.CITATION]],
            on=Col.FLAT_PART_ID,
            how="left",
        )
        result_df = merged_df.groupby("index").agg({Col.CITATION: tuple})
        concept_df = pd.merge(
            concept_df,
            result_df,
            left_index=True,
            right_index=True,
            how="left",
        )

        concept_df[Col.ANSWER] = concept_df.apply(
            self._replace_index_with_chunk_id,
            axis=1,
            args=(flat_part_df[Col.CITATION],),
        )

        synth_answers = concept_df.apply(
            self._repr_question_w_answer, axis=1
        ).tolist()
        batched_answers = self._split_list_into_batches(synth_answers)
        logging.info("Prettification")
        chain = ChainCreator.choose_chain(Ct.PRETTIFIER)
        pretty_answers = await self.process_answers_with_retry(
            batched_answers, chain
        )
        concept_df[Col.ANSWER_STR] = pretty_answers
        return concept_df

    @staticmethod
    def _det_max_cluster_size(
        chunk_df: pd.DataFrame,
        # num_chunks: int
    ) -> int:
        chunk_concept_ids = chunk_df[Col.CONCEPT_IDS].dropna()
        max_num_chunk_concepts = chunk_concept_ids.apply(len).max()

        if max_num_chunk_concepts <= 10:
            # if max_num_chunk_concepts < num_chunks <= 10:
            return 5
        return max_num_chunk_concepts

    async def _cluster_concepts(
        self,
        concept_df: pd.DataFrame,
        relation_df: pd.DataFrame,
        max_cluster_size: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logging.info("Creating concept cluster hierarchy.")

        (
            concept_df,
            relation_df,
            root_df,
        ) = await self._cluster_and_synthesize(
            concept_df,
            relation_df,
            max_cluster_size,
        )
        num_concepts = len(concept_df)
        num_relations = len(relation_df)

        logging.info(
            f"Created hierarchy with {num_concepts} concepts "
            f"and {num_relations} relationships"
        )
        return concept_df, relation_df, root_df

    async def _create_concept_hierarchy(
        self,
        chunk_df: pd.DataFrame,
        concept_df: pd.DataFrame,
        relation_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        num_chunks = len(chunk_df)
        is_single_chunk = num_chunks == 1

        num_concepts = len(concept_df)
        is_num_concepts_small = num_concepts < 4

        is_graph_trivial = is_single_chunk or is_num_concepts_small
        if not is_graph_trivial:
            max_cluster_size = self._det_max_cluster_size(
                chunk_df,
                # num_chunks
            )
            return await self._cluster_concepts(
                concept_df, relation_df, max_cluster_size
            )

        logging.info(
            "Skipping hierarchy creation for small graph "
            f"({num_concepts} concepts)"
        )
        concept_df[Col.LVL] = 1
        root_df = None
        return concept_df, relation_df, root_df

    async def format_concept_df(
        self, concept_df: pd.DataFrame, flat_part_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create the final concept DataFrame
        by filtering active concepts and adding citations.

        Args:
            concept_df: DataFrame containing concepts
            flat_part_df: DataFrame containing citation information

        Returns:
            Final concept DataFrame with active concepts, citations,
                and embeddings
        """
        logging.info("Creating final concept DataFrame")

        is_active_concepts = concept_df[Col.IS_ACTIVE_CONCEPT]
        active_mask = is_active_concepts == Col.ACTIVE_CONCEPT_TRUE_VAL

        concept_df = concept_df.loc[active_mask]
        concept_df = await self._augment_wth_cit(concept_df, flat_part_df)
        return self.embed_active_concepts(concept_df)

    async def wrap_up(
        self,
        concept_df: pd.DataFrame,
        relation_df: pd.DataFrame,
        root_index: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, int]:
        """
        Perform final wrap-up operations and return results.

        Args:
            concept_df: Final node DataFrame
            relation_df: Final relationship DataFrame
            root_index: Index of the root node

        Returns:
            Tuple of (concept_df, relation_df, root_index)
        """
        llm_cost_handler = cur_llm_cost_handler.get()
        logging.info(f"=Postprocessing LLM costs:\n{llm_cost_handler}")
        logging.info(
            f"Final Mind Map contains {len(concept_df)} nodes "
            f"and {len(relation_df)} relationships"
        )
        await self.queue.put(None)
        return concept_df, relation_df, root_index

    async def process_concepts(
        self,
        concept_df: pd.DataFrame,
        relation_df: pd.DataFrame,
        chunk_df: pd.DataFrame,
        flat_part_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, int]:
        """
        Process concepts to create a hierarchical mind map:
            1. Prepare concept data
            2. Create a hierarchical cluster structure (if needed)
                depends on the number of concepts
            3. Define a root concept
            4. Resolve concepts with duplicate names
            5. Create the final concept DataFrame with citations

        Args:
            concept_df: DataFrame containing concepts
            relation_df: DataFrame containing relations between concepts
            chunk_df: DataFrame containing document chunks
            flat_part_df: DataFrame containing flat part ids
                with citation information

        Returns:
            Tuple of (final_concept_df, final_relation_df, root_index)
        """
        logging.info("Starting concept processing pipeline")

        concept_df = self._prep_concept_df(concept_df)

        (
            concept_df,
            relation_df,
            root_df,
        ) = await self._create_concept_hierarchy(
            chunk_df,
            concept_df,
            relation_df,
        )

        concept_df, relation_df, root_index = await self.define_root(
            concept_df,
            relation_df,
            root_df,
        )

        concept_df, relation_df, root_index = await self.dedup_concepts(
            concept_df, relation_df, root_index
        )

        concept_df = await self.format_concept_df(concept_df, flat_part_df)

        return await self.wrap_up(concept_df, relation_df, root_index)

    async def process_concepts_with_retry(
        self, batched_answers: list[list[str]], chain: RunnableSerializable
    ) -> list:
        """
        Process all answer batches
        with automatic retry for failed batches.

        Args:
            batched_answers: List of batches,
                where each batch is a list of answers
            chain: Chain to use for processing

        Returns:
            List of processed answers
        """
        batch_info = [(i, batch) for i, batch in enumerate(batched_answers)]
        result_map = {}

        inputs = [
            {Pi.QAPAIRS: answer_batch, Pi.NUM_ANSWERS: len(answer_batch)}
            for _, answer_batch in batch_info
        ]
        chains_w_inputs = [(chain, inputs)]

        batch_results = await ChainRunner().run_chains_w_status_updates(
            chains_w_inputs, self._put_prettification_status
        )

        retry_batches = []
        for i, ((original_idx, answer_batch), batch_result) in enumerate(
            zip(batch_info, batch_results)
        ):
            processed_concepts = (
                batch_result.pretty_concepts
                if hasattr(batch_result, Fn.PRETTY_CONCEPTS)
                else []
            )

            if len(processed_concepts) == len(answer_batch):
                result_map[original_idx] = processed_concepts
            else:
                retry_batches.append((original_idx, answer_batch))

        if retry_batches:
            for original_idx, answer_batch in retry_batches:
                processed_batch = await self._recursive_process_batch(
                    answer_batch, chain
                )
                result_map[original_idx] = processed_batch

        all_pretty_answers = []
        for i in range(len(batched_answers)):
            all_pretty_answers.extend(result_map[i])

        return all_pretty_answers

    async def process_del_changes(
        self, concept_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Process only the rows that were modified during filtering.

        Parameters:
         concept_df: DataFrame with answers and modification flags

        Returns:
         DataFrame with updated answer strings for modified rows
        """
        modified_rows = concept_df[concept_df[Col.MODIFIED] == 1]
        if len(modified_rows) > 0:
            # Only process modified rows
            synth_answers = modified_rows.apply(
                self._repr_question_w_answer, axis=1
            ).tolist()

            batched_answers = self._split_list_into_batches(synth_answers)
            logging.info("Concept Prettification")

            chain = ChainCreator.choose_chain(Ct.CONCEPT_PRETTIFIER)
            pretty_concepts = await self.process_concepts_with_retry(
                batched_answers, chain
            )

            # Update only the modified rows in the result DataFrame
            modified_indices = modified_rows.index
            for i, idx in enumerate(modified_indices):
                concept_df.at[idx, Col.NAME] = pretty_concepts[i].name
                concept_df.at[idx, Col.QUESTION] = pretty_concepts[i].question
                concept_df.at[idx, Col.ANSWER_STR] = pretty_concepts[i].answer
        else:
            logging.info("No modified answers to prettify")

        return concept_df


class AddConceptProcessor(ConceptProcessor):
    @staticmethod
    def _replace_index_with_chunk_id(
        row: pd.Series,
        citation_df: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """
        Replace indices in answer citations with chunk IDs.

        Args:
            row: DataFrame row containing answers
            citation_df: DataFrame with 2 columns where first column contains indices
                        and second column contains chunk IDs

        Returns:
            List of answer parts with updated citations
        """
        new_annotated_answer = []
        # Get the column names for easier reference
        index_col = citation_df.columns[0]
        chunk_id_col = citation_df.columns[1]

        for answer in row[Col.ANSWER]:
            if isinstance(answer, dict):
                text = answer[FACT]
                if len(answer) > 1:
                    chunk_ids = set()
                    indices = answer[SOURCE_IDS]
                    if indices is not None and isinstance(indices, int):
                        # Look up the index in the first column and get chunk ID from second column
                        matching_rows = citation_df[
                            citation_df[index_col] == indices
                        ]
                        if not matching_rows.empty:
                            id_value = matching_rows.iloc[0][chunk_id_col]
                            if isinstance(id_value, list):
                                chunk_ids.update(id_value)
                            else:
                                chunk_ids.add(id_value)
                    elif indices is not None and isinstance(indices, list):
                        for index in indices:
                            index = int(index)
                            # Look up the index in the first column and get chunk ID from second column
                            matching_rows = citation_df[
                                citation_df[index_col] == index
                            ]
                            if not matching_rows.empty:
                                id_value = matching_rows.iloc[0][chunk_id_col]
                                if isinstance(id_value, list):
                                    chunk_ids.update(id_value)
                                else:
                                    chunk_ids.add(id_value)
                    new_annotated_answer.append(
                        {
                            FACT: text,
                            SOURCE_IDS: sorted(tuple(chunk_ids)),
                        }
                    )
                else:
                    new_annotated_answer.append([text, []])
            elif isinstance(answer, SynthesizedAnswerPart):
                text = getattr(answer, Fn.SYNTH_FACT)
                chunk_ids = set()
                indices = getattr(answer, Fn.CITATIONS)
                if indices is not None:
                    if isinstance(indices, int):
                        # Look up the index in the first column and get chunk ID from second column
                        matching_rows = citation_df[
                            citation_df[index_col] == indices
                        ]
                        if not matching_rows.empty:
                            id_value = matching_rows.iloc[0][chunk_id_col]
                            if isinstance(id_value, list):
                                chunk_ids.update(id_value)
                            else:
                                chunk_ids.add(id_value)
                    elif isinstance(indices, list):
                        for index in indices:
                            index = int(index)
                            # Look up the index in the first column and get chunk ID from second column
                            matching_rows = citation_df[
                                citation_df[index_col] == index
                            ]
                            if not matching_rows.empty:
                                id_value = matching_rows.iloc[0][chunk_id_col]
                                if isinstance(id_value, list):
                                    chunk_ids.update(id_value)
                                else:
                                    chunk_ids.add(id_value)
                    new_annotated_answer.append(
                        {
                            FACT: text,
                            SOURCE_IDS: sorted(tuple(chunk_ids)),
                        }
                    )
                else:
                    new_annotated_answer.append([text, []])
        return new_annotated_answer

    @staticmethod
    def _replace_chunk_id_with_index(
        row: pd.Series,
        citation_df: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """
        Replace chunk IDs in answer citations with flat part indices.

        Args:
            row: DataFrame row containing answers with chunk IDs
            citation_df: DataFrame with 2 columns where first column contains indices
                        and second column contains chunk IDs

        Returns:
            List of answer parts with citations converted back to indices
        """
        new_annotated_answer = []
        # Get the column names for easier reference
        index_col = citation_df.columns[0]  # flat_part_id column
        chunk_id_col = citation_df.columns[1]  # citation column

        for answer in row[Col.ANSWER]:
            if isinstance(answer, dict):
                text = answer[FACT]
                if SOURCE_IDS in answer:
                    chunk_ids = answer[SOURCE_IDS]
                    # If chunk_ids is None or empty, preserve the original answer
                    if not chunk_ids:
                        new_annotated_answer.append(answer)
                        continue

                    indices = set()
                    if isinstance(chunk_ids, (str, int)):
                        # Look up the chunk ID and get the corresponding index
                        matching_rows = citation_df[
                            citation_df[chunk_id_col] == chunk_ids
                        ]
                        if not matching_rows.empty:
                            for _, match_row in matching_rows.iterrows():
                                id_value = match_row[index_col]
                                indices.add(id_value)
                        else:
                            # If no matching row found, preserve the original chunk ID
                            new_annotated_answer.append(answer)
                            continue
                    elif isinstance(chunk_ids, list):
                        found_all = True
                        for chunk_id in chunk_ids:
                            # Look up the chunk ID and get the corresponding index
                            matching_rows = citation_df[
                                citation_df[chunk_id_col] == chunk_id
                            ]
                            if not matching_rows.empty:
                                for _, match_row in matching_rows.iterrows():
                                    id_value = match_row[index_col]
                                    indices.add(id_value)
                            else:
                                # If any chunk ID doesn't have a matching index, preserve original
                                found_all = False

                        if not found_all:
                            new_annotated_answer.append(answer)
                            continue

                    new_annotated_answer.append(
                        {
                            FACT: text,
                            SOURCE_IDS: sorted(list(indices)),
                        }
                    )
                else:
                    # Preserve the original answer if it doesn't have SOURCE_IDS
                    new_annotated_answer.append(answer)
            elif isinstance(answer, SynthesizedAnswerPart):
                text = getattr(answer, Fn.SYNTH_FACT)
                chunk_ids = getattr(answer, Fn.CITATIONS, None)

                # If chunk_ids is None or empty, preserve the original answer part
                if not chunk_ids:
                    new_annotated_answer.append(answer)
                    continue

                indices = set()
                convert_successful = True

                if isinstance(chunk_ids, (str, int)):
                    # Look up the chunk ID and get the corresponding index
                    matching_rows = citation_df[
                        citation_df[chunk_id_col] == chunk_ids
                    ]
                    if not matching_rows.empty:
                        for _, match_row in matching_rows.iterrows():
                            id_value = match_row[index_col]
                            indices.add(id_value)
                    else:
                        # If no matching row found, preserve the original
                        convert_successful = False
                elif isinstance(chunk_ids, list):
                    for chunk_id in chunk_ids:
                        # Look up the chunk ID and get the corresponding index
                        matching_rows = citation_df[
                            citation_df[chunk_id_col] == chunk_id
                        ]
                        if not matching_rows.empty:
                            for _, match_row in matching_rows.iterrows():
                                id_value = match_row[index_col]
                                indices.add(id_value)
                        else:
                            # If any chunk ID doesn't match, preserve original
                            convert_successful = False
                            break

                if convert_successful:
                    new_annotated_answer.append(
                        {
                            FACT: text,
                            SOURCE_IDS: sorted(list(indices)),
                        }
                    )
                else:
                    # Preserve original when conversion isn't successful
                    new_annotated_answer.append(answer)
            else:
                # Preserve any other types unchanged
                new_annotated_answer.append(answer)

        return new_annotated_answer

    @classmethod
    def _create_lvl_clusters(
        cls,
        embedding_matrix: NDArray,
        concept_indices: NDArray,
        max_cluster_size: int,
    ) -> tuple[int, list[list[int]]]:
        """
        Create hierarchical level clusters based on embedding matrix.

        Args:
            embedding_matrix: Matrix of embeddings for clustering
            concept_indices: Indices of concepts
                corresponding to embeddings
            max_cluster_size: Maximum allowed size for any cluster

        Returns:
            Tuple of (level_multiplier, combined_clusters)
            where combined_clusters
            is a list of lists containing indices grouped into clusters
        """
        max_cluster_size = max(max_cluster_size, 10)
        num_lvl_qapair = len(concept_indices)
        estim_num_clusters = num_lvl_qapair / max_cluster_size
        quarter_estim_num_clusters = estim_num_clusters / 4
        lvl_mult_float = math.log(quarter_estim_num_clusters, 3)
        lvl_mult = max(math.ceil(lvl_mult_float), 0)

        while True:
            max_num_lvl_clusters = 4 * 3**lvl_mult
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
                lvl_mult += 1

    async def _cluster_and_synthesize(
        self,
        concept_df: pd.DataFrame,
        relation_df: pd.DataFrame,
        max_chunk_size: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Synthesize hierarchical clusters of concepts.

        Iteratively creates levels of clusters until a root level with
        manageable number of concepts is reached.

        Args:
            concept_df: DataFrame containing concepts
            relation_df: DataFrame containing relationships between concepts
            max_chunk_size: Maximum number of QA pairs in a cluster

        Returns:
            Tuple of (concept_df, relation_df, root_df)
        """

        are_sub_roots_waisted = False
        is_root_waisted = False
        max_old_lvl = concept_df[Col.LVL].max()
        lvl = 1
        lvl_concept_df = concept_df[concept_df["new"] == 1].copy()
        old_concept_df = concept_df[concept_df["new"] == 0].copy()

        while True:
            if not is_root_waisted:
                lvl_old_concept_df = old_concept_df[
                    old_concept_df[Col.LVL] == lvl
                ]
                if lvl_old_concept_df.empty and not are_sub_roots_waisted:
                    lvl_old_concept_df = old_concept_df[
                        (old_concept_df[Col.LVL] == -1)
                        & (old_concept_df[Col.CLUSTER_ID] != -1)
                    ]
                    are_sub_roots_waisted = True
                if lvl_old_concept_df.empty:
                    lvl_old_concept_df = old_concept_df[
                        (old_concept_df[Col.LVL] == -1)
                        & (old_concept_df[Col.CLUSTER_ID] == 1)
                    ]
                    is_root_waisted = True
                lvl_concept_df = pd.concat([lvl_old_concept_df, lvl_concept_df])

            lvl_concept_df[Col.LVL] = lvl
            lvl_concept_ids = lvl_concept_df.index

            lvl_concept_df = self.embed_active_concepts(lvl_concept_df)
            embedding_matrix = np.stack(lvl_concept_df[Col.EMBEDDING])
            lvl_qapair_indices_arr = lvl_concept_ids.to_numpy()

            logging.info(f"Clusters creation. Details: Level {lvl}")
            lvl_mult, clusters = self._create_lvl_clusters(
                embedding_matrix,
                lvl_qapair_indices_arr,
                max_chunk_size,
            )
            qapair_to_cluster = {
                qapair_id: cluster_id
                for cluster_id, cluster in enumerate(clusters)
                for qapair_id in cluster
            }
            cluster_ids = lvl_concept_ids.map(qapair_to_cluster)
            lvl_concept_df[Col.CLUSTER_ID] = cluster_ids

            status_msg = f"Synthesizing clusters"
            logging.info(f"{status_msg}. Details: Level {lvl}")
            synthesized_clusters = await self._synth_clusters(
                lvl_concept_df, clusters, lvl
            )

            (
                lvl_concept_df,
                next_lvl_concept_df,
                relation_df,
            ) = await self._process_synth_clusters(
                synthesized_clusters,
                lvl_concept_df,
                relation_df,
                lvl,
                concept_df.index.max(),
            )

            level_text = "level was" if lvl == 1 else "levels were"
            status_msg = f"{lvl} hierarchy {level_text} created"
            logging.info(status_msg)

            if lvl > 1:
                concept_df = pd.concat([concept_df, lvl_concept_df])
            else:
                concept_df = lvl_concept_df

            # Root is created for each cluster
            is_num_root_fit = len(clusters) <= max_chunk_size
            if not (lvl_mult == 0 or is_num_root_fit):
                lvl += 1
                lvl_concept_df = next_lvl_concept_df
            else:
                next_level_old_concept_df = None
                if not are_sub_roots_waisted:
                    next_level_old_concept_df = old_concept_df[
                        (old_concept_df[Col.LVL] == -1)
                        & (old_concept_df[Col.CLUSTER_ID] != -1)
                    ]
                elif not is_root_waisted:
                    next_level_old_concept_df = old_concept_df[
                        (old_concept_df[Col.LVL] == -1)
                        & (old_concept_df[Col.CLUSTER_ID] == -1)
                    ]
                    is_root_waisted = True
                elif max_old_lvl > lvl:
                    # TODO: temp workaround (bad)
                    next_level_old_concept_df = old_concept_df[
                        (
                            (old_concept_df[Col.LVL] == -1)
                            & (old_concept_df[Col.CLUSTER_ID] != -1)
                        )
                        | (
                            (old_concept_df[Col.LVL] > lvl)
                            & (old_concept_df[Col.CLUSTER_ID] == -1)
                        )
                    ]
                self.is_root_waisted = is_root_waisted
                if next_level_old_concept_df is not None:
                    return (
                        concept_df,
                        relation_df,
                        pd.concat(
                            [next_lvl_concept_df, next_level_old_concept_df]
                        ),
                    )
                else:
                    return (
                        concept_df,
                        relation_df,
                        next_lvl_concept_df,
                    )

    async def _prep_roots_w_old_root(
        self,
        root_df: pd.DataFrame,
        root_synth_results: RootClusterSynthesisResult,
        concept_name_to_id: dict[str, int],
        old_root_concept: pd.DataFrame,
    ) -> list[dict]:
        """
        Prepare root concepts from synthesis results.

        Args:
            root_df: DataFrame containing root-level concepts
            root_synth_results: Results from root synthesis
            concept_name_to_id: Mapping from concept names to IDs

        Returns:
            List of prepared root concept dictionaries
        """
        main_root = [getattr(root_synth_results, Fn.ROOT_CONCEPT)]
        if not self.is_root_waisted:
            old_root_info = self._repr_concepts(old_root_concept)
            inputs = {
                "old_concept": old_root_info,
                "new_concept": str(main_root),
            }
            chain = ChainCreator.choose_chain(Ct.NEW_ROOT)
            result = await ChainRunner().run_chain(chain, inputs)
            main_root = [getattr(result, Fn.ROOT_CONCEPT)]

        roots = []
        for sub_root in (
            getattr(root_synth_results, Fn.SYNTH_CONCEPTS) + main_root
        ):
            qapair = dict(sub_root)
            qapair = self._rename_concept_fields(qapair, concept_name_to_id)
            roots.append(
                {
                    **qapair,
                    Col.CLUSTER_ID: -1,
                    Col.LVL: root_df[Col.LVL].max(),
                }
            )

        roots[-1][Col.LVL] = -1
        return roots

    async def _define_root_w_old_root(
        self,
        other_concept_df: pd.DataFrame,
        rel_df: pd.DataFrame,
        old_root_concept: pd.DataFrame,
        root_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, int]:
        """
        Define the root concept and its relationships to other concepts.

        Args:
            other_concept_df: DataFrame containing non-root concepts
            rel_df: DataFrame containing relationships
            root_df: Optional DataFrame containing root-level QA pairs

        Returns:
            Tuple of (updated_concept_df, updated_rel_df, root_index)
        """
        if root_df is not None:
            concept_df = pd.concat([other_concept_df, root_df])
        else:
            root_df = concept_df = other_concept_df

        root_synth_results = await self._synthesize_roots(root_df)

        root_ids = root_df.index
        root_names = root_df[Col.NAME]
        qapair_name_to_id = pd.Series(root_ids, index=root_names).to_dict()
        sub_roots = await self._prep_roots_w_old_root(
            root_df, root_synth_results, qapair_name_to_id, old_root_concept
        )
        concept_df = self._add_concepts(concept_df, sub_roots)
        concept_df = self._make_unique_names(concept_df)

        root_index = concept_df.index[-1]
        synth_answers = concept_df.apply(self._repr_answer, axis=1)
        concept_df[Col.ANSWER_STR] = synth_answers

        root_rels = []
        cluster_id_mask = concept_df[Col.CLUSTER_ID] == -1
        root_cluster_index = concept_df.loc[cluster_id_mask].index
        num_rels = min(len(root_cluster_index), 4)
        for idx in root_cluster_index[:num_rels]:
            root_rels.append(
                {
                    Col.ORIGIN_CONCEPT_ID: root_index,
                    Col.TARGET_CONCEPT_ID: idx,
                    Col.WEIGHT: 4.0,
                }
            )
        root_rel_df = pd.DataFrame(root_rels)

        rel_df = self._update_rel_df(rel_df, root_rel_df, concept_df)

        rel_df[Col.TYPE] = Col.RELATED_TYPE_VAL

        return concept_df, rel_df, root_index

    async def _augment_wth_cit(
        self,
        concept_df: pd.DataFrame,
        flat_part_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Augment concept answers with citations from flat part DataFrame.

        This method:
        1. Explodes the FLAT_PART_ID column
            to handle multiple citations per node
        2. Merges with citation data from flat_part_df
        3. Aggregates citations back by node
        4. Updates answers with proper citation references
        5. Applies prettification to the answers

        Args:
            concept_df: DataFrame containing nodes to augment
            flat_part_df: DataFrame containing citation information

        Returns:
            DataFrame with augmented citations and prettified answers
        """
        node_df_exploded = concept_df.explode(Col.FLAT_PART_ID).reset_index()
        merged_df = pd.merge(
            node_df_exploded,
            flat_part_df[[Col.FLAT_PART_ID, Col.CITATION]],
            on=Col.FLAT_PART_ID,
            how="left",
        )
        result_df = merged_df.groupby("index").agg({Col.CITATION: tuple})
        concept_df = pd.merge(
            concept_df,
            result_df,
            left_index=True,
            right_index=True,
            how="left",
        )

        concept_df[Col.ANSWER] = concept_df.apply(
            self._replace_index_with_chunk_id,
            axis=1,
            args=(flat_part_df[[Col.FLAT_PART_ID, Col.CITATION]],),
        )

        synth_answers = concept_df.apply(
            self._repr_question_w_answer, axis=1
        ).tolist()
        batched_answers = self._split_list_into_batches(synth_answers)
        logging.info("Prettification")
        chain = ChainCreator.choose_chain(Ct.PRETTIFIER)
        pretty_answers = await self.process_answers_with_retry(
            batched_answers, chain
        )
        concept_df[Col.ANSWER_STR] = pretty_answers
        return concept_df

    async def _cluster_concepts(
        self,
        concept_df: pd.DataFrame,
        relation_df: pd.DataFrame,
        max_cluster_size: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logging.info("Creating concept cluster hierarchy.")

        (
            concept_df,
            relation_df,
            root_df,
        ) = await self._cluster_and_synthesize(
            concept_df,
            relation_df,
            max_cluster_size,
        )
        num_concepts = len(concept_df)
        num_relations = len(relation_df)

        logging.info(
            f"Created hierarchy with {num_concepts} concepts "
            f"and {num_relations} relationships"
        )
        return concept_df, relation_df, root_df

    async def _create_concept_hierarchy_w_old_chunks(
        self,
        chunk_df: pd.DataFrame,
        concept_df: pd.DataFrame,
        relation_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        num_concepts = len(concept_df[concept_df["new"] == 1])
        is_num_concepts_small = num_concepts < 4

        # num_chunks = len(chunk_df)

        is_graph_trivial = is_num_concepts_small
        if not is_graph_trivial:
            max_cluster_size = self._det_max_cluster_size(
                chunk_df,
                # num_chunks
            )
            return await self._cluster_concepts(
                concept_df, relation_df, max_cluster_size
            )

        logging.info(
            "Skipping hierarchy creation for small graph "
            f"({num_concepts} concepts)"
        )
        self.is_root_waisted = False
        concept_df[Col.LVL] = 1
        root_df = None
        return concept_df, relation_df, root_df

    async def wrap_up(
        self,
        concept_df: pd.DataFrame,
        relation_df: pd.DataFrame,
        root_index: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, int]:
        """
        Perform final wrap-up operations and return results.

        Args:
            concept_df: Final node DataFrame
            relation_df: Final relationship DataFrame
            root_index: Index of the root node

        Returns:
            Tuple of (concept_df, relation_df, root_index)
        """
        llm_cost_handler = cur_llm_cost_handler.get()
        logging.info(f"=Postprocessing LLM costs:\n{llm_cost_handler}")
        logging.info(
            f"Final Mind Map contains {len(concept_df)} nodes "
            f"and {len(relation_df)} relationships"
        )
        valid_indices = set(concept_df.index)
        relation_df = relation_df[
            relation_df[Col.ORIGIN_CONCEPT_ID].isin(valid_indices)
            & relation_df[Col.TARGET_CONCEPT_ID].isin(valid_indices)
        ]
        await self.queue.put(None)
        return concept_df, relation_df, root_index

    async def process_concepts(
        self,
        concept_df: pd.DataFrame,
        relation_df: pd.DataFrame,
        chunk_df: pd.DataFrame,
        flat_part_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, int]:
        """
        Process concepts to create a hierarchical mind map:
            1. Prepare concept data
            2. Create a hierarchical cluster structure (if needed)
                depends on the number of concepts
            3. Define a root concept
            4. Resolve concepts with duplicate names
            5. Create the final concept DataFrame with citations

        Args:
            concept_df: DataFrame containing concepts
            relation_df: DataFrame containing relations between concepts
            chunk_df: DataFrame containing document chunks
            flat_part_df: DataFrame containing flat part ids
                with citation information

        Returns:
            Tuple of (final_concept_df, final_relation_df, root_index)
        """
        logging.info("Starting concept processing pipeline")
        concept_df[Col.EMBEDDING] = np.nan
        concept_df[Col.IS_ACTIVE_CONCEPT] = Col.ACTIVE_CONCEPT_TRUE_VAL
        mapping = {}
        for _, row in concept_df.iterrows():
            # Only process rows where flat_part_id has a single element
            if len(row["flat_part_id"]) == 1:
                if pd.notna(row["citation"]):
                    mapping[row["flat_part_id"][0]] = row["citation"][0]

        for _, row in concept_df.iterrows():
            if len(row["flat_part_id"]) > 1:
                # Extract relationships from compound entries
                for i, part_id in enumerate(row["flat_part_id"]):
                    if part_id not in mapping:
                        # Only add if this atomic relationship isn't already known
                        if isinstance(row["citation"], tuple):
                            mapping[part_id] = row["citation"][i]

        # mapping = {tuple(row[Col.FLAT_PART_ID]): row[Col.CITATION] for _, row in concept_df.iterrows()}
        concept_df.drop(columns=[Col.CITATION], inplace=True)
        new_concept_df = concept_df[concept_df["new"] == 1]
        concept_df[concept_df["new"] == 1] = self._prep_concept_df(
            new_concept_df
        )

        def add_to_existing_df(df, data_tuples):
            # Create temporary dataframe with filtered data
            temp_data = []

            for fpi, cit in data_tuples.items():
                if pd.notna(cit):
                    temp_data.append({Col.FLAT_PART_ID: fpi, Col.CITATION: cit})

            temp_df = pd.DataFrame(temp_data)

            # Append to existing dataframe
            return pd.concat([df, temp_df], ignore_index=True)

        flat_part_df = add_to_existing_df(flat_part_df, mapping)

        concept_df[Col.ANSWER] = concept_df.apply(
            self._replace_chunk_id_with_index,
            axis=1,
            args=(flat_part_df[[Col.FLAT_PART_ID, Col.CITATION]],),
        )

        old_root_concept = concept_df[
            (concept_df[Col.LVL] == -1) & (concept_df[Col.CLUSTER_ID] == -1)
        ]

        (
            concept_df,
            relation_df,
            root_df,
        ) = await self._create_concept_hierarchy_w_old_chunks(
            chunk_df,
            concept_df,
            relation_df,
        )

        (
            concept_df,
            relation_df,
            root_index,
        ) = await self._define_root_w_old_root(
            concept_df,
            relation_df,
            old_root_concept,
            root_df,
        )

        concept_df, relation_df, root_index = await self.dedup_concepts(
            concept_df, relation_df, root_index
        )

        concept_df = await self.format_concept_df(concept_df, flat_part_df)

        return await self.wrap_up(concept_df, relation_df, root_index)
