import json
import textwrap
from typing import Any, Optional

import pandas as pd

from generator.chainer import ChainCreator, ChainRunner
from generator.chainer.response_formats import RootDeduplicationResult
from generator.chainer.utils.constants import ChainTypes as Ct
from generator.common.constants import ColVals
from generator.common.constants import DataFrameCols as Col
from generator.common.constants import FieldNames as Fn
from generator.common.logger import logging
from generator.core.structs import RawMindMapData
from generator.core.utils.constants import Pi

from . import utils


class ConceptDeduplicator:
    """
    Handles the merging and renaming of duplicate concepts.

    This class provides two main strategies for deduplication:
    1. `deduplicate`: A structural merge of concepts sharing a name.
    2. `make_names_unique`: A renaming of concepts to ensure unique
    labels.
    """

    def __init__(self, chain_creator: ChainCreator):
        """Initializes the deduplicator with a reusable ChainRunner."""
        self.chain_creator = chain_creator
        self.chain_runner = ChainRunner()

    async def deduplicate(self, data: RawMindMapData) -> RawMindMapData:
        """
        Merges concepts that share the same name using an LLM.

        This method performs a deep, structural deduplication. It first
        reveals true duplicates by stripping name suffixes (e.g., `_1`).
        It then iteratively finds groups of active concepts with
        identical names and uses an LLM to synthesize a single, new
        concept that merges their information.

        The original duplicate concepts are deactivated, and all
        relationships are updated to point to the new, merged concept.
        The process repeats for a fixed number of iterations or until no
        duplicates remain.

        Args:
            data: The raw mind map data containing concepts and
            relations.

        Returns:
            The mind map data with duplicate concepts merged.
        """
        concept_df = data.concept_df
        relation_df = data.relation_df
        root_id = data.root_index

        # Remove temporary suffixes to reveal true duplicates.
        concept_df[Col.NAME] = concept_df[Col.NAME].str.replace(
            r"_.*", "", regex=True
        )

        for _ in range(10):  # Max 10 iterations of deduplication.
            active_df = concept_df[
                concept_df[Col.IS_ACTIVE_CONCEPT] == ColVals.TRUE_INT
            ]
            groups, group_indices, dup_indices, root_group_id = (
                self._get_dup_groups(active_df, root_id)
            )

            if not dup_indices:
                logging.info(
                    "Deduplication complete. No more duplicates found."
                )
                break

            max_cluster_ids, max_lvls = self._prepare_dedup_inputs(
                group_indices, concept_df, root_id
            )
            dedup_results, root_result = await self._run_deduplication_chains(
                groups, root_group_id
            )
            new_concepts, new_rels_by_name = self._process_dedup_results(
                dedup_results,
                root_result,
                root_group_id,
                max_cluster_ids,
                max_lvls,
            )

            if not new_concepts:
                continue

            # Add new concepts and get their assigned IDs.
            last_id = concept_df.index.max()
            concept_df = utils.add_concepts(concept_df, new_concepts)
            concept_df[Col.ANSWER_STR] = concept_df.apply(
                utils.repr_answer, axis=1
            )

            # Resolve relationship names to the new IDs.
            newly_added_mask = concept_df.index > last_id
            name_to_new_id = pd.Series(
                concept_df.index[newly_added_mask],
                index=concept_df.loc[newly_added_mask, Col.NAME],
            ).to_dict()

            new_rel_rows = [
                {
                    Col.ORIGIN_CONCEPT_ID: name_to_new_id[rel["origin_name"]],
                    Col.TARGET_CONCEPT_ID: name_to_new_id[rel["target_name"]],
                    Col.TYPE: ColVals.RELATED,
                    Col.WEIGHT: 3.0,
                }
                for rel in new_rels_by_name
                if rel["origin_name"] in name_to_new_id
                and rel["target_name"] in name_to_new_id
            ]
            relation_df = utils.update_rel_df(
                relation_df, pd.DataFrame(new_rel_rows), concept_df
            )

        # Final cleanup: find the new root and filter out invalid relations.
        active_mask = concept_df[Col.IS_ACTIVE_CONCEPT] == ColVals.TRUE_INT
        root_mask = (concept_df[Col.LVL] == -1) & (
            concept_df[Col.CLUSTER_ID] == -1
        )
        new_root_index = int(
            concept_df.loc[root_mask & active_mask].index.values[0]
        )

        inactive_ids = concept_df[
            concept_df[Col.IS_ACTIVE_CONCEPT] == ColVals.FALSE_INT
        ].index
        invalid_rel_mask = relation_df[
            [Col.ORIGIN_CONCEPT_ID, Col.TARGET_CONCEPT_ID]
        ].isin(inactive_ids).any(axis=1) | (
            relation_df[Col.ORIGIN_CONCEPT_ID]
            == relation_df[Col.TARGET_CONCEPT_ID]
        )
        return RawMindMapData(
            concept_df=concept_df,
            relation_df=relation_df[~invalid_rel_mask],
            root_df=data.root_df,
            chunk_df=data.chunk_df,
            flat_part_df=data.flat_part_df,
            root_index=new_root_index,
        )

    async def make_names_unique(
        self, data: RawMindMapData, max_tries: int = 3, batch_size: int = 10
    ) -> RawMindMapData:
        """
        Iteratively renames concepts with duplicate names using an LLM.

        This method performs a "soft" deduplication. It identifies
        groups of concepts sharing the same name and uses an LLM to
        generate a new, more descriptive, and unique name for each
        concept based on its specific content. This does not merge
        concepts or change the graph structure.

        To improve performance, it processes concepts in batches using
        an asynchronous ChainRunner. The process repeats until all names
        are unique or a maximum number of tries is reached.

        Args:
            data: The raw mind map data.
            max_tries: The maximum number of renaming iterations.
            batch_size: The number of concepts to process in each LLM
                call.

        Returns:
            The mind map data with concept names updated to be unique.
        """
        concept_df = data.concept_df

        chain_runner = ChainRunner()
        df_processed = concept_df.copy()
        for i in range(max_tries):
            logging.info(
                f"--- Starting Deduplication Iteration: {i + 1}/{max_tries} ---"
            )

            name_counts = df_processed[Col.NAME].value_counts()
            duplicate_names = name_counts[name_counts > 1].index.tolist()

            if not duplicate_names:
                logging.info(
                    "Success! No duplicate names found. Process finished."
                )
                return RawMindMapData(
                    concept_df=df_processed,
                    relation_df=data.relation_df,
                    root_df=data.root_df,
                    chunk_df=data.chunk_df,
                    flat_part_df=data.flat_part_df,
                    root_index=data.root_index,
                )

            logging.warning(
                f"Found {len(duplicate_names)} groups of duplicates to "
                f"process: {duplicate_names}"
            )

            df_processed = await self._rename_single_pass_async(
                chain_runner=chain_runner,
                df_to_process=df_processed,
                duplicate_names=duplicate_names,
                batch_size=batch_size,
            )
            logging.info(f"--- Finished Iteration: {i + 1}/{max_tries} ---\n")

        # Final check after all iterations are complete.
        final_counts = df_processed[Col.NAME].value_counts()
        remaining_duplicates = final_counts[final_counts > 1].index.tolist()
        if remaining_duplicates:
            logging.error(
                f"Process finished after {max_tries} tries, but "
                f"{len(remaining_duplicates)} duplicates remain: "
                f"{remaining_duplicates}"
            )
        else:
            logging.info(
                "Success! All duplicates resolved within {max_tries} tries."
            )

        return RawMindMapData(
            concept_df=df_processed,
            relation_df=data.relation_df,
            root_df=data.root_df,
            chunk_df=data.chunk_df,
            flat_part_df=data.flat_part_df,
            root_index=data.root_index,
        )

    @classmethod
    def _get_dedup_qapairs(
        cls,
        dedup_results: list,
        max_cluster_ids: list[int],
        max_lvls: list[int],
    ) -> list[dict]:
        """
        Extracts concept data from LLM deduplication results.

        Args:
            dedup_results: A list of structured result objects from the
                deduplication LLM chain.
            max_cluster_ids: A list of cluster IDs to assign to the new
                merged concepts, aligned with `dedup_results`.
            max_lvls: A list of levels to assign to the new merged
                concepts, aligned with `dedup_results`.

        Returns:
            A list of new, merged concept dictionaries.
        """
        concepts = []
        for dedup_result, cluster_id, lvl in zip(
            dedup_results, max_cluster_ids, max_lvls
        ):
            # Use getattr for safe access to the result attribute.
            for concept in getattr(dedup_result, Fn.SYNTH_CONCEPTS, []):
                concept_dict = utils.rename_concept_fields(concept.model_dump())
                concepts.append(
                    {**concept_dict, Col.CLUSTER_ID: cluster_id, Col.LVL: lvl}
                )

        return concepts

    @classmethod
    def _get_dup_groups(
        cls, concept_df: pd.DataFrame, root_id: int
    ) -> (
        tuple[list[list[str]], list[list[int]], list[int], int | None]
        | tuple[None, None, None, None]
    ):
        """
        Identifies groups of concepts with duplicate names.

        Args:
            concept_df: DataFrame containing active concepts.
            root_id: The ID of the main root concept.

        Returns:
            A tuple containing:
            - A list of string representations for each duplicate group.
            - A list of index lists for each duplicate group.
            - A flat list of all duplicate indices.
            - The index of the group containing the root, or None.
            Returns (None, None, None, None) if no duplicates exist.
        """
        grouped_by_name = concept_df.groupby(Col.NAME)
        dup_groups = grouped_by_name.filter(lambda group: len(group) > 1)

        if dup_groups.empty:
            return None, None, None, None

        # Re-group the filtered DataFrame to work with duplicate groups only.
        grouped_dup = dup_groups.groupby(Col.NAME)
        group_indices = [group.index.tolist() for _, group in grouped_dup]
        all_dup_indices = dup_groups.index.tolist()

        concept_group_repr = [
            cls._repr_dup_group_as_list(group, root_id=root_id)
            for _, group in grouped_dup
        ]

        # Find which group of duplicates contains the main root concept.
        root_group_idx = next(
            (
                i
                for i, indices in enumerate(group_indices)
                if root_id in indices
            ),
            None,
        )

        return (
            concept_group_repr,
            group_indices,
            all_dup_indices,
            root_group_idx,
        )

    @classmethod
    def _prepare_dedup_inputs(
        cls,
        groups_of_dup_indices: list[list[int]],
        concept_df: pd.DataFrame,
        root_id: int,
    ) -> tuple[list[int], list[int]]:
        """
        Prepares cluster IDs and levels for new merged concepts.

        For each group of duplicates, it determines the `cluster_id` and
        `lvl` for the new merged concept. The rule is to take the
        maximum value from the concepts being merged, ensuring the new
        concept resides at the highest level of its constituents.

        Args:
            groups_of_dup_indices: A list of groups, where each group is
                a list of concept IDs to be merged.
            concept_df: The main DataFrame of all concepts.
            root_id: The ID of the main root concept.

        Returns:
            A tuple containing a list of max cluster IDs and a list of
            max levels, one for each group.
        """
        max_cluster_ids_per_group = []
        max_lvls_per_group = []

        for group in groups_of_dup_indices:
            # Collect cluster IDs and levels from all concepts in the group,
            # excluding the root concept itself if present.
            group_cluster_ids = [
                utils.get_max_value_or_negative_one(
                    pd.Series(concept_df.loc[dup_id, Col.CLUSTER_ID])
                )
                for dup_id in group
                if dup_id != root_id
            ]
            group_lvls = [
                utils.get_max_value_or_negative_one(
                    pd.Series(concept_df.loc[dup_id, Col.LVL])
                )
                for dup_id in group
                if dup_id != root_id
            ]

            max_cluster_ids_per_group.append(
                utils.get_max_value_or_negative_one(
                    pd.Series(group_cluster_ids)
                )
            )
            max_lvls_per_group.append(
                utils.get_max_value_or_negative_one(pd.Series(group_lvls))
            )

        return max_cluster_ids_per_group, max_lvls_per_group

    @classmethod
    def _process_dedup_results(
        cls,
        dedup_results: list[Any],
        root_ded_result: Optional[RootDeduplicationResult],
        root_group_id: int | None,
        max_cluster_ids: list[int],
        max_lvls: list[int],
    ) -> tuple[list[dict], list[dict]]:
        """
        Processes LLM results to get new concepts and relations.

        This function parses the structured output from both the
        standard and root deduplication chains, standardizing them into
        lists of new concept dictionaries and new relationship
        dictionaries.

        Args:
            dedup_results: List of standard deduplication results.
            root_ded_result: The special result for the root group.
            root_group_id: The original index of the root group.
            max_cluster_ids: List of cluster IDs for the new concepts.
            max_lvls: List of levels for the new concepts.

        Returns:
            A tuple containing the list of new concepts and the list of
            new relationships (by name, to be resolved later).
        """
        all_results = dedup_results
        if root_ded_result is not None and root_group_id is not None:
            # Re-insert the root result at its original position to align
            # with the max_cluster_ids and max_lvls lists.
            all_results.insert(root_group_id, root_ded_result)

        new_concepts = cls._get_dedup_qapairs(
            all_results, max_cluster_ids, max_lvls
        )
        new_relations = []

        # Extract the main root concept if it was created.
        if root_ded_result:
            root_concept = getattr(root_ded_result, Fn.ROOT_CONCEPT, None)
            if root_concept:
                root_dict = utils.rename_concept_fields(
                    root_concept.model_dump()
                )
                new_concepts.append(
                    {**root_dict, Col.CLUSTER_ID: -1, Col.LVL: -1}
                )

        # Extract relationships from all results.
        for result in all_results:
            for rel in getattr(result, "relations", []):
                new_relations.append(
                    {
                        # The names will be resolved to IDs later.
                        "origin_name": getattr(rel, Fn.ORIGIN_CONCEPT_NAME),
                        "target_name": getattr(rel, Fn.TARGET_CONCEPT_NAME),
                    }
                )
        return new_concepts, new_relations

    async def _rename_single_pass_async(
        self,
        chain_runner: ChainRunner,
        df_to_process: pd.DataFrame,
        duplicate_names: list[str],
        batch_size: int,
    ) -> pd.DataFrame:
        """
        Performs one round of renaming duplicate concepts via an LLM.

        This method orchestrates a single pass of the renaming process:
        1. Collects all concepts that have duplicate names.
        2. Batches them for efficient processing.
        3. Submits all batches to the ChainRunner for concurrent
        execution.
        4. Updates the DataFrame with the new, unique names from the
        results.

        Args:
            chain_runner: An instance of the ChainRunner for async
                calls.
            df_to_process: The DataFrame to modify.
            duplicate_names: A list of names that have duplicates.
            batch_size: The size of batches to send to the LLM.

        Returns:
            The DataFrame with names updated from this pass.
        """
        all_chain_inputs = []

        # Step 1: Collect all inputs from all groups and batches.
        for name in duplicate_names:
            group_df = df_to_process[df_to_process[Col.NAME] == name]
            for i in range(0, len(group_df), batch_size):
                batch_df = group_df.iloc[i : i + batch_size]

                concepts_to_rename = [
                    {
                        "concept_index": idx,
                        Col.QUESTION: row[Col.QUESTION],
                        Col.ANSWER_STR: row[Col.ANSWER_STR],
                    }
                    for idx, row in batch_df.iterrows()
                ]

                input_data = {
                    "original_name": name,
                    "concepts_json": json.dumps(concepts_to_rename, indent=2),
                }
                all_chain_inputs.append(input_data)

        if not all_chain_inputs:
            return df_to_process

        # Step 2: Dispatch all collected inputs to the ChainRunner.
        logging.info(
            f"Submitting {len(all_chain_inputs)} batches to the "
            "ChainRunner for concurrent processing."
        )
        chain = self.chain_creator.choose_chain(Ct.DEDUPLICATE_NAMES)
        results = await chain_runner.run_chain_on_batch(chain, all_chain_inputs)

        # Step 3: Update the DataFrame with the results.
        logging.info(
            "All batches processed. Updating DataFrame with new names."
        )
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"A batch failed with an exception: {result}")
                continue

            if result is None or not hasattr(result, "renamed_concepts"):
                logging.warning(
                    f"Skipping a null or malformed batch result: {result}"
                )
                continue

            for item in result.renamed_concepts:
                concept_index = getattr(item, "concept_index", None)
                new_name = getattr(item, "new_name", None)

                if concept_index is not None and new_name is not None:
                    df_to_process.loc[concept_index, Col.NAME] = new_name
                else:
                    logging.warning(
                        f"Skipping malformed item in result: {item}"
                    )

        return df_to_process

    @staticmethod
    def _repr_dup_group_as_list(group: pd.DataFrame, root_id: int) -> list[str]:
        """
        Creates a structured string representation of a duplicate group.

        This formats a group of concepts sharing the same name into a
        human-readable block for an LLM, clearly marking if one of the
        concepts is the designated root of the entire mind map.

        Args:
            group: DataFrame of duplicate entries for a single name.
            root_id: The ID of the designated main root concept.

        Returns:
            A list of structured string representations for the LLM.
        """
        representations = []
        for index, row in group.iterrows():
            is_root = index == root_id
            root_marker = " **[THIS IS THE ROOT CONCEPT]**" if is_root else ""

            concept_str = textwrap.dedent(
                f"""\
                Concept (ID: {index}){root_marker}
                --------------------
                Name: "{row[Col.NAME]}"
                Question: {row[Col.QUESTION]}
                Answer: {row[Col.ANSWER]}"""
            )
            representations.append(concept_str)
        return representations

    async def _run_deduplication_chains(
        self,
        groups_of_dupes: list[list[str]],
        root_group_id: int | None,
    ) -> tuple[list[Any], Optional[RootDeduplicationResult]]:
        """
        Runs the appropriate deduplication chains on concept groups.

        This method orchestrates the LLM calls for merging concepts. It
        separates the group containing the main root concept (if any)
        to be processed by a specialized, more careful `DEDUP_ROOT`
        chain, while all other groups are processed by the standard
        `DEDUP` chain.

        Args:
            groups_of_dupes: A list of string representations for each
                group of duplicate concepts.
            root_group_id: The index of the group containing the root,
                or None if no group contains the root.

        Returns:
            A tuple containing the list of standard deduplication
            results and the special result for the root group (if any).
        """
        root_ded_results = None
        if root_group_id is not None:
            root_group_repr = groups_of_dupes.pop(root_group_id)
            root_chain = self.chain_creator.choose_chain(Ct.DEDUP_ROOT)
            root_inputs = {Pi.CLUSTER_SYNTH: root_group_repr}
            root_ded_results = await ChainRunner().run_chain_w_retries(
                root_chain, root_inputs, retry_field_name=Fn.SOURCE_IDS
            )

        # Run the standard deduplication chain on all other groups.
        dedup_chain = self.chain_creator.choose_chain(Ct.DEDUP)
        inputs = [{Pi.CLUSTER_SYNTH: group} for group in groups_of_dupes]
        dedup_results = await ChainRunner().run_chain_on_batch(
            dedup_chain, inputs
        )

        return dedup_results, root_ded_results
