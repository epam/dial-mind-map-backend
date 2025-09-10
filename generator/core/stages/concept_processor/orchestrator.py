import asyncio as aio
from typing import Any

import numpy as np
import pandas as pd

from generator.chainer import ChainCreator, ChainRunner
from generator.chainer.response_formats import (
    RootClusterSynthesisResultProtocol,
)
from generator.chainer.utils.constants import ChainTypes as Ct
from generator.common.constants import ColVals
from generator.common.constants import DataFrameCols as Col
from generator.common.constants import FieldNames as Fn
from generator.common.context import cur_llm_cost_handler
from generator.common.logger import logging
from generator.core.structs import RawMindMapData
from generator.core.utils.constants import Pi

from . import utils
from .clusterer import AddConceptClusterer, ConceptClusterer
from .deduplicator import ConceptDeduplicator
from .formatter import ConceptFormatter


class ProcessingOrchestrator:
    """
    Orchestrates the multi-stage processing of concepts.

    This class defines the high-level pipeline for generating a complete
    and polished mind map from a raw, flat set of concepts. It calls
    specialized processors in a specific sequence to handle clustering,
    root definition, deduplication, and formatting.
    """

    def __init__(
        self,
        chain_creator: ChainCreator,
        queue: aio.Queue,
        style_instructions: dict[str, Any],
    ):
        """
        Initializes the orchestrator and its specialized processors.

        Args:
            queue: An asyncio queue for sending frontend status updates.
            style_instructions: A dictionary of instructions for the
                LLM to guide the final "prettification" style.
        """
        self.chain_creator = chain_creator

        self.clusterer = ConceptClusterer(chain_creator, queue)
        self.deduplicator = ConceptDeduplicator(chain_creator)
        self.formatter = ConceptFormatter(
            chain_creator, queue, style_instructions
        )

        self.queue = queue

    async def process_concepts(
        self,
        concept_df: pd.DataFrame,
        relation_df: pd.DataFrame,
        chunk_df: pd.DataFrame,
        flat_part_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, int]:
        """
        Processes concepts to create a hierarchical mind map from
        scratch.

        This method orchestrates the end-to-end pipeline:
        1.  **Preparation**: Prepares the initial concept DataFrame.
        2.  **Clustering**: Builds a multi-level hierarchy of concepts
            by
            iteratively clustering and synthesizing parent concepts.
        3.  **Root Definition**: Synthesizes a single, overarching root
            concept for the entire mind map from the top-level clusters.
        4.  **Deduplication (Merge)**: Finds concepts with identical
            names and merges them into single, synthesized concepts.
        5.  **Formatting**: Adds final citations and uses an LLM to
            "prettify" the language of all concepts.
        6.  **Deduplication (Rename)**: Finds any remaining concepts
            with duplicate names and renames them to be unique.
        7.  **Finalization**: Returns the final data structures.

        Args:
            concept_df: DataFrame of raw concepts.
            relation_df: DataFrame of initial relations.
            chunk_df: DataFrame of source document chunks.
            flat_part_df: DataFrame with citation information.

        Returns:
            A tuple containing the final concept DataFrame, the final
            relation DataFrame, and the index of the root concept.
        """
        logging.info("Concept processing: Start")

        data = RawMindMapData(
            concept_df=self._prep_concept_df(concept_df),
            relation_df=relation_df,
            chunk_df=chunk_df,
            flat_part_df=flat_part_df,
        )

        data = await self.clusterer.create_hierarchy(data)
        data = await self.define_root(data)
        data = await self.deduplicator.deduplicate(data)
        data = await self.formatter.format_final_concepts(data)
        data = await self.deduplicator.make_names_unique(data)

        return await self.wrap_up(data)

    @staticmethod
    def _prep_concept_df(concept_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the concept DataFrame by initializing required columns.
        """
        concept_df[Col.IS_ACTIVE_CONCEPT] = ColVals.TRUE_INT
        concept_df[Col.EMBEDDING] = None
        concept_df[Col.ANSWER_STR] = np.nan

        answers_w_flat_part_ids = concept_df[[Col.ANSWER, Col.FLAT_PART_ID]]
        answers = answers_w_flat_part_ids.apply(utils.format_source_ids, axis=1)
        concept_df[Col.ANSWER] = answers

        str_answers = concept_df.apply(utils.repr_answer, axis=1)
        concept_df[Col.ANSWER_STR] = str_answers
        return concept_df

    async def define_root(self, data: RawMindMapData) -> RawMindMapData:
        """Defines the main root concept for the entire mind map.

        This method takes the highest-level concepts produced by the
        clustering step and uses an LLM to synthesize a single, all-
        encompassing root concept from them. It then establishes
        relationships from this new main root to its immediate children.

        Args:
            data: The mind map data after the clustering stage.

        Returns:
            The mind map data with a single main root concept defined.
        """
        other_concept_df = data.concept_df
        rel_df = data.relation_df
        root_df = data.root_df

        if root_df is not None and not root_df.empty:
            concept_df = pd.concat([other_concept_df, root_df])
        else:
            # If no separate root_df is provided, use the main df.
            root_df = concept_df = other_concept_df

        root_synth_results = await self._synthesize_roots(root_df)

        qapair_name_to_id = pd.Series(
            root_df.index, index=root_df[Col.NAME]
        ).to_dict()
        sub_roots = await self._prep_roots(
            root_df, root_synth_results, qapair_name_to_id
        )
        concept_df = utils.add_concepts(concept_df, sub_roots)
        concept_df = utils.make_unique_names(concept_df)

        # The main root is the last one added.
        root_index = int(concept_df.index.values[-1])
        synth_answers = concept_df.apply(utils.repr_answer, axis=1)
        concept_df[Col.ANSWER_STR] = synth_answers

        # Create relationships from the main root to its sub-roots.
        root_cluster_mask = concept_df[Col.CLUSTER_ID] == -1
        # Exclude the main root itself from being a target.
        sub_root_indices = concept_df.loc[
            root_cluster_mask & (concept_df.index != root_index)
        ].index
        num_rels_to_create = min(len(sub_root_indices), 4)

        root_rels = [
            {
                Col.ORIGIN_CONCEPT_ID: root_index,
                Col.TARGET_CONCEPT_ID: idx,
                # High weight for primary relationships.
                Col.WEIGHT: 4.0,
            }
            for idx in sub_root_indices[:num_rels_to_create]
        ]
        root_rel_df = pd.DataFrame(root_rels)
        rel_df = utils.update_rel_df(rel_df, root_rel_df, concept_df)
        rel_df[Col.TYPE] = ColVals.RELATED

        return RawMindMapData(
            concept_df=concept_df,
            relation_df=rel_df,
            root_df=root_df,
            chunk_df=data.chunk_df,
            flat_part_df=data.flat_part_df,
            root_index=root_index,
        )

    async def _synthesize_roots(
        self, root_df: pd.DataFrame
    ) -> RootClusterSynthesisResultProtocol:
        """Synthesizes a single root from all top-level concepts."""
        inputs = {Pi.CLUSTER_SYNTH: utils.repr_concepts(root_df)}
        chain = self.chain_creator.choose_chain(Ct.ROOT_CLUSTER_SYNTH)
        return await ChainRunner().run_chain_w_retries(
            chain, inputs, max_retries=3, retry_field_name=Fn.SOURCE_IDS
        )

    @classmethod
    async def _prep_roots(
        cls,
        root_df: pd.DataFrame,
        root_synth_results: RootClusterSynthesisResultProtocol,
        concept_name_to_id: dict[str, int],
    ) -> list[dict]:
        """Prepares the final list of root concepts from LLM results.

        This includes the main synthesized root and the sub-roots it was
        derived from, which are now its direct children.

        Args:
            root_df: DataFrame of the original top-level concepts.
            root_synth_results: Results from the root synthesis chain.
            concept_name_to_id: Mapping from concept names to their IDs.

        Returns:
            A list of prepared root concept dictionaries.
        """
        main_root = getattr(root_synth_results, Fn.ROOT_CONCEPT, None)
        sub_roots_from_synth = getattr(
            root_synth_results, Fn.SYNTH_CONCEPTS, []
        )

        all_roots_to_process = sub_roots_from_synth + (
            [main_root] if main_root else []
        )
        if not all_roots_to_process:
            return []

        prepared_roots = []
        max_lvl = root_df[Col.LVL].max()
        for sub_root in all_roots_to_process:
            qapair = utils.rename_concept_fields(
                sub_root.model_dump(), concept_name_to_id
            )
            prepared_roots.append(
                {**qapair, Col.CLUSTER_ID: -1, Col.LVL: max_lvl}
            )

        # The last concept in the list is the main root; mark its level
        # as -1.
        prepared_roots[-1][Col.LVL] = -1
        return prepared_roots

    async def wrap_up(
        self, data: RawMindMapData
    ) -> tuple[pd.DataFrame, pd.DataFrame, int]:
        """Performs final logging and returns results."""
        concept_df = data.concept_df
        relation_df = data.relation_df
        root_index = data.root_index

        llm_cost_handler = cur_llm_cost_handler.get()
        logging.info(f"=Postprocessing LLM costs:\n{llm_cost_handler}")
        logging.info(
            f"Final Mind Map: {len(concept_df)} nodes, {len(relation_df)} "
            "relationships"
        )
        await self.queue.put(None)  # Signal completion
        return concept_df, relation_df, root_index


class AddConceptOrchestrator(ProcessingOrchestrator):
    """
    Orchestrates adding new concepts to an existing mind map.

    This class modifies the standard processing pipeline to handle the
    more complex task of merging a set of new concepts into a
    pre-existing, multi-level concept hierarchy.
    """

    def __init__(
        self,
        chain_creator: ChainCreator,
        queue: aio.Queue,
        style_instructions: dict[str, Any],
    ):
        """
        Initializes the orchestrator for adding concepts.

        This overrides the standard `ConceptClusterer` with the
        specialized `AddConceptClusterer` designed for merging.
        """
        super().__init__(chain_creator, queue, style_instructions)
        self.clusterer = AddConceptClusterer(chain_creator, queue)

    async def process_concepts(
        self,
        concept_df: pd.DataFrame,
        relation_df: pd.DataFrame,
        chunk_df: pd.DataFrame,
        flat_part_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, int]:
        """
        Processes concepts by merging new ones into an existing
        hierarchy.

        This method orchestrates a modified pipeline:
        1.  **Preparation**: Prepares new and old concepts separately.
        2.  **Clustering (Merge)**: Rebuilds the hierarchy
            level-by-level, merging new concepts with old ones at each
            stage.
        3.  **Root Definition (Merge)**: Synthesizes a new main root and
            then merges it with the existing mind map's old root.
        4.  **Deduplication & Formatting**: Runs the standard
            deduplication and formatting steps on the newly merged
            hierarchy.

        Args:
            concept_df: DataFrame containing both old and new concepts,
                distinguished by a 'new' column.
            relation_df: DataFrame of existing relations.
            chunk_df: DataFrame of source document chunks.
            flat_part_df: DataFrame with citation information.

        Returns:
            A tuple containing the final concept DataFrame, the final
            relation DataFrame, and the index of the root concept.
        """
        logging.info(
            "Starting concept processing pipeline for adding concepts."
        )
        concept_df, flat_part_df = await self._prepare_initial_dataframes(
            concept_df, flat_part_df
        )

        old_root_concept = concept_df[
            (concept_df[Col.LVL] == -1) & (concept_df[Col.CLUSTER_ID] == -1)
        ]

        data = RawMindMapData(
            # concept_df=concept_df,  # Possibly prep is redundant here
            concept_df=self._prep_concept_df(concept_df),
            relation_df=relation_df,
            chunk_df=chunk_df,
            flat_part_df=flat_part_df,
            old_root_concept=old_root_concept,
        )

        data = await self.clusterer.create_hierarchy(data)
        data = await self._define_root_w_old_root(data)
        data = await self.deduplicator.deduplicate(data)
        data = await self.formatter.format_final_concepts(data)

        return await self.wrap_up(data)

    async def _prepare_initial_dataframes(
        self, concept_df: pd.DataFrame, flat_part_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Prepares initial DataFrames for the 'add' pipeline."""
        concept_df = concept_df.copy()

        concept_df[Col.EMBEDDING] = np.nan
        concept_df[Col.IS_ACTIVE_CONCEPT] = ColVals.TRUE_INT

        # Create a mapping from flat_part_id to citation.
        exploded_citations = concept_df.explode(
            [Col.FLAT_PART_ID, Col.CITATION]
        )
        citation_map_df = (
            exploded_citations[[Col.FLAT_PART_ID, Col.CITATION]]
            .dropna()
            .drop_duplicates(subset=Col.FLAT_PART_ID)
        )
        flat_part_df = pd.concat(
            [flat_part_df, citation_map_df], ignore_index=True
        ).drop_duplicates(subset=Col.FLAT_PART_ID)

        concept_df = concept_df.drop(columns=[Col.CITATION])

        new_concepts_mask = concept_df["new"] == 1

        # Check if there are any new concepts to process to avoid
        # errors on an empty slice.
        if new_concepts_mask.any():
            # Prepare the data from the slice
            prepped_data = self._prep_concept_df(
                concept_df.loc[new_concepts_mask]
            )
            # Assign the prepared data back to the original DataFrame
            # using the mask
            concept_df.loc[new_concepts_mask] = prepped_data

        concept_df[Col.ANSWER] = concept_df.apply(
            utils.replace_chunk_id_with_index,
            axis=1,
            args=(flat_part_df[[Col.FLAT_PART_ID, Col.CITATION]],),
        )

        return concept_df, flat_part_df

    async def _define_root_w_old_root(
        self, data: RawMindMapData
    ) -> RawMindMapData:
        """Defines the root, merging the new root with the old one."""
        root_df = data.root_df
        other_concept_df = data.concept_df
        rel_df = data.relation_df
        old_root_concept = data.old_root_concept

        if root_df is not None and not root_df.empty:
            concept_df = pd.concat([other_concept_df, root_df])
        else:
            root_df = concept_df = other_concept_df

        root_synth_results = await self._synthesize_roots(root_df)
        qapair_name_to_id = pd.Series(
            root_df.index, index=root_df[Col.NAME]
        ).to_dict()

        sub_roots = await self._prep_roots_w_old_root(
            root_df, root_synth_results, qapair_name_to_id, old_root_concept
        )
        concept_df = utils.add_concepts(concept_df, sub_roots)
        concept_df = utils.make_unique_names(concept_df)

        root_index = int(concept_df[concept_df[Col.LVL] == -1].index.values[0])
        concept_df[Col.ANSWER_STR] = concept_df.apply(utils.repr_answer, axis=1)

        # Create relationships from the main root to its sub-roots.
        root_cluster_mask = concept_df[Col.CLUSTER_ID] == -1
        sub_root_indices = concept_df.loc[
            root_cluster_mask & (concept_df.index != root_index)
        ].index
        num_rels = min(len(sub_root_indices), 4)
        root_rels = [
            {
                Col.ORIGIN_CONCEPT_ID: root_index,
                Col.TARGET_CONCEPT_ID: idx,
                Col.WEIGHT: 4.0,
            }
            for idx in sub_root_indices[:num_rels]
        ]
        rel_df = utils.update_rel_df(
            rel_df, pd.DataFrame(root_rels), concept_df
        )
        rel_df[Col.TYPE] = ColVals.RELATED
        return RawMindMapData(
            concept_df=concept_df,
            relation_df=rel_df,
            chunk_df=data.chunk_df,
            flat_part_df=data.flat_part_df,
            old_root_concept=data.old_root_concept,
            root_df=data.root_df,
            root_index=root_index,
        )

    async def _prep_roots_w_old_root(
        self,
        root_df: pd.DataFrame,
        root_synth_results: RootClusterSynthesisResultProtocol,
        concept_name_to_id: dict[str, int],
        old_root_concept: pd.DataFrame,
    ) -> list[dict]:
        """Prepares roots, merging the new main root with the old one."""
        main_root_obj = getattr(root_synth_results, Fn.ROOT_CONCEPT, None)
        if not main_root_obj:
            return []

        # If an old root exists, use an LLM to merge it with the new
        # one.
        if not old_root_concept.empty:
            inputs = {
                Pi.OLD_CONCEPT: utils.repr_concepts(old_root_concept),
                Pi.NEW_CONCEPT: str(main_root_obj.model_dump()),
            }
            chain = self.chain_creator.choose_chain(Ct.NEW_ROOT)
            result = await ChainRunner().run_chain_w_retries(
                chain, inputs, max_retries=3, retry_field_name=Fn.SOURCE_IDS
            )
            main_root_obj = getattr(result, Fn.ROOT_CONCEPT, main_root_obj)

        sub_roots = getattr(root_synth_results, Fn.SYNTH_CONCEPTS, [])
        all_roots = sub_roots + [main_root_obj]

        prepared_roots = []
        max_lvl = root_df[Col.LVL].max()
        for root in all_roots:
            qapair = utils.rename_concept_fields(
                root.model_dump(), concept_name_to_id
            )
            prepared_roots.append(
                {**qapair, Col.CLUSTER_ID: -1, Col.LVL: max_lvl}
            )

        # Mark the main root (the last one in the list) with level -1.
        if prepared_roots:
            prepared_roots[-1][Col.LVL] = -1
        return prepared_roots
