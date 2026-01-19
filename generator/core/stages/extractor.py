import asyncio as aio
from typing import Optional

import pandas as pd

from common_utils.logger_config import logger
from generator.chainer import ChainCreator
from generator.chainer import ChainRunner as Cr
from generator.chainer.utils.constants import ChainTypes as Ct
from generator.common.constants import DataFrameCols as Col
from generator.common.constants import FieldNames as Fn
from generator.common.context import cur_llm_cost_handler
from generator.core.actions.llm import form_multimodal_inputs, form_text_inputs
from generator.core.stages.doc_handler.constants import DocCategories as DocCat
from generator.core.structs import ExtractionProduct
from generator.core.utils.constants import DefaultValues as Dv
from generator.core.utils.constants import FrontEndStatuses as Fes
from generator.core.utils.frontend_handler import put_status
from generator.core.utils.misc import concat_tuples


class Extractor:
    """
    Extracts structured information (concepts and relations) from
    various document types to build a mind map.

    This class orchestrates the process of running document chunks
    through Language Models (LLMs), parsing the structured output, and
    formatting it into pandas DataFrames for further processing.
    """

    def __init__(self, chain_creator: ChainCreator, queue: aio.Queue):
        """
        Initializes the Extractor.

        Args:
            queue: The asyncio queue for sending frontend status
                updates.
        """
        self.chain_creator = chain_creator
        self.queue = queue

    async def extract_mindmap_elements(
        self,
        chunk_df: pd.DataFrame,
        flat_part_df: pd.DataFrame,
        instructions: dict[str, str | dict[str, str]],
        is_add_mode: bool = False,
    ) -> ExtractionProduct:
        """
        Extracts concepts and relations from chunked documents.

        This is the main entry point for the extraction process.

        Args:
            chunk_df: DataFrame containing document chunks.
            flat_part_df: DataFrame containing flat document parts for
                metadata.
            instructions: A dictionary of instructions for the LLM.
            is_add_mode: If True, calculates offsets for adding to an
                existing map. Otherwise, assumes a fresh extraction.

        Returns:
            An ExtractionProduct object containing the concept_df,
            relation_df, and the updated chunk_df. Returns a result with
            None for DataFrames if no concepts are extracted.
        """
        if not is_add_mode:
            logger.info("Extraction: Start")
        else:
            logger.info("Add Extraction: Start")
            if chunk_df.empty:
                logger.warning(
                    "Add mode was selected, but the provided chunk_df is empty."
                )
                return await self._wrap_up(chunk_df)

        chunk_offset = Dv.START_FLAT_PART_ID

        extraction_results = await self._llm_extract(chunk_df, instructions)

        concepts, relations = self._parse_extraction_results(
            extraction_results, start_index=chunk_offset
        )

        if not concepts:
            logger.warning("No concepts were extracted from the documents.")
            return await self._wrap_up(chunk_df)

        concept_df = self._create_concept_df(concepts, chunk_df, flat_part_df)
        relation_df = self._create_relation_df(relations, concept_df)

        chunk_df, concept_df, relation_df = self._prepare_dfs(
            chunk_df, concept_df, relation_df, chunk_offset=chunk_offset
        )

        return await self._wrap_up(chunk_df, concept_df, relation_df)

    async def _llm_extract(
        self,
        chunk_df: pd.DataFrame,
        instructions: dict[str, str | dict[str, str]],
    ) -> list:
        """
        Dispatches chunks to the appropriate LLM chains based on document
        type and returns the re-ordered results.
        """
        if chunk_df.empty:
            return []

        doc_type_configs = [
            {
                "mask": (chunk_df[Col.DOC_CAT] == DocCat.LINK)
                | (chunk_df[Col.DOC_CAT] == DocCat.HTML)
                | (chunk_df[Col.DOC_CAT] == DocCat.TXT),
                "chain_type": Ct.TEXTUAL_EXTRACTION,
                "input_formatter": lambda df: form_text_inputs(
                    df, instructions
                ),
            },
            {
                "mask": chunk_df[Col.DOC_CAT] == DocCat.PPTX,
                "chain_type": Ct.PPTX_EXTRACTION,
                "input_formatter": lambda df: form_multimodal_inputs(
                    df, DocCat.PPTX, instructions
                ),
            },
            {
                "mask": chunk_df[Col.DOC_CAT] == DocCat.PDF,
                "chain_type": Ct.PDF_EXTRACTION,
                "input_formatter": lambda df: form_multimodal_inputs(
                    df, DocCat.PDF, instructions
                ),
            },
        ]

        chains_with_inputs = []
        all_indices = []

        for config in doc_type_configs:
            masked_df = chunk_df.loc[config["mask"]]
            if not masked_df.empty:
                indices = list(masked_df.index)
                inputs = config["input_formatter"](masked_df)
                chain = self.chain_creator.choose_chain(config["chain_type"])
                chains_with_inputs.append((chain, inputs))
                all_indices.extend(indices)

        if not chains_with_inputs:
            logger.warning("No processable chunks found for extraction.")
            return []

        # Run extraction with frontend status updates
        status_details = Fes.PROGRESS.format(0)
        await put_status(self.queue, Fes.ANALYZE_DOCS, status_details)

        results = await Cr().run_chains_w_status_updates(
            chains_with_inputs,
            self._put_extraction_status,
        )

        # Re-sort results to match the original chunk_df order
        indexed_results = list(zip(all_indices, results))
        sorted_results = sorted(indexed_results, key=lambda x: x[0])
        return [result for _, result in sorted_results]

    @staticmethod
    def _parse_extraction_results(
        extraction_results: list, start_index: int = 1
    ) -> tuple[list[dict], list[dict]]:
        """Parses LLM results, assigning a unique, offset chunk ID."""
        concepts, relations = [], []
        for chunk_id, result in enumerate(
            extraction_results, start=start_index
        ):
            concepts.extend(
                {
                    **concept.model_dump(),
                    Col.FLAT_CHUNK_ID: chunk_id,
                    Col.PAGE_ID: tuple(
                        sorted(
                            set(
                                source_id
                                for part in concept.answer
                                for source_id in getattr(
                                    part, Fn.SOURCE_IDS, []
                                )
                            )
                        )
                    ),
                }
                for concept in result.concepts
            )
            relations.extend(
                {**relation.model_dump(), Col.FLAT_CHUNK_ID: chunk_id}
                for relation in result.relations
            )
        return concepts, relations

    @classmethod
    def _prepare_dfs(
        cls,
        chunk_df: pd.DataFrame,
        concept_df: pd.DataFrame,
        relation_df: pd.DataFrame,
        chunk_offset: int = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Finalizes DataFrames by linking them, making concept names
        unique, and cleaning up temporary columns.
        """
        concept_df = cls._make_names_unique(concept_df)

        # Use a temporary column to map concepts/relations back to the
        # original chunk_df index.
        concept_df[Col.TEMP_CHUNK_ID] = (
            concept_df[Col.FLAT_CHUNK_ID] - chunk_offset
        )
        chunk_grouped_concept_df = concept_df.groupby(Col.TEMP_CHUNK_ID)
        chunk_df[Col.CONCEPT_IDS] = chunk_grouped_concept_df.apply(
            lambda df: df.index.tolist() if not df.empty else []
        )

        # And use it here as well
        relation_df[Col.TEMP_CHUNK_ID] = (
            relation_df[Col.FLAT_CHUNK_ID] - chunk_offset
        )
        if not relation_df.empty:
            chunk_grouped_rel_df = relation_df.groupby(Col.TEMP_CHUNK_ID)
            chunk_df[Col.RELATION_IDS] = chunk_grouped_rel_df.apply(
                lambda df: df.index.tolist() if not df.empty else []
            )
        else:
            chunk_df[Col.RELATION_IDS] = [[] for _ in range(len(chunk_df))]

        concept_df[Col.FLAT_CHUNK_ID] = concept_df[Col.FLAT_CHUNK_ID].apply(
            lambda flat_chunk_id: (flat_chunk_id,)
        )

        # Clean up temporary columns
        concept_df.drop(columns=[Col.TEMP_CHUNK_ID], inplace=True)
        relation_df.drop(
            columns=[Col.TEMP_CHUNK_ID, Col.FLAT_CHUNK_ID], inplace=True
        )

        return chunk_df, concept_df, relation_df

    @staticmethod
    def _make_names_unique(concept_df: pd.DataFrame) -> pd.DataFrame:
        """Ensures all concept names are unique by appending a suffix."""
        # Using .items() is idiomatic for iterating over a Series
        for name, count in concept_df[Col.NAME].value_counts().items():
            if count > 1:
                indices = concept_df[concept_df[Col.NAME] == name].index
                for i, idx in enumerate(indices, 1):
                    concept_df.loc[idx, Col.NAME] = f"{name}_{i}"
        return concept_df

    @classmethod
    def _create_concept_df(
        cls,
        concepts: list[dict],
        chunk_df: pd.DataFrame,
        flat_part_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Builds the concept DataFrame from parsed LLM results."""
        if not concepts:
            return pd.DataFrame()

        concept_df = pd.DataFrame(concepts)
        concept_df = pd.merge(
            concept_df,
            chunk_df[[Col.FLAT_CHUNK_ID, Col.DOC_CAT]],
            on=Col.FLAT_CHUNK_ID,
        )

        pptx_mask = concept_df[Col.DOC_CAT] == DocCat.PPTX
        non_pptx_mask = ~pptx_mask

        # Improvement: Process PPTX and non-PPTX concepts separately
        pptx_concepts = cls._process_pptx_concepts(
            concept_df[pptx_mask], flat_part_df
        )
        non_pptx_concepts = cls._process_non_pptx_concepts(
            concept_df[non_pptx_mask], flat_part_df
        )

        return pd.concat([non_pptx_concepts, pptx_concepts], ignore_index=True)

    @staticmethod
    def _process_pptx_concepts(
        pptx_concept_df: pd.DataFrame, flat_part_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Handles the specific logic for concepts from PPTX documents."""
        if pptx_concept_df.empty:
            return pd.DataFrame()

        pptx_concept_df[Col.PAGE_ID] = pptx_concept_df[Col.ANSWER].apply(
            lambda answer: tuple(
                set(
                    id_
                    for part in answer
                    for id_ in part.get(Fn.SOURCE_IDS, [])
                )
            )
        )

        exploded_df = pptx_concept_df.explode(Col.PAGE_ID)
        exploded_df[Col.PAGE_ID] = exploded_df[Col.PAGE_ID].apply(
            lambda x: (x,)
        )

        merged_df = pd.merge(
            exploded_df,
            flat_part_df[[Col.FLAT_CHUNK_ID, Col.PAGE_ID, Col.FLAT_PART_ID]],
            on=[Col.FLAT_CHUNK_ID, Col.PAGE_ID],
        )

        group_cols = [
            Col.NAME,
            Col.QUESTION,
            Col.FLAT_CHUNK_ID,
            Col.DOC_CAT,
        ]
        agg_rules = {
            Col.ANSWER: "first",
            Col.FLAT_PART_ID: lambda x: sorted(list(x)),
            Col.PAGE_ID: lambda x: concat_tuples(x),
        }
        return merged_df.groupby(group_cols).agg(agg_rules).reset_index()

    @staticmethod
    def _process_non_pptx_concepts(
        non_pptx_concept_df: pd.DataFrame, flat_part_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Handles concept creation for non-PPTX documents."""
        if non_pptx_concept_df.empty:
            return non_pptx_concept_df

        merged_df = pd.merge(
            non_pptx_concept_df,
            flat_part_df[[Col.FLAT_CHUNK_ID, Col.PAGE_ID, Col.FLAT_PART_ID]],
            on=[Col.FLAT_CHUNK_ID, Col.PAGE_ID],
        )
        merged_df[Col.FLAT_PART_ID] = merged_df[Col.FLAT_PART_ID].apply(
            lambda x: [x]
        )
        return merged_df

    @classmethod
    def _create_relation_df(
        cls, relations: list[dict], concept_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Builds the relation DataFrame and links it to concepts."""
        if not relations:
            return pd.DataFrame(
                columns=[
                    Col.ORIGIN_CONCEPT_ID,
                    Col.TARGET_CONCEPT_ID,
                    Col.WEIGHT,
                    Col.FLAT_CHUNK_ID,
                ]
            )

        rel_df = pd.DataFrame(relations)

        concept_lookup = (
            concept_df[[Col.NAME, Col.FLAT_CHUNK_ID]]
            .reset_index()
            .rename(columns={"index": Col.CONCEPT_ID})
        )

        # Merge for origin concepts
        rel_df = pd.merge(
            rel_df,
            concept_lookup,
            left_on=[Fn.ORIGIN_CONCEPT_NAME, Col.FLAT_CHUNK_ID],
            right_on=[Col.NAME, Col.FLAT_CHUNK_ID],
            how="left",
        ).rename(columns={Col.CONCEPT_ID: Col.ORIGIN_CONCEPT_ID})

        # Merge for target concepts
        rel_df = pd.merge(
            rel_df,
            concept_lookup,
            left_on=[Fn.TARGET_CONCEPT_NAME, Col.FLAT_CHUNK_ID],
            right_on=[Col.NAME, Col.FLAT_CHUNK_ID],
            how="left",
            suffixes=("", "_target"),
        ).rename(columns={Col.CONCEPT_ID: Col.TARGET_CONCEPT_ID})

        rel_df.drop(
            columns=[
                Col.NAME,
                f"{Col.NAME}_target",
                Fn.ORIGIN_CONCEPT_NAME,
                Fn.TARGET_CONCEPT_NAME,
            ],
            inplace=True,
        )

        rel_df[Col.WEIGHT] = 3.0

        rows_with_nans = rel_df[rel_df.isna().any(axis=1)]
        if not rows_with_nans.empty:
            logger.warning(
                "The following relations with nonexistent concepts "
                f"were removed:\n{rows_with_nans}"
            )
        rel_df.dropna(inplace=True, ignore_index=True)

        rel_df[Col.ORIGIN_CONCEPT_ID] = rel_df[Col.ORIGIN_CONCEPT_ID].astype(
            int
        )
        rel_df[Col.TARGET_CONCEPT_ID] = rel_df[Col.TARGET_CONCEPT_ID].astype(
            int
        )

        return rel_df[
            [
                Col.ORIGIN_CONCEPT_ID,
                Col.TARGET_CONCEPT_ID,
                Col.WEIGHT,
                Col.FLAT_CHUNK_ID,
            ]
        ]

    async def _put_extraction_status(self, completed: int, total: int) -> None:
        """Updates the frontend with the extraction progress."""
        chunk_text = "chunk" if completed == 1 else "chunks"
        status_msg = f"{completed} {chunk_text} out of {total} processed"
        logger.info(status_msg)

        raw_percentage = (completed / total) * 100 if total > 0 else 100
        rounded_percentage = max(5, 5 * round(raw_percentage / 5))

        status_details = Fes.PROGRESS.format(rounded_percentage)
        await put_status(self.queue, Fes.ANALYZE_DOCS, status_details)

    async def _wrap_up(
        self,
        chunk_df: pd.DataFrame,
        concept_df: Optional[pd.DataFrame] = None,
        relation_df: Optional[pd.DataFrame] = None,
    ) -> ExtractionProduct:
        """
        Logs final results, signals completion, and returns the data.
        """
        if concept_df is not None:
            concept_count = len(concept_df)
            node_text = "node" if concept_count == 1 else "nodes"
            verb_text = "was" if concept_count == 1 else "were"
            logger.info(f"{concept_count} {node_text} {verb_text} extracted.")

            llm_cost_handler = cur_llm_cost_handler.get()
            logger.info(f"Extraction LLM costs:\n{llm_cost_handler}")
            logger.info("Extraction: End")

        await self.queue.put(None)

        return ExtractionProduct(concept_df, relation_df, chunk_df)
