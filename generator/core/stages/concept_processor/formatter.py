import asyncio as aio
import itertools
from typing import Any

import pandas as pd
from langchain_core.runnables import RunnableSerializable

from generator.chainer import ChainCreator, ChainRunner
from generator.chainer.utils.constants import ChainTypes as Ct
from generator.common.constants import ColVals
from generator.common.constants import DataFrameCols as Col
from generator.common.logger import logging
from generator.core.utils.constants import FACT, SOURCE_IDS
from generator.core.utils.constants import FrontEndStatuses as Fes
from generator.core.utils.constants import Pi
from generator.core.utils.frontend_handler import put_status

from ...structs import RawMindMapData
from . import utils


class ConceptFormatter:
    """
    Handles final formatting and citation of concepts.

    This class is the final step in the concept processing pipeline. It
    takes structured concept data and performs two main actions:
    1.  **Citation Augmentation**: Replaces internal source IDs with
        final, user-facing citation markers.
    2.  **Prettification**: Uses an LLM to polish the language of
        concept names, questions, and answers to match a specified
        style, without altering factual content.
    """

    def __init__(
        self,
        chain_creator: ChainCreator,
        queue: aio.Queue,
        style_instructions: dict,
    ):
        """Initializes the formatter.

        Args:
            queue: An asyncio queue for sending frontend status updates.
            style_instructions: A dictionary of instructions for the
                LLM to guide the "prettification" style.
        """
        self.chain_creator = chain_creator
        self.queue = queue
        self.style_instructions = style_instructions
        self.chain_runner = ChainRunner()

    async def format_final_concepts(
        self, data: RawMindMapData
    ) -> RawMindMapData:
        """
        Creates the final concept DataFrame.

        This is the main entry point for formatting a full set of
        concepts from scratch. It filters for active concepts, adds
        citations, applies stylistic prettification via an LLM, and
        finally generates embeddings for the polished concepts.

        Args:
            data: The raw mind map data containing unformatted concepts.

        Returns:
            The mind map data with a fully formatted and embedded
            concept DataFrame.
        """
        concept_df = data.concept_df
        flat_part_df = data.flat_part_df

        logging.info("Creating final concept DataFrame")
        active_mask = concept_df[Col.IS_ACTIVE_CONCEPT] == ColVals.TRUE_INT
        concept_df = concept_df.loc[active_mask].copy()
        concept_df = await self._augment_and_prettify(concept_df, flat_part_df)
        return RawMindMapData(
            concept_df=utils.embed_active_concepts(concept_df),
            relation_df=data.relation_df,
            root_df=data.root_df,
            chunk_df=data.chunk_df,
            flat_part_df=data.flat_part_df,
            root_index=data.root_index,
        )

    async def process_del_changes(
        self, concept_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prettifies only the concepts that were marked as modified.

        This method is an optimization for incremental updates. It finds
        rows marked with a 'modified' flag and sends only those concepts
        to the LLM for stylistic polishing, avoiding the cost of
        re-processing unchanged concepts.

        Args:
            concept_df: The concept DataFrame, potentially with some
                rows marked as modified.

        Returns:
            The concept DataFrame with modified concepts updated
            in-place.
        """
        modified_rows = concept_df[concept_df.get(Col.MODIFIED) == 1]
        if modified_rows.empty:
            logging.info("No modified answers to prettify.")
            return concept_df

        concepts_to_prettify = modified_rows.apply(
            self._repr_question_w_answer, axis=1
        ).tolist()
        batched_concepts = utils.split_list_into_batches(concepts_to_prettify)
        logging.info(f"Prettifying {len(modified_rows)} modified concepts.")

        chain = self.chain_creator.choose_chain(Ct.APPLY_CONCEPT_PRETTIFIER)
        pretty_concepts = await self.prettify_concepts_with_retry(
            batched_concepts, chain
        )

        for i, idx in enumerate(modified_rows.index):
            concept = pretty_concepts[i]
            if hasattr(concept, "name"):
                concept_df.at[idx, Col.NAME] = concept.name
                concept_df.at[idx, Col.QUESTION] = concept.question
                concept_df.at[idx, Col.ANSWER_STR] = concept.answer
        return concept_df

    async def _augment_and_prettify(
        self, concept_df: pd.DataFrame, flat_part_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Augments answers with citations and applies LLM prettification.

        This method first performs the data manipulation needed to map
        internal `flat_part_id`s to final citation text. It then
        prepares the concepts and sends them to the LLM for stylistic
        polishing.

        Args:
            concept_df: The DataFrame of active concepts.
            flat_part_df: The DataFrame mapping part IDs to citations.

        Returns:
            The fully cited and prettified concept DataFrame.
        """
        if (
            Col.CITATION in concept_df.columns
            and concept_df[Col.CITATION].isna().any()
        ):
            concept_df.drop(Col.CITATION, axis=1, inplace=True)
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
            concept_df, result_df, left_index=True, right_index=True, how="left"
        )

        id_to_citation_map = pd.Series(
            flat_part_df[Col.CITATION].values,
            index=flat_part_df[Col.FLAT_PART_ID],
        )

        concept_df[Col.ANSWER] = concept_df.apply(
            self._replace_id_with_chunk_id,
            axis=1,
            args=(id_to_citation_map,),
        )

        concepts_to_prettify = concept_df.apply(
            self._repr_question_w_answer, axis=1
        ).tolist()
        batched_concepts = utils.split_list_into_batches(concepts_to_prettify)
        logging.info("Concept Prettification")
        chain = self.chain_creator.choose_chain(Ct.CONCEPT_PRETTIFIER)
        pretty_concepts = await self.prettify_concepts_with_retry(
            batched_concepts, chain
        )

        for i, idx in enumerate(concept_df.index):
            concept = pretty_concepts[i]
            # Defensively update only if prettification was successful
            if hasattr(concept, "name"):
                concept_df.at[idx, Col.NAME] = concept.name
                concept_df.at[idx, Col.QUESTION] = concept.question
                concept_df.at[idx, Col.ANSWER_STR] = concept.answer
        return concept_df

    async def _process_single_concept(
        self, concept: dict, chain: RunnableSerializable
    ) -> list:
        """
        Processes a single concept through the prettification chain.

        This is the base case for the recursive batch processing logic.

        Args:
            concept: The concept dictionary to process.
            chain: The processing chain to apply.

        Returns:
            A list containing the processed concept object, or the
            original concept dict in a list if processing fails.
        """
        inputs = [
            {
                Pi.QAPAIRS: [concept],
                Pi.NUM_ANSWERS: 1,
                Pi.STYLE: self.style_instructions,
            }
        ]
        results = await ChainRunner().run_chains_w_status_updates(
            [(chain, inputs)], self._put_prettification_status
        )
        if (
            results
            and hasattr(results[0], "pretty_concepts")
            and (processed := results[0].pretty_concepts)
        ):
            return processed

        logging.warning(f"Was not able to prettify node: {concept.get('name')}")
        return [concept]  # Return original dict on failure

    async def _put_prettification_status(
        self, completed: int, total: int
    ) -> None:
        """
        Updates status for the concept prettification progress.

        This callback sends progress updates to a frontend via a queue.
        It rounds the percentage to the nearest 5% to avoid sending an
        excessive number of status updates for small changes.

        Args:
            completed: Number of completed items.
            total: Total number of items to process.
        """
        raw_percentage = (completed / total) * 100
        rounded_percentage = max(5, 5 * round(raw_percentage / 5))
        status_details = Fes.PROGRESS.format(rounded_percentage)
        await put_status(self.queue, Fes.PRETTIFY, status_details)

    async def _recursive_process_batch(
        self, concept_batch: list[dict], chain: RunnableSerializable
    ) -> list:
        """
        Recursively processes a batch, splitting it if processing fails.

        This divide-and-conquer approach provides robustness. If the LLM
        fails to process a large batch (e.g., due to context length or
        a malformed item), this method splits the batch in half and
        retries each half separately. This continues until the failing
        item is isolated and processed individually.

        Args:
            concept_batch: A batch of concept dictionaries to process.
            chain: The LLM chain to use for processing.

        Returns:
            A list of processed concept objects (or original dicts on
            failure).
        """
        if len(concept_batch) == 1:
            return await self._process_single_concept(concept_batch[0], chain)

        inputs = [
            {
                Pi.QAPAIRS: concept_batch,
                Pi.NUM_ANSWERS: len(concept_batch),
                Pi.STYLE: self.style_instructions,
            }
        ]
        results = await ChainRunner().run_chains_w_status_updates(
            [(chain, inputs)], self._put_prettification_status
        )
        result = results[0]
        processed_concepts = getattr(result, "pretty_concepts", [])

        if len(processed_concepts) == len(concept_batch):
            return processed_concepts

        logging.warning(
            f"Batch of size {len(concept_batch)} failed, splitting."
        )
        mid = len(concept_batch) // 2
        first_half_task = self._recursive_process_batch(
            concept_batch[:mid], chain
        )
        second_half_task = self._recursive_process_batch(
            concept_batch[mid:], chain
        )
        first_results, second_results = await aio.gather(
            first_half_task, second_half_task
        )
        return first_results + second_results

    @staticmethod
    def _replace_id_with_chunk_id(
        row: pd.Series,
        id_to_citation_map: pd.Series,
    ) -> list[dict[str, Any]]:
        """
        Replaces internal source IDs in an answer with final chunk IDs.

        This function is applied to each concept row. It iterates
        through the facts in the `ANSWER` field and uses a lookup map to
        convert the internal `source_ids` (like `flat_part_id`) to the
        final, user-facing citation identifiers.

        Args:
            row: A DataFrame row containing an `ANSWER` column.
            id_to_citation_map: A pandas Series mapping an internal ID
                (index) to its corresponding final citation (value).

        Returns:
            A new `ANSWER` list with updated citation chunk IDs.
        """
        new_annotated_answer = []
        for answer_part in row[Col.ANSWER]:
            text = answer_part.get(FACT, "")
            source_ids = answer_part.get(SOURCE_IDS)

            if source_ids is None:
                new_annotated_answer.append({FACT: text, SOURCE_IDS: []})
                continue

            chunk_ids = set()
            # Ensure IDs are always in a list for consistent iteration.
            ids_list = (
                [source_ids]
                if isinstance(source_ids, (int, str))
                else source_ids
            )

            for source_id in ids_list:
                try:
                    id_value = id_to_citation_map.loc[source_id]
                    if isinstance(id_value, list):
                        chunk_ids.update(id_value)
                    else:
                        chunk_ids.add(id_value)
                except KeyError:
                    logging.warning(f"Invalid citation ID {source_id} found.")

            new_annotated_answer.append(
                {FACT: text, SOURCE_IDS: sorted(list(chunk_ids))}
            )

        return new_annotated_answer

    @staticmethod
    def _repr_question_w_answer(row: pd.Series) -> dict[str, Any]:
        """
        Creates a dict representation of a concept for the LLM.

        This function formats a concept's answer for an LLM prompt. It
        groups facts that share the exact same set of sources, joining
        their text. This creates a more compact and readable input,
        e.g., "Fact A. Fact B. [^1^][^2^]".

        Args:
            row: A DataFrame row with `QUESTION`, `ANSWER`, and `NAME`.

        Returns:
            A dictionary containing the name, question, and a single
            formatted answer string with citations.
        """

        def get_sources(part: dict | Any) -> tuple:
            """
            Helper to extract sources consistently from dict or
            object.
            """
            if isinstance(part, dict):
                return tuple(sorted(part.get(SOURCE_IDS, [])))
            return tuple(sorted(getattr(part, "citations", [])))

        def get_fact(part: dict | Any) -> str:
            """Helper to extract fact text consistently."""
            return (
                part.get(FACT, "")
                if isinstance(part, dict)
                else getattr(part, "synth_fact", "")
            )

        repr_facts = []
        # Group consecutive answer parts by their source IDs.
        for sources, group in itertools.groupby(
            row[Col.ANSWER], key=get_sources
        ):
            # Join the facts from all parts in the current group.
            full_fact = " ".join(get_fact(part) for part in group).strip()

            if full_fact:
                if sources:
                    # Format citations like: ^[^1^]^^[^2^]^
                    sources_str = f"^[^{']^^['.join(map(str, sources))}]^"
                    full_fact += sources_str
                repr_facts.append(full_fact)

        formatted_answer = " ".join(repr_facts)

        return {
            "name": row.get(Col.NAME),
            "question": row[Col.QUESTION],
            "answer": formatted_answer,
        }

    async def prettify_concepts_with_retry(
        self, batched_concepts: list[list[dict]], chain: RunnableSerializable
    ) -> list:
        """
        Processes all concept batches with a robust retry mechanism.

        This method first attempts to process all batches concurrently.
        For any batches that fail, it triggers a more robust, recursive
        divide-and-conquer processing strategy
        (`_recursive_process_batch`) to ensure maximum completion.

        Args:
            batched_concepts: A list of concept batches to be
                prettified.
            chain: The LLM chain to use for processing.

        Returns:
            A flattened list of all successfully processed concepts.
        """
        batch_info = list(enumerate(batched_concepts))
        result_map = {}

        inputs = [
            {
                Pi.QAPAIRS: batch,
                Pi.NUM_ANSWERS: len(batch),
                Pi.STYLE: self.style_instructions,
            }
            for _, batch in batch_info
        ]
        batch_results = await ChainRunner().run_chains_w_status_updates(
            [(chain, inputs)], self._put_prettification_status
        )

        retry_batches = []
        for i, ((original_idx, concept_batch), result) in enumerate(
            zip(batch_info, batch_results)
        ):
            processed = getattr(result, "pretty_concepts", [])
            if len(processed) == len(concept_batch):
                result_map[original_idx] = processed
            else:
                retry_batches.append((original_idx, concept_batch))

        if retry_batches:
            for original_idx, concept_batch in retry_batches:
                processed_batch = await self._recursive_process_batch(
                    concept_batch, chain
                )
                result_map[original_idx] = processed_batch

        all_pretty_concepts = []
        for i in range(len(batched_concepts)):
            all_pretty_concepts.extend(result_map.get(i, []))
        return all_pretty_concepts
