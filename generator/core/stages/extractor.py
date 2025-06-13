import asyncio as aio

import pandas as pd

from ...chainer.chain_factory import ChainCreator as Cc
from ...chainer.chain_factory import ChainRunner as Cr
from ...chainer.model_handler import ModelUtils as Mu
from ...utils.constants import ChainTypes as Ct
from ...utils.constants import DataFrameCols as Col
from ...utils.constants import DocCategories as DocCat
from ...utils.constants import Fn
from ...utils.constants import FrontEndStatuses as Fes
from ...utils.context import cur_llm_cost_handler
from ...utils.frontend_handler import put_status
from ...utils.logger import logging
from ...utils.misc import concat_tuples


class Extractor:
    def __init__(self, queue: aio.Queue):
        self.queue = queue

    @staticmethod
    def _parse_extraction_results(
        extraction_results: list,
    ) -> tuple[list[dict], list[dict]]:
        concepts, relations = [], []
        for chunk_id, result in enumerate(extraction_results, start=1):
            concepts.extend(
                {**concept.model_dump(), Col.FLAT_CHUNK_ID: chunk_id}
                for concept in result.concepts
            )
            relations.extend(
                {**relation.model_dump(), Col.FLAT_CHUNK_ID: chunk_id}
                for relation in result.relations
            )
        return concepts, relations

    @staticmethod
    def _make_names_unique(concept_df: pd.DataFrame) -> pd.DataFrame:
        val_counts = concept_df[Col.NAME].value_counts()
        duplicates = val_counts[val_counts > 1]
        for name in duplicates.index:
            indices = concept_df[concept_df[Col.NAME] == name].index
            for i, idx in enumerate(indices, 1):
                concept_df.loc[idx, Col.NAME] = f"{name}_{i}"
        return concept_df

    @staticmethod
    def _merge_qapair_ids(
        relation_df: pd.DataFrame,
        concept_index_df: pd.DataFrame,
        concept_name_column: str,
        new_id_column: str,
    ) -> pd.DataFrame:
        return (
            relation_df.merge(
                concept_index_df,
                left_on=[concept_name_column, Col.FLAT_CHUNK_ID],
                right_on=[Col.NAME, Col.FLAT_CHUNK_ID],
                how="left",
            )
            .drop(columns=[Col.NAME, concept_name_column])
            .rename(columns={Col.QAPAIR_ID: new_id_column})
        )

    @staticmethod
    def _update_chunk_metadata(row: pd.Series) -> pd.Series:
        # row.name is the index (id)
        row[Col.CHUNK].metadata[Col.ID] = str(row.name)
        row[Col.CHUNK].metadata[Col.DOC_TITLE] = row[Col.DOC_TITLE]
        return row

    async def _put_extraction_status(self, completed: int, total: int) -> None:
        """
        Update the extraction status with progress information.

        Args:
            completed: Number of completed tasks
            total: Total number of tasks
        """
        chunk_text = "chunk" if completed == 1 else "chunks"
        status_msg = f"{completed} {chunk_text} out of {total} processed"
        logging.info(status_msg)

        raw_percentage = (completed / total) * 100
        rounded_percentage = max(5, 5 * round(raw_percentage / 5))

        status_details = Fes.PROGRESS.format(rounded_percentage)
        await put_status(self.queue, Fes.ANALYZE_DOCS, status_details)

    async def _llm_extract(self, chunk_df: pd.DataFrame) -> list:
        link_mask = chunk_df[Col.DOC_CAT] == DocCat.LINK
        html_mask = chunk_df[Col.DOC_CAT] == DocCat.HTML
        text_mask = link_mask | html_mask

        text_contents = chunk_df.loc[text_mask, Col.CONTENT]
        text_indices = chunk_df.loc[text_mask].index
        text_inputs = Mu.form_text_inputs(text_contents)
        textual_chain = Cc.choose_chain(Ct.TEXTUAL_EXTRACTION)
        textual_chain_w_inputs = textual_chain, text_inputs

        pptx_mask = chunk_df[Col.DOC_CAT] == DocCat.PPTX

        pptx_contents = chunk_df.loc[pptx_mask, Col.CONTENT]
        pptx_indices = chunk_df.loc[pptx_mask].index
        pptx_inputs = Mu.form_multimodal_inputs(pptx_contents, DocCat.PPTX)
        pptx_chain = Cc.choose_chain(Ct.PPTX_EXTRACTION)
        pptx_chain_w_inputs = pptx_chain, pptx_inputs

        pdf_mask = chunk_df[Col.DOC_CAT] == DocCat.PDF

        pdf_contents = chunk_df.loc[pdf_mask, Col.CONTENT]
        pdf_indices = chunk_df.loc[pdf_mask].index
        pdf_inputs = Mu.form_multimodal_inputs(pdf_contents, DocCat.PDF)
        pdf_chain = Cc.choose_chain(Ct.PDF_EXTRACTION)
        pdf_chain_w_inputs = pdf_chain, pdf_inputs

        status_details = Fes.PROGRESS.format(0)
        await put_status(self.queue, Fes.ANALYZE_DOCS, status_details)
        results = await Cr().run_chains_w_status_updates(
            [textual_chain_w_inputs, pptx_chain_w_inputs, pdf_chain_w_inputs],
            self._put_extraction_status,
        )

        all_indices = (
            list(text_indices) + list(pptx_indices) + list(pdf_indices)
        )
        indexed_results = list(zip(all_indices, results))
        sorted_results = sorted(indexed_results, key=lambda x: x[0])
        return [result for _, result in sorted_results]

    @classmethod
    def _create_concept_df(
        cls,
        concepts: list[dict],
        chunk_df: pd.DataFrame,
        flat_part_df: pd.DataFrame,
    ) -> pd.DataFrame:
        concept_df = pd.DataFrame(concepts)
        chunk_cat_df = chunk_df[[Col.FLAT_CHUNK_ID, Col.DOC_CAT]]
        concept_df = pd.merge(concept_df, chunk_cat_df, on=Col.FLAT_CHUNK_ID)
        concept_df[Col.PAGE_ID] = [tuple() for _ in range(len(concept_df))]

        pptx_mask = concept_df[Col.DOC_CAT] == DocCat.PPTX
        pptx_answers = concept_df.loc[pptx_mask, Col.ANSWER]
        concept_df.loc[pptx_mask, Col.PAGE_ID] = pptx_answers.apply(
            lambda answer: tuple(
                set(
                    [
                        id_
                        for answer_part in answer
                        for id_ in answer_part[Fn.SOURCE_IDS]
                    ]
                )
            )
        )

        exploded_pptx_qapair_df = concept_df.explode(Col.PAGE_ID)
        exploded_slide_ids = exploded_pptx_qapair_df[Col.PAGE_ID]
        exploded_pptx_qapair_df[Col.PAGE_ID] = exploded_slide_ids.apply(
            lambda slide_id: (slide_id,)
        )
        exploded_pptx_qapair_df = pd.merge(
            exploded_pptx_qapair_df,
            flat_part_df[[Col.FLAT_CHUNK_ID, Col.PAGE_ID, Col.FLAT_PART_ID]],
            on=[Col.FLAT_CHUNK_ID, Col.PAGE_ID],
        )
        grouped_pptx_qapair_df = exploded_pptx_qapair_df.groupby(
            [Col.NAME, Col.QUESTION, Col.FLAT_CHUNK_ID, Col.DOC_CAT]
        )

        pptx_qapair_df = grouped_pptx_qapair_df.agg(
            {
                Col.ANSWER: "first",
                Col.FLAT_PART_ID: lambda x: list(x),
                Col.PAGE_ID: lambda x: concat_tuples(x),
            }
        ).reset_index()

        non_pptx_qapair_df = pd.merge(
            concept_df[~pptx_mask],
            flat_part_df[[Col.FLAT_CHUNK_ID, Col.PAGE_ID, Col.FLAT_PART_ID]],
            on=[Col.FLAT_CHUNK_ID, Col.PAGE_ID],
        )
        flat_part_ids = non_pptx_qapair_df[Col.FLAT_PART_ID]
        flat_part_ids_as_list = flat_part_ids.apply(lambda x: [x])
        non_pptx_qapair_df[Col.FLAT_PART_ID] = flat_part_ids_as_list

        return pd.concat(
            [non_pptx_qapair_df, pptx_qapair_df], ignore_index=True
        )

    @classmethod
    def _create_relation_df(
        cls, relations: list[dict], concept_df: pd.DataFrame
    ) -> pd.DataFrame:
        if relations:
            rel_df = pd.DataFrame(relations)
        else:
            rel_df = pd.DataFrame(
                columns=[
                    Fn.ORIGIN_CONCEPT_NAME,
                    Fn.TARGET_CONCEPT_NAME,
                    Col.FLAT_CHUNK_ID,
                ]
            )

        concept_index_df = (
            concept_df[[Col.NAME, Col.FLAT_CHUNK_ID]]
            .reset_index()
            .rename(columns={"index": Col.QAPAIR_ID})
        )

        rel_df = cls._merge_qapair_ids(
            rel_df,
            concept_index_df,
            Col.SOURCE_QAPAIR_NAME,
            Col.ORIGIN_CONCEPT_ID,
        )
        rel_df = cls._merge_qapair_ids(
            rel_df,
            concept_index_df,
            Col.TARGET_QAPAIR_NAME,
            Col.TARGET_CONCEPT_ID,
        )

        rel_df[Col.WEIGHT] = 3.0

        rows_with_nans = rel_df[rel_df.isna().any(axis=1)]
        if not rows_with_nans.empty:
            logging.warning(
                "The following relations with nonexistent qapairs "
                f"were removed: {rows_with_nans}"
            )
        rel_df.dropna(inplace=True, ignore_index=True)

        column_order = [
            Col.ORIGIN_CONCEPT_ID,
            Col.TARGET_CONCEPT_ID,
            Col.WEIGHT,
            Col.FLAT_CHUNK_ID,
        ]

        return rel_df[column_order]

    @classmethod
    def _prepare_dfs(
        cls,
        chunk_df: pd.DataFrame,
        concept_df: pd.DataFrame,
        relation_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        concept_df = cls._make_names_unique(concept_df)
        concept_df[Col.TEMP_CHUNK_ID] = concept_df[Col.FLAT_CHUNK_ID] - 1
        chunk_grouped_concept_df = concept_df.groupby(Col.TEMP_CHUNK_ID)
        chunk_df[Col.CONCEPT_IDS] = chunk_grouped_concept_df.apply(
            lambda df: df.index.tolist() if not df.empty else []
        )

        relation_df[Col.TEMP_CHUNK_ID] = relation_df[Col.FLAT_CHUNK_ID] - 1
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
        concept_df.drop(columns=[Col.TEMP_CHUNK_ID], inplace=True)
        relation_df.drop(
            columns=[Col.TEMP_CHUNK_ID, Col.FLAT_CHUNK_ID], inplace=True
        )

        return chunk_df, concept_df, relation_df

    @classmethod
    def _prepare_doc_series(cls, chunk_df: pd.DataFrame) -> pd.Series:
        chunk_df = chunk_df.apply(cls._update_chunk_metadata, axis=1)
        return chunk_df.groupby(Col.DOC_ID)[Col.CHUNK].agg(list)

    @classmethod
    async def _output_extraction_result(cls, concept_count: int):
        node_text = "node" if concept_count == 1 else "nodes"
        verb_text = "was" if concept_count == 1 else "were"
        logging.info(f"{concept_count} {node_text} {verb_text} extracted")

    async def _wrap_up(
        self,
        chunk_df: pd.DataFrame,
        concept_df: pd.DataFrame | None = None,
        relation_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
        if concept_df is not None:
            qapair_count = len(concept_df)
            await self._output_extraction_result(qapair_count)
            llm_cost_handler = cur_llm_cost_handler.get()
            logging.info(f"=Extraction LLM costs:\n{llm_cost_handler}")
            await self.queue.put(None)
            logging.info("Extraction end")
            return concept_df, relation_df, chunk_df
        return None, None, None

    async def extract_mindmap_elements(
        self, chunk_df: pd.DataFrame, flat_part_df: pd.DataFrame
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
        logging.info("Extraction start")
        extraction_results = await self._llm_extract(chunk_df)
        concepts, relations = self._parse_extraction_results(extraction_results)

        if not concepts:
            return await self._wrap_up(chunk_df)

        concept_df = self._create_concept_df(concepts, chunk_df, flat_part_df)
        relation_df = self._create_relation_df(relations, concept_df)

        chunk_df, concept_df, relation_df = self._prepare_dfs(
            chunk_df, concept_df, relation_df
        )

        return await self._wrap_up(chunk_df, concept_df, relation_df)


class AddExtractor(Extractor):
    @staticmethod
    def _parse_extraction_results_w_min_chunk(
        extraction_results: list,
        min_chunk: int,
    ) -> tuple[list[dict], list[dict]]:
        concepts, relations = [], []
        for chunk_id, result in enumerate(extraction_results, start=min_chunk):
            concepts.extend(
                {**concept.model_dump(), Col.FLAT_CHUNK_ID: chunk_id}
                for concept in result.concepts
            )
            relations.extend(
                {**relation.model_dump(), Col.FLAT_CHUNK_ID: chunk_id}
                for relation in result.relations
            )
        return concepts, relations

    @classmethod
    def _prepare_dfs_w_min_chunk(
        cls,
        chunk_df: pd.DataFrame,
        concept_df: pd.DataFrame,
        relation_df: pd.DataFrame,
        min_chunk: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        concept_df = cls._make_names_unique(concept_df)
        concept_df[Col.TEMP_CHUNK_ID] = (
            concept_df[Col.FLAT_CHUNK_ID] - min_chunk
        )
        chunk_grouped_concept_df = concept_df.groupby(Col.TEMP_CHUNK_ID)
        chunk_df[Col.CONCEPT_IDS] = chunk_grouped_concept_df.apply(
            lambda df: df.index.tolist() if not df.empty else []
        )

        relation_df[Col.TEMP_CHUNK_ID] = (
            relation_df[Col.FLAT_CHUNK_ID] - min_chunk
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
        concept_df.drop(columns=[Col.TEMP_CHUNK_ID], inplace=True)
        relation_df.drop(
            columns=[Col.TEMP_CHUNK_ID, Col.FLAT_CHUNK_ID], inplace=True
        )

        return chunk_df, concept_df, relation_df

    async def extract_mindmap_elements(
        self, chunk_df: pd.DataFrame, flat_part_df: pd.DataFrame
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
        logging.info("Extraction start")
        extraction_results = await self._llm_extract(chunk_df)
        min_chunk = chunk_df[Col.FLAT_CHUNK_ID].min()
        concepts, relations = self._parse_extraction_results_w_min_chunk(
            extraction_results, min_chunk
        )

        if not concepts:
            return await self._wrap_up(chunk_df)

        concept_df = self._create_concept_df(concepts, chunk_df, flat_part_df)
        relation_df = self._create_relation_df(relations, concept_df)

        chunk_df, concept_df, relation_df = self._prepare_dfs_w_min_chunk(
            chunk_df, concept_df, relation_df, min_chunk
        )

        return await self._wrap_up(chunk_df, concept_df, relation_df)
