import asyncio as aio
from typing import AsyncGenerator
from uuid import uuid4

import numpy as np
import pandas as pd

from general_mindmap.models.request import (
    Document,
    EdgeData,
    InitMindmapRequest,
    NodeData,
    NodeMetadata,
)
from general_mindmap.v2.dial.client import DialClient
from general_mindmap.v2.generator.base import RootNodeChunk, StatusChunk

from ..utils.constants import EMPTY_NODE_DATA
from ..utils.constants import DataFrameCols as Col
from ..utils.constants import OtherBackEndConstants as Bec
from ..utils.exceptions import AddPipeException, GenPipeException
from ..utils.logger import logging
from ..utils.misc import reindex_node_df_for_duplicates
from ..utils.task_utils import cancel_tasks, process_task_w_queue
from .actions.graph import filter_by_document_id, prep_node_df_for_add
from .actions.references import fix_llm_references, transform_sources
from .stages import (
    AddConceptProcessor,
    AddExtractor,
    ConceptProcessor,
    DocChunker,
    EdgeProcessor,
    Extractor,
)
from .structs import MindMapData


class PipelineHandler:
    def __init__(self, queue: aio.Queue, dial_client: DialClient):
        self.status_queue = queue
        self.dial_client = dial_client

    @staticmethod
    async def process_results(
        mind_map_data: MindMapData,
    ) -> AsyncGenerator[NodeData | EdgeData | RootNodeChunk, None]:
        """Process Mind Map results (nodes and edges) and yield them."""
        logging.info("Process Mind Map Results: Start")

        # Set with both problematic and valid node ids
        node_ids = set()

        problematic_nodes = mind_map_data.problematic_nodes
        if problematic_nodes is not None:
            for node in problematic_nodes:
                node_ids.add(node[Bec.NODE_ID])
                yield NodeData(
                    id=node[Bec.NODE_ID],
                    label=node[Bec.NAME],
                    question=node[Bec.QUESTION],
                    details=node[Bec.ANSWER_STR],
                    metadata=node.get(Bec.METADATA),
                )

        node_df = mind_map_data.node_df
        node_df = reindex_node_df_for_duplicates(node_df, node_ids)
        if node_df is None:
            yield EMPTY_NODE_DATA
            yield RootNodeChunk(root_id=EMPTY_NODE_DATA.id)
        else:
            for idx, node in node_df.iterrows():
                answer = node[Col.ANSWER_STR]
                final_answer = fix_llm_references(answer)

                yield NodeData(
                    id=str(idx),
                    label=node[Col.NAME],
                    question=node[Col.QUESTION],
                    details=final_answer,
                    metadata=NodeMetadata(
                        answer=node[Col.ANSWER],
                        level=node[Col.LVL],
                        cluster_id=node[Col.CLUSTER_ID],
                    ),
                )

            root_id = mind_map_data.root_id
            yield RootNodeChunk(root_id=str(root_id))

        edge_df = mind_map_data.edge_df
        if edge_df is not None:
            for _, edge in edge_df.iterrows():
                edge_type = (
                    Bec.GENERATED
                    if str(edge[Col.TYPE]) == Col.ART_EDGE_TYPE_VAL
                    else Bec.INIT
                )

                yield EdgeData(
                    id=str(uuid4()),
                    source=str(edge[Col.ORIGIN_CONCEPT_ID]),
                    target=str(edge[Col.TARGET_CONCEPT_ID]),
                    type=edge_type,
                    weight=str(edge[Col.WEIGHT]),
                )

        logging.info("Process Mind Map Results: End")

    async def generate(
        self, request: InitMindmapRequest
    ) -> AsyncGenerator[StatusChunk | MindMapData, None]:
        """
        Runs the mind map generation pipeline stages.
        Yields intermediate status updates
        and finally yields the result tuple.

         'Generate' pipeline is split into 4 stages:
         1. Document chunking
         2. Mind map elements extraction
         3. Concept composing
         4. Edge processing
        """
        logging.info("'Generate' pipeline: Start")

        tasks: list[aio.Task] = []

        try:
            # 1: Document chunking
            chunking_results = None
            doc_chunker = DocChunker(self.status_queue)
            chunk_docs_coro = doc_chunker.chunk_docs(
                request.documents, self.dial_client
            )
            chunk_docs_task = aio.create_task(chunk_docs_coro)
            tasks.append(chunk_docs_task)

            async for result in process_task_w_queue(
                self.status_queue, chunk_docs_task
            ):
                if isinstance(result, StatusChunk):
                    yield result  # Yield intermediate status updates
                elif isinstance(result, tuple):
                    chunking_results = (
                        result  # Capture the final result of this stage
                    )

            if chunking_results is None:
                raise GenPipeException("Failed to chunk documents")

            chunk_df, flat_part_df = chunking_results

            # 2: Mind map elements extraction
            extraction_results = None
            extractor = Extractor(self.status_queue)
            extraction_coro = extractor.extract_mindmap_elements(
                chunk_df, flat_part_df
            )
            extraction_task = aio.create_task(extraction_coro)
            tasks.append(extraction_task)

            async for result in process_task_w_queue(
                self.status_queue, extraction_task
            ):
                if isinstance(result, StatusChunk):
                    yield result  # Yield intermediate status updates
                elif isinstance(result, tuple):
                    extraction_results = (
                        result  # Capture the final result of this stage
                    )

            concept_df, relation_df, chunk_df = extraction_results
            if concept_df is None:
                raise GenPipeException("Failed to extract mindmap elements")

            # 3: Concept composing
            concept_processing_result = None
            concept_processor = ConceptProcessor(self.status_queue)
            processor_coro = concept_processor.process_concepts(
                concept_df, relation_df, chunk_df, flat_part_df
            )
            processor_task = aio.create_task(processor_coro)
            tasks.append(processor_task)

            async for result in process_task_w_queue(
                self.status_queue, processor_task
            ):
                if isinstance(result, StatusChunk):
                    yield result  # Yield intermediate status updates
                elif isinstance(result, tuple):
                    concept_processing_result = (
                        result  # Capture the final result of this stage
                    )

            node_df_stage3, edge_df_stage3, root_id_stage3 = (
                concept_processing_result
            )
            if node_df_stage3.empty:
                raise GenPipeException("Failed to compose concepts")

            # 4: Edges processing
            edge_result = None
            edge_processor = EdgeProcessor(self.status_queue)
            process_edges_coro = edge_processor.enhance_edges(
                node_df_stage3, edge_df_stage3, root_id_stage3
            )
            process_edges_task = aio.create_task(process_edges_coro)
            tasks.append(process_edges_task)

            async for result in process_task_w_queue(
                self.status_queue, process_edges_task
            ):
                if isinstance(result, StatusChunk):
                    yield result
                elif isinstance(result, tuple):
                    edge_result = result

            edge_df_final, root_id_final = edge_result
            node_df, edge_df, root_id = (
                node_df_stage3,
                edge_df_final,
                root_id_final,
            )
            yield MindMapData(node_df=node_df, edge_df=edge_df, root_id=root_id)

        finally:
            cancel_tasks(tasks)

        logging.info("'Generate' pipeline: End")

    async def delete(
        self,
        del_docs: list[Document],
        add_docs: list[Document],
        graph_data: MindMapData,
    ) -> AsyncGenerator[StatusChunk | MindMapData, None]:
        logging.info("'Delete' pipeline: Start")

        # Use del_doc ids to delete facts referencing them
        del_doc_ids = [doc.id for doc in del_docs]
        node_df = graph_data.node_df
        node_df = filter_by_document_id(node_df, del_doc_ids)

        root_id = graph_data.root_id
        # If root node was removed due to referencing deleted documents
        if root_id not in node_df.index:
            root_id = None

        # Process changes and determine still valid node ids
        concept_processor = ConceptProcessor(self.status_queue)
        node_df = await concept_processor.process_del_changes(node_df)
        valid_indices = set(node_df.index)

        edge_df = graph_data.edge_df
        if not edge_df.empty:
            edge_df = edge_df[
                edge_df[Col.ORIGIN_CONCEPT_ID].isin(valid_indices)
                & edge_df[Col.TARGET_CONCEPT_ID].isin(valid_indices)
            ]

        if not node_df.empty:
            if not add_docs or root_id not in node_df.index:
                node_df[Col.EMBEDDING] = np.nan
                node_df[Col.IS_ACTIVE_CONCEPT] = Col.ACTIVE_CONCEPT_TRUE_VAL
                node_df = concept_processor.embed_active_concepts(node_df)

            flat_data = []

            for index, row in node_df.iterrows():
                citations_tuple = row[Col.CITATION]
                part_ids_list = row[Col.FLAT_PART_ID]

                for citation, part_id in zip(citations_tuple, part_ids_list):
                    flat_data.append(
                        {
                            Col.CITATION: citation,
                            Col.FLAT_PART_ID: part_id,
                        }
                    )

            flat_part_df = pd.DataFrame(flat_data).drop_duplicates()

            citation_to_part_id_map = flat_part_df.set_index(Col.CITATION)[
                Col.FLAT_PART_ID
            ].to_dict()

            if root_id not in node_df.index:
                node_df[Col.ANSWER] = node_df[Col.ANSWER].apply(
                    lambda x: transform_sources(x, citation_to_part_id_map)
                )
                if edge_df.empty:
                    root_df = None
                else:
                    node_connections = pd.concat(
                        [
                            edge_df[Col.ORIGIN_CONCEPT_ID],
                            edge_df[Col.TARGET_CONCEPT_ID],
                        ]
                    )
                    node_degree = node_connections.value_counts()

                    degree_df = node_degree.rename("num_neighbours").to_frame()

                    temp_sorting_df = node_df.merge(
                        degree_df, left_index=True, right_index=True, how="left"
                    )

                    temp_sorting_df["num_neighbours"] = (
                        temp_sorting_df["num_neighbours"].fillna(0).astype(int)
                    )

                    sorted_temp_df = temp_sorting_df.sort_values(
                        by=[Col.LVL, "num_neighbours"], ascending=[False, False]
                    )

                    if not node_degree.empty:
                        unique_nodes_in_edges = len(node_degree)
                        top_n_selection_count = min(unique_nodes_in_edges, 5)
                    else:
                        top_n_selection_count = 0

                    top_node_ids = sorted_temp_df.head(
                        top_n_selection_count
                    ).index.tolist()

                    if top_node_ids:
                        root_df = node_df.loc[top_node_ids].copy()
                    else:
                        root_df = pd.DataFrame(columns=node_df.columns)

                    if top_node_ids:
                        node_df.drop(top_node_ids, inplace=True)

                node_df, edge_df, root_id = await concept_processor.define_root(
                    node_df, edge_df, root_df
                )

                concept_df, relation_df, root_index = (
                    await concept_processor.dedup_concepts(
                        node_df, edge_df, root_id
                    )
                )

                concept_df = await concept_processor.format_concept_df(
                    concept_df, flat_part_df
                )

                node_df, edge_df, root_id = await concept_processor.wrap_up(
                    concept_df, relation_df, root_index
                )
                node_df = concept_processor.embed_active_concepts(node_df)

            if not add_docs:
                if Col.WEIGHT not in edge_df.columns:
                    edge_df[Col.WEIGHT] = None

                empty_weight_mask = edge_df[Col.WEIGHT].isnull()
                edge_df.loc[empty_weight_mask, Col.WEIGHT] = 0.0

                art_type_condition = empty_weight_mask & (
                    edge_df[Col.TYPE] == Col.ART_EDGE_TYPE_VAL
                )
                edge_df.loc[art_type_condition] = 0.5

                related_type_condition = empty_weight_mask & (
                    edge_df[Col.TYPE] == Col.RELATED_TYPE_VAL
                )
                edge_df.loc[related_type_condition] = 3.0

                mask = (
                    (edge_df[Col.ORIGIN_CONCEPT_ID] == root_id)
                    | (edge_df[Col.TARGET_CONCEPT_ID] == root_id)
                ) & (edge_df[Col.TYPE] == Col.RELATED_TYPE_VAL)
                edge_df.loc[mask, Col.WEIGHT] = 5.0

                edge_processor = EdgeProcessor(self.status_queue)
                process_edges_task = aio.create_task(
                    edge_processor.enhance_edges(node_df, edge_df, root_id)
                )

                async for result in process_task_w_queue(
                    self.status_queue, process_edges_task
                ):
                    if isinstance(result, StatusChunk):
                        await self.status_queue.put(result)
                    if isinstance(result, tuple):
                        edge_df, root_id = result

        logging.info("'Delete' pipeline: End")

        yield MindMapData(node_df=node_df, edge_df=edge_df, root_id=root_id)

    async def add(
        self, add_docs: list[Document], graph_data: MindMapData
    ) -> AsyncGenerator[StatusChunk | MindMapData, None]:
        logging.info("'Add' pipeline: Start")

        node_df = graph_data.node_df
        edge_df = graph_data.edge_df

        tasks: list[aio.Task] = []

        try:
            node_df, max_prev_part_id, new_node_id = prep_node_df_for_add(
                node_df
            )

            # 1: Input preparation
            chunking_results = None

            add_doc_chunker = DocChunker(self.status_queue)
            chunk_docs_task = aio.create_task(
                add_doc_chunker.chunk_add_docs(
                    add_docs, self.dial_client, max_prev_part_id
                )
            )
            tasks.append(chunk_docs_task)

            async for result in process_task_w_queue(
                self.status_queue, chunk_docs_task
            ):
                if isinstance(result, StatusChunk):
                    await self.status_queue.put(result)
                if isinstance(result, tuple):
                    chunking_results = result

            if chunking_results is None:
                raise AddPipeException("Failed to chunk documents")

            chunk_df, flat_part_df = chunking_results

            # 2: Information extraction
            extraction_results = None

            add_extractor = AddExtractor(self.status_queue)
            extract_concepts_task = aio.create_task(
                add_extractor.extract_mindmap_elements(chunk_df, flat_part_df)
            )
            tasks.append(extract_concepts_task)

            async for result in process_task_w_queue(
                self.status_queue, extract_concepts_task
            ):
                if isinstance(result, StatusChunk):
                    await self.status_queue.put(result)
                if isinstance(result, tuple):
                    extraction_results = result

            concept_df, relation_df, chunk_df = extraction_results
            if concept_df is None:
                raise AddPipeException("Failed to extract mindmap elements")

            concept_df["new"] = 1
            concept_df.index += new_node_id
            concept_df = pd.concat([node_df, concept_df])

            relation_df[Col.ORIGIN_CONCEPT_ID] += new_node_id
            relation_df[Col.TARGET_CONCEPT_ID] += new_node_id
            relation_df = pd.concat([edge_df, relation_df], ignore_index=True)

            # 3: Information processing
            concept_processing_result = None

            add_concept_processor = AddConceptProcessor(self.status_queue)
            turn_into_mindmap_task = aio.create_task(
                add_concept_processor.process_concepts(
                    concept_df, relation_df, chunk_df, flat_part_df
                )
            )
            tasks.append(turn_into_mindmap_task)

            async for result in process_task_w_queue(
                self.status_queue, turn_into_mindmap_task
            ):
                if isinstance(result, StatusChunk):
                    await self.status_queue.put(result)
                if isinstance(result, tuple):
                    concept_processing_result = result

            if concept_processing_result is None:
                raise AddPipeException("Failed to compose concepts")

            # 4: Edges processing
            mindmap_results = None

            edge_processor = EdgeProcessor(self.status_queue)
            process_edges_task = aio.create_task(
                edge_processor.enhance_edges(*concept_processing_result)
            )
            tasks.append(process_edges_task)

            async for result in process_task_w_queue(
                self.status_queue, process_edges_task
            ):
                if isinstance(result, StatusChunk):
                    await self.status_queue.put(result)
                if isinstance(result, tuple):
                    mindmap_results = result
            if mindmap_results is None:
                raise AddPipeException("Failed process edges")

            node_df = concept_processing_result[0]
            edge_df, root_id = mindmap_results

        finally:
            cancel_tasks(tasks)

        logging.info("'Add' pipeline: End")

        yield MindMapData(node_df=node_df, edge_df=edge_df, root_id=root_id)
