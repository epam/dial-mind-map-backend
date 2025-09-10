import asyncio
import logging
from typing import AsyncGenerator, Optional, Union
from uuid import uuid4

import numpy as np
import pandas as pd

from generator.adapter import GMContract as Gmc
from generator.chainer import ChainCreator
from generator.common import type_vars as tv
from generator.common.constants import ColVals
from generator.common.constants import DataFrameCols as Col
from generator.common.exceptions import (
    AddPipeException,
    EmptyDataException,
    GenPipeException,
)
from generator.common.interfaces import FileStorage
from generator.common.structs import (
    Document,
    EdgeData,
    InitMindmapRequest,
    MMRequest,
    NodeData,
    NodeMetadata,
    RootNodeChunk,
    StatusChunk,
)

from .actions.docs import fetch_all_docs_content
from .actions.graph import (
    filter_graph_by_docs,
    prep_node_df_for_add,
    recalculate_edge_weights,
    reindex_node_df_for_duplicates,
)
from .actions.references import (
    create_flat_part_df,
    find_new_root_candidates,
    fix_llm_references,
    transform_sources,
)
from .actions.user_instructions import process_user_filter, process_user_style
from .stages import (
    AddConceptProcessor,
    ConceptDeduplicator,
    ConceptFormatter,
    ConceptProcessor,
    DocHandler,
    EdgeProcessor,
    Extractor,
    embed_active_concepts,
)
from .structs import ExtractionProduct, MindMapData, RawMindMapData
from .utils.constants import EMPTY_NODE_DATA
from .utils.task_utils import cancel_tasks, process_task_w_queue


class PipelineHandler:
    """
    Orchestrates the multi-stage pipelines for generating and modifying
    mind maps.
    """

    def __init__(
        self,
        queue: asyncio.Queue,
        file_storage: FileStorage,
        filter_instructions: dict[str, str],
        style_instructions: dict[str, str],
        request: MMRequest,
        chain_creator: ChainCreator,
    ):
        self.status_queue = queue
        self.file_storage = file_storage
        self.filter_instructions = filter_instructions
        self.style_instructions = style_instructions
        self.request = request
        self.chain_creator = chain_creator

    @classmethod
    async def create(
        cls,
        queue: asyncio.Queue,
        file_storage: FileStorage,
        request: MMRequest,
    ):
        """
        An asynchronous factory method to create a configured instance.
        """
        filter_instructions = await process_user_filter(request)
        style_instructions = await process_user_style(request, file_storage)
        chain_creator = ChainCreator("gpt-4.1-2025-04-14")
        return cls(
            queue,
            file_storage,
            filter_instructions,
            style_instructions,
            request,
            chain_creator,
        )

    async def generate(
        self, request: Optional[InitMindmapRequest] = None
    ) -> AsyncGenerator[Union[StatusChunk, MindMapData], None]:
        """
        Runs the mind map generation pipeline stages. Yields
        intermediate status updates and finally yields the result.

         'Generate' pipeline is split into 4 stages:
         1. Document chunking
         2. Mind map elements extraction
         3. Concept composing
         4. Edge processing
        """
        logging.info("'Generate' pipeline: Start")
        tasks: list[asyncio.Task] = []
        if request is None:
            request = self.request
        try:
            docs_content = await fetch_all_docs_content(
                request.documents, self.file_storage
            )

            # Stage 1: Document Chunking
            chunker = DocHandler(self.status_queue)
            chunk_task = asyncio.create_task(chunker.chunk_docs(docs_content))
            tasks.append(chunk_task)
            chunking_results: tv.ChunkingResult | None = None
            async for res in process_task_w_queue(
                self.status_queue, chunk_task
            ):
                if isinstance(res, StatusChunk):
                    yield res
                else:
                    chunking_results = res
            if chunking_results is None:
                raise GenPipeException("Failed to chunk documents")
            chunk_df, flat_part_df = chunking_results

            # Stage 2: Element Extraction
            extractor = Extractor(self.chain_creator, self.status_queue)
            extract_task = asyncio.create_task(
                extractor.extract_mindmap_elements(
                    chunk_df, flat_part_df, self.filter_instructions
                )
            )
            tasks.append(extract_task)
            extraction_product: ExtractionProduct | None = None
            async for res in process_task_w_queue(
                self.status_queue, extract_task
            ):
                if isinstance(res, StatusChunk):
                    yield res
                elif isinstance(res, ExtractionProduct):
                    extraction_product = res
            if extraction_product is None:
                raise EmptyDataException("Failed to extract mindmap elements")
            concept_df, relation_df, chunk_df = extraction_product
            if concept_df is None:
                raise EmptyDataException("Extractor returned no concepts")

            # Stage 3: Concept Processing
            processor = ConceptProcessor(
                self.chain_creator, self.status_queue, self.style_instructions
            )
            process_task = asyncio.create_task(
                processor.process_concepts(
                    concept_df, relation_df, chunk_df, flat_part_df
                )
            )
            tasks.append(process_task)
            concept_results: tv.ConceptResult | None = None
            async for res in process_task_w_queue(
                self.status_queue, process_task
            ):
                if isinstance(res, StatusChunk):
                    yield res
                else:
                    concept_results = res
            if concept_results is None or concept_results[0].empty:
                raise GenPipeException("Failed to compose concepts")
            node_df, edge_df, root_id = concept_results

            # Stage 4: Edge Processing
            edge_processor = EdgeProcessor(self.status_queue)
            edge_task = asyncio.create_task(
                edge_processor.enhance_edges(node_df, edge_df, root_id)
            )
            tasks.append(edge_task)
            edge_results: tv.EdgeResult | None = None
            async for res in process_task_w_queue(self.status_queue, edge_task):
                if isinstance(res, StatusChunk):
                    yield res
                else:
                    edge_results = res
            if edge_results is None:
                raise GenPipeException("Failed to process edges")
            edge_df, root_id = edge_results

            yield MindMapData(node_df=node_df, edge_df=edge_df, root_id=root_id)

        except EmptyDataException as e:
            logging.warning(
                f"Pipeline resulted in empty data: {e}. Returning empty graph."
            )
            yield MindMapData(
                node_df=pd.DataFrame(), edge_df=pd.DataFrame(), root_id=0
            )
        finally:
            cancel_tasks(tasks)
            logging.info("'Generate' pipeline: End")

    async def delete(
        self,
        del_docs: list[Document],
        graph_data: MindMapData,
        is_just_del: bool,
    ) -> AsyncGenerator[Union[StatusChunk, MindMapData], None]:
        """
        Runs the 'delete' pipeline to remove information from the mind
        map.
        """
        logging.info("'Delete' pipeline: Start")
        tasks: list[asyncio.Task] = []
        try:
            node_df, edge_df = filter_graph_by_docs(
                graph_data, [doc.id for doc in del_docs]
            )

            concept_processor = ConceptProcessor(
                self.chain_creator, self.status_queue, self.style_instructions
            )
            concept_formatter = ConceptFormatter(
                self.chain_creator, self.status_queue, self.style_instructions
            )
            node_df = await concept_formatter.process_del_changes(node_df)

            root_id = (
                graph_data.root_id
                if graph_data.root_id in node_df.index
                else None
            )

            if not node_df.empty:
                if root_id is None:
                    node_df, edge_df, root_id = (
                        await self._rebuild_graph_after_root_deletion(
                            node_df, edge_df, concept_processor
                        )
                    )

                if not is_just_del:
                    node_df[Col.EMBEDDING] = np.nan
                    node_df[Col.IS_ACTIVE_CONCEPT] = ColVals.TRUE_INT

                    node_df = embed_active_concepts(node_df)

                    edge_df = recalculate_edge_weights(edge_df, root_id)

                    edge_processor = EdgeProcessor(self.status_queue)
                    edge_task = asyncio.create_task(
                        edge_processor.enhance_edges(node_df, edge_df, root_id)
                    )
                    tasks.append(edge_task)
                    edge_results: tv.EdgeResult | None = None
                    async for res in process_task_w_queue(
                        self.status_queue, edge_task
                    ):
                        if isinstance(res, StatusChunk):
                            yield res
                        else:
                            edge_results = res
                    if edge_results:
                        edge_df, root_id = edge_results

            yield MindMapData(node_df=node_df, edge_df=edge_df, root_id=root_id)
        finally:
            cancel_tasks(tasks)
            logging.info("'Delete' pipeline: End")

    async def _rebuild_graph_after_root_deletion(
        self,
        node_df: pd.DataFrame,
        edge_df: pd.DataFrame,
        concept_processor: ConceptProcessor,
    ) -> tuple[pd.DataFrame, pd.DataFrame, int]:
        """
        When the root is deleted, this finds a new root and
        restructures the graph.
        """
        flat_part_df = create_flat_part_df(node_df)
        citation_map = flat_part_df.set_index(Col.CITATION)[
            Col.FLAT_PART_ID
        ].to_dict()
        node_df[Col.ANSWER] = node_df[Col.ANSWER].apply(
            lambda x: transform_sources(x, citation_map)
        )

        root_candidates_df = find_new_root_candidates(node_df, edge_df)
        if root_candidates_df is not None:
            node_df.drop(root_candidates_df.index, inplace=True)

        data = RawMindMapData(
            concept_df=node_df,
            relation_df=edge_df,
            root_df=root_candidates_df,
            chunk_df=None,
            flat_part_df=flat_part_df,
        )

        data = await concept_processor.define_root(data)
        data = await ConceptDeduplicator(self.chain_creator).deduplicate(data)

        concept_formatter = ConceptFormatter(
            self.chain_creator, self.status_queue, self.style_instructions
        )
        data = await concept_formatter.format_final_concepts(data)
        node_df, edge_df, root_id = await concept_processor.wrap_up(data)
        node_df = embed_active_concepts(node_df)

        return node_df, edge_df, root_id

    async def add(
        self,
        add_docs: list[Document],
        graph_data: MindMapData,
    ) -> AsyncGenerator[Union[StatusChunk, MindMapData], None]:
        """Runs the 'add' pipeline to integrate new documents."""
        logging.info("'Add' pipeline: Start")
        tasks: list[asyncio.Task] = []
        try:
            node_df, edge_df = graph_data.node_df, graph_data.edge_df
            node_df, max_prev_part_id, new_node_id = prep_node_df_for_add(
                node_df
            )
            docs_content = await fetch_all_docs_content(
                add_docs, self.file_storage
            )

            # Stage 1: Document Chunking
            chunker = DocHandler(self.status_queue)
            chunk_task = asyncio.create_task(
                chunker.chunk_docs(docs_content, start_part_id=max_prev_part_id)
            )
            tasks.append(chunk_task)
            chunking_results: tv.ChunkingResult | None = None
            async for res in process_task_w_queue(
                self.status_queue, chunk_task
            ):
                if isinstance(res, StatusChunk):
                    yield res
                else:
                    chunking_results = res
            if chunking_results is None:
                raise AddPipeException("Failed to chunk documents for add")
            chunk_df, flat_part_df = chunking_results

            # Stage 2: Element Extraction
            extractor = Extractor(self.chain_creator, self.status_queue)
            extract_task = asyncio.create_task(
                extractor.extract_mindmap_elements(
                    chunk_df,
                    flat_part_df,
                    self.filter_instructions,
                    is_add_mode=True,
                )
            )
            tasks.append(extract_task)
            extraction_product: ExtractionProduct | None = None
            async for res in process_task_w_queue(
                self.status_queue, extract_task
            ):
                if isinstance(res, StatusChunk):
                    yield res
                elif isinstance(res, ExtractionProduct):
                    extraction_product = res
            if extraction_product is None:
                raise EmptyDataException("Failed to extract elements for add")
            concept_df, relation_df, _ = extraction_product
            if concept_df is None:
                raise EmptyDataException(
                    "Extractor returned no concepts for add"
                )

            # Combine old and new dataframes
            concept_df["new"] = 1
            concept_df.index += new_node_id
            concept_df = pd.concat([node_df, concept_df])
            relation_df[Col.ORIGIN_CONCEPT_ID] += new_node_id
            relation_df[Col.TARGET_CONCEPT_ID] += new_node_id
            relation_df = pd.concat([edge_df, relation_df], ignore_index=True)

            # Stage 3: Process Combined Information
            processor = AddConceptProcessor(
                self.chain_creator, self.status_queue, self.style_instructions
            )
            process_task = asyncio.create_task(
                processor.process_concepts(
                    concept_df, relation_df, chunk_df, flat_part_df
                )
            )
            tasks.append(process_task)
            concept_results: tv.ConceptResult | None = None
            async for res in process_task_w_queue(
                self.status_queue, process_task
            ):
                if isinstance(res, StatusChunk):
                    yield res
                else:
                    concept_results = res
            if concept_results is None:
                raise AddPipeException("Failed to compose concepts for add")

            # Stage 4: Final Edge Processing
            edge_processor = EdgeProcessor(self.status_queue)
            edge_task = asyncio.create_task(
                edge_processor.enhance_edges(*concept_results)
            )
            tasks.append(edge_task)
            edge_results: tv.EdgeResult | None = None
            async for res in process_task_w_queue(self.status_queue, edge_task):
                if isinstance(res, StatusChunk):
                    yield res
                else:
                    edge_results = res
            if edge_results is None:
                raise AddPipeException("Failed to process final edges for add")
            node_df, _, _ = concept_results
            edge_df, root_id = edge_results

            yield MindMapData(node_df=node_df, edge_df=edge_df, root_id=root_id)

        except EmptyDataException:
            logging.warning(
                "Add pipeline resulted in no new elements; returning original "
                "graph data."
            )
            yield graph_data
        finally:
            cancel_tasks(tasks)
            logging.info("'Add' pipeline: End")

    @staticmethod
    async def process_results(
        mind_map_data: MindMapData,
    ) -> AsyncGenerator[NodeData | EdgeData | RootNodeChunk, None]:
        """
        Processes the final MindMapData object and yields structured
        node/edge data.
        """
        logging.info("Processing Mind Map Results: Start")

        node_ids = set()
        if mind_map_data.problematic_nodes:
            async for node in PipelineHandler._yield_problematic_nodes(
                mind_map_data.problematic_nodes
            ):
                node_ids.add(node.id)
                yield node

        node_df = reindex_node_df_for_duplicates(
            mind_map_data.node_df, node_ids
        )
        async for item in PipelineHandler._yield_graph_nodes(
            node_df,
            mind_map_data.root_id,
            bool(mind_map_data.problematic_nodes),
        ):
            yield item

        async for edge in PipelineHandler._yield_graph_edges(
            mind_map_data.edge_df
        ):
            yield edge

        logging.info("Processing Mind Map Results: End")

    @staticmethod
    async def _yield_problematic_nodes(
        nodes: list,
    ) -> AsyncGenerator[NodeData, None]:
        """Yields NodeData for problematic nodes."""
        for node in nodes:
            yield NodeData(
                id=node[Gmc.NODE_ID],
                label=node[Gmc.NAME],
                question=node[Gmc.QUESTION],
                details=node[Gmc.ANSWER_STR],
                metadata=node.get(Gmc.METADATA),
            )

    @staticmethod
    async def _yield_graph_nodes(
        node_df: Optional[pd.DataFrame],
        root_id: int,
        has_problematic_nodes: bool,
    ) -> AsyncGenerator[NodeData | RootNodeChunk, None]:
        """Yields NodeData for valid graph nodes and the RootNodeChunk."""
        if node_df is None or node_df.empty:
            if not has_problematic_nodes:
                yield EMPTY_NODE_DATA
                yield RootNodeChunk(root_id=EMPTY_NODE_DATA.id)
            return

        for idx, node in node_df.iterrows():
            # noinspection PyTypeChecker
            yield NodeData(
                id=str(idx),
                label=str(node[Col.NAME]),
                question=str(node[Col.QUESTION]),
                details=fix_llm_references(str(node[Col.ANSWER_STR])),
                metadata=NodeMetadata(
                    answer=list(node[Col.ANSWER]),
                    level=int(node[Col.LVL]),
                    cluster_id=int(node[Col.CLUSTER_ID]),
                ),
            )
        yield RootNodeChunk(root_id=str(root_id))

    @staticmethod
    async def _yield_graph_edges(
        edge_df: Optional[pd.DataFrame],
    ) -> AsyncGenerator[EdgeData, None]:
        """Yields EdgeData for all edges in the graph."""
        if edge_df is not None and not edge_df.empty:
            for _, edge in edge_df.iterrows():
                yield EdgeData(
                    id=str(uuid4()),
                    source=str(edge[Col.ORIGIN_CONCEPT_ID]),
                    target=str(edge[Col.TARGET_CONCEPT_ID]),
                    type=(
                        Gmc.GENERATED
                        if str(edge[Col.TYPE]) == ColVals.ARTIFICIAL
                        else Gmc.INIT
                    ),
                    weight=str(edge[Col.WEIGHT]),
                )
