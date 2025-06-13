import asyncio as aio
from typing import AsyncGenerator

from general_mindmap.models.request import (
    ApplyMindmapRequest,
    EdgeData,
    InitMindmapRequest,
    NodeData,
)
from general_mindmap.v2.dial.client import DialClient
from general_mindmap.v2.generator.base import (
    DocStatusChunk,
    Generator,
    RootNodeChunk,
    StatusChunk,
)

from .core.actions import process_graph, restore_original_ids, validate_graph
from .core.pipelines import PipelineHandler
from .core.structs import MindMapData
from .utils.configurator import Configurator
from .utils.constants import FrontEndStatuses as Status
from .utils.context import cur_llm_cost_handler
from .utils.exceptions import (
    AddPipeException,
    ApplyException,
    DelPipeException,
    GenPipeException,
)
from .utils.logger import logging


class MindMapGenerator(Generator):
    """
    Mind Map Generator has 2 main functions:
     - Generate a mind map from scratch.
     - Apply changes made to the sources to the mind map.
    """

    def __init__(self, dial_url: str, api_key: str, dial_client: DialClient):
        super().__init__(dial_url, api_key)

        Configurator.configure_generator()

        # dial_client is used to read files in DIAL storage.
        self.dial_client = dial_client
        self.status_queue = aio.Queue()

    async def generate(
        self, request: InitMindmapRequest
    ) -> AsyncGenerator[
        StatusChunk | NodeData | EdgeData | DocStatusChunk | RootNodeChunk, None
    ]:
        """Generates a mind map from scratch."""
        logging.info("Generation: Start")
        mind_map_data = None
        pipeline_handler = PipelineHandler(self.status_queue, self.dial_client)

        # noinspection PyUnreachableCode
        try:
            async for item in pipeline_handler.generate(request):
                if isinstance(item, StatusChunk):
                    yield item
                elif isinstance(item, MindMapData):
                    mind_map_data = item

        except GenPipeException as e:
            logging.exception(e.msg)

        finally:
            llm_cost_handler = cur_llm_cost_handler.get()
            logging.info(f"Total LLM costs:\n{llm_cost_handler}")
            yield StatusChunk(title=Status.SAVE_GENERATION_RESULTS)
            async for result in pipeline_handler.process_results(mind_map_data):
                yield result

        logging.info("Generation: End")

    async def apply(
        self, request: ApplyMindmapRequest
    ) -> AsyncGenerator[
        StatusChunk | NodeData | EdgeData | DocStatusChunk | RootNodeChunk, None
    ]:
        """
        Apply includes:
        1. Add - add documents
        and integrate info from them to the existing mind map
        2. Delete - delete documents
        and remove information from them from the existing mind map

        Update is done as Delete + Add
        """
        logging.info("Apply: Start")
        pipeline_handler = PipelineHandler(self.status_queue, self.dial_client)

        try:
            # To make sure that at least 1 pipeline was run
            pipeline_run = False

            graph_files = request.graph_files
            graph_files, problematic_nodes = validate_graph(graph_files)
            mind_map_data, id_map = process_graph(graph_files)

            del_docs = request.del_documents
            add_docs = request.add_documents
            if del_docs:
                pipeline_run = True
                async for item in pipeline_handler.delete(
                    del_docs, add_docs, mind_map_data
                ):
                    if isinstance(item, StatusChunk):
                        yield item
                    elif isinstance(item, MindMapData):
                        mind_map_data = item

            if mind_map_data.node_df.empty:
                pipeline_run = True
                request = InitMindmapRequest(documents=add_docs)
                async for item in pipeline_handler.generate(request):
                    if isinstance(item, StatusChunk):
                        yield item
                    elif isinstance(item, MindMapData):
                        mind_map_data = item

            elif add_docs:
                pipeline_run = True
                async for item in pipeline_handler.add(add_docs, mind_map_data):
                    if isinstance(item, StatusChunk):
                        yield item
                    elif isinstance(item, MindMapData):
                        mind_map_data = item

            if pipeline_run is False:
                raise ApplyException("Apply was called without changes.")

            node_df, edge_df = restore_original_ids(mind_map_data, id_map)
            root_id = mind_map_data.root_id
            mind_map_data = MindMapData(
                node_df=node_df,
                edge_df=edge_df,
                root_id=root_id,
                problematic_nodes=problematic_nodes,
            )

            yield StatusChunk(title=Status.SAVE_APPLY_RESULTS)
            async for result in pipeline_handler.process_results(mind_map_data):
                yield result

        except (
            GenPipeException,
            AddPipeException,
            DelPipeException,
            ApplyException,
        ) as e:
            logging.exception(e.msg)
            raise e

        finally:
            llm_cost_handler = cur_llm_cost_handler.get()
            logging.info(f"Total LLM costs:\n{llm_cost_handler}")

        logging.info("Apply: End")
