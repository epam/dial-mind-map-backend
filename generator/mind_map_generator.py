import asyncio as aio
from typing import AsyncGenerator

from .common.exceptions import ApplyException, GenerationException
from .common.interfaces import FileStorage
from .common.structs import (
    ApplyMindmapRequest,
    Generator,
    GeneratorStream,
    InitMindmapRequest,
    StatusChunk,
)
from .core.actions import process_graph, restore_original_ids, validate_graph
from .core.pipelines import PipelineHandler
from .core.structs import MindMapData
from .core.utils.constants import ExceptionMessage as Exc_msg
from .core.utils.constants import FrontEndStatuses as Status
from .core.utils.generator import (
    GeneratorConfigurator,
    handle_exceptions_and_logs,
)


class MindMapGenerator(Generator):
    """
    Mind Map Generator has 2 main functions:
     - Generate a mind map from scratch.
     - Apply changes made to the sources to the mind map.
    """

    def __init__(self, dial_url: str, api_key: str, file_storage: FileStorage):
        super().__init__(dial_url, api_key)
        GeneratorConfigurator.configure()
        self.status_queue = aio.Queue()
        self.file_storage = file_storage

    @handle_exceptions_and_logs
    async def generate(
        self, request: InitMindmapRequest
    ) -> AsyncGenerator[GeneratorStream, None]:
        """Generates a mind map from a set of source documents."""
        pipeline_handler = await PipelineHandler.create(
            self.status_queue, self.file_storage, request
        )

        mind_map_data = None
        async for item in pipeline_handler.generate():
            if isinstance(item, StatusChunk):
                yield item
            elif isinstance(item, MindMapData):
                mind_map_data = item

        if mind_map_data is None:
            raise GenerationException(Exc_msg.NO_MIND_MAP_DATA)

        yield StatusChunk(title=Status.SAVE_GENERATION_RESULTS)
        async for result in pipeline_handler.process_results(mind_map_data):
            yield result

    @handle_exceptions_and_logs
    async def apply(
        self, request: ApplyMindmapRequest
    ) -> AsyncGenerator[GeneratorStream, None]:
        """
        Applies document changes to an existing mind map.

        Apply includes:
        1. Add - add documents
        and integrate info from them to the existing mind map
        2. Delete - delete documents
        and remove information from them from the existing mind map
        3. Update is done as Delete + Add
        """
        if not request.add_documents and not request.del_documents:
            raise ApplyException(
                "Apply method was called without any changes to documents."
            )

        pipeline_handler = await PipelineHandler.create(
            self.status_queue, self.file_storage, request
        )

        graph_files, problematic_nodes = validate_graph(request.graph_files)
        mind_map_data, id_map = process_graph(graph_files)

        if request.del_documents:
            stream = self._stream_deletions(pipeline_handler, mind_map_data)
            async for item in stream:
                if isinstance(item, StatusChunk):
                    yield item
                elif isinstance(item, MindMapData):
                    mind_map_data = item

        if request.add_documents:
            stream = self._stream_additions(pipeline_handler, mind_map_data)
            async for item in stream:
                if isinstance(item, StatusChunk):
                    yield item
                elif isinstance(item, MindMapData):
                    mind_map_data = item

        final_mind_map_data = MindMapData(
            node_df=restore_original_ids(mind_map_data, id_map)[0],
            edge_df=restore_original_ids(mind_map_data, id_map)[1],
            root_id=mind_map_data.root_id,
            problematic_nodes=problematic_nodes,
        )

        yield StatusChunk(title=Status.SAVE_APPLY_RESULTS)
        async for result in pipeline_handler.process_results(
            final_mind_map_data
        ):
            yield result

    @staticmethod
    async def _stream_deletions(
        pipeline_handler: PipelineHandler,
        mind_map_data: MindMapData,
    ) -> AsyncGenerator[GeneratorStream | MindMapData, None]:
        """Streams all results from the deletion pipeline."""
        is_just_del = bool(pipeline_handler.request.add_documents)
        async for item in pipeline_handler.delete(
            pipeline_handler.request.del_documents, mind_map_data, is_just_del
        ):
            yield item

    @staticmethod
    async def _stream_additions(
        pipeline_handler: PipelineHandler,
        mind_map_data: MindMapData,
    ) -> AsyncGenerator[GeneratorStream | MindMapData, None]:
        """
        Streams all results from the addition pipeline, handling the
        empty graph case.
        """
        if mind_map_data.node_df.empty:
            gen_request = InitMindmapRequest(
                documents=pipeline_handler.request.add_documents
            )
            async for item in pipeline_handler.generate(gen_request):
                yield item
        else:
            async for item in pipeline_handler.add(
                pipeline_handler.request.add_documents, mind_map_data
            ):
                yield item
