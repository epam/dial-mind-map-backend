import asyncio
import logging
from email.policy import EmailPolicy
from typing import Iterable, List

from docarray import DocList

from dial_rag.attachment_link import AttachmentLink
from dial_rag.content_stream import StreamWithPrefix, SupportsWriteStr
from dial_rag.dial_config import DialConfig
from dial_rag.document_loaders import (
    load_attachment,
    load_dial_document_metadata,
    parse_document,
)
from dial_rag.document_record import (
    FORMAT_VERSION,
    Chunk,
    DocumentRecord,
    IndexSettings,
    build_chunks_list,
)
from dial_rag.errors import InvalidDocumentError, convert_and_log_exceptions
from dial_rag.image_processor.extract_pages import is_image
from dial_rag.index_storage import IndexStorage
from dial_rag.print_stats import print_chunks_stats
from dial_rag.request_context import RequestContext
from dial_rag.resources.dial_limited_resources import DialLimitedResources
from dial_rag.retrievers.bm25_retriever import BM25Retriever
from dial_rag.retrievers.description_retriever.description_retriever import (
    DescriptionIndexConfig,
    DescriptionRetriever,
)
from dial_rag.retrievers.multimodal_retriever import (
    MultimodalIndexConfig,
    MultimodalRetriever,
)
from dial_rag.retrievers.semantic_retriever import SemanticRetriever
from dial_rag.utils import format_size, timed_stage

logger = logging.getLogger(__name__)


class FailStageException(Exception):
    pass


async def check_document_access(
    request_context: RequestContext, attachment_link: AttachmentLink
):
    # Try to load document metadata to check the access to the document for the documents in the Dial filesystem.
    if not attachment_link.is_dial_document:
        return

    with timed_stage(
        request_context.choice,
        f"Access document '{attachment_link.display_name}'",
    ) as access_stage:
        try:
            await load_dial_document_metadata(request_context, attachment_link)
        except InvalidDocumentError as e:
            access_stage.append_content(e.message)
            raise


def parse_content_type(content_type):
    header = EmailPolicy.header_factory("content-type", content_type)
    return header.content_type, dict(header.params)


def get_default_image_chunk(attachment_link: AttachmentLink):
    return Chunk(
        text="",
        metadata={
            "page_number": 1,
            "source_display_name": attachment_link.display_name,
            "source": attachment_link.dial_link,
        },
    )


async def load_document_impl(
    dial_config: DialConfig,
    dial_limited_resources: DialLimitedResources,
    attachment_link: AttachmentLink,
    io_stream: SupportsWriteStr,
    index_settings: IndexSettings,
    multimodal_index_config: MultimodalIndexConfig | None,
    description_index_config: DescriptionIndexConfig | None,
) -> DocumentRecord:
    absolute_url = attachment_link.absolute_url
    headers = (
        {"api-key": dial_config.api_key.get_secret_value()}
        if absolute_url.startswith(dial_config.dial_url)
        else {}
    )

    file_name, content_type, doc_bytes = await load_attachment(
        attachment_link, headers
    )
    logger.debug(f"Successfuly loaded document {file_name} of {content_type}")
    mime_type, _ = parse_content_type(content_type)

    print(f"Document size: {format_size(len(doc_bytes))}\n", file=io_stream)
    async with asyncio.TaskGroup() as tg:

        multimodal_index_task = None
        if index_settings.multimodal_index:
            assert multimodal_index_config is not None
            multimodal_index_task = tg.create_task(
                MultimodalRetriever.build_index(
                    dial_config,
                    dial_limited_resources,
                    multimodal_index_config,
                    mime_type,
                    doc_bytes,
                    StreamWithPrefix(io_stream, "MultimodalRetriever: "),
                )
            )

        description_index_task = None
        if index_settings.use_description_index:
            assert description_index_config is not None
            description_index_task = tg.create_task(
                DescriptionRetriever.build_index(
                    dial_config,
                    dial_limited_resources,
                    description_index_config,
                    doc_bytes,
                    mime_type,
                    StreamWithPrefix(io_stream, "DescriptionRetriever: "),
                )
            )

        # TODO: try to move is_image check to the parse_document since another loader is not exposed here from the document_loaders.py
        if is_image(content_type):
            chunks_list = [get_default_image_chunk(attachment_link)]
        else:
            chunks = await parse_document(
                StreamWithPrefix(io_stream, "Parser: "),
                doc_bytes,
                mime_type,
                attachment_link,
            )
            chunks_list = await build_chunks_list(chunks)

        text_index_task = tg.create_task(
            BM25Retriever.build_index(
                chunks_list, StreamWithPrefix(io_stream, "BM25Retriever: ")
            )
        )

        embeddings_index_task = tg.create_task(
            SemanticRetriever.build_index(
                chunks_list, StreamWithPrefix(io_stream, "SemanticRetriever: ")
            )
        )

    multimodal_index = (
        multimodal_index_task.result() if multimodal_index_task else None
    )
    description_indexes = (
        description_index_task.result() if description_index_task else None
    )

    return DocumentRecord(
        format_version=FORMAT_VERSION,
        index_settings=index_settings,
        chunks=DocList(chunks_list),
        text_index=text_index_task.result(),
        embeddings_index=embeddings_index_task.result(),
        multimodal_embeddings_index=multimodal_index,
        description_embeddings_index=description_indexes,
        original_file_name=file_name,
        original_document=doc_bytes,
        mime_type=mime_type,
    )


async def load_document(
    request_context: RequestContext,
    attachment_link: AttachmentLink,
    index_storage: IndexStorage,
    index_settings: IndexSettings,
    multimodal_index_config: MultimodalIndexConfig | None,
    description_index_config: DescriptionIndexConfig | None,
) -> DocumentRecord:
    with convert_and_log_exceptions(logger):
        choice = request_context.choice

        await check_document_access(request_context, attachment_link)

        doc_record = None
        # aidial-sdk does not allow to do stage.close(Status.FAILED) inside with-statement
        try:
            with timed_stage(
                choice, f"Load indexes for '{attachment_link.display_name}'"
            ) as load_stage:
                doc_record = await index_storage.load(
                    attachment_link, index_settings, request_context
                )
                if doc_record is None:
                    raise FailStageException()
                print_chunks_stats(load_stage.content_stream, doc_record.chunks)
        except FailStageException:
            pass

        if doc_record is None:
            with timed_stage(
                choice, f"Processing document '{attachment_link.display_name}'"
            ) as doc_stage:
                io_stream = doc_stage.content_stream
                try:
                    doc_record = await load_document_impl(
                        request_context.dial_config,
                        request_context.dial_limited_resources,
                        attachment_link,
                        io_stream,
                        index_settings,
                        multimodal_index_config,
                        description_index_config,
                    )
                except InvalidDocumentError as e:
                    doc_stage.append_content(e.message)
                    raise

                print_chunks_stats(io_stream, doc_record.chunks)

            with timed_stage(
                choice, f"Store indexes for '{attachment_link.display_name}'"
            ):
                await index_storage.store(
                    attachment_link, doc_record, request_context
                )

        return doc_record


async def load_documents(
    request_context: RequestContext,
    attachment_links: Iterable[AttachmentLink],
    index_storage: IndexStorage,
    index_settings: IndexSettings,
    multimodal_index_config: MultimodalIndexConfig,
    description_index_config: DescriptionIndexConfig,
) -> List[DocumentRecord | BaseException]:
    return await asyncio.gather(
        *[
            load_document(
                request_context,
                attachment_link,
                index_storage,
                index_settings,
                multimodal_index_config,
                description_index_config,
            )
            for attachment_link in attachment_links
        ],
        return_exceptions=True,
    )
