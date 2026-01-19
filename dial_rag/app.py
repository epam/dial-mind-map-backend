import logging
import os

# Enable basic logging, so we can have StreamHandler logging to stdout
# before the proper logging configuration is set up by aidial_sdk
logging.basicConfig(level=logging.INFO)

import asyncio
from contextlib import asynccontextmanager
from typing import List, Tuple

import uvicorn
from aidial_sdk import DIALApp, HTTPException
from aidial_sdk.chat_completion import ChatCompletion, Request, Response
from aidial_sdk.telemetry.types import (
    MetricsConfig,
    TelemetryConfig,
    TracingConfig,
)
from langchain.retrievers import EnsembleRetriever
from langchain.schema import BaseRetriever, Document

from dial_rag.attachment_link import (
    AttachmentLink,
    format_document_loading_errors,
    get_attachment_links,
)
from dial_rag.commands import process_commands
from dial_rag.document_record import (
    Chunk,
    DocumentRecord,
    IndexSettings,
    MultimodalIndexSettings,
)
from dial_rag.documents import load_documents
from dial_rag.index_record import ChunkMetadata, RetrievalType
from dial_rag.index_storage import IndexStorage
from dial_rag.llm import LlmConfig
from dial_rag.qa_chain import ChatChainConfig, QAChainConfig, generate_answer
from dial_rag.query_chain import QueryChainConfig
from dial_rag.request_context import RequestContext, create_request_context
from dial_rag.resources.cpu_pools import warmup_cpu_pools
from dial_rag.retrievers.all_documents_retriever import AllDocumentsRetriever
from dial_rag.retrievers.bm25_retriever import BM25Retriever
from dial_rag.retrievers.description_retriever.description_retriever import (
    DescriptionIndexConfig,
    DescriptionRetriever,
)
from dial_rag.retrievers.multimodal_retriever import (
    Metric,
    MultimodalIndexConfig,
    MultimodalRetriever,
)
from dial_rag.retrievers.semantic_retriever import SemanticRetriever
from dial_rag.stages import RetrieverStage
from dial_rag.utils import (
    bool_env_var,
    float_env_var,
    int_env_var,
    profiler_if_enabled,
    timed_stage,
)

APP_NAME = "dial-rag"

DIAL_URL: str = os.getenv("DIAL_URL", "http://dial-proxy.dial-proxy")
ENABLE_DEBUG_COMMANDS: bool = bool_env_var(
    "ENABLE_DEBUG_COMMANDS", default=False
)


QA_CHAIN_CONFIG = QAChainConfig(
    chat_chain_config=ChatChainConfig(
        llm_config=LlmConfig(
            model_deployment_name=os.environ.get(
                "CHAT_DEPLOYMENT_NAME", "gpt-4.1-2025-04-14"
            ),
            max_prompt_tokens=int_env_var(
                "CHAT_HISTORY_MAX_PROMPTS_TOKENS", 16000
            ),
        ),
        use_history=bool_env_var("CHAT_USE_HISTORY", default=True),
        num_page_images_to_use=int_env_var("CHAT_NUM_PAGE_IMAGES_TO_USE", 4),
        page_image_size=int_env_var("CHAT_PAGE_IMAGE_SIZE", 1536),
    ),
    query_chain_config=QueryChainConfig(
        llm_config=LlmConfig(
            model_deployment_name=os.environ.get(
                "QUERY_DEPLOYMENT_NAME", "gpt-4.1-2025-04-14"
            ),
            max_prompt_tokens=int_env_var(
                "QUERY_HISTORY_MAX_PROMPTS_TOKENS", 8000
            ),
        ),
        use_history=bool_env_var("QUERY_USE_HISTORY", default=True),
    ),
)

USE_MULTIMODAL_INDEX: bool = bool_env_var("USE_MULTIMODAL_INDEX", default=False)
MULTIMODAL_INDEX_CONFIG = MultimodalIndexConfig(
    embeddings_model=os.environ.get(
        "MULTIMODAL_INDEX_EMBEDDINGS_MODEL", "multimodalembedding@001"
    ),
    metric=Metric(
        os.environ.get("MULTIMODAL_INDEX_METRIC", Metric.SQEUCLIDEAN_DIST)
    ),
    image_size=int_env_var("MULTIMODAL_INDEX_IMAGE_SIZE", 1536),
    estimated_task_tokens=int_env_var(
        "MULTIMODAL_INDEX_ESTIMATED_TASK_TOKENS", 500
    ),
    time_limit_multiplier=float_env_var(
        "MULTIMODAL_INDEX_TIME_LIMIT_MULTIPLIER", 1.5
    ),
    min_time_limit_sec=float_env_var(
        "MULTIMODAL_INDEX_MIN_TIME_LIMIT_SEC", 5 * 60
    ),
)

USE_DESCRIPTION_INDEX: bool = bool_env_var(
    "USE_DESCRIPTION_INDEX", default=True
)
DESCRIPTION_INDEX_CONFIG = DescriptionIndexConfig(
    llm_config=LlmConfig(
        model_deployment_name=os.environ.get(
            "DESCRIPTION_INDEX_DEPLOYMENT_NAME", "gpt-4.1-mini-2025-04-14"
        ),
        max_retries=int_env_var("DESCRIPTION_INDEX_LLM_MAX_RETRIES", 3),
        max_prompt_tokens=0,  # No limits since history is not used for description generation
    ),
    estimated_task_tokens=int_env_var(
        "DESCRIPTION_INDEX_ESTIMATED_TASK_TOKENS", 4000
    ),
    time_limit_multiplier=float_env_var(
        "DESCRIPTION_INDEX_TIME_LIMIT_MULTIPLIER", 1.5
    ),
    min_time_limit_sec=float_env_var(
        "DESCRIPTION_INDEX_MIN_TIME_LIMIT_SEC", 5 * 60
    ),
)

OTLP_EXPORT_ENABLED: bool = (
    os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") is not None
)

logger = logging.getLogger(__name__)
logger.setLevel("INFO")


def doc_to_attach(
    metadata_doc: Document, document_records: List[DocumentRecord], index=None
) -> dict | None:
    metadata = ChunkMetadata(**metadata_doc.metadata)

    doc_record: DocumentRecord = document_records[metadata["doc_id"]]
    chunk: Chunk = doc_record.chunks[metadata["chunk_id"]]
    if index is None:
        index = f"{metadata['doc_id']}.{metadata['chunk_id']}"

    if metadata["retrieval_type"] == RetrievalType.TEXT:
        type = "text/markdown"
        data = f"{chunk.text}"
    elif metadata["retrieval_type"] == RetrievalType.IMAGE:
        data = (
            f"[Image of the page {chunk.metadata['page_number']}]"
            if "page_number" in chunk.metadata
            else "[Image]"
        )
        type = "text/markdown"

    # aidial_sdk has a bug with empty string as an attachment data
    # https://github.com/epam/ai-dial-sdk/issues/167
    data = data or " "

    return dict(
        type=type,
        data=data,
        title="[{index}] {source_display_name}".format(
            **chunk.metadata, index=index
        ),
        reference_url=chunk.metadata["source"],
    )


def process_load_errors(
    docs_and_errors: List[DocumentRecord | BaseException],
    attachment_links: List[AttachmentLink],
) -> Tuple[List[DocumentRecord], List[Tuple[BaseException, AttachmentLink]]]:
    document_records: List[DocumentRecord] = []
    loading_errors: List[Tuple[BaseException, AttachmentLink]] = []

    for doc_or_error, link in zip(docs_and_errors, attachment_links):
        if isinstance(doc_or_error, DocumentRecord):
            document_records.append(doc_or_error)
        elif isinstance(doc_or_error, Exception):
            loading_errors.append((doc_or_error, link))
        else:
            # If the error is BaseException, but not Exception:
            # GeneratorExit, KeyboardInterrupt, SystemExit etc.
            raise HTTPException(
                message=f"Internal error during document loading: {str(doc_or_error)}",
                status_code=500,
            ) from doc_or_error

    return document_records, loading_errors


# TODO: Refactor this function to avoid using request_context
def create_retriever(
    request_context: RequestContext,
    document_records: List[DocumentRecord],
    multimodal_index_config: MultimodalIndexConfig | None,
) -> BaseRetriever:
    stage = lambda retriever, name: (
        RetrieverStage(
            choice=request_context.choice,
            stage_name=name,
            document_records=document_records,
            retriever=retriever,
            doc_to_attach=doc_to_attach,
        )
        if request_context.choice
        else retriever
    )

    if not AllDocumentsRetriever.is_within_limit(document_records):
        semantic_retriever = stage(
            SemanticRetriever(document_records, 7), "Embeddings search"
        )
        retrievers = [semantic_retriever]
        weights = [1.0]

        if BM25Retriever.has_index(document_records):
            bm25_retriever = stage(
                BM25Retriever(document_records, 7), "Keywords search"
            )
            retrievers.append(bm25_retriever)
            weights.append(1.0)

        if MultimodalRetriever.has_index(document_records):
            assert multimodal_index_config
            multimodal_retriever = stage(
                MultimodalRetriever(
                    request_context.dial_config,
                    multimodal_index_config,
                    document_records,
                    7,
                ),
                "Multimodal search",
            )
            retrievers.append(multimodal_retriever)
            weights.append(1.0)

        if DescriptionRetriever.has_index(document_records):
            description_retriever = stage(
                DescriptionRetriever(document_records, 7), "Page image search"
            )
            retrievers.append(description_retriever)
            weights.append(1.0)

        retriever = stage(
            EnsembleRetriever(
                retrievers=retrievers,
                weights=weights,
            ),
            "Combined search",
        )
    else:
        retriever = stage(
            AllDocumentsRetriever(document_records), "All documents"
        )

    return retriever


class DialRAGApplication(ChatCompletion):
    dial_url: str
    index_storage: IndexStorage
    enable_debug_commands: bool

    def __init__(self, dial_url: str, enable_debug_commands: bool):
        self.dial_url = dial_url
        self.index_storage = IndexStorage(self.dial_url)
        self.enable_debug_commands = enable_debug_commands
        super().__init__()

    async def chat_completion(
        self, request: Request, response: Response
    ) -> None:
        loop = asyncio.get_running_loop()
        with create_request_context(
            self.dial_url, request, response
        ) as request_context:
            choice = request_context.choice
            assert choice is not None

            messages, config = await loop.run_in_executor(
                None,
                process_commands,
                request.messages,
                self.enable_debug_commands,
            )
            attachment_links = list(
                get_attachment_links(request_context, messages)
            )

            # TODO: Need something better for the relation between IndexConfig used for the retriever and
            # the IndexSettings - the part of the IndexConfig which requires the stored index to be recalculated
            index_settings = IndexSettings(
                use_description_index=USE_DESCRIPTION_INDEX,
                multimodal_index=(
                    MultimodalIndexSettings(
                        embeddings_model=MULTIMODAL_INDEX_CONFIG.embeddings_model
                    )
                    if USE_MULTIMODAL_INDEX
                    else None
                ),
            )

            docs_and_errors = await load_documents(
                request_context,
                attachment_links,
                self.index_storage,
                index_settings,
                multimodal_index_config=MULTIMODAL_INDEX_CONFIG,
                description_index_config=DESCRIPTION_INDEX_CONFIG,
            )
            document_records, loading_errors = process_load_errors(
                docs_and_errors, attachment_links
            )

            if (
                len(loading_errors) > 0
                and not config.conf.ignore_document_loading_errors
            ):
                choice.append_content(
                    format_document_loading_errors(loading_errors)
                )
                return

            with timed_stage(choice, "Prepare indexes for search"):
                retriever = await loop.run_in_executor(
                    None,
                    create_retriever,
                    request_context,
                    document_records,
                    MULTIMODAL_INDEX_CONFIG,
                )

            last_message_content = messages[-1].content
            if last_message_content is None or not last_message_content.strip():
                return

            # TODO: Add helpers for merging configs from different sources
            properties = await request.request_dial_application_properties()
            from general_mindmap.v2.dial.client import DialClient

            client = await DialClient.create_without_request(
                DIAL_URL or "", request.headers.get("etag", ""), properties
            )

            qa_chain_config = QA_CHAIN_CONFIG.copy(deep=True)
            qa_chain_config.chat_chain_config.llm_config.model_deployment_name = client._metadata.params.get(
                "chat_model"
            ) or os.getenv(
                "RAG_MODEL", default="gpt-4.1-2025-04-14"
            )
            qa_chain_config.query_chain_config.llm_config.model_deployment_name = client._metadata.params.get(
                "chat_model"
            ) or os.getenv(
                "RAG_MODEL", default="gpt-4.1-2025-04-14"
            )
            if config.debug.model:
                qa_chain_config.chat_chain_config.llm_config.model_deployment_name = (
                    config.debug.model
                )

            if config.debug.query_model:
                qa_chain_config.query_chain_config.model_deployment_name = (
                    config.debug.query_model
                )

            with profiler_if_enabled(choice, config.debug.profile):
                reference_items = await generate_answer(
                    request_context=request_context,
                    qa_chain_config=qa_chain_config,
                    retriever=retriever,
                    messages=messages,
                    content_callback=choice.append_content,
                    document_records=document_records,
                )

            # Answer has already been streamed to the user, so we don't need to do anything here.
            for i, reference_item in enumerate(reference_items):
                if attachment := doc_to_attach(
                    reference_item, document_records, index=(i + 1)
                ):
                    choice.add_attachment(**attachment)


@asynccontextmanager
async def lifespan(app):
    await warmup_cpu_pools()
    yield


def create_app(
    dial_url: str, enable_debug_commands: bool = ENABLE_DEBUG_COMMANDS
) -> DIALApp:
    result_app = DIALApp(
        dial_url,
        propagate_auth_headers=False,
        telemetry_config=TelemetryConfig(
            service_name=APP_NAME,
            tracing=TracingConfig(
                logging=True,
                oltp_export=OTLP_EXPORT_ENABLED,
            ),
            metrics=MetricsConfig(),
        ),
        lifespan=lifespan,
    )
    result_app.add_chat_completion(
        APP_NAME, DialRAGApplication(dial_url, enable_debug_commands)
    )
    return result_app


if __name__ == "__main__":
    app = create_app(DIAL_URL)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    uvicorn.run(app, host="0.0.0.0", port=5000)
