import sys
import traceback
from logging import Logger

from aiohttp import (
    ClientConnectorDNSError,
    ClientConnectorError,
    ClientOSError,
    ClientResponseError,
)
from pydantic import SecretStr

from dial_rag.app import (
    DESCRIPTION_INDEX_CONFIG,
    MULTIMODAL_INDEX_CONFIG,
    USE_DESCRIPTION_INDEX,
    USE_MULTIMODAL_INDEX,
)
from dial_rag.attachment_link import AttachmentLink
from dial_rag.content_stream import SupportsWriteStr
from dial_rag.dial_config import DialConfig
from dial_rag.dial_user_limits import get_user_limits_for_model
from dial_rag.document_record import (
    DocumentRecord,
    IndexSettings,
    MultimodalIndexSettings,
)
from dial_rag.documents import load_document_impl
from dial_rag.errors import NotEnoughDailyTokensError
from dial_rag.resources.dial_limited_resources import DialLimitedResources
from general_mindmap.utils.log_config import logger

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


class LoggerWriter:
    def __init__(self, logger: Logger):
        self.logger = logger

    def write(self, message: str):
        if message.strip():
            self.logger.info(message.strip())


async def build_index(
    dial_url: str, api_key: SecretStr, link: str, display_name: str
) -> DocumentRecord | None:
    dial_config = DialConfig(dial_url=dial_url, api_key=api_key)

    try:
        return await load_document_impl(
            dial_config=dial_config,
            dial_limited_resources=DialLimitedResources(
                lambda model_name: get_user_limits_for_model(
                    dial_config, model_name
                )
            ),
            attachment_link=AttachmentLink(
                dial_link=link, absolute_url=link, display_name=display_name
            ),
            io_stream=LoggerWriter(logger),
            index_settings=index_settings,
            multimodal_index_config=MULTIMODAL_INDEX_CONFIG,
            description_index_config=DESCRIPTION_INDEX_CONFIG,
        )
    except ExceptionGroup as e:
        for exception in e.exceptions:
            if isinstance(exception, NotEnoughDailyTokensError):
                raise exception

        logger.exception(e)

        return None
    except (
        ClientConnectorDNSError,
        ClientConnectorError,
        ClientOSError,
        ClientResponseError,
    ):
        return None
