from contextlib import contextmanager
from pydantic import BaseModel, SecretStr
from aidial_sdk.chat_completion import Request, Response, Choice

from dial_rag.dial_config import DialConfig
from dial_rag.errors import convert_and_log_exceptions
from dial_rag.resources.dial_limited_resources import DialLimitedResources
from dial_rag.dial_user_limits import get_user_limits_for_model


class RequestContext(BaseModel):
    dial_url: str
    api_key: str  # Do not use SecretStr here to avoid mixing pydantic versions
    chat_guardrails_enabled: bool
    chat_prompt: str
    chat_guardrails_prompt: str
    chat_guardrails_response_prompt: str
    choice: Choice | None  # TODO
    dial_limited_resources: DialLimitedResources

    class Config:
        # aidial_sdk.chat_completion.Choice is not a pydantic model
        arbitrary_types_allowed = True

    def is_dial_url(self, url: str) -> bool:
        return url.startswith(self.dial_url)

    @property
    def dial_base_url(self) -> str:
        return f"{self.dial_url}/v1/"

    @property
    def dial_metadata_base_url(self) -> str:
        return f"{self.dial_base_url}/metadata/"

    @property
    def dial_config(self) -> DialConfig:
        return DialConfig(dial_url=self.dial_url, api_key=SecretStr(self.api_key))

    def get_file_access_headers(self, url: str) -> dict:
        if not self.is_dial_url(url):
            return {}

        return self.get_api_key_headers()

    def get_api_key_headers(self) -> dict:
        return {"api-key": self.api_key}


@contextmanager
def create_request_context(dial_url: str, request: Request, response: Response):
    with convert_and_log_exceptions():
        with response.create_single_choice() as choice:
            dial_config = DialConfig(dial_url=dial_url, api_key=SecretStr(request.api_key))

            request_context = RequestContext(
                dial_url=dial_url,
                api_key=request.api_key,
                choice=choice,
                dial_limited_resources=DialLimitedResources(
                    lambda model_name: get_user_limits_for_model(dial_config, model_name)
                ),
            )
            yield request_context
