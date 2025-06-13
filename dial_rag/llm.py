from langchain_core.pydantic_v1 import BaseModel
from langchain_openai.chat_models import AzureChatOpenAI

from dial_rag.dial_config import DialConfig


class LlmConfig(BaseModel):
    model_deployment_name: str  # model_name is reserved in pydantic
    max_prompt_tokens: int = 0  # 0 means no limit
    max_retries: int = 2


def create_llm(dial_config: DialConfig, llm_config: LlmConfig):
    extra_body = {}
    if llm_config.max_prompt_tokens:
        extra_body["max_prompt_tokens"] = llm_config.max_prompt_tokens

    llm = AzureChatOpenAI(
        azure_endpoint=dial_config.dial_url,
        api_key=dial_config.api_key,
        model=llm_config.model_deployment_name,
        api_version="2023-03-15-preview",
        openai_api_type="azure",
        temperature=0,
        streaming=True,
        max_retries=llm_config.max_retries,
        extra_body=extra_body,
    )
    return llm
