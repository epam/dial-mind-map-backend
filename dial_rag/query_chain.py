import logging
from operator import attrgetter
from typing import List

from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import chain

from dial_rag.llm import LlmConfig, create_llm
from dial_rag.request_context import RequestContext
from dial_rag.utils import timed_stage


class QueryChainConfig(BaseModel):
    llm_config: LlmConfig
    use_history: bool = True


class StandaloneQuestionCallback(BaseModel):
    question: str = Field(description="reformulated standalone question")


QUERY_SYSTEM_TEMPLATE = """
Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history.
Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
Call the StandaloneQuestionCallback to return the reformulated standalone question.
"""


EXTRACT_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(QUERY_SYSTEM_TEMPLATE),
        MessagesPlaceholder("chat_history"),
    ]
)


def get_number_of_user_messages(history: List[BaseMessage]):
    return sum(m.type == "human" for m in history)


@chain
def get_last_message(input):
    return input["chat_history"][-1].content


@chain
def log_fallback_error(input):
    logging.warning(f"Failed to extract query: {input['error']}")
    return input


def get_original_question(input: dict) -> str:
    """
    Extracts the content of the very last message in the chat history.
    """
    chat_history = input.get("chat_history", [])

    if chat_history:
        last_message = chat_history[-1]

        if hasattr(last_message, "content"):
            return last_message.content

        elif isinstance(last_message, dict):
            return last_message.get("content", "")

    return input.get("question", "")


def create_get_query_chain(
    request_context: RequestContext, query_chain_config: QueryChainConfig
):
    llm = create_llm(request_context.dial_config, query_chain_config.llm_config)

    extract_query_chain = (
        EXTRACT_QUERY_PROMPT
        | llm.with_structured_output(StandaloneQuestionCallback)
        | attrgetter("question")
    ).with_fallbacks(
        [log_fallback_error | get_last_message], exception_key="error"
    )

    @chain
    async def get_query_chain(input):
        chat_history = input["chat_history"]
        with timed_stage(
            request_context.choice, "Standalone question"
        ) as stage:
            query = await get_last_message.ainvoke(input)

            user_messages_num = get_number_of_user_messages(chat_history)
            if query_chain_config.use_history and user_messages_num > 1:
                query = await extract_query_chain.ainvoke(input)

            stage.append_content(query)
            return query

    return get_query_chain
