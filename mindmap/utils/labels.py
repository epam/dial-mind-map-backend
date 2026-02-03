import os

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema.output_parser import StrOutputParser
from langchain_core.language_models import ParrotFakeChatModel
from langchain_core.runnables import ConfigurableField, Runnable, chain
from langchain_openai import AzureChatOpenAI
from pydantic.types import SecretStr

LABEL_PROMPT = ChatPromptTemplate.from_messages(
    [
        HumanMessagePromptTemplate.from_template(
            """You are label generator.
            You task is to generate a short (up to 3 words) label for a snippet which consists of the user's question and the answer generated from the sources.
            The label should be a short phrase that captures the essence of the snippet. It should be concise and informative.

            Snippet:
            {question}
            {answer}
            """
        ),
    ]
)


@chain
def strip_output(x: str) -> str:
    return x.strip()


def create_label_chain(
    dial_url: str, api_key: SecretStr, rag_model: str
) -> Runnable:
    llm = AzureChatOpenAI(
        temperature=0,
        model=rag_model or os.getenv("RAG_MODEL", default="gpt-4.1-2025-04-14"),
        api_version="2023-03-15-preview",
        openai_api_type="azure",
        azure_endpoint=dial_url,
        api_key=api_key,
        streaming=True,
    ).configurable_alternatives(
        ConfigurableField(id="llm"),
        default_key="llm",
        fake_llm=ParrotFakeChatModel(),
    )
    return LABEL_PROMPT | llm | StrOutputParser() | strip_output
