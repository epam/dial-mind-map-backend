import json
import os
import re
from operator import itemgetter
from typing import Any, AsyncIterator, Dict, List, Sequence, cast

from aidial_sdk.chat_completion import Message
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.language_models import ParrotFakeChatModel
from langchain_core.messages import HumanMessage, merge_content
from langchain_core.runnables import (
    ConfigurableField,
    Runnable,
    RunnableAssign,
    RunnableLambda,
    RunnableMap,
    chain,
)
from langchain_openai import AzureChatOpenAI
from pydantic.types import SecretStr

from dial_rag.app import create_retriever
from dial_rag.document_record import DocumentRecord
from dial_rag.embeddings.embeddings import BGE_EMBEDDINGS_MODEL_NAME_OR_PATH
from dial_rag.index_record import ChunkMetadata
from dial_rag.qa_chain import (
    create_docs_message,
    make_image_by_page,
    text_element,
)
from dial_rag.request_context import RequestContext
from general_mindmap.models.attachment import GRAPH_CONTENT_TYPE
from general_mindmap.models.graph import Graph
from general_mindmap.utils.cards_retriever import create_cards_retriever
from general_mindmap.utils.graph_patch import GraphPatcher
from general_mindmap.utils.labels import create_label_chain

SYSTEM_TEMPLATE = """
You are chatbot for the Mindmap.
User can ask you questions about the mindmap and the answer you give will be automatically added to the Mindmap as a new node for user to see and interact with.

You have already found the chunks of the information which may be relevant for the user question.
Using this information, you should answer the user question about the information on the Mindmap.
If users question is not clear, don't try to guess, ask for clarification.
If the question contains contradiction or contradiction with the information you found, ask for clarification.
Provide the answer only based on the information you found. If the question cannot be answered with the information you found, say that you do not know the answer, don't try to make up an answer.
Provide the links to the relevant sources or extra documentation in markdown format if the links are available.
Do not use phrase "provided" for the documentation or information you found in your answers. From the user perspective, you are the only chatbot they interact with and they do not provide any documentation to you.
Do not mention mindmap nodes in the output, because the user thinks that mindmap is also a part of the documentation.

If the user question is already answered in one of the mindmap nodes, use the existing node text as an answer. Preserve the images and links in the answer.

Anything between the 'context' and 'mindmap' xml blocks is retrieved from a knowledge bank, not part of the conversation with the user.

Cite pieces of context using [number] notation (like [2]). Cite pieces of mindmap using "{{number}}" notation (like {{1}}).
Only cite the most relevant pieces of context that answer the question accurately.
Place these citations at the end of the sentence or paragraph that reference them - do not put them all at the end.
If different citations refer to different entities within the same name, write separate answers for each entity.
If you want to cite multiple pieces of context for the same sentence, format it as `[number1] [number2]`.
However, you should NEVER do this with the same number - if you want to cite `number1` multiple times for a sentence, only do `[number1]` not `[number1] [number1]`.
"""

PROMPT_TEMPLATE = "{question}"

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
        HumanMessagePromptTemplate.from_template(PROMPT_TEMPLATE),
    ]
)


INCLUDED_ATTRIBUTES = ["source", "page_number", "title", "label", "question"]

MODEL_NAME = os.getenv("RAG_MODEL", "gpt-4o-2024-05-13")


embeddings_model = HuggingFaceBgeEmbeddings(
    model_name=BGE_EMBEDDINGS_MODEL_NAME_OR_PATH,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
    show_progress=True,
)


def format_doc(i: int, doc: Document) -> str:
    attributes = [
        (k, v) for k, v in doc.metadata.items() if k in INCLUDED_ATTRIBUTES
    ]
    attributes = [("id", f"{i + 1}")] + attributes
    doc_attributes = " ".join(f"{k}='{v}'" for k, v in attributes)
    return f"<doc {doc_attributes}>\n{doc.page_content}\n</doc>\n"


def format_docs(docs: Sequence[Document]) -> str:
    formated_docs = "\n".join(format_doc(i, doc) for i, doc in enumerate(docs))
    return f"<mindmap>\n{formated_docs}\n</mindmap>"


@chain
async def source_documents_to_attachments(input: dict) -> AsyncIterator[dict]:
    yield dict(
        type="sources",
        content={
            "chunks": input["source_documents"],
            "nodes": input["cards"],
        },
    )


def get_graph_attachment(last_message: Message) -> list | None:
    custom_content = last_message.custom_content
    if not custom_content:
        return None
    attachments = custom_content.attachments
    if not attachments:
        return None

    input_graph_attach = next(
        a.data for a in attachments if a.type == GRAPH_CONTENT_TYPE
    )
    if not input_graph_attach:
        return None

    return json.loads(input_graph_attach)


def get_node_id(last_message: Message) -> str | None:
    graph_attachment = get_graph_attachment(last_message)
    if not graph_attachment:
        return None
    assert len(graph_attachment) == 1
    return graph_attachment[0]["data"]["id"]


@chain
def get_last_message(input: dict) -> dict:
    last_message: Message = input["messages"][-1]
    new_node_id = get_node_id(last_message)
    return {"question": last_message.content, "node_id": new_node_id}


def create_llm_chain(dial_url: str, api_key: SecretStr):
    return AzureChatOpenAI(
        temperature=0,
        model=MODEL_NAME,
        azure_endpoint=dial_url,
        api_key=api_key,
        api_version="2023-03-15-preview",
        openai_api_type="azure",
        streaming=True,
    ).configurable_alternatives(
        ConfigurableField(id="llm"),
        default_key="llm",
        fake_llm=ParrotFakeChatModel(),
    )


def create_retrieval_chain(docs_retriever, extra_retriever=None):
    if extra_retriever:
        return RunnableAssign(
            RunnableMap(
                {
                    "source_documents": itemgetter("question") | docs_retriever,
                    "cards": itemgetter("question") | extra_retriever,
                }
            )
        )

    return RunnableAssign(
        RunnableMap(
            {
                "source_documents": itemgetter("question") | docs_retriever,
                "cards": lambda _: [],
            }
        )
    )


async def create_prompt(inputs):
    records = inputs["records"]
    context = inputs["context"]
    question = inputs["question"]
    mindmap = format_docs(inputs["mindmap"])

    prompt_messages = CHAT_PROMPT.invoke({"question": question}).to_messages()

    chunks_metadatas = [
        ChunkMetadata(**index_item.metadata) for index_item in context
    ]

    image_by_page = await make_image_by_page(
        records,
        chunks_metadatas,
        4,
        1536,
    )

    docs_message = create_docs_message(records, chunks_metadatas, image_by_page)

    merged_content = merge_content(
        cast(List[str | Dict], [text_element(question)]),
        cast(List[str | Dict], [text_element(mindmap)]),
        cast(List[str | Dict], [text_element("\n")]),
        cast(List[str | Dict], docs_message),
    )

    prompt_messages[-1] = HumanMessage(content=merged_content)

    return prompt_messages


def create_qa_chain(
    records,
    dial_url: str,
    api_key: SecretStr,
    docs_retriever,
    extra_retriever=None,
):
    qa_chain = create_retrieval_chain(docs_retriever, extra_retriever).assign(
        answer=RunnableLambda(
            lambda inputs: {
                "records": records,
                "context": inputs["source_documents"],
                "mindmap": inputs["cards"],
                "question": inputs["question"],
            }
        )
        | create_prompt
        | create_llm_chain(dial_url, api_key)
        | StrOutputParser(),
        attachment=source_documents_to_attachments,
    )

    return qa_chain


class Rag:
    def __init__(
        self,
        nodes_docstore: FAISS | None,
        patch_docstore: FAISS | None,
        graph_data: Graph,
        records: List[DocumentRecord],
        request_context: RequestContext,
    ) -> None:
        self.graph_data = graph_data
        self.graph_patcher = GraphPatcher(graph_data, patch_docstore)

        self.docs_retriever = create_retriever(
            request_context=request_context,
            document_records=records,
            multimodal_index_config=None,
        )

        self.cards_retriever = create_cards_retriever(
            graph_data, nodes_docstore
        )

    def create_chain(self, records, dial_url: str, api_key: SecretStr):
        qa_chain = create_qa_chain(
            records,
            dial_url,
            api_key,
            self.docs_retriever,
            self.cards_retriever,
        )

        rag_chain = get_last_message | qa_chain.assign(
            label=create_label_chain(dial_url, api_key)
        ).assign(attachment_graph=self.graph_patcher.create_chain())

        return rag_chain
