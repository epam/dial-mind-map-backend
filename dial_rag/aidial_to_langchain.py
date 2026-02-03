import json
import os
from typing import Callable, Iterable, List

from aidial_sdk.chat_completion import Message
from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import convert_to_messages


def parse_tool_message(message: Message) -> str:
    if not message.tool_calls:
        raise ValueError("Message does not contain any tool calls.")

    arguments_str = message.tool_calls[0].function.arguments
    arguments_dict = json.loads(arguments_str)
    return arguments_dict.get("query")


def get_content_parser() -> Callable[[Message], str]:
    if os.getenv("IS_RAG_TEST_RUN") == "1":
        return parse_tool_message
    return lambda message: str(message.content)


def to_langchain_messages(
    messages: Iterable[Message],
    content_parser: Callable[[Message], str] | None = None,
) -> List[BaseMessage]:
    if content_parser is None:
        content_parser = get_content_parser()

    message_tuples = [
        (message.role.value, content_parser(message)) for message in messages
    ]
    return convert_to_messages(message_tuples)
