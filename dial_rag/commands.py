from typing import Any, List, Tuple, get_args, get_origin
from types import UnionType
from copy import deepcopy
from pydantic import BaseModel
from aidial_sdk.chat_completion import Message, Role, CustomContent, Attachment


def parse_primitive_type(type_, value: str) -> Any:
    if type_ is bool:
        return value.lower() in ["true", ""]
    return type_(value)


def consume_line(model: BaseModel, line: str) -> bool:
    for name, field in model.model_fields.items():
        cmd = f"/{name}"
        if line.startswith(cmd):
            cmd_value = line.lstrip(cmd).strip()
            if get_origin(field.annotation) is list:
                element = parse_primitive_type(
                    get_args(field.annotation)[0], cmd_value
                )
                getattr(model, name).append(element)
            elif get_origin(field.annotation) is UnionType:
                arg_type = get_args(field.annotation)[0]
                value = parse_primitive_type(arg_type, cmd_value)
                setattr(model, name, value)
            else:
                value = parse_primitive_type(field.annotation, cmd_value)
                setattr(model, name, value)

            return True
    return False


class DebugCommands(BaseModel):
    profile: bool = False
    model: str | None = None
    query_model: str | None = None


class ConfCommands(BaseModel):
    ignore_document_loading_errors: bool = False


class AttachmentCommands(BaseModel):
    attach: List[str] = []


class Commands(BaseModel):
    debug: DebugCommands = DebugCommands()
    conf: ConfCommands = ConfCommands()


def process_message_commands(
    message: Message,
    commands: Commands,
) -> Message:
    if message.content is None or message.role != Role.USER:
        return message

    lines = message.content.split("\n")
    content_lines = []

    attachment_commands = AttachmentCommands()

    command_consumers = [commands.debug, commands.conf, attachment_commands]

    for line in lines:
        if not any(consume_line(consumer, line) for consumer in command_consumers):
            content_lines.append(line)

    if len(content_lines) == len(lines):
        return message

    content = "\n".join(content_lines)

    new_message = deepcopy(message)
    new_message.content = content

    attachments = attachment_commands.attach

    if not attachments:
        # Do not add custom_content if there was not /attach commands
        return new_message

    if not new_message.custom_content:
        new_message.custom_content = CustomContent()
    if not new_message.custom_content.attachments:
        new_message.custom_content.attachments = []

    new_message.custom_content.attachments.extend(
        [Attachment(type=None, url=url) for url in attachments]
    )

    return new_message


def process_commands(
    messages: List[Message], enable_debug_commands: bool
) -> Tuple[List[Message], Commands]:

    commands = Commands()

    result_messages = [
        process_message_commands(message, commands) for message in messages
    ]

    if not enable_debug_commands:
        commands.debug = DebugCommands()

    return result_messages, commands
