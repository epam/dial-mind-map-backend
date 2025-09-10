import json
from abc import ABC
from typing import Any, ClassVar

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from pydantic import BaseModel

from ...common.constants import EnvConsts
from .constants import PromptInputs as Pi
from .constants import PromptPartKeys as Ppk
from .constants import PromptPartTypes as Ppt
from .utils import load_prompt_part, save_composed_prompt

# region #################### Core Base Classes ####################


class PromptHandler(ABC):
    """
    Abstract base class for all prompt handlers.
    Provides a common interface for generating prompt components.
    """

    def get_pt_gen(
        self, inputs: dict[str, Any]
    ) -> list[SystemMessage | HumanMessage]:
        """Generates a dynamic list of messages from inputs."""
        raise NotImplementedError(
            f"{self.__class__.__name__} is designed for static prompt "
            f"templates. Use get_cpt() instead."
        )

    def get_cpt(self) -> ChatPromptTemplate:
        """Generates a static ChatPromptTemplate."""
        raise NotImplementedError(
            f"{self.__class__.__name__} is designed for dynamic message "
            f"generation. Use get_pt_gen() instead."
        )


class _BaseSystemMessageBuilder(PromptHandler):
    """
    A base class that automates the creation of system messages from
    declaratively defined prompt parts.
    """

    _default_part_key: ClassVar[str] = Ppk.DEFAULT
    _system_message_parts: ClassVar[list[tuple[str, str | None]]] = []
    _part_key_overrides: ClassVar[dict[str, str]] = {}

    @classmethod
    def _get_part_key(cls, part_type: str) -> str:
        """Gets the key for a part, allowing for overrides."""
        return cls._part_key_overrides.get(part_type, cls._default_part_key)

    @classmethod
    def _build_system_message_content(cls) -> str:
        """Builds the final string content for the system message."""
        parts = []
        for part_type, part_key_or_none in cls._system_message_parts:
            key = part_key_or_none or cls._get_part_key(part_type)
            try:
                part_content = load_prompt_part(part_type, key)
                if part_type == Ppt.EXAMPLE:
                    part_content += "\n"
                parts.append(str(part_content))
            except (KeyError, FileNotFoundError):
                continue

        return "\n".join(filter(None, parts))

    @classmethod
    def create_sys_msg(
        cls, as_template: bool = False
    ) -> SystemMessage | SystemMessagePromptTemplate:
        """
        Creates the system message, saves it for inspection, and
        returns it.
        """
        content = cls._build_system_message_content()
        if EnvConsts.SAVE_PROMPTS:
            save_composed_prompt(
                handler_name=cls.__name__, full_template=content
            )

        if as_template:
            return SystemMessagePromptTemplate(
                prompt=PromptTemplate.from_template(content)
            )
        return SystemMessage(content=content)


class _BaseHumanMessageBuilder(PromptHandler):
    """
    A mixin for handlers that build a complex human message from
    numbered sections.
    """

    _input_wrapper_key: ClassVar[str]

    @classmethod
    def _get_input_wrapper(cls) -> dict[str, str]:
        """Loads the input wrapper template from YAML."""
        return load_prompt_part(Ppt.INPUT_WRAPPER, cls._input_wrapper_key)

    @classmethod
    def _build_sections(cls, sections_data: list[tuple[str, Any]]) -> str:
        """Builds a string of numbered sections."""
        body_sections = []
        for i, (header, content) in enumerate(sections_data, 1):
            if content:
                body_sections.append(f"**{i}. {header}**\n{content}")
        return "\n\n".join(body_sections)

    @classmethod
    def _format_pydantic_or_dict(cls, data: Any) -> str:
        """Dumps a Pydantic model or dict to a YAML string."""
        if isinstance(data, BaseModel):
            dump_data = data.model_dump(exclude={"validation"})
            data = dump_data.get("refined_query", dump_data)

        return yaml.dump(
            data, sort_keys=False, default_flow_style=False, indent=2
        ).strip()


class _BaseSimpleQueryHandler(_BaseSystemMessageBuilder):
    """
    Base class for simple handlers that take a single 'query' input.
    """

    @classmethod
    def get_pt_gen(
        cls, inputs: dict[str, Any]
    ) -> list[SystemMessage | HumanMessage]:
        system_msg = cls.create_sys_msg()
        human_msg = HumanMessage(content=f"user_query: {inputs[Pi.QUERY]}")
        return [system_msg, human_msg]


# endregion

# region #################### Extraction Handlers ####################


class _BaseExtractionHandler(
    _BaseSystemMessageBuilder, _BaseHumanMessageBuilder
):
    """Base for all extraction-related handlers."""

    _default_part_key: ClassVar[str] = Ppk.MULTIMODAL
    _steps_desc_key: ClassVar[str] = _default_part_key

    _system_message_parts: ClassVar[list[tuple[str, str | None]]] = [
        (Ppt.ROLE, None),
        (Ppt.GOAL, None),
        (Ppt.EXAMPLE, Ppk.FILTER_EXAMPLE),
        (Ppt.STEPS_HEADER, Ppk.MULTIMODAL),
        (Ppt.STEPS_DESC, None),
    ]

    @classmethod
    def _get_part_key(cls, part_type: str) -> str:
        if part_type == Ppt.STEPS_DESC:
            return cls._steps_desc_key
        return super()._get_part_key(part_type)


class TextualExtraction(_BaseExtractionHandler):
    """Handler for extracting knowledge fragments from text."""

    _default_part_key: ClassVar[str] = Ppk.TEXT
    _steps_desc_key: ClassVar[str] = Ppk.TEXT
    _input_wrapper_key: ClassVar[str] = Ppk.TEXT

    @classmethod
    def get_pt_gen(
        cls, inputs: dict[str, Any]
    ) -> list[SystemMessage | HumanMessage]:
        system_msg = cls.create_sys_msg()
        input_tp = cls._get_input_wrapper()

        sections = [
            (input_tp["doc_description_header"], inputs.get(Pi.DOC_DESC)),
            (
                input_tp["filtering_criteria_header"],
                cls._format_pydantic_or_dict(inputs[Pi.FILTER]),
            ),
            (input_tp["textual_content_header"], inputs[Pi.TEXTUAL_EXTRACTION]),
        ]
        body_string = cls._build_sections(sections)

        human_content = (
            f"{input_tp['separator']}\n"
            f"{input_tp['main_title']}\n\n\n"
            f"{body_string}\n"
            f"{input_tp['separator']}"
        )
        return [system_msg, HumanMessage(content=human_content)]


class MultimodalExtractionBase(_BaseExtractionHandler):
    """Base class for multimodal (e.g., PDF, PPTX) extraction
    handlers."""

    _input_wrapper_key: ClassVar[str] = Ppk.MULTIMODAL

    @classmethod
    def get_pt_gen(
        cls, inputs: dict[str, Any]
    ) -> list[SystemMessage | HumanMessage]:
        system_msg = cls.create_sys_msg()
        input_tp = cls._get_input_wrapper()

        sections = [
            (input_tp["doc_description_header"], inputs.get(Pi.DOC_DESC)),
            (
                input_tp["filtering_criteria_header"],
                cls._format_pydantic_or_dict(inputs[Pi.FILTER]),
            ),
            (
                f"{input_tp['document_content_header'].strip()}\n"
                f"{input_tp['separator']}",
                None,
            ),
        ]
        body_string = cls._build_sections(sections)

        final_text_wrapper = (
            f"{input_tp['separator']}\n"
            f"{input_tp['main_title']}\n\n\n"
            f"{body_string}"
        )

        human_content: list[dict[str, Any]] = [
            {"type": "text", "text": final_text_wrapper}
        ]
        human_content.extend(inputs.get("multimodal_content", []))

        return [system_msg, HumanMessage(content=human_content)]


class PDFExtraction(MultimodalExtractionBase):
    _steps_desc_key: ClassVar[str] = Ppk.PDF


class PPTXExtraction(MultimodalExtractionBase):
    _steps_desc_key: ClassVar[str] = Ppk.PPT


# endregion

# region ############# Synthesis & Deduplication Handlers #############


class _BaseExampleInjectorSystemBuilder(_BaseSystemMessageBuilder):
    """
    A specialized builder for prompts that inject an example into the
    steps.
    """

    _example_part_key: ClassVar[str]
    _steps_desc_key: ClassVar[str]

    @classmethod
    def _build_system_message_content(cls) -> str:
        role = load_prompt_part(Ppt.ROLE, cls._get_part_key(Ppt.ROLE))
        steps_header = load_prompt_part(
            Ppt.STEPS_HEADER, cls._get_part_key(Ppt.STEPS_HEADER)
        )
        overall_instruction = load_prompt_part(
            Ppt.OVERALL_INSTRUCTION, cls._get_part_key(Ppt.OVERALL_INSTRUCTION)
        )

        example = load_prompt_part(Ppt.EXAMPLE, cls._example_part_key)
        raw_steps = load_prompt_part(Ppt.STEPS_DESC, cls._steps_desc_key)
        custom_steps = (
            f"{raw_steps[Ppk.BEFORE_EXAMPLE]}\n"
            f"{example}\n"
            f"{raw_steps[Ppk.AFTER_EXAMPLE]}"
        )

        parts = [role, overall_instruction, steps_header, custom_steps]
        if any(p[0] == Ppt.GOAL for p in cls._system_message_parts):
            goal = load_prompt_part(Ppt.GOAL, cls._get_part_key(Ppt.GOAL))
            parts.insert(1, goal)

        return "\n".join(filter(None, parts))


class _BaseSynthesisHandler(_BaseExampleInjectorSystemBuilder):
    """Base for Cluster and Root Cluster Synthesis."""

    _example_part_key: ClassVar[str] = Ppk.SOURCE_IDS_EXAMPLE
    _part_key_overrides: ClassVar[dict[str, str]] = {
        Ppt.STEPS_HEADER: Ppk.CLUSTER_SYNTH,
    }

    @classmethod
    def get_pt_gen(
        cls, inputs: dict[str, Any]
    ) -> list[SystemMessage | HumanMessage]:
        system_msg = cls.create_sys_msg()
        concepts_string = json.dumps(inputs[Pi.CLUSTER_SYNTH], indent=2)
        human_msg = HumanMessage(
            content=f'**Concepts:**: """{concepts_string}"""'
        )
        return [system_msg, human_msg]


class ClusterSynthesis(_BaseSynthesisHandler):
    _default_part_key: ClassVar[str] = Ppk.CLUSTER_SYNTH
    _steps_desc_key: ClassVar[str] = Ppk.CLUSTER_SYNTH


class RootClusterSynthesis(_BaseSynthesisHandler):
    _example_part_key: ClassVar[str] = Ppk.SOURCE_IDS_VS_ROOT_HAL_EXAMPLE
    _default_part_key: ClassVar[str] = Ppk.ROOT_CLUSTER_SYNTH
    _steps_desc_key: ClassVar[str] = Ppk.ROOT_CLUSTER_SYNTH


class _BaseDeduplicatorHandler(_BaseExampleInjectorSystemBuilder):
    """Base for all deduplication handlers."""

    _part_key_overrides: ClassVar[dict[str, str]] = {
        Ppt.ROLE: Ppk.CLUSTER_SYNTH
    }
    _human_message_template: ClassVar[str]

    @classmethod
    def get_cpt(cls) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                cls.create_sys_msg(as_template=True),
                ("human", cls._human_message_template),
            ]
        )


class Deduplicator(_BaseDeduplicatorHandler):
    _default_part_key: ClassVar[str] = Ppk.DEDUPLICATOR
    _steps_desc_key: ClassVar[str] = Ppk.DEDUPLICATOR
    _example_part_key: ClassVar[str] = Ppk.DEDUP_EXAMPLE
    _human_message_template: ClassVar[str] = (
        f'**Concepts:**: """{{{Pi.CLUSTER_SYNTH}}}"""'
    )


class RootDeduplicator(_BaseDeduplicatorHandler):
    _default_part_key: ClassVar[str] = Ppk.ROOT_DEDUPLICATOR
    _steps_desc_key: ClassVar[str] = Ppk.ROOT_DEDUPLICATOR
    _example_part_key: ClassVar[str] = Ppk.ROOT_DEDUP_EXAMPLE
    _human_message_template: ClassVar[str] = (
        f'**Concepts:**: """{{{Pi.CLUSTER_SYNTH}}}"""'
    )


class AddRootDeduplicator(_BaseDeduplicatorHandler):
    _default_part_key: ClassVar[str] = Ppk.ADD_ROOT_DEDUPLICATOR
    _steps_desc_key: ClassVar[str] = Ppk.ADD_ROOT_DEDUPLICATOR
    _example_part_key: ClassVar[str] = Ppk.ROOT_DEDUP_EXAMPLE
    _system_message_parts: ClassVar[list[tuple[str, str | None]]] = [
        (Ppt.GOAL, None)
    ]
    _human_message_template: ClassVar[str] = (
        '"""New concept: {new_concept}. Old concept: {old_concept}."""'
    )


# endregion

# region #################### Prettifier Handlers ####################


class _BasePrettifierHandler(
    _BaseSystemMessageBuilder, _BaseHumanMessageBuilder
):
    """Base class for Prettifier and ApplyPrettifier."""

    _default_part_key: ClassVar[str] = Ppk.PRETTIFICATION
    _input_wrapper_key: ClassVar[str] = Ppk.PRETTIFICATION
    _steps_desc_key: ClassVar[str]

    _system_message_parts: ClassVar[list[tuple[str, str | None]]] = [
        (Ppt.ROLE, None),
        (Ppt.GOAL, None),
        (Ppt.EXAMPLE, Ppk.STYLE_EXAMPLE),
        (Ppt.STEPS_HEADER, None),
        (Ppt.STEPS_DESC, None),
    ]

    @classmethod
    def _get_part_key(cls, part_type: str) -> str:
        if part_type == Ppt.STEPS_DESC:
            return cls._steps_desc_key
        return super()._get_part_key(part_type)

    @classmethod
    def get_pt_gen(
        cls, inputs: dict[str, Any]
    ) -> list[SystemMessage | HumanMessage]:
        system_msg = cls.create_sys_msg()
        input_tp = cls._get_input_wrapper()

        sections = [
            (
                input_tp["styling_instructions_header"],
                cls._format_pydantic_or_dict(inputs[Pi.STYLE]),
            ),
            (input_tp["num_concepts_header"], inputs.get(Pi.NUM_ANSWERS)),
            (
                input_tp["input_concepts_header"],
                json.dumps(inputs[Pi.QAPAIRS], indent=2),
            ),
        ]
        body_string = cls._build_sections(sections)

        human_content = (
            f"{input_tp['separator']}\n"
            f"{input_tp['main_title']}\n\n"
            f"{body_string}\n"
            f"{input_tp['separator']}"
        )
        return [system_msg, HumanMessage(content=human_content)]


class Prettifier(_BasePrettifierHandler):
    _steps_desc_key: ClassVar[str] = Ppk.PRETTIFICATION


class ApplyPrettifier(_BasePrettifierHandler):
    _steps_desc_key: ClassVar[str] = Ppk.APPLY_PRETTIFICATION


# endregion

# region ############# Validation and Refinement Handlers #############


class ValidateFilter(_BaseSimpleQueryHandler):
    _default_part_key: ClassVar[str] = Ppk.VALIDATE_FILTER
    _system_message_parts: ClassVar[list[tuple[str, str | None]]] = [
        (Ppt.ROLE, None),
        (Ppt.OVERALL_INSTRUCTION, None),
        (Ppt.EXAMPLE, Ppk.VALIDATE_FILTER_EXAMPLE),
    ]


class ValidateStyle(_BaseSimpleQueryHandler):
    _default_part_key: ClassVar[str] = Ppk.VALIDATE_STYLE
    _system_message_parts: ClassVar[list[tuple[str, str | None]]] = [
        (Ppt.ROLE, None),
        (Ppt.OVERALL_INSTRUCTION, None),
        (Ppt.EXAMPLE, Ppk.VALIDATE_STYLE_EXAMPLE),
    ]


class RefineFilter(_BaseSimpleQueryHandler):
    _default_part_key: ClassVar[str] = Ppk.REFINE_FILTER
    _system_message_parts: ClassVar[list[tuple[str, str | None]]] = [
        (Ppt.ROLE, None),
        (Ppt.GOAL, None),
        (Ppt.OVERALL_INSTRUCTION, None),
        (Ppt.EXAMPLE, Ppk.REFINE_FILTER_EXAMPLE),
    ]


class StyleDirective(_BaseSimpleQueryHandler):
    _default_part_key: ClassVar[str] = Ppk.REFINE_STYLE
    _system_message_parts: ClassVar[list[tuple[str, str | None]]] = [
        (Ppt.ROLE, None),
        (Ppt.GOAL, None),
        (Ppt.STEPS_DESC, None),
        (Ppt.EXAMPLE, Ppk.REFINE_STYLE_EXAMPLE),
    ]


# endregion

# region #################### Other Handlers ####################


class RenameConcepts(_BaseSystemMessageBuilder):
    _default_part_key: ClassVar[str] = Ppk.RENAME_CONCEPTS
    _system_message_parts: ClassVar[list[tuple[str, str | None]]] = [
        (Ppt.ROLE, None),
        (Ppt.GOAL, None),
        (Ppt.OVERALL_INSTRUCTION, None),
    ]

    @classmethod
    def get_pt_gen(
        cls, inputs: dict[str, Any]
    ) -> list[SystemMessage | HumanMessage]:
        system_msg = cls.create_sys_msg()
        human_content = (
            f"The following concepts all have the same non-unique name: "
            f"'{inputs[Pi.ORIGINAL_NAME]}'.\n\n"
            f"Here is the list of concepts to rename:\n"
            f"{inputs[Pi.CONCEPTS_JSON]}"
        )
        return [system_msg, HumanMessage(content=human_content)]


# endregion
