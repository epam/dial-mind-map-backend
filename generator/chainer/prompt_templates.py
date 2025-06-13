from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from .constants import FieldNames as Fn
from .constants import PromptInputs


class Examples:
    """Example strings used across multiple prompt handlers."""

    answer_citations_example = (
        "Example of including citations:\n"
        "<example>\n"
        f"Input {Fn.CONCEPTS}: ["
        '"Name: Inflation in the USA: '
        "Question: What was the inflation in the USA in 2019 - "
        "Answer: "
        "[{'fact': 'Inflation in the USA was 3% in 2019.', "
        "'source_ids': [1]}]\","
        '"Name: Inflation in the USA_2: '
        "Question: What was the inflation in the USA around 2019 - "
        "Answer: "
        "[{'fact': 'Inflation in the USA was around 3% in 2019.', "
        "'source_ids': [1]}]\","
        '"Name: Inflation in the USA in December of 2019: '
        "Question: What was the inflation in the USA in December of 2019 - "
        "Answer: "
        "[{'fact': 'Inflation in the USA was 3% in December of 2019.',"
        " 'source_ids': [3]}]\"]"
        "\n"
        "Output: QAPair(synth_name='Inflation in the USA', "
        "synth_question='What was the inflation in the USA around 2019?', "
        "synth_answer=["
        "SynthAnswerPart("
        "synth_fact='Inflation in the USA was 3% in 2019, around 2019, "
        "and December of 2019.', citations=[1, 3])]"
        "\n</example>"
    )

    url_text = (
        "In XML tags you can find a good and a bad examples "
        "of handling URLs. "
        "Strive for the good one and avoid the bad one. "
        "URL usage examples: "
        "<good_example>[google](https://www.google.com/)</good_example> "
        "<bad_example>[here](https://www.google.com/)</bad_example>"
    )


class PromptHandler(ABC):
    """Base class for all prompt handlers with common functionality."""

    # Common prompt components
    role = (
        "You are a highly capable, thoughtful, and precise multilingual "
        "assistant. Always respond in the language of the user's input."
    )

    goal = (
        "Your goal is to deeply understand the user's intent, "
        "ask clarifying questions when needed, "
        "think step-by-step through complex problems, "
        "provide clear and accurate answers, "
        "and proactively anticipate helpful follow-up information."
    )

    overall_instruction = (
        "Always prioritize being truthful, nuanced, insightful, "
        "and efficient, tailoring your responses specifically "
        "to the user's needs and preferences."
    )

    steps_header = (
        "# Use the following step-by-step instructions "
        "to respond to user input."
    )

    steps = ""

    language = (
        "Respond in the same language as the user input. "
        "Ensure that all concept names, questions, and answers "
        "are provided in the user's language. "
        "If the language is unclear or uncertain, default to English."
    )

    @classmethod
    def create_sys_msg(cls) -> SystemMessage:
        """Creates a system message from the class's prompt components."""
        return SystemMessage(
            content="\n".join(
                filter(
                    None,
                    (
                        cls.role,
                        cls.goal,
                        cls.overall_instruction,
                        cls.steps_header,
                        cls.steps,
                        cls.language,
                    ),
                )
            )
        )

    @classmethod
    @abstractmethod
    def get_cpt(cls) -> ChatPromptTemplate:
        """Returns a ChatPromptTemplate using the handler's components."""
        pass


class TextualExtraction(PromptHandler):
    """Handler for extracting concepts from text."""

    goal = (
        "Your goal is to represent all concepts "
        "and relations between them "
        "from the text delimited by triple quotes."
    )

    overall_instruction = (
        "Your response has to adhere strictly to the given format. "
        "Think step-by-step and provide clear and accurate answers. "
        "Always prioritize being truthful, nuanced, insightful, "
        "and efficient, tailoring your responses specifically "
        "to the provided format. "
        "Treat the text as part of a broader domain or corpus. "
        "Ensure that your extraction is exhaustive and faithful "
        "to the text, "
        "without adding any information not explicitly mentioned. "
        "The concepts and relations you extract "
        "have to accurately reflect the content of the provided text "
        "and, where possible, the broader context of the entire corpus. "
        "Your output should be clear and accessible to a broad audience."
    )

    steps = (
        "## Step 1: Extract Concepts\n"
        "Identify all concepts mentioned within the text "
        "enclosed by triple quotes. "
        "Ensure that your output is exhaustive, "
        "capturing every detail. "
        "If a URL is associated with a concept make sure to include it "
        "in Markdown format to the answer associated with the concept.\n"
        "## Step 2: Determine relations\n"
        "Analyze the text enclosed by triple quotes "
        "to determine the relations between the identified concepts "
        "indicating how one may inform or relate to another.\n"
        "## Step 3: Structure the Information\n"
        "For each concept, formulate a relevant question. "
        "Compose an answer for each question "
        "using the information provided in the text "
        "enclosed by triple quotes. "
        "Each answer has to consist from parts. "
        "Each answer part has to consist from a fact. "
        "Facts must encompass all available data, such as URLs, "
        "among other details. "
        "There are no sources for the facts "
        "from the text delimited by triple quotes. "
        "Assign a concise and unique name to each concept. "
        "Use the names of the origin and target concepts "
        "to represent each relation.\n"
        "## Step 4: Adhere to Response Format\n"
        "Revise questions, answers, and concept names as necessary "
        "to comply with the specified response format."
    )

    @classmethod
    def get_cpt(cls) -> ChatPromptTemplate:
        """Returns a ChatPromptTemplate for textual extraction."""
        return ChatPromptTemplate(
            [
                cls.create_sys_msg(),
                ("human", f'"""{{{PromptInputs.TEXTUAL_EXTRACTION}}}"""'),
            ]
        )


class ClusterSynthesis(PromptHandler):
    """Handler for synthesizing clusters of concepts."""

    goal = (
        "Your goal is to synthesize a coherent and meaningful cluster "
        "from the concepts delimited by triple quotes. "
        "Each concept consists of a name, question, and answer. "
        "A cluster is a group of concepts where each concept is unique, "
        "the root concept that represents the broadest, "
        "most general idea in the cluster is defined, and "
        "relations between related concepts are determined."
    )

    overall_instruction = (
        "Your response has to adhere strictly to the given format. "
        "Think step-by-step and provide clear and accurate answers. "
        "Always prioritize being truthful, nuanced, insightful, "
        "and efficient, tailoring your responses specifically "
        "to the provided format. "
        "Ensure that your output is comprehensive and faithful "
        "to the input, "
        "without adding any information not explicitly mentioned. "
        "The concepts and relations you output "
        "have to accurately reflect the content of the source concepts. "
        "Your output should be clear and accessible to a broad audience."
    )

    steps = (
        "## Step 1: Synthesize Unique Concepts\n"
        "Iteratively review and combine similar concepts "
        "provided within triple quotes "
        "until all concepts are unique and represent distinct, ideas. "
        "Combination of concepts means combining their names, questions, "
        "and answers. For each answer part make sure to include citations "
        "as numbers in square brackets from the source answers. "
        "Facts themselves must not have source inside, they should be "
        "specified separately in citations field."
        f"\n{Examples.answer_citations_example}\n"
        "Ensure that each synthesized concept is clear and well-defined.\n"
        "## Step 2: Identify relations\n"
        "Analyze the newly synthesized concepts "
        "and define the relations between them. "
        "## Step 3: Define the Root Concept\n"
        "Identify the root concept, "
        "which represents the broadest and most general idea "
        "within the cluster. "
        "You may:\n"
        "- Select one of the newly synthesized concepts "
        "and refine it to serve as the root.\n"
        "- Create a completely new concept "
        "if none of the existing ones are broad enough to fulfill this role.\n"
        "Ensure the root concept is distinct "
        "and serves as the foundation for the cluster.\n"
        "## Step 4: Adhere to Response Format\n"
        "Revise questions, answers, and concept names as necessary "
        "to comply with the specified response format. "
        "Exclude the root concept from the list of synthesized concepts."
    )

    language = None

    @classmethod
    def get_cpt(cls) -> ChatPromptTemplate:
        """Returns a ChatPromptTemplate for cluster synthesis."""
        return ChatPromptTemplate(
            [
                cls.create_sys_msg(),
                ("human", f'Concepts: """{{{PromptInputs.CLUSTER_SYNTH}}}"""'),
            ]
        )


class Deduplicator(PromptHandler):
    """Prompt Handler for deduplicating concepts."""

    goal = (
        "Your goal is to synthesize a coherent and meaningful cluster "
        "from the concepts delimited by triple quotes. "
        "Each concept consists of a name, question, and answer. "
        "A cluster is a group of concepts where each concept is unique, "
        "and has a unique name "
        "and relations between related concepts are determined."
    )

    overall_instruction = ClusterSynthesis.overall_instruction

    steps = (
        "## Step 1: Synthesize Unique Concepts\n"
        "Iteratively review and combine similar concepts "
        "provided within triple quotes "
        "until all concepts are unique and represent distinct, ideas. "
        "Combination of concepts means combining their names, questions, "
        "and answers. For each answer part make sure to include citations "
        "as numbers in square brackets from the source answers. "
        "Facts themselves must not have source inside, they should be "
        "specified separately in citations field."
        f"\n{Examples.answer_citations_example}\n"
        "Ensure that each synthesized concept has a unique name and "
        "is clear and well-defined.\n"
        "## Step 2: Identify Relations\n"
        "Analyze the newly synthesized concepts "
        "and define the relations between them. "
        "## Step 3: Adhere to Response Format\n"
        "Revise questions, answers, and concept names as necessary "
        "to comply with the specified response format. "
    )

    language = None

    @classmethod
    def get_cpt(cls) -> ChatPromptTemplate:
        """Returns a ChatPromptTemplate for deduplication."""
        return ChatPromptTemplate(
            [
                cls.create_sys_msg(),
                ("human", f'Concepts: """{{{PromptInputs.CLUSTER_SYNTH}}}"""'),
            ]
        )


class RootDeduplicator(PromptHandler):
    """Handler for deduplicating concepts with a root concept."""

    goal = (
        "Your goal is to synthesize a coherent and meaningful cluster "
        "from the root and other concepts delimited by triple quotes. "
        "Each concept consists of a name, question, and answer. "
        "A cluster is a group of concepts where each concept is unique, "
        "the root concept that represents the broadest, "
        "most general idea in the cluster is defined, and "
        "relations between related concepts are determined."
    )

    overall_instruction = ClusterSynthesis.overall_instruction

    steps = (
        "## Step 1: Synthesize Unique Concepts\n"
        "Iteratively review and combine similar concepts "
        "provided within triple quotes "
        "until all concepts are unique and represent distinct, ideas. "
        "Combination of concepts means combining their names, questions, "
        "and answers. Do not use the root concept as a source for merging; "
        "only incorporate new information into it if applicable. "
        "For each answer part make sure to include citations "
        "as numbers in square brackets from the source answers. "
        "Facts themselves must not have source inside, they should be "
        "specified separately in citations field."
        f"\n{Examples.answer_citations_example}\n"
        "Ensure that each synthesized concept is clear and well-defined.\n"
        "## Step 2: Identify relations\n"
        "Analyze the newly synthesized concepts "
        "and define the relations between them. "
        "## Step 3: Adhere to Response Format\n"
        "Revise questions, answers, and concept names as necessary "
        "to comply with the specified response format. "
        "Exclude the root concept from the list of synthesized concepts."
    )

    language = None

    @classmethod
    def get_cpt(cls) -> ChatPromptTemplate:
        """Returns a ChatPromptTemplate for root deduplication."""
        return ChatPromptTemplate(
            [
                cls.create_sys_msg(),
                ("human", f'Concepts: """{{{PromptInputs.CLUSTER_SYNTH}}}"""'),
            ]
        )


class AddRootDeduplicator(PromptHandler):
    """Prompt Handler for deduplicating concepts."""

    goal = (
        "Your goal is to synthesize a coherent and meaningful cluster "
        "from the concepts delimited by triple quotes. "
        "Each concept consists of a name, question, and answer. "
        "A cluster is a group of concepts where each concept is unique, "
        "and has a unique name "
        "and relations between related concepts are determined."
    )

    overall_instruction = ClusterSynthesis.overall_instruction

    steps = (
        "## Step 1: Update the New Concept\n"
        "Combination of concepts means combining their names, questions, "
        "and answers. For each answer part make sure to include citations "
        "as numbers in square brackets from the source answers. "
        "Facts themselves must not have source inside, they should be "
        "specified separately in citations field."
        f"\n{Examples.answer_citations_example}\n"
        "Ensure that the new concept absorbs information from the "
        "old concept if it does not have it included already. "
        "## Step 2: Adhere to Response Format\n"
        "Revise questions, answers, and concept names as necessary "
        "to comply with the specified response format. "
    )

    language = None

    @classmethod
    def get_cpt(cls) -> ChatPromptTemplate:
        """Returns a ChatPromptTemplate for deduplication."""
        return ChatPromptTemplate(
            [
                cls.create_sys_msg(),
                (
                    "human",
                    '"""New concept: {new_concept}. Old concept: {old_concept}."""',
                ),
            ]
        )


class Prettifier(PromptHandler):
    """Handler for prettifying answers in question-answer pairs."""

    goal = (
        "Your goal is to refine the answers in the "
        f"{{{PromptInputs.NUM_ANSWERS}}} question answer pairs "
        f"within triple quotes. "
        "Correct any errors and enhance the logical structure "
        "to improve its overall presentation, "
        "ensuring that all original details and content are preserved "
        "and no new information is added. "
        "Make sure to include a refined answer for each of the provided "
        "question answer pairs."
    )

    overall_instruction = (
        "Always prioritize being truthful, nuanced, insightful, "
        "and efficient, tailoring your responses specifically "
        "to the user's needs and preferences."
        "Freely use Markdown format features like lists and text formatting. "
        "Make sure that all URLs you provided are formatted using Markdown "
        "and the text used to represent the URL "
        "is descriptive and provides context about what the URL links to. "
        f"\n{Examples.url_text}\n"
        "If you don't know what text to use better use the raw URL itself "
        'instead of generic terms like "here" as link text. '
        'Never use "here" as text used to represent a URL.\n'
        "Retain all citations from the original answers. "
        "Structure answers as straight responses to the questions "
        "without a header in the beginning. "
        "Make sure that all the citations from initial answers are used "
        "for relevant parts of the answer as in the sources. "
        "Cite pieces of the answer using the notation ^[marker]^. "
        "The 'marker' inside the brackets should be either a single number "
        "(e.g., ^[2]^) or three numbers separated by dots (e.g., ^[12.1.1]^)."
        "Place these citations at the end of the sentence "
        "or paragraph that reference them - do not put them all at the end. "
        "Place references before the period at the end of the sentence. "
        "If you want to cite multiple pieces of context "
        "for the same sentence, format it as `^[marker1]^ ^[marker2]^`."
        "However, you should NEVER do this with the same marker - "
        "if you want to cite `marker1` multiple times for a sentence, "
        "only do `^[marker1]^` not `^[marker1]^ ^[marker1]^`. "
        "Return only answers. "
        "Order of the answers has to be the same. "
        "Make sure to include a refined answer for each of the provided "
        "question answer pairs."
    )

    language = None

    @classmethod
    def get_cpt(cls) -> ChatPromptTemplate:
        """Returns a ChatPromptTemplate for prettifying."""
        return ChatPromptTemplate(
            [
                cls.create_sys_msg(),
                ("human", f'"""{{{PromptInputs.QAPAIRS}}}"""'),
            ]
        )


class FullPrettifier(PromptHandler):
    """Handler for prettifying question-answer pairs."""

    goal = (
        "Your goal is to refine the answers in the "
        f"{{{PromptInputs.NUM_ANSWERS}}} question answer pairs "
        f"within triple quotes. Also generate appropriate name "
        f"and question for it. "
        "Correct any errors and enhance the logical structure "
        "to improve its overall presentation, "
        "ensuring that all original details and content are preserved "
        "and no new information is added. "
        "Make sure to include a refined answer for each of the provided "
        "question answer pairs."
    )

    overall_instruction = (
        "Always prioritize being truthful, nuanced, insightful, "
        "and efficient, tailoring your responses specifically "
        "to the user's needs and preferences."
        "Freely use Markdown format features like lists and text formatting. "
        "Make sure that all URLs you provided are formatted using Markdown "
        "and the text used to represent the URL "
        "is descriptive and provides context about what the URL links to. "
        f"\n{Examples.url_text}\n"
        "If you don't know what text to use better use the raw URL itself "
        'instead of generic terms like "here" as link text. '
        'Never use "here" as text used to represent a URL.\n'
        "Retain all citations from the original answers. "
        "Structure answers as straight responses to the questions "
        "without a header in the beginning. "
        "Make sure that all the citations from initial answers are used "
        "for relevant parts of the answer as in the sources. "
        "Cite pieces of the answer using the notation ^[marker]^. "
        "The 'marker' inside the brackets should be either a single number "
        "(e.g., ^[2]^) or three numbers separated by dots (e.g., ^[12.1.1]^)."
        "Place these citations at the end of the sentence "
        "or paragraph that reference them - do not put them all at the end. "
        "Place references before the period at the end of the sentence. "
        "If you want to cite multiple pieces of context "
        "for the same sentence, format it as `^[marker1]^ ^[marker2]^`."
        "However, you should NEVER do this with the same marker - "
        "if you want to cite `marker1` multiple times for a sentence, "
        "only do `^[marker1]^` not `^[marker1]^ ^[marker1]^`. "
        "Return only answers. "
        "Order of the answers has to be the same. "
        "Make sure to include a refined answer for each of the provided "
        "question answer pairs with its new name and question."
    )

    language = None

    @classmethod
    def get_cpt(cls) -> ChatPromptTemplate:
        """Returns a ChatPromptTemplate for prettifying."""
        return ChatPromptTemplate(
            [
                cls.create_sys_msg(),
                ("human", f'"""{{{PromptInputs.QAPAIRS}}}"""'),
            ]
        )


class MultimodalExtractionBase(PromptHandler):
    """Base class for multimodal extraction handlers."""

    overall_instruction = (
        "Your response has to adhere strictly to the given format. "
        "Think step-by-step and provide clear and accurate answers. "
        "Always prioritize being truthful, nuanced, insightful, "
        "and efficient, tailoring your responses specifically "
        "to the provided format. "
        "{content_type_description}"
        "Ensure that your extraction is exhaustive and faithful "
        "to the provided text and images, "
        "without adding any information not explicitly mentioned. "
        "The concepts and relations you extract "
        "have to accurately reflect the content of the provided "
        "{content_source} and, where possible, "
        "the broader context of the entire {document_type}. "
        "Your output should be clear and accessible to a broad audience."
    )

    steps = (
        "## Step 1: Extract Concepts\n"
        "Identify all concepts mentioned within the provided text and images. "
        "Ensure that your output is exhaustive, "
        "capturing every detail. "
        "## Step 2: Determine Relations\n"
        "Analyze the provided text and images "
        "to determine the relations between the identified concepts "
        "indicating how one may inform or relate to another.\n"
        "## Step 3: Structure the Information\n"
        "For each concept, formulate a relevant question. "
        "Compose an answer for each question "
        "using the information provided in the text and images. "
        "Each answer has to consist from parts. "
        "Each answer part has to consist from a fact. "
        "Facts must encompass all available data, such as URLs, "
        "among other details. "
        "Note that the source {source_id_type} ids of the text and images "
        "are explicitly stated, therefore, "
        "cite them as sources in your output. "
        "Facts themselves must not have source inside, they should be "
        "specified separately in source_ids field."
        "Assign a concise and unique name to each concept. "
        "Use the names of the origin and target concepts "
        "to represent each relation.\n"
        "## Step 4: Adhere to Response Format\n"
        "Revise questions, answers, and concept names as necessary "
        "to comply with the specified response format."
    )

    @classmethod
    def get_cpt(cls) -> ChatPromptTemplate:
        """Implementation to satisfy abstract method requirement."""
        raise NotImplementedError(
            "Use get_pt_gen method for multimodal extraction"
        )

    @classmethod
    def get_pt_gen(
        cls, inputs: dict[str, Any]
    ) -> list[SystemMessage | HumanMessage]:
        """
        Generate prompt messages for multimodal extraction.

        Args:
            inputs: Dictionary containing multimodal content

        Returns:
            List of messages for the language model
        """
        human_msg = HumanMessage(
            content=inputs[PromptInputs.MULTIMODAL_CONTENT]
        )
        return [cls.create_sys_msg(), human_msg]


class PDFExtraction(MultimodalExtractionBase):
    """Handler for extracting concepts from PDFs."""

    goal = (
        "Your goal is to represent all concepts "
        "and relations between them "
        "from the text and images in human's input."
    )

    content_type_description = (
        "Treat the provided text and images "
        "as pages of a PDF document. "
        "All text and images following the specified page number "
        "originate from that page. "
        "If there is no explicit mention of another page, "
        "then it all relates to the one initially referenced. "
    )

    @classmethod
    def create_sys_msg(cls) -> SystemMessage:
        """Creates a system message tailored for PDF extraction."""
        filled_instruction = cls.overall_instruction.format(
            content_type_description=cls.content_type_description,
            content_source="pages",
            document_type="document",
        )

        filled_steps = cls.steps.format(source_id_type="page")

        return SystemMessage(
            content="\n".join(
                filter(
                    None,
                    (
                        cls.role,
                        cls.goal,
                        filled_instruction,
                        cls.steps_header,
                        filled_steps,
                        cls.language,
                    ),
                )
            )
        )


class PPTXExtraction(MultimodalExtractionBase):
    """Handler for extracting concepts from PPTX presentations."""

    goal = (
        "Your goal is to represent all concepts "
        "and relations between them "
        "from the text and images in human's input."
    )

    # Customize the content description for PPTX
    content_type_description = (
        "Treat the provided text and images "
        "as slides of a PowerPoint presentation. "
        "All text and images following the specified slide number "
        "originate from that slide. "
        "If there is no explicit mention of another slide, "
        "then it all relates to the one initially referenced. "
    )

    @classmethod
    def create_sys_msg(cls) -> SystemMessage:
        """Creates a system message tailored for PPTX extraction."""
        filled_instruction = cls.overall_instruction.format(
            content_type_description=cls.content_type_description,
            content_source="slides",
            document_type="presentation",
        )

        filled_steps = cls.steps.format(source_id_type="slide")

        return SystemMessage(
            content="\n".join(
                filter(
                    None,
                    (
                        cls.role,
                        cls.goal,
                        filled_instruction,
                        cls.steps_header,
                        filled_steps,
                        cls.language,
                    ),
                )
            )
        )
