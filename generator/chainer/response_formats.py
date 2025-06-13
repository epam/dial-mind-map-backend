from pydantic import Field

from .constants import FieldNames as Fn
from .utils import get_pydantic_model

AnswerPart = get_pydantic_model(
    "AnswerPart",
    {
        Fn.FACT: (
            str,
            Field(
                description="The factual content of the answer part "
                "with URLs represented in Markdown format."
            ),
        ),
        Fn.SOURCE_IDS: (
            list[int] | None,
            Field(
                default=None,
                description="List of source IDs "
                "that support this fact, if available. "
                "Sources are limited to slides.",
            ),
        ),
    },
    "A part of an answer that includes a specific fact "
    "and its corresponding sources if available.",
)


Concept = get_pydantic_model(
    "Concept",
    {
        Fn.CONCEPT_NAME: (
            str,
            Field(
                description="A unique, concise, human-readable name "
                "for the QA Pair. Typically 1-2 words without underscores "
                "or camelcase compound words."
            ),
        ),
        Fn.QUESTION: (
            str,
            Field(
                description="The question derived from the data. "
                "Typically 1-2 sentences long."
            ),
        ),
        Fn.ANSWER: (
            list[AnswerPart],
            Field(
                description="A list of answer parts that together "
                "form a comprehensive answer to the question."
            ),
        ),
    },
    "Concept representation that includes its name question about it "
    "and an answer to the question.",
)


Relation = get_pydantic_model(
    "Relation",
    {
        Fn.ORIGIN_CONCEPT_NAME: (
            str,
            Field(description="Name of the origin concept."),
        ),
        Fn.TARGET_CONCEPT_NAME: (
            str,
            Field(description="Name of the target concept."),
        ),
    },
    "Relation between two concepts.",
)


ExtractionResult = get_pydantic_model(
    "ExtractionResult",
    {
        Fn.CONCEPTS: (
            list[Concept],
            Field(description="List of concepts extracted from the data."),
        ),
        Fn.RELATIONS: (
            list[Relation],
            Field(
                description="A list of relations representing "
                "the connections between different concepts."
            ),
        ),
    },
)


SynthesizedAnswerPart = get_pydantic_model(
    "SynthesizedAnswerPart",
    {
        Fn.SYNTH_FACT: (
            str,
            Field(
                description="The factual statement contained "
                "within the synthesized answer part "
                "with URLs in Markdown format."
            ),
        ),
        Fn.CITATIONS: (
            list[int],
            Field(
                description="A list of numbers in square brackets "
                "representing reference indices.",
            ),
        ),
    },
    "A part of a synthesized answer that includes a specific fact "
    "along with its supporting citations "
    "sourced from multiple original answers.",
)


SynthesizedConcept = get_pydantic_model(
    "SynthesizedConcept",
    {
        Fn.SYNTH_CONCEPT_NAME: (
            str,
            Field(
                description="A unique, concise name "
                "for the synthesized concept."
            ),
        ),
        Fn.SYNTH_QUESTION: (
            str,
            Field(
                description="The formulated question "
                "derived by integrating or refining questions."
            ),
        ),
        Fn.SYNTH_ANSWER: (
            list[SynthesizedAnswerPart],
            Field(description="A list of synthesized answer components"),
        ),
        Fn.SOURCE_CONCEPT_NAMES: (
            list[str],
            Field(description="A list of Names of the source concepts"),
        ),
    },
    "A synthesized concepts created by combining elements "
    "from multiple source concepts or modifying a single source concept "
    "to generate a new, unique concept.",
)


ClusterSynthesisResult = get_pydantic_model(
    "ClusterSynthesisResult",
    {
        Fn.SYNTH_CONCEPTS: (
            list[SynthesizedConcept],
            Field(description="A list of synthesized concepts."),
        ),
        Fn.RELATIONS: (
            list[Relation],
            Field(
                description="A list of relations defining the connections "
                "between concepts."
            ),
        ),
        Fn.ROOT_CONCEPT: (
            SynthesizedConcept,
            Field(description="The root concept."),
        ),
    },
)


DedQAPair = get_pydantic_model(
    "SynthesizedConcept",
    {
        Fn.SYNTH_CONCEPT_NAME: (
            str,
            Field(
                description="A unique, concise name "
                "for the synthesized concept."
            ),
        ),
        Fn.SYNTH_QUESTION: (
            str,
            Field(
                description="The formulated question derived "
                "by integrating or refining questions."
            ),
        ),
        Fn.SYNTH_ANSWER: (
            list[SynthesizedAnswerPart],
            Field(description="A list of synthesized answer components"),
        ),
        Fn.SOURCE_CONCEPT_IDS: (
            list[int],
            Field(description="A list of IDs of the source concepts"),
        ),
    },
    "A synthesized concept created by combining elements "
    "from multiple source concepts, or modifying a single source concept "
    "to generate a new, unique concept.",
)


RootClusterSynthesisResult = get_pydantic_model(
    "RootClusterSynthesisResult",
    {
        Fn.ROOT_CONCEPT: (
            SynthesizedConcept,
            Field(description="The central QA Pair."),
        ),
        Fn.SYNTH_CONCEPTS: (
            list[SynthesizedConcept],
            Field(description="A set of subordinate QA Pairs."),
        ),
    },
)


DeduplicationResult = get_pydantic_model(
    "DeduplicationResult",
    {
        Fn.SYNTH_CONCEPTS: (
            list[DedQAPair],
            Field(description="A list of synthesized QA Pairs."),
        ),
        Fn.RELATIONS: (
            list[Relation],
            Field(description="A list of relations defining the connections."),
        ),
    },
)


RootDeduplicationResult = get_pydantic_model(
    "RootDeduplicationResult",
    {
        Fn.PRETTY_CONCEPTS: (
            list[DedQAPair],
            Field(description="A list of synthesized QA Pairs."),
        )
    },
)


PrettyAnswers = get_pydantic_model(
    "PrettyAnswers",
    {
        Fn.ANSWERS: (
            list[str],
            Field(description="List of prettified answers."),
        )
    },
)


PrettyConcept = get_pydantic_model(
    "PrettyConcept",
    {
        Fn.CONCEPT_NAME: (
            str,
            Field(
                description="A unique, concise name "
                "for the synthesized concept."
            ),
        ),
        Fn.QUESTION: (
            str,
            Field(
                description="The formulated question derived "
                "by integrating or refining questions."
            ),
        ),
        Fn.ANSWER: (
            str,
            Field(description="A string answer."),
        ),
    },
)

PrettyConcepts = get_pydantic_model(
    "PrettyConcepts",
    {
        Fn.PRETTY_CONCEPTS: (
            list[PrettyConcept],
            Field(description="List of prettified concepts."),
        )
    },
)


ResultRoot = get_pydantic_model(
    Fn.ROOT_CONCEPT,
    {
        Fn.ROOT_CONCEPT: (
            SynthesizedConcept,
            Field(description="The new updated concept"),
        ),
    },
)
