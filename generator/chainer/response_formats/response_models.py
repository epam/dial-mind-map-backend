from typing import List, Type

from pydantic import BaseModel, Field, create_model

from generator.common.constants import FieldNames as Fn
from generator.common.type_vars import PydanticFieldDefinition

from .constants import ModelNames as Mn
from .simple_fields import (
    Answer,
    ConceptName,
    Fact,
    FieldDefinitions,
    Question,
    Relation,
    Source,
)


def _build_model(
    model_name: str,
    *fields: PydanticFieldDefinition,
    docstring: str | None = None,
    base: Type[BaseModel] | None = None,
) -> Type[BaseModel]:
    """
    Dynamically create a Pydantic model.

    Args:
        model_name: The name of the model to create.
        *fields: A variable number of field definition tuples.
        docstring: The docstring for the new model.
        base: A base model to inherit from.

    Returns:
        The created Pydantic model class.
    """
    field_definitions = dict(fields)
    kwargs = {}
    if docstring:
        kwargs["__doc__"] = docstring
    if base:
        kwargs["__base__"] = base

    return create_model(model_name, **field_definitions, **kwargs)


# Common docstrings and descriptions for reuse.
_ANSWER_PART_SOURCES_DOC = "A specific factual statement and its sources."
_CONCEPT_ANSWER_DESC = (
    "A list of AnswerPart objects that collectively form the complete "
    "answer."
)
_CONCEPT_NODE_DOC = (
    "Encapsulates a distinct piece of information, equivalent to a node "
    "in a mind map."
)
_CONCEPTS_DESC = (
    "A list of all distinct concepts identified in the document segment."
)


class AnswerPart:
    Text = _build_model(
        Mn.ANSWER_PART,
        Fact.TEXT,
        docstring="A specific factual statement.",
    )

    Multimodal = _build_model(
        Mn.ANSWER_PART,
        Fact.MULTIMODAL,
        Source.IDS,
        docstring=_ANSWER_PART_SOURCES_DOC,
    )

    Synthesized = _build_model(
        Mn.ANSWER_PART,
        Fact.SYNTH,
        Source.CITATIONS,
        docstring=_ANSWER_PART_SOURCES_DOC,
    )


class ConceptAnswer:
    TEXT = (
        Fn.ANSWER,
        (
            List[AnswerPart.Text],
            Field(description=_CONCEPT_ANSWER_DESC),
        ),
    )

    MULTIMODAL = (
        Fn.ANSWER,
        (
            List[AnswerPart.Multimodal],
            Field(min_length=1, description=_CONCEPT_ANSWER_DESC),
        ),
    )

    SYNTH = (
        Fn.ANSWER,
        (
            List[AnswerPart.Synthesized],
            Field(description="A list of synthesized answer components."),
        ),
    )


class Concept:
    Text = _build_model(
        Mn.CONCEPT,
        ConceptName.EXTRACTED,
        Question.EXTRACTED,
        ConceptAnswer.TEXT,
        docstring=_CONCEPT_NODE_DOC,
    )

    Multimodal = _build_model(
        Mn.CONCEPT,
        ConceptName.EXTRACTED,
        Question.EXTRACTED,
        ConceptAnswer.MULTIMODAL,
        docstring=_CONCEPT_NODE_DOC,
    )

    Synthesized = _build_model(
        Mn.CONCEPT,
        ConceptName.SYNTH,
        Question.SYNTH,
        ConceptAnswer.SYNTH,
        Source.CONCEPT_NAMES,
        docstring=(
            "Represents a synthesized concept created by clustering "
            "information."
        ),
    )

    DedConcept = _build_model(
        Mn.CONCEPT,
        ConceptName.SYNTH,
        Question.SYNTH,
        ConceptAnswer.SYNTH,
        Source.CONCEPT_IDS,
        docstring=(
            "Represents a distinct concept created by clustering "
            "information."
        ),
    )

    Pretty = _build_model(
        "PrettyConcept",
        ConceptName.PRETTY,
        Question.PRETTY,
        Answer.PRETTY,
    )

    _RenamedConcept = _build_model(
        "RenamedConcept",
        FieldDefinitions.CONCEPT_INDEX,
        FieldDefinitions.NEW_NAME,
        docstring=(
            "A single concept with its original index and a new, "
            "specific name."
        ),
        base=BaseModel,
    )

    TEXT = (
        Fn.CONCEPTS,
        (List[Text], Field(min_length=1, description=_CONCEPTS_DESC)),
    )

    MULTIMODAL = (
        Fn.CONCEPTS,
        (List[Multimodal], Field(description=_CONCEPTS_DESC)),
    )

    SYNTH = (
        Fn.SYNTH_CONCEPTS,
        (
            List[Synthesized],
            Field(
                description=(
                    "A list of distinct concepts created by synthesizing "
                    "information from the provided ones."
                )
            ),
        ),
    )

    DED_CONCEPTS = (
        Fn.SYNTH_CONCEPTS,
        (
            List[DedConcept],
            Field(description="A list of synthesized QA Pairs."),
        ),
    )

    PRETTY = (Fn.PRETTY_CONCEPTS, (List[Pretty], Field(description="")))

    RENAMED_CONCEPTS = (
        "renamed_concepts",
        (
            List[_RenamedConcept],
            Field(description="A list of concepts with their new names."),
        ),
    )


class ConceptRelation:
    Default = _build_model(
        "Relation",
        Relation.ORIGIN_CONCEPT_NAME,
        Relation.TARGET_CONCEPT_NAME,
        docstring="Represents a directional connection between two concepts.",
    )

    DEFAULT = (
        Fn.RELATIONS,
        (
            List[Default],
            Field(
                default_factory=list,
                description=(
                    "A list of directional relations connecting the concepts."
                ),
            ),
        ),
    )


class Root:
    CONCEPT = (
        Fn.ROOT_CONCEPT,
        (Concept.Synthesized, Field(description="The root concept.")),
    )

    DED = (
        Fn.ROOT_CONCEPT,
        (
            Concept.DedConcept,
            Field(
                description=(
                    "The single, preserved, and synthesized root concept. "
                    "It must be present."
                )
            ),
        ),
    )


class Results:
    ExtractionResult = _build_model(
        Mn.KNOWLEDGE_FRAGMENT, Concept.TEXT, ConceptRelation.DEFAULT
    )

    KnowledgeFragment = _build_model(
        Mn.KNOWLEDGE_FRAGMENT,
        Concept.MULTIMODAL,
        ConceptRelation.DEFAULT,
        docstring=(
            "The complete, structured knowledge fragment extracted from a "
            "document segment."
        ),
    )

    ClusterSynthesisResult = _build_model(
        Mn.CONCEPT_CLUSTER,
        Concept.SYNTH,
        ConceptRelation.DEFAULT,
        Root.CONCEPT,
    )

    RootClusterSynthesisResult = _build_model(
        "RootClusterSynthesisResult", Root.CONCEPT, Concept.SYNTH
    )

    DeduplicationResult = _build_model(
        Mn.CONCEPT_GRAPH, Concept.DED_CONCEPTS, ConceptRelation.DEFAULT
    )

    RootDeduplicationResult = _build_model(
        Mn.CONCEPT_CLUSTER, Concept.DED_CONCEPTS, Root.DED
    )

    ResultRoot = _build_model(Fn.ROOT_CONCEPT, Root.CONCEPT)

    PrettyConcepts = _build_model(
        Mn.PRETTY_CONCEPTS_RESULT,
        Concept.PRETTY,
        docstring="A container for a list of prettified concepts.",
    )

    ValidationResult = _build_model(
        "ValidationResult",
        FieldDefinitions.IS_SAFE,
        FieldDefinitions.REASON,
        docstring="The result of a security validation check.",
        base=BaseModel,
    )

    VALIDATION = ("validation", (ValidationResult, ...))

    RefinedFilterResult = _build_model(
        "RefinedFilterResult",
        VALIDATION,
        FieldDefinitions.REFINED_QUERY,
        docstring=(
            "The result of refining a 'basic' user query, including "
            "validation."
        ),
        base=BaseModel,
    )

    RefinedStyleResult = _build_model(
        "RefinedStyleResult",
        VALIDATION,
        FieldDefinitions.LANGUAGE,
        FieldDefinitions.PERSONA,
        FieldDefinitions.TONE,
        FieldDefinitions.OTHER_INSTRUCTIONS,
        docstring=(
            "Deconstructed styling instructions, extracted from a user's "
            "single request.\nIncludes security validation."
        ),
        base=BaseModel,
    )

    RenamingResult = _build_model(
        "RenamingResult",
        Concept.RENAMED_CONCEPTS,
        docstring="A list of renamed concepts.",
        base=BaseModel,
    )
