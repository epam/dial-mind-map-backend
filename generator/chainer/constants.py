"""
Constants used for field names and prompt inputs
throughout the application.
"""


class FieldNames:
    """String constants for field names in output formats."""

    # Core concept fields
    CONCEPT_NAME = "name"
    QUESTION = "question"
    ANSWER = "answer"
    FACT = "fact"
    SOURCE_IDS = "source_ids"

    # Relation fields
    ORIGIN_CONCEPT_NAME = "origin_concept_name"
    TARGET_CONCEPT_NAME = "target_concept_name"

    # Structure fields
    CONCEPTS = "concepts"
    RELATIONS = "relations"
    ANSWERS = "answers"
    ROOT_CONCEPT = "root_concept"

    # Synthesis-related fields
    SYNTH_CONCEPT_NAME = "synthesized_name"
    SYNTH_QUESTION = "synthesized_question"
    SYNTH_ANSWER = "synthesized_answer"
    SYNTH_FACT = "synthesized_fact"
    SOURCE_CONCEPT_NAMES = "source_concept_names"
    SOURCE_CONCEPT_IDS = "source_concept_ids"
    CITATIONS = "citations"
    SYNTH_CONCEPTS = "synthesized_concepts"
    PRETTY_CONCEPTS = "pretty_concepts"


class PromptInputs:
    """String constants for prompt inputs and parameters."""

    # Input content types
    TEXTUAL_EXTRACTION = "text"
    MULTIMODAL_CONTENT = "multimodal_content"
    CLUSTER_SYNTH = "concepts"

    # Input parameters
    NUM_ANSWERS = "num_answers"
    QAPAIRS = "question_answer_pairs"
