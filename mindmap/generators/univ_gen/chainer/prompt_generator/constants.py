class PromptInputs:
    """String constants for prompt inputs and parameters."""

    # Input content types
    TEXTUAL_EXTRACTION = "text"
    MULTIMODAL_CONTENT = "multimodal_content"
    CLUSTER_SYNTH = "concepts"

    # Input parameters
    NUM_ANSWERS = "num_answers"
    QAPAIRS = "question_answer_pairs"
    TARGET_LANGUAGE = "target_language"
    TONE = "TONE"
    PERSONA = "PERSONA"
    OLD_CONCEPT = "old_concept"
    NEW_CONCEPT = "new_concept"

    STYLE = "style_instruction"

    FILTER = "filters"
    DOC_DESC = "doc_desc"

    QUERY = "query"

    ORIGINAL_NAME = "original_name"
    CONCEPTS_JSON = "concepts_json"


class PromptPartTypes:
    ROLE = "role"
    GOAL = "goal"
    LANGUAGE = "language"
    OVERALL_INSTRUCTION = "overall_instruction"
    STEPS_DESC = "steps_desc"
    STEPS_HEADER = "steps_header"

    INPUT_WRAPPER = "input_wrapper"

    EXAMPLE = "example"


class PromptPartKeys:
    DEFAULT = "default"
    TEXT = "textual_extraction"
    MULTIMODAL = "multimodal_extraction"
    DEDUPLICATOR = "deduplicator"
    PPT = "ppt_extraction"
    PDF = "pdf_extraction"
    CLUSTER_SYNTH = "cluster_synth"
    ROOT_CLUSTER_SYNTH = "root_cluster_synth"
    ROOT_DEDUPLICATOR = "root_deduplicator"
    PRETTIFICATION = "prettification"

    VALIDATE_FILTER = "validate_filter"
    VALIDATE_STYLE = "validate_style"
    REFINE_FILTER = "refine_filter"
    REFINE_STYLE = "refine_style"
    RENAME_CONCEPTS = "rename_concepts"

    ADD_ROOT_DEDUPLICATOR = "add_root_deduplicator"
    APPLY_PRETTIFICATION = "apply_prettification"

    SOURCE_IDS_EXAMPLE = "source_ids_synth_example"
    SOURCE_IDS_VS_ROOT_HAL_EXAMPLE = "source_ids_synth_example_vs_root_hal"
    SOURCE_IDS_T_EXAMPLE = "source_ids_template_example"
    DEDUP_EXAMPLE = "deduplication"
    ROOT_DEDUP_EXAMPLE = "root_deduplication"
    FILTER_EXAMPLE = "filtering_criteria_example"
    STYLE_EXAMPLE = "styling_criteria_example"
    VALIDATE_FILTER_EXAMPLE = "validate_filter_example"
    VALIDATE_STYLE_EXAMPLE = "validate_style_example"
    REFINE_FILTER_EXAMPLE = "refine_filter_example"
    REFINE_STYLE_EXAMPLE = "refine_style_example"

    BEFORE_EXAMPLE = "before_example"
    AFTER_EXAMPLE = "after_example"


READY_PTS_DIR = "generated_prompts"
PROMPT_PARTS_DIR = "prompt_parts"
YML_EXTENSION = ".yml"
