"""
Constants used for field names and prompt inputs
throughout the application.
"""

from enum import Enum

from generator.common.constants import EnvConsts, FieldNames

DEFAULT_CHAT_MODEL_NAME = EnvConsts.DEFAULT_CHAT_MODEL_NAME
DEFAULT_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHAT_MODEL_SEED = "CHAT_MODEL_SEED"
IS_EMBED_CACHE = "IS_EMBED_CACHE"

IS_LLM_CACHE = "IS_LLM_CACHE"
IS_LANGCHAIN_DEBUG = "IS_LANGCHAIN_DEBUG"
IS_LANGCHAIN_VERBOSE = "IS_LANGCHAIN_VERBOSE"
IS_LOCAL_RUN = "IS_LOCAL_RUN"


class FieldValidation:
    ERROR_TYPES = "error_types"
    DESCRIPTION = "description"


FIELD_TO_ERRORS = {
    FieldNames.SOURCE_IDS: {
        FieldValidation.ERROR_TYPES: {
            "missing",
            "list_too_short",
            "int_parsing",
            "list_type",
        },
        FieldValidation.DESCRIPTION: (
            "list of integer source IDs for every fact"
        ),
    }
}


class ChainTypes(Enum):
    TEXTUAL_EXTRACTION = "TEXTUAL_EXTRACTION"
    PDF_EXTRACTION = "PDF_EXTRACTION"
    PPTX_EXTRACTION = "PPTX_EXTRACTION"
    CLUSTER_SYNTH = "CLUSTER_SYNTHESIS"
    ROOT_CLUSTER_SYNTH = "ROOT_CLUSTER_SYNTHESIS"
    DEDUP = "DEDUPLICATION"
    DEDUP_ROOT = "ROOT_DEDUPLICATION"
    PRETTIFIER = "ANSWER_PRETTIFIER"
    CONCEPT_PRETTIFIER = "CONCEPT_PRETTIFIER"
    APPLY_CONCEPT_PRETTIFIER = "APPLY_CONCEPT_PRETTIFIER"
    NEW_ROOT = "NEW_ROOT"
    REFINE_FILTER = "REFINE_FILTER"
    VALIDATE_FILTER = "VALIDATE_FILTER"
    REFINE_STYLE = "REFINE_STYLE"
    VALIDATE_STYLE = "VALIDATE_STYLE"
    DEDUPLICATE_NAMES = "DEDUPLICATE_NAMES"
