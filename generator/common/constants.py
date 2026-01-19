import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent


class EnvConsts:
    @classmethod
    def get_consts_from_env(cls):
        load_dotenv()
        # Name of the application that uses the generator
        cls.APP_NAME = os.getenv("APP_NAME", "GM")

        # DIAL
        cls.DIAL_URL = os.getenv("DIAL_URL")
        cls.DEFAULT_CHAT_MODEL_NAME = os.getenv(
            "GENERATOR_MODEL", "gpt-4.1-2025-04-14"
        )

        # Prompts
        cls.SAVE_PROMPTS = cls._env_to_bool("SAVE_PROMPTS")

        # Logging
        cls.IS_LOG = cls._env_to_bool("IS_LOG")
        cls.LOGS_DIR = ROOT_DIR / os.getenv("LOGS_DIR", "logs")

        # Caching
        cls.CACHE_DIR = ROOT_DIR / os.getenv("CACHE_DIR", "cache")

        # LLM caching
        cls.IS_LLM_CACHE = cls._env_to_bool("IS_LLM_CACHE")
        cls.LLM_CACHE_PATH = cls.CACHE_DIR / os.getenv(
            "LLM_CACHE_FILE_NAME", ".langchain.db"
        )

        # Embedding caching
        cls.IS_EMBED_CACHE = cls._env_to_bool("IS_EMBED_CACHE")
        cls.EMBEDDING_CACHE_DIR_PATH = cls.CACHE_DIR / os.getenv(
            "EMBEDDING_CACHE_DIR_PATH", "embeddings"
        )

        # Langchain telemetry
        cls.IS_LANGCHAIN_DEBUG = cls._env_to_bool("IS_LANGCHAIN_DEBUG")
        cls.IS_LANGCHAIN_VERBOSE = cls._env_to_bool("IS_LANGCHAIN_VERBOSE")

        # Clustering settings
        cls.IS_STABLE_AGGLOMERATIVE = cls._env_to_bool(
            "IS_STABLE_AGGLOMERATIVE"
        )

        # Simple generator settings
        cls.IS_STREAM_SIMPLE_GEN = cls._env_to_bool(
            "IS_STREAM_SIMPLE_GEN", default=True
        )

    @staticmethod
    def _env_to_bool(key: str, *, default: bool = False) -> bool:
        if (value := os.environ.get(key)) is None:
            return default
        return value.lower() in {"1", "t", "on", "true"}


class FieldNames:
    """String constants for field names in response formats."""

    # Concept
    CONCEPT_NAME = "name"
    QUESTION = "question"
    ANSWER = "answer"
    FACT = "fact"
    SOURCE_IDS = "source_ids"

    # Synthesis
    SOURCE_CONCEPT_NAMES = "source_concept_names"
    SOURCE_CONCEPT_IDS = "source_concept_ids"
    CITATIONS = "source_ids"

    # Relation
    ORIGIN_CONCEPT_NAME = "source_name"
    TARGET_CONCEPT_NAME = "target_name"

    # Structure
    CONCEPTS = "concepts"
    RELATIONS = "relations"
    ANSWERS = "answers"
    ROOT_CONCEPT = "root_concept"
    SYNTH_CONCEPTS = "concepts"
    PRETTY_CONCEPTS = "pretty_concepts"

    # User instructions
    IS_SAFE = "is_safe"
    REASON = "reason"

    REFINED_QUERY = "refined_query"

    LANGUAGE = "language"
    PERSONA = "persona"
    TONE = "tone"
    OTHER_INSTRUCTIONS = "other_instructions"

    # Rename
    CONCEPT_INDEX = "concept_index"
    NEW_NAME = "new_name"


class DataFrameCols:
    """Defines standardized column names for use in pandas DataFrames."""

    ID = "id"

    # Doc
    DOC_ID = "doc_id"
    DOC_CAT = "doc_cat"

    # Chunks and parts
    CHUNK_ID = "chunk_id"
    FLAT_CHUNK_ID = "flat_chunk_id"
    TEMP_CHUNK_ID = "temp_chunk_id"
    DOC_DESC = "doc_desc"
    CONTENT = "content"
    CONCEPT_IDS = "concept_ids"
    RELATION_IDS = "relation_ids"
    FLAT_PART_ID = "flat_part_id"
    CITATION = "citation"

    # Concepts
    CONCEPT_ID = "id"
    NAME = "name"
    QUESTION = "question"
    ANSWER_STR = "str_answer"
    EMBEDDING = "embedding"
    CLUSTER_ID = "cluster_id"
    SOURCE_IDS = "source_qapair_names"
    LVL = "level"
    PAGE_ID = "page_ids"
    ANSWER = "answer"
    IS_ACTIVE_CONCEPT = "is_active"
    MODIFIED = "modified"

    # Edges
    ORIGIN_CONCEPT_ID = "origin_qapair_id"
    TARGET_CONCEPT_ID = "target_qapair_id"
    WEIGHT = "weight"
    TYPE = "type"


class ColVals:
    # Flags
    TRUE_INT = 1
    FALSE_INT = 0
    UNDEFINED = -1

    # Edge Types
    RELATED = "related"
    ARTIFICIAL = "artificial"
