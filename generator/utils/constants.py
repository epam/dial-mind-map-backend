import os
from enum import Enum

from general_mindmap.models.request import NodeData
from ..chainer.constants import FieldNames as Fn
from ..chainer.constants import PromptInputs as Pi

# Models
DEFAULT_CHAT_MODEL_NAME = os.getenv("GENERATOR_MODEL", "gpt-4o-2024-08-06")
DEFAULT_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHAT_MODEL_SEED = "CHAT_MODEL_SEED"

# Cache
CACHE_DIR = "../../generator/cache/"
LLM_CACHE_PATH = CACHE_DIR + ".langchain.db"
EMBEDDING_CACHE_DIR_PATH = (
    "D:/PythonProjects/general-mindmap/generator/cache/embeddings"
)
LOGS_DIR = "../../generator/logs"

FACT = "fact"
SOURCE_IDS = "source_ids"
SYNTH_NAME = "synth_name"
SYNTH_QUESTION = "synth_question"
SYNTH_ANSWER = "synth_answer"

TEXT_INPUT = Pi.TEXTUAL_EXTRACTION


class DefaultValues:
    START_FLAT_PART_ID = 1
    START_CLUSTER_ID = 0
    START_NODE_ID = 0

    NO_SOURCE_ID = "0.0"


class OtherBackEndConstants:
    DATA_KEY = "data"

    # Files
    NODES_FILE = "nodes_file"
    EDGES_FILE = "edges_file"

    # File keys
    NODES_KEY = "nodes"
    ROOT_KEY = "root"
    EDGE_KEY = "edges"

    # Edge types
    GENERATED = "Generated"
    INIT = "Init"
    MANUAL = "Manual"

    # Node data keys
    NODE_ID = "id"
    NAME = "label"
    QUESTION = "question"
    ANSWER_STR = "details"
    METADATA = "metadata"
    ANSWER = "answer"
    LVL = "level"
    CLUSTER_ID = "cluster_id"

    # Edge data keys
    EDGE_ID = "id"
    SOURCE = "source"
    TARGET = "target"
    TYPE = "type"
    WEIGHT = "weight"


class FrontEndStatuses:
    SAVE_GENERATION_RESULTS = "Saving generation results"
    SAVE_APPLY_RESULTS = "Saving changes"
    LOAD_DOCS = "Loading documents"
    ANALYZE_DOCS = "Analyzing documents"
    DEFINE_CONCEPTS = "Defining concepts. Step {}"
    PRETTIFY = "Prettification"
    CREATE_CONNECTIONS = "Creating connections"
    PROGRESS = "Progress: {}%"


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
    NEW_ROOT = "NEW_ROOT"


class ClusteringMethods:
    AGGLOMERATIVE = "agglomerative"
    KMEANS = "kmeans"


class DataFrameCols:
    ID = "id"

    DOC_ID = "doc_id"
    DOC_URL = "doc_url"
    DOC_CAT = "doc_cat"
    DOC_TITLE = "doc_title"

    CHUNK_ID = "chunk_id"
    FLAT_CHUNK_ID = "flat_chunk_id"
    TEMP_CHUNK_ID = "temp_chunk_id"
    CHUNK = "lc_doc"
    CONTENT = "content"
    SLIDE_CONTENT = "slide_contents"
    CONCEPT_IDS = "concept_ids"
    RELATION_IDS = "relation_ids"
    FLAT_PART_ID = "flat_part_id"
    CITATION = "citation"

    QAPAIR_ID = "id"
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
    ACTIVE_CONCEPT_TRUE_VAL = 1
    ACTIVE_CONCEPT_FALSE_VAL = 0
    MODIFIED = "modified"
    ROOT_LVL_VAL = -1
    ROOT_CLUSTER_VAL = -1

    SOURCE_QAPAIR_NAME = Fn.ORIGIN_CONCEPT_NAME
    ORIGIN_CONCEPT_ID = "origin_qapair_id"
    TARGET_QAPAIR_NAME = Fn.TARGET_CONCEPT_NAME
    TARGET_CONCEPT_ID = "target_qapair_id"
    WEIGHT = "weight"
    TYPE = "type"
    RELATED_TYPE_VAL = "related"
    ART_EDGE_TYPE_VAL = "art_related"


class DocCategories:
    LINK = "LINK"
    PPTX = "PPTX"
    HTML = "HTML"
    PDF = "PDF"
    UNSUPPORTED = "UNSUPPORTED"


class DocTypes:
    LINK = "LINK"
    FILE = "FILE"


class DocContentTypes:
    PRESENTATION = (
        "application/vnd.openxmlformats-officedocument."
        "presentationml.presentation"
    )
    HTML = "text/html"
    PDF = "application/pdf"


class Patterns:
    GENERAL_CITATION = r"\[[^\]]+\]"
    # Updated pattern to match both X.Y.Z format and simple integers
    FRONT_CITATION = r"\^\[(\d+(?:\.\d+)*)\]\^"
    NO_PUNCT = r"[^\w\s]"


# Model environment Variables
IS_LLM_CACHE = "IS_LLM_CACHE"
IS_EMBED_CACHE = "IS_EMBED_CACHE"
IS_LANGCHAIN_DEBUG = "IS_LANGCHAIN_DEBUG"
IS_LANGCHAIN_VERBOSE = "IS_LANGCHAIN_VERBOSE"
IS_LOCAL_RUN = "IS_LOCAL_RUN"

# Web
USER_AGENT = "USER_AGENT"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/114.0.5735.199 Safari/537.36"
)

IS_LOG = "IS_LOG"
STABLE_AGGLOMERATIVE = "STABLE_AGGLOMERATIVE"

EMPTY_NODE_DATA = NodeData(
    id="0",
    label="Empty Graph",
    question=" ",
    details=" ",
    metadata=None,
)
