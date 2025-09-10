from generator.adapter import GMContract as Gmc
from generator.chainer.prompt_generator.constants import PromptInputs as Pi
from generator.common.structs import NodeData


class ExceptionMessage:
    NO_MIND_MAP_DATA = (
        "Mind map data was not received from the generation pipeline."
    )


# Cache
CACHE_DIR = "../../cache/"
LLM_CACHE_PATH = CACHE_DIR + ".langchain.db"
EMBEDDING_CACHE_DIR_PATH = (
    "D:/PythonProjects/general-mindmap/generator/cache/embeddings"
)

# Logging
LOGS_DIR = "../../logs"

# Chunking
HEADER_TO_SPLIT_ON = "Header 1"

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


class LoggingMessages:
    LLM_COST = "Total LLM costs:\n{}"


class FrontEndStatuses:
    SAVE_GENERATION_RESULTS = "Saving generation results"
    SAVE_APPLY_RESULTS = "Saving changes"
    LOAD_DOCS = "Loading documents"
    ANALYZE_DOCS = "Analyzing documents"
    DEFINE_CONCEPTS = "Defining concepts. Step {}"
    PRETTIFY = "Prettification"
    CREATE_CONNECTIONS = "Creating connections"
    PROGRESS = "Progress: {}%"

    LOAD_ADD_DOCS = "Loading new documents"


class ClusteringMethods:
    AGGLOMERATIVE = "agglomerative"
    KMEANS = "kmeans"


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

DEFAULT_STYLE = {
    # "Act like the USA, but in tone of a pirate and use English"
    # "Write this from the third-person view in formal tone using British English."
    Gmc.STYLE_PROMPT_FIELD: "",
    Gmc.IS_FINAL_FIELD: False,
}

DEFAULT_FILTER = {
    # "Only extract USA-related information"
    # "Macroeconomics, interest rates, and inflation in the Americas from 2023."
    Gmc.FILTER_PROMPT_FIELD: "",
    Gmc.IS_FINAL_FIELD: True,
}

IS_LOG = "IS_LOG"
STABLE_AGGLOMERATIVE = "STABLE_AGGLOMERATIVE"

EMPTY_NODE_DATA = NodeData(
    id="0",
    label="Empty Graph",
    question=" ",
    details=" ",
    metadata=None,
)


class EdgeWeights:
    ARTIFICIAL_EDGE_WEIGHT = 0.5
    RELATED_EDGE_WEIGHT = 3.0
    ROOT_RELATED_EDGE_WEIGHT = 5.0
    DEFAULT_EDGE_WEIGHT = 0.0
