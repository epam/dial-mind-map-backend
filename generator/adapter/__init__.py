from generator.common.constants import EnvConsts

from .base import AbstractAdapter

if EnvConsts.APP_NAME == "GM":
    from .gm_adapter import GeneralMindmapAdapter

    adapter: AbstractAdapter = GeneralMindmapAdapter()

elif EnvConsts.APP_NAME == "OTHER_BACKEND":
    raise NotImplementedError(
        "The 'OTHER_BACKEND' adapter is not yet implemented."
    )

else:
    raise ValueError(f"Unknown or unsupported APP_NAME: '{EnvConsts.APP_NAME}'")

# Constants
GMContract = adapter.gm_contract

# Structures
GraphFilesAdapter = adapter.graph_files

# Embedding model
embeddings_model = adapter.get_embeddings_model()

# Functions
translate_graph_files = adapter.translate_graph_files
