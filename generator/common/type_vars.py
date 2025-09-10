from typing import Any, Type, TypeAlias

import pandas as pd

ChunkingResult: TypeAlias = tuple[pd.DataFrame, pd.DataFrame]
ConceptResult: TypeAlias = tuple[pd.DataFrame, pd.DataFrame, Any]
EdgeResult: TypeAlias = tuple[pd.DataFrame, Any]
PydanticFieldDefinition: TypeAlias = tuple[str, tuple[Type, Any]]
