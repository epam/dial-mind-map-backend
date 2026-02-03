from dataclasses import dataclass
from typing import NamedTuple, Optional, Union

import pandas as pd


class ExtractionProduct(NamedTuple):
    """
    A structured container for the results of the extraction process.
    """

    concept_df: Union[pd.DataFrame, None]
    relation_df: Union[pd.DataFrame, None]
    chunk_df: Union[pd.DataFrame, None]


@dataclass
class RawMindMapData:
    """
    A container for all dataframes and state related to a mind map
    being processed.
    """

    # Core dataframes that are modified throughout the pipeline
    concept_df: pd.DataFrame
    relation_df: pd.DataFrame

    # Inputs used by specific steps
    chunk_df: Optional[pd.DataFrame]
    flat_part_df: Optional[pd.DataFrame]

    # Root data
    root_df: Optional[pd.DataFrame] = None
    root_index: int | None = None

    old_root_concept: Optional[pd.DataFrame] = None


@dataclass
class MindMapData:
    node_df: pd.DataFrame
    edge_df: pd.DataFrame
    root_id: int
    problematic_nodes: list[dict] | None = None
