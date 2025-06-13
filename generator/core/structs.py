from dataclasses import dataclass

import pandas as pd


@dataclass
class MindMapData:
    node_df: pd.DataFrame
    edge_df: pd.DataFrame
    root_id: int
    problematic_nodes: list[dict] | None = None
