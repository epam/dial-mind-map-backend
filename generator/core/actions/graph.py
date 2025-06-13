import re
from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException

from ...chainer.constants import FieldNames as Fn
from ...utils.constants import DataFrameCols as Col
from ...utils.constants import DefaultValues as Dv
from ...utils.constants import OtherBackEndConstants as Bec
from ...utils.constants import Patterns as Pat
from ...utils.exceptions import ApplyException
from ...utils.logger import logging
from ...utils.misc import is_valid_uuid, split_list_by_indices
from ..structs import MindMapData
from .references import (
    extract_citations,
    extract_facts_with_sources,
    extract_facts_with_sources_and_fix_old,
)


def validate_graph(
    graph_files: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str | Any]]]:
    """
    Determine problematic nodes, remove them from graph files,
    and return their data together with updated graph files.
    """
    logging.info("Validate graph: Start")
    nodes_file = graph_files[Bec.NODES_FILE]

    # Mind map without root is invalid
    root_id = nodes_file[Bec.ROOT_KEY]
    if root_id is None:
        raise HTTPException(status_code=400, detail="No root node")

    # Convert to node_df for easier data modifications
    nodes = nodes_file[Bec.NODES_KEY]
    nodes_data = [node[Bec.DATA_KEY] for node in nodes]
    node_df = pd.DataFrame(nodes_data)

    # Convert node details to list of facts and sources
    node_df[Col.ANSWER] = None
    empty_answers = node_df[Col.ANSWER].isna()
    node_df.loc[empty_answers, Col.ANSWER] = node_df.loc[
        empty_answers, Bec.ANSWER_STR
    ].apply(lambda answer_str: extract_facts_with_sources(answer_str))

    is_metadata_in_cols = Bec.METADATA in node_df.columns

    problematic_ids = set()
    node_ids = node_df[Bec.NODE_ID].tolist()
    # Iterative process to handle cascading problems
    while True:
        iteration_problematic_ids = set()
        for _, row in node_df.iterrows():
            is_metadata_present = isinstance(row[Bec.METADATA], dict)

            node_id = row[Bec.NODE_ID]
            if node_id in problematic_ids:
                continue

            facts_list: list[dict[str, str | list[str]]] = row[Col.ANSWER]
            for fact in facts_list:
                is_fact_valid = True
                for source_id in fact[Fn.SOURCE_IDS]:
                    # If source is a node
                    if source_id.isdigit() or is_valid_uuid(source_id):
                        if source_id not in node_ids + problematic_ids:
                            iteration_problematic_ids.add(node_id)
                            is_fact_valid = False
                            break
                    # If no source
                    elif source_id == Dv.NO_SOURCE_ID:
                        iteration_problematic_ids.add(node_id)
                        is_fact_valid = False
                        break
                    # If node has metadata, it is not None,
                    # and source is document: make sure that
                    # it has a version defined (has 3 numbers, not 2)
                    elif (
                        is_metadata_in_cols
                        and is_metadata_present
                        and "." in source_id
                    ):
                        parts = source_id.split(".")
                        if len(parts) != 3 or any(
                            not part.isdigit() for part in parts
                        ):
                            iteration_problematic_ids.add(node_id)
                            is_fact_valid = False
                            break
                    else:
                        continue
                if not is_fact_valid:
                    break

        if not iteration_problematic_ids:
            break

        problematic_ids.update(iteration_problematic_ids)

    # Determine problematic list indices by problematic node_ids
    mask = node_df[Bec.NODE_ID].isin(problematic_ids)
    problematic_indices = node_df.index[mask].tolist()

    # Get list of valid nodes and data of problematic nodes
    problematic_nodes, other_nodes = split_list_by_indices(
        nodes, problematic_indices
    )
    problematic_nodes_data = [node[Bec.DATA_KEY] for node in problematic_nodes]

    # If root is invalid
    if root_id in problematic_ids:
        raise HTTPException(status_code=400, detail="Root is invalid")

    # Pack results back to graph files
    nodes_file[Bec.NODES_KEY] = other_nodes
    nodes_file[Bec.ROOT_KEY] = root_id
    graph_files[Bec.NODES_FILE] = nodes_file

    logging.info("Validate graph: End")

    return graph_files, problematic_nodes_data


def prep_node_df_for_add(
    node_df: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int]:
    node_df["new"] = 0
    new_node_id = int(node_df.index.max()) + 1

    all_part_ids = [
        item
        for sublist in node_df[Col.FLAT_PART_ID]
        if isinstance(sublist, list) and len(sublist) > 0
        for item in sublist
    ]
    max_prev_part_id = max(all_part_ids) if all_part_ids else 0
    return node_df, max_prev_part_id, new_node_id


def handle_manual_nodes(node_df: pd.DataFrame) -> pd.DataFrame:
    inconsistent_nan_mask = (
        node_df[Col.LVL].isna() != node_df[Col.CLUSTER_ID].isna()
    )

    if np.any(inconsistent_nan_mask):
        raise ApplyException(
            f"Inconsistent NaN values found between '{Col.LVL}'"
            f" and '{Col.CLUSTER_ID}'. "
            "Expected them to be either both NaN or both non-NaN."
        )

    mask = node_df[Col.LVL].isna() & node_df[Col.CLUSTER_ID].isna()

    if not mask.any():
        return node_df

    if node_df[Col.CLUSTER_ID].notna().any():
        max_cluster_id = node_df[Col.CLUSTER_ID].max()
        start_cluster_id = int(max_cluster_id) + 1
    else:
        start_cluster_id = Dv.START_CLUSTER_ID

    num_rows_to_update = mask.sum()

    # Generate cluster_ids in groups of 5
    new_cluster_ids = [
        start_cluster_id + (i // 5) for i in range(num_rows_to_update)
    ]

    node_df.loc[mask, Col.LVL] = 1
    node_df.loc[mask, Col.CLUSTER_ID] = new_cluster_ids

    return node_df


def form_flat_part_ids(node_df: pd.DataFrame) -> pd.DataFrame:
    citation_col = node_df[Col.CITATION]
    citations = sorted(set().union(*citation_col))
    citation_to_flat_part_id_ser = pd.Series(
        {
            citation: flat_part_id  # Start flat_part_id from 1
            for flat_part_id, citation in enumerate(
                citations, Dv.START_FLAT_PART_ID
            )
        }
    )
    node_df[Col.FLAT_PART_ID] = citation_col.apply(
        lambda _citations: citation_to_flat_part_id_ser.reindex(
            _citations
        ).tolist()
    )

    return node_df


def process_graph(
    graph_files: dict[str, Any],
) -> tuple[MindMapData, dict[str, int]]:
    """ID Map saves correspondence between int ids and uuids."""
    nodes_file = graph_files[Bec.NODES_FILE]

    # Get Root ID
    root_id = nodes_file[Bec.ROOT_KEY]
    if root_id is None:
        raise HTTPException(status_code=400, detail="No root node")

    # Create Node DF
    nodes_data = []
    nodes = nodes_file[Bec.NODES_KEY]
    for node in nodes:
        node_data_row = node[Bec.DATA_KEY].copy()
        if (
            Bec.METADATA in node_data_row.keys()
            and node_data_row[Bec.METADATA] is not None
        ):
            # Extract metadata fields and add them to the main node data
            metadata = node_data_row.pop(Bec.METADATA)
            # Bug with source in fact.
            for answer in metadata[Bec.ANSWER]:
                fact_string = answer.get(Fn.FACT, "")
                cleaned_fact = re.sub(Pat.GENERAL_CITATION, "", fact_string)
                answer[Fn.FACT] = cleaned_fact
            node_data_row.update(metadata)
        else:
            keys_to_set = [Bec.ANSWER, Bec.LVL, Bec.CLUSTER_ID]
            for key in keys_to_set:
                node_data_row[key] = None
        nodes_data.append(node_data_row)
    node_df = pd.DataFrame(nodes_data)

    # Save original node IDs
    node_id_ser = node_df[Col.ID]
    original_ids = node_id_ser.copy()
    id_map = {}  # original_id -> new_integer_id mapping

    # Define start_id for new nodes
    numeric_ids = pd.to_numeric(node_id_ser, errors="coerce")
    max_original_id = numeric_ids.max()
    if pd.notna(max_original_id):
        new_id = int(max_original_id) + 1
    else:
        new_id = Dv.START_NODE_ID

    # Create mapping of original to new IDs
    ids_to_replace = numeric_ids.isna()
    new_id_inc = 0
    for _, (idx, original_id) in enumerate(zip(ids_to_replace, original_ids)):
        if idx:
            new_id = new_id + new_id_inc
            new_id_inc += 1
            id_map[original_id] = new_id
        else:
            id_map[original_id] = int(original_id)

    # Apply the mapping to the node dataframe
    node_df[Col.ID] = node_df[Col.ID].map(id_map).astype(int)
    node_df.set_index(Col.ID, inplace=True)

    # Deal with node_df columns
    node_col_map = {
        Bec.NAME: Col.NAME,
        Bec.QUESTION: Col.QUESTION,
        Bec.ANSWER_STR: Col.ANSWER_STR,
        Bec.ANSWER: Col.ANSWER,
        Bec.LVL: Col.LVL,
        Bec.CLUSTER_ID: Col.CLUSTER_ID,
    }
    cols_to_keep = [
        col for col in node_col_map.keys() if col in node_df.columns
    ]
    node_df = node_df[cols_to_keep].rename(columns=node_col_map)

    # Deal with missing answers
    node_df[Col.MODIFIED] = 0
    empty_answers = node_df[Col.ANSWER].isna()
    if empty_answers.any():
        results_and_flags_series = node_df.loc[
            empty_answers, Col.ANSWER_STR
        ].apply(
            lambda answer_str: extract_facts_with_sources_and_fix_old(
                answer_str
            )
        )
        node_df.loc[results_and_flags_series.index, Col.ANSWER] = (
            results_and_flags_series.apply(lambda x: x[0])
        )
        node_df.loc[results_and_flags_series.index, Col.MODIFIED] = (
            results_and_flags_series.apply(lambda x: x[1]).astype(int)
        )

    # Extract citations and flat_part_ids
    node_df[Col.CITATION] = node_df[Col.ANSWER].apply(extract_citations)
    node_df = form_flat_part_ids(node_df)

    # Map Root to new ID
    root_id = id_map.get(root_id, int(root_id))
    # Add root to the nodes
    node_df.loc[root_id, [Col.LVL, Col.CLUSTER_ID]] = (
        Col.ROOT_LVL_VAL,
        Col.ROOT_CLUSTER_VAL,
    )
    node_df = handle_manual_nodes(node_df)

    # Create Edge DF
    edges_file = graph_files[Bec.EDGES_FILE]

    edges_data = []
    edges = edges_file[Bec.EDGE_KEY]
    for edge in edges:
        edge_data_row = dict()
        edge_data = edge[Bec.DATA_KEY]

        edge_data_row[Col.ID] = edge_data[Bec.EDGE_ID]

        if edge_data[Bec.SOURCE] == "nan" or edge_data[Bec.TARGET] == "nan":
            continue
        edge_data_row[Col.ORIGIN_CONCEPT_ID] = edge_data[Bec.SOURCE]
        edge_data_row[Col.TARGET_CONCEPT_ID] = edge_data[Bec.TARGET]

        edge_type = edge_data.get(Bec.TYPE)
        # noinspection PyUnreachableCode
        match edge_type:
            case Bec.INIT:
                edge_data_row[Col.TYPE] = Col.ART_EDGE_TYPE_VAL
            case Bec.GENERATED | Bec.MANUAL:
                edge_data_row[Col.TYPE] = Col.RELATED_TYPE_VAL
            case _:
                raise ApplyException("Invalid edge type in the mind map.")

        edge_weight = edge_data.get(Bec.WEIGHT)
        if edge_weight is not None and edge_weight != "nan":
            edge_data_row[Col.WEIGHT] = float(edge_weight)
        else:
            edge_data_row[Col.WEIGHT] = np.nan

        edges_data.append(edge_data_row)
    edge_df = pd.DataFrame(edges_data)

    if not edge_df.empty:
        edge_df.drop(columns=[Col.ID], inplace=True)

        # Update edge source and target columns with the new IDs
        origin_concept_ids = edge_df[Col.ORIGIN_CONCEPT_ID]
        edge_df[Col.ORIGIN_CONCEPT_ID] = origin_concept_ids.replace(id_map)

        target_concept_ids = edge_df[Col.TARGET_CONCEPT_ID]
        edge_df[Col.TARGET_CONCEPT_ID] = target_concept_ids.replace(id_map)

    return (
        MindMapData(node_df=node_df, edge_df=edge_df, root_id=root_id),
        id_map,
    )


def filter_by_document_id(
    node_df: pd.DataFrame, doc_ids: list[str]
) -> pd.DataFrame:
    """
    Filter facts and sources associated with given document_ids or row indices
    from the DataFrame's 'str_answer' column and filter citations
    from the 'citations' column
    that reference any of the document_ids. Flag modified rows.

    Args:
     node_df: pandas DataFrame with 'str_answer' column and optionally
        'citations' column
     doc_ids: List[str], the document ids to filter out

    Returns:
    - Filtered DataFrame with an additional column indicating modifications
    """
    new_node_df = node_df.copy()
    rows_to_drop: set[int] = set()

    # Iterative process to handle cascading deletions
    while True:
        iteration_drops: set[int] = set()

        for idx, row in new_node_df.iterrows():
            idx: int
            if idx in rows_to_drop:
                continue

            facts_list: list[dict[str, str | list[str]]] = row[Col.ANSWER]
            original_fact_count = len(facts_list)
            filtered_facts: list[dict[str, str | list[str]]] = []

            for fact in facts_list:
                initial_source_count = len(fact[Fn.SOURCE_IDS])
                valid_sources: list[str] = []

                for source_id in fact[Fn.SOURCE_IDS]:
                    source_is_dropped = False
                    if source_id.isdigit():
                        source_node = int(source_id)
                        if source_node in rows_to_drop:
                            source_is_dropped = True
                    else:
                        if any(
                            source_id.startswith(f"{doc_id}.")
                            for doc_id in doc_ids
                        ):
                            source_is_dropped = True

                    if not source_is_dropped:
                        valid_sources.append(source_id)

                if valid_sources:
                    fact_copy = fact.copy()
                    fact_copy[Fn.SOURCE_IDS] = valid_sources
                    filtered_facts.append(fact_copy)

                    if len(valid_sources) < initial_source_count:
                        new_node_df.at[idx, Col.MODIFIED] = 1
                elif not valid_sources:
                    new_node_df.at[idx, Col.MODIFIED] = 1

            if not filtered_facts:
                iteration_drops.add(idx)
                new_node_df.at[idx, Col.MODIFIED] = 1
            else:
                new_node_df.at[idx, Col.ANSWER] = filtered_facts
                if len(filtered_facts) < original_fact_count:
                    new_node_df.at[idx, Col.MODIFIED] = 1

            if idx not in iteration_drops:
                if Col.CITATION in row and row[Col.CITATION] is not None:
                    citation_tuple: tuple[str, ...] = row[Col.CITATION]
                    original_citation_count = len(citation_tuple)

                    filtered_citations_list = []
                    for citation_val in citation_tuple:
                        if not any(
                            str(citation_val).startswith(f"{doc_id}.")
                            for doc_id in doc_ids
                        ):
                            filtered_citations_list.append(citation_val)

                    filtered_citations = tuple(filtered_citations_list)
                    new_node_df.at[idx, Col.CITATION] = filtered_citations

                    if len(filtered_citations) < original_citation_count:
                        new_node_df.at[idx, Col.MODIFIED] = 1

        if not iteration_drops:
            break

        rows_to_drop.update(iteration_drops)

    final_df = new_node_df.drop(list(rows_to_drop))
    return form_flat_part_ids(final_df)
