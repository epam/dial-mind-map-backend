import logging
import re
from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException
from pydantic import ValidationError

from common_utils.logger_config import logger
from generator.adapter import GMContract as Gmc
from generator.adapter import GraphFilesAdapter, translate_graph_files
from generator.common.constants import ColVals
from generator.common.constants import DataFrameCols as Col
from generator.common.constants import FieldNames as Fn
from generator.common.exceptions import ApplyException
from generator.common.structs import EdgesFile, GraphFiles, NodesFile
from generator.core.structs import MindMapData
from generator.core.utils.constants import DefaultValues as Dv
from generator.core.utils.constants import EdgeWeights as Ew
from generator.core.utils.constants import Patterns as Pat
from generator.core.utils.misc import is_valid_uuid

from .references import (
    extract_facts_with_sources,
    extract_facts_with_sources_and_fix_old,
    extract_source_ids,
)


def validate_graph(
    in_graph_files: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str | Any]]]:
    """
    Validates graph nodes, removing those with invalid sources.

    An invalid source is one that:
    - Points to a non-existent or already invalidated node.
    - Is explicitly marked as 'no_source'.
    - Is a document citation with an improper format (e.g., not
      "1.2.3").

    The function iteratively removes problematic nodes and any other
    nodes that depend on them, handling cascading invalidations.

    Args:
        in_graph_files: A dictionary containing graph data, including
          nodes and the root ID.

    Returns:
        A tuple containing the updated graph_files with invalid nodes
        removed, and a list of the data from the removed nodes.

    Raises:
        HTTPException: If the root node is missing or becomes invalid.
    """
    logger.info("Validate graph: Start")
    try:
        graph_files = translate_graph_files(in_graph_files)
    except ValidationError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid graph data structure: {e}"
        )

    nodes_file = graph_files.nodes_file
    root_id = nodes_file.root_id

    if not root_id:
        raise HTTPException(
            status_code=400, detail="No root node found in graph files."
        )

    # Convert to node_df for easier data modifications
    nodes = nodes_file.nodes
    nodes_data = [node.data.model_dump() for node in nodes]
    node_df = pd.DataFrame(nodes_data)

    # Convert node details to list of facts and sources
    node_df[Col.ANSWER] = None
    empty_answers = node_df[Col.ANSWER].isna()
    node_df.loc[empty_answers, Col.ANSWER] = node_df.loc[
        empty_answers, Gmc.ANSWER_STR
    ].apply(lambda answer_str: extract_facts_with_sources(answer_str))

    problematic_ids = set()
    all_node_ids = set(node_df[Gmc.NODE_ID].tolist())

    # Iterative process to handle cascading problems
    while True:
        iteration_problematic_ids = set()
        valid_node_ids = all_node_ids - problematic_ids

        for _, row in node_df.iterrows():
            has_metadata = row[Gmc.METADATA] is not None

            node_id = row[Gmc.NODE_ID]
            if node_id in problematic_ids:
                continue

            facts_list: list[dict[str, str | list[str]]] = list(row[Col.ANSWER])
            is_node_problematic = False

            for fact in facts_list:
                for source_id in fact[Fn.SOURCE_IDS]:
                    # If source is a node
                    if source_id.isdigit() or is_valid_uuid(source_id):
                        if source_id not in valid_node_ids:
                            is_node_problematic = True
                            break
                    # If no source
                    elif source_id == Dv.NO_SOURCE_ID:
                        is_node_problematic = True
                        break
                    # If node has metadata, it is not None,
                    # and source is document: make sure that
                    # it has a version defined (has 3 numbers, not 2)
                    elif has_metadata and "." in source_id:
                        parts = source_id.split(".")
                        if len(parts) != 3 or any(
                            not part.isdigit() for part in parts
                        ):
                            is_node_problematic = True
                            break
                if is_node_problematic:
                    iteration_problematic_ids.add(node_id)
                    break

        if not iteration_problematic_ids:
            break

        problematic_ids.update(iteration_problematic_ids)

    valid_nodes = [
        node for node in nodes_file.nodes if node.data.id not in problematic_ids
    ]
    problematic_nodes_data = [
        node.data.model_dump()
        for node in nodes_file.nodes
        if node.data.id in problematic_ids
    ]

    # If root is invalid
    if root_id in problematic_ids:
        raise HTTPException(status_code=400, detail="Root is invalid")

    # Get the original edges from the input file
    original_edges = graph_files.edges_file.edges

    # Filter out edges that connect to a problematic node
    valid_edges = [
        edge
        for edge in original_edges
        if edge.data.source not in problematic_ids
        and edge.data.target not in problematic_ids
    ]

    # Create a new EdgesFile object with only the valid edges
    updated_edges_file = EdgesFile(edges=valid_edges)

    # Pack results back to graph files
    updated_nodes_file = NodesFile(nodes=valid_nodes, root_id=root_id)
    updated_graph_files = GraphFiles(
        nodes_file=updated_nodes_file, edges_file=updated_edges_file
    )

    aliased_model_instance = GraphFilesAdapter.model_validate(
        updated_graph_files
    )
    out_graph_files = aliased_model_instance.model_dump(by_alias=True)

    logger.info("Validate graph: End")

    return out_graph_files, problematic_nodes_data


def process_graph(
    graph_files: dict[str, Any],
) -> tuple[MindMapData, dict[str, int]]:
    """ID Map saves correspondence between int ids and uuids."""
    nodes_file = graph_files[Gmc.NODES_FILE]

    # Get Root ID
    root_id = nodes_file[Gmc.ROOT_KEY]
    if root_id is None:
        raise HTTPException(status_code=400, detail="No root node")

    # Create Node DF
    nodes_data = []
    nodes = nodes_file[Gmc.NODES_KEY]
    for node in nodes:
        node_data_row = node[Gmc.DATA_KEY].copy()
        if (
            Gmc.METADATA in node_data_row.keys()
            and node_data_row[Gmc.METADATA] is not None
        ):
            # Extract metadata fields and add them to the main node data
            metadata = node_data_row.pop(Gmc.METADATA)
            # Bug with source in fact.
            for answer in metadata[Gmc.ANSWER]:
                fact_string = answer.get(Fn.FACT, "")
                cleaned_fact = re.sub(Pat.GENERAL_CITATION, "", fact_string)
                answer[Fn.FACT] = cleaned_fact
            node_data_row.update(metadata)
        else:
            keys_to_set = [Gmc.ANSWER, Gmc.LVL, Gmc.CLUSTER_ID]
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
        Gmc.NAME: Col.NAME,
        Gmc.QUESTION: Col.QUESTION,
        Gmc.ANSWER_STR: Col.ANSWER_STR,
        Gmc.ANSWER: Col.ANSWER,
        Gmc.LVL: Col.LVL,
        Gmc.CLUSTER_ID: Col.CLUSTER_ID,
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
    node_df[Col.CITATION] = node_df[Col.ANSWER].apply(extract_source_ids)
    node_df = _form_flat_part_ids(node_df)

    # Map Root to new ID
    valid_root_id = id_map.get(root_id)
    if valid_root_id is None:
        valid_root_id = int(root_id)
    # Add root to the nodes
    node_df.loc[valid_root_id, [Col.LVL, Col.CLUSTER_ID]] = [
        ColVals.UNDEFINED,
        ColVals.UNDEFINED,
    ]
    node_df = _handle_manual_nodes(node_df)

    # Create Edge DF
    edges_file = graph_files[Gmc.EDGES_FILE]

    edges_data = []
    edges = edges_file[Gmc.EDGE_KEY]
    for edge in edges:
        edge_data_row = dict()
        edge_data = edge[Gmc.DATA_KEY]

        edge_data_row[Col.ID] = edge_data[Gmc.EDGE_ID]

        if edge_data[Gmc.SOURCE] == "nan" or edge_data[Gmc.TARGET] == "nan":
            continue
        edge_data_row[Col.ORIGIN_CONCEPT_ID] = edge_data[Gmc.SOURCE]
        edge_data_row[Col.TARGET_CONCEPT_ID] = edge_data[Gmc.TARGET]

        edge_type = edge_data.get(Gmc.TYPE)
        # noinspection PyUnreachableCode
        match edge_type:
            case Gmc.INIT:
                edge_data_row[Col.TYPE] = ColVals.ARTIFICIAL
            case Gmc.GENERATED | Gmc.MANUAL:
                edge_data_row[Col.TYPE] = ColVals.RELATED
            case _:
                raise ApplyException("Invalid edge type in the mind map.")

        edge_weight = edge_data.get(Gmc.WEIGHT)
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

        # Get the set of all valid node IDs from the node_df's index
        valid_node_ids = set(node_df.index)

        # Check which edges have both a valid source and a valid target
        is_source_valid = edge_df[Col.ORIGIN_CONCEPT_ID].isin(valid_node_ids)
        is_target_valid = edge_df[Col.TARGET_CONCEPT_ID].isin(valid_node_ids)

        # Keep only the edges where both source and target nodes exist
        edge_df = edge_df[is_source_valid & is_target_valid].copy()

    return (
        MindMapData(node_df=node_df, edge_df=edge_df, root_id=valid_root_id),
        id_map,
    )


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


def filter_graph_by_docs(
    graph_data: MindMapData, del_doc_ids: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filters nodes and edges, removing any associated with the given
    document IDs.
    """
    node_df = _filter_by_document_id(graph_data.node_df, del_doc_ids)
    valid_indices = set(node_df.index)
    edge_df = graph_data.edge_df
    if not edge_df.empty:
        edge_df = edge_df[
            edge_df[Col.ORIGIN_CONCEPT_ID].isin(valid_indices)
            & edge_df[Col.TARGET_CONCEPT_ID].isin(valid_indices)
        ]
    return node_df, edge_df


def recalculate_edge_weights(
    edge_df: pd.DataFrame, root_id: Any
) -> pd.DataFrame:
    """Applies default weights to edges that don't have them."""
    if Col.WEIGHT not in edge_df.columns:
        edge_df[Col.WEIGHT] = np.nan
    empty_weight_mask = edge_df[Col.WEIGHT].isnull()
    if not empty_weight_mask.any():
        return edge_df

    edge_df.loc[empty_weight_mask, Col.WEIGHT] = Ew.DEFAULT_EDGE_WEIGHT
    edge_df.loc[
        empty_weight_mask & (edge_df[Col.TYPE] == ColVals.ARTIFICIAL),
        Col.WEIGHT,
    ] = Ew.ARTIFICIAL_EDGE_WEIGHT
    edge_df.loc[
        empty_weight_mask & (edge_df[Col.TYPE] == ColVals.RELATED),
        Col.WEIGHT,
    ] = Ew.RELATED_EDGE_WEIGHT
    root_mask = (edge_df[Col.ORIGIN_CONCEPT_ID] == root_id) | (
        edge_df[Col.TARGET_CONCEPT_ID] == root_id
    )
    edge_df.loc[
        root_mask & (edge_df[Col.TYPE] == ColVals.RELATED), Col.WEIGHT
    ] = Ew.ROOT_RELATED_EDGE_WEIGHT
    return edge_df


def reindex_node_df_for_duplicates(
    node_df: pd.DataFrame, used_node_ids: set[int]
) -> pd.DataFrame:
    positions_to_replace = node_df.index.isin(used_node_ids)
    if positions_to_replace.any():
        numeric_index = pd.to_numeric(node_df.index.values, errors="coerce")
        max_int = numeric_index.max()
        start_new_index = int(max_int) + 1 if pd.notna(max_int) else 0

        num_replacements = positions_to_replace.sum()
        new_indices_sequence = range(
            start_new_index, start_new_index + num_replacements
        )
        new_indices_iterator = iter(new_indices_sequence)
        new_full_index = node_df.index.tolist()

        for i in range(len(new_full_index)):
            if positions_to_replace[i]:
                new_full_index[i] = next(new_indices_iterator)

        node_df.index = new_full_index

    return node_df


def _form_flat_part_ids(node_df: pd.DataFrame) -> pd.DataFrame:
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


def _handle_manual_nodes(node_df: pd.DataFrame) -> pd.DataFrame:
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


def _filter_by_document_id(
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

            facts_list: list[dict[str, str | list[str]]] = list(row[Col.ANSWER])
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
                    citation_tuple: tuple[str, ...] = tuple(row[Col.CITATION])
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
    return _form_flat_part_ids(final_df)
