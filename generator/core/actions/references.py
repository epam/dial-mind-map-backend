import re

import pandas as pd

from ...chainer.constants import FieldNames as Fn
from ...utils.constants import DataFrameCols as Col
from ...utils.constants import DefaultValues as Dv
from ...utils.constants import Patterns as Pat
from ...utils.str_utils import is_trivial_content
from ..structs import MindMapData


def extract_citations(fact_list: list[dict]) -> tuple:
    """Extract all source_ids from the structured answers"""
    answer_source_ids = []
    for fact in fact_list:
        answer_source_ids.extend(fact[Fn.SOURCE_IDS])

    return tuple(sorted(set(answer_source_ids)))


def transform_sources(list_of_dicts: list[dict], mapping: dict) -> list[dict]:
    """
    Transforms a list of dictionaries by replacing source_id strings
    with their corresponding flat_part_id lists using the provided mapping.
    If a source_id is not found in the mapping, it is kept as the original string.
    """
    transformed_list = []
    if not isinstance(list_of_dicts, list):
        # Handle cases where the cell might not be a list (e.g., NaN, None)
        return (
            list_of_dicts  # Or return [] or None, depending on desired behavior
        )

    for item in list_of_dicts:
        if not isinstance(item, dict):
            # Handle cases where an item in the list is not a dictionary
            transformed_list.append(item)  # Keep the original item
            continue

        original_source_ids = item.get(
            "source_ids", []
        )  # Get source_ids, default to empty list if key missing
        transformed_source_ids = []

        if isinstance(original_source_ids, list):
            for source_id_str in original_source_ids:
                # Look up the source_id_str in the mapping
                # Use .get() to handle cases where the source_id_str is not in the map
                # If not found, .get() returns None by default. We'll keep the original string.
                mapped_id = mapping.get(
                    source_id_str, source_id_str
                )  # Default to original string if not found
                transformed_source_ids.append(mapped_id)
        else:
            # Handle cases where source_ids is not a list
            transformed_source_ids = original_source_ids  # Keep original value

        # Create a new dictionary with the transformed source_ids
        # Copy other keys from the original dictionary
        new_item = item.copy()
        new_item["source_ids"] = transformed_source_ids
        transformed_list.append(new_item)

    return transformed_list


def restore_original_ids(
    mind_map_data: MindMapData, id_map: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Restores the original IDs in node_df and edge_df using the provided mapping.

    Args:
        mind_map_data: Contains nodes and edges
        id_map: Dictionary mapping original IDs to integer IDs

    Returns:
        Tuple of (node_df, edge_df) with original IDs restored
    """
    node_df = mind_map_data.node_df
    edge_df = mind_map_data.edge_df

    # Create reverse mapping: integer_id -> original_id
    reverse_mapping = {v: k for k, v in id_map.items()}

    # Create a copy to avoid modifying the original
    node_df_restored = node_df.copy()
    edge_df_restored = edge_df.copy()

    # Replace integer IDs with original IDs in the index
    node_df_restored = node_df_restored.rename(index=reverse_mapping)

    # Update edge source and target columns
    edge_df_restored[Col.ORIGIN_CONCEPT_ID] = edge_df_restored[
        Col.ORIGIN_CONCEPT_ID
    ].replace(reverse_mapping)

    edge_df_restored[Col.TARGET_CONCEPT_ID] = edge_df_restored[
        Col.TARGET_CONCEPT_ID
    ].replace(reverse_mapping)

    return node_df_restored, edge_df_restored


def fix_llm_references(a: str):
    # Remove 0 references assuming they only mean user created the fact.
    zero_ref_patterns = [r"\^\[0\]\^", r"\^\[0\]\ \^"]

    for pattern in zero_ref_patterns:
        a = re.sub(pattern, "", a)

    left = a.find("^")

    ranges = []
    while True:
        right = a.find("^", left + 1)

        if right == -1:
            break

        if re.match(r"\^\[\d+\.\d+\]\^$", a[left : right + 1]):
            ranges.append([left, right])
        elif re.match(r"\^\[\d+\.\d+\]\ \^$", a[left : right + 1]):
            ranges.append([left, right])
        elif re.match(r"\^\[\d+\.\d+\.\d+\]\^$", a[left : right + 1]):
            ranges.append([left, right])
        elif re.match(r"\^\[\d+\.\d+\.\d+\]\ \^$", a[left : right + 1]):
            ranges.append([left, right])

        left = right

    positions = []
    for i in range(len(ranges) - 1):
        r1 = ranges[i]
        r2 = ranges[i + 1]

        if r1[1] == r2[0]:
            positions.append(r1[1])

    deleted = 0
    for i, pos in enumerate(positions):
        if a[pos + i - deleted - 1] == " ":
            deleted += 1
            a = a[: pos + i - deleted] + "^" + a[pos + i - deleted + 1 :]
        else:
            a = a[: pos + i - deleted] + "^" + a[pos + i - deleted :]

    a = a.replace("]^ ^[", "]^^[")

    return a


def extract_facts_with_sources(
    answer_str: str,
) -> list[dict[str, str | list[str]]]:
    results = []
    refs = list(re.finditer(Pat.FRONT_CITATION, answer_str))

    # If no references, the entire text is a fact with "No source" value
    if not refs:
        if answer_str.strip():
            results.append(
                {Fn.FACT: answer_str.strip(), Fn.SOURCE_IDS: [Dv.NO_SOURCE_ID]}
            )
        return results

    cur_char_idx = 0
    for ref in refs:
        # Text before this reference is part of a fact
        fact_text = answer_str[cur_char_idx : ref.start()].strip()
        cur_char_idx = ref.end()
        # Extract source ID from the reference
        source_id = ref.group(1)

        # If there's meaningful text (more than just punctuation),
        # add it with its source
        if fact_text and not is_trivial_content(fact_text):
            results.append({Fn.FACT: fact_text, Fn.SOURCE_IDS: [source_id]})

    # Handle any remaining text after the last reference
    if cur_char_idx < len(answer_str):
        remaining_text = answer_str[cur_char_idx:].strip()
        if remaining_text and not is_trivial_content(remaining_text):
            results.append(
                {Fn.FACT: remaining_text, Fn.SOURCE_IDS: [Dv.NO_SOURCE_ID]}
            )

    return results


def extract_facts_with_sources_and_fix_old(
    answer_str: str,
) -> tuple[list[dict[str, str | list[str]]], bool]:
    results = []
    refs = list(re.finditer(Pat.FRONT_CITATION, answer_str))
    source_updated = False

    # If no references, the entire text is a fact with "No source" value
    if not refs:
        if answer_str.strip():
            results.append(
                {Fn.FACT: answer_str.strip(), Fn.SOURCE_IDS: [Dv.NO_SOURCE_ID]}
            )
        return results, source_updated

    cur_char_idx = 0
    for ref in refs:
        # Text before this reference is part of a fact
        fact_text = answer_str[cur_char_idx : ref.start()].strip()
        cur_char_idx = ref.end()
        # Extract source ID from the reference
        source_id = ref.group(1)

        parts = source_id.split(".")
        if len(parts) == 2:
            # If there are exactly two parts (e.g., "1.2"),
            # append ".1"
            source_id = source_id + ".1"
            source_updated = True

        # If there's meaningful text (more than just punctuation),
        # add it with its source
        if fact_text and not is_trivial_content(fact_text):
            results.append({Fn.FACT: fact_text, Fn.SOURCE_IDS: [source_id]})

    # Handle any remaining text after the last reference
    if cur_char_idx < len(answer_str):
        remaining_text = answer_str[cur_char_idx:].strip()
        if remaining_text and not is_trivial_content(remaining_text):
            results.append(
                {Fn.FACT: remaining_text, Fn.SOURCE_IDS: [Dv.NO_SOURCE_ID]}
            )

    return results, source_updated
