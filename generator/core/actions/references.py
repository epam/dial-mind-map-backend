import re
from typing import Optional

import pandas as pd

from generator.common.constants import DataFrameCols as Col
from generator.common.constants import FieldNames as Fn
from generator.core.structs import MindMapData
from generator.core.utils.constants import DefaultValues as Dv
from generator.core.utils.constants import Patterns as Pat
from generator.core.utils.str_utils import is_trivial_content


def extract_source_ids(answer: list[dict]) -> tuple:
    """Extract all citations from the structured answers"""
    source_ids = []
    for answer_part in answer:
        source_ids.extend(answer_part[Fn.SOURCE_IDS])

    return tuple(sorted(set(source_ids)))


# noinspection PyUnreachableCode
def transform_sources(answer: list[dict], mapping: dict) -> list[dict]:
    """
    Transforms an answer by replacing source ids with their
    corresponding flat part ids using the provided mapping.
    If a source id is not found in the mapping, it is kept as the
    original string.
    """
    transformed_answer = []
    if not isinstance(answer, list):
        return answer

    for answer_part in answer:
        if not isinstance(answer_part, dict):
            transformed_answer.append(answer_part)
            continue

        source_ids = answer_part.get(Fn.SOURCE_IDS, [])
        transformed_source_ids = []

        if isinstance(source_ids, list):
            for source_id in source_ids:
                mapped_id = mapping.get(source_id, source_id)
                transformed_source_ids.append(mapped_id)
        else:
            transformed_source_ids = source_ids

        transformed_answer_part = answer_part.copy()
        transformed_answer_part[Fn.SOURCE_IDS] = transformed_source_ids
        transformed_answer.append(transformed_answer_part)

    return transformed_answer


def restore_original_ids(
    mind_map_data: MindMapData, id_map: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Restores the original IDs in node_df and edge_df using the provided
    mapping.

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

    node_df_restored = node_df.copy()
    edge_df_restored = edge_df.copy()

    node_df_restored = node_df_restored.rename(index=reverse_mapping)

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

        if re.match(r"\^\[\d+\.\d+]\^$", a[left : right + 1]):
            ranges.append([left, right])
        elif re.match(r"\^\[\d+\.\d+] \^$", a[left : right + 1]):
            ranges.append([left, right])
        elif re.match(r"\^\[\d+\.\d+\.\d+]\^$", a[left : right + 1]):
            ranges.append([left, right])
        elif re.match(r"\^\[\d+\.\d+\.\d+] \^$", a[left : right + 1]):
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

    return a.replace("]^ ^[", "]^^[")


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
            # insert ".1" between the two parts, resulting in "1.1.2"
            source_id = parts[0] + ".1." + parts[1]
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


def create_flat_part_df(node_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a DataFrame mapping citations to part IDs from the node
    data.
    """
    flat_data = [
        {Col.CITATION: cit, Col.FLAT_PART_ID: pid}
        for _, row in node_df.iterrows()
        for cit, pid in zip(row[Col.CITATION], row[Col.FLAT_PART_ID])
    ]
    return pd.DataFrame(flat_data).drop_duplicates()


def find_new_root_candidates(
    node_df: pd.DataFrame, edge_df: pd.DataFrame
) -> Optional[pd.DataFrame]:
    """
    Determines the best candidate(s) for a new root node based on
    connectivity and level.
    """
    if edge_df.empty:
        return None

    node_connections = pd.concat(
        [edge_df[Col.ORIGIN_CONCEPT_ID], edge_df[Col.TARGET_CONCEPT_ID]]
    )
    node_degree = (
        node_connections.value_counts().rename("num_neighbours").to_frame()
    )

    temp_df = node_df.merge(
        node_degree, left_index=True, right_index=True, how="left"
    )
    temp_df["num_neighbours"] = temp_df["num_neighbours"].fillna(0).astype(int)

    sorted_df = temp_df.sort_values(
        by=[Col.LVL, "num_neighbours"], ascending=[False, False]
    )

    top_n = min(len(node_degree), 5)
    if top_n == 0:
        return None

    top_node_ids = sorted_df.head(top_n).index
    return node_df.loc[top_node_ids].copy()
