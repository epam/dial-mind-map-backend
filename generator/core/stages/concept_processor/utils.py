import logging
from typing import Any, Hashable, Union

import numpy as np
import pandas as pd

from generator.chainer import Embedder
from generator.common.constants import ColVals
from generator.common.constants import DataFrameCols as Col
from generator.common.constants import FieldNames as Fn
from generator.core.utils.constants import FACT, SOURCE_IDS


def add_concepts(
    concept_df: pd.DataFrame,
    concepts_to_add: list[dict | str],
    max_id: int | None = None,
) -> pd.DataFrame:
    """
    Adds new concepts to a DataFrame, deactivating their sources.

    This function orchestrates adding new concepts. It iterates through
    a list of instructions, where each is either a dictionary defining
    a new synthesized concept or a string naming an existing concept to
    duplicate.

    For each new concept, its source concepts are marked as inactive.
    New concepts are assigned unique IDs starting from the maximum
    existing ID.

    Args:
        concept_df: The primary DataFrame of all existing concepts. It
            must have an integer index serving as the concept ID.
        concepts_to_add: A list of instructions for creating concepts.
            - If an item is a dict, it defines a new concept.
            - If an item is a str, it names an existing concept to
            duplicate.
        max_id: An optional integer. If provided, new concept IDs will
            start from `max(max_existing_id, max_id) + 1`.

    Returns:
        A new DataFrame with original and new concepts. Source concepts
        are marked as inactive. Returns the original DataFrame if no
        valid concepts could be added.
    """
    new_concepts_list = []
    ids_to_deactivate = set()

    for item in concepts_to_add:
        if isinstance(item, dict):
            new_concept, source_ids = prepare_new_concept_from_dict(
                item, concept_df
            )
        else:  # Assumes item is a string (source_name)
            new_concept, source_ids = prepare_new_concept_from_name(
                item, concept_df
            )

        if new_concept:
            new_concepts_list.append(new_concept)
            ids_to_deactivate.update(source_ids)

    if not new_concepts_list:
        return concept_df

    # Deactivate all source concepts in a single, efficient operation.
    if ids_to_deactivate:
        concept_df.loc[list(ids_to_deactivate), Col.IS_ACTIVE_CONCEPT] = (
            ColVals.FALSE_INT
        )

    max_existing_id = max(concept_df.index.max(), max_id or -1)
    new_indices = range(
        max_existing_id + 1, max_existing_id + 1 + len(new_concepts_list)
    )
    new_concepts_df = pd.DataFrame(new_concepts_list, index=new_indices)

    return pd.concat([concept_df, new_concepts_df])


def format_source_ids(row: pd.Series) -> list[dict]:
    """
    Formats an answer by associating each fact with all source parts.

    This transforms a concept's `ANSWER` field. It assumes the concept
    row has a `FLAT_PART_ID` column (a list of all source document
    parts for the whole concept). It then creates a new `ANSWER` where
    each fact is explicitly paired with this complete list of source
    part IDs.

    Args:
        row: A pandas Series for a single concept. Must contain
            `Col.ANSWER` and `Col.FLAT_PART_ID`.

    Returns:
        A list of dicts, where each dict contains a 'fact' and its
        associated 'source_ids', e.g.,
        `[{'fact': '...', 'source_ids': [1, 2, 3]}, ...]`.
    """
    return [
        {
            FACT: answer_part[Fn.FACT],
            SOURCE_IDS: sorted(row[Col.FLAT_PART_ID]),
        }
        for answer_part in row[Col.ANSWER]
    ]


def get_concept_id_changes(concept_df: pd.DataFrame) -> dict[int, list[int]]:
    """
    Maps old source concept IDs to their new synthesized concept IDs.

    When new concepts are created from old ones (synthesis), the new
    concepts store parent IDs in the `SOURCE_IDS` column. This function
    inverts that relationship to create a lookup table:
    `{old_id: [new_id_1, new_id_2, ...]}`.

    This is useful for propagating changes from old concepts to the new
    ones that replaced them.

    Args:
        concept_df: DataFrame of concepts after synthesis. Must have a
            `SOURCE_IDS` column containing lists of parent concept IDs.

    Returns:
        A dict mapping each source concept ID to a list of new concept
        IDs that were created from it.
    """
    exploded_df = concept_df.explode(Col.SOURCE_IDS)
    grouped = exploded_df.groupby(Col.SOURCE_IDS)
    return grouped.apply(lambda group: list(group.index)).to_dict()


def make_unique_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures all concept names are unique by suffixing duplicates.

    This function identifies concepts with duplicate names and renames
    them based on their `IS_ACTIVE_CONCEPT` status. The logic is:
    1.  One active, multiple inactive: Suffix inactive with "_".
    2.  Multiple active: Suffix active with "_1", "_2", etc., and
        any inactive with "_".
    3.  All inactive: Suffix all with "_".

    This preserves the original name for a single active concept where
    possible.

    Args:
        df: The concept DataFrame, modified in-place. Requires columns
            `Col.NAME` and `Col.IS_ACTIVE_CONCEPT`.

    Returns:
        The DataFrame with modified names (operation is in-place).
    """
    name_counts = df[Col.NAME].value_counts()
    duplicates = name_counts[name_counts > 1].index

    for name in duplicates:
        name_mask = df[Col.NAME] == name
        active_col = df.loc[name_mask, Col.IS_ACTIVE_CONCEPT]
        active_mask = pd.Series(active_col == ColVals.TRUE_INT)
        inactive_mask = pd.Series(active_col == ColVals.FALSE_INT)
        num_active = active_mask.sum()
        inactive_dup_mask = inactive_mask & name_mask

        if num_active == 1 and inactive_mask.any():
            df.loc[inactive_dup_mask, Col.NAME] += "_"
        elif num_active > 1:
            active_indices = df[active_mask & name_mask].index
            for i, idx in enumerate(active_indices, 1):
                df.at[idx, Col.NAME] = f"{name}_{i}"
            df.loc[inactive_dup_mask, Col.NAME] += "_"
        else:
            df.loc[name_mask, Col.NAME] += "_"
    return df


def prepare_new_concept_from_dict(
    concept_dict: dict, concept_df: pd.DataFrame
) -> tuple[dict, list[int]]:
    """
    Prepares a new concept from a dict, synthesizing data from sources.

    This helper processes a concept defined as a dictionary. It
    validates `source_ids`, aggregates `flat_part_id`s from all
    sources, and sets standard fields.

    Optimization: If the new concept derives from a single source and
    the question is unchanged, the source's embedding is reused to
    avoid a costly re-embedding operation.

    Args:
        concept_dict: A dictionary defining the new concept. Must
            contain `QUESTION` and may contain `SOURCE_IDS`.
        concept_df: The existing DataFrame of all concepts, used to
            look up source concept data.

    Returns:
        A tuple containing:
        - A dict for the prepared new concept.
        - A list of valid source IDs that should be deactivated.
    """
    valid_source_ids = []
    for source_id in concept_dict.get(Col.SOURCE_IDS, []):
        if source_id in concept_df.index:
            valid_source_ids.append(source_id)
        else:
            logging.warning(f"Nonexistent source_id: {source_id} skipped.")

    flat_part_ids = set()
    embedding = None
    for source_id in valid_source_ids:
        source_concept = concept_df.loc[source_id]
        flat_part_ids.update(source_concept.get(Col.FLAT_PART_ID, []))

    # Inherit embedding only if the new concept is synthesized from a
    # single source and the question has not been changed.
    if len(valid_source_ids) == 1:
        source_concept = concept_df.loc[valid_source_ids[0]]
        is_question_diff = bool(
            source_concept[Col.QUESTION] != concept_dict[Col.QUESTION]
        )
        if not is_question_diff:
            embedding = (
                source_concept[Col.EMBEDDING]
                if Col.EMBEDDING in source_concept
                else None
            )

    concept_dict[Col.FLAT_PART_ID] = sorted(list(flat_part_ids))
    concept_dict[Col.EMBEDDING] = embedding
    concept_dict[Col.IS_ACTIVE_CONCEPT] = ColVals.TRUE_INT
    concept_dict[Col.SOURCE_IDS] = valid_source_ids

    return concept_dict, valid_source_ids


def prepare_new_concept_from_name(
    source_name: str, concept_df: pd.DataFrame
) -> tuple[dict | None, list[int]]:
    """
    Prepares a new concept by duplicating an existing active concept.

    Finds an active concept by its name, copies it, and prepares it as
    a "new" version. The new concept's level (`LVL`) is incremented,
    and its `SOURCE_IDS` is set to the ID of the original.

    This is used for "evolving" a concept, where the original is
    deactivated and replaced by a new version.

    Args:
        source_name: The name of the active concept to duplicate.
        concept_df: The existing DataFrame of all concepts.

    Returns:
        A tuple containing:
        - A dict for the new concept, or None if not found.
        - A list with the single source ID to be deactivated.
    """
    active_mask = concept_df[Col.IS_ACTIVE_CONCEPT] == ColVals.TRUE_INT
    name_mask = concept_df[Col.NAME] == source_name
    # The use of .index[0] assumes a unique active concept exists
    # for the given name. This is a strong but necessary assumption
    # for this logic path.
    source_ids = concept_df.loc[name_mask & active_mask].index
    if source_ids.empty:
        logging.warning(f"No active concept found for name: {source_name}")
        return None, []

    source_id = int(source_ids.values[0])
    new_concept = concept_df.loc[source_id].to_dict()
    new_concept[Col.LVL] += 1
    new_concept[Col.SOURCE_IDS] = [source_id]

    return new_concept, [source_id]


def rename_concept_fields(
    concept: dict[str, Any], qapair_name_to_id: dict[str, int] | None = None
) -> dict[str, Any]:
    """
    Renames fields in a concept dict to the standard format.

    This acts as an adapter, translating field names from an external
    source (e.g., `concept_name`) to the internal, standard names
    (e.g., `NAME`). It can also convert source concept names to IDs if a
    mapping is provided.

    Args:
        concept: A dict representing a concept with non-standard fields.
        qapair_name_to_id: An optional mapping from concept names to IDs
            to convert `source_concept_names` to `SOURCE_IDS`.

    Returns:
        The same dictionary, modified in-place, with renamed fields.
    """
    concept[Col.NAME] = concept.pop(Fn.CONCEPT_NAME)
    concept[Col.QUESTION] = concept.pop(Fn.QUESTION)
    concept[Col.ANSWER] = concept.pop(Fn.ANSWER)

    if qapair_name_to_id:
        concept[Col.SOURCE_IDS] = [
            qapair_name_to_id[s]
            for s in concept.pop(Fn.SOURCE_CONCEPT_NAMES, [])
            if s in qapair_name_to_id
        ]
    else:
        concept[Col.SOURCE_IDS] = concept.pop(Fn.SOURCE_CONCEPT_IDS, [])
    return concept


def repr_answer(row: pd.Series) -> str:
    """
    Creates a human-readable string of an answer with citations.

    Processes a concept's `ANSWER` column, which contains structured
    facts and sources. It concatenates facts into a single string,
    appending source citations in brackets for each fact.
    Example: "The sky is blue[1][2]. Photosynthesis needs light[3]."

    It handles answer parts that are either dicts or objects.

    Args:
        row: A pandas Series for a single concept, containing an
            `ANSWER` column.

    Returns:
        A single string representing the full answer with citations.
    """
    repr_facts = []
    for part in row.get(Col.ANSWER, []):
        if isinstance(part, dict):
            fact = part.get(FACT, "")
            sources = part.get(SOURCE_IDS, [])
        else:
            fact = getattr(part, Fn.FACT, "")
            sources = getattr(part, Fn.CITATIONS, [])

        sources_str = f"[{']['.join(map(str, sources))}]"
        repr_facts.append(fact + sources_str)
    return " ".join(repr_facts)


def repr_concepts(
    concept_df: pd.DataFrame, concept_ids: list[int] | None = None
) -> list[dict[Hashable, Any]]:
    """
    Creates a list of dictionaries representing selected concepts.

    Extracts key info (`NAME`, `QUESTION`, `ANSWER`) from the concept
    DataFrame for presentation or serialization. It can operate on the
    entire DataFrame or a specified subset of concepts.

    Args:
        concept_df: The DataFrame containing all concept data.
        concept_ids: An optional list of concept IDs to filter by. If
            provided, only these concepts will be included.

    Returns:
        A list of dictionaries, where each dict represents a concept.
    """
    target_df = (
        concept_df if concept_ids is None else concept_df.loc[concept_ids]
    )

    return target_df[[Col.NAME, Col.QUESTION, Col.ANSWER]].to_dict(
        orient="records"
    )


def split_list_into_batches(
    items: list[Any], batch_size: int = 10
) -> list[list[Any]]:
    """
    Splits a list into smaller, evenly-sized batches.

    A simple utility for batch processing, useful for feeding data to
    models or APIs that have batch size limits.

    Args:
        items: The list of items to split.
        batch_size: The maximum number of items in each batch.

    Returns:
        A list of lists, where each inner list is a batch. The last
        batch may be smaller. Returns an empty list for an empty input.
    """
    if not items:
        return []
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def update_rel_df(
    rel_df: pd.DataFrame,
    new_rel_df: pd.DataFrame,
    concept_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Updates the relationships DataFrame with new and propagated
    relations.

    This performs a three-step update of concept relationships:
    1. Adds new relationships from `new_rel_df`.
    2. Propagates existing relationships from old concepts to the new
       ones that synthesized them (via `update_rels`).
    3. Deduplicates relationships, keeping the one with the highest
       `WEIGHT` for any given (origin, target) pair.

    Args:
        rel_df: The DataFrame of existing concept relationships.
        new_rel_df: A DataFrame of new relationships to add.
        concept_df: The latest concept DataFrame, used to resolve ID
            changes from synthesis.

    Returns:
        The fully updated and deduplicated relationships DataFrame.
    """
    if not new_rel_df.empty:
        rel_df = pd.concat([rel_df, new_rel_df], ignore_index=True)

    qapair_id_changes = get_concept_id_changes(concept_df)
    rel_df = update_rels(rel_df, qapair_id_changes)

    # Sort by weight to ensure the strongest relationship is kept.
    rel_df.sort_values(Col.WEIGHT, ascending=False, inplace=True)
    # Remove duplicate relationships, keeping the one with the highest weight.
    return rel_df.drop_duplicates(
        subset=[Col.ORIGIN_CONCEPT_ID, Col.TARGET_CONCEPT_ID],
        keep="first",
        ignore_index=True,
    )


def update_rels(
    rel_df: pd.DataFrame, source_id_dict: dict[int, list[int]]
) -> pd.DataFrame:
    """Propagates relationships from old to new synthesized concepts.

    For an existing relationship A -> B, if A was used to create A1
    and A2, and B was used to create B1, this function generates new
    relationships: A1 -> B1 and A2 -> B1.

    The weight of propagated relationships is slightly decayed
    (multiplied by 0.99) to reflect a potential loss of specificity.

    Args:
        rel_df: The DataFrame of relationships to update.
        source_id_dict: A mapping from old concept IDs to the new IDs
            that replaced them (from `get_concept_id_changes`).

    Returns:
        A new DataFrame with original and propagated relationships.
        Self-referential relationships (e.g., A1 -> A1) are excluded.
    """
    new_relations = []
    for _, row in rel_df.iterrows():
        origin_id = int(row.at[Col.ORIGIN_CONCEPT_ID])
        target_id = int(row.at[Col.TARGET_CONCEPT_ID])

        new_origins = source_id_dict.get(origin_id, [origin_id])
        new_targets = source_id_dict.get(target_id, [target_id])
        was_updated = origin_id in source_id_dict or target_id in source_id_dict

        for new_o in new_origins:
            for new_t in new_targets:
                if new_o == new_t:
                    continue
                new_row = row.copy()
                (
                    new_row[Col.ORIGIN_CONCEPT_ID],
                    new_row[Col.TARGET_CONCEPT_ID],
                ) = (new_o, new_t)
                if was_updated:
                    new_row[Col.WEIGHT] *= 0.99  # Decay weight
                new_relations.append(new_row)

    return (
        pd.DataFrame(new_relations).reset_index(drop=True)
        if new_relations
        else pd.DataFrame(columns=rel_df.columns)
    )


def get_max_value_or_negative_one(
    series: Union[pd.Series, list, int, float, np.number],
) -> int:
    """Gets max value from a collection, with special handling for -1.

    Calculates the maximum value of a list or pandas Series. However,
    if -1 is present anywhere in the collection, the function
    immediately returns -1. This is useful when -1 is a sentinel value
    indicating a special state that must take precedence.

    Args:
        series: A pandas Series, list of numbers, or a single number.

    Returns:
        The integer maximum value, or -1 if the collection is empty or
        contains the value -1.
    """
    # Handle single number inputs first.
    if isinstance(series, (int, float, np.number)):
        return -1 if series == -1 else int(series)

    # Ensure we are working with a pandas Series.
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    if series.empty:
        return -1

    # The core rule: if -1 is present, it takes precedence.
    if -1 in series.values:
        return -1
    else:
        return int(series.max())


def embed_active_concepts(qapair_df: pd.DataFrame) -> pd.DataFrame:
    """Generates vector embeddings for active concepts that lack them.

    This identifies all "active" concepts (`IS_ACTIVE_CONCEPT` is true)
    that do not yet have an embedding (`EMBEDDING` is null). It then
    extracts their `QUESTION` text and uses the `Embedder` to generate
    embeddings in a single batch call. The new embeddings are then
    placed into the DataFrame.

    Args:
        qapair_df: The DataFrame containing all concepts.

    Returns:
        The DataFrame with updated embeddings for active concepts. The
        operation is performed in-place.
    """
    # The use of boolean masking is highly efficient for selecting
    # data in pandas and follows best practices.
    qapairs_wo_embedding_mask = qapair_df[Col.EMBEDDING].isna()
    active_qapairs_mask = qapair_df[Col.IS_ACTIVE_CONCEPT] == ColVals.TRUE_INT

    qapairs_to_embed_mask = qapairs_wo_embedding_mask & active_qapairs_mask
    questions_to_embed = qapair_df.loc[qapairs_to_embed_mask, Col.QUESTION]

    if not questions_to_embed.empty:
        embeddings_list = Embedder.embed(pd.Series(questions_to_embed).tolist())
        embeddings = pd.Series(embeddings_list, index=questions_to_embed.index)
        qapair_df.loc[qapairs_to_embed_mask, Col.EMBEDDING] = embeddings

    return qapair_df


def replace_chunk_id_with_index(
    row: pd.Series,
    citation_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Replaces chunk IDs in answer citations with structured indices.

    To be applied to a concept row, this function processes the `ANSWER`
    field. It converts raw `chunk_id` citations to flat indices using a
    provided `citation_df` mapping.

    An optimized `chunk_id -> index` map is used for fast lookups. If
    a `chunk_id` cannot be found, the original answer part for that
    fact is preserved to prevent data loss.

    Args:
        row: A DataFrame row (Series) containing an `ANSWER` column.
        citation_df: A DataFrame mapping chunk IDs to indices. Expected
            to have two columns: [index_col, chunk_id_col].

    Returns:
        A new list for the `ANSWER` field, where citations have been
        converted from chunk IDs to indices.
    """
    # REFACTOR: This method was completely rewritten for clarity and
    # efficiency. Instead of looping and filtering the DataFrame for
    # every single chunk_id, we create a reverse mapping once, which
    # allows for a much faster and cleaner lookup.
    index_col = citation_df.columns[0]
    chunk_id_col = citation_df.columns[1]
    chunk_id_to_index_map = pd.Series(
        citation_df[index_col].values, index=citation_df[chunk_id_col]
    ).to_dict()

    new_annotated_answer = []
    for answer_part in row[Col.ANSWER]:
        # Helper to consistently extract data from dict or object.
        text = getattr(answer_part, Fn.FACT, answer_part.get(FACT))
        chunk_ids = getattr(
            answer_part, Fn.CITATIONS, answer_part.get(SOURCE_IDS)
        )

        if not chunk_ids:
            new_annotated_answer.append(answer_part)
            continue

        # Ensure chunk_ids is always a list for consistent iteration.
        chunk_ids_list = (
            [chunk_ids] if isinstance(chunk_ids, (str, int)) else chunk_ids
        )
        found_indices = set()
        conversion_successful = True

        for chunk_id in chunk_ids_list:
            found_index = chunk_id_to_index_map.get(chunk_id)
            if found_index is not None:
                found_indices.add(found_index)
            else:
                # If any chunk_id cannot be found, the conversion for
                # this entire answer part fails.
                conversion_successful = False
                break

        if conversion_successful:
            new_annotated_answer.append(
                {FACT: text, SOURCE_IDS: sorted(list(found_indices))}
            )
        else:
            # If conversion failed, preserve the original answer part.
            new_annotated_answer.append(answer_part)

    return new_annotated_answer
