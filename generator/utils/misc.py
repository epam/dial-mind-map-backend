import os
import threading
import time
import uuid
from typing import ClassVar, Sequence

import pandas as pd


class HybridIDGenerator:
    """
    A thread-safe hybrid ID generator
    that combines timestamp and a counter.

    Generates unique IDs by using microsecond-precision timestamps
    and a counter that increments when multiple IDs are requested
    within the same microsecond.
    """

    _prev_time: ClassVar[int] = 0
    _counter: ClassVar[int] = 0
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def get_next_id(cls) -> str:
        """
        Generate a unique ID by combining current time and a counter.

        Returns:
            str: A unique ID in the format "timestamp-counter"
        """
        with cls._lock:
            # Use time_ns() for better precision, convert to microseconds
            current_time = time.time_ns() // 1000

            if current_time == cls._prev_time:
                cls._counter += 1
            else:
                cls._counter = 0
                cls._prev_time = current_time

            return f"{current_time}-{cls._counter}"


class ContextPreservingAsyncIterator:
    """Wrapper that preserves context variables across async boundaries."""

    def __init__(self, iterator, context_vars):
        self.iterator = iterator
        self.context_vars = context_vars

    def __aiter__(self):
        return self

    # noinspection PyUnreachableCode
    async def __anext__(self):
        tokens = []
        try:
            for var, value in self.context_vars.items():
                tokens.append((var, var.set(value)))

            return await self.iterator.__anext__()
        except StopAsyncIteration:
            raise
        finally:
            for var, token in reversed(tokens):
                var.reset(token)


def concat_tuples(tuple_seq: Sequence[tuple]) -> tuple:
    final_tuple = ()
    for tuple_ in tuple_seq:
        final_tuple += tuple_
    return final_tuple


def env_to_bool(key: str, *, default: bool = False) -> bool:
    if (value := os.environ.get(key)) is None:
        return default
    return value.lower() in {"1", "t", "on", "true"}


def split_list_by_indices(
    my_list: list, indices_to_extract: set[int]
) -> tuple[list, list]:
    """
    Splits a list into two based on a list of indices.

    Args:
        my_list: The original list to split.
        indices_to_extract: A set of 0-based integer indices.

    Returns:
        A tuple containing two lists:
        - The first list contains elements from my_list at the specified indices.
        - The second list contains the remaining elements.
    """
    extracted_list = []
    remaining_list = []

    for index, item in enumerate(my_list):
        if index in indices_to_extract:
            extracted_list.append(item)
        else:
            remaining_list.append(item)

    return extracted_list, remaining_list


def is_valid_uuid(uuid_string: str, version=None) -> bool:
    """
    Checks if a string is a valid UUID.

    Args:
        uuid_string: The string to check.
        version: Optional. If specified (e.g., 4), checks if the UUID
                 is of that specific version.

    Returns:
        True if the string is a valid UUID (and matches the version if specified),
        False otherwise.
    """
    try:
        uuid_obj = uuid.UUID(uuid_string)
        if version is not None and uuid_obj.version != version:
            return False
        return True

    except ValueError:
        return False


def reindex_node_df_for_duplicates(
    node_df: pd.DataFrame, used_node_ids: set[int]
) -> pd.DataFrame:
    positions_to_replace = node_df.index.isin(used_node_ids)
    if positions_to_replace.any():
        numeric_index = pd.to_numeric(node_df.index, errors="coerce")
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
