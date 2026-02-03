import uuid
from typing import Sequence


def concat_tuples(tuple_seq: Sequence[tuple]) -> tuple:
    final_tuple = ()
    for tuple_ in tuple_seq:
        final_tuple += tuple_
    return final_tuple


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
        - The first list contains elements from my_list at the specified
        indices.
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
        True if the string is a valid UUID (and matches the version if
        specified),
        False otherwise.
    """
    try:
        uuid_obj = uuid.UUID(uuid_string)
        if version is not None and uuid_obj.version != version:
            return False
        return True

    except ValueError:
        return False
