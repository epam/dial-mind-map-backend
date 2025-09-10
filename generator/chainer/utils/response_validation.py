from pydantic import ValidationError


def is_field_invalid(
    error: ValidationError, field_name: str, error_types: set[str]
) -> bool:
    """
    Inspects a Pydantic validation error
    to check if it was caused by the provided field being invalid.
    """
    for e in error.errors():
        if field_name in e["loc"] and e["type"] in error_types:
            return True
    return False
