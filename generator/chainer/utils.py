from pydantic import BaseModel, Field


def get_pydantic_model(
    model_name: str,
    fields_config: dict[str, tuple[type | str, Field]],
    docstring: str | None = None,
) -> type:
    """
    Creates a Pydantic BaseModel class with configurable field names.

    Args:
        model_name: Name of the class to create
        fields_config: Dictionary mapping field names
            to tuples of (type, Field)
        docstring: Optional documentation string for the class

    Returns:
        A new Pydantic BaseModel class
    """
    return type(
        model_name,
        (BaseModel,),
        {
            "__doc__": docstring,
            "__annotations__": {
                name: field_type
                for name, (field_type, _) in fields_config.items()
            },
            **{name: field for name, (_, field) in fields_config.items()},
        },
    )
