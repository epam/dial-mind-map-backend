import os


def env_to_bool(key: str, *, default: bool = False) -> bool:
    if (value := os.environ.get(key)) is None:
        return default
    return value.lower() in {"1", "t", "on", "true"}
