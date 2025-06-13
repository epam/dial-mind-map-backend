import functools

from anyio import fail_after
from fastapi import HTTPException, Response

from general_mindmap.v2.dial.client import PROCESSING_LIMIT

INCORRECT_JSON_RESPONSE_ERROR = Response(
    status_code=400,
    content="The body should contain a valid json",
)


def timeout_after(timeout: float = PROCESSING_LIMIT):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                with fail_after(timeout):
                    return await func(*args, **kwargs)
            except TimeoutError:
                raise HTTPException(status_code=408, detail="Timeout")

        return wrapper

    return decorator
