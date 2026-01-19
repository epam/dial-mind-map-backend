from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from common_utils.context import HybridIDGenerator, cur_run_id


class ContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        run_id = HybridIDGenerator.get_next_id()
        cur_run_id.set(run_id)
        return await call_next(request)
