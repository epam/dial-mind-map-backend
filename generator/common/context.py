from contextvars import ContextVar

from .llm import LLMCostHandler

cur_run_id = ContextVar[str]("run_id", default="none")
cur_llm_cost_handler = ContextVar[LLMCostHandler]("llm_cost_handler")
