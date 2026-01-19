from contextvars import ContextVar

from .llm import LLMCostHandler

cur_llm_cost_handler = ContextVar[LLMCostHandler]("llm_cost_handler")
