from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler


class PromptLoggerCallback(AsyncCallbackHandler):
    """A callback handler that captures the prompts sent to the LLM."""

    def __init__(self):
        super().__init__()
        self._prompts: list[str] = []

    async def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        """
        Fires when the LLM is about to be called, capturing the prompts.
        """
        self._prompts.extend(prompts)

    def get_last_prompt(self) -> str | None:
        """Returns the last captured prompt, if any."""
        return self._prompts[-1] if self._prompts else None
