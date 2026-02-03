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
