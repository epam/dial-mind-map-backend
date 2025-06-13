import asyncio
from langchain.schema import BaseRetriever


class AsyncRetriever(BaseRetriever):
    inner: BaseRetriever

    def __init__(self, inner: BaseRetriever):
        super().__init__(inner=inner)

    def _get_relevant_documents(self, query: str, *args, **kwargs):
        return self.inner.get_relevant_documents(query, *args, **kwargs)

    async def _aget_relevant_documents(self, query: str, *args, **kwargs):
        try:
            return await self.inner.aget_relevant_documents(query, *args, **kwargs)
        except NotImplementedError:
            pass
        return await asyncio.get_running_loop().run_in_executor(
            None,
            self.inner.get_relevant_documents, query, *args, **kwargs)
