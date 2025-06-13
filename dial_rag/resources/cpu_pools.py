import asyncio
import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import cache
from multiprocessing import get_context

from dial_rag.utils import int_env_var

CPU_COUNT = os.cpu_count() or 1


# 1 thread here, because inference is already parallelized inside the model,
# and we expect the batching to be done by the caller
EMBED_DOCUMENTS_WORKERS = int_env_var("EMBED_DOCUMENTS_WORKERS", 1)

# We do not want question answering to be blocked by the documents indexing
EMBED_QUERY_WORKERS = int_env_var("EMBED_QUERY_WORKERS", 1)

# Legacy way to set number of workers
DOCUMENT_LOADERS_WORKERS = int_env_var("DOCUMENT_LOADERS_WORKERS", max(1, CPU_COUNT - 2))


logger = logging.getLogger(__name__)


class UnpicklableExceptionError(RuntimeError):
    pass


def _run_in_process_wrapper(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        try:
            # python has an issue if unpicklable exception will be passed between processes
            # https://github.com/python/cpython/issues/120810
            # The exception created with kwargs could cause the issue
            pickle.loads(pickle.dumps(e))
        except Exception as pe:
            logger.exception(pe)
            # Unpicklable exception could break the process pool and cause the following error:
            # `concurrent.futures.process.BrokenProcessPool: A child process terminated abruptly, the process pool is not usable anymore`
            # To avoid this, we raise a custom exception with the original traceback in the __cause__ attribute
            raise UnpicklableExceptionError("Unpicklable exception raised in subprocess") from e
        raise


class CpuPools:
    indexing_cpu_pool: ProcessPoolExecutor
    indexing_embeddings_pool: ThreadPoolExecutor
    query_embeddings_pool: ThreadPoolExecutor

    def __init__(self) -> None:
        # Using process pool for indexing to avoid GIL limitations
        self.indexing_cpu_pool = ProcessPoolExecutor(
            max_workers=DOCUMENT_LOADERS_WORKERS,
            # Spawn is used to avoid inheriting the file descriptors from the parent process
            mp_context=get_context("spawn"),
        )

        self.indexing_embeddings_pool = ThreadPoolExecutor(
            max_workers=EMBED_DOCUMENTS_WORKERS,
            thread_name_prefix="indexing_embeddings"
        )

        # TODO: Do we need a separate pool for query embeddings?
        self.query_embeddings_pool = ThreadPoolExecutor(
            max_workers=EMBED_QUERY_WORKERS,
            thread_name_prefix="query_embeddings"
        )

    def _run_in_pool(self, pool, func, *args, **kwargs):
        return asyncio.get_running_loop().run_in_executor(pool, func, *args, **kwargs)

    def run_in_indexing_cpu_pool(self, func, *args, **kwargs):
        return self._run_in_pool(self.indexing_cpu_pool, _run_in_process_wrapper, func, *args, **kwargs)

    def run_in_indexing_embeddings_pool(self, func, *args, **kwargs):
        return self._run_in_pool(self.indexing_embeddings_pool, func, *args, **kwargs)

    def run_in_query_embeddings_pool(self, func, *args, **kwargs):
        return self._run_in_pool(self.query_embeddings_pool, func, *args, **kwargs)

    @staticmethod
    @cache
    def instance():
        return CpuPools()


async def warmup_cpu_pools():
    """Warm up the pools to avoid the first call overhead"""
    cpu_pools = CpuPools.instance()
    await cpu_pools.run_in_indexing_cpu_pool(sum, range(10))
    await cpu_pools.run_in_indexing_embeddings_pool(sum, range(10))
    await cpu_pools.run_in_query_embeddings_pool(sum, range(10))


def run_in_indexing_cpu_pool(func, *args, **kwargs):
    return CpuPools.instance().run_in_indexing_cpu_pool(func, *args, **kwargs)


def run_in_indexing_embeddings_pool(func, *args, **kwargs):
    return CpuPools.instance().run_in_indexing_embeddings_pool(func, *args, **kwargs)


def run_in_query_embeddings_pool(func, *args, **kwargs):
    return CpuPools.instance().run_in_query_embeddings_pool(func, *args, **kwargs)
