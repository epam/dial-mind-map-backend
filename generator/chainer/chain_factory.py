"""
Factory module for creating and running language model chains.
Provides utilities for creating specialized chain types
and executing them with progress monitoring.
"""

import asyncio as aio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Awaitable, TypeVar

from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel

import generator.chainer.model_handler as model_handler
import generator.chainer.prompt_templates as pt
import generator.chainer.response_formats as form
from generator.utils.constants import ChainTypes as Ct
from generator.utils.context import cur_llm_cost_handler

T = TypeVar("T")


@dataclass
class ChainComponents:
    """
    Represents the components necessary
    to create a specific type of chain.

    Attributes:
        prompt_gen: Template or callable for generating prompts.
        response_format: The data structure format expected as output.
    """

    prompt_gen: Callable | ChatPromptTemplate
    response_format: type[BaseModel] | None


class ChainCreator:
    """
    Factory class for creating different types of language model chains.
    Each chain is composed of a prompt generator
    and a structured output format.
    """

    default_chat_model = model_handler.ModelCreator.get_chat_model()

    CHAIN_TYPE_TO_COMPONENTS = {
        Ct.TEXTUAL_EXTRACTION: ChainComponents(
            pt.TextualExtraction.get_cpt(), form.ExtractionResult
        ),
        Ct.PDF_EXTRACTION: ChainComponents(
            pt.PDFExtraction.get_pt_gen, form.ExtractionResult
        ),
        Ct.PPTX_EXTRACTION: ChainComponents(
            pt.PPTXExtraction.get_pt_gen, form.ExtractionResult
        ),
        Ct.CLUSTER_SYNTH: ChainComponents(
            pt.ClusterSynthesis.get_cpt(), form.ClusterSynthesisResult
        ),
        Ct.ROOT_CLUSTER_SYNTH: ChainComponents(
            pt.ClusterSynthesis.get_cpt(), form.RootClusterSynthesisResult
        ),
        Ct.DEDUP: ChainComponents(
            pt.Deduplicator.get_cpt(), form.DeduplicationResult
        ),
        Ct.DEDUP_ROOT: ChainComponents(
            pt.RootDeduplicator.get_cpt(), form.RootDeduplicationResult
        ),
        Ct.PRETTIFIER: ChainComponents(
            pt.Prettifier.get_cpt(), form.PrettyAnswers
        ),
        Ct.CONCEPT_PRETTIFIER: ChainComponents(
            pt.FullPrettifier.get_cpt(), form.PrettyConcepts
        ),
        Ct.NEW_ROOT: ChainComponents(
            pt.AddRootDeduplicator.get_cpt(), form.ResultRoot
        ),
    }

    @classmethod
    def _choose_chat_model(
        cls, model_name: str | None = None
    ) -> AzureChatOpenAI:
        """
        Selects a chat model based on the provided model name.

        Args:
            model_name: Optional model name to select a specific model.
                If None, uses the default model.

        Returns:
            An instance of AzureChatOpenAI model.
        """
        if model_name is None:
            return cls.default_chat_model
        return model_handler.ModelCreator.get_chat_model(model_name)

    @classmethod
    def choose_chain(
        cls, chain_type: Ct, model_name: str | None = None
    ) -> RunnableSerializable:
        """
        Creates and configures a chain of a specific type.

        Args:
            chain_type: Type of chain to create from ChainTypes enum.
            model_name: Optional chat model name; uses default if None.

        Returns:
            A configured chain ready to be executed.

        Raises:
            KeyError: If the specified chain_type
                does not exist in the mapping.
            ValueError: If chain components are invalid or incompatible.
        """
        try:
            chat_model = cls._choose_chat_model(model_name)
            chain_components = cls.CHAIN_TYPE_TO_COMPONENTS[chain_type]

            prompt_gen = chain_components.prompt_gen
            output_format = chain_components.response_format

            if output_format is None:
                return prompt_gen | chat_model

            form_chat_model = chat_model.with_structured_output(output_format)
            return prompt_gen | form_chat_model
        except KeyError:
            raise KeyError(f"Unknown chain type: {chain_type}")
        except Exception as e:
            msg = f"Failed to create chain of type {chain_type}: {str(e)}"
            raise ValueError(msg)


class ChainRunner:
    """
    Executes LLM chains with cost monitoring and progress tracking.
    Provides methods for running chains on single inputs or batches.
    """

    def __init__(self):
        """
        Initialize the chain runner with a lock
        for thread safety and a task counter.
        """
        self.lock = aio.Lock()
        self.task_counter = 0

    async def run_chain(
        self, chain: RunnableSerializable, input_data: dict[str, Any]
    ) -> Any:
        with get_openai_callback() as cb_handler:
            result = await chain.ainvoke(input_data)
            async with self.lock:
                llm_cost_handler = cur_llm_cost_handler.get()
                llm_cost_handler.update_costs(cb_handler)
            return result

    async def run_chain_on_batch(
        self, chain: RunnableSerializable, inputs: list[dict]
    ) -> list[Any]:
        """
        Run a chain on multiple inputs asynchronously.

        Args:
            chain: The chain to execute.
            inputs: List of input dictionaries.

        Returns:
            List of results corresponding to each input.
        """
        tasks = [self.run_chain(chain, input_data) for input_data in inputs]
        return await aio.gather(*tasks)

    async def _increment_and_update_status(
        self, num_total: int, status_update_func: Callable, **status_args
    ) -> None:
        """
        Increment the task counter and call the status update function.

        Args:
            num_total: Total number of tasks.
            status_update_func: Function to call for status updates.
            status_args: Additional arguments for the status function.
        """
        async with self.lock:
            self.task_counter += 1
            await status_update_func(
                self.task_counter, num_total, **status_args
            )

    async def run_chains_w_status_updates(
        self,
        chains_w_inputs: list[
            tuple[RunnableSerializable, list[dict[str, Any]]]
        ],
        status_update_func: callable,
        **status_args,
    ) -> list[Any]:
        """
        Run multiple chains with multiple inputs,
        providing status updates.

        Args:
            chains_w_inputs: List of tuples
                containing (chain, list_of_inputs).
            status_update_func: Callback function for status updates.
            status_args: Additional arguments
                passed to the status update function.

        Returns:
            List of results from all chain executions.
        """
        tasks = []
        total_tasks = sum(len(inputs) for _, inputs in chains_w_inputs)

        for chain, inputs in chains_w_inputs:
            for input_data in inputs:
                chain_task = self.run_chain(chain, input_data)

                async def execute_w_status(cur_chain_task: Awaitable[T]) -> T:
                    try:
                        result = await cur_chain_task
                        await self._increment_and_update_status(
                            total_tasks, status_update_func, **status_args
                        )
                        return result

                    except Exception as e:
                        await self._increment_and_update_status(
                            total_tasks, status_update_func, **status_args
                        )
                        raise e

                task = aio.create_task(execute_w_status(chain_task))
                tasks.append(task)

        results = await aio.gather(*tasks)

        self.task_counter = 0  # Reset instance counter

        return results
