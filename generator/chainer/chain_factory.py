"""
Factory module for creating and running language model chains.
Provides utilities for creating specialized chain types
and executing them with progress monitoring.
"""

import asyncio as aio
import logging
from typing import Any, Callable

from langchain_community.callbacks import get_openai_callback
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from openai import LengthFinishReasonError, RateLimitError
from pydantic import ValidationError

from common_utils.logger_config import logger
from generator.common.context import cur_llm_cost_handler
from generator.common.exceptions import GeneratorException

from . import model_handler
from . import prompt_generator as pt
from . import response_formats as form
from .structs import ChainComponents, ProgressTracker
from .utils.callbacks import PromptLoggerCallback
from .utils.constants import FIELD_TO_ERRORS
from .utils.constants import ChainTypes as Ct
from .utils.constants import FieldNames as Fn
from .utils.constants import FieldValidation as Fv
from .utils.response_validation import is_field_invalid


class ChainCreator:
    """
    Factory class for creating different types of language model chains.
    Each chain is composed of a prompt generator
    and a structured output format.
    """

    default_chat_model = model_handler.ModelCreator.get_chat_model()

    _CHAIN_TYPE_TO_COMPONENTS = {
        Ct.TEXTUAL_EXTRACTION: ChainComponents(
            pt.TextualExtraction.get_pt_gen, form.ExtractionResult
        ),
        Ct.PDF_EXTRACTION: ChainComponents(
            pt.PDFExtraction.get_pt_gen, form.KnowledgeFragment
        ),
        Ct.PPTX_EXTRACTION: ChainComponents(
            pt.PPTXExtraction.get_pt_gen, form.KnowledgeFragment
        ),
        Ct.CLUSTER_SYNTH: ChainComponents(
            pt.ClusterSynthesis.get_pt_gen, form.ClusterSynthesisResult
        ),
        Ct.ROOT_CLUSTER_SYNTH: ChainComponents(
            pt.RootClusterSynthesis.get_pt_gen, form.RootClusterSynthesisResult
        ),
        Ct.DEDUP: ChainComponents(
            pt.Deduplicator.get_cpt(), form.DeduplicationResult
        ),
        Ct.DEDUP_ROOT: ChainComponents(
            pt.RootDeduplicator.get_cpt(), form.RootDeduplicationResult
        ),
        Ct.CONCEPT_PRETTIFIER: ChainComponents(
            pt.Prettifier.get_pt_gen, form.PrettyConcepts
        ),
        Ct.APPLY_CONCEPT_PRETTIFIER: ChainComponents(
            pt.ApplyPrettifier.get_pt_gen, form.PrettyConcepts
        ),
        Ct.NEW_ROOT: ChainComponents(
            pt.AddRootDeduplicator.get_cpt(), form.ResultRoot
        ),
        Ct.REFINE_FILTER: ChainComponents(
            pt.RefineFilter.get_pt_gen, form.RefinedFilterResult
        ),
        Ct.VALIDATE_FILTER: ChainComponents(
            pt.ValidateFilter.get_pt_gen, form.ValidationResult
        ),
        Ct.REFINE_STYLE: ChainComponents(
            pt.StyleDirective.get_pt_gen, form.RefinedStyleResult
        ),
        Ct.VALIDATE_STYLE: ChainComponents(
            pt.ValidateStyle.get_pt_gen, form.ValidationResult
        ),
        Ct.DEDUPLICATE_NAMES: ChainComponents(
            pt.RenameConcepts.get_pt_gen, form.RenamingResult
        ),
    }

    def __init__(self, model_name: str | None = None):
        """
        Initializes the ChainCreator with a specific chat model.

        Args:
            model_name: The name of the model to use. If None, the
                class-level default model is used.
        """
        if model_name is None:
            self.chat_model = self.default_chat_model
        else:
            self.chat_model = model_handler.ModelCreator.get_chat_model(
                model_name
            )

    # noinspection PyUnreachableCode
    @classmethod
    def register_chain_type(
        cls, chain_type: Ct, components: ChainComponents
    ) -> None:
        """
        Dynamically registers a new chain type or overwrites an
        existing one.

        Args:
            chain_type: The enum member from ChainTypes for the new
                chain.
            components: A ChainComponents instance for the new chain.
        """
        if not isinstance(chain_type, Ct):
            raise TypeError(
                "chain_type must be a member of the ChainTypes enum."
            )
        if not isinstance(components, ChainComponents):
            raise TypeError(
                "components must be an instance of ChainComponents."
            )
        cls._CHAIN_TYPE_TO_COMPONENTS[chain_type] = components

    def choose_chain(self, chain_type: Ct) -> RunnableSerializable:
        """
        Creates and configures a chain of a specific type using the
        model this creator was initialized with.

        Args:
            chain_type: Type of chain to create from ChainTypes enum.

        Returns:
            A configured chain ready to be executed.
        """
        try:
            # Uses the model configured during instantiation
            chat_model = self.chat_model
            chain_components = self._CHAIN_TYPE_TO_COMPONENTS[chain_type]

            prompt_gen = chain_components.prompt_gen
            output_format = chain_components.response_format

            if output_format is None:
                return prompt_gen | chat_model

            form_chat_model = chat_model.with_structured_output(output_format)
            return prompt_gen | form_chat_model
        except KeyError as e:
            raise KeyError(f"Unknown chain type: {chain_type}") from e
        except Exception as e:
            msg = f"Failed to create chain of type {chain_type}: {str(e)}"
            raise ValueError(msg) from e


class ChainRunner:
    """
    Executes LLM chains with cost monitoring and progress tracking.
    Provides methods for running chains on single inputs or batches.
    """

    def __init__(self):
        """
        Initialize the chain runner with a lock for thread-safe cost
        updates.
        """
        self.cost_update_lock = aio.Lock()

    async def _run_chain(
        self,
        chain: RunnableSerializable,
        input_data: dict[str, Any],
        use_cache: bool = True,
        callbacks: list[AsyncCallbackHandler] | None = None,
    ) -> Any:
        """
        Invokes the chain with or without caching and with custom
        callbacks.
        """
        config: RunnableConfig = {"callbacks": callbacks or []}

        if not use_cache:
            # To disable caching, we pass a config that turns it off.
            config["configurable"] = {"llm_cache": None}

        with get_openai_callback() as cb_handler:
            # Add the OpenAI cost callback handler to the list
            config["callbacks"].append(cb_handler)

            result = await chain.ainvoke(input_data, config=config)

            async with self.cost_update_lock:
                llm_cost_handler = cur_llm_cost_handler.get()
                llm_cost_handler.update_costs(cb_handler)
            return result

    async def run_chain_w_retries(
        self,
        chain: RunnableSerializable,
        input_data: dict[str, Any],
        max_retries: int = 1,
        retry_field_name: str | None = None,
    ) -> Any:
        """
        Runs a chain and implements a retry mechanism for validation
        errors.

        Args:
            chain: The runnable chain to execute.
            input_data: The dictionary of input data for the chain.
            max_retries: The maximum number of retry attempts.
            retry_field_name: If specified, retries will be triggered on
                Pydantic ValidationErrors related to this field.
        """
        last_known_malformed_result = None
        # Make a copy of the input data to avoid modifying the original
        # dict in case of retries.
        current_input = input_data.copy()

        for attempt in range(max_retries + 1):
            prompt_logger = PromptLoggerCallback()
            try:
                use_cache = attempt == 0
                return await self._run_chain(
                    chain, current_input, use_cache, callbacks=[prompt_logger]
                )

            except ValidationError as e:
                logger.warning(f"Validation error on attempt {attempt + 1}. ")
                # log.debug(
                #     "LLM prompt that caused the error: "
                #     f"{prompt_logger.get_last_prompt()}"
                # )

                last_known_malformed_result = {
                    ".".join(map(str, err["loc"])): err["input"]
                    for err in e.errors()
                }

                if attempt < max_retries and retry_field_name:
                    field_error_config = FIELD_TO_ERRORS.get(retry_field_name)
                    if not field_error_config:
                        logger.warning(
                            "No retry configuration for field "
                            f"'{retry_field_name}'."
                        )
                        break

                    if is_field_invalid(
                        error=e,
                        field_name=retry_field_name,
                        error_types=field_error_config.get(Fv.ERROR_TYPES),
                    ):
                        logger.warning(
                            f"Attempt {attempt + 1} failed on validation for "
                            f"'{retry_field_name}'. Retrying without cache."
                        )

                        if not isinstance(current_input.get("messages"), list):
                            logger.warning(
                                "Cannot retry: 'messages' key not found in "
                                "input_data or is not a list. Skipping retries."
                            )
                            break

                        description = field_error_config.get(
                            Fv.DESCRIPTION, "valid format"
                        )
                        error_message = (
                            "Your previous response was invalid because the "
                            f"'{retry_field_name}' field was missing or "
                            "malformed. This field is mandatory and must be a "
                            f"{description}. Please review your instructions "
                            "and regenerate the entire response, ensuring "
                            "every part of the answer includes the "
                            f"'{retry_field_name}' field correctly."
                        )

                        # Modify the copy of the input for the next attempt
                        current_input["messages"] = list(
                            current_input["messages"]
                        ) + [HumanMessage(content=error_message)]
                        continue

                logger.warning(
                    "Chain execution failed Pydantic validation after all "
                    "retries. Returning the last-known malformed result."
                )
                break

            except RateLimitError:
                logger.error("Rate limit error encountered.")
                # log.debug(
                #     "LLM prompt that caused the error: "
                #     f"{prompt_logger.get_last_prompt()}"
                # )
                raise

            except LengthFinishReasonError as e:

                logger.warning(
                    f"Attempt {attempt + 1} failed because the model's output "
                    "was truncated (hit the token limit)."
                )
                # log.warning(
                #     "LLM prompt that caused the error: "
                #     f"{prompt_logger.get_last_prompt()}"
                # )

                if e.completion and e.completion.choices:
                    # partial_completion = e.completion.choices[0].message.content
                    # Log the partial output to see what the model was trying to generate.
                    # log.warning(
                    #     "Partial completion received before truncation:\n"
                    #     f"--- START ---\n{partial_completion}\n--- END ---"
                    # )

                    usage_info = e.completion.usage
                    logger.warning(f"Token usage for this attempt: {usage_info}")
                else:
                    logger.warning(
                        "No partial completion or usage info was available in the error."
                    )

                if attempt < max_retries:
                    logger.warning(
                        "Retrying with an instruction to be more concise."
                    )

                    if not isinstance(current_input.get("messages"), list):
                        logger.error(
                            "Cannot retry for length: 'messages' key not found in "
                            "input_data or is not a list. Failing."
                        )
                        break

                    error_message = (
                        "Your previous response was too long and was cut off. "
                        "Please regenerate the entire response, but be more "
                        "concise."
                    )

                    current_input["messages"] = list(
                        current_input["messages"]
                    ) + [HumanMessage(content=error_message)]

                    continue

            except Exception as e:
                logger.error("An error occurred during chain execution: " f"{e}.")
                # log.debug(
                #     "LLM prompt that caused the error: "
                #     f"{prompt_logger.get_last_prompt()}"
                # )
                raise GeneratorException(
                    f"An unexpected error occurred during chain execution: {e}"
                ) from e

        return last_known_malformed_result

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
        tasks = [
            self.run_chain_w_retries(chain, input_data) for input_data in inputs
        ]
        return await aio.gather(*tasks)

    async def run_chains_w_status_updates(
        self,
        chains_w_inputs: list[
            tuple[RunnableSerializable, list[dict[str, Any]]]
        ],
        status_update_func: Callable,
        **status_args,
    ) -> list[Any]:
        """
        Run multiple chains with multiple inputs, providing status
        updates.

        Args:
            chains_w_inputs: List of tuples containing
                (chain, list_of_inputs).
            status_update_func: Callback function for status updates.
            status_args: Additional arguments for the status update
                function.

        Returns:
            List of results from all chain executions.
        """
        all_inputs = [
            (chain, input_data)
            for chain, inputs in chains_w_inputs
            for input_data in inputs
        ]
        total_tasks = len(all_inputs)

        if total_tasks == 0:
            return []

        progress = ProgressTracker(
            total=total_tasks,
            status_update_func=status_update_func,
            status_args=status_args,
        )

        async def execute_w_status(
            chain: RunnableSerializable, input_data: dict[str, Any]
        ) -> Any:
            """
            Wrapper to run a chain and update progress upon completion.
            """
            try:
                return await self.run_chain_w_retries(
                    chain, input_data, retry_field_name=Fn.SOURCE_IDS
                )
            finally:
                await progress.increment()

        tasks = [
            aio.create_task(execute_w_status(chain, input_data))
            for chain, input_data in all_inputs
        ]

        return await aio.gather(*tasks)
