import asyncio
import json
import re
import time
from collections import Counter, defaultdict, deque
from typing import Any, AsyncGenerator, Dict, List, Tuple, TypeAlias
from uuid import uuid4

from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from common_utils.logger_config import logger
from general_mindmap.v2.dial.client import DialClient
from general_mindmap.v2.generator.constants import (
    OLD_PROMPT_MAP,
    QA_PROMPT_MAP,
    STRUCTURE_PROMPT_MAP,
)
from general_mindmap.v2.generator.exceptions import AnswerCountMismatchError
from general_mindmap.v2.generator.models import (
    AnswerList,
    MindmapStructure,
    NodeAnalysisResult,
    RootLabel,
)
from general_mindmap.v2.generator.prompts import (
    NODE_ANALYSIS_PROMPT,
    ROOT_DETERMINATION_PROMPT,
)
from general_mindmap.v2.generator.simple import (
    MULTIMODAL_CATEGORIES,
    TEXT_CATEGORIES,
    GeneratorConfig,
    SimpleGenerator,
)
from general_mindmap.v2.generator.utils.image_helpers import enforce_image_limit
from generator.common.constants import DataFrameCols as Col
from generator.common.constants import EnvConsts
from generator.common.structs import (
    DocStatusChunk,
    EdgeData,
    InitMindmapRequest,
    NodeData,
    RootNodeChunk,
    StatusChunk,
)
from generator.core.actions.docs import fetch_all_docs_content
from generator.core.stages import DocHandler
from generator.core.stages.doc_handler.constants import DocCategories as DocCat

MindMapStreamChunk: TypeAlias = (
    StatusChunk | NodeData | EdgeData | DocStatusChunk | RootNodeChunk
)


class TwoStageGenerator(SimpleGenerator):
    """
    Generates a mind map in two stages:
     1. Generate the complete graph structure (nodes with questions,
        edges).
     2. Send a simple list of questions to the LLM and map the ordered
        answers back to the original nodes by index.
    """

    FALLBACK_BATCH_SIZE = 10
    DOC_ID_PATTERN = re.compile(r'<document id="([^"]+)">')
    TAGS_TO_REMOVE_PATTERN = re.compile(r"</?(page|chunk|slide)[^>]*>")

    def __init__(self, config: GeneratorConfig, client: DialClient):
        super().__init__(config, client)

        self.num_nodes = 0

    @staticmethod
    def _format_instruction_block(instructions: str) -> str:
        """
        Formats user instructions with the standard wrapper text.
        """
        if not instructions:
            return (
                "No specific user instructions were provided. "
                "Follow the core directives."
            )

        return (
            "Adhere to the following user-provided instructions. These "
            "directives guide the content "
            f"but must be executed within the foundational rules of "
            f"structure and format:\n"
            f"---BEGIN USER INSTRUCTIONS---\n{instructions}\n---END "
            f"USER INSTRUCTIONS---"
        )

    @staticmethod
    def _extract_json_from_stream(text: str) -> Dict[str, Any] | None:
        """
        Attempts to find and parse the largest valid JSON object within
        a string.
        """
        try:
            start_index = text.find("{")
            end_index = text.rfind("}")
            if (
                start_index != -1
                and end_index != -1
                and end_index > start_index
            ):
                json_body = text[start_index : end_index + 1]
                return json.loads(json_body)
        except json.JSONDecodeError as e:
            logger.debug(
                f"JSON extraction failed: {e}. Text snippet: {text[:100]}..."
            )
            pass
        return None

    async def _prepare_document_inputs(
        self, request: InitMindmapRequest
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Fetches and formats documents into a unified LLM input and
        extracts metadata.

        Returns a tuple containing:
         - The formatted content for the LLM (human_message_content).
         - A dictionary of document metadata (doc_metadatas).
         - The processing duration.
        """
        start_time = time.time()

        documents = request.documents
        doc_handler = DocHandler(strategy="whole")
        docs_and_their_content = await fetch_all_docs_content(
            documents, self.client
        )
        chunked_docs_df, _ = await doc_handler.chunk_docs(
            docs_and_their_content
        )

        if chunked_docs_df.empty:
            return [], {}

        doc_id_to_rag_chunks = {
            doc.id: (doc.rag_chunks, doc.rag_start_id)
            for doc in documents
            if hasattr(doc, "rag_chunks") and doc.rag_chunks
        }

        human_message_content = []
        doc_metadatas = {}
        processed_doc_ids = set()

        rows = chunked_docs_df.to_dict("records")

        for row in rows:
            doc_id = row[Col.DOC_ID]
            if doc_id in processed_doc_ids:
                continue

            processed_doc_ids.add(doc_id)
            doc_cat = row[Col.DOC_CAT]
            content_from_df = row[Col.CONTENT]

            doc_metadatas[doc_id] = {"file_type": doc_cat}

            if doc_cat == DocCat.UNSUPPORTED:
                logger.warning(
                    f"Skipping unsupported document with ID: {doc_id}"
                )
                continue

            human_message_content.append(
                {"type": "text", "text": f'<document id="{doc_id}">'}
            )

            if doc_cat in TEXT_CATEGORIES:
                rag_info = doc_id_to_rag_chunks.get(doc_id)
                if rag_info:
                    rag_chunks, start_id = rag_info
                    for chunk in rag_chunks:
                        chunk_text = self._format_chunk_text_with_links(chunk)

                        raw_chunk_id = chunk.metadata.get("chunk_id")
                        safe_chunk_id = (
                            int(raw_chunk_id) if raw_chunk_id is not None else 0
                        )

                        chunk_id = str(safe_chunk_id + start_id + 1)

                        chunk_content = self._format_chunk_content(
                            texts=[chunk_text], images=[], chunk_id=chunk_id
                        )
                        human_message_content.extend(chunk_content)
                elif content_from_df:
                    chunk_content = self._format_chunk_content(
                        texts=[content_from_df], images=[], chunk_id=1
                    )
                    human_message_content.extend(chunk_content)

            elif doc_cat in MULTIMODAL_CATEGORIES:
                if isinstance(content_from_df, list):
                    for page in content_from_df:
                        page_content = self._format_chunk_content(
                            texts=page.get("texts", []),
                            images=page.get("images", []),
                            chunk_id=page.get("page_id"),
                        )
                        if page_content:
                            human_message_content.extend(page_content)

            human_message_content.append(
                {"type": "text", "text": "</document>"}
            )

        final_content = enforce_image_limit(human_message_content)

        duration = time.time() - start_time
        logger.info(f"Document processing took {duration:.2f} seconds.")

        return final_content, doc_metadatas

    @staticmethod
    def _chunk_list(data: List[Any], size: int) -> List[List[Any]]:
        """Splits a list into chunks of a specified size."""
        if size <= 0:
            raise ValueError("Chunk size must be positive.")
        return [data[i : i + size] for i in range(0, len(data), size)]

    @classmethod
    def _create_simplified_input_for_structure(
        cls,
        human_message_content: List[Dict[str, Any]],
        doc_metadatas: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Creates a simplified input for Stage 1 (structure generation).

        We don't need to know sources at this stage:
         - Keeps <document> and </document> tags.
         - Removes <page>, <chunk>, and <slide> tags.
         - For PDF and PPTX documents, keeps image content.
         - For other documents, removes image content.
        """
        logger.info(
            "Creating simplified input for Stage 1 structure generation."
        )

        simplified_input = []
        current_doc_id = None
        current_doc_type = ""

        for item in human_message_content:
            item_type = item.get("type")

            if item_type == "text":
                text = item.get("text", "")

                # Clean the text by removing page/chunk/slide tags
                cleaned_text = cls.TAGS_TO_REMOVE_PATTERN.sub("", text)
                if not cleaned_text.strip():
                    continue

                # Check for doc ID
                doc_id_match = cls.DOC_ID_PATTERN.search(cleaned_text)
                if doc_id_match:
                    current_doc_id = doc_id_match.group(1)
                    doc_info = doc_metadatas.get(current_doc_id, {})
                    current_doc_type = doc_info.get("file_type", "").lower()
                    logger.debug(
                        f"Entering document {current_doc_id} of type {current_doc_type}"
                    )

                # Append the cleaned text item
                simplified_input.append({"type": "text", "text": cleaned_text})

                # Check if this item closes the document
                if "</document>" in cleaned_text:
                    logger.debug(f"Exiting document {current_doc_id}")
                    current_doc_id = None
                    current_doc_type = ""

            elif item_type == "image_url":
                if current_doc_type in ["pdf_as_a_whole", "pptx_as_a_whole"]:
                    logger.debug(
                        f"Including image from document {current_doc_id} (type: {current_doc_type})"
                    )
                    simplified_input.append(item)
                else:
                    logger.debug(
                        f"Skipping image from document {current_doc_id} (type: {current_doc_type})"
                    )

        return simplified_input

    def _get_prompt_template(
        self,
        prompt_map: Dict[str, str],
        use_backwards_compatible: bool = False,
    ) -> str:
        """
        Selects the best prompt template based on the configured model name.
        If use_backwards_compatible is True, it returns the appropriate old prompt.
        """
        if use_backwards_compatible:
            logger.info(
                "Using backwards-compatible old prompt for A/B testing."
            )
            if prompt_map is STRUCTURE_PROMPT_MAP:
                return OLD_PROMPT_MAP["structure"]
            else:
                return OLD_PROMPT_MAP["qa"]

        model_name_lower = self.config.model.lower()
        for key, prompt in prompt_map.items():
            if key != "default" and model_name_lower.startswith(key):
                logger.info(f"Using model-specific prompt for: {key}")
                return prompt

        logger.info("Using universal/default prompt.")
        return prompt_map["default"]

    @staticmethod
    def _validate_graph_structure(graph: dict) -> tuple[bool, str]:
        """
        Validates the raw dictionary and returns ALL found errors.
        Returns (True, "") if valid, or (False, combined_error_string).
        """
        errors = []
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        root_label = graph.get("root")

        if not nodes:
            return False, "The graph contains no nodes."

        labels = [n.get("label") for n in nodes if n.get("label")]
        counts = Counter(labels)
        duplicates = [k for k, v in counts.items() if v > 1]

        if duplicates:
            errors.append(
                f"Duplicate node labels found: {duplicates}. "
                f"All labels must be unique."
            )

        valid_labels = set(labels)

        if not root_label:
            errors.append("The 'root' field is missing or empty.")
        elif root_label not in valid_labels:
            errors.append(
                f"Root label '{root_label}' is not found in the node list."
            )

        adjacency = defaultdict(list)
        dangling_sources = set()
        dangling_targets = set()

        for edge in edges:
            src = edge.get("source")
            tgt = edge.get("target")

            src_exists = src in valid_labels
            tgt_exists = tgt in valid_labels

            if not src_exists:
                dangling_sources.add(src)
            if not tgt_exists:
                dangling_targets.add(tgt)

            if src_exists and tgt_exists:
                adjacency[src].append(tgt)
                adjacency[tgt].append(src)

        if dangling_sources:
            errors.append(
                f"Edges reference non-existent source nodes: "
                f"{list(dangling_sources)}."
            )
        if dangling_targets:
            errors.append(
                f"Edges reference non-existent target nodes: "
                f"{list(dangling_targets)}."
            )

        visited_global = set()
        connected_components = 0
        cycle_detected = False

        sorted_nodes = sorted(list(valid_labels))

        for start_node in sorted_nodes:
            if start_node in visited_global:
                continue

            connected_components += 1

            queue = deque([(start_node, None)])
            visited_in_this_component = {start_node}
            visited_global.add(start_node)

            while queue:
                curr, parent = queue.popleft()

                for neighbor in adjacency[curr]:
                    if neighbor == parent:
                        continue

                    if neighbor in visited_in_this_component:
                        if not cycle_detected:
                            errors.append(
                                f"Cycle detected involving '{curr}' and "
                                f"'{neighbor}'. Trees must be acyclic."
                            )
                            cycle_detected = True

                    elif neighbor not in visited_global:
                        visited_in_this_component.add(neighbor)
                        visited_global.add(neighbor)
                        queue.append((neighbor, curr))

        if connected_components > 1:
            errors.append(
                f"Graph is disconnected. It consists of {connected_components} "
                f"separate islands. Ensure all nodes connect back to the root."
            )

        if errors:
            if len(errors) == 1:
                return False, errors[0]
            error_summary = "Multiple issues found:\n" + "\n".join(
                [f"- {e}" for e in errors]
            )
            return False, error_summary

        return True, ""

    async def _generate_structure(
        self,
        human_message_content: List[Dict[str, Any]],
        user_instructions: str,
        use_backwards_compatible_prompts: bool = False,
    ) -> AsyncGenerator[StatusChunk | dict[str, Any], None]:
        """
        Stage 1: Generates the mind map's structure
        with self-correction.

        This method is an asynchronous generator that yields
        StatusChunk objects for progress updates and returns the final
        graph structure.
        """
        logger.info("Stage 1: Generating mind map structure")
        start_time = time.time()

        system_prompt_template = self._get_prompt_template(
            STRUCTURE_PROMPT_MAP,
            use_backwards_compatible=use_backwards_compatible_prompts,
        )
        instructions_text = self._format_instruction_block(user_instructions)
        final_system_prompt = system_prompt_template.format(
            user_instructions=instructions_text
        )

        messages: list[BaseMessage] = [
            SystemMessage(content=final_system_prompt),
            HumanMessage(content=human_message_content),
        ]

        structured_llm = self.llm.with_structured_output(MindmapStructure)

        max_retries = 1
        graph_structure = {}

        for attempt in range(max_retries + 1):
            raw_output = None
            content_chunks = []
            node_count = 0

            with get_openai_callback() as cb:
                if EnvConsts.IS_STREAM_SIMPLE_GEN:
                    event_stream = structured_llm.astream_events(messages)

                    async for event in event_stream:
                        if event["event"] == "on_chat_model_stream":
                            chunk = event["data"].get("chunk")
                            if (
                                isinstance(chunk, AIMessageChunk)
                                and chunk.content
                            ):
                                content_chunks.append(chunk.content)
                                current_json_str = "".join(content_chunks)
                                current_node_count = current_json_str.count(
                                    '"question":'
                                )
                                if current_node_count > node_count:
                                    node_count = current_node_count
                                    if attempt > 0:
                                        status_msg = (
                                            f"Fixing structure: adding nodes "
                                            f"({node_count} nodes built)..."
                                        )
                                    else:
                                        status_msg = (
                                            f"Adding nodes ({node_count} nodes "
                                            f"built)..."
                                        )
                                    yield StatusChunk(title=status_msg)
                else:
                    raw_output = await structured_llm.ainvoke(messages)

                self._log_llm_usage(cb)

            graph_structure = {}

            if raw_output and isinstance(raw_output, MindmapStructure):
                graph_structure = raw_output.model_dump()
            elif content_chunks:
                full_output_str = "".join(content_chunks)
                extracted_json = self._extract_json_from_stream(full_output_str)
                if extracted_json:
                    graph_structure = extracted_json
                else:
                    logger.warning(
                        f"Failed to decode JSON stream: "
                        f"{full_output_str[:50]}..."
                    )
            else:
                logger.warning(
                    f"Expected MindmapStructure, got {type(raw_output)}"
                )

            is_valid, error_msg = self._validate_graph_structure(
                graph_structure
            )

            if is_valid:
                break

            logger.debug(
                f"Validation failed (Attempt {attempt + 1}): {error_msg}"
            )

            if attempt == max_retries:
                logger.debug(
                    "Max retries reached. Returning best-effort structure."
                )
                break

            if content_chunks:
                previous_response_str = "".join(content_chunks)
            else:
                previous_response_str = json.dumps(graph_structure)

            messages.append(AIMessage(content=previous_response_str))

            correction_msg = (
                f"The JSON structure generated has errors.\n"
                f"ERROR DETAIL: {error_msg}\n"
                "Please REGENERATE the entire structure. "
                "Ensure all node labels are unique, every edge connects "
                "existing nodes, and the graph is a single connected tree "
                "(no cycles, no detached parts)."
            )
            messages.append(HumanMessage(content=correction_msg))

        duration = time.time() - start_time
        final_node_count = (
            len(graph_structure.get("nodes", []))
            if isinstance(graph_structure, dict)
            else 0
        )
        logger.info(
            f"Stage 1: Completed in {duration:.2f}s with {final_node_count} "
            f"nodes."
        )

        yield graph_structure

    async def _fetch_answers_for_question_list(
        self,
        human_message_content: List[Dict[str, Any]],
        questions: List[str],
        user_instructions: str,
    ) -> AsyncGenerator[StatusChunk | List[Dict[str, Any]], None]:
        """
        Stage 2 (Robust): Fetches answers, yielding progress, with a cascading
        fallback strategy. The final yielded chunk contains the complete answer list.
        """
        logger.info(f"Stage 2: Fetching answers for {len(questions)} questions")
        start_time = time.time()
        max_retries = 2
        answers = None

        try:
            # Attempt 1: Process the entire batch with retries
            full_batch_stream = self._execute_llm_batch_request(
                human_message_content,
                questions,
                user_instructions,
                max_retries=max_retries,
            )

            async for chunk in full_batch_stream:
                if isinstance(chunk, StatusChunk):
                    yield chunk
                else:
                    answers = chunk

            if answers is not None:
                duration = time.time() - start_time
                logger.info(
                    "Stage 2: Successfully fetched all answers in a single batch "
                    f"in {duration:.2f}s."
                )
                yield answers
                return

            raise ValueError(
                "Batch processing stream completed without a final answer list."
            )

        except Exception as e:
            logger.warning(
                f"Stage 2: Full batch processing failed after {max_retries} "
                f"attempts ({e}). "
                "Switching to fallback strategy with batch size "
                f"{self.FALLBACK_BATCH_SIZE}."
            )
            # Attempt 2: Cascading fallback
            questions_with_indices = list(enumerate(questions))
            fallback_stream = self._handle_cascading_fallback(
                human_message_content, questions_with_indices, user_instructions
            )

            final_answers = None
            async for chunk in fallback_stream:
                if isinstance(chunk, StatusChunk):
                    yield chunk
                else:
                    final_answers = chunk  # Capture the final re-assembled list

            duration = time.time() - start_time
            logger.info(
                f"Stage 2: Answer fetching completed via fallback in {duration:.2f}s."
            )
            if final_answers is not None:
                yield final_answers

    async def _execute_llm_batch_request(
        self,
        human_message_content: List[Dict[str, Any]],
        questions: List[str],
        user_instructions: str,
        max_retries: int = 0,
        use_backwards_compatible_prompts: bool = False,
    ) -> AsyncGenerator[StatusChunk | List[Dict[str, Any]], None]:
        """
        Executes a single LLM request for a batch of questions, yielding progress.
        """
        if not questions:
            yield []
            return

        num_questions = len(questions)

        system_prompt_template = self._get_prompt_template(
            QA_PROMPT_MAP, use_backwards_compatible_prompts
        )

        instructions_text = self._format_instruction_block(user_instructions)

        final_system_prompt = system_prompt_template.format(
            user_instructions=instructions_text, num_questions=num_questions
        )

        content_with_questions = [
            {
                "type": "text",
                "text": "Here is the JSON list of questions to answer:",
            },
            {"type": "text", "text": f"```json\n{json.dumps(questions)}\n```"},
            {"type": "text", "text": "\nHere are the source documents:"},
            *human_message_content,
        ]

        structured_llm = self.llm.with_structured_output(AnswerList)

        llm_input = [
            ("system", final_system_prompt),
            ("human", content_with_questions),
        ]

        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                raw_output = None
                content_chunks = []
                answer_count = 0
                response_json = {}

                with get_openai_callback() as cb:
                    if EnvConsts.IS_STREAM_SIMPLE_GEN:
                        event_stream = structured_llm.astream_events(llm_input)

                        async for event in event_stream:
                            if event["event"] == "on_chat_model_stream":
                                chunk = event["data"].get("chunk")
                                chunk_text = ""

                                if isinstance(
                                    chunk, AIMessageChunk
                                ) and isinstance(chunk.content, str):
                                    chunk_text = chunk.content

                                if chunk_text:
                                    content_chunks.append(chunk_text)
                                    current_json_str = "".join(content_chunks)
                                    current_answer_count = (
                                        current_json_str.count(
                                            '"answer_parts":'
                                        )
                                    )

                                    if current_answer_count > answer_count:
                                        answer_count = current_answer_count
                                        yield StatusChunk(
                                            title=f"Generating content ({answer_count}/{self.num_nodes})..."
                                        )
                    else:
                        raw_output = await structured_llm.ainvoke(llm_input)
                    self._log_llm_usage(cb)

                if content_chunks:
                    full_output_str = "".join(content_chunks)
                    response_json = self._extract_json_from_stream(
                        full_output_str
                    )

                elif raw_output:
                    if isinstance(raw_output, AnswerList):
                        response_json = raw_output.model_dump()
                    elif (
                        isinstance(raw_output, AIMessage)
                        and raw_output.tool_calls
                    ):

                        # noinspection PyTypeHints
                        response_json = raw_output.tool_calls[0]["args"]
                    else:
                        raise ValueError(
                            f"LLM output was an unexpected type: {type(raw_output)}"
                        )

                if not response_json:
                    raise ValueError("LLM returned an empty response.")

                answers = response_json.get("answers", [])
                if len(answers) != num_questions:
                    raise AnswerCountMismatchError(
                        n_questions=num_questions, n_answers=len(answers)
                    )

                yield answers
                return  # Success, exit the retry loop

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"LLM batch request for {num_questions} questions failed "
                    f"on attempt {attempt + 1}/{max_retries + 1}: {e}"
                )
                if attempt < max_retries:
                    await asyncio.sleep(2 * (2**attempt))

        raise last_exception from None

    async def _handle_cascading_fallback(
        self,
        human_message_content: List[Dict[str, Any]],
        questions_with_indices: List[Tuple[int, str]],
        user_instructions: str,
    ) -> AsyncGenerator[StatusChunk | List[Dict[str, Any]], None]:
        """
        Handles fallback logic by yielding aggregate progress from mini-batches
        and one-by-one processing.
        """
        final_answers = {}
        total_questions_to_find = len(questions_with_indices)
        failed_in_batch = []

        # Level 1 Fallback: Mini-batches
        batches = self._chunk_list(
            questions_with_indices, self.FALLBACK_BATCH_SIZE
        )
        for i, batch in enumerate(batches):
            batch_questions = [q for _, q in batch]
            batch_indices = [idx for idx, _ in batch]

            try:
                batch_stream = self._execute_llm_batch_request(
                    human_message_content,
                    batch_questions,
                    user_instructions,
                    max_retries=1,
                )

                batch_answers = None
                async for chunk in batch_stream:
                    if isinstance(chunk, StatusChunk):
                        pass
                    else:
                        batch_answers = chunk

                if batch_answers and len(batch_answers) == len(batch_indices):
                    for j, answer in enumerate(batch_answers):
                        original_index = batch_indices[j]
                        final_answers[original_index] = answer
                else:
                    # If answers are missing or mismatched, mark entire batch for one-by-one fallback
                    failed_in_batch.extend(batch)

            except Exception as e:
                logger.warning(
                    f"Fallback batch {i + 1} failed: {e}. Marking questions for individual processing."
                )
                failed_in_batch.extend(batch)

            # Yield aggregate progress after each batch attempt
            yield StatusChunk(
                title=f"Generating answers ({len(final_answers)}/{self.num_nodes} questions)"
            )

        # Level 2 Fallback: One-by-one
        if failed_in_batch:
            logger.info(
                f"Processing {len(failed_in_batch)} remaining questions one-by-one."
            )
            for original_index, question_text in failed_in_batch:
                try:
                    single_q_stream = self._execute_llm_batch_request(
                        human_message_content,
                        [question_text],
                        user_instructions,
                        max_retries=1,
                    )

                    single_answer_list = None
                    async for chunk in single_q_stream:
                        if not isinstance(chunk, StatusChunk):
                            single_answer_list = chunk

                    if single_answer_list:
                        final_answers[original_index] = single_answer_list[0]

                except Exception as e:
                    logger.error(
                        f"Single question fallback failed for index {original_index}: {e}"
                    )

                # Update and yield overall progress after each single attempt
                yield StatusChunk(
                    title=f"Generating answers ({len(final_answers)}/{self.num_nodes} questions)"
                )

        # Reassemble and yield the final result
        full_answer_list = [
            final_answers.get(i) for i in range(total_questions_to_find)
        ]
        yield full_answer_list

    async def _analyze_nodes(
        self, generated_nodes: List[NodeData]
    ) -> NodeAnalysisResult:
        """
        Analyzes generated nodes for quality issues like duplicate
        content and low-substance answers using an LLM.

        Args:
            generated_nodes: A list of all populated NodeData objects.

        Returns:
            A NodeAnalysisResult object containing lists of problematic
            nodes.
        """
        logger.info(
            "Starting optional analysis for node quality on "
            f"{len(generated_nodes)} nodes."
        )
        start_time = time.time()

        if len(generated_nodes) < 2:
            logger.info(
                "Not enough nodes to perform quality analysis. Skipping."
            )
            return NodeAnalysisResult()

        nodes_for_analysis = [
            {
                "label": node.label,
                "question": node.question,
                "details": node.details,
            }
            for node in generated_nodes
        ]

        structured_llm = self.llm.with_structured_output(NodeAnalysisResult)
        system_prompt = NODE_ANALYSIS_PROMPT.split("# OUTPUT FORMAT")[0].strip()

        human_message = [
            {
                "type": "text",
                "text": "Analyze the following mind map nodes for duplicate and low-substance content:",
            },
            {
                "type": "text",
                "text": f"```json\n{json.dumps(nodes_for_analysis, indent=2)}\n```",
            },
        ]

        llm_input = [("system", system_prompt), ("human", human_message)]

        try:
            raw_output = None
            with get_openai_callback() as cb:
                raw_output = await structured_llm.ainvoke(llm_input)
                self._log_llm_usage(cb)

            if not raw_output:
                raise ValueError(
                    "LLM returned an empty response during analysis."
                )

            analysis_result = NodeAnalysisResult.model_validate(raw_output)

            duration = time.time() - start_time
            num_duplicates = len(analysis_result.duplicate_nodes)
            num_low_substance = len(analysis_result.low_substance_nodes)
            logger.info(
                f"Node analysis completed in {duration:.2f}s. "
                f"Found {num_duplicates} duplicate nodes and {num_low_substance} low-substance nodes."
            )
            return analysis_result

        except Exception as e:
            logger.error(f"An error occurred during node quality analysis: {e}")
            return NodeAnalysisResult()

    async def _determine_root_node(
        self,
        graph_structure: Dict[str, Any],
        user_instructions: str,
    ) -> str | None:
        """
        Uses an LLM to determine the most logical root node for a given
        graph structure.
        """
        logger.info("Attempting to determine a new root node via LLM call...")

        structured_llm = self.llm.with_structured_output(RootLabel)
        system_prompt = ROOT_DETERMINATION_PROMPT.split("# OUTPUT FORMAT")[
            0
        ].strip()

        human_message = [
            {
                "type": "text",
                "text": "The original user instructions were:\n"
                f'"""\n{user_instructions or "No instructions provided."}\n"""',
            },
            {
                "type": "text",
                "text": "\nHere is the graph structure that was generated. "
                "Please identify the correct root node from it:",
            },
            {
                "type": "text",
                "text": f"```json\n{json.dumps(graph_structure, indent=2)}\n```",
            },
        ]

        llm_input = [("system", system_prompt), ("human", human_message)]

        try:
            with get_openai_callback() as cb:
                raw_output = await structured_llm.ainvoke(llm_input)
                self._log_llm_usage(cb)

            if not raw_output:
                raise ValueError(
                    "LLM returned an empty response for root determination."
                )

            result = raw_output.model_dump()
            new_root = result.get("root_label")
            logger.debug(f"LLM selected '{new_root}' as the new root node.")
            return new_root

        except Exception as e:
            logger.error(f"Failed to determine new root node via LLM: {e}")
            return None

    async def generate(
        self,
        request: InitMindmapRequest,
        run_post_analysis: bool = False,
    ) -> AsyncGenerator[MindMapStreamChunk, None]:
        start_time = time.time()

        logger.info("Starting two-step mind map generation.")
        yield StatusChunk(title="Let’s create your mind map!")

        gen_prompt = self.config.gen_prompt
        rag_prompt = self.config.rag_prompt

        yield StatusChunk(title="Getting your sources ready...")
        full_inputs, doc_metadatas = await self._prepare_document_inputs(
            request
        )
        if not full_inputs:
            logger.warning("No processable inputs; aborting generation.")
            return

        # --- STAGE 1: GENERATE STRUCTURE ---
        yield StatusChunk(title="Outlining the main ideas...")
        simplified_human_message = self._create_simplified_input_for_structure(
            full_inputs, doc_metadatas
        )

        yield StatusChunk(title=f"Outlining the main ideas...")
        graph_structure = None

        structure_stream = self._generate_structure(
            simplified_human_message,
            user_instructions=gen_prompt,
        )
        async for status in structure_stream:
            if isinstance(status, dict):
                graph_structure = status
            else:
                yield status

        if (
            not isinstance(graph_structure, dict)
            or "nodes" not in graph_structure
        ):
            error_log_message = (
                "Invalid graph structure. Expected a dictionary "
                "with a 'nodes' key, "
                f"but received type {type(graph_structure).__name__}."
            )
            logger.error(error_log_message)

            if graph_structure is None:
                raise ValueError(
                    "Mind map structure generation failed to produce "
                    "a final result."
                )

            yield StatusChunk(
                title="Oops — couldn’t build the structure. Try again?"
            )
            return

        nodes = graph_structure.get("nodes", [])
        label_to_first_id = {}
        valid_nodes_with_ids = []

        self.num_nodes = len(nodes)
        for node in nodes:
            label = node.get("label")
            if not label:
                continue

            node_id = str(uuid4())

            node["_id"] = node_id
            valid_nodes_with_ids.append(node)

            if label not in label_to_first_id:
                label_to_first_id[label] = node_id

        # --- Validate and fix root node ---
        root_label = graph_structure.get("root")

        root_id = label_to_first_id.get(root_label)
        if not root_id and valid_nodes_with_ids:
            root_id = valid_nodes_with_ids[0]["_id"]

        if root_id:
            yield RootNodeChunk(root_id=root_id)

        # --- YIELD EDGES ---
        edges = graph_structure.get("edges", [])
        processed_pairs = set()

        for edge in edges:
            source_label = edge.get("source")
            target_label = edge.get("target")

            # Connect edges to the FIRST occurrence of that label
            source_id = label_to_first_id.get(source_label)
            target_id = label_to_first_id.get(target_label)

            if source_id and target_id and source_id != target_id:
                pair_key = tuple(sorted((source_id, target_id)))

                if pair_key in processed_pairs:
                    continue

                processed_pairs.add(pair_key)

                yield EdgeData(
                    id=str(uuid4()), source=source_id, target=target_id
                )
                yield EdgeData(
                    id=str(uuid4()), source=target_id, target=source_id
                )

        logger.info("Yielded initial graph skeleton (edges, root).")

        # --- STAGE 2: FETCH AND YIELD CONTENT ---
        yield StatusChunk(title="Generating content...")
        questions_to_ask = [node.get("question", "") for node in nodes]
        answers = None
        answer_stream = self._fetch_answers_for_question_list(
            full_inputs,
            questions_to_ask,
            user_instructions=rag_prompt,
        )
        async for status in answer_stream:
            if isinstance(status, list):
                answers = status
            else:
                yield status

        if not answers or len(answers) != len(nodes):
            logger.error(
                f"Failed to augment graph: Mismatch in question/answer count. "
                f"Sent {len(nodes)} questions, received {len(answers)} answers."
            )
            return

        logger.info(f"Received and processing {len(answers)} answers.")
        all_generated_nodes: List[NodeData] = []
        for node, answer_obj in zip(nodes, answers):
            node_id = node.get("_id")

            if not node_id:
                continue

            answer_parts = answer_obj.get("answer_parts", [])
            details_parts = []
            for part in answer_parts:
                statement = part.get("statement")
                doc_id = part.get("doc_id")
                chunk_id = part.get("chunk_id")

                if not statement:
                    continue

                citation = ""
                if doc_id:
                    if chunk_id is not None:
                        citation = f" ^[{doc_id}.{chunk_id}]^"
                    else:
                        citation = f" ^[{doc_id}]^"

                details_parts.append(statement + citation)

            details_text = " ".join(details_parts)

            node_data = NodeData(
                id=node_id,
                label=node["label"],
                question=node["question"],
                details=details_text,
            )
            all_generated_nodes.append(node_data)
            yield node_data

        # --- STAGE 3 (OPTIONAL): ANALYZE FOR QUALITY ISSUES ---
        if run_post_analysis:
            analysis_result = await self._analyze_nodes(all_generated_nodes)

            if analysis_result.duplicate_nodes:
                summary = (
                    f"Post-analysis found {len(analysis_result.duplicate_nodes)} "
                    "potentially duplicate nodes."
                )
                logger.warning(summary)
                for node in analysis_result.duplicate_nodes:
                    logger.warning(
                        f"  - Node '{node.label}' is a likely duplicate of '{node.duplicate_of}'."
                    )

            if analysis_result.low_substance_nodes:
                summary = (
                    f"Post-analysis found {len(analysis_result.low_substance_nodes)} "
                    "low-substance or tautological nodes."
                )
                logger.warning(summary)
                for node in analysis_result.low_substance_nodes:
                    logger.warning(
                        f"  - Node '{node.label}' is low-substance. Reason: {node.reason}"
                    )

        total_duration = time.time() - start_time
        logger.info(f"Total generation time: {total_duration:.2f} seconds.")
        yield StatusChunk(title="Saving your mind map...")
