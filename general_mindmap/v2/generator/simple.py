import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Set, Tuple
from uuid import uuid4

from langchain_community.callbacks import get_openai_callback
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from pydantic import BaseModel, Field

from general_mindmap.v2.dial.client import DialClient
from generator.chainer.model_handler import ModelCreator
from generator.common.constants import DataFrameCols as Col
from generator.common.structs import (
    DocStatusChunk,
    EdgeData,
    Generator,
    InitMindmapRequest,
    NodeData,
    RootNodeChunk,
    StatusChunk,
)
from generator.core.actions.docs import fetch_all_docs_content
from generator.core.stages.doc_handler import DocHandler
from generator.core.stages.doc_handler.constants import DocCategories as DocCat

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _chunk_list(data: List[Any], size: int) -> List[List[Any]]:
    """Splits a list into chunks of a specified size."""
    if size <= 0:
        raise ValueError("Chunk size must be positive.")
    return [data[i : i + size] for i in range(0, len(data), size)]


class AnswerCountMismatchError(Exception):
    """Custom exception for when LLM answer count doesn't match question count."""

    def __init__(self, n_questions, n_answers, raw_output=None):
        self.n_questions = n_questions
        self.n_answers = n_answers
        self.raw_output = raw_output
        super().__init__(
            f"Mismatch in question/answer count. Sent {n_questions} questions, received {n_answers} answers."
        )


# --- Pydantic Models for Structured Output ---


class AnswerPart(BaseModel):
    """A single, discrete factual statement with its source."""

    statement: str = Field(
        ...,
        description="A single, discrete factual statement extracted directly from the text.",
    )
    doc_id: str = Field(
        ..., description="The document ID from the source <document> tag."
    )
    page: int = Field(
        ...,
        description="The page number from the source <page> tag where the fact was found.",
    )


class Node(BaseModel):
    """Represents a node in the mind map."""

    label: str = Field(..., description="Unique label for the node.")
    question: str = Field(
        ..., description="A question that this node's content answers."
    )
    answer_parts: List[AnswerPart] = Field(
        default_factory=list,
        description="List of factual statements that form the answer.",
    )


class Edge(BaseModel):
    """Represents a directed edge between two nodes in the mind map."""

    source: str = Field(..., description="The label of the parent node.")
    target: str = Field(..., description="The label of the child node.")


class Mindmap(BaseModel):
    """The complete mind map structure."""

    root: str = Field(..., description="The label of the root node.")
    nodes: List[Node] = Field(
        ..., description="A list of all nodes in the graph."
    )
    edges: List[Edge] = Field(
        ..., description="A list of all edges connecting the nodes."
    )


# --- Pydantic Models for Two-Stage Generation ---


class NodeStructure(BaseModel):
    """Represents the structure of a node without the detailed answer."""

    label: str = Field(..., description="Unique label for the node.")
    question: str = Field(
        ..., description="A question that this node's content will answer."
    )


class MindmapStructure(BaseModel):
    """The structure of the mind map, without detailed answers."""

    root: str = Field(..., description="The label of the root node.")
    nodes: List[NodeStructure] = Field(
        ..., description="A list of nodes with their questions."
    )
    edges: List[Edge] = Field(
        ..., description="A list of all edges connecting the nodes."
    )


class Answer(BaseModel):
    """Contains the answer parts for a single question."""

    answer_parts: List[AnswerPart] = Field(
        ...,
        description="A list of factual statements answering a specific question.",
    )


class AnswerList(BaseModel):
    """A list of answers corresponding to a list of questions."""

    answers: List[Answer] = Field(
        ...,
        description="A list of answer objects, in the same order as the input questions.",
    )


# --- Constants and Configuration ---

TOKENS_PER_MILLION = 1_000_000
PROMPT = "prompt"
COMPLETION = "completion"

MULTIMODAL_CATEGORIES: Set[str] = {
    DocCat.PDF_AS_A_WHOLE,
    DocCat.PPTX_AS_A_WHOLE,
}
TEXT_CATEGORIES: Set[str] = {
    DocCat.TXT_AS_A_WHOLE,
    DocCat.HTML_AS_A_WHOLE,
    DocCat.LINK_AS_A_WHOLE,
}

DEFAULT_PRICING_MAP = {
    "gemini-2.5-pro": {PROMPT: 1.25, COMPLETION: 10.00},
    "gpt-5-2025-08-07": {PROMPT: 1.25, COMPLETION: 10.00},
    "gpt-5-mini-2025-08-07": {PROMPT: 0.25, COMPLETION: 2.00},
    "gpt-5-nano-2025-08-07": {PROMPT: 0.05, COMPLETION: 0.40},
}

SYSTEM_PROMPT = """
# ROLE
You are an AI architect that transforms raw documents into perfectly structured, highly detailed JSON mind maps.

# OBJECTIVE
Your mission is to create the most detailed and structurally perfect mind map possible from the source text. Success is measured by a balance of three critical factors: absolute textual accuracy, maximum logical granularity, and perfect structural integrity.

# NON-NEGOTIABLE CORE PRINCIPLES
You must adhere to all of the following principles. A failure in any one of these areas is a failure of the entire task.

1.  **Principle 1: Textual Fidelity (Accuracy and Content are Paramount)**
    *   **Source-Grounded Answers:** While all information must originate *exclusively* from the source text, you are permitted to rephrase it to form a natural and direct answer to the node's `question`. You should synthesize facts from the text into a coherent statement, but you are **strictly forbidden** from adding any information, speculation, or meaning not explicitly present in the source. The goal is a natural answer, not a robotic copy-paste.
    *   **Content Mandate:** Every node **must** provide a meaningful answer. The `answer_parts` array cannot be empty and must contain at least one factual `statement`. A node without an answer is an invalid node.
    *   **Primacy of Accuracy:** This principle overrides all others. If a choice is between creating a more natural-sounding answer by altering the facts and a less natural one by keeping the facts intact, you **must** choose to keep the facts perfectly intact.

2.  **Principle 2: Maximal Decomposition (Pursue Granularity Relentlessly)**
    *   Within the bounds of accuracy, your goal is to create the most granular map possible. Deconstruct the source text to its maximum logical depth.
    *   **Decomposition Mandate:** If a concept, list, definition, process, or statement can be logically broken down into 'N' distinct facts, its corresponding node **must** branch into 'N' child nodes.
    *   **No Consolidation:** Never consolidate distinct facts into a single node. Always err on the side of creating more, smaller, more specific nodes.
    *   A node should only be a leaf (have zero children) if the text provides no further decomposable details about it. There is no upper limit to how many children a node can have.

3.  **Principle 3: Structural Integrity (The Blueprint Must Be Perfect)**
    *   **Valid JSON:** The final output must be a single, perfectly valid JSON object.
    *   **Unique Labels:** Every node's `label` must be unique across the entire map.
    *   **Tree Structure:** The graph must be a single connected tree. Every node except the root must have **exactly one** parent (i.e., appear as a `target` in exactly one `edges` object).
    *   **Root Node:** The `root` field must contain the `label` of the root node. The root node object must be in the `nodes` array and have at least three outgoing edges.

4.  **Principle 4: Adherence to Task-Specific Instructions**
    *   **User-Defined Constraints:** You must strictly follow all user-provided instructions that define the final output's characteristics. This includes directives on:
        *   **Content & Structure:** How the final graph should be organized (e.g., using predefined categories for first-level nodes, following a specific hierarchy).
        *   **Style & Persona:** The desired language, tone, point of view, or persona for the output's content.
    *   **Precise Source Attribution:** Every `statement` must be paired with its correct `doc_id` and `page` number from the source text. A missing `doc_id` or `page` for any statement is a critical failure.
    *   **Hyperlink Integration:** All hyperlinks from the source must be correctly formatted in the `statement` using Markdown: `[anchor text](URL)`.

# OUTPUT FORMAT
The output must be a single, valid JSON object. All fields shown are mandatory.
```json
{
  "root": "Root Node Label",
  "nodes": [
    {
      "label": "Unique Node Label",
      "question": "A question that this node's content answers, based on decomposition logic or user instructions.",
      "answer_parts": [
        {
          "statement": "A single, discrete factual statement extracted directly from the text.",
          "doc_id": "doc_1",
          "page": 1
        }
      ]
    }
  ],
  "edges": [
    { "source": "Parent Node Label", "target": "Child Node Label" }
  ]
}
```
"""

# --- PROMPTS FOR TWO-STAGE GENERATION ---

STRUCTURE_SYSTEM_PROMPT = """
# ROLE
You are an AI architect that designs the structural blueprint for highly detailed JSON mind maps.

# OBJECTIVE
Your mission is to architect the deepest, most granular, and structurally perfect mind map blueprint possible from the source text. Your output will define the complete graph structure, consisting of nodes (each with a unique `label` and a specific `question`) and the `edges` that connect them. **Your primary measure of success is the granularity and depth of the map.**

# NON-NEGOTIABLE CORE PRINCIPLES

1.  **Principle 1: Relentless Decomposition (Your Core Task)**
    *   You are not just organizing topics; you are deconstructing the source text into its most fundamental, atomic facts.
    *   **The Litmus Test for Decomposition:** For any potential node, you must internally ask: **"Does the source text provide multiple distinct facts, details, or examples for this topic?"**
        *   If the answer is YES, you **must** break that node down into multiple child nodes. Each child node will represent one of those distinct facts.
        *   If the answer is NO, and the text only provides a single, indivisible piece of information, only then can you create a single leaf node.
    *   **AVOID CONSOLIDATION AT ALL COSTS:** Never create a single "list" node for a topic that has multiple items. For example, if the text states "The system has three components: A, B, and C," you must NOT create one node answering "What are the three components?". Instead, you MUST create a parent node ("Components of the System") with three children: "Component A," "Component B," and "Component C."
    *   **Err on the Side of Granularity:** When in doubt, always choose to split a node. Creating too many nodes is better than creating too few.

2.  **Principle 2: Question Specificity**
    *   Every `question` must be highly specific and ask for a single, discrete piece of information.
    *   **AVOID 'Umbrella' Questions:** Do not ask questions like "What are the details of X?" or "Tell me about Y." Instead, ask targeted questions like "What is the primary function of X?" or "In what year was Y established?". The specificity of your questions will naturally guide you to a more granular structure.

3.  **Principle 3: Structural Integrity (The Blueprint Must Be Perfect)**
    *   **Valid JSON:** The final output must be a single, perfectly valid JSON object.
    *   **Unique Labels:** Every node's `label` must be unique across the entire map.
    *   **Tree Structure:** The graph must be a single connected tree. Every node except the root must have **exactly one** parent.
    *   **Root Node:** The `root` field must contain the `label` of the root node.

4.  **Principle 4: Adherence to Task Directives**
    *   **User-Defined Constraints:** You must strictly follow all user-provided instructions that define the final output's characteristics (e.g., predefined first-level categories, language, tone).

# OUTPUT FORMAT
The output must be a single, valid JSON object conforming to the schema below. All fields shown are mandatory.
```json
{
  "root": "Root Node Label",
  "nodes": [
    {
      "label": "Unique Node Label",
      "question": "A specific, targeted question that asks for a single, atomic fact."
    }
  ],
  "edges": [
    { "source": "Parent Node Label", "target": "Child Node Label" }
  ]
}
```
"""

QUESTION_ANSWERING_PROMPT = """
# ROLE
You are a detail-oriented AI researcher. Your task is to answer a list of questions with specific, factual information from a provided document.

# OBJECTIVE
You will be given a JSON list of questions and the source document(s). For each question, you must find the precise answer within the documents and return it with all required source citations.

# NON-NEGOTIABLE CORE PRINCIPLES
A failure in any of these areas is a failure of the entire task.

1.  **Principle 1: Content and Accuracy**
    *   **Answer Mandate:** You **must** find a relevant factual answer for every question provided. An empty `answer_parts` array is an invalid response and constitutes a task failure. The preceding step has guaranteed that an answer exists for every question.
    *   **Verbatim Extraction:** All `statement` values must be extracted directly and verbatim from the source text. You are strictly forbidden from summarizing, rephrasing, or inventing information.

2.  **Principle 2: Sourcing and Formatting**
    *   **Precise Source Citation:** Every `statement` **must** be paired with its correct `doc_id` from the `<document>` tag and the `page` number from the `<page>` tag where the fact was found.
    *   **Hyperlink Integration:** If the source text contains hyperlinks, you **must** integrate them into the `statement` using Markdown format: `[anchor text](URL)`.

3.  **Principle 3: Structural Fidelity**
    *   **Strict Order Preservation:** The output list of answers **must** be the same length and in the exact same order as the input list of questions. The first answer object corresponds to the first question, the second to the second, and so on.
    *   **JSON Schema Adherence:** Your entire output must be a single, valid JSON object conforming exactly to the schema in the `OUTPUT FORMAT` section.

# OUTPUT FORMAT
Your output must be a single JSON object. The `"answers"` list must contain exactly one object for each question in the input.
```json
{
  "answers": [
    {
      "answer_parts": [
        {
          "statement": "This is a single, discrete factual statement extracted directly from the text.",
          "doc_id": "doc_1",
          "page": 1
        }
      ]
    }
  ]
}
```
"""


@dataclass
class GeneratorConfig:
    """Configuration for the SimpleGenerator."""

    model: str
    prompt: str
    dial_url: str
    api_key: str
    pricing_map: Dict[str, Dict[str, float]] = field(default_factory=dict)
    llm_total_timeout: float = 1500.0
    llm_connect_timeout: float = 5.0


class SimpleGenerator(Generator):
    """
    A generator that processes a batch of mixed-type documents, invokes
    an LLM to create a graph with precise citations, and yields graph
    components.
    """

    def __init__(self, config: GeneratorConfig):
        super().__init__(dial_url=config.dial_url, api_key=config.api_key)
        self.config = config
        self._pricing_map = DEFAULT_PRICING_MAP.copy()
        self._pricing_map.update(config.pricing_map)
        self.client = DialClient(
            dial_url=self.config.dial_url, api_key="auto", etag=""
        )
        self.llm = ModelCreator.get_chat_model(
            self.config.model,
            total_timeout=self.config.llm_total_timeout,
            read_timeout=1500.0,
            connect_timeout=self.config.llm_connect_timeout,
        )

    def _log_llm_usage(self, cb: OpenAICallbackHandler):
        """Logs token usage and calculates cost."""
        total_tokens = getattr(cb, "total_tokens", 0)
        prompt_tokens = getattr(cb, "prompt_tokens", 0)
        completion_tokens = getattr(cb, "completion_tokens", 0)

        logger.info(
            f"LLM Usage: Total={total_tokens} "
            f"(Prompt={prompt_tokens}, Completion={completion_tokens})"
        )
        model_name = self.config.model.lower()
        if model_name in self._pricing_map:
            prices = self._pricing_map[model_name]
            prompt_cost = prices.get(PROMPT, 0.0)
            completion_cost = prices.get(COMPLETION, 0.0)
            cost = (prompt_tokens / TOKENS_PER_MILLION) * prompt_cost + (
                completion_tokens / TOKENS_PER_MILLION
            ) * completion_cost
            logger.info(f"Estimated Cost (from custom map): ${cost:.6f}")
        elif "gpt" in model_name and cb.total_cost > 0:
            cost = cb.total_cost
            logger.info(f"Estimated Cost (from OpenAI callback): ${cost:.6f}")
        else:
            logger.warning(
                "Cost calculation not available for model "
                f"'{self.config.model}'."
            )

    @staticmethod
    def _format_page_content(
        texts: List[str], images: List[str], page_id: int, tag_name: str
    ) -> List[Dict[str, Any]]:
        """
        Formats the content for a single page or slide.
        This method is guaranteed to return a list.
        """
        content = []
        page_text = "\n".join(texts)
        content.append(
            {
                "type": "text",
                "text": f'<{tag_name} number="{page_id}">\n\n{page_text}',
            }
        )
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_data}"},
            }
            for img_data in images
        )
        content.append({"type": "text", "text": f"</{tag_name}>"})
        return content

    async def _prepare_document_inputs(
        self, request: InitMindmapRequest
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Fetches and formats a batch of mixed-type documents into a
        unified LLM input, using document IDs for tracking.
        """
        start_time = time.time()
        doc_handler = DocHandler(strategy="whole")
        docs_and_their_content = await fetch_all_docs_content(
            request.documents, self.client
        )
        chunked_docs_df, _ = await doc_handler.chunk_docs(
            docs_and_their_content
        )

        if chunked_docs_df.empty:
            return [], 0.0

        human_message_content = []
        for _, row in chunked_docs_df.iterrows():
            doc_cat = row[Col.DOC_CAT]
            doc_id = row[Col.DOC_ID]
            content = row[Col.CONTENT]

            if doc_cat == DocCat.UNSUPPORTED:
                logger.warning(
                    f"Skipping unsupported document with ID: {doc_id}"
                )
                continue

            human_message_content.append(
                {"type": "text", "text": f'<document id="{doc_id}">'}
            )

            if doc_cat in MULTIMODAL_CATEGORIES:
                tag_name = "slide" if doc_cat == DocCat.PPTX else "page"
                for page in content:
                    page_content = self._format_page_content(
                        texts=page.get("texts", []),
                        images=page.get("images", []),
                        page_id=page.get("page_id"),
                        tag_name=tag_name,
                    )
                    if page_content:
                        human_message_content.extend(page_content)

            elif doc_cat in TEXT_CATEGORIES:
                if content:
                    human_message_content.append(
                        {"type": "text", "text": content}
                    )

            human_message_content.append(
                {"type": "text", "text": "</document>"}
            )

        duration = time.time() - start_time
        logger.info(f"Document processing took {duration:.2f} seconds.")
        return human_message_content, duration

    async def _invoke_llm(
        self, human_message_content: List[Dict[str, Any]], is_stream: bool
    ) -> Tuple[Dict[str, Any], float]:
        """
        Invokes the LLM with the prepared content and returns the graph.
        """
        start_time = time.time()
        raw_output = None
        if not human_message_content:
            logger.error("Cannot invoke LLM with no content.")
            return {}, 0.0

        is_gpt_model = "gpt" in self.config.model.lower()
        if is_gpt_model:
            logger.info(
                "Using Pydantic model for structured output with GPT model."
            )
            structured_llm = self.llm.with_structured_output(Mindmap)
            system_prompt = SYSTEM_PROMPT.split("# OUTPUT FORMAT")[0].strip()
        else:
            logger.info(
                "Using json_mode for structured output with non-GPT model."
            )
            structured_llm = self.llm.with_structured_output(method="json_mode")
            system_prompt = SYSTEM_PROMPT

        llm_input = [
            ("system", system_prompt + "\n\n" + self.config.prompt),
            ("human", human_message_content),
        ]

        with get_openai_callback() as cb:
            if is_stream:
                async for chunk in structured_llm.astream(llm_input):
                    raw_output = chunk
                    logging.debug("Chunk generated")
            else:
                raw_output = await structured_llm.ainvoke(llm_input)
            self._log_llm_usage(cb)

        graph_dict = {}
        if raw_output:
            if is_gpt_model:
                graph_dict = raw_output.model_dump()
            else:
                graph_dict = raw_output

        duration = time.time() - start_time
        logger.info(f"LLM generation completed in {duration:.2f} seconds.")
        return graph_dict, duration

    @staticmethod
    async def _process_graph_response(
        graph: Dict[str, Any],
    ) -> AsyncGenerator[NodeData | EdgeData | RootNodeChunk, None]:
        """
        Processes the graph response, assembling node details and
        programmatically creating inline citations in the format
        ^[doc_id.page]^.
        """
        start_time = time.time()
        try:
            nodes = graph.get("nodes", [])
            edges = graph.get("edges", [])
            root_label = graph.get("root")

            num_nodes, num_edges = len(nodes), len(edges)
            logger.info(
                f"Graph received with {num_nodes} nodes and {num_edges} edges."
            )

            label_to_id = {node["label"]: str(uuid4()) for node in nodes}

            for node in nodes:
                node_id = label_to_id.get(node["label"])
                if not node_id:
                    continue

                answer_parts = node.get("answer_parts", [])
                details_parts = []
                for part in answer_parts:
                    statement = part.get("statement")
                    doc_id = part.get("doc_id")
                    page = part.get("page")

                    if not statement:
                        continue

                    if doc_id and page is not None:
                        citation = f" ^[{doc_id}.{page}]^"
                        details_parts.append(statement + citation)
                    else:
                        details_parts.append(statement)

                details_text = " ".join(details_parts)

                yield NodeData(
                    id=node_id,
                    label=node["label"],
                    question=node.get("question", ""),
                    details=details_text,
                )

            for edge in edges:
                source_id = label_to_id.get(edge.get("source"))
                target_id = label_to_id.get(edge.get("target"))
                if source_id and target_id:
                    yield EdgeData(
                        id=str(uuid4()), source=source_id, target=target_id
                    )
                    yield EdgeData(
                        id=str(uuid4()), source=target_id, target=source_id
                    )

            if root_id := label_to_id.get(root_label, ""):
                yield RootNodeChunk(root_id=root_id)

        except (KeyError, TypeError) as e:
            logger.error(
                f"Error processing graph response: {e}. Response: {graph}"
            )
            return
        duration = time.time() - start_time
        logger.info(f"Graph processing took {duration:.2f} seconds.")

    async def generate(
        self, request: InitMindmapRequest, is_stream: bool = True
    ) -> AsyncGenerator[
        StatusChunk | NodeData | EdgeData | DocStatusChunk | RootNodeChunk, None
    ]:
        """
        Main generation pipeline for handling mixed-document batches.
        """
        total_start_time = time.time()

        inputs, _ = await self._prepare_document_inputs(request)
        if not inputs:
            logger.warning("No processable inputs generated from documents.")
            return

        graph, _ = await self._invoke_llm(inputs, is_stream)
        if not graph:
            logger.error("LLM failed to generate a graph.")
            return

        async for chunk in self._process_graph_response(graph):
            yield chunk

        total_duration = time.time() - total_start_time
        logger.info(f"Total generation time: {total_duration:.2f} seconds.")


class TwoStageGenerator(SimpleGenerator):
    """
    Generates a mind map in two highly efficient stages:
    1.  Generate the complete graph structure (nodes with questions,
    edges).
    2.  Send a simple list of questions to the LLM and map the ordered
        answers back to the original nodes by index.
    """

    FALLBACK_BATCH_SIZE = 10

    def __init__(self, config: GeneratorConfig):
        super().__init__(config)

    async def _generate_structure(
        self, human_message_content: List[Dict[str, Any]], is_stream: bool
    ) -> Dict[str, Any]:
        """Stage 1: Generates the mind map's structure."""
        logger.info("Stage 1: Generating mind map structure...")
        start_time = time.time()
        raw_output = None

        is_gpt_model = "gpt" in self.config.model.lower()
        if is_gpt_model:
            logger.info(
                "Stage 1: Using Pydantic model for structured output with GPT model."
            )
            structured_llm = self.llm.with_structured_output(MindmapStructure)
            system_prompt_base = STRUCTURE_SYSTEM_PROMPT.split(
                "# OUTPUT FORMAT"
            )[0].strip()
        else:
            logger.info(
                "Stage 1: Using json_mode for structured output with non-GPT model."
            )
            structured_llm = self.llm.with_structured_output(method="json_mode")
            system_prompt_base = STRUCTURE_SYSTEM_PROMPT

        final_system_prompt = system_prompt_base
        if self.config.prompt:
            final_system_prompt += (
                "\n\n# IMPORTANT USER-PROVIDED INSTRUCTIONS\n"
                "The user has provided the following instructions. When performing your task, "
                "you MUST pay close attention to any directives related to the **structure, hierarchy, topics, and overall organization** of the mind map. "
                "You should IGNORE instructions related to the answers.\n\n"
                "## User Instructions:\n"
                f'"""\n{self.config.prompt}\n"""'
            )

        llm_input = [
            ("system", final_system_prompt),
            ("human", human_message_content),
        ]
        with get_openai_callback() as cb:
            if is_stream:
                async for chunk in structured_llm.astream(llm_input):
                    raw_output = chunk
                    logging.debug("Chunk generated")
            else:
                raw_output = await structured_llm.ainvoke(llm_input)
            self._log_llm_usage(cb)

        graph_structure = {}
        if raw_output:
            if is_gpt_model:
                graph_structure = raw_output.model_dump()
            else:
                graph_structure = raw_output

        duration = time.time() - start_time
        logger.info(
            f"Stage 1: Structure generation completed in {duration:.2f}s."
        )
        return graph_structure

    async def _fetch_answers_for_question_list(
        self,
        human_message_content: List[Dict[str, Any]],
        questions: List[str],
        is_stream: bool,
    ) -> List[Dict[str, Any]]:
        """
        Stage 2 (Robust): Sends a list of questions to the LLM and gets
        an ordered list of answers back, with a cascading fallback
        strategy.
        """
        logger.info(
            f"Stage 2: Fetching answers for {len(questions)} questions..."
        )
        start_time = time.time()
        max_retries = 2

        try:
            # Attempt 1: Process the entire batch with retries
            answers = await self._execute_llm_batch_request(
                human_message_content,
                questions,
                is_stream,
                max_retries=max_retries,
            )
            duration = time.time() - start_time
            logger.info(
                f"Stage 2: Successfully fetched all answers in a single batch in {duration:.2f}s."
            )
            return answers

        except Exception as e:
            logger.warning(
                f"Stage 2: Full batch processing failed after {max_retries} attempts ({e}). "
                f"Switching to fallback strategy with batch size {self.FALLBACK_BATCH_SIZE}."
            )
            # Attempt 2: Cascading fallback
            questions_with_indices = list(enumerate(questions))
            answers = await self._handle_cascading_fallback(
                human_message_content, questions_with_indices, is_stream
            )

            duration = time.time() - start_time
            logger.info(
                f"Stage 2: Answer fetching completed via fallback in {duration:.2f}s."
            )
            return answers

    async def _execute_llm_batch_request(
        self,
        human_message_content: List[Dict[str, Any]],
        questions: List[str],
        is_stream: bool,
        max_retries: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Executes a single LLM request for a batch of questions, with optional retries.
        This is the core execution unit.
        """
        if not questions:
            return []

        is_gpt_model = "gpt" in self.config.model.lower()
        num_questions = len(questions)

        # Start with the base QA prompt
        system_prompt = QUESTION_ANSWERING_PROMPT

        # Inject the full user prompt with specific guidance for this stage.
        if self.config.prompt:
            system_prompt += (
                "\n\n# IMPORTANT USER-PROVIDED INSTRUCTIONS\n"
                "The user has provided the following instructions. When performing your task, "
                "you MUST pay close attention to any directives related to the **content, style, tone, and formatting of the answers**. "
                "For example, look for constraints on conciseness, which data sources to use (e.g., specific pages or 'only from images'), or the desired answer language. "
                "You should IGNORE instructions related to the overall structure (nodes and edges).\n\n"
                "## User Instructions:\n"
                f'"""\n{self.config.prompt}\n"""'
            )

        # Add the dynamic prompt enhancement for batch size
        prompt_enhancement = (
            f"\n\nYou will be given {num_questions} questions. "
            f"You MUST provide exactly {num_questions} answers in the final JSON output, "
            "preserving the original order."
        )
        system_prompt += prompt_enhancement

        content_with_questions = [
            {
                "type": "text",
                "text": "Here is the JSON list of questions to answer:",
            },
            {"type": "text", "text": f"```json\n{questions}\n```"},
            {"type": "text", "text": "\nHere are the source documents:"},
            *human_message_content,
        ]

        if is_gpt_model:
            structured_llm = self.llm.with_structured_output(AnswerList)
        else:
            structured_llm = self.llm.with_structured_output(method="json_mode")

        llm_input = [
            ("system", system_prompt),
            ("human", content_with_questions),
        ]

        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                raw_output = None
                with get_openai_callback() as cb:
                    if is_stream:
                        async for chunk in structured_llm.astream(llm_input):
                            raw_output = chunk
                    else:
                        raw_output = await structured_llm.ainvoke(llm_input)
                    self._log_llm_usage(cb)

                if not raw_output:
                    raise ValueError("LLM returned an empty response.")

                response_json = (
                    raw_output.model_dump() if is_gpt_model else raw_output
                )
                answers = response_json.get("answers", [])

                if len(answers) == num_questions:
                    return answers  # Success! The only successful exit point.

                # If validation fails, raise an error to be caught by the except block.
                raise AnswerCountMismatchError(
                    n_questions=num_questions, n_answers=len(answers)
                )

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"LLM batch request for {num_questions} questions failed on attempt {attempt + 1}/{max_retries + 1}: {e}"
                )
                if attempt < max_retries:
                    await asyncio.sleep(2 * (2**attempt))  # Exponential backoff

        # This part is now reachable. If the loop completes, all retries have failed.
        # We re-raise the last exception that was caught.
        raise last_exception from None

    async def _handle_cascading_fallback(
        self,
        human_message_content: List[Dict[str, Any]],
        questions_with_indices: List[Tuple[int, str]],
        is_stream: bool,
    ) -> List[Dict[str, Any]]:
        """
        Handles the fallback logic: first in mini-batches, then one-by-one for any failed batches.
        """
        final_answers = {}

        # --- Level 1 Fallback: Mini-batches ---
        batches = _chunk_list(questions_with_indices, self.FALLBACK_BATCH_SIZE)

        tasks = [
            self._execute_llm_batch_request(
                human_message_content, [q for _, q in batch], is_stream
            )
            for batch in batches
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # --- Process results and identify remaining failures ---
        failed_questions_with_indices = []
        for i, result in enumerate(results):
            batch = batches[i]
            if isinstance(result, Exception):
                logger.warning(
                    f"Mini-batch of {len(batch)} questions failed. Adding them to the one-by-one queue."
                )
                failed_questions_with_indices.extend(batch)
            else:
                for j, answer in enumerate(result):
                    original_index = batch[j][0]
                    final_answers[original_index] = answer

        # --- Level 2 Fallback: One-by-one ---
        if failed_questions_with_indices:
            logger.info(
                f"Processing {len(failed_questions_with_indices)} remaining questions one-by-one."
            )
            single_tasks = [
                self._execute_llm_batch_request(
                    human_message_content,
                    [question],  # A batch of one
                    is_stream,
                    max_retries=1,
                )
                for _, question in failed_questions_with_indices
            ]
            single_results = await asyncio.gather(
                *single_tasks, return_exceptions=True
            )

            for i, result in enumerate(single_results):
                original_index, question_text = failed_questions_with_indices[i]
                if isinstance(result, Exception):
                    logger.error(
                        f"Final attempt for question '{question_text}' failed: {result}"
                    )
                    final_answers[original_index] = {
                        "answer_parts": [
                            {
                                "statement": "Error: Failed to generate a valid answer for this question after multiple fallbacks.",
                                "doc_id": "N/A",
                                "page": 0,
                            }
                        ]
                    }
                else:
                    final_answers[original_index] = result[0]

        # Reassemble the final list in the correct order
        total_questions = len(questions_with_indices)
        return [final_answers[i] for i in range(total_questions)]

    async def generate(
        self, request: InitMindmapRequest, is_stream: bool = True
    ) -> AsyncGenerator[
        StatusChunk | NodeData | EdgeData | DocStatusChunk | RootNodeChunk, None
    ]:
        """
        Main generation pipeline implementing the efficient two-stage
        process.
        """
        total_start_time = time.time()
        logger.info("Starting efficient two-stage mind map generation.")

        inputs, _ = await self._prepare_document_inputs(request)
        if not inputs:
            logger.warning("No processable inputs; aborting generation.")
            return

        # --- STAGE 1: GENERATE STRUCTURE ---
        graph_structure = await self._generate_structure(inputs, is_stream)
        if not graph_structure or "nodes" not in graph_structure:
            logger.error(
                "Failed to generate a valid graph structure. Aborting."
            )
            return

        nodes = graph_structure.get("nodes", [])
        edges = graph_structure.get("edges", [])
        root_label = graph_structure.get("root")
        logger.info(
            f"Structure received: {len(nodes)} nodes, {len(edges)} edges."
        )

        label_to_id = {node["label"]: str(uuid4()) for node in nodes}

        # --- YIELD SKELETON: Yield root, edges, and nodes with empty
        # details ---
        if root_id := label_to_id.get(root_label):
            yield RootNodeChunk(root_id=root_id)

        for edge in edges:
            source_id = label_to_id.get(edge.get("source"))
            target_id = label_to_id.get(edge.get("target"))
            if source_id and target_id:
                yield EdgeData(
                    id=str(uuid4()), source=source_id, target=target_id
                )
                yield EdgeData(
                    id=str(uuid4()), source=target_id, target=source_id
                )

        logger.info("Yielded initial graph skeleton (edges, root).")

        # --- STAGE 2: FETCH AND YIELD CONTENT ---
        questions_to_ask = [node.get("question", "") for node in nodes]
        answers = await self._fetch_answers_for_question_list(
            inputs, questions_to_ask, is_stream
        )

        if not answers or len(answers) != len(nodes):
            logger.error(
                f"Failed to augment graph: Mismatch in question/answer count. "
                f"Sent {len(nodes)} questions, received {len(answers)} answers."
            )
            return

        logger.info(f"Received and processing {len(answers)} answers.")
        # Map answers back to nodes by index using zip
        for node, answer_obj in zip(nodes, answers):
            node_id = label_to_id.get(node["label"])
            if not node_id:
                continue

            answer_parts = answer_obj.get("answer_parts", [])
            details_parts = []
            for part in answer_parts:
                statement = part.get("statement")
                doc_id = part.get("doc_id")
                page = part.get("page")

                if not statement:
                    continue

                citation = (
                    f" ^[{doc_id}.{page}]^"
                    if doc_id and page is not None
                    else ""
                )
                details_parts.append(statement + citation)

            details_text = " ".join(details_parts)

            # Yield an updated NodeData object to fill in the details
            yield NodeData(
                id=node_id,
                label=node["label"],
                question=node["question"],
                details=details_text,
            )

        total_duration = time.time() - total_start_time
        logger.info(f"Total generation time: {total_duration:.2f} seconds.")
