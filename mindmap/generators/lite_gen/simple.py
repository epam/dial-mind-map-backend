import logging
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Set
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit
from uuid import uuid4

from langchain_community.callbacks.openai_info import OpenAICallbackHandler

from mindmap.dial.client import DialClient
from mindmap.generators.base import (
    EdgeData,
    Generator,
    InitMindmapRequest,
    MindMapStreamChunk,
    NodeData,
    RootNodeChunk,
)
from mindmap.generators.univ_gen.chainer.model_handler import ModelCreator
from mindmap.generators.univ_gen.core.stages.doc_handler.constants import (
    DocCategories as DocCat,
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

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Prompt ---

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
    *   **Precise Source Attribution:** Every `statement` must be paired with its correct `doc_id` and `chunk` number from the source text. A missing `doc_id` or `page` for any statement is a critical failure.
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


@dataclass
class GeneratorConfig:
    """Configuration for the SimpleGenerator."""

    model: str
    gen_prompt: str
    rag_prompt: str
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

    def __init__(self, config: GeneratorConfig, client: DialClient):
        super().__init__(dial_url=config.dial_url, api_key=config.api_key)
        self.config = config
        self._pricing_map = DEFAULT_PRICING_MAP.copy()
        self._pricing_map.update(config.pricing_map)
        self.client = client
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
    def _format_chunk_content(
        texts: List[str], images: List[str], chunk_id: int | str
    ) -> List[Dict[str, Any]]:
        """
        Formats the content for a single chunk (from a page, slide, or
        text block). This method is guaranteed to return a list.
        """
        content = []
        chunk_text = "\n".join(texts)

        if not chunk_text and not images:
            return []

        content.append(
            {"type": "text", "text": f'<chunk id="{chunk_id}">\n\n{chunk_text}'}
        )

        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_data}"},
                "metadata": {"source_chunk_id": chunk_id},
            }
            for img_data in images
        )

        content.append({"type": "text", "text": "</chunk>"})
        return content

    @staticmethod
    def _format_chunk_text_with_links(chunk: Any) -> str:
        """
        Processes a chunk's text, embedding hyperlinks from its metadata
        as Markdown-formatted absolute URLs.
        """
        chunk_text = chunk.text
        metadata = chunk.metadata

        link_texts = metadata.get("link_texts")
        link_urls = metadata.get("link_urls")
        base_url = metadata.get("source_display_name")

        if not (link_texts and link_urls and base_url):
            return chunk_text

        links_to_replace = {}
        for anchor, relative_url in zip(link_texts, link_urls):
            absolute_url = urljoin(base_url, relative_url)

            parts = urlsplit(absolute_url)

            unquoted_path = unquote(parts.path)

            quoted_path = quote(unquoted_path, safe="/:")

            final_url = urlunsplit(parts._replace(path=quoted_path))

            links_to_replace[anchor] = f"[{anchor}]({final_url})"

        sorted_anchors = sorted(links_to_replace.keys(), key=len, reverse=True)
        for anchor in sorted_anchors:
            chunk_text = chunk_text.replace(anchor, links_to_replace[anchor])

        return chunk_text

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

    @abstractmethod
    async def generate(
        self, request: InitMindmapRequest
    ) -> AsyncGenerator[MindMapStreamChunk, None]:
        empty_iter: list[MindMapStreamChunk] = []
        for chunk in empty_iter:
            yield chunk
