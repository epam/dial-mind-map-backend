import logging
import re
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_simplified_input_for_structure(
        human_message_content: List[Dict[str, Any]],
        doc_metadatas: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Creates a simplified input for Stage 1 (structure generation).
    """
    logger.info("Creating simplified input for Stage 1 structure generation.")
    simplified_input = []
    current_doc_id, current_doc_type = None, ""

    doc_id_regex = re.compile(r'<document id="([^"]+)">')
    tags_to_remove_regex = re.compile(r"</?(page|chunk|slide)[^>]*>")

    for item in human_message_content:
        item_type = item.get("type")
        if item_type == "text":
            cleaned_text = tags_to_remove_regex.sub("", item.get("text", ""))
            if not cleaned_text.strip(): continue

            doc_id_match = doc_id_regex.search(cleaned_text)
            if doc_id_match:
                current_doc_id = doc_id_match.group(1)
                current_doc_type = doc_metadatas.get(current_doc_id, {}).get("file_type", "").lower()

            simplified_input.append({"type": "text", "text": cleaned_text})

            if "</document>" in cleaned_text:
                current_doc_id, current_doc_type = None, ""

        elif item_type == "image_url":
            if current_doc_type in ["pdf_as_a_whole", "pptx_as_a_whole"]:
                simplified_input.append(item)

    return simplified_input


def chunk_list(data: List[Any], size: int) -> List[List[Any]]:
    """Splits a list into chunks of a specified size."""
    if size <= 0: raise ValueError("Chunk size must be positive.")
    return [data[i: i + size] for i in range(0, len(data), size)]