import logging
import re
from typing import Dict, Set

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from generator.adapter import GMContract as Gmc
from generator.chainer import ChainCreator as Cc
from generator.chainer import ChainRunner as Cr
from generator.chainer import ChainTypes as Ct
from generator.chainer.model_handler import ModelCreator
from generator.chainer.prompt_generator.constants import PromptInputs as Pi
from generator.chainer.response_formats import (
    RefinedStyleResult,
    RefinedStyleResultProtocol,
)
from generator.common.constants import DataFrameCols as Col
from generator.common.constants import FieldNames as Fn
from generator.common.interfaces import FileStorage
from generator.common.structs import MMRequest
from generator.core.actions.docs import fetch_all_docs_content
from generator.core.stages import DocHandler
from generator.core.stages.doc_handler.constants import DocCategories as DocCat
from generator.core.utils.constants import DEFAULT_FILTER, DEFAULT_STYLE

MULTIMODAL_CATEGORIES: Set[str] = {
    DocCat.PDF_AS_A_WHOLE,
    DocCat.PPTX_AS_A_WHOLE,
}
TEXT_CATEGORIES: Set[str] = {
    DocCat.TXT_AS_A_WHOLE,
    DocCat.HTML_AS_A_WHOLE,
    DocCat.LINK_AS_A_WHOLE,
}


class DominantLanguageResponse(BaseModel):
    """Defines the structured output for language detection."""

    dominant_language: str = Field(
        description="The single dominant language identified from the text collection, e.g., 'English', 'Spanish'."
    )


async def _extract_text_from_documents(
    request: MMRequest, file_storage: FileStorage
) -> Dict[str, str]:
    """
    Fetches and processes documents to extract plain text content for each one.

    This function reuses the document processing pipeline but strips away all
    formatting, tags, and non-text elements.

    Returns:
        A dictionary mapping each document ID to its full, plain text content.
    """

    # Step 1: Reuse existing fetching and processing logic
    doc_handler = DocHandler(strategy="whole")
    if getattr(request, "del_documents", None):
        documents = [
            doc
            for doc in request.documents
            if doc.id not in [doc.id for doc in request.del_documents]
        ]
    else:
        documents = request.documents

    if not documents:
        return {}

    docs_and_their_content = await fetch_all_docs_content(
        documents, file_storage
    )
    # The chunker is responsible for the initial text/image extraction
    chunked_docs_df, _ = await doc_handler.chunk_docs(docs_and_their_content)

    if chunked_docs_df.empty:
        return {}

    # Step 2: Iterate through processed docs and aggregate only text
    extracted_texts: Dict[str, str] = {}
    processed_doc_ids = set()

    for _, row in chunked_docs_df.iterrows():
        doc_id = row[Col.DOC_ID]
        if doc_id in processed_doc_ids:
            continue

        processed_doc_ids.add(doc_id)
        doc_cat = row[Col.DOC_CAT]
        content_from_df = row[Col.CONTENT]

        # Skip unsupported docs or docs with no content
        if doc_cat == DocCat.UNSUPPORTED or not content_from_df:
            continue

        text_parts = []

        # Handle text-based documents (HTML, TXT, etc.)
        if doc_cat in TEXT_CATEGORIES:
            # The 'content' column for text docs should already be a string
            if isinstance(content_from_df, str):
                text_parts.append(content_from_df)

        # Handle multimodal documents (PDF, PPTX)
        elif doc_cat in MULTIMODAL_CATEGORIES:
            # The 'content' is a list of pages/slides
            if isinstance(content_from_df, list):
                for page_or_slide in content_from_df:
                    # Each page/slide has a 'texts' key with a list of strings
                    page_texts = page_or_slide.get("texts", [])
                    if page_texts:
                        text_parts.extend(page_texts)

        if text_parts:
            # Join all collected text parts into a single string for the document
            extracted_texts[doc_id] = "\n".join(text_parts)

    return extracted_texts


def _clean_text_for_language_detection(text: str) -> str:
    """
    Cleans text by removing URLs and Markdown links to improve language detection accuracy.
    """
    if not text:
        return ""

    # Pattern 1: Remove Markdown-style links, e.g., [link text](url)
    # This removes the entire construct.
    text = re.sub(r"\[.*?]\(.*?\)", "", text)

    # Pattern 2: Remove standalone URLs (http, https, www)
    # This catches any URLs that might not be in Markdown format.
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Optional: Clean up extra whitespace and newlines left after removal
    text = re.sub(
        r"\s{2,}", " ", text
    )  # Replace multiple spaces with a single space
    text = re.sub(
        r"\n{2,}", "\n", text
    )  # Replace multiple newlines with a single one

    return text.strip()


def _create_language_batch_id_chain():
    """
    Creates a LangChain chain to identify the dominant language from a
    batch of texts, structured to output a Pydantic model.
    """
    model = ModelCreator.get_chat_model()

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a language detection expert. Your task is to identify the dominant language from a collection of text samples.
You will be given a series of text samples, each enclosed in a <document> tag with a unique ID.
Analyze all documents to determine the single dominant language across the entire collection by considering the language of the majority of the text.""",
            ),
            (
                "human",
                "Analyze the following text samples and determine the dominant language.\n\n{batch_text}",
            ),
        ]
    )

    structured_llm = model.with_structured_output(DominantLanguageResponse)

    chain = prompt_template | structured_llm
    return chain


async def _determine_dominant_language_single_call(
    texts_by_doc_id: Dict[str, str],
) -> str:
    """
    Determines the dominant language from multiple document texts
    using a single, batched LLM call with proportional sampling to
    respect document size.
    """
    if not texts_by_doc_id:
        return "English"

    # Clean texts and filter out empty ones
    cleaned_texts = {}
    total_chars = 0
    for doc_id, text in texts_by_doc_id.items():
        if text and text.strip():
            cleaned_text = _clean_text_for_language_detection(text)
            if cleaned_text:
                cleaned_texts[doc_id] = cleaned_text
                total_chars += len(cleaned_text)

    if not cleaned_texts:
        return "English"

    # Define a total budget for the prompt to avoid excessive
    # length/cost
    total_prompt_budget = 15000
    prompt_parts = []

    for doc_id, text in cleaned_texts.items():
        doc_length = len(text)
        # Calculate the proportional sample size for this document
        proportional_sample_size = int(
            (doc_length / total_chars) * total_prompt_budget
        )

        # Ensure a minimum sample size for very small docs in a large
        # collection and a maximum cap equal to the document's actual
        # length.
        sample_size = min(doc_length, max(100, proportional_sample_size))

        sample = text[:sample_size]
        prompt_parts.append(f'<document id="{doc_id}">\n{sample}\n</document>')

    batch_text_prompt = "\n\n".join(prompt_parts)
    language_id_chain = _create_language_batch_id_chain()

    try:
        response_obj = await language_id_chain.ainvoke(
            {"batch_text": batch_text_prompt}
        )
        if response_obj and isinstance(response_obj, DominantLanguageResponse):
            lang = response_obj.dominant_language
            if lang and isinstance(lang, str):
                return lang.strip().capitalize()

        logging.warning(
            "LLM did not return a valid DominantLanguageResponse object."
        )
        return "English"

    except Exception as e:
        logging.warning(f"Language detection via single-call LLM failed: {e}")
        return "English"


def _format_directives_as_string(directives: RefinedStyleResultProtocol) -> str:
    """
    Formats StylingDirectives into a single, human-readable string.
    """
    output_lines: list[str] = []
    directive_values = directives.model_dump()

    fields_to_format = {
        Fn.LANGUAGE: "Language",
        Fn.PERSONA: "Persona",
        Fn.TONE: "Tone",
        Fn.OTHER_INSTRUCTIONS: "Other Instructions",
    }

    for field_name, label in fields_to_format.items():
        value = directive_values.get(field_name)
        # Add the line only if the value is present
        if value and str(value).strip():
            output_lines.append(f"{label}: {value}")

    return "\n".join(output_lines)


async def process_user_style(
    request: MMRequest, file_storage: FileStorage
) -> dict[str, str]:
    """
    Processes style instructions from a request, validates or refines
    the style prompt, and returns the final, safe style prompt.

    -- THIS FUNCTION NOW CORRECTLY USES THE REVISED SINGLE-CALL METHOD --
    """
    style_section = getattr(request, Gmc.STYLE_SECTION, DEFAULT_STYLE)
    user_prompt = style_section.get(Gmc.STYLE_PROMPT_FIELD, "")
    is_final = style_section.get(Gmc.IS_FINAL_FIELD, False)

    if not user_prompt.strip():
        # 1. Extract clean text from documents
        texts_by_doc_id = await _extract_text_from_documents(
            request, file_storage
        )

        # 2. Determine the dominant language
        dominant_language = await _determine_dominant_language_single_call(
            texts_by_doc_id
        )

        # 3. Use the result to dynamically configure your style
        request_default_directives = RefinedStyleResult(
            validation={"is_safe": True, "reason": "Default for request."},
            language=dominant_language,
            persona="Third-person neutral",
            tone="Formal",
            other_instructions="",
        )
        default_style_prompt = _format_directives_as_string(
            request_default_directives
        )
        return {Pi.STYLE: default_style_prompt}

    # 1. Select and run the appropriate processing chain
    chain_type = Ct.VALIDATE_STYLE if is_final else Ct.REFINE_STYLE
    chain = Cc().choose_chain(chain_type)
    chain_inputs = {Pi.QUERY: user_prompt}
    result = await Cr().run_chain_w_retries(chain, chain_inputs)

    # 2. Normalize the result from either chain
    if is_final:
        # Validation chain returns a result with 'is_safe' and 'reason'
        is_safe = result.is_safe
        reason_for_failure = result.reason
        result_prompt = user_prompt
    else:
        validation_details = result.validation

        is_safe = validation_details.is_safe
        reason_for_failure = validation_details.reason
        result_prompt = _format_directives_as_string(result)

    # 3. Return the successful prompt or raise an error
    if not is_safe:
        raise ValueError(reason_for_failure)

    return {Pi.STYLE: result_prompt}


async def process_user_filter(request: MMRequest) -> dict[str, str]:
    """
    Processes filter instructions from a request, validates or refines
    the prompt, and returns the final prompt.
    """
    # 1. Extract data from the request
    filter_section = getattr(request, Gmc.FILTER_SECTION, DEFAULT_FILTER)
    user_prompt = filter_section.get(Gmc.FILTER_PROMPT_FIELD, "")
    is_final = filter_section.get(Gmc.IS_FINAL_FIELD, False)

    if not user_prompt.strip():
        return {Pi.FILTER: ""}

    # 2. Select and run the appropriate processing chain
    chain_type = Ct.VALIDATE_FILTER if is_final else Ct.REFINE_FILTER
    chain = Cc().choose_chain(chain_type)
    chain_inputs = {Pi.QUERY: user_prompt}
    result = await Cr().run_chain_w_retries(chain, chain_inputs)

    # 3. Normalize the result from either chain
    if is_final:
        # Validation chain returns a result with 'is_safe' and 'reason'
        is_safe = result.is_safe
        reason_for_failure = result.reason
        result_prompt = user_prompt
    else:
        # Refinement chain returns a result with a nested validation
        # object and a 'refined_query'
        validation_details = result.validation

        is_safe = validation_details.is_safe
        reason_for_failure = validation_details.reason
        result_prompt = result.refined_query

    # 4. Return the successful prompt or raise an error
    if not is_safe:
        raise ValueError(reason_for_failure)

    return {Pi.FILTER: result_prompt}
