import pandas as pd

from generator.chainer.prompt_generator.constants import PromptInputs as Pi
from generator.common.constants import DataFrameCols as Col
from generator.core.stages.doc_handler.constants import DocCategories


def form_text_inputs(
    text_chunks_df: pd.DataFrame,
    extraction_instructions: dict[str, str | dict[str, str]],
) -> list:
    inputs = []
    for _, row in text_chunks_df.iterrows():
        content = row[Col.CONTENT]
        doc_desc = row.get(Col.DOC_DESC)

        chunk_instructions = extraction_instructions.copy()
        if doc_desc:
            chunk_instructions[Pi.DOC_DESC] = doc_desc

        inputs.append({Pi.TEXTUAL_EXTRACTION: content, **chunk_instructions})
    return inputs


def form_multimodal_inputs(
    multimodal_chunks_df: pd.DataFrame,
    cat: str,
    extraction_instructions: dict[str, str | dict[str, str]],
    include_page_numbers: bool = True,
) -> list:
    """
    Forms a list of multimodal inputs for processing, with an option
    to include page/slide numbers.

    Args:
        multimodal_chunks_df (pd.DataFrame): DataFrame containing the
            chunks.
        cat (str): The document category (e.g., PPTX, PDF).
        extraction_instructions (dict): Instructions for the model.
        include_page_numbers (bool): If True, prepends a "Page X" or
            "Slide X" header to the text of each page/slide.
            Defaults to True.

    Returns:
        list: A list of formatted multimodal inputs ready for the model.
    """
    if cat == DocCategories.PPTX:
        page_pref = "Slide"
    else:
        page_pref = "Page"

    multimodal_inputs = []
    for _, row in multimodal_chunks_df.iterrows():
        file_content = row[Col.CONTENT]
        doc_desc = row.get(Col.DOC_DESC)

        multimodal_content = []
        for slide in file_content:
            slide_text = "\n".join(slide.get("texts"))

            if include_page_numbers:
                final_text = (
                    f"==={page_pref} {slide.get('page_id')}===:\n\n"
                    f"{slide_text}"
                )
            else:
                final_text = slide_text

            slide_text_part = {
                "type": "text",
                "text": final_text,
            }
            multimodal_content.append(slide_text_part)
            slide_img_part = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    },
                }
                for image_data in slide.get("images")
            ]
            multimodal_content.extend(slide_img_part)

        # Create specific instructions for this chunk
        chunk_instructions = extraction_instructions.copy()
        if doc_desc:
            # Add the document description to the instructions.
            # The LLM prompt template should be designed to use this key.
            chunk_instructions[Pi.DOC_DESC] = doc_desc

        multimodal_inputs.append(
            {
                Pi.MULTIMODAL_CONTENT: multimodal_content,
                **chunk_instructions,
            }
        )
    return multimodal_inputs
