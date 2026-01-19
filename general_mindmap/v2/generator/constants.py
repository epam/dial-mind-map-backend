from typing import Set

from . import prompts

# --- Document Categories ---

MULTIMODAL_CATEGORIES: Set[str] = {
    "pdf_as_a_whole",
    "pptx_as_a_whole",
}
TEXT_CATEGORIES: Set[str] = {
    "txt_as_a_whole",
    "html_as_a_whole",
    "link_as_a_whole",
}

# --- Define Prompt Maps ---

STRUCTURE_PROMPT_MAP = {
    "gemini-2.5-pro": prompts.STRUCTURE_SYSTEM_PROMPT_GEMINI_2_5_PRO,
    "gpt-5": prompts.STRUCTURE_SYSTEM_PROMPT_GPT_5_LR,
    "gpt-4.1": prompts.STRUCTURE_SYSTEM_PROMPT_GPT_4_1,
    "default": prompts.UNIVERSAL_STRUCTURE_SYSTEM_PROMPT,
}

QA_PROMPT_MAP = {
    "gemini-2.5-pro": prompts.QUESTION_ANSWERING_PROMPT_GEMINI_2_5_PRO,
    "gpt-5": prompts.QUESTION_ANSWERING_PROMPT_GPT_5_LR,
    "gpt-4.1": prompts.QUESTION_ANSWERING_PROMPT_GPT_4_1,
    "default": prompts.UNIVERSAL_QUESTION_ANSWERING_PROMPT,
}

OLD_PROMPT_MAP = {
    "structure": prompts.OLD_STRUCTURE_SYSTEM_PROMPT,
    "qa": prompts.OLD_QUESTION_ANSWERING_PROMPT,
}

# --- Other Constants ---

IMAGE_LIMIT = 50
