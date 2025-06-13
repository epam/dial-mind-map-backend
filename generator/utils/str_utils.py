import re

from generator.utils.constants import Patterns as Pat


def is_trivial_content(text: str) -> bool:
    """
    Check if the content is trivial (just punctuation or very short)
    """
    # Remove all punctuation and check if there's anything left
    no_punct = re.sub(Pat.NO_PUNCT, "", text)
    if not no_punct.strip():
        return True

    # Very short content that's likely not meaningful
    if len(text.strip()) <= 1:
        return True

    return False
