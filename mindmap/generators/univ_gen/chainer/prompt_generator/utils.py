from pathlib import Path

import yaml

from .constants import PROMPT_PARTS_DIR, READY_PTS_DIR, YML_EXTENSION


def save_composed_prompt(
    handler_name: str,
    full_template: str,
    output_subdir: str = READY_PTS_DIR,
) -> None:
    """
    Saves a composed prompt and its source components to a YAML file
    for debugging.

    Args:
        handler_name: The name of the handler class, used for the
            filename and metadata.
        full_template: The final, fully composed prompt string.
        output_subdir: The name of the subdirectory to save the file in.

    Returns:
        The Path object of the newly created file.
    """
    output_dir = Path(__file__).resolve().parent / output_subdir
    output_dir.mkdir(exist_ok=True)

    file_path = output_dir / f"{handler_name}{YML_EXTENSION}"

    output_data = {"full_template": full_template}

    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(
            output_data,
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
            default_style="|",
            width=80,
        )


def load_prompt_part(part_type: str, part_key: str) -> str | dict[str, str]:
    """
    Loads a specific prompt component from a YAML file.

    Args:
        part_type: The name of the YAML file.
        part_key: The key of the prompt component to load.

    Returns:
        A prompt string
    """
    prompt_part_dir = Path(__file__).resolve().parent / PROMPT_PARTS_DIR
    file_path = prompt_part_dir / (part_type + YML_EXTENSION)
    with open(file_path, "r") as f:
        all_prompts = yaml.safe_load(f)

    prompt_data = all_prompts.get(part_key)

    if prompt_data is None:
        raise KeyError(
            f"Prompt part with key '{part_key}' not found in {file_path}"
        )

    return prompt_data
