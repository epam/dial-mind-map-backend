import base64
import io
import logging
from typing import Any, Dict, List

from PIL import Image

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def merge_base64_images(b64_image1: str, b64_image2: str) -> str:
    """Merges two base64 encoded images vertically."""
    img1_data = base64.b64decode(b64_image1)
    img2_data = base64.b64decode(b64_image2)
    img1 = Image.open(io.BytesIO(img1_data)).convert("RGB")
    img2 = Image.open(io.BytesIO(img2_data)).convert("RGB")
    new_width = max(img1.width, img2.width)
    new_height = img1.height + img2.height
    new_img = Image.new("RGB", (new_width, new_height), (255, 255, 255))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (0, img1.height))
    buffer = io.BytesIO()
    new_img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def enforce_image_limit(
    content: List[Dict[str, Any]],
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Ensures image count is within the limit by performing balanced, paired
    merges of consecutive images.
    """
    # 1. Find all images and their original indices
    image_locations = [
        (i, item)
        for i, item in enumerate(content)
        if item.get("type") == "image_url"
    ]

    images_to_remove = len(image_locations) - limit
    if images_to_remove <= 0:
        # If we are within the limit, just clean up metadata and return
        for item in content:
            if "metadata" in item:
                del item["metadata"]
        return content

    logger.info(
        f"Image count ({len(image_locations)}) exceeds limit ({limit}). "
        f"Performing {images_to_remove} paired merges."
    )

    # 2. Plan all replacements and deletions up front
    replacements = {}
    deletions = set()

    for i in range(images_to_remove):
        # Identify the next available pair to merge
        pair_idx1 = 2 * i
        pair_idx2 = 2 * i + 1

        if pair_idx2 >= len(image_locations):
            break  # Stop if we run out of pairs

        idx1, item1 = image_locations[pair_idx1]
        idx2, item2 = image_locations[pair_idx2]

        # Ensure we don't try to modify the same index twice
        if (
            idx1 in replacements
            or idx1 in deletions
            or idx2 in replacements
            or idx2 in deletions
        ):
            continue

        page1 = item1["metadata"]["source_chunk_id"]
        page2 = item2["metadata"]["source_chunk_id"]

        # Merge the image data
        b64_img1 = item1["image_url"]["url"].split(",", 1)[1]
        b64_img2 = item2["image_url"]["url"].split(",", 1)[1]
        merged_b64 = merge_base64_images(b64_img1, b64_img2)

        # Create the notification text and the new merged image item
        notification_text = (
            f"\n\n> **Note:** To meet processing limits, the image for page/chunk `{page1}` "
            f"has been merged with the one from page/chunk `{page2}`. The combined "
            f"image is shown below."
        )
        notification_item = {"type": "text", "text": notification_text}

        merged_item = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{merged_b64}"},
            "metadata": {
                "source_chunk_id": page1,
                "merged_pages": [page1, page2],
            },
        }

        # Plan the operations: replace the first image, delete the second
        replacements[idx1] = [merged_item, notification_item]
        deletions.add(idx2)

    # 3. Execute the plan by building a new content list
    new_content = []
    for i, item in enumerate(content):
        if i in deletions:
            continue  # Skip items marked for deletion
        if i in replacements:
            # Add the replacement content. Note: We place the merged image
            # first, then the note, to keep it close to its original chunk.
            new_content.extend(replacements[i])
        else:
            new_content.append(item)

    # 4. Final cleanup of all temporary metadata before returning
    final_content = new_content
    for item in final_content:
        if "metadata" in item:
            del item["metadata"]

    return final_content
