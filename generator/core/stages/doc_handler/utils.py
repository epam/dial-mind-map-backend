import base64
import binascii
import io
import logging
import os
import subprocess
import tempfile
from time import sleep
from typing import Iterator, List

import pdf2image
import PIL.Image
import psutil
from PIL import Image as PIL_Image
from pptx import Presentation
from pptx.parts.image import Image
from pptx.presentation import Presentation as IPresentation
from pptx.slide import Slide

from common_utils.logger_config import logger
from generator.chainer.model_handler import LLMUtils


def extract_doc_title(doc_url: str) -> str:
    filename = os.path.basename(doc_url)
    return os.path.splitext(filename)[0]


def calculate_image_tokens(base64_str: str) -> int:
    """
    Calculates the token cost for an image given its base64 representation.
    Handles both raw base64 strings and data URIs.
    """
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]

    # Fix for incorrect padding
    missing_padding = len(base64_str) % 4
    if missing_padding:
        base64_str += "=" * (4 - missing_padding)

    try:
        img_bytes = base64.b64decode(base64_str)
        with PIL_Image.open(io.BytesIO(img_bytes)) as img:
            width, height = img.size
        return LLMUtils.calculate_img_tokens(width, height)
    # Catch binascii.Error directly instead of through base64
    except (binascii.Error, IOError) as e:
        logger.error(f"Could not process image for token calculation: {e}")
        return 0  # Return 0 tokens if the image is invalid


def load_presentation_from_bytes(content: bytes) -> IPresentation:
    """
    Loads a Presentation object from in-memory bytes.

    This method is synchronous as it only performs an in-memory
    operation.
    """
    return Presentation(io.BytesIO(content))


def get_visible_slides(presentation: IPresentation) -> Iterator[Slide]:
    return (
        slide
        for slide in presentation.slides
        if slide.element.get("show", None) != "0"
    )


def process_image(image: Image) -> str | None:
    img_bytes_io = io.BytesIO(image.blob)
    with PIL_Image.open(img_bytes_io) as img:
        if img.format == "WMF":  # WMF is not supported by OpenAI
            return None

    img_bytes_io.seek(0)
    return base64.b64encode(img_bytes_io.read()).decode()


async def _convert_pptx_to_pdf(pptx: bytes) -> bytes:
    with tempfile.TemporaryDirectory() as tmpdirname:
        input_path = os.path.join(tmpdirname, "input.pptx")
        output_path = os.path.join(tmpdirname, "input.pdf")

        with open(input_path, "wb") as f:
            f.write(pptx)

        command = [
            "soffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            tmpdirname,
            input_path,
        ]

        # only one soffice process can be ran
        wait_time = 0
        sleep_time = 0.1
        output = subprocess.run(command, capture_output=True)
        message = output.stdout.decode().strip()
        # we can't rely on returncode unfortunately because on macOS it would return 0 even when the
        # command failed to run; instead we have to rely on the stdout being empty as a sign of the
        # process failed
        while (wait_time < 60) and (message == ""):
            wait_time += sleep_time
            if _is_soffice_running():
                sleep(sleep_time)
            else:
                output = subprocess.run(command, capture_output=True)
                message = output.stdout.decode().strip()

        with open(output_path, "rb") as f:
            pdf_bytes = f.read()

    return pdf_bytes


def _is_soffice_running():
    for proc in psutil.process_iter():
        try:
            if "soffice" in proc.name().lower():
                return True
        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            psutil.ZombieProcess,
        ):
            pass
    return False


def _convert_pdf_to_images(pdf: bytes) -> List[PIL.Image.Image]:
    return pdf2image.convert_from_bytes(pdf)


async def convert_pptx_to_images(pptx: bytes) -> List[PIL.Image.Image]:
    pdf = await _convert_pptx_to_pdf(pptx)

    return _convert_pdf_to_images(pdf)
