import asyncio
import os
import subprocess
import tempfile
from time import sleep
from typing import List

import pdf2image
import PIL
import PIL.Image
import psutil


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


async def convert_pptx_to_pdf(pptx: bytes) -> bytes:
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


async def convert_pptx_to_images(pptx: bytes) -> List[PIL.Image.Image]:
    pdf = await convert_pptx_to_pdf(pptx)

    return convert_pdf_to_images(pdf)


def convert_pdf_to_images(pdf: bytes) -> List[PIL.Image.Image]:
    return pdf2image.convert_from_bytes(pdf)
