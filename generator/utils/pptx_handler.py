import base64
import io
from typing import Iterator

from PIL import Image as PIL_Image
from pptx import Presentation
from pptx.parts.image import Image
from pptx.slide import Slide

from general_mindmap.v2.dial.client import DialClient
from generator.chainer.model_handler import ModelUtils
from generator.utils.dial import read_file_bytes_from_dial


class PPTXHandler:
    @staticmethod
    async def load_presentation(
        dial_client: DialClient, doc_url: str
    ) -> Presentation:
        doc_bytes = await read_file_bytes_from_dial(dial_client, doc_url)
        return Presentation(io.BytesIO(doc_bytes))

    @staticmethod
    def get_visible_slides(presentation: Presentation) -> Iterator[Slide]:
        return (
            slide
            for slide in presentation.slides
            if slide.element.get("show", None) != "0"
        )

    @staticmethod
    def process_image(image: Image) -> str | None:
        img_bytes_io = io.BytesIO(image.blob)
        with PIL_Image.open(img_bytes_io) as img:
            if img.format == "WMF":  # WMF is not supported by OpenAI
                return None

        img_bytes_io.seek(0)
        return base64.b64encode(img_bytes_io.read()).decode()

    @staticmethod
    def calculate_image_tokens(base64_str: str) -> int:
        img_bytes = base64.b64decode(base64_str)
        with PIL_Image.open(io.BytesIO(img_bytes)) as img:
            width, height = img.size
        return ModelUtils.calculate_img_tokens(width, height)
