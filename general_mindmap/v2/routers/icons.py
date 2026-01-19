from io import BytesIO

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import StreamingResponse

from general_mindmap.v2.config import DIAL_URL
from general_mindmap.v2.dial.client import DialClient
from general_mindmap.v2.routers.utils.errors import timeout_after

router = APIRouter()


@router.put("/v1/icons/{icon_path:path}")
@timeout_after()
async def add_file(
    request: Request,
    icon_path: str,
    file: UploadFile = File(),
):
    async with await DialClient.create(DIAL_URL, request) as client:
        await client.write_raw_file(
            f"icons/{icon_path}",
            await file.read(),
            content_type=file.content_type,
        )


@router.get("/v1/icons/{icon_path:path}")
@timeout_after()
async def get_file(
    request: Request,
    icon_path: str,
):
    client = await DialClient.create(DIAL_URL, request)

    return StreamingResponse(
        BytesIO(
            await client.read_raw_file_by_url(
                client.make_url_without_extension(f"icons/{icon_path}")
            )
        ),
    )
