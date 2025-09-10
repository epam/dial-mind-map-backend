import io
import zipfile
from io import BytesIO
from time import time

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import JSONResponse, StreamingResponse

from dial_rag.app import DIAL_URL
from general_mindmap.v2.dial.client import DialClient
from general_mindmap.v2.models.metadata import (
    HistoryItem,
    HistoryItemType,
    HistoryStep,
    Metadata,
)
from general_mindmap.v2.routers.utils.errors import timeout_after
from general_mindmap.v2.utils.batch_file_reader import (
    BatchFileReader,
    BatchRawFileReader,
)
from general_mindmap.v2.utils.batch_file_writer import BatchFileWriter

router = APIRouter()


@router.get("/mindmaps/{mindmap:path}/appearances/themes/{theme}")
@timeout_after()
async def get_appearances(request: Request, theme: str):
    client = await DialClient.create_with_folder(
        DIAL_URL,
        "auto",
        request.headers["x-mindmap"],
        request.headers.get("etag", ""),
    )

    await client.read_metadata()

    if client._metadata.appearances_file is None:
        raise HTTPException(status_code=404)

    appearances, etag = await client.read_file_by_name_and_etag(
        client._metadata.appearances_file
    )

    if theme not in appearances:
        raise HTTPException(status_code=404)

    return JSONResponse(appearances[theme], headers={"ETag": etag})


@router.post("/mindmaps/{mindmap:path}/appearances/themes/{theme}")
@timeout_after()
async def change_appearances(request: Request, theme: str):
    start_time = time()

    async with await DialClient.create_with_folder(
        DIAL_URL,
        "auto",
        request.headers["x-mindmap"],
        request.headers.get("etag", ""),
    ) as client:
        appearances = {}
        old_file_name = client._metadata.appearances_file
        if old_file_name:
            appearances, _ = await client.read_file_by_name_and_etag(
                old_file_name
            )
        else:
            old_file_name = ""

        appearances[theme] = await request.json()

        new_file_name = client._metadata.appearances_file = (
            f"appearances/{start_time}_config"
        )
        await client.write_file(
            new_file_name,
            appearances,
            request.headers.get("etag", ""),
        )

        client._metadata.history.append(
            client._metadata,
            HistoryStep(
                user="USER",
                changes=[
                    HistoryItem(
                        old_value=old_file_name,
                        new_value=new_file_name,
                        type=HistoryItemType.APPEARANCES,
                    )
                ],
            ),
        )

        return Response(headers={"ETag": await client.close()})


@router.post("/mindmaps/{mindmap:path}/appearances/themes/{theme}/events")
async def subscribe_to_appearances(request: Request, theme: str):
    start_time = time()

    client = await DialClient.create_with_folder(
        DIAL_URL,
        "auto",
        request.headers["x-mindmap"],
        "",
    )

    await client.read_metadata()

    appearances = {}
    if client._metadata.appearances_file:
        appearances, _ = await client.read_file_by_name_and_etag(
            client._metadata.appearances_file, request.headers.get("etag", "")
        )
    else:
        async with client:
            client._metadata.appearances_file = (
                f"appearances/{start_time}_config"
            )
            await client.write_file(
                client._metadata.appearances_file, appearances
            )
            await client.close()

    return StreamingResponse(
        client.subscribe_to_appearances(
            request, appearances, client._metadata.appearances_file, theme
        ),
        media_type="text/event-stream",
    )


@router.post(
    "/mindmaps/{mindmap:path}/appearances/themes/{theme}/storage/{file_name}"
)
@timeout_after()
async def add_file(
    request: Request,
    theme: str,
    file_name: str,
    file: UploadFile = File(),
):
    async with await DialClient.create_with_folder(
        DIAL_URL,
        "auto",
        request.headers["x-mindmap"],
        request.headers.get("etag", ""),
    ) as client:
        await client.write_raw_file(
            f"appearances/themes/{theme}/storage/{file_name}",
            await file.read(),
            content_type=file.content_type,
        )


@router.get(
    "/mindmaps/{mindmap:path}/appearances/themes/{theme}/storage/{file_name}"
)
@timeout_after()
async def get_file(
    request: Request,
    theme: str,
    file_name: str,
):
    client = await DialClient.create_with_folder(
        DIAL_URL,
        "auto",
        request.headers["x-mindmap"],
        request.headers.get("etag", ""),
    )

    return StreamingResponse(
        BytesIO(
            await client.read_raw_file_by_url(
                client.make_url_without_extension(
                    f"appearances/themes/{theme}/storage/{file_name}"
                )
            )
        ),
    )


@router.get("/mindmaps/{mindmap:path}/appearances/export")
@timeout_after()
async def export(
    request: Request,
):
    client = await DialClient.create_with_folder(
        DIAL_URL,
        "auto",
        request.headers["x-mindmap"],
        request.headers.get("etag", ""),
    )

    await client.read_metadata()

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        try:
            themes = (await client.get_files_list("appearances/themes/"))[0]
        except HTTPException as e:
            if e.status_code == 404:
                themes = []
            else:
                raise e

        file_reader = BatchRawFileReader(client)
        file_url_to_path = {}

        if client._metadata.appearances_file:
            full_url_to_file = client.make_url(
                client._metadata.appearances_file
            )
            file_reader.add_file(full_url_to_file)
            file_url_to_path[full_url_to_file] = "config.json"
        else:
            zipf.writestr("config.json", "{}")

        for theme in themes:
            files = (
                await client.get_files_list(
                    f"appearances/themes/{theme['name']}/storage/"
                )
            )[0]

            for file in files:
                full_url_to_file = f"{DIAL_URL}/v1/{file['url']}"

                file_reader.add_file(full_url_to_file)
                file_url_to_path[full_url_to_file] = (
                    f"themes/{theme['name']}/storage/{file['name']}"
                )

        for result in await file_reader.read():
            zipf.writestr(file_url_to_path[result[0]], result[1])

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=appearances.zip"},
    )


@router.post("/mindmaps/{mindmap:path}/appearances")
@timeout_after()
async def import_file(
    request: Request,
    file: UploadFile,
):
    current_time = time()

    async with await DialClient.create_with_folder(
        DIAL_URL,
        "auto",
        request.headers["x-mindmap"],
        request.headers.get("etag", ""),
    ) as client:
        zip_buffer = io.BytesIO(await file.read())

        file_writer = BatchFileWriter(client)

        old_file_name = client._metadata.appearances_file
        new_file_name = ""
        with zipfile.ZipFile(zip_buffer, "r") as zipf:
            for name in zipf.namelist():
                if name == "config.json":
                    new_file_name = client._metadata.appearances_file = (
                        f"appearances/{current_time}_config"
                    )
                    file_writer.add_raw_file(
                        f"{new_file_name}.json",
                        zipf.read(name),
                    )
                else:
                    file_writer.add_raw_file(
                        f"appearances/{name}", zipf.read(name)
                    )

        await file_writer.write()

        client._metadata.history.append(
            client._metadata,
            HistoryStep(
                user="USER",
                changes=[
                    HistoryItem(
                        old_value=old_file_name,
                        new_value=new_file_name,
                        type=HistoryItemType.APPEARANCES,
                    )
                ],
            ),
        )

        return Response(headers={"ETag": await client.close()})
