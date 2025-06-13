from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from general_mindmap.v2.config import DIAL_URL
from general_mindmap.v2.dial.client import DialClient
from general_mindmap.v2.routers.history import read_graph
from general_mindmap.v2.routers.utils.errors import timeout_after

router = APIRouter()


@router.get("/mindmaps/{mindmap:path}/graph")
@timeout_after()
async def get_graph(request: Request, metainfo: str = "True"):
    metainfo_flag = metainfo.lower() == "true"

    client = await DialClient.create_with_folder(
        DIAL_URL,
        "auto",
        request.headers["x-mindmap"],
        request.headers.get("etag", ""),
    )

    await client.read_metadata()

    graph = await read_graph(client)

    for node in graph["nodes"]:
        node = node["data"]

        if "question" in node:
            if "questions" not in node or not node["questions"]:
                node["questions"] = [node["question"]]
            del node["question"]

    if not metainfo_flag:
        for node in graph["nodes"]:
            node = node["data"]

            if "question" in node:
                del node["question"]
            if "details" in node:
                del node["details"]
            if "link" in node:
                del node["link"]

    return JSONResponse(content=graph, headers={"ETag": client._etag})


@router.post("/mindmaps/{mindmap:path}/subscribe")
async def subscribe(request: Request, mindmap: str):
    client = await DialClient.create_with_folder(
        DIAL_URL if DIAL_URL else "",
        request.headers.get("authorization", "-"),
        request.headers["x-mindmap"],
        "",
    )

    return StreamingResponse(
        client.subscribe(request), media_type="text/event-stream"
    )
