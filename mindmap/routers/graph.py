from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from mindmap.config import DIAL_URL
from mindmap.dial.client import DialClient
from mindmap.routers.history import read_graph
from mindmap.routers.utils.errors import timeout_after

router = APIRouter()


@router.get("/v1/graph")
@timeout_after()
async def get_graph(request: Request, metainfo: str = "True"):
    metainfo_flag = metainfo.lower() == "true"

    client = await DialClient.create(DIAL_URL, request)

    await client.read_metadata()

    graph = await read_graph(client)

    for node in graph["nodes"]:
        if "questions" not in node["data"] and "question" in node["data"]:
            node["data"]["questions"] = [node["data"]["question"]]

        if "question" in "node":
            del node["data"]["question"]

    if not metainfo_flag:
        for node in graph["nodes"]:
            node = node["data"]

            if "questions" in node:
                del node["questions"]
            if "details" in node:
                del node["details"]
            if "link" in node:
                del node["link"]

    return JSONResponse(content=graph, headers={"ETag": client._etag})


@router.get("/v1/subscribe")
async def subscribe(request: Request):
    client = await DialClient.create(DIAL_URL, request)

    return StreamingResponse(
        client.subscribe(request), media_type="text/event-stream"
    )
