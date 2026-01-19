from aidial_sdk import DIALApp

from common_utils.logger_config import configure_loggers
from general_mindmap.v2.routers.utils.request_id import ContextMiddleware

configure_loggers()

from general_mindmap.v2.completion.app import Mindmap
from general_mindmap.v2.config import DIAL_URL
from general_mindmap.v2.routers import (
    appearances,
    edges,
    generate,
    graph,
    history,
    icons,
    nodes,
    sources,
)
from general_mindmap.v2.utils.header_propagator import HeaderPropagator

GENERATED_TYPE = "Generated"
MANUAL_TYPE = "Manual"

app = DIALApp(
    dial_url=DIAL_URL,
    add_healthcheck=True,
)

app.add_middleware(ContextMiddleware)

HeaderPropagator(app, DIAL_URL).enable()

app.add_api_route(
    "/openai/deployments/{deployment_name:path}/chat/completions",
    app._chat_completion(
        "mindmap",
        Mindmap(DIAL_URL or ""),
        heartbeat_interval=None,
    ),
    methods=["POST"],
)

app.include_router(edges.router)
app.include_router(graph.router)
app.include_router(nodes.router)
app.include_router(history.router)
app.include_router(generate.router)
app.include_router(sources.router)
app.include_router(appearances.router)
app.include_router(icons.router)
