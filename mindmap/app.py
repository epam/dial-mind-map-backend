from aidial_sdk import DIALApp

from mindmap.routers.utils.request_id import ContextMiddleware
from mindmap.utils.logger_config import configure_loggers

configure_loggers()

from mindmap.completion.app import Mindmap  # noqa: E402
from mindmap.config import DIAL_URL  # noqa: E402
from mindmap.dial.header_propagator import HeaderPropagator  # noqa: E402
from mindmap.routers import (  # noqa: E402
    appearances,
    edges,
    generate,
    graph,
    history,
    icons,
    nodes,
    sources,
)

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
