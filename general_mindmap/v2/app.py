from aidial_sdk import DIALApp

from general_mindmap.utils.log_config import configure_loggers
from general_mindmap.v2.completion.app import Mindmap
from general_mindmap.v2.config import DIAL_URL
from general_mindmap.v2.routers import (
    appearances,
    edges,
    generate,
    graph,
    history,
    nodes,
    sources,
)
from general_mindmap.v2.routers.old import appearances as old_appearances
from general_mindmap.v2.routers.old import edges as old_edges
from general_mindmap.v2.routers.old import generate as old_generate
from general_mindmap.v2.routers.old import graph as old_graph
from general_mindmap.v2.routers.old import history as old_history
from general_mindmap.v2.routers.old import nodes as old_nodes
from general_mindmap.v2.routers.old import sources as old_sources
from general_mindmap.v2.utils.header_propagator import HeaderPropagator

GENERATED_TYPE = "Generated"
MANUAL_TYPE = "Manual"

configure_loggers()

app = DIALApp(
    dial_url=DIAL_URL,
    add_healthcheck=True,
)


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

app.include_router(old_edges.router)
app.include_router(old_graph.router)
app.include_router(old_nodes.router)
app.include_router(old_history.router)
app.include_router(old_generate.router)
app.include_router(old_sources.router)
app.include_router(old_appearances.router)
