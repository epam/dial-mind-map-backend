import asyncio
import time
from uuid import uuid4

import streamlit as st
import streamlit.components.v1 as components

from config import MIND_MAP_FRONTEND_URL
from dial_api import add_source, create_app, export, get_graph, import_state, read_state
from dial_api import run_generation as run_mind_map_generation
from dial_api import set_params, write_state

SVG_LOADER = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200"><circle fill="#3D0EFF" stroke="#3D0EFF" stroke-width="3" r="15" cx="40" cy="100"><animate attributeName="opacity" calcMode="spline" dur="2" values="1;0;1;" keySplines=".5 0 .5 1;.5 0 .5 1" repeatCount="indefinite" begin="-.4"></animate></circle><circle fill="#3D0EFF" stroke="#3D0EFF" stroke-width="3" r="15" cx="100" cy="100"><animate attributeName="opacity" calcMode="spline" dur="2" values="1;0;1;" keySplines=".5 0 .5 1;.5 0 .5 1" repeatCount="indefinite" begin="-.2"></animate></circle><circle fill="#3D0EFF" stroke="#3D0EFF" stroke-width="3" r="15" cx="160" cy="100"><animate attributeName="opacity" calcMode="spline" dur="2" values="1;0;1;" keySplines=".5 0 .5 1;.5 0 .5 1" repeatCount="indefinite" begin="0"></animate></circle></svg>'

state, etag = read_state()


@st.dialog("Confirmation")
def delete_confirm_dialog(id: int):
    cols = st.columns([1, 1])

    if cols[0].button("Cancel"):
        st.rerun()

    if cols[1].button("Delete", type="primary"):
        state["runs"] = [run for run in state["runs"] if run["run_id"] != id]

        write_state(state, etag)
        st.rerun()


async def run_generation(
    sem: asyncio.Semaphore,
    run_id: str,
    start_time: float,
    template: bytes,
    run_name: str,
) -> None:
    async with sem:
        app_id = await create_app()

        await import_state(app_id, template)
        generation_task = asyncio.create_task(run_mind_map_generation(app_id))

        while True:
            state, etag = read_state()

            if "runs" not in state:
                state["runs"] = []
            state["runs"].append(
                {
                    "run_id": run_id,
                    "start_time": start_time,
                    "app_id": app_id,
                    "run_name": run_name,
                }
            )

            if write_state(state, etag) != 412:
                break

        await generation_task


async def run_generations(
    number_runs: int,
    model: str,
    prompt: str,
    rag_prompt: str,
    source_bytes: bytes,
    run_name: str,
):
    run_id = str(uuid4())
    start_time = time.time()

    template_app_id = await create_app()
    await add_source(template_app_id, source_bytes)
    await set_params(template_app_id, model, prompt, rag_prompt)
    template = await export(template_app_id)

    sem = asyncio.Semaphore(5)

    tasks = [
        run_generation(sem, run_id, start_time, template, run_name)
        for _ in range(number_runs)
    ]
    await asyncio.gather(*tasks)


with st.container(border=True):
    st.subheader("New Run")

    run_name = st.text_input("Name")

    settings = st.columns([1, 1], vertical_alignment="top")

    number_runs = settings[0].number_input("Number runs", value=1, min_value=1)
    model = settings[0].text_input("Model", "gpt-5-2025-08-07")
    source = settings[1].file_uploader("Choose a source", type=["pdf"])

    prompts = st.columns([1, 1])

    prompt = prompts[0].text_area(
        "Prompt",
    )
    rag_prompt = prompts[1].text_area(
        "Rag Prompt",
    )

    if st.button("Run"):
        state["run"] = True
        asyncio.run(
            run_generations(
                number_runs, model, prompt, rag_prompt, source.getvalue(), run_name
            )
        )


async def load_graphs():
    tasks = [get_graph(run["app_id"]) for run in state["runs"]]
    results = await asyncio.gather(*tasks)
    return {run["app_id"]: graph for run, graph in zip(state["runs"], results)}


if state.get("run"):
    runs_map = {}
    run_id_to_run_name = {}
    for run in reversed(state["runs"]):
        run_id_to_run_name[run["run_id"]] = run.get("run_name", run["run_id"])
        if run["run_id"] not in runs_map:
            runs_map[run["run_id"]] = [run]
        else:
            runs_map[run["run_id"]].append(run)

    graphs = asyncio.run(load_graphs())

    for run_id, runs in runs_map.items():
        with st.expander(f"{run_id_to_run_name[run_id]}"):
            with st.container(border=True):
                cols = st.columns(3)
                cols[0].write("Number of nodes")
                cols[1].write("Number of edges")

            for run in runs:
                with st.container(border=True):
                    cols = st.columns(3)

                    graph = graphs[run["app_id"]]

                    if len(graph["nodes"]):
                        cols[0].write(str(len(graph["nodes"])))
                        cols[1].write(str(len(graph["edges"]) // 2))
                    else:
                        cols[0].image(
                            SVG_LOADER,
                            width=40,
                        )
                        cols[1].image(
                            SVG_LOADER,
                            width=40,
                        )

                    buttons = cols[2].columns([1, 1])
                    if buttons[0].button(f"Show the graph", key=run["app_id"]):
                        components.iframe(
                            f"{MIND_MAP_FRONTEND_URL}/content?id={run['app_id']}&theme=dark",
                            height=700,
                        )
                    if buttons[1].button(
                        f"Show the chat", key=f"Chat - {run['app_id']}"
                    ):
                        components.iframe(
                            f"{MIND_MAP_FRONTEND_URL}/chat?id={run['app_id']}&theme=dark",
                            height=700,
                        )

            if st.button("Delete", type="primary", key=f"Delete run #{run_id}"):
                delete_confirm_dialog(run_id)
