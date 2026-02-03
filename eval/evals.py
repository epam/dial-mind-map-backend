import asyncio
import json
from copy import deepcopy
from typing import Any, Dict, List
from uuid import uuid4

import streamlit as st
import streamlit.components.v1 as components
import tqdm
from openai import AzureOpenAI

from config import DIAL_API_KEY, DIAL_URL, MIND_MAP_FRONTEND_URL
from dial_api import get_graph, read_state, write_state
from graders import DEFAULT_CODE

SYSTEM_PROMPT = """
    You are evaluating whether the provided graph meets the specific criterion.
    You will receive the desciption of the criterion and the graph.
    Respond **only** in JSON format as follows:
    {
        "reasoning": "A concise explanation of why this decision was made",
        "result": "pass" or "fail",
    }
"""


@st.dialog("Confirmation")
def delete_confirm_dialog(id: str):
    cols = st.columns([1, 1])

    if cols[0].button("Cancel"):
        st.rerun()

    if cols[1].button("Delete", type="primary"):
        state["evals"] = [eval for eval in state["evals"] if eval["id"] != id]

        write_state(state, etag)
        st.rerun()


state, etag = read_state()

if "evals" not in state:
    state["evals"] = []

client = AzureOpenAI(
    api_version="2024-10-21", azure_endpoint=DIAL_URL, api_key=DIAL_API_KEY
)

if "graders" not in state:
    state["graders"] = []
if "runs" not in state:
    state["runs"] = []


async def load_graphs(app_ids: List[str]) -> Dict[str, Any]:
    tasks = [get_graph(app_id) for app_id in app_ids]
    results = await asyncio.gather(*tasks)
    return {app_id: graph for app_id, graph in zip(app_ids, results)}


def reassign_node_ids_to_label(graph: Any) -> Any:
    graph = deepcopy(graph)

    node_id_to_node_label = {
        node["data"]["id"]: node["data"]["label"] for node in graph["nodes"]
    }

    if graph.get("root", None):
        graph["root"] = node_id_to_node_label[graph["root"]]

    for edge in graph["edges"]:
        edge_data = edge["data"]

        for key in ["source", "target"]:
            edge_data[key] = node_id_to_node_label[edge_data[key]]

    return graph


async def single_evaluation(sem: asyncio.Semaphore, graph: Any, grader: Any) -> Any:
    async with sem:
        if grader["type"] == "Code":
            context_graph = {}

            context_graph["root"] = graph.get("root", None)

            context_graph["nodes"] = [
                {
                    "id": node["data"]["id"],
                    "label": node["data"]["label"],
                    "question": node["data"]["questions"][0],
                    "answer": node["data"]["details"],
                }
                for node in graph["nodes"]
            ]

            if grader["edges"]:
                context_graph["edges"] = [
                    {"source": edge["data"]["source"], "target": edge["data"]["target"]}
                    for i, edge in enumerate(graph["edges"])
                    if i % 2 == 0
                ]

            namespace = {}
            exec(
                grader.get("code", None) if grader.get("code", None) else DEFAULT_CODE,
                namespace,
            )

            result = namespace["grade"](context_graph)

            return {"result": "pass" if result[0] else "fail", "reasoning": result[1]}

        graph = reassign_node_ids_to_label(graph)

        context_graph = {}

        context_graph["root"] = graph.get("root", None)

        context_graph["nodes"] = [
            {}
            | ({"label": node["data"]["label"]} if grader["labels"] else {})
            | (
                {"question": node["data"]["questions"][0]}
                if grader["questions"]
                else {}
            )
            | ({"answer": node["data"]["details"]} if grader["answers"] else {})
            for node in graph["nodes"]
        ]

        if grader["edges"]:
            context_graph["edges"] = [
                {"source": edge["data"]["source"], "target": edge["data"]["target"]}
                for i, edge in enumerate(graph["edges"])
                if i % 2 == 0
            ]

        user_prompt = f"""
            The graph:
            {json.dumps(context_graph, indent=2)}

            The desciption of the criterion:
            {grader['prompt']}
        """

        result = client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            response_format={"type": "json_object"},
        )

        return json.loads(result.choices[0].message.content)


async def full_evaluation(
    sem: asyncio.Semaphore,
    run_name: str,
    app_ids: List[str],
    graders: List[Any],
    graphs: Dict[str, Any],
):
    tasks = []
    for app_id in app_ids:
        for grader in graders:
            tasks.append(single_evaluation(sem, graphs[app_id], grader))

    results = await asyncio.gather(*tasks)

    id = 0
    mapped_results = []
    for app_id in app_ids:
        for grader in graders:
            mapped_results.append(
                {
                    "app_id": app_id,
                    "run_name": run_name,
                    "grader_name": grader["name"],
                    "result": results[id],
                }
            )

            id += 1

    return mapped_results


async def evaluate(run_names: List[str], grader_names: List[str]):
    sem = asyncio.Semaphore(10)

    tasks = []

    async def with_index(i, coro):
        return i, await coro

    state, etag = read_state()

    graders = [grader for grader in state["graders"] if grader["name"] in grader_names]

    for run_name in run_names:
        app_ids = [
            run["app_id"]
            for run in state["runs"]
            if run.get("run_name", run["run_id"]) == run_name
        ]
        graphs = await load_graphs(app_ids)

        for app_id in app_ids:
            for grader in graders:
                tasks.append(single_evaluation(sem, graphs[app_id], grader))

    tasks = [asyncio.create_task(with_index(i, coro)) for i, coro in enumerate(tasks)]

    results = [None] * len(tasks)
    for task in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        i, t = await task
        results[i] = t

    id = 0
    mapped_results = []
    for run_name in run_names:
        mapped_results.append([])
        app_ids = [
            run["app_id"]
            for run in state["runs"]
            if run.get("run_name", run["run_id"]) == run_name
        ]

        for app_id in app_ids:
            for grader in graders:
                mapped_results[-1].append(
                    {
                        "app_id": app_id,
                        "run_name": run_name,
                        "grader_name": grader["name"],
                        "result": results[id],
                    }
                )

                id += 1

    while True:
        state, etag = read_state()

        state["evals"].append(
            {
                "name": evaluation_name,
                "results": mapped_results,
                "run_names": run_names,
                "graders": graders,
                "id": str(uuid4()),
            }
        )

        if write_state(state, etag) != 412:
            break


with st.container(border=True):
    st.subheader("New Eval")

    evaluation_name = st.text_input("Name")

    run_names = st.multiselect(
        "Select runs",
        list(
            dict.fromkeys(
                reversed([run.get("run_name", run["run_id"]) for run in state["runs"]])
            )
        ),
    )

    all_grader_names = [
        grader["name"] if grader.get("name", None) else f"Grader #{i}"
        for i, grader in enumerate(state["graders"])
    ]

    grader_names = st.multiselect(
        "Select graders",
        all_grader_names,
        all_grader_names,
        key="graders_select",
    )

    if st.button("Evaluate"):
        asyncio.run(evaluate(run_names, grader_names))

if len(state["evals"]):
    all_app_ids = {
        result["app_id"]
        for eval in state["evals"]
        for result in eval["results"]
        if type(result) != list
    }

    all_app_ids = all_app_ids | {
        result["app_id"]
        for eval in state["evals"]
        for run_result in eval["results"]
        if type(run_result) == list
        for result in run_result
    }

    graphs = asyncio.run(load_graphs(all_app_ids))

    for eval in reversed(state["evals"]):
        eval_id = eval["id"]

        graders = eval.get("graders", [])

        # Map the old format to the new
        if len(eval["results"]) and type(eval["results"][0]) != list:
            eval["results"] = [eval["results"]]

        with st.expander(eval["name"]):
            cols = st.columns([1, 1, 1])

            if cols[2].button("Delete", type="primary", key=f"Delete eval #{eval_id}"):
                delete_confirm_dialog(eval_id)

            for run_result in eval["results"]:
                results_mapping = {}
                for result in run_result:
                    if result["app_id"] not in results_mapping:
                        results_mapping[result["app_id"]] = {}

                    results_mapping[result["app_id"]][result["grader_name"]] = result[
                        "result"
                    ]

                with st.expander(run_result[0].get("run_name", "Run")):
                    with st.container(border=True):
                        cols = st.columns(4)
                        cols[0].write("Number of nodes")
                        cols[1].write("Number of edges")
                        cols[2].write("Graders")

                    app_ids = {result["app_id"] for result in run_result}

                    for app_id in app_ids:
                        with st.container(border=True):
                            cols = st.columns(4)

                            graph = graphs[app_id]

                            cols[0].write(str(len(graph["nodes"])))
                            cols[1].write(str(len(graph["edges"]) // 2))

                            html = """
                            <style>
                            .tooltip{position:relative;display:inline-block;}
                            .tooltip .tooltiptext{
                            visibility:hidden;width:250px;background:#333;color:#fff;text-align:left;
                            border-radius:8px;padding:8px 10px;position:absolute;z-index:1;
                            bottom:125%;left:50%;transform:translateX(-50%);
                            opacity:0;transition:opacity 0.3s ease 0.5s;font-size:14px;line-height:1.3;white-space:normal;
                            }
                            .tooltip:hover .tooltiptext{
                            visibility:visible;opacity:1;transition-delay:0s;
                            }
                            </style>
                            <div style='display:flex;gap:10px;margin-bottom:10px;flex-wrap:wrap;'>
                            """

                            for grader in graders:
                                result = results_mapping[app_id][grader["name"]]

                                color = (
                                    "green"
                                    if result["result"].lower() == "pass"
                                    else "red"
                                )

                                reasoning = f"<strong>{grader['name']}</strong><br><br>{result['reasoning']}"

                                html += f"<div class='tooltip'><div style='width:30px;height:30px;background:{color};border-radius:6px;'></div><span class='tooltiptext' style=\"white-space: pre-line;\">{reasoning}</span></div>"

                            cols[2].markdown(html, unsafe_allow_html=True)

                            buttons = cols[3].columns([1, 1])
                            if buttons[0].button(
                                f"Show the graph", key=f"{eval_id}-{app_id}-content"
                            ):
                                components.iframe(
                                    f"{MIND_MAP_FRONTEND_URL}/content?id={app_id}&theme=dark",
                                    height=700,
                                )
                            if buttons[1].button(
                                f"Show the chat", key=f"{eval_id}-{app_id}-chat"
                            ):
                                components.iframe(
                                    f"{MIND_MAP_FRONTEND_URL}/chat?id={app_id}&theme=dark",
                                    height=700,
                                )
