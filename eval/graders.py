from uuid import uuid4

import streamlit as st
from code_editor import code_editor

from dial_api import read_state, write_state

DEFAULT_CODE = """
GRAPH_EXAMPLE = {
  "root": "1",
  "nodes": [
    {
      "id": "1",
      "label": "Node A",
      "question": "Example question",
      "answer": "Example answer"
    },
    {
      "id": "2",
      "label": "Node B",
      "question": "Example question",
      "answer": "Example answer"
    }
  ],
  "edges": [
    {
      "source": "1",
      "target": "2"
    }
  ]
}

def grade(graph: any) -> tuple[bool, str]:
    return True, "The reason"
"""

state, etag = read_state()


@st.dialog("Confirmation")
def delete_confirm_dialog(id: str):
    cols = st.columns([1, 1])

    if cols[0].button("Cancel"):
        st.rerun()

    if cols[1].button("Delete", type="primary"):
        state["graders"] = [grader for grader in state["graders"] if grader["id"] != id]

        write_state(state, etag)
        st.rerun()


def change_grader():
    changed = False

    for grader in state["graders"]:
        for field in {
            "Type",
            "Name",
            "Prompt",
            "Labels",
            "Questions",
            "Answers",
            "Edges",
            "Code",
            "Description",
        }:
            if f"Grader #{grader['id']}. {field}" in st.session_state:
                val = st.session_state[f"Grader #{grader['id']}. {field}"]
                if field == "Code":
                    if val:
                        val = val["text"]
                    else:
                        val = grader.get("code", None)

                if grader.get(field.lower(), None) != val:
                    grader[field.lower()] = val
                    changed = True

    if changed:
        write_state(state, etag)


if "graders" not in state:
    state["graders"] = []

for i, grader in enumerate(state["graders"]):
    with st.container(border=True):
        st.text_input(
            "Name",
            value=grader.get("name", ""),
            placeholder=f"Grader #{i + 1}",
            key=f"Grader #{grader['id']}. Name",
            on_change=change_grader,
        )

        with st.expander(
            label="Settings",
        ):
            option = st.selectbox(
                "Type",
                ("Graph", "Code"),
                index=("Graph", "Code").index(grader.get("type", "Graph")),
                key=f"Grader #{grader['id']}. Type",
                on_change=change_grader,
            )

            if option == "Graph":
                st.text("Context")

                st.toggle(
                    "Labels",
                    value=grader.get("labels", True),
                    key=f"Grader #{grader['id']}. Labels",
                    on_change=change_grader,
                )
                st.toggle(
                    "Questions",
                    value=grader.get("questions", True),
                    key=f"Grader #{grader['id']}. Questions",
                    on_change=change_grader,
                )
                st.toggle(
                    "Answers",
                    value=grader.get("answers", True),
                    key=f"Grader #{grader['id']}. Answers",
                    on_change=change_grader,
                )
                st.toggle(
                    "Edges",
                    value=grader.get("edges", True),
                    key=f"Grader #{grader['id']}. Edges",
                    on_change=change_grader,
                )

                st.text_area(
                    "Prompt",
                    value=grader.get("prompt", ""),
                    placeholder="Describe the pass/fail criteria",
                    key=f"Grader #{grader['id']}. Prompt",
                    on_change=change_grader,
                )
            elif option == "Code":
                code = code_editor(
                    grader["code"] if grader.get("code", None) else DEFAULT_CODE,
                    focus=True,
                    lang="python",
                    response_mode="debounce",
                    key=f"Grader #{grader['id']}. Code",
                )

                change_grader()

            st.text_area(
                "Description",
                value=grader.get("description", ""),
                key=f"Grader #{grader['id']}. Description",
                on_change=change_grader,
            )

            if st.button(
                "Delete",
                key=f"Grader #{grader['id']}. Delete",
                type="primary",
            ):
                delete_confirm_dialog(grader["id"])

if st.button("Add grader"):
    state["graders"].append({"id": str(uuid4())})

    write_state(state, etag)
    st.rerun()
