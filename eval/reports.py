import streamlit as st

from dial_api import read_state
from report import prepare_report

state, etag = read_state()

eval_names = st.multiselect(
    "Select evals",
    list(dict.fromkeys(reversed([eval["name"] for eval in state["evals"]]))),
)

if st.button("Prepare report"):
    evals = [eval for eval in state["evals"] if eval["name"] in eval_names]

    aggregated_eval = {"results": [], "graders": []}
    for eval in evals:
        for result in eval["results"]:
            aggregated_eval["results"].append(result)

        for grader in eval["graders"]:
            if grader not in aggregated_eval["graders"]:
                aggregated_eval["graders"].append(grader)

    st.download_button(
        "Download report",
        data=prepare_report(aggregated_eval),
        file_name="report.xlsx",
        on_click="ignore",
        icon=":material/download:",
    )
