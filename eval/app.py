import streamlit as st

graders_page = st.Page("graders.py", title="Graders")
runs_page = st.Page("runs.py", title="Runs")
evals_page = st.Page("evals.py", title="Evals")
reports_page = st.Page("reports.py", title="Report")
state_page = st.Page("state.py", title="State")

pg = st.navigation([graders_page, runs_page, evals_page, reports_page, state_page])

st.set_page_config(
    layout="wide",
    page_title="Mind Map Eval",
)

pg.run()
