import json

import streamlit as st

from dial_api import read_state, write_state

state, etag = read_state()

new_state = st.text_area("State", json.dumps(state, indent=2), height="content")

if st.button("Save"):
    write_state(json.loads(new_state), etag)
