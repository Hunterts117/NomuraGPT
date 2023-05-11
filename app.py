import os
import tempfile
import streamlit as st
from streamlit_chat import message
from NomuraGPT import NomuraGPT

st.set_page_config(page_title="NomuraGPT")
os.environ['OPENAI_API_KEY'] = "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"

def display_messages():
    st.subheader("NomuraGPT")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["agent"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def main():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")

        st.session_state["agent"] = NomuraGPT(st.session_state["OPENAI_API_KEY"])

    st.header("NomuraGPT")

    display_messages()
    
    st.text_input("Message", key="user_input", on_change=process_input)

    st.divider()

if __name__ == "__main__":
    main()
