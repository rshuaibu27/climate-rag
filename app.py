import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
from src.rag import RAGPipeline

st.set_page_config(
    page_title="Climate Research Navigator",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Climate Research Navigator")
st.caption("Ask questions about climate science, grounded in real academic research.")

@st.cache_resource
def load_pipeline():
    return RAGPipeline()

pipeline = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("📄 Retrieved Papers")
    st.caption("Papers retrieved for your last question.")

    if st.button("Reset conversation"):
        st.session_state.messages = []
        pipeline.reset()
        st.rerun()

    last_papers = None
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "assistant":
            last_papers = msg.get("papers")
            break

    if last_papers:
        for p in last_papers:
            with st.expander(f"{p['title'][:60]}... ({p['year']})"):
                st.write(f"**Authors:** {p['authors']}")
                st.write(f"**Relevance score:** {p['score']}")
    else:
        st.info("Ask a question to see retrieved papers here.")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Ask a climate science question..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Searching research and generating answer..."):
            reply, papers = pipeline.chat(user_input)
            st.markdown(reply)
            st.session_state.messages.append({
                "role": "assistant",
                "content": reply,
                "papers": papers
            })
            st.rerun()