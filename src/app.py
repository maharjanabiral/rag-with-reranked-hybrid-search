import streamlit as st
import requests
from dotenv import load_dotenv
import os

load_dotenv()
API_URL = os.getenv("API_URL")

st.set_page_config(
    page_title="Vaultify",
    layout="wide",
)

st.markdown(
    """
<style>
    .stChatMessage { padding: 0.5rem; }
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("Vaultify")
    st.caption("Ask Anything related to the documents.")
    st.divider()

    show_debug = st.toggle("Show pipeline details", value=False)
    show_docs = st.toggle("Show retrieved docs", value=False)

    st.divider()

    st.subheader("Ingest Documents")
    ingest_path = st.text_input("Path", value="data/")
    if st.button("Ingest", use_container_width=True):
        with st.spinner("Ingesting..."):
            try:
                res = requests.post(f"{API_URL}/ingest", json={"path": ingest_path})
                res.raise_for_status()
                st.success(f"Ingested {res.json()['files_ingested']} files")
            except Exception as e:
                st.error(f"Ingest failed: {e}")

    st.divider()

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Vaultify")
st.caption("Ask Anything.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if msg["role"] == "assistant" and show_debug and "meta" in msg:
            with st.expander("Pipeline details"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Relevance", msg["meta"]["relevance_score"])
                col2.metric("Hallucination", msg["meta"]["hallucination_score"])
                col3.metric("Answer", msg["meta"]["answer_score"])
                col4.metric("Retries", msg["meta"]["retries"])

        if msg["role"] == "assistant" and show_docs and "docs" in msg:
            with st.expander(f"Retrieved docs ({len(msg['docs'])})"):
                for i, doc in enumerate(msg["docs"]):
                    st.markdown(f"**Doc {i + 1}** — `{doc['source']}`")
                    st.info(doc["content"][:400])

question = st.chat_input("Ask about monetary policy, GDP, inflation, banking...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = requests.post(
                    f"{API_URL}/chat",
                    json={"question": question},
                    timeout=60,
                )
                res.raise_for_status()
                data = res.json()

                st.write(data["generation"])

                if show_debug:
                    with st.expander("Pipeline details"):
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Relevance", data["relevance_score"])
                        col2.metric("Hallucination", data["hallucination_score"])
                        col3.metric("Answer", data["answer_score"])
                        col4.metric("Retries", data["retries"])

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": data["generation"],
                        "meta": {
                            "relevance_score": data["relevance_score"],
                            "hallucination_score": data["hallucination_score"],
                            "answer_score": data["answer_score"],
                            "retries": data["retries"],
                        },
                        "docs": data.get("docs", []),
                    }
                )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to API — make sure FastAPI is running on port 8000"
                )
            except Exception as e:
                st.error(f"Error: {e}")
