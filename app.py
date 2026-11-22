"""Streamlit user interface for the AI-Powered PDF Study Assistant.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from config import BASE_DIR, get_settings
from src.embeddings import load_embedding_model
from src.llm_handler import MissingAPIKeyError, get_llm
from src.models import FileReport, SourceReference
from src.rag_pipeline import RAGError, RAGPipeline
from src.utils import group_pages_by_file

st.set_page_config(page_title="PDF Study Assistant", page_icon="📚", layout="wide")


# --------------------------------------------------------------------------- helpers
def load_css() -> None:
    """Inject assets/style.css into the page."""
    css_path = BASE_DIR / "assets" / "style.css"
    try:
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    except OSError:
        pass  # Styling is optional; the app still works without it.


@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~90 MB) ...")
def get_embeddings(model_name: str):
    """Load the embedding model once per server process and share it across sessions."""
    return load_embedding_model(model_name)


def init_state() -> None:
    """Create session_state keys (Streamlit reruns this script on every interaction)."""
    st.session_state.setdefault("messages", [])        # chat transcript
    st.session_state.setdefault("reports", [])         # per-file processing results
    st.session_state.setdefault("pipeline", None)      # RAGPipeline once docs are processed
    st.session_state.setdefault("uploader_key", 0)     # bump to reset the file uploader


def render_sources(sources: list[SourceReference]) -> None:
    """Show which PDF files/pages supported an answer, plus the retrieved excerpts."""
    if not sources:
        return
    grouped = group_pages_by_file(sources)
    tags = "".join(
        f'<span class="source-tag">📄 {name} · p. {", ".join(map(str, pages))}</span>'
        for name, pages in grouped.items()
    )
    st.markdown(f'<div class="source-row">{tags}</div>', unsafe_allow_html=True)
    with st.expander("View retrieved excerpts"):
        for source in sources:
            st.markdown(f"**{source.filename} — page {source.page}** (distance {source.score:.2f})")
            st.caption(source.excerpt)


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message.get("is_error"):
            st.error(message["content"])
        else:
            st.markdown(message["content"])
            render_sources(message.get("sources", []))


def render_file_report(report: FileReport) -> None:
    if report.ok:
        st.markdown(
            f'<div class="doc-chip ok">✅ <b>{report.filename}</b><br>'
            f"<small>{report.pages_with_text}/{report.total_pages} pages with text · "
            f"{report.chunks} chunks</small></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="doc-chip bad">❌ <b>{report.filename}</b><br><small>{report.error}</small></div>',
            unsafe_allow_html=True,
        )


def process_documents(uploaded_files, settings) -> None:
    """Run the ingestion half of the RAG pipeline for the uploaded files."""
    files = [(f.name, f.getvalue()) for f in uploaded_files]
    with st.status("Processing documents ...", expanded=True) as status:
        try:
            embeddings = get_embeddings(settings.embedding_model)
            try:
                llm = get_llm(settings)
            except MissingAPIKeyError:
                llm = None  # documents can still be indexed; asking questions will be blocked
            pipeline = RAGPipeline(embeddings, llm, settings)
            report = pipeline.ingest(files, progress_callback=status.write)
        except (RAGError, RuntimeError) as exc:
            status.update(label="Processing failed", state="error")
            st.error(str(exc))
            return

        st.session_state.reports = report.files
        st.session_state.messages = []  # old chat refers to old documents
        if report.ready:
            st.session_state.pipeline = pipeline
            status.update(label=f"Ready: {report.total_chunks} chunks indexed", state="complete")
        else:
            st.session_state.pipeline = None
            status.update(label="No readable documents", state="error")


def reset_documents() -> None:
    st.session_state.pipeline = None
    st.session_state.reports = []
    st.session_state.messages = []
    st.session_state.uploader_key += 1


# --------------------------------------------------------------------------- page
load_css()
init_state()
settings = get_settings()

st.markdown(
    '<div class="app-header"><h1>📚 PDF Study Assistant</h1>'
    "<p>Upload your study material and ask questions. Answers come only from your documents, "
    "with page references.</p></div>",
    unsafe_allow_html=True,
)

if not settings.has_api_key:
    st.warning(
        "`GROQ_API_KEY` is not set. You can upload and process PDFs, but you need a key in your "
        "`.env` file (then restart the app) to ask questions."
    )

with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if st.button("Process documents", type="primary", disabled=not uploaded):
        process_documents(uploaded, settings)

    if st.session_state.reports:
        st.subheader("Uploaded files")
        for file_report in st.session_state.reports:
            render_file_report(file_report)

    st.divider()
    if st.button("Clear chat", disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.rerun()
    if st.button("Remove all documents", disabled=st.session_state.pipeline is None and not st.session_state.reports):
        reset_documents()
        st.rerun()

pipeline = st.session_state.pipeline
if pipeline is None:
    st.info("👈 Upload one or more PDFs in the sidebar and click **Process documents** to begin.")

for stored_message in st.session_state.messages:
    render_message(stored_message)

question = st.chat_input("Ask a question about your documents ...", disabled=pipeline is None)
if question:
    # Only successful turns are sent to the pipeline as conversation memory.
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if not m.get("is_error")
    ]
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents ..."):
            try:
                result = pipeline.ask(question, history)
                reply = {"role": "assistant", "content": result.answer, "sources": result.sources}
            except RAGError as exc:
                reply = {"role": "assistant", "content": str(exc), "is_error": True}
        if reply.get("is_error"):
            st.error(reply["content"])
        else:
            st.markdown(reply["content"])
            render_sources(reply["sources"])
    st.session_state.messages.append(reply)
