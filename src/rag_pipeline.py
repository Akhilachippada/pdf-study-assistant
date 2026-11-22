"""The RAG pipeline: ties extraction, chunking, embeddings, FAISS and the LLM together."""
from __future__ import annotations

import logging
from collections import Counter
from typing import Callable, Mapping, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from config import Settings, get_settings
from src.models import FileReport, IngestionReport, RAGAnswer, SourceReference
from src.pdf_processor import PDFProcessingError, extract_pages
from src.prompt_templates import CONDENSE_PROMPT, NOT_FOUND_MESSAGE, QA_PROMPT
from src.text_splitter import split_pages
from src.utils import format_history, truncate
from src.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class RAGError(Exception):
    """User-presentable error raised by the pipeline."""


def build_context(hits: Sequence[tuple[Document, float]]) -> str:
    """Format retrieved chunks as labelled excerpts for the prompt."""
    blocks = []
    for doc, _score in hits:
        label = f"[Source: {doc.metadata.get('source', 'unknown')}, page {doc.metadata.get('page', '?')}]"
        blocks.append(f"{label}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def build_sources(hits: Sequence[tuple[Document, float]]) -> list[SourceReference]:
    """Create unique (file, page) references, keeping the best-scoring chunk per page."""
    best: dict[tuple[str, int], SourceReference] = {}
    for doc, score in hits:
        key = (doc.metadata.get("source", "unknown"), int(doc.metadata.get("page", 0)))
        if key not in best or score < best[key].score:
            best[key] = SourceReference(key[0], key[1], score, truncate(doc.page_content))
    return sorted(best.values(), key=lambda s: s.score)


class RAGPipeline:
    """Ingest PDFs once, then answer many questions against them.

    `embeddings` and `llm` are injected, which keeps the class easy to unit-test
    with fakes. `llm` may be None: documents can still be processed without an API key.
    """

    def __init__(self, embeddings, llm=None, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._store = VectorStoreManager(embeddings)
        self._llm = llm
        parser = StrOutputParser()
        # LCEL: "prompt | model | parser" builds a runnable chain.
        self._qa_chain = (QA_PROMPT | llm | parser) if llm is not None else None
        self._condense_chain = (CONDENSE_PROMPT | llm | parser) if llm is not None else None

    @property
    def is_ready(self) -> bool:
        return self._store.is_ready

    # ------------------------------------------------------------------ ingestion
    def ingest(
        self,
        files: Sequence[tuple[str, bytes]],
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> IngestionReport:
        """PDF bytes -> text -> clean -> chunks -> embeddings -> FAISS index.

        `files` is a list of (filename, raw_bytes). Unreadable files are reported and
        skipped, so one bad PDF never blocks the others.
        """
        notify = progress_callback or (lambda _message: None)
        reports: list[FileReport] = []
        pages = []

        for filename, data in files:
            notify(f"Reading {filename} ...")
            try:
                extraction = extract_pages(filename, data, self._settings.max_upload_mb)
            except PDFProcessingError as exc:
                logger.info("Rejected %s: %s", filename, exc)
                reports.append(FileReport(filename=filename, error=str(exc)))
                continue
            pages.extend(extraction.pages)
            reports.append(
                FileReport(
                    filename=filename,
                    total_pages=extraction.total_pages,
                    pages_with_text=len(extraction.pages),
                )
            )

        if not pages:
            self._store.clear()
            return IngestionReport(files=reports, total_chunks=0)

        notify("Splitting text into chunks ...")
        chunks = split_pages(pages, self._settings.chunk_size, self._settings.chunk_overlap)

        notify(f"Embedding {len(chunks)} chunks and building the FAISS index ...")
        try:
            self._store.build(chunks)
        except Exception as exc:  # noqa: BLE001 - model download / memory problems, etc.
            raise RAGError(f"Failed to build the vector index: {exc}") from exc

        chunk_counts = Counter(chunk.metadata["source"] for chunk in chunks)
        for report in reports:
            if report.ok:
                report.chunks = chunk_counts.get(report.filename, 0)
        notify("Ready!")
        return IngestionReport(files=reports, total_chunks=len(chunks))

    # ------------------------------------------------------------------ question answering
    def _condense_question(self, question: str, history: Sequence[Mapping[str, str]]) -> str:
        """Rewrite a follow-up ("what about its limits?") into a standalone search query."""
        if not history or self._condense_chain is None:
            return question
        try:
            rewritten = self._condense_chain.invoke(
                {"chat_history": format_history(history), "question": question}
            ).strip()
            return rewritten or question
        except Exception as exc:  # noqa: BLE001 - fall back to the raw question
            logger.warning("Question rewriting failed, using original question: %s", exc)
            return question

    def ask(
        self, question: str, chat_history: Optional[Sequence[Mapping[str, str]]] = None
    ) -> RAGAnswer:
        """Retrieve relevant chunks and generate a grounded answer with sources."""
        question = (question or "").strip()
        if not question:
            raise RAGError("Please type a question.")
        if not self._store.is_ready:
            raise RAGError("Upload and process at least one PDF before asking questions.")
        if self._qa_chain is None:
            raise RAGError("The language model is not configured. Set GROQ_API_KEY in your .env file.")

        history = list(chat_history or [])[-self._settings.max_history_messages:]
        standalone = self._condense_question(question, history)

        hits = self._store.search(standalone, self._settings.top_k, self._settings.max_distance)
        if not hits:
            # Nothing similar enough: answer honestly without calling the LLM at all.
            return RAGAnswer(NOT_FOUND_MESSAGE, [], standalone, found=False)

        try:
            answer = self._qa_chain.invoke({"context": build_context(hits), "question": standalone}).strip()
        except Exception as exc:  # noqa: BLE001 - network, auth, rate limit ...
            raise RAGError(f"The language model request failed: {exc}") from exc

        if not answer:
            raise RAGError("The language model returned an empty response. Please try again.")
        if NOT_FOUND_MESSAGE in answer:
            return RAGAnswer(NOT_FOUND_MESSAGE, [], standalone, found=False)
        return RAGAnswer(answer, build_sources(hits), standalone, found=True)

    def reset(self) -> None:
        """Drop the FAISS index (e.g. when the user removes all documents)."""
        self._store.clear()
