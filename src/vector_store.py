"""FAISS vector store wrapper (index building + similarity search)."""
from __future__ import annotations

from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class VectorStoreManager:
    """Owns one in-memory FAISS index for the current set of uploaded documents."""

    def __init__(self, embeddings) -> None:
        self._embeddings = embeddings
        self._store: Optional[FAISS] = None

    @property
    def is_ready(self) -> bool:
        return self._store is not None

    def build(self, documents: list[Document]) -> None:
        """Embed every chunk and index the vectors (replaces any previous index)."""
        if not documents:
            raise ValueError("Cannot build a vector index from zero documents.")
        self._store = FAISS.from_documents(documents, self._embeddings)

    def search(
        self, query: str, k: int = 4, max_distance: Optional[float] = None
    ) -> list[tuple[Document, float]]:
        """Return up to `k` (chunk, distance) pairs, most similar first.

        The query is embedded with the same model as the chunks. FAISS then does an exact
        nearest-neighbour search. Chunks farther than `max_distance` are dropped so that
        unrelated questions retrieve nothing instead of the "least bad" chunks.
        """
        if self._store is None:
            raise RuntimeError("The vector store is empty. Process documents first.")
        results = self._store.similarity_search_with_score(query, k=k)
        pairs = [(doc, float(score)) for doc, score in results]
        if max_distance is not None:
            pairs = [(doc, score) for doc, score in pairs if score <= max_distance]
        return pairs

    def clear(self) -> None:
        self._store = None
