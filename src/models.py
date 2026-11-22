"""Plain data containers shared across the project (no third-party dependencies)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SourceReference:
    """One retrieved chunk that was used as evidence for an answer."""

    filename: str
    page: int
    score: float      # squared L2 distance from FAISS: LOWER means more similar
    excerpt: str


@dataclass
class RAGAnswer:
    """Result of asking one question."""

    answer: str
    sources: list[SourceReference] = field(default_factory=list)
    standalone_question: str = ""
    found: bool = True   # False when the documents did not contain the answer


@dataclass
class FileReport:
    """Processing outcome for one uploaded PDF."""

    filename: str
    total_pages: int = 0
    pages_with_text: int = 0
    chunks: int = 0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class IngestionReport:
    """Processing outcome for a whole upload batch."""

    files: list[FileReport] = field(default_factory=list)
    total_chunks: int = 0

    @property
    def ready(self) -> bool:
        return self.total_chunks > 0
