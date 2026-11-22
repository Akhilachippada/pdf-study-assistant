"""Small helper functions that do not depend on LangChain."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

from src.models import SourceReference


def format_history(history: Iterable[Mapping[str, str]]) -> str:
    """Turn [{'role': 'user', 'content': '...'}, ...] into a plain-text transcript."""
    lines = []
    for message in history:
        speaker = "User" if message.get("role") == "user" else "Assistant"
        lines.append(f"{speaker}: {message.get('content', '').strip()}")
    return "\n".join(lines)


def group_pages_by_file(sources: Iterable[SourceReference]) -> dict[str, list[int]]:
    """Group source references as {filename: [sorted unique page numbers]}."""
    grouped: dict[str, set[int]] = defaultdict(set)
    for source in sources:
        grouped[source.filename].add(source.page)
    return {name: sorted(pages) for name, pages in grouped.items()}


def truncate(text: str, limit: int = 300) -> str:
    """Shorten text for display, adding an ellipsis when it was cut."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
