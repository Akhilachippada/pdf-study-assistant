"""Creates the chat model used to generate answers (Groq-hosted open models)."""
from __future__ import annotations

from config import Settings


class MissingAPIKeyError(Exception):
    """Raised when GROQ_API_KEY is not configured."""


def get_llm(settings: Settings):
    """Build a LangChain chat model for Groq.

    Raises:
        MissingAPIKeyError: if no API key is configured.
        RuntimeError: if the langchain-groq package is not installed.
    """
    if not settings.has_api_key:
        raise MissingAPIKeyError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:
        raise RuntimeError("langchain-groq is not installed. Run: pip install -r requirements.txt") from exc

    return ChatGroq(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
        temperature=settings.llm_temperature,
        max_retries=2,
    )
