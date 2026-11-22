"""Central configuration for the PDF Study Assistant.

All tunable values live here and can be overridden with environment variables
(or a local `.env` file). API keys are NEVER hardcoded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Load variables from .env (if the file exists) into the process environment.
load_dotenv(BASE_DIR / ".env")


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to a default."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable, falling back to a default."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Immutable bundle of application settings (defaults are sensible for a laptop)."""

    groq_api_key: str = ""
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.0          # 0 = most deterministic / least creative
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000                # characters per chunk
    chunk_overlap: int = 200              # characters shared between neighbouring chunks
    top_k: int = 6                        # chunks retrieved per question
    # FAISS returns SQUARED L2 distance. For normalised vectors: d = 2 - 2*cosine.
    # 1.4 therefore keeps chunks with cosine similarity above ~0.30.
    max_distance: float = 4.0
    max_upload_mb: int = 25
    max_history_messages: int = 6         # recent chat messages used for follow-ups

    @property
    def has_api_key(self) -> bool:
        return bool(self.groq_api_key.strip())


def get_settings() -> Settings:
    """Build Settings from environment variables (read fresh on every call)."""
    settings = Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        llm_model=os.getenv("LLM_MODEL", Settings.llm_model).strip(),
        llm_temperature=_env_float("LLM_TEMPERATURE", Settings.llm_temperature),
        embedding_model=os.getenv("EMBEDDING_MODEL", Settings.embedding_model).strip(),
        chunk_size=_env_int("CHUNK_SIZE", Settings.chunk_size),
        chunk_overlap=_env_int("CHUNK_OVERLAP", Settings.chunk_overlap),
        top_k=_env_int("TOP_K", Settings.top_k),
        max_distance=_env_float("MAX_DISTANCE", Settings.max_distance),
        max_upload_mb=_env_int("MAX_UPLOAD_MB", Settings.max_upload_mb),
        max_history_messages=_env_int("MAX_HISTORY_MESSAGES", Settings.max_history_messages),
    )
    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
    if settings.top_k < 1:
        raise ValueError("TOP_K must be at least 1")
    return settings
