"""Sentence-Transformers embedding model, wrapped for LangChain."""
from __future__ import annotations


def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Create the embedding model (runs locally on CPU; no API key needed).

    The model is downloaded from Hugging Face on first use (~90 MB) and cached afterwards.
    Embeddings are L2-normalised, so FAISS L2 distance maps directly to cosine similarity.

    The import is lazy because importing sentence-transformers/torch takes several seconds.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise RuntimeError(
            "langchain-huggingface / sentence-transformers is not installed. "
            "Run: pip install -r requirements.txt"
        ) from exc

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
