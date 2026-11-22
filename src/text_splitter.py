"""Split page-level text into overlapping chunks suitable for embedding."""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.pdf_processor import PageContent


def split_pages(
    pages: list[PageContent],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Convert pages into LangChain `Document` chunks that keep filename + page metadata.

    RecursiveCharacterTextSplitter tries the separators in order (paragraph, line,
    sentence, word, character), so chunks break at the most natural boundary that fits
    in `chunk_size`. The overlap keeps context that straddles a boundary retrievable.
    Chunks never span two pages, which keeps page citations exact.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    page_documents = [
        Document(page_content=page.text, metadata={"source": page.filename, "page": page.page_number})
        for page in pages
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(page_documents)
    for chunk_id, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = chunk_id  # handy for debugging / future features
    return chunks
