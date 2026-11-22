"""PDF validation, text extraction and text cleaning (PyPDF)."""
from __future__ import annotations

import io
import logging
import re
import unicodedata
from dataclasses import dataclass, field

from pypdf import PdfReader

logger = logging.getLogger(__name__)


class PDFProcessingError(Exception):
    """Raised when a PDF is invalid, unreadable, or contains no extractable text."""


@dataclass(frozen=True)
class PageContent:
    """Cleaned text of a single PDF page. `page_number` is 1-based (as shown in viewers)."""

    filename: str
    page_number: int
    text: str


@dataclass
class ExtractionResult:
    filename: str
    total_pages: int
    pages: list[PageContent] = field(default_factory=list)


def validate_pdf(filename: str, data: bytes, max_size_mb: int = 25) -> None:
    """Cheap sanity checks done BEFORE parsing, so bad uploads fail fast with clear messages."""
    if not filename.lower().endswith(".pdf"):
        raise PDFProcessingError("Only .pdf files are supported.")
    if not data:
        raise PDFProcessingError("The file is empty.")
    if len(data) > max_size_mb * 1024 * 1024:
        raise PDFProcessingError(f"The file is larger than the {max_size_mb} MB limit.")
    # Real PDFs contain the '%PDF-' marker within the first KB (some have leading junk).
    if b"%PDF-" not in data[:1024]:
        raise PDFProcessingError("This does not look like a valid PDF file.")


def clean_text(text: str) -> str:
    """Normalise raw extracted text so chunks are cleaner and embed better.

    * NFKC normalisation turns ligatures (e.g. 'ﬁ') into plain letters.
    * Control characters are removed.
    * Words broken across lines with a hyphen ('exam-\\nple') are re-joined.
    * Hard line wraps inside a paragraph become spaces; blank lines are kept as
      paragraph breaks, which the text splitter later uses as natural boundaries.
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        joined = " ".join(line.strip() for line in block.split("\n") if line.strip())
        joined = re.sub(r"[ \t]+", " ", joined)
        if joined:
            paragraphs.append(joined)
    return "\n\n".join(paragraphs)


def extract_pages(filename: str, data: bytes, max_size_mb: int = 25) -> ExtractionResult:
    """Validate a PDF and return the cleaned text of each page that has any text.

    Raises:
        PDFProcessingError: invalid, corrupt, password-protected, or image-only PDFs.
    """
    validate_pdf(filename, data, max_size_mb)

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            # Many "protected" PDFs open with an empty password; try that first.
            try:
                unlocked = reader.decrypt("")
            except Exception:  # noqa: BLE001 - missing crypto backend, bad algorithm, etc.
                unlocked = 0
            if not unlocked:
                raise PDFProcessingError("The PDF is password-protected.")
        total_pages = len(reader.pages)
    except PDFProcessingError:
        raise
    except Exception as exc:  # noqa: BLE001 - pypdf can raise many different error types
        raise PDFProcessingError(f"The PDF could not be read (it may be corrupted): {exc}") from exc

    result = ExtractionResult(filename=filename, total_pages=total_pages)
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - one bad page must not kill the whole file
            logger.warning("Skipping page %d of %s: %s", page_number, filename, exc)
            continue
        cleaned = clean_text(raw_text)
        if cleaned:
            result.pages.append(PageContent(filename, page_number, cleaned))

    if not result.pages:
        raise PDFProcessingError(
            "No extractable text found. The PDF may be a scanned image (OCR is not supported)."
        )
    return result
