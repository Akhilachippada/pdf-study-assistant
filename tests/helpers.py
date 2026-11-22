"""Helpers that build tiny PDFs in memory for the tests."""
import io

from pypdf import PdfWriter

try:
    from reportlab.pdfgen import canvas
except ImportError:  # reportlab is a dev-only dependency
    canvas = None


def make_text_pdf(pages: list[str]) -> bytes:
    """Create a PDF with one line of text per page (needs reportlab)."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    for text in pages:
        pdf.drawString(72, 750, text)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def make_blank_pdf(num_pages: int = 1) -> bytes:
    """Create a valid PDF whose pages contain no text (like a scanned document)."""
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=612, height=792)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()
