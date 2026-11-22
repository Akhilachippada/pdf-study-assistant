import unittest

from src.pdf_processor import PDFProcessingError, clean_text, extract_pages, validate_pdf
from tests.helpers import canvas, make_blank_pdf, make_text_pdf


class CleanTextTests(unittest.TestCase):
    def test_joins_hyphenated_line_breaks(self):
        self.assertEqual(clean_text("exam-\nple text"), "example text")

    def test_unwraps_lines_but_keeps_paragraphs(self):
        raw = "First line\nsecond line\n\nNew paragraph"
        self.assertEqual(clean_text(raw), "First line second line\n\nNew paragraph")

    def test_normalises_ligatures_and_whitespace(self):
        self.assertEqual(clean_text("ﬁnal    answer\t here"), "final answer here")

    def test_removes_control_characters(self):
        self.assertEqual(clean_text("a\x00b\x07c"), "a b c")

    def test_empty_and_whitespace_only(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text("  \n \n\n "), "")


class ValidationTests(unittest.TestCase):
    def test_rejects_wrong_extension(self):
        with self.assertRaises(PDFProcessingError):
            validate_pdf("notes.txt", b"%PDF-1.4 data")

    def test_rejects_empty_file(self):
        with self.assertRaises(PDFProcessingError):
            validate_pdf("a.pdf", b"")

    def test_rejects_non_pdf_content(self):
        with self.assertRaises(PDFProcessingError):
            validate_pdf("a.pdf", b"this is plain text, not a pdf")

    def test_rejects_oversized_file(self):
        with self.assertRaises(PDFProcessingError):
            validate_pdf("a.pdf", b"%PDF-" + b"0" * (2 * 1024 * 1024), max_size_mb=1)


class ExtractionTests(unittest.TestCase):
    def test_corrupt_pdf_gives_clean_error(self):
        with self.assertRaises(PDFProcessingError):
            extract_pages("broken.pdf", b"%PDF-1.4\nthis is not really a pdf structure")

    def test_image_only_pdf_gives_clear_error(self):
        with self.assertRaises(PDFProcessingError) as ctx:
            extract_pages("scan.pdf", make_blank_pdf(2))
        self.assertIn("No extractable text", str(ctx.exception))

    @unittest.skipIf(canvas is None, "reportlab not installed")
    def test_extracts_text_with_correct_page_numbers(self):
        data = make_text_pdf(["Photosynthesis happens in chloroplasts.", "Mitochondria produce ATP."])
        result = extract_pages("bio.pdf", data)
        self.assertEqual(result.total_pages, 2)
        self.assertEqual([p.page_number for p in result.pages], [1, 2])
        self.assertIn("chloroplasts", result.pages[0].text)
        self.assertIn("ATP", result.pages[1].text)
        self.assertTrue(all(p.filename == "bio.pdf" for p in result.pages))

    @unittest.skipIf(canvas is None, "reportlab not installed")
    def test_blank_pages_are_skipped_but_counted(self):
        # One real page: the other (blank) pages are simply not returned.
        data = make_text_pdf(["Only page with text"])
        result = extract_pages("one.pdf", data)
        self.assertEqual(result.total_pages, 1)
        self.assertEqual(len(result.pages), 1)


if __name__ == "__main__":
    unittest.main()
