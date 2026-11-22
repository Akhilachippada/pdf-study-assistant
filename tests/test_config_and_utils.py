import os
import unittest
from unittest import mock

from config import Settings, get_settings
from src.models import SourceReference
from src.utils import format_history, group_pages_by_file, truncate


class ConfigTests(unittest.TestCase):
    def test_defaults_and_overrides(self):
        env = {"GROQ_API_KEY": " abc ", "CHUNK_SIZE": "500", "CHUNK_OVERLAP": "50", "TOP_K": "6"}
        with mock.patch.dict(os.environ, env, clear=True):
            settings = get_settings()
        self.assertEqual(settings.groq_api_key, "abc")
        self.assertTrue(settings.has_api_key)
        self.assertEqual((settings.chunk_size, settings.chunk_overlap, settings.top_k), (500, 50, 6))

    def test_missing_key_is_reported(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(Settings().has_api_key)

    def test_overlap_must_be_smaller_than_chunk_size(self):
        with mock.patch.dict(os.environ, {"CHUNK_SIZE": "100", "CHUNK_OVERLAP": "100"}, clear=True):
            with self.assertRaises(ValueError):
                get_settings()

    def test_non_numeric_value_raises(self):
        with mock.patch.dict(os.environ, {"TOP_K": "many"}, clear=True):
            with self.assertRaises(ValueError):
                get_settings()


class UtilsTests(unittest.TestCase):
    def test_format_history(self):
        history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
        self.assertEqual(format_history(history), "User: Hi\nAssistant: Hello")

    def test_group_pages_by_file(self):
        refs = [
            SourceReference("a.pdf", 5, 0.4, "x"),
            SourceReference("a.pdf", 2, 0.5, "y"),
            SourceReference("a.pdf", 5, 0.6, "z"),
            SourceReference("b.pdf", 1, 0.7, "w"),
        ]
        self.assertEqual(group_pages_by_file(refs), {"a.pdf": [2, 5], "b.pdf": [1]})

    def test_truncate(self):
        self.assertEqual(truncate("short text"), "short text")
        self.assertTrue(truncate("word " * 200, 50).endswith("…"))
        self.assertLessEqual(len(truncate("word " * 200, 50)), 50)


if __name__ == "__main__":
    unittest.main()
