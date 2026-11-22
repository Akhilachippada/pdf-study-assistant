"""End-to-end pipeline tests using a fake embedding model and a fake LLM.

These need the real LangChain + FAISS packages (pip install -r requirements-dev.txt), but
no internet, no API key and no model download.
"""
import re
import unittest
import zlib
from dataclasses import replace

try:
    import numpy as np
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from config import Settings
    from src.rag_pipeline import RAGError, RAGPipeline
    from src.prompt_templates import NOT_FOUND_MESSAGE
    from tests.helpers import canvas, make_blank_pdf, make_text_pdf

    DEPS_AVAILABLE = canvas is not None
except ImportError:  # pragma: no cover - dependencies missing
    DEPS_AVAILABLE = False
    Embeddings = object  # lets the class below be defined


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings: texts sharing words get similar vectors."""

    DIM = 256

    def _embed(self, text: str) -> list[float]:
        vec = np.zeros(self.DIM, dtype="float32")
        for token in re.findall(r"\w+", text.lower()):
            vec[zlib.crc32(token.encode()) % self.DIM] += 1.0
        norm = np.linalg.norm(vec)
        return (vec / norm if norm else vec).tolist()

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


@unittest.skipUnless(DEPS_AVAILABLE, "LangChain/FAISS/reportlab not installed")
class RAGPipelineTests(unittest.TestCase):
    def setUp(self):
        self.settings = replace(Settings(), chunk_size=200, chunk_overlap=20, top_k=3, max_distance=1.4)
        self.files = [
            ("bio.pdf", make_text_pdf(["Photosynthesis converts light energy into chemical energy in chloroplasts."])),
            ("history.pdf", make_text_pdf(["The French Revolution began in 1789 in Paris."])),
        ]

    def _pipeline(self, responses):
        llm = FakeListChatModel(responses=responses)
        return RAGPipeline(HashingEmbeddings(), llm, self.settings)

    def test_ingest_reports_each_file(self):
        pipeline = self._pipeline(["unused"])
        bad = ("blank.pdf", make_blank_pdf())
        report = pipeline.ingest(self.files + [bad, ("x.txt", b"hello")])
        self.assertTrue(pipeline.is_ready)
        self.assertEqual([f.ok for f in report.files], [True, True, False, False])
        self.assertGreater(report.total_chunks, 1)

    def test_ingest_with_only_bad_files_is_not_ready(self):
        pipeline = self._pipeline(["unused"])
        report = pipeline.ingest([("blank.pdf", make_blank_pdf())])
        self.assertFalse(report.ready)
        self.assertFalse(pipeline.is_ready)

    def test_answer_comes_with_correct_source(self):
        pipeline = self._pipeline(["Light energy becomes chemical energy."])
        pipeline.ingest(self.files)
        result = pipeline.ask("What does photosynthesis convert into chemical energy?")
        self.assertTrue(result.found)
        self.assertEqual(result.answer, "Light energy becomes chemical energy.")
        self.assertEqual(result.sources[0].filename, "bio.pdf")
        self.assertEqual(result.sources[0].page, 1)

    def test_unrelated_question_returns_not_found_without_llm(self):
        pipeline = self._pipeline(["THIS MUST NOT BE USED"])
        pipeline.ingest(self.files)
        result = pipeline.ask("Explain quantum computing qubits")
        self.assertFalse(result.found)
        self.assertEqual(result.answer, NOT_FOUND_MESSAGE)
        self.assertEqual(result.sources, [])

    def test_model_saying_not_found_hides_sources(self):
        pipeline = self._pipeline([NOT_FOUND_MESSAGE])
        pipeline.ingest(self.files)
        result = pipeline.ask("When did the French Revolution begin?")
        self.assertFalse(result.found)
        self.assertEqual(result.sources, [])

    def test_follow_up_is_rewritten_before_retrieval(self):
        # 1st fake response = rewritten question, 2nd = final answer.
        pipeline = self._pipeline(["What does photosynthesis convert into chemical energy?", "Final answer"])
        pipeline.ingest(self.files)
        history = [
            {"role": "user", "content": "What is photosynthesis?"},
            {"role": "assistant", "content": "A process in plants."},
        ]
        result = pipeline.ask("What does it convert?", history)
        self.assertEqual(result.standalone_question, "What does photosynthesis convert into chemical energy?")
        self.assertEqual(result.answer, "Final answer")
        self.assertEqual(result.sources[0].filename, "bio.pdf")

    def test_ask_before_ingest_raises(self):
        with self.assertRaises(RAGError):
            self._pipeline(["x"]).ask("Anything?")

    def test_ask_without_llm_raises(self):
        pipeline = RAGPipeline(HashingEmbeddings(), None, self.settings)
        pipeline.ingest(self.files)
        with self.assertRaises(RAGError):
            pipeline.ask("What is photosynthesis?")

    def test_empty_question_raises(self):
        pipeline = self._pipeline(["x"])
        pipeline.ingest(self.files)
        with self.assertRaises(RAGError):
            pipeline.ask("   ")


if __name__ == "__main__":
    unittest.main()
