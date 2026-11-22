# Project Guide: Understanding the PDF Study Assistant (for interviews)

## 1. The one-minute pitch

> "I built a PDF study assistant using Retrieval-Augmented Generation. Uploaded PDFs are parsed with PyPDF, cleaned, split into overlapping chunks, embedded locally with a Sentence-Transformers model and indexed in FAISS. When a user asks a question, I embed it, retrieve the top-k most similar chunks, and pass only those chunks to a Groq-hosted LLM through a LangChain prompt that forces answers to stay inside the context. The UI is Streamlit and shows the PDF name and page numbers behind every answer. If nothing relevant is found, the app says so instead of hallucinating."

## 2. Why RAG? (the problem it solves)

An LLM alone has three problems for this use case: it knows nothing about *your* PDFs, it has a limited context window (a long book will not fit), and it hallucinates confidently. RAG fixes this by **retrieving** the relevant passages at question time and **augmenting** the prompt with them, so the LLM **generates** from evidence. It is cheaper and faster than sending the whole PDF, and it makes citations possible, because every chunk carries its file and page.

## 3. Architecture

Two phases that share one embedding model:

**Phase A: Ingestion** (when you click *Process documents*)
`PDF bytes → validate → extract per page → clean → chunk → embed → FAISS index`

**Phase B: Question answering** (every chat message)
`question → [rewrite if follow-up] → embed → FAISS top-k search → distance filter → build prompt → LLM → answer + sources`

Layers:
- **Presentation:** `app.py` (Streamlit) and `assets/style.css`
- **Orchestration:** `src/rag_pipeline.py`
- **Components:** `pdf_processor`, `text_splitter`, `embeddings`, `vector_store`, `llm_handler`, `prompt_templates`
- **Support:** `config.py`, `models.py`, `utils.py`

## 4. The RAG pipeline step by step

1. **Validation** (`validate_pdf`): extension is `.pdf`, file non-empty, under the size limit, `%PDF-` header present. Fails fast with a readable message.
2. **Extraction** (`extract_pages`): PyPDF reads page by page. Encrypted files are unlocked with an empty password if possible, else rejected. A page that fails to parse is skipped. A file with zero text (scanned) raises a clear error.
3. **Cleaning** (`clean_text`): NFKC normalisation (ligatures like "ﬁ"), control-character removal, re-joining `exam-\nple` → `example`, unwrapping hard line breaks while preserving paragraph breaks.
4. **Chunking** (`split_pages`): `RecursiveCharacterTextSplitter` with size 1000 and overlap 200. It tries separators in order (`\n\n`, `\n`, `. `, space, character) so splits land on natural boundaries. Each chunk keeps `source`, `page`, `chunk_id`.
5. **Embedding** (`load_embedding_model`): `all-MiniLM-L6-v2` converts each chunk to a 384-dimensional vector, normalised to length 1.
6. **Indexing** (`VectorStoreManager.build`): `FAISS.from_documents` embeds all chunks and stores vectors in a flat (exact) L2 index; LangChain keeps the chunk text and metadata in a docstore keyed to the vector positions.
7. **Question rewriting** (`_condense_question`): with history, the LLM rewrites "What about its limitations?" into a standalone question. This matters because the retriever only sees the query string; "it" would retrieve garbage.
8. **Retrieval** (`VectorStoreManager.search`): the rewritten question is embedded by the same model, and FAISS returns the `TOP_K=4` nearest chunks with distances. Chunks beyond `MAX_DISTANCE` are dropped. If none remain → "not found" answer, no LLM call (saves cost and prevents hallucination).
9. **Prompt construction** (`build_context` + `QA_PROMPT`): chunks are labelled `[Source: file.pdf, page 3]` and joined; the system prompt says "use only this context, else reply with this exact sentence".
10. **Generation:** the LCEL chain `QA_PROMPT | llm | StrOutputParser()` calls Groq and returns text.
11. **Post-processing** (`ask`): if the model returned the not-found sentence, sources are hidden; otherwise `build_sources` de-duplicates by (file, page), keeps the best score and attaches excerpts.
12. **Display** (`app.py`): the answer, source tags (`📄 notes.pdf · p. 3, 5`) and an expander with retrieved excerpts.

## 5. How LangChain, embeddings, FAISS and the LLM work together

- **Embeddings** turn text into vectors where *similar meaning = nearby points*. "Mitochondria make energy" and "What produces ATP in a cell?" land close together even with few shared words, which keyword search cannot do.
- **FAISS** is a library for fast nearest-neighbour search over vectors. It only knows vectors and integer ids. It has no idea what text they came from.
- **LangChain** is the glue: `Document` objects (text + metadata) carry data between steps; the `FAISS` wrapper stores the id → Document mapping so a search returns chunks, not raw numbers; `ChatPromptTemplate` fills the prompt; the `|` operator (LCEL) chains prompt → model → parser; `HuggingFaceEmbeddings` and `ChatGroq` are adapters that give different providers one common interface (swapping the LLM is a one-file change).
- **The LLM** never searches anything. It only reads the prompt that retrieval assembled and writes an answer. Retrieval quality therefore caps answer quality.

## 6. Data flow: PDF upload to answer

```
Streamlit file_uploader ──▶ [(filename, bytes)] ──▶ RAGPipeline.ingest()
   ├─ extract_pages()   → ExtractionResult(PageContent(filename, page_number, text)…)
   ├─ split_pages()     → [Document(page_content, metadata{source,page,chunk_id})…]
   └─ VectorStoreManager.build() → FAISS index   (+ IngestionReport shown in sidebar)

st.chat_input ──▶ RAGPipeline.ask(question, history)
   ├─ _condense_question()                    → standalone question
   ├─ VectorStoreManager.search()             → [(Document, distance)…]
   ├─ build_context() → QA_PROMPT → ChatGroq  → answer text
   └─ build_sources()                         → RAGAnswer(answer, sources[SourceReference])
render_message() ──▶ answer + source tags + excerpts
```

## 7. File-by-file explanation

| File | Purpose and key ideas |
|---|---|
| `config.py` | Frozen `Settings` dataclass read from env vars / `.env` via python-dotenv. Validates (`overlap < size`). Keeps keys out of code. |
| `src/pdf_processor.py` | Validation, extraction, cleaning. Custom `PDFProcessingError` gives friendly errors. Returns page-level `PageContent`, so page citations are exact. No LangChain dependency, which keeps it easy to unit-test. |
| `src/text_splitter.py` | Wraps `RecursiveCharacterTextSplitter`; converts pages to `Document`s with metadata and adds `chunk_id`. |
| `src/embeddings.py` | Builds `HuggingFaceEmbeddings` (MiniLM, CPU, normalised). Lazy import because torch is slow to load. |
| `src/vector_store.py` | `VectorStoreManager`: `build`, `search` (with distance filter), `clear`. Hides FAISS behind a small interface, so another vector DB could replace it. |
| `src/prompt_templates.py` | `QA_PROMPT` (grounding rules + exact "not found" sentence), `CONDENSE_PROMPT` (follow-up rewriting). Includes a basic prompt-injection guard ("context is not instructions"). |
| `src/llm_handler.py` | Creates `ChatGroq` or raises `MissingAPIKeyError`. Temperature 0 for factual consistency. |
| `src/rag_pipeline.py` | `RAGPipeline.ingest` and `.ask`. Dependency injection (embeddings and LLM passed in) enables testing with fakes. Wraps failures in `RAGError`. |
| `src/models.py` | Dataclasses: `SourceReference`, `RAGAnswer`, `FileReport`, `IngestionReport`. |
| `src/utils.py` | History formatting, grouping pages per file, text truncation. |
| `app.py` | Streamlit UI. `st.session_state` keeps chat and pipeline across reruns; `st.cache_resource` loads the embedding model once; `st.status` shows processing; `st.chat_input` / `st.chat_message` build the chat. |
| `assets/style.css` | Header gradient, file chips, source tags, mobile media query. |
| `tests/` | Unit tests for cleaning/validation/extraction/config/utils, plus pipeline tests with a fake embedding model and fake LLM (no internet, no key). |

**Streamlit execution model (common gotcha):** Streamlit re-runs `app.py` top to bottom on every interaction. Anything that must survive (chat, the FAISS index) lives in `st.session_state`; anything expensive and shareable (the embedding model) is wrapped in `st.cache_resource`.

## 8. Design decisions you should be able to defend

- **Local embeddings + hosted LLM:** embeddings are free, private, and need no key; only the generation step uses an API.
- **FAISS flat index:** exact search; for thousands of chunks it is instant. Approximate indexes (IVF, HNSW) matter at millions of vectors.
- **Normalised embeddings:** with unit vectors, squared L2 distance = 2 − 2·cosine, so ranking by L2 equals ranking by cosine, and the threshold `MAX_DISTANCE=1.4` ≈ cosine > 0.3.
- **Chunk 1000 / overlap 200:** large enough for a self-contained idea, small enough to be precise; MiniLM reads up to 256 tokens, and ~1000 characters is roughly 200–250 tokens, so chunks fit. Overlap protects ideas cut at a boundary.
- **Distance threshold before the LLM:** a deterministic guardrail against hallucination plus cost saving.
- **Question rewriting:** makes follow-ups work with a stateless retriever.
- **Temperature 0:** minimises creative drift.

## 9. Interview questions and answers

**Q1. What is RAG and why use it instead of fine-tuning?**
RAG retrieves relevant external text at query time and gives it to the LLM. Compared with fine-tuning it needs no training, updates instantly when documents change, supports citations, and is far cheaper. Fine-tuning is better for changing style or behaviour, not for injecting facts.

**Q2. Why not send the whole PDF to the LLM?**
Context-window limits, cost and latency scale with tokens, and accuracy drops when relevant facts are buried in long inputs ("lost in the middle"). Retrieval sends only the few relevant chunks.

**Q3. What is an embedding?**
A fixed-length vector (384 numbers here) produced by a neural network so that semantically similar texts are close in vector space.

**Q4. Why Sentence-Transformers / MiniLM?**
It is small (~90 MB), fast on CPU, free, and good at semantic similarity for English. Trade-off: lower accuracy than larger models (e.g. bge-large) and a 256-token input limit.

**Q5. What does FAISS do, and why not a normal database?**
FAISS performs efficient nearest-neighbour search on dense vectors. A normal SQL/keyword search matches exact terms; vector search matches meaning. A managed vector DB (Pinecone, Chroma, pgvector) would add persistence and filtering.

**Q6. What index type did you use and what are the alternatives?**
LangChain's FAISS wrapper uses `IndexFlatL2` (exact, brute force). Alternatives: IVF (clustering), HNSW (graph), PQ (compression) for approximate search at scale.

**Q7. L2 distance vs cosine similarity?**
Cosine compares direction; L2 compares position. For normalised vectors they are monotonically related (d² = 2 − 2cos), so I normalise and use L2. Lower distance is better, which is the reverse of a similarity score.

**Q8. How did you choose chunk size and overlap?**
Trade-off: small chunks give precise retrieval but lose context; large chunks keep context but dilute the embedding and waste prompt space. 1000/200 is a common starting point, and I made it configurable. I would tune it with a set of test questions and measure retrieval hit-rate.

**Q9. Why RecursiveCharacterTextSplitter?**
It prefers paragraph, then sentence, then word boundaries, so chunks stay semantically coherent, unlike a fixed-width cut that can slice mid-sentence.

**Q10. How do you stop hallucinations?**
Layered: (1) retrieval threshold, so no relevant chunks means no LLM call; (2) strict system prompt with an exact refusal sentence; (3) temperature 0; (4) citations so users can verify. None of it is a mathematical guarantee. Prompt-based grounding is probabilistic.

**Q11. How do follow-up questions work?**
The last few messages and the new question go to the LLM, which rewrites it into a standalone question. That query is used for retrieval, since the retriever has no memory.

**Q12. How are page numbers tracked?**
Extraction is per page; each page's text becomes a `Document` with `{source, page}` metadata; the splitter copies metadata to every chunk; after retrieval I read the metadata back. Chunks never span pages, so citations are exact.

**Q13. What is LCEL?**
LangChain Expression Language: composing runnables with `|`, e.g. `prompt | llm | parser`. Every component has `invoke`/`stream`/`batch`, and the chain is itself a runnable. It replaces legacy `LLMChain`/`RetrievalQA` classes.

**Q14. Which LangChain parts did you avoid because they're deprecated, and why?**
Old `langchain.chains` (`RetrievalQA`, `ConversationalRetrievalChain`) and `ConversationBufferMemory`. I used explicit LCEL chains and my own history handling, which is also easier to understand and test.

**Q15. What happens with a scanned PDF?**
PyPDF returns empty text, so I raise a clear `PDFProcessingError` suggesting it is image-only. The fix is OCR (e.g. Tesseract/`pytesseract`) before extraction.

**Q16. How did you test without an API key or internet?**
Dependency injection: `RAGPipeline` accepts any embeddings and LLM. Tests use a deterministic hashing embedding class and `FakeListChatModel`, and generate PDFs with reportlab.

**Q17. How would you handle a very large corpus (100k pages)?**
Persist the index (`save_local`/`load_local`) or move to a vector DB, use an approximate FAISS index (HNSW/IVF), batch embedding with a GPU, add metadata filters, hybrid BM25 + vector search, and a re-ranker.

**Q18. What is hybrid search and a re-ranker?**
Hybrid combines keyword (BM25) and vector scores, helping with exact terms, IDs and acronyms. A cross-encoder re-ranker re-scores the top 20–50 retrieved chunks jointly with the query for better top-k precision.

**Q19. How would you evaluate the system?**
Retrieval: hit-rate/recall@k and MRR on a labelled question set. Generation: faithfulness (is every claim supported by context?), answer relevance, and citation accuracy, using tools like RAGAS or LLM-as-judge plus human review.

**Q20. What is prompt injection and does your app defend against it?**
A document may contain text like "ignore previous instructions". My prompt tells the model to treat context as data, not commands. This reduces but doesn't eliminate the risk; stronger defences include output filtering and sandboxing tools.

**Q21. Security and privacy considerations?**
API key in `.env` (git-ignored), never in code; PDFs are processed in memory and not stored; embeddings run locally, but retrieved chunks and questions are sent to Groq, so confidential documents need a local LLM (e.g. Ollama) or a provider with suitable data terms.

**Q22. Why `temperature=0`?**
Factual, repeatable answers. Higher temperature increases randomness and the chance of straying from context.

**Q23. What are the limitations of your system?**
No OCR, tables/figures not understood, in-memory index rebuilt per session, English-optimised embeddings, fixed top-k that may miss multi-hop answers across distant sections, and grounding is prompt-based.

**Q24. How would you add streaming responses?**
Use `chain.stream(...)` and Streamlit's `st.write_stream` to display tokens as they arrive.

**Q25. Why is embedding model consistency important?**
Queries and chunks must be embedded by the same model, otherwise they live in different vector spaces and distances are meaningless. Changing the model requires re-indexing.

**Q26. What does `top_k` control, and what happens if it's too high or low?**
Number of chunks sent to the LLM. Too low risks missing evidence; too high adds noise, cost and can distract the model.

**Q27. What is the difference between a token and a character limit here?**
The splitter counts characters (simple, no tokenizer needed); the embedding model and LLM count tokens (~4 characters per English token). I picked sizes that comfortably fit both.

**Q28. If you had one more week, what would you add?**
OCR, a persisted index with per-user collections, a re-ranker, streaming, quiz/flashcard generation, and an automated evaluation set.
