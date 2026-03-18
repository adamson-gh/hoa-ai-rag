# HOA Document AI

> AI assistant that answers HOA rules questions from governing documents using retrieval-augmented generation (RAG).

A retrieval-augmented AI assistant for answering homeowner questions from HOA governing documents such as declarations, bylaws, rules, and policies.

This project ingests HOA PDFs, chunks and indexes them with embeddings, retrieves the most relevant excerpts for a user question, and asks an LLM to answer **only from the retrieved evidence**. It also distinguishes between answers that are **explicitly stated** and answers that are **inferred from related rules**.

## What this project demonstrates

- End-to-end RAG pipeline design for messy, real-world documents
- PDF ingestion, cleaning, chunking, and metadata extraction
- Vector search with reranking and context trimming
- Evidence-aware prompting and answer calibration
- Debug logging for retrieval inspection and iteration
- Multi-corpus support so different HOAs can be indexed separately

## Repository contents

- `app.py` — command-line HOA assistant
- `ingest_hoa_docs.py` — ingestion and indexing pipeline
- `prompts/system.txt` — system prompt used for answer behavior
- `requirements.txt` — Python dependencies

## Architecture

- **Ingestion** → PDF text extraction, cleaning, chunking, quality scoring, metadata
- **Embeddings** → `sentence-transformers` with `all-MiniLM-L6-v2`
- **Indexing** → FAISS vector index per HOA corpus
- **Retrieval** → semantic retrieval plus lexical/title/metadata boosts
- **Reranking** → chunk quality, overlap scoring, and context trimming
- **Generation** → local Ollama model (`phi4:14b`) with grounded prompting
- **Calibration** → explicit vs inferred handling and refusal when evidence is weak

## How it works

1. Put HOA source PDFs into `data/<hoa_name>/hoa_docs/`
2. Run ingestion to extract text, chunk documents, score quality, and build an index
3. Ask questions against that HOA’s indexed corpus
4. The assistant retrieves the best excerpts, trims weak context, and answers with uncertainty handling

## Key design choices

### Grounded answers only
The assistant is instructed to answer only from retrieved excerpts and to avoid inventing rules.

### Explicit vs inferred handling
If a rule is only implied by related language, the assistant says so clearly instead of presenting it as an explicit rule.

### Retrieval tuning over hardcoded Q&A
The system uses embeddings, metadata, lexical overlap, section titles, and reranking rather than brittle question-answer templates.

### Separate corpora per HOA
Each HOA can have its own source documents, processed chunks, and vector index, which prevents cross-HOA contamination.

## Example usage

### 1) Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Place source documents

Put PDFs here:

```text
data/arora/hoa_docs/
data/milestone/hoa_docs/
```

### 4) Build an index

```powershell
python .\ingest_hoa_docs.py arora
python .\ingest_hoa_docs.py milestone
```

### 5) Run the assistant

```powershell
python .\app.py arora
python .\app.py milestone
```

## Example questions

- `Can I park overnight on the street?`
- `Do I need approval to change my exterior?`
- `What happens if I violate the rules?`
- `Can owners rent out their homes?`

## Example Q&A

### Example 1 — Pet nuisance

**Question:** Is my dog allowed to bark a lot?

**Answer (shortened):**  
No, not if the barking becomes a nuisance. The retrieved excerpts allow domestic pets but also prohibit nuisances and support enforcement if a violation occurs.

**Why this matters:**  
This shows the assistant can combine a specific section like **Animals** with broader nuisance language and still answer cautiously.

---

### Example 2 — Leasing

**Question:** Can owners rent out their homes?

**Answer (shortened):**  
Leasing appears to be contemplated by the governing documents, but the retrieved text does not define a dedicated leasing-rules section or specific leasing restrictions.

**Why this matters:**  
This shows the assistant distinguishes between **indirect support** and an **explicit rule**, instead of overstating the answer.

---

### Example 3 — Missing information

**Question:** What should I do if I lose my mailbox key?

**Answer (shortened):**  
I couldn’t find a strong enough match in the HOA documents to answer confidently.

**Why this matters:**  
This demonstrates refusal behavior when the evidence is missing, instead of guessing.

## Notes

- This repo intentionally does **not** include real HOA documents, indexes, logs, or generated chunks.
- The app expects a local Ollama server for generation.
- Default model in `app.py`: `phi4:14b`
- Default embedding model: `all-MiniLM-L6-v2`

## Ollama setup

Install Ollama, then pull the model used by the app:

```powershell
ollama pull phi4:14b
```

Start Ollama if needed, then run the app.

## Main dependencies

- `sentence-transformers`
- `faiss-cpu`
- `numpy`
- `pypdf`

## What I learned building this

- Retrieval quality matters more than model choice
- Titles and metadata improve legal/document search significantly
- Good AI systems must distinguish explicit evidence from inference
- Missing information should be surfaced honestly, not guessed
- Logs are essential for improving retrieval and answer behavior

## Future improvements

- Streamlit or web UI
- Better section-title extraction during ingestion
- More systematic evaluation set and metrics
- Optional lexical/BM25 retrieval layer
- Deployment path for public HOA self-service use

## License

MIT
