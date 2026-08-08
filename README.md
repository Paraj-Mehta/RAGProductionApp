# QueryDocs

A local, production-oriented Retrieval-Augmented Generation (RAG) application. Upload PDF documents, ask questions in natural language, and receive answers grounded strictly in the content of your uploaded files — powered by BGE-M3 embeddings, Qdrant vector storage, and Google Gemini for answer generation.

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [RAG Pipelines](#rag-pipelines)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Component Setup](#component-setup)
  - [Qdrant (Vector Database)](#qdrant-vector-database)
  - [BGE-M3 (Embedding Model)](#bge-m3-embedding-model)
  - [Google Gemini (LLM)](#google-gemini-llm)
  - [Inngest (Workflow Orchestration)](#inngest-workflow-orchestration)
  - [FastAPI (Backend)](#fastapi-backend)
  - [Streamlit (Frontend)](#streamlit-frontend)
- [Running the Project](#running-the-project)
- [Author](#author)
---

## Features

**Currently Implemented**

- Upload PDF documents via a Streamlit web interface
- Automatic PDF text extraction and sentence-level chunking
- Local embedding generation using BGE-M3 (1024-dimensional vectors, CPU/GPU)
- Vector storage and similarity search using a local Qdrant instance
- Event-driven ingestion and query workflows via Inngest
- Answer generation using Google Gemini, strictly based on retrieved context
- Source chunk display alongside generated answers
- Configurable top-K retrieval parameter


---

## Architecture Overview
Streamlit Frontend  :8501
        │
        │  Inngest Events
        ▼
Inngest Dev Server  :8288
        │
        │  HTTP
        ▼
  FastAPI Backend   :8000
        │
        ├──────────────────────────┐
        ▼                          ▼
  BGE-M3 (local)           Google Gemini API
  Embeddings               Answer Generation
        │
        ▼
  Qdrant (Docker)   :6333
  Vector Storage
  ---

## RAG Pipelines

### PDF Ingestion Pipeline
PDF Upload → Save PDF → Inngest Event (rag/ingest_pdf)
→ PDFReader → Extract Text → SentenceSplitter → Chunks
→ BGE-M3 → 1024-dim Embeddings → Qdrant (Store Vectors + Payload)

---

### Question-Answering Pipeline

User Question → Inngest Event (rag/query_pdf_ai) → BGE-M3
→ Query Embedding → Qdrant query_points() → Top-K Chunks
→ Build Context → Google Gemini → Answer + Sources → Streamlit

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **Backend** | FastAPI + Uvicorn |
| **Workflow Orchestration** | Inngest (local dev server) |
| **PDF Processing** | LlamaIndex PDFReader, SentenceSplitter |
| **Embedding Model** | BAAI/bge-m3 (Sentence Transformers, local) |
| **Vector Database** | Qdrant (Docker) |
| **LLM (Generation)** | Google Gemini (`google-genai` SDK) |
| **Package Manager** | uv |
| **Runtime** | Python 3.11+ |
| **Inngest Dev Server** | Node.js / npm |

---

## Project Structure
RAGProductionApp/
│
├── .venv/                        # Virtual environment (not committed)
├── uploads/                      # Locally saved uploaded PDFs (not committed)
│
├── src/
│   └── ragproductionapp/
│       ├── __init__.py           # Package init
│       ├── main.py               # FastAPI app + Inngest endpoint registration
│       ├── data_loader.py        # PDF reading and chunking logic (PDFReader + SentenceSplitter)
│       ├── vector_db.py          # Qdrant client setup, collection management, vector operations
│       └── ...                   # Additional modules as the project grows
│
├── frontend.py                   # Streamlit frontend: upload, query, display answers
├── .env                          # Environment variables (never committed)
├── .gitignore
├── .python-version               # Pinned Python version for uv
├── pyproject.toml                # Project metadata and dependencies
├── uv.lock                       # Locked dependency versions (commit this)
└── README.md

> **Note:** The project structure may evolve as new features are added.

**Key files explained:**

- **`main.py`** — Registers the FastAPI app and the Inngest event handler endpoint (`/api/inngest`). Defines Inngest functions that respond to `rag/ingest_pdf` and `rag/query_pdf_ai` events.
- **`data_loader.py`** — Uses LlamaIndex `PDFReader` to extract text and `SentenceSplitter` to split it into overlapping chunks suitable for embedding.
- **`vector_db.py`** — Manages the Qdrant client connection, collection creation/validation, upserting vectors with payloads, and querying for nearest neighbours.
- **`frontend.py`** — Streamlit app that handles file uploads, sends Inngest events, polls for results, and renders answers and sources.

---

## Prerequisites

Ensure the following are installed on your machine before proceeding:

| Tool | Purpose | Minimum Version |
|---|---|---|
| Python | Runtime | 3.11+ |
| [uv](https://docs.astral.sh/uv/) | Package & venv manager | Latest |
| Node.js / npm | Run Inngest Dev Server | LTS |
| Docker Desktop | Run Qdrant locally | Latest |
| Git | Clone the repository | Latest |

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd RAGProductionApp
```

### 2. Create a virtual environment

```bash
uv venv
```

### 3. Activate the virtual environment (PowerShell on Windows)

```powershell
.venv\Scripts\Activate.ps1
```

> On macOS/Linux: `source .venv/bin/activate`

### 4. Install all dependencies

```bash
uv add fastapi uvicorn streamlit inngest llama-index-core llama-index-readers-file torch sentence-transformers qdrant-client google-genai python-dotenv requests
```

> `uv.lock` is automatically updated and should be committed to Git to ensure reproducible installs across environments.

---

## Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
INNGEST_API_BASE=http://127.0.0.1:8288/v1
```

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Your Google Gemini API key for answer generation |
| `INNGEST_API_BASE` | Base URL for the local Inngest Dev Server API |

> **Never commit `.env` to version control.** The `.gitignore` below covers this.

**Recommended `.gitignore` entries:**

```gitignore
.venv/
.env
__pycache__/
uploads/
.vscode/
*.log
.pytest_cache/
.mypy_cache/
```

---

## Component Setup

### Qdrant (Vector Database)

Qdrant is an open-source, high-performance vector database. It stores the 1024-dimensional embeddings produced by BGE-M3 along with metadata payloads (source filename, raw text chunk). Similarity search uses **cosine distance**.

**First-time setup — create and start the Qdrant container:**

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

| Flag | Purpose |
|---|---|
| `-d` | Run in detached (background) mode |
| `--name qdrant` | Name the container for easy reference |
| `-p 6333:6333` | Expose REST API port |
| `-p 6334:6334` | Expose gRPC port |
| `-v qdrant_storage:/qdrant/storage` | Persist data across restarts |

**Subsequent starts (after the container already exists):**

```bash
docker start qdrant
```

**Stop Qdrant:**

```bash
docker stop qdrant
```

**Check container status:**

```bash
docker ps
```

**Qdrant Web Dashboard:** [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

**Collection details:**

| Setting | Value |
|---|---|
| Collection name | `docs` |
| Vector dimension | `1024` |
| Distance metric | `COSINE` |

> **Important:** If you previously created the `docs` collection with a different embedding model (e.g., OpenAI `text-embedding-3-large` which produces **3072-dimensional** vectors), you must delete and recreate the collection before using BGE-M3 (1024 dimensions). Mixing vectors of different dimensions in the same collection will cause errors. Delete the collection via the Qdrant dashboard or the Qdrant Python client before re-ingesting documents.

---

### BGE-M3 (Embedding Model)

[BGE-M3](https://huggingface.co/BAAI/bge-m3) is a state-of-the-art multilingual embedding model developed by BAAI. It converts text chunks and user questions into 1024-dimensional dense vectors that capture semantic meaning.

**Why BGE-M3 is used locally:**

- Runs entirely on your machine — no paid embedding API required
- Supports CPU and GPU (automatically uses GPU if CUDA is available)
- Produces high-quality 1024-dimensional embeddings suitable for semantic search
- Completely separate from the Gemini LLM — BGE-M3 handles *understanding and retrieval*, Gemini handles *generation*

**Role separation:**

**First-time use:** The model weights are downloaded from Hugging Face on first run and cached locally. This may take a few minutes depending on your connection. Subsequent runs load from cache and are fast.

---

### Google Gemini (LLM)

Gemini is used exclusively for **answer generation** — it receives the retrieved text chunks as context along with the user's question and produces a grounded natural-language answer. It is not used for embeddings.

**Install the SDK (already included in the `uv add` command above):**

```bash
uv add google-genai
```

**Set your API key in `.env`:**

```env
GEMINI_API_KEY=your_gemini_api_key
```

Obtain an API key from [Google AI Studio](https://aistudio.google.com/).

> The Gemini API call is made directly using the `google-genai` Python SDK inside an Inngest workflow step. The current installed version of the Inngest Python SDK (`inngest v0.5.19`) does not include an `ai.gemini.Adapter`, so the SDK is called directly.

---

### Inngest (Workflow Orchestration)

[Inngest](https://www.inngest.com/) provides event-driven, durable workflow orchestration. Instead of handling PDF ingestion and RAG queries in synchronous HTTP endpoints, the application sends named events that trigger background Inngest functions. This gives:

- **Background processing** — long-running embedding and retrieval steps don't block the UI
- **Durable steps** — each step within a function is individually retried on failure
- **Observability** — the Inngest Dev Server dashboard shows every event, function run, and step result in real time

**Current events:**

| Event Name | Triggered By | Purpose |
|---|---|---|
| `rag/ingest_pdf` | Streamlit upload | Runs the full PDF → chunk → embed → store pipeline |
| `rag/query_pdf_ai` | Streamlit question submission | Runs embed query → retrieve → generate answer pipeline |

**Ingestion event payload:**

```python
inngest.Event(
    name="rag/ingest_pdf",
    data={
        "pdf_path": str(pdf_path.resolve()),  # Absolute local path to the saved PDF
        "source_id": pdf_path.name,           # Filename used as metadata/source identifier
    },
)
```

**Query event payload:**

```python
inngest.Event(
    name="rag/query_pdf_ai",
    data={
        "question": question,   # The user's natural language question
        "top_k": top_k,         # Number of chunks to retrieve (default: 5)
    },
)
```

**Start the Inngest Dev Server** (requires Node.js/npm):

```bash
npx inngest-cli@latest dev --url http://localhost:8000/api/inngest
```

This tells the Inngest Dev Server to connect to your local FastAPI backend at port 8000. Keep this running in its own terminal.

**Inngest Dashboard:** [http://localhost:8288](http://localhost:8288)

To stop: press `Ctrl+C` in the terminal. To restart: run the same command again.

---

### FastAPI (Backend)

The FastAPI application lives in `src/ragproductionapp/main.py`. It serves as the bridge between the Inngest Dev Server and the application logic (PDF processing, embedding, vector search, Gemini calls).

FastAPI exposes the `/api/inngest` endpoint that the Inngest Dev Server calls to trigger registered functions. All business logic runs inside Inngest function steps called by this endpoint.

**Start the backend:**

```bash
uv run uvicorn --app-dir src ragproductionapp.main:app --reload
```

| URL | Purpose |
|---|---|
| [http://localhost:8000](http://localhost:8000) | FastAPI root |
| [http://localhost:8000/docs](http://localhost:8000/docs) | Swagger / interactive API docs |
| [http://localhost:8000/api/inngest](http://localhost:8000/api/inngest) | Inngest event handler endpoint |

---

### Streamlit (Frontend)

The Streamlit frontend (`frontend.py`) is the user-facing interface. It:

- Accepts PDF file uploads and saves them to the local `uploads/` directory
- Sends the `rag/ingest_pdf` Inngest event to trigger ingestion
- Accepts natural language questions and a configurable `top_k` value
- Sends the `rag/query_pdf_ai` Inngest event to trigger retrieval and generation
- Polls the local Inngest API for function run completion
- Displays the generated answer and the source chunks used to produce it

The frontend communicates exclusively with the Inngest Dev Server and is designed for **local development use**.

**Start the frontend:**

```bash
uv run streamlit run streamlit_app.py
```

**URL:** [http://localhost:8501](http://localhost:8501)

---

## Running the Project

Four services must be running simultaneously. Open four separate terminals.

**Start Qdrant (Docker):**

```bash
docker start qdrant
```

> If starting for the first time, use the full `docker run` command from the [Qdrant setup section](#qdrant-vector-database).

**Terminal 1 — FastAPI Backend:**

```bash
uv run uvicorn --app-dir src ragproductionapp.main:app --reload
```

**Terminal 2 — Inngest Dev Server:**

```bash
npx inngest-cli@latest dev --url http://localhost:8000/api/inngest
```

**Terminal 3 — Streamlit Frontend:**

```bash
uv run streamlit run streamlit_app.py
```


## Author

**Paraj Mehta**  
Computer Science & Engineering  
Pandit Deendayal Energy University (PDEU)

---
