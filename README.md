# JF_RAG — Jeffrey's Archive

> A production-grade fullstack **MultiModal RAG** (Retrieval-Augmented Generation) system purpose-built for investigative and legal document research.  
> Ingests **PDFs, scanned images, and CSV records**. Retrieves using hybrid BM25 + FAISS with HyDE, query rewriting, cross-encoder reranking, and parent-child context expansion. Generates answers via Groq or Ollama. Ships with a polished React + TypeScript frontend and a Vapi-powered AI voice assistant.

---

## 🚀 Live Deployment

| Service | URL | Platform | Status |
|---|---|---|---|
| **Frontend** | [https://jfrag.vercel.app](https://jfrag.vercel.app) | Vercel | ✅ Live |
| **Backend API** | [https://adikaprojects--jf-rag-backend-fastapi-app.modal.run](https://adikaprojects--jf-rag-backend-fastapi-app.modal.run) | Modal.run | ✅ Live |
| **API Health** | [/health](https://adikaprojects--jf-rag-backend-fastapi-app.modal.run/health) | Modal.run | ✅ Live |
| **API Docs** | [/docs](https://adikaprojects--jf-rag-backend-fastapi-app.modal.run/docs) | Modal.run | ✅ Live |

### Infrastructure

| Component | Detail |
|---|---|
| **Frontend host** | Vercel — auto-deploys from `main` branch on every push |
| **Backend host** | Modal.run — Python 3.11 container, 4 GB RAM, 2 vCPU, scales to zero |
| **Vector indexes** | Cloudflare R2 (`jf-rag-data` bucket) — FAISS text, BM25, image FAISS |
| **LLM** | Groq API (`llama-3.3-70b-versatile`) with key rotation |
| **Cold start** | ~30s (models pre-baked into Modal image layer) |
| **Idle scale-down** | 5 minutes after last request |

### Redeploy (backend)

```powershell
cd d:\JF_RAG
Remove-Item Env:MODAL_TOKEN_ID -ErrorAction SilentlyContinue
Remove-Item Env:MODAL_TOKEN_SECRET -ErrorAction SilentlyContinue
modal deploy modal_app.py
```

Frontend redeploys automatically on every `git push` to `main`.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Features](#features)
4. [Environment Variables](#environment-variables)
5. [Setup — Backend](#setup--backend)
6. [Setup — Frontend](#setup--frontend)
7. [Running the System](#running-the-system)
8. [Data Ingestion Pipelines](#data-ingestion-pipelines)
9. [Retrieval System](#retrieval-system)
10. [LLM Router](#llm-router)
11. [Guardrails](#guardrails)
12. [API Reference](#api-reference)
13. [Frontend](#frontend)
14. [Voice Assistant (Vapi)](#voice-assistant-vapi)
15. [Cloud Storage (Cloudflare R2)](#cloud-storage-cloudflare-r2)
16. [Unredaction Tools](#unredaction-tools)
17. [Deployment](#deployment)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        React + TypeScript UI                        │
│   (Vite · Tailwind · Radix UI · framer-motion · react-markdown)     │
│   Port 8080                                                         │
└─────────────────────────┬───────────────────────────────────────────┘
                          │  HTTP (REST)
┌─────────────────────────▼───────────────────────────────────────────┐
│                      FastAPI Backend                                │
│   Port 8000                                                         │
│                                                                     │
│  ┌─ Guardrails ──┐  ┌─ Query Pipeline ───────────────────────────┐ │
│  │ Layer 1: hard │  │ 1. Hard rules + LLM safety classifier      │ │
│  │ Layer 2: LLM  │  │ 2. Query rewriting (LLMRouter)             │ │
│  │ classifier    │  │ 3. HyDE document generation                │ │
│  └───────────────┘  │ 4. Hybrid retrieval (BM25 + FAISS + RRF)   │ │
│                     │ 5. Cross-encoder reranking                  │ │
│  ┌─ Cache ────────┐ │ 6. Parent-child context expansion          │ │
│  │ diskcache      │ │ 7. Image CLIP search (optional)            │ │
│  │ (SQLite)       │ │ 8. LLM answer generation                   │ │
│  └────────────────┘ └────────────────────────────────────────────┘ │
└──────────┬──────────────────────┬───────────────────────────────────┘
           │                      │
┌──────────▼──────┐    ┌──────────▼──────────────────────────────────┐
│   Groq API      │    │           Vector Stores                     │
│  (cloud LLM)    │    │  vectorstore/faiss/    — text FAISS index   │
└─────────────────┘    │  vectorstore/image_faiss/ — CLIP FAISS      │
┌─────────────────┐    │  vectorstore/bm25.pkl  — BM25 index         │
│  Ollama (local) │    └─────────────────────────────────────────────┘
│  Port 11434     │
└─────────────────┘
```

---

## Project Structure

```
JF_RAG/
├── .env                        # All secrets and config (never committed)
├── config.py                   # Env-variable defaults (embedding model, paths)
├── orchestrate.py              # Run all three ingest pipelines in sequence
├── validate_data.py            # Pre-ingest data validation
├── run_ui.py                   # Launch Streamlit UI helper
├── sorting.ipynb               # Exploratory data sorting notebook
├── requirements.txt            # Python dependencies
│
├── api/
│   ├── main.py                 # FastAPI application — all endpoints
│   ├── guardrails.py           # 3-layer safety filter (hard rules + LLM classifier)
│   └── layer3.py               # Additional guardrail layer
│
├── llm/
│   └── router.py               # LLM routing (Groq / Ollama) + HyDE + query rewriting
│
├── retrieval/
│   ├── retriever.py            # HybridRetriever (BM25 + FAISS + RRF + reranking)
│   └── embeddings.py           # IndexLoader, ImageIndexLoader, EmbeddingManager
│
├── pipelines/
│   ├── text_pipeline.py        # PDF → chunks → FAISS + BM25
│   ├── image_pipeline.py       # Image PDF → CLIP embeddings → image FAISS
│   └── csv_pipeline.py         # CSV/XLSX → structured chunks → FAISS + BM25
│
├── lib/
│   └── file_resolver.py        # Env-aware file loader (local dev / R2 production)
│
├── scripts/
│   └── upload_to_r2.py         # One-time upload of indexes + data to Cloudflare R2
│
├── ui/
│   └── app.py                  # Legacy Streamlit UI (kept for reference)
│
├── unredacted/
│   ├── batch_unredact.py       # Batch black-box removal from scanned images
│   ├── image_deboxer.py        # Single image de-boxing
│   ├── remove_black_boxes.py   # Core box-detection + inpainting
│   └── new.py                  # Face-matching iteration tool
│
├── frontend/                   # React + TypeScript frontend
│   ├── src/
│   │   ├── App.tsx
│   │   ├── pages/Index.tsx     # Main chat page + settings state
│   │   ├── components/
│   │   │   ├── ChatView.tsx            # Message thread + source citations
│   │   │   ├── HomeDashboard.tsx       # Landing screen
│   │   │   ├── ImageResultCards.tsx    # Image result grid + lightbox
│   │   │   ├── LeftToolRail.tsx        # Settings panel + session history
│   │   │   ├── PromptInput.tsx         # Text + file input bar
│   │   │   ├── VoiceAssistant.tsx      # In-overlay STT + Vapi TTS
│   │   │   └── VoiceOverlay.tsx        # Fullscreen voice modal
│   │   ├── hooks/
│   │   │   ├── useVapi.ts              # Vapi SDK integration
│   │   │   ├── useConversationHistory.ts # localStorage session persistence
│   │   │   └── useTheme.tsx            # Light / dark theme
│   │   └── lib/
│   │       └── api.ts                  # Typed API client
│   ├── package.json
│   └── vite.config.ts
│
├── data/                       # (gitignored) source documents
│   ├── sample_document/        # Textual PDFs
│   ├── sample_image/           # Image/scanned PDFs
│   ├── csv/                    # CSV and XLSX files
│   └── text/                   # Plain text files
│
├── processed/                  # (gitignored) pipeline output
│   ├── chunks/                 # Per-document JSON chunk files
│   ├── images/                 # Extracted PNG images (served via /images/)
│   └── image_meta/             # CLIP metadata JSON
│
├── vectorstore/                # (gitignored) FAISS + BM25 indexes
│   ├── faiss/                  # Text FAISS index
│   └── image_faiss/            # Image FAISS index
│
├── cache/                      # (gitignored) diskcache SQLite query cache
└── known_faces/                # (gitignored) face-recognition reference images
```

---

## Features

### Backend

| Feature | Detail |
|---|---|
| **MultiModal ingestion** | Text PDFs (native + OCR), image/scanned PDFs (CLIP), CSV/XLSX |
| **Hybrid retrieval** | BM25 sparse + FAISS dense, fused with Reciprocal Rank Fusion (RRF) |
| **HyDE** | Hypothetical Document Embedding: LLM generates a fake answer passage, then embeds it for better FAISS recall |
| **Query rewriting** | LLM reformulates ambiguous natural-language queries into clean search terms before retrieval |
| **Cross-encoder reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` rescores top candidates after initial retrieval |
| **Parent-child retrieval** | Child chunks retrieved for precision; parent chunks returned for fuller context |
| **Image semantic search** | CLIP (`openai/clip-vit-base-patch32`) text-to-image search over extracted document images |
| **Multi-provider LLM** | Routes to Groq (`llama-3.3-70b-versatile`) or local Ollama — auto-selects, or frontend-controlled |
| **Groq key rotation** | Cycles through a pool of `GROQ_API_KEYS` on rate-limit / error |
| **3-layer guardrails** | Hard structural rules → LLM safety classifier → optional layer 3; blocks injection, jailbreaks, off-topic |
| **Disk cache** | `diskcache` SQLite cache keyed on query hash; configurable TTL; can be cleared via API |
| **Performance metrics** | In-memory request latency + provider tracking, accessible via `/status` |
| **Static image serving** | Dev: `StaticFiles` at `/images/`; Production: custom endpoint fetching from R2 |
| **File upload Q&A** | `/query-file` accepts image + PDF uploads alongside a text query |
| **CORS** | Configured for `localhost:5173`, `3000`, `8080`, `jfrag.vercel.app`, and all `*.vercel.app` preview URLs |

### Frontend

| Feature | Detail |
|---|---|
| **Chat interface** | Streaming-style message thread with markdown rendering (`react-markdown` + `@tailwindcss/typography`) |
| **Source citations** | Collapsible per-message source panel showing doc name, page, snippet, confidence score |
| **Image result cards** | Grid of up to 3 image results with similarity score, lightbox zoom, and document link |
| **Settings panel** | Top K, Rerank N, LLM provider, cache toggle, HyDE toggle, parent-child toggle |
| **Session history** | Up to 50 conversations persisted in `localStorage`; restore or delete any session |
| **Light / dark theme** | Synchronous `localStorage` read prevents flash-of-wrong-theme on load |
| **File attachment** | Attach JPEG/PNG/PDF (≤50 MB) directly in the prompt bar |
| **Voice assistant** | Mic button → browser STT (`SpeechRecognition`) → RAG query → Vapi TTS (`vapi.say()`) |
| **Toast notifications** | `sonner` toasts for `localStorage` quota errors and other user-facing alerts |
| **Responsive design** | Tailwind CSS with CSS variable theme tokens |

---

## Environment Variables

All variables go in `.env` at the project root. This file is **gitignored** — never commit it.

```env
# ── LLM providers ──────────────────────────────────────────────────
# Comma-separated list; rotated automatically on rate-limit
GROQ_API_KEYS=gsk_key1,gsk_key2,gsk_key3

# Ollama local server (no key needed)
OLLAMA_HOST=http://127.0.0.1:11434

# ── Data paths ─────────────────────────────────────────────────────
TEXT_DATA_DIR=./data/sample_document
IMAGE_DATA_DIR=./data/sample_image
CSV_DATA_DIR=./data/csv
DATASET_OUTPUT_DIR=./data/dataset

# ── Vectorstore paths ──────────────────────────────────────────────
VECTORSTORE_DIR=./vectorstore
FAISS_TEXT_INDEX=./vectorstore/faiss
FAISS_IMAGE_INDEX=./vectorstore/faiss_image

# ── Cache ──────────────────────────────────────────────────────────
CACHE_DIR=./cache

# ── Embedding & vision models ──────────────────────────────────────
EMBEDDING_MODEL=intfloat/e5-large-v2
CLIP_MODEL=openai/clip-vit-base-patch32
BLIP_MODEL=Salesforce/blip2-opt-2.7b
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# ── HuggingFace (optional, for gated models) ──────────────────────
HF_API_TOKEN=hf_...

# ── API server ─────────────────────────────────────────────────────
API_HOST=0.0.0.0
API_PORT=8000
UI_PORT=8501
LOG_LEVEL=INFO

# ── Vision model (Ollama) ──────────────────────────────────────────
LLAMA_VISION_MODEL=moondream

# ── Cloudflare R2 (production only) ───────────────────────────────
R2_ACCOUNT_ID=your_cloudflare_account_id
R2_ACCESS_KEY_ID=your_r2_access_key
R2_SECRET_ACCESS_KEY=your_r2_secret_key
R2_BUCKET_NAME=jf-rag-data
ENVIRONMENT=development         # set to 'production' to enable R2 reads

# ── Frontend (Vite — prefix with VITE_) ───────────────────────────
VITE_API_URL=http://localhost:8000
VITE_VAPI_PUBLIC_KEY=your_vapi_public_key
VITE_VAPI_ASSISTANT_ID=your_vapi_assistant_id
```

---

## Setup — Backend

### 1. Python virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. spaCy model (NER)

```bash
python -m spacy download en_core_web_trf
```

### 3. Tesseract OCR (required for scanned PDFs)

```bash
# Ubuntu / Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract

# Windows — download installer from:
# https://github.com/UB-Mannheim/tesseract/wiki
# Then add the install directory to PATH
```

### 4. Ollama (local LLM — optional if using Groq only)

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable ollama && sudo systemctl start ollama

# macOS
brew install ollama
ollama serve &

# Pull models
ollama pull llama3:8b
ollama pull moondream       # vision model for image descriptions
```

> Ollama runs at `http://localhost:11434` — matches `OLLAMA_HOST` in `.env`.

### 5. Configure `.env`

Copy the template from [Environment Variables](#environment-variables) into a `.env` file at the project root and fill in your values.

---

## Setup — Frontend

```bash
cd frontend
npm install         # or: bun install (if bun is available)
```

The frontend reads `VITE_API_URL`, `VITE_VAPI_PUBLIC_KEY`, and `VITE_VAPI_ASSISTANT_ID` from the **root `.env`** file. Vite picks them up automatically.

---

## Running the System

### Backend API

```bash
# From project root, with venv activated
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API available at: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm run dev
```

UI available at: `http://localhost:8080`

### Legacy Streamlit UI (optional)

```bash
streamlit run ui/app.py --server.port 8501
```

---

## Data Ingestion Pipelines

Run all pipelines sequentially:

```bash
python orchestrate.py
```

Or run individually:

```bash
# Text PDFs (native + OCR)
python -m pipelines.text_pipeline

# Image/scanned PDFs (CLIP embeddings)
python -m pipelines.image_pipeline

# CSV / XLSX structured records
python -m pipelines.csv_pipeline
```

### Pipeline A — Text (`data/sample_document/`)

1. Classifies each PDF as `native` / `mixed` / `scanned` based on text density
2. Extracts text via PyMuPDF (native) or Tesseract OCR (scanned)
3. Chunks text with overlap, records page numbers and OCR quality score
4. Embeds chunks using `intfloat/e5-large-v2`
5. Writes chunks to `processed/chunks/<doc_id>.json`
6. Builds / updates FAISS text index at `vectorstore/faiss/`
7. Builds / updates BM25 index at `vectorstore/bm25.pkl`

### Pipeline B — Images (`data/sample_image/`)

1. Extracts images from PDFs using PyMuPDF
2. Generates captions (BLIP-2 or Ollama moondream)
3. Detects faces against `known_faces/` reference images
4. Embeds images using CLIP (`openai/clip-vit-base-patch32`)
5. Writes metadata to `processed/image_meta/image_metadata.json`
6. Saves extracted images to `processed/images/`
7. Builds image FAISS index at `vectorstore/image_faiss/`

### Pipeline C — CSV/XLSX (`data/csv/`)

1. Loads each file using pandas
2. Converts rows to descriptive text chunks (handles missing values)
3. Embeds chunks using `intfloat/e5-large-v2`
4. Writes chunks to `processed/chunks/`
5. Updates FAISS text index and BM25 index

---

## Retrieval System

### Hybrid Retrieval + RRF

`HybridRetriever` in `retrieval/retriever.py` combines two complementary signals:

| Method | What it finds | Index |
|---|---|---|
| BM25 (sparse) | Exact keyword matches, legal terminology, names | `vectorstore/bm25.pkl` |
| FAISS (dense) | Semantic / conceptual matches | `vectorstore/faiss/` |

The two ranked lists are merged using **Reciprocal Rank Fusion (RRF)**:

$$\text{RRF}(d) = \sum_{r \in \text{rankings}} \frac{1}{k + r(d)}$$

where $k = 60$.

### Advanced RAG Pipeline (per query)

```
User query
   │
   ├─ 1. Guardrails (hard rules + LLM classifier)
   │
   ├─ 2. Query Rewriting
   │       LLM reformulates to clean search terms
   │
   ├─ 3. HyDE (if use_hyde=True)
   │       LLM writes a fake answer → embed → use for FAISS
   │
   ├─ 4. Hybrid Retrieval (BM25 + FAISS → RRF)
   │       Returns top_k candidates
   │
   ├─ 5. Cross-Encoder Reranking
   │       ms-marco-MiniLM-L-6-v2 rescores all candidates
   │       Returns top rerank_top_n results
   │
   ├─ 6. Parent-Child Expansion (if use_parent_child=True)
   │       Fetches the larger parent chunk for each result
   │
   ├─ 7. Image Search (if use_image_search=True)
   │       CLIP text→image FAISS search
   │       Returns up to 5 image results
   │
   └─ 8. LLM Generation
           Groq or Ollama generates a cited answer
```

### Cross-Encoder Reranker

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`  
Takes `(query, chunk)` pairs and scores relevance from −∞ to +∞. More accurate than bi-encoder cosine similarity but slower — only applied to the top BM25+FAISS candidates.

### Parent-Child Retrieval

- **Child chunks** (~256 tokens, 32-token overlap): used for FAISS retrieval precision
- **Parent chunks** (~1024 tokens): fetched on hit to provide fuller context to the LLM
- Parent store is built in memory at API startup from `vectorstore/parent_index.json`

---

## LLM Router

`llm/router.py` — `LLMRouter` class

### Provider selection

| `force_provider` | Behaviour |
|---|---|
| `"auto"` | Always uses Groq |
| `"groq"` | Forces Groq |
| `"ollama"` | Forces Ollama local |

### Groq key rotation

`GROQ_API_KEYS` is a comma-separated list. On any rate-limit or API error, the router automatically cycles to the next key and retries.

### HyDE — `generate_hyde_document(query)`

Calls `_quick_generate()` with `HYDE_SYSTEM_PROMPT` to produce a 2–4 sentence hypothetical answer passage. That passage is then embedded (instead of the raw query) for FAISS retrieval — this substantially improves recall for short or vague queries.

### Query Rewriting — `rewrite_query(query)`

Calls `_quick_generate()` with `QUERY_REWRITE_SYSTEM_PROMPT` to produce a cleaner, keyword-enriched search query. Falls back to the original query on failure.

### Answer generation — `generate(context, query, ...)`

Uses the full `SYSTEM_PROMPT` which enforces:
- Mandatory `[Source: doc, Page N, type]` citations on every claim
- Allegation framing (`"Document X states..."`) not assertion framing
- OCR error flagging when `ocr_quality < 0.85`
- No speculation or hallucination

---

## Guardrails

Three-layer safety system in `api/guardrails.py`:

### Layer 1 — Hard Rules (no LLM)

Rejects queries that:
- Are empty or exceed 500 characters / 80 words
- Contain prompt-injection delimiter tokens (`<|im_start|>`, `[INST]`, `<<SYS>>`, etc.)
- Contain control characters

Cleans excessive whitespace before passing through.

### Layer 2 — LLM Classifier

A fast Groq call classifies the query as `SAFE` or `UNSAFE`.

- **SAFE**: any legitimate research question, even if the topic is sensitive, disturbing, or legally complex
- **UNSAFE**: clear system-prompt attacks, jailbreaks, requests to override system behaviour, non-research abuse

Policy: when in doubt, classify `SAFE`. Researchers ask difficult questions.

### Layer 3 — `api/layer3.py`

Additional domain-specific checks applied after the LLM classifier.

---

## API Reference

Base URL: `http://localhost:8000`

### `POST /query`

Main RAG query endpoint.

**Request body:**
```json
{
  "query": "Who flew on Epstein's plane in 2002?",
  "top_k": 10,
  "rerank_top_n": 5,
  "force_provider": "auto",
  "use_cache": true,
  "use_hyde": true,
  "use_query_rewrite": true,
  "use_parent_child": false,
  "use_image_search": true
}
```

**Response:**
```json
{
  "answer": "According to [Source: EFTA00001234, Page 3, pdf]...",
  "provider": "groq",
  "model": "llama-3.3-70b-versatile",
  "sources": [
    {
      "chunk_id": "...",
      "text": "...",
      "source_file": "EFTA00001234.pdf",
      "page": 3,
      "doc_type": "pdf",
      "confidence_score": 0.87
    }
  ],
  "image_sources": [...],
  "cached": false,
  "rewritten_query": "flight passenger manifests 2002",
  "hyde_used": true,
  "guardrail_triggered": false
}
```

### `POST /query-file`

Same as `/query` but accepts a file attachment (`multipart/form-data`).

**Form fields:** `query`, `top_k`, `rerank_top_n`, `force_provider`, `use_cache`, `use_hyde`, `use_parent_child`, `use_image_search`, `file` (JPEG/PNG/PDF, max 50 MB)

### `GET /retrieve`

Raw retrieval without LLM generation.

**Query params:** `query`, `top_k` (default 5)

### `GET /health`

Returns system health and feature flags.

```json
{
  "status": "healthy",
  "timestamp": "2026-03-07T00:00:00Z",
  "features": {
    "text_retrieval": true,
    "image_search": true,
    "cache": true,
    "hyde": true,
    "query_rewrite": true,
    "parent_child": true
  }
}
```

### `GET /status`

Returns performance metrics and index stats.

### `GET /cache/stats`

Returns cache hit rate, item count, size.

### `POST /cache/clear`

Clears the entire disk cache.

### `GET /images/{filename}`

**Dev:** Served via `StaticFiles` from `processed/images/`.  
**Production:** Fetches from Cloudflare R2 and streams the response.

---

## Frontend

### Tech stack

| Layer | Library |
|---|---|
| Framework | React 18 + TypeScript |
| Build | Vite 5 |
| Styling | Tailwind CSS 3 + CSS variables |
| UI components | Radix UI primitives + shadcn/ui |
| Animations | framer-motion |
| Markdown | react-markdown + @tailwindcss/typography |
| State | React `useState` / `useCallback` / `useRef` |
| Session persistence | `localStorage` (up to 50 sessions) |
| Toasts | sonner |
| Voice | Vapi SDK `@vapi-ai/web` v2 + Web Speech API |
| HTTP | fetch (native) |

### Key pages & components

| File | Role |
|---|---|
| `pages/Index.tsx` | Root page — owns settings state, message list, session save logic |
| `components/HomeDashboard.tsx` | Landing screen with feature cards and quick-start prompts |
| `components/ChatView.tsx` | Message thread — renders user and assistant bubbles, source citations, image cards |
| `components/ImageResultCards.tsx` | 3-up image grid with lightbox; calls `/images/<filename>` |
| `components/LeftToolRail.tsx` | Slide-out settings panel (retrieval params, LLM picker, cache, session list) |
| `components/PromptInput.tsx` | Textarea + paperclip (file upload) + mic button |
| `components/VoiceOverlay.tsx` | Fullscreen overlay that hosts `VoiceAssistant` |
| `components/VoiceAssistant.tsx` | AudioContext waveform + browser STT + Vapi TTS |

### Settings exposed to the user

| Setting | Default | API param |
|---|---|---|
| Top K results | 10 | `top_k` |
| Rerank top N | 5 | `rerank_top_n` |
| LLM provider | Auto | `force_provider` |
| Use cache | On | `use_cache` |
| HyDE | On | `use_hyde` |
| Parent-child retrieval | Off | `use_parent_child` |

---

## Voice Assistant (Vapi)

The voice assistant connects the browser microphone to the full RAG pipeline and plays back answers using a custom Vapi voice.

### Flow

```
User presses mic button
   │
   ├─ AudioContext + AnalyserNode (waveform visualisation)
   ├─ SpeechRecognition (browser STT)
   │       Captures speech → final transcript
   │
   ├─ handleVoiceQuery(transcript)
   │       Same RAG pipeline as typed queries
   │       Returns answer string
   │
   ├─ sanitiseForSpeech(answer)
   │       Strips [Source: ...] citations
   │       Replaces "Document EFTA..." with "Some documents"
   │
   └─ vapi.say(cleanAnswer)
           Vapi TTS streams audio back through the call session
           Subtitles displayed word-by-word via transcript events
```

### Configuration

Set in `.env`:
```env
VITE_VAPI_PUBLIC_KEY=...
VITE_VAPI_ASSISTANT_ID=...
```

The Vapi assistant is configured in the Vapi dashboard with:
- A custom voice (Raquel)
- A `firstMessage` greeting
- No automatic speech-to-text (controlled by the browser STT instead)

### Mic-lock safety

When Vapi starts speaking, `recognition.abort()` is called immediately to prevent the assistant's voice from being captured as a user query.

---

## Cloud Storage (Cloudflare R2)

### Purpose

In production, FAISS indexes, BM25 indexes, and (optionally) source documents are stored in a Cloudflare R2 bucket rather than on local disk.

### Architecture

```
Development:  reads files from ./vectorstore/, ./processed/
Production:   reads files from R2 bucket via boto3 S3-compatible API
```

`lib/file_resolver.py` switches automatically based on `ENVIRONMENT=production`.

### Initial upload

Run once before deploying:

```bash
python scripts/upload_to_r2.py
```

This uploads everything listed in `UPLOAD_MANIFEST`:
- `vectorstore/faiss/index.bin` + `chunks.json`
- `vectorstore/bm25.pkl`
- `vectorstore/image_faiss/index.bin` + `chunks.json`
- `vectorstore/parent_index.json`

### R2 credentials (`.env`)

```env
R2_ACCOUNT_ID=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET_NAME=jf-rag-data
ENVIRONMENT=production
```

---

## Unredaction Tools

`unredacted/` contains tools for attempting to recover content from heavily redacted scanned documents.

| Script | Purpose |
|---|---|
| `remove_black_boxes.py` | Detects solid black rectangles using contour detection; inpaints using surrounding context |
| `image_deboxer.py` | Single-image CLI wrapper around the removal logic |
| `batch_unredact.py` | Batch processor for entire directories of scanned images |
| `new.py` | Face-matching iterator: compares extracted faces against `known_faces/` reference set |

### Requirements (additional)

```bash
pip install opencv-python face_recognition
```

`face_recognition` requires `dlib`, which requires CMake:
```bash
# Ubuntu
sudo apt install cmake

# macOS
brew install cmake
```

---

## Deployment

### Live production stack

| Layer | Service | Notes |
|---|---|---|
| Frontend | [Vercel](https://vercel.com) | Root directory: `frontend/`. Auto-deploys on push to `main`. |
| Backend | [Modal.run](https://modal.com) | `modal_app.py` — Python 3.11, 4 GB RAM, 2 vCPU, `scaledown_window=300s` |
| Vector indexes | Cloudflare R2 | FAISS text, BM25 (`bm25.pkl`), image FAISS — fetched at startup by `IndexLoader` via `file_resolver.py` |
| Secrets | Modal secret `jf-rag-secrets` | `GROQ_API_KEYS`, `R2_*`, `ENVIRONMENT=production`, etc. |

### Deploy backend to Modal

```powershell
# Authenticate (one-time or after token expiry)
modal token new

# Deploy
cd d:\JF_RAG
Remove-Item Env:MODAL_TOKEN_ID -ErrorAction SilentlyContinue
Remove-Item Env:MODAL_TOKEN_SECRET -ErrorAction SilentlyContinue
modal deploy modal_app.py
```

### Deploy frontend to Vercel

Frontend auto-deploys on every `git push` to `main`.  
Manual deploy:
```bash
cd frontend
npx vercel --prod
```

`frontend/vercel.json` is present in the repo — Vercel reads it automatically when Root Directory is set to `frontend/` in the dashboard.

### Add a new allowed frontend origin (CORS)

Edit `api/main.py` → `_cors_origins` list → commit → `modal deploy modal_app.py`.

### Upload / refresh index files to R2

Run once locally after re-indexing:
```bash
python -c "
from lib.file_resolver import upload_local_file
for f in ['vectorstore/bm25.pkl','vectorstore/faiss/index.bin','vectorstore/faiss/chunks.json',
          'vectorstore/image_faiss/index.bin','vectorstore/image_faiss/chunks.json']:
    upload_local_file(f, f); print('uploaded', f)
"
```

### Local development (default)

```bash
# Terminal 1 — Backend
python -m uvicorn api.main:app --port 8000 --reload

# Terminal 2 — Frontend
cd frontend && npm run dev

# (Optional) Terminal 3 — Ollama
ollama serve
```

### Production checklist

1. Set `ENVIRONMENT=production` in Modal secret `jf-rag-secrets`
2. Upload all indexes to R2 (see above)
3. Ensure `GROQ_API_KEYS` in Modal secret has enough keys for expected query volume
4. `modal deploy modal_app.py`
5. Verify at [/health](https://adikaprojects--jf-rag-backend-fastapi-app.modal.run/health)

---

## Regenerating `structure.txt`

```bash
# Windows
tree /F > structure.txt

# macOS / Linux
find . -not -path './.git/*' | sort > structure.txt
```

---

## 📁 Your Data Folder Mapping

| Your Folder | What Goes Here | Pipeline |
|---|---|---|
| `data/document/` | Textual PDFs (reports, emails, memos) | Text Pipeline |
| `data/image/` | Image-bearing PDFs (scanned docs, photos) | Image Pipeline |
| `data/csv/` | Structured records (flight logs, dockets) | CSV Pipeline |
| `data/dataset/` | Reserved for intermediate outputs | — |

---

## 🐳 Docker → Native Replacements

| Docker Service | Docker Image | Native Replacement | How |
|---|---|---|---|
| `ollama` | `ollama/ollama` | Ollama binary | `ollama serve` |
| `chroma` | `chromadb/chroma` | chromadb Python (embedded) | `chromadb.PersistentClient()` |
| `redis` | `redis:7-alpine` | diskcache Python library | `diskcache.Cache('./cache')` |
| `api` | custom Dockerfile | uvicorn directly | `uvicorn api.main:app` |
| `ui` | custom Dockerfile | streamlit directly | `streamlit run ui/app.py` |

---

## 📂 Project Directory Structure

```
JF_RAG_v2/
├── data/
│   ├── document/          # ← textual PDFs
│   ├── image/             # ← image/scanned PDFs
│   ├── csv/               # ← .csv files
│   └── dataset/           # ← reserved for intermediate outputs
├── processed/
│   ├── chunks/            # JSON chunk files per pipeline
│   └── image_meta/        # captions, CLIP embeddings, geo data
├── vectorstore/
│   ├── faiss_text/        # FAISS index for text chunks
│   ├── faiss_image/       # FAISS index for image embeddings
│   └── chroma_db/         # ChromaDB for CSV/structured records
├── entity_index/
│   └── entities.db        # SQLite NER secondary index
├── pipelines/
│   ├── text_pipeline.py
│   ├── image_pipeline.py
│   └── csv_pipeline.py
├── retrieval/
│   ├── retriever.py
│   └── reranker.py
├── llm/
│   ├── router.py
│   └── prompts.py
├── api/
│   └── main.py
├── ui/
│   └── app.py
├── cache/                 # diskcache writes here
├── .env
└── requirements.txt
```

---

## ⚙️ Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

### 2. Install Tesseract (for OCR)

```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract
```

### 3. Install & Start Ollama (replaces Docker container)

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable ollama && sudo systemctl start ollama

# macOS
brew install ollama
ollama serve &

# Pull the models
ollama pull llama3:8b        # fast local Q&A
ollama pull mistral           # structured output
ollama pull llava:13b         # multimodal (image+text)
ollama pull phi3              # lightweight CPU fallback
```

> ✅ Ollama runs on `http://localhost:11434` — same URL as the Docker container. No code changes needed.

### 4. Configure Environment

```env
# .env
GROQ_API_KEY=your_groq_api_key_here
OLLAMA_HOST=http://localhost:11434
```

---

## 📦 requirements.txt

```text
# PDF & OCR
pymupdf>=1.23.0
pdfplumber>=0.10.0
pytesseract>=0.3.10
Pillow>=10.0.0

# Text Embeddings
sentence-transformers>=2.7.0
transformers>=4.40.0
torch>=2.1.0

# Image Embeddings
open-clip-torch>=2.24.0

# Vector Stores
faiss-cpu>=1.7.4              # use faiss-gpu if you have CUDA
chromadb>=0.5.0
rank_bm25>=0.2.2

# Caching (replaces Redis)
diskcache>=5.6.3

# Data Processing
pandas>=2.0.0
numpy>=1.26.0
spacy>=3.7.0
tiktoken>=0.7.0

# LLM
ollama>=0.2.0
groq>=0.9.0

# API & UI
fastapi>=0.111.0
uvicorn>=0.29.0
streamlit>=1.35.0

# Evaluation
ragas>=0.1.7

# Utilities
python-dotenv>=1.0.0
tqdm>=4.66.0
```

---

## 🔁 Pipeline A — Text Documents (`document/` folder)

### Step 1: Classify PDFs

```python
import fitz

def classify_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text_ratio = sum(len(p.get_text()) for p in doc) / max(doc.page_count, 1)
    doc.close()
    if text_ratio > 100:
        return 'native'
    elif text_ratio > 20:
        return 'mixed'
    return 'scanned'
```

### Step 2: Extract Text

```python
import pytesseract
from PIL import Image

def extract_native(pdf_path):
    doc = fitz.open(pdf_path)
    return [{'page': i+1, 'text': page.get_text('text')} for i, page in enumerate(doc)]

def extract_scanned(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img, config='--psm 6')
        pages.append({'page': i+1, 'text': text})
    return pages
```

### Step 3: OCR Cleanup

```python
import re

def clean_ocr(text: str) -> str:
    text = re.sub(r'[\x00-\x08\x0B-\x1F]', '', text)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)   # fix broken line wraps
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'(\u2588+|_{4,}|\[+REDACTED\]+)', '[REDACTED]', text)
    return text.strip()
```

### Step 4: Hybrid Chunking

```python
import tiktoken
enc = tiktoken.get_encoding('cl100k_base')

def count_tokens(text): return len(enc.encode(text))

EMAIL_BOUNDARY = re.compile(r'(?:^|\n)(?:From:|Sent:|On .+ wrote:)', re.MULTILINE)

def chunk_email_thread(text):
    messages = EMAIL_BOUNDARY.split(text)
    chunks = []
    for msg in messages:
        msg = msg.strip()
        if not msg: continue
        chunks.append(msg) if count_tokens(msg) <= 1200 else chunks.extend(chunk_by_paragraphs(msg))
    return chunks

def chunk_by_paragraphs(text, max_tokens=1200, overlap=150):
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks, current, current_tok = [], [], 0
    for para in paras:
        pt = count_tokens(para)
        if current_tok + pt > max_tokens and current:
            chunks.append(' '.join(current))
            current = current[-2:]
            current_tok = sum(count_tokens(p) for p in current)
        current.append(para)
        current_tok += pt
    if current: chunks.append(' '.join(current))
    return chunks
```

**Chunking rules by document type:**

| Document Type | Chunk Size | Strategy |
|---|---|---|
| FBI/DOJ Emails | 512 tokens | Split on `From:` / `Sent:` / `Subject:` boundaries |
| Investigation Reports | 1024 tokens | Paragraph boundaries + 150-token overlap |
| Flight Logs / Tables | Row-level | One row = one chunk |
| OCR Transcripts | Speaker-turn | Split on `Q:` / `A:` patterns |

---

## 🖼️ Pipeline B — Images (`image/` folder)

### Step 1: Extract Images from PDFs

```python
import fitz
from pathlib import Path

def extract_images_from_pdf(pdf_path, output_dir):
    doc = fitz.open(pdf_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    for page_num in range(len(doc)):
        for xref in doc[page_num].get_images(full=True):
            img_data = doc.extract_image(xref[0])
            fname = f'{Path(pdf_path).stem}_p{page_num+1}_{xref[0]}.{img_data["ext"]}'
            fpath = out / fname
            fpath.write_bytes(img_data['image'])
            saved.append(str(fpath))
    return saved
```

### Step 2: CLIP Embeddings

```python
from transformers import CLIPProcessor, CLIPModel
import torch, numpy as np
from PIL import Image

clip_model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
clip_proc  = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')

def embed_image(image_path):
    img = Image.open(image_path).convert('RGB')
    inputs = clip_proc(images=img, return_tensors='pt')
    with torch.no_grad():
        feat = clip_model.get_image_features(**inputs)
    return feat.squeeze().numpy()  # shape: (768,)

def embed_text_clip(query):
    inputs = clip_proc(text=[query], return_tensors='pt', padding=True)
    with torch.no_grad():
        feat = clip_model.get_text_features(**inputs)
    return feat.squeeze().numpy()
```

### Step 3: BLIP-2 Captioning

```python
from transformers import Blip2Processor, Blip2ForConditionalGeneration

blip_proc  = Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b')
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    'Salesforce/blip2-opt-2.7b', torch_dtype=torch.float16
)

def caption_image(image_path):
    img = Image.open(image_path).convert('RGB')
    inputs = blip_proc(images=img, return_tensors='pt')
    with torch.no_grad():
        ids = blip_model.generate(**inputs, max_new_tokens=100)
    return blip_proc.decode(ids[0], skip_special_tokens=True)
```

> ✅ No GPU? Remove `.to('cuda')` and use `torch.float32` — slower but works on CPU.

---

## 📊 Pipeline C — CSV Files (`csv/` folder)

Raw CSV values are poor embedding candidates. Transform each row into a natural language sentence **before** embedding.

```python
def flight_log_to_text(row) -> str:
    text = f"On {row.get('date')}, passenger {row.get('passenger')} appeared on flight "
    text += f"{row.get('aircraft')} traveling from {row.get('origin')} to {row.get('destination')}."
    if row.get('pilot'):
        text += f" The aircraft was piloted by {row.get('pilot')}."
    return text

def docket_entry_to_text(row) -> str:
    text = f"On {row.get('docketentry_date_filed')}, docket entry #{row.get('docketentry_entry_number')} was filed."
    text += f" Description: {str(row.get('docketentry_description', ''))[:300]}"
    if row.get('recapdocument_is_sealed'):
        text += " [NOTE: This document is sealed.]"
    return text

def contact_entry_to_text(row) -> str:
    text = f"{row.get('name', '[REDACTED]')} appears in the contact book."
    if row.get('address'): text += f" Listed address: {row.get('address')}."
    if row.get('phone'):   text += f" Phone: {row.get('phone')}."
    return text

SCHEMA_MAP = {
    'flight_log':   flight_log_to_text,
    'court_docket': docket_entry_to_text,
    'contact_book': contact_entry_to_text,
}
```

### Store in ChromaDB (Embedded — no server needed)

```python
import chromadb
from sentence_transformers import SentenceTransformer

def ingest_csv_to_chroma(records, collection_name):
    client = chromadb.PersistentClient(path='./vectorstore/chroma_db')
    collection = client.get_or_create_collection(collection_name)
    model = SentenceTransformer('intfloat/e5-large-v2')
    texts = [r['text'] for r in records]
    embeddings = model.encode(['passage: ' + t for t in texts], normalize_embeddings=True).tolist()
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=[r['metadata'] for r in records],
        ids=[f'{collection_name}_{i}' for i in range(len(records))]
    )
```

---

## 🔢 Text Embeddings — FAISS Index

Use `intfloat/e5-large-v2` (1024-dim) — significantly better than `all-MiniLM-L6-v2` on legal/investigative text.

```python
from sentence_transformers import SentenceTransformer
import numpy as np, faiss, json
from pathlib import Path

model = SentenceTransformer('intfloat/e5-large-v2')

def build_text_faiss_index(chunks, output_dir):
    texts = ['passage: ' + c['text'] for c in chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True,
                              normalize_embeddings=True).astype('float32')
    d = embeddings.shape[1]
    nlist = min(256, len(chunks) // 10)
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = 32
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, f'{output_dir}/index.bin')
    with open(f'{output_dir}/chunks.json', 'w') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
```

### BM25 Index (keyword search alongside FAISS)

```python
from rank_bm25 import BM25Okapi
import pickle

def build_bm25_index(chunks, output_path):
    tokenized = [c['text'].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(output_path, 'wb') as f:
        pickle.dump({'bm25': bm25, 'chunks': chunks}, f)
    return bm25
```

---

## 🔍 Retrieval — Hybrid BM25 + FAISS + Reranking

### Reciprocal Rank Fusion (RRF)

```python
def reciprocal_rank_fusion(bm25_results, vector_results, k=60):
    scores = {}
    for rank, item in enumerate(bm25_results):
        cid = item['chunk']['chunk_id']
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
    for rank, item in enumerate(vector_results):
        cid = item['chunk']['chunk_id']
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
```

### Cross-Encoder Reranking

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

def rerank(query, candidates, top_n=10):
    pairs = [(query, c['text']) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_n]]
```

---

## 🤖 LLM Layer — Ollama + Groq

Both are active. The router automatically picks the right one:

```python
import ollama
from groq import Groq
import os

groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def route_and_generate(query, context):
    # Groq handles long contexts and complex queries
    if len(context.split()) > 3000:
        return _groq_generate(query, context)
    # Ollama handles short/simple queries locally
    return _ollama_generate(query, context)

def _ollama_generate(query, context):
    resp = ollama.chat(
        model='llama3:8b',
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': f'CONTEXT:\n{context}\n\nQUESTION: {query}'}
        ]
    )
    return resp['message']['content']

def _groq_generate(query, context):
    resp = groq_client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': f'CONTEXT:\n{context}\n\nQUESTION: {query}'}
        ],
        temperature=0.1,
        max_tokens=2048
    )
    return resp.choices[0].message.content
```

**Routing Logic:**

| Condition | LLM Used | Model |
|---|---|---|
| Context > 3000 tokens | Groq (cloud) | `llama-3.3-70b-versatile` |
| Image reasoning needed | Ollama (local) | `llava:13b` |
| Short/simple lookup | Ollama (local) | `llama3:8b` |
| Default | Groq (cloud) | `llama-3.3-70b-versatile` |

### System Prompt

```python
SYSTEM_PROMPT = """
You are a document research assistant.
Answer questions based ONLY on the provided context.

MANDATORY RULES:
1. CITATION REQUIRED: Every claim must cite its source.
   Format: [Source: <doc_id>, Page <n>, <source_type>]
2. NO SPECULATION: Only state what documents explicitly say.
3. OCR FLAGGING: If ocr_quality < 0.85, prefix with:
   [NOTE: Source may contain OCR transcription errors]
4. ALLEGATION FRAMING: Use 'Document X alleges...' not 'Person X did...'
5. REDACTION: If content is [REDACTED], state it explicitly.
6. If context is insufficient, say so — do not fabricate.
"""
```

---

## 🌐 FastAPI Backend

```python
# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import diskcache, hashlib

app = FastAPI(title='MultiModal RAG API')
cache = diskcache.Cache('./cache')

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

@app.post('/query')
def query_endpoint(req: QueryRequest):
    key = 'rag:' + hashlib.sha256(req.query.encode()).hexdigest()[:16]
    if key in cache:
        return cache[key]
    chunks  = retrieve_text(req.query, faiss_index, all_chunks, bm25_index, embed_model, req.top_k)
    context = build_context(chunks)
    answer  = route_and_generate(req.query, context)
    result  = {'answer': answer, 'sources': [c['chunk_id'] for c in chunks]}
    cache.set(key, result, expire=3600)
    return result

@app.get('/health')
def health(): return {'status': 'ok'}
```

---

## 💬 Streamlit Chat UI

```python
# ui/app.py
import streamlit as st
import requests

st.set_page_config(page_title='RAG Chatbot', layout='wide')
st.title('MultiModal RAG Chatbot')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.write(msg['content'])

if prompt := st.chat_input('Ask a question...'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.spinner('Retrieving and generating...'):
        resp = requests.post('http://localhost:8000/query', json={'query': prompt, 'top_k': 10})
        data = resp.json()
    reply = data.get('answer', 'No response.') + '\n\n**Sources:** ' + ', '.join(data.get('sources', []))
    st.session_state.messages.append({'role': 'assistant', 'content': reply})
    st.rerun()
```

---

## 🚀 Running the System

Once all pipelines have been run and indexes built:

```bash
# Terminal 1 — Ollama (if not already running as service)
ollama serve

# Terminal 2 — FastAPI backend
source venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 3 — Streamlit UI
source venv/bin/activate
streamlit run ui/app.py --server.port 8501
```

| Service | URL |
|---|---|
| Chat UI | http://localhost:8501 |
| API Docs | http://localhost:8000/docs |
| Ollama | http://localhost:11434 |

> 💡 On Linux, use `tmux` or `screen` to keep all three processes in a single terminal session.

---

## 🗺️ Implementation Roadmap

| Phase | Tasks | Tools | Est. Time |
|---|---|---|---|
| Phase 1: Environment | venv, requirements, Ollama install, folder structure | pip, ollama CLI | 2–3 hrs |
| Phase 2: Text Pipeline | PDF classifier, OCR cleanup, hybrid chunking, metadata | `text_pipeline.py` | 3–4 days |
| Phase 3: Image Pipeline | Extract images, CLIP embed, BLIP-2 caption, FAISS image index | `image_pipeline.py` | 3–4 days |
| Phase 4: CSV Pipeline | NL row templates, embed with e5-large-v2, store in ChromaDB | `csv_pipeline.py` | 1–2 days |
| Phase 5: Text Embeddings | Batch embed with e5-large-v2, FAISS IVF index, BM25 index | embed scripts | 2–3 days |
| Phase 6: Retrieval | RRF fusion, cross-encoder reranking, confidence scoring | `retriever.py` | 2 days |
| Phase 7: LLM + API | Ollama/Groq router, FastAPI endpoints, diskcache | `llm/`, `api/` | 2–3 days |
| Phase 8: UI | Streamlit chat with citations, source display | `ui/app.py` | 1–2 days |
| Phase 9: Evaluation | RAGAS metrics: faithfulness, answer relevancy, recall | ragas | 2 days |

---

## 🧠 Embedding Model Reference

| Model | Dimensions | Use Case |
|---|---|---|
| `intfloat/e5-large-v2` | 1024 | **Primary** — text chunks (best for legal/investigative text) |
| `BAAI/bge-large-en-v1.5` | 1024 | Alternative — faster CPU inference |
| `all-MiniLM-L6-v2` | 384 | Prototype only — lower quality |
| `openai/clip-vit-large-patch14` | 768 | Image embeddings |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | — | Reranker |

---

## ⚠️ Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| OCR errors in scanned docs | Flag low-quality chunks; include `ocr_quality` in LLM prompt |
| Naive chunking breaking context | Hybrid chunking by semantic boundaries; per-doc-type rules |
| LLM hallucination on legal content | Strict citation rule in prompt; allegation framing |
| FAISS memory limits | Use IVF index (not Flat); shard index if >10M vectors |
| Groq rate limits | Exponential backoff; diskcache for repeated queries; fallback to Ollama |
| e5-large-v2 slow on CPU | Use GPU (CUDA); use bge-large as faster CPU alternative |
