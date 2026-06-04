# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session Initialization

Before working in this repo, also load:
- `.claude/CLAUDE.md` — comprehensive project guide (commands per agent, env vars, security model, Celery, recent fixes)
- `.claude/pipeline_current_architecture.md` — detailed pipeline architecture and roadmap
- `.claude/docs/` (optional, topic-specific) — read only the file that matches the task (CSV ingestion, Zotero prompt, SSE debugging, clustering, UI tech, etc.)

Avant tout `git commit`, documenter les nouvelles fonctions Python avec docstrings au format standard (PEP 257 / Google ou NumPy style).

## Common commands

```bash
# Run the web UI (dev)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# or, from the parent directory of ragpy/
./ragpy_cli.sh start | close | kill

# Docker (recommended for prod-like dev)
docker compose up -d                    # FastAPI + Redis + Celery worker + Beat + Flower (:5555)
docker compose logs -f ragpy
docker compose exec ragpy bash

# Tests (pytest, no pytest config file — defaults apply)
pytest                                  # run all
pytest tests/test_csv_ingestion.py      # one file
pytest tests/test_zotero_client.py::test_name -v
pytest -k "citation"                    # by keyword

# CLI pipeline (4 stages — same logic the web UI orchestrates)
python scripts/rad_dataframe.py --json sources/X/X.json --dir sources/X --output sources/X/output.csv
python scripts/rad_chunk.py     --input sources/X/output.csv --output sources/X --phase all   # initial | dense | sparse | all
python scripts/rad_clustering.py --input sources/X/output_chunks_with_embeddings.json --output sources/X --session-name S
# Vector DB upload is invoked as a function — see scripts/rad_vectordb.py (insert_to_pinecone / insert_to_weaviate_hybrid / insert_to_qdrant)
```

## Big-picture architecture

RAGpy is a FastAPI app that wraps a 4-stage academic-document RAG pipeline. The same pipeline runs three ways — direct CLI scripts, FastAPI subprocesses (with SSE progress), or Celery tasks — and the routes layer is the orchestration boundary.

**Pipeline stages** (each stage's output is the next stage's input, all artifacts persisted as files so a session can resume mid-flow):
1. `scripts/rad_dataframe.py` — Zotero JSON + PDFs → OCR (Mistral primary, OpenAI vision fallback, PyMuPDF legacy) → `output.csv`. Provider tracked in `texteocr_provider` column.
2. `scripts/rad_chunk.py` — three sub-phases on the CSV: `initial` (chunking + GPT recoding, **skipped when `texteocr_provider in {"mistral", "csv"}`** for cost), `dense` (OpenAI `text-embedding-3-large`, 3072 dims), `sparse` (spaCy `fr_core_news_md` features). Concurrent via `ThreadPoolExecutor` with thread-safe `SAVE_LOCK`.
3. `scripts/rad_clustering.py` — UMAP + HDBSCAN on dense embeddings → cluster IDs + auto Zotero tags.
4. `scripts/rad_vectordb.py` — uploads `*_chunks_with_embeddings_sparse.json` to Pinecone / Weaviate (multi-tenant) / Qdrant. Indexes/collections must be pre-created (no auto-create for Pinecone).

**Three execution modes share the same scripts:**
- **Direct CLI** — for batch/dev use.
- **Subprocess via routes** (`app/routes/processing.py`) — default web mode, SSE-streamed progress, `build_subprocess_env()` injects per-user credentials.
- **Celery** (`app/tasks/*.py`, endpoints under `/api/celery/`) — production mode, gated by `ENABLE_CELERY=true`. Redis broker, Flower at `:5555`, periodic cleanup via Beat. Frontend chooses subprocess (SSE) or Celery (polling) per request.

**FastAPI app layout** (`app/`):
- `main.py` registers all routers and lifespan.
- `routes/` — one file per domain: `auth`, `users`, `admin`, `projects`, `pages` (Jinja templates), `pipeline`, `ingestion`, `processing` (subprocess pipeline + SSE), `celery_tasks` (Celery variant), `settings` (credentials UI + vector DB), `citations` (Zotero notes/citations), `background_tasks`.
- `core/credentials.py` — **central security boundary**. All credential access goes through `get_credential_or_env()` (per-user → `.env` for ADMIN only) and `build_subprocess_env()` (sanitized env for subprocesses). Direct `os.getenv()` for credentials is a bug.
- `tasks/` — Celery task wrappers around the same `scripts/rad_*` logic.
- `utils/` — `zotero_client`, `llm_note_generator` (async, uses global LLM semaphore `MAX_CONCURRENT_LLM_CALLS`), `citation_filter`, `pdf_downloader` (Playwright with cookie-banner dismissal).
- `database/`, `models/`, `schemas/` — SQLAlchemy + Pydantic. SQLite at `data/`.
- `ingestion/csv_ingestion.py` — bypass-OCR path; sets `texteocr_provider="csv"` so stage 2 skips GPT recoding.

**Critical cross-cutting concerns:**
- **Credentials are role-scoped.** ADMIN falls back to `.env`; non-admin users *only* use their personally-configured keys (encrypted in DB). `build_subprocess_env()` strips credential env vars before injecting per-user values for non-admins. Routes must return HTTP 403 with `credential_required` field on missing keys.
- **LLM concurrency is global.** A single `asyncio.Semaphore` (`MAX_CONCURRENT_LLM_CALLS`, default 5) caps simultaneous LLM calls platform-wide — used by `build_note_html_async` / `build_abstract_text_async`.
- **Hardcoded chunk metadata is a known limitation.** `rad_chunk.py` (~L250-263) and the three vector-DB connectors hardcode 8-10 metadata fields, dropping custom CSV columns. See `.claude/pipeline_current_architecture.md` for the recommended dynamic-injection fix.
- **OpenRouter fallback for cost.** Models with `provider/model` slug auto-route to OpenRouter (~75% cheaper for GPT recoding); falls back to OpenAI if unavailable.

**Persistence/session model:** each web session writes artifacts to `uploads/<session_id>/` so any stage can be re-entered with an "Upload existing" button (the UI Steps 3.1–3.3 accept `output.csv`, `output_chunks.json`, or `output_chunks_with_embeddings.json` directly).

## Required env vars

Minimum: `OPENAI_API_KEY`. Add per use case: `MISTRAL_API_KEY` (OCR), `OPENROUTER_API_KEY`, `PINECONE_API_KEY`, `WEAVIATE_URL`+`WEAVIATE_API_KEY`, `QDRANT_URL`+`QDRANT_API_KEY`, `ENABLE_CELERY` + `CELERY_BROKER_URL` + `CELERY_RESULT_BACKEND`. See `.env.example`. The app prompts for missing OpenAI keys interactively on CLI use and offers to write them via `python-dotenv`.
