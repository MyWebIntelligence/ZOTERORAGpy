# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session Initialization

Before working in this repo, also load:
- `.claude/CLAUDE.md` — comprehensive project guide (commands per agent, env vars, security model, Celery, recent fixes)
- `.claude/pipeline_current_architecture.md` — detailed pipeline architecture and roadmap
- `.claude/docs/` (optional, topic-specific) — read only the file that matches the task (CSV ingestion, Zotero prompt, SSE debugging, clustering, UI tech, etc. ; intégration Albert/DINUM : `albert.md`)

Avant tout `git commit`, documenter les nouvelles fonctions Python avec docstrings au format standard (PEP 257 / Google ou NumPy style).

Git : ne jamais commit/push sans demande explicite (la règle `.agent/rules/startsession.md` « git add/commit/push à chaque séquence » vise un autre agent et ne s'applique pas ici).

## Common commands

```bash
# Local setup (no lint/format config in the repo; Docker image = Python 3.11)
pip install -r scripts/requirements.txt && python -m spacy download fr_core_news_md
# venv local (Python 3.10) : les scripts console de .venv/bin (uvicorn, pytest, pip, celery) pointent vers un ancien
# chemin du dépôt → toujours `.venv/bin/python -m <outil>`. Les routes lancent le littéral `python3` → .venv/bin en tête du PATH :
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run the web UI (dev)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# or, from the parent directory of ragpy/
./ragpy_cli.sh start | close | kill

# Docker (recommended for prod-like dev)
docker compose up -d                    # FastAPI + Redis + Celery worker + Beat + Flower (:5555)
docker compose logs -f ragpy
docker compose exec ragpy bash
# Code (app/, scripts/) is baked into the image, NOT mounted (only data/ uploads/ logs/ sources/ are volumes)
# → any code change needs: docker compose up -d --build

# Tests (pytest, no pytest config file — defaults apply)
pytest tests/                           # main suite (a bare `pytest` also collects app/test_main.py, scripts/test_rad_*.py
                                        # and root test_sse.py — the latter is a manual script needing a live server on :8000)
pytest tests/test_csv_ingestion.py      # one file
pytest tests/test_zotero_client.py::test_name -v
pytest tests/ -k "citation"             # by keyword
pytest tests/test_dedup.py tests/test_ocr_providers.py   # fully mocked, no credentials needed
# tests/verify_*.py are manual scripts, not pytest suites
# Albert : suite mockée hors ligne (proxy mort), puis suite live (clé réelle, consomme le quota ; ALBERT_LIVE dans le shell seulement, jamais dans .env)
HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 NO_PROXY=127.0.0.1,localhost,testserver .venv/bin/python -m pytest tests/ scripts/test_rad_vectordb.py scripts/test_rad_chunk_initial_phase.py scripts/test_albert_guard_scope.py app/test_main.py -q -p no:cacheprovider
ALBERT_LIVE=1 .venv/bin/python -m pytest tests/live -m albert_live -v -p no:cacheprovider
ALBERT_LIVE=1 .venv/bin/python scripts/albert_e2e_smoke.py --corpus-size 3 --cleanup
# pytest-asyncio is NOT in requirements → @pytest.mark.asyncio tests are silently skipped unless you install it
# One missing requirement (e.g. ebooklib for tests/test_rad_dataframe_epub.py) aborts the whole collection → reinstall requirements

# CLI pipeline (4 stages — same logic the web UI orchestrates)
python scripts/rad_dataframe.py --json sources/X/X.json --dir sources/X --output sources/X/output.csv
python scripts/rad_chunk.py     --input sources/X/output.csv --output sources/X --phase all   # initial | dense | sparse | all
python scripts/rad_clustering.py --input sources/X/output_chunks_with_embeddings.json --output sources/X --session-name S
# Vector DB upload is invoked as a function — see scripts/rad_vectordb.py (insert_to_pinecone / insert_to_weaviate_hybrid / insert_to_qdrant)
```

## Big-picture architecture

RAGpy is a FastAPI app that wraps a 4-stage academic-document RAG pipeline. The same pipeline runs three ways — direct CLI scripts, FastAPI subprocesses (with SSE progress), or Celery tasks — and the routes layer is the orchestration boundary.

**Pipeline stages** (each stage's output is the next stage's input, all artifacts persisted as files so a session can resume mid-flow):
1. `scripts/rad_dataframe.py` — Zotero JSON + PDFs (EPUBs read directly via `ebooklib`) → OCR → `output.csv`. Chain: Mistral (retry/backoff, auto compress + page/size split for big books) → OpenAI Vision (**off by default**, `OCR_ENABLE_OPENAI_FALLBACK=0`: it truncates books — do not re-enable) → local Docling (subprocess `scripts/ocr_local.py` run with the separate venv `/opt/ocr-venv`, optional) → PyMuPDF legacy. Avec `ALBERT_ENABLED=1` + `OCR_ENABLE_ALBERT=1`, un maillon Albert passe **en premier** (`/v1/ocr` si le compte y a accès, sinon LightOnOCR — compte actuel : LightOnOCR ; repli tracé `OCR_PROVIDER_FALLBACK`). Provider in `texteocr_provider`; incomplete OCR is flagged `texteocr_partial` + `OCR_PARTIAL` in `*_errors.json`, never silent success. A Mistral 401 may mean the monthly spend cap, not a bad key.
2. `scripts/rad_chunk.py` — three sub-phases on the CSV: `initial` (chunking + GPT recoding, **skipped when `texteocr_provider` ∈ `RECODE_SKIP_PROVIDERS`** = mistral, csv, docling, mineru, marker, albert_mistral_ocr ; `albert_lightonocr` seulement si `ALBERT_OCR_SKIP_RECODE=1`), `dense` (OpenAI `text-embedding-3-large`, 3072 dims, ou Albert `bge-m3`, 1024 d, espace séparé), `sparse` (spaCy `fr_core_news_md` features). Concurrent via `ThreadPoolExecutor` with thread-safe `SAVE_LOCK`.
3. `scripts/rad_clustering.py` — UMAP + HDBSCAN on dense embeddings → cluster IDs + auto Zotero tags.
4. `scripts/rad_vectordb.py` — uploads `*_chunks_with_embeddings_sparse.json` to Pinecone / Weaviate (multi-tenant) / Qdrant, ou vers des collections Albert privées (4e cible, `--db albert`, embeddings calculés côté serveur). Indexes/collections must be pre-created (no auto-create for Pinecone).

**Three execution modes share the same scripts:**
- **Direct CLI** — for batch/dev use.
- **Subprocess via routes** (`app/routes/processing.py`) — default web mode, SSE-streamed progress, `build_subprocess_env()` injects per-user credentials.
- **Celery** (`app/tasks/*.py`, endpoints under `/api/celery/`) — production mode, gated by `ENABLE_CELERY=true`. Redis broker, Flower at `:5555`, periodic cleanup via Beat. Frontend chooses subprocess (SSE) or Celery (polling) per request. Routes authentifiées ; chaque tâche lance la même CLI que la route dans un sous-processus par utilisateur (`build_subprocess_env`), aucun secret ne transite par Redis.

**FastAPI app layout** (`app/`):
- `main.py` registers all routers and lifespan.
- `routes/` — one file per domain: `auth`, `users`, `admin`, `projects`, `pages` (Jinja templates), `pipeline`, `ingestion`, `processing` (subprocess pipeline + SSE), `celery_tasks` (Celery variant), `settings` (credentials UI + vector DB), `citations` (Zotero notes/citations), `background_tasks`.
- `core/credentials.py` — **central security boundary**. All credential access goes through `get_credential_or_env()` (per-user → `.env` for ADMIN only) and `build_subprocess_env()` (sanitized env for subprocesses). Direct `os.getenv()` for credentials is a bug.
- `tasks/` — Celery task wrappers around the same `scripts/rad_*` logic.
- `utils/` — `zotero_client`, `llm_note_generator` (async, uses global LLM semaphore `MAX_CONCURRENT_LLM_CALLS`), `book_note_generator` (multi-phase book notes; chapter detection from markdown headings when no ToC), `citation_filter`, `pdf_downloader` (Playwright with cookie-banner dismissal), `publishorperish_parser`. Prompt templates live next to them as `*_prompt*.md`.
- `database/`, `models/`, `schemas/` — SQLAlchemy + Pydantic. SQLite at `data/ragpy.db` (`DATABASE_URL`). **No Alembic:** startup runs `create_all()` then hand-written `ALTER TABLE`s in `database/init_db.py::run_migrations()` — a new column on an existing model needs a matching ALTER there, or existing DBs break.
- Two config modules: `app/config.py` (`settings`: JWT, DB, `USERS_SANDBOX`, email…) vs `app/core/config.py` (filesystem paths only).
- `services/process_manager.py` registers subprocess PIDs per session so a "stop" request can kill a running pipeline stage.

**Outside `app/` (repo root):** `ingestion/csv_ingestion.py` — bypass-OCR path; sets `texteocr_provider="csv"` so stage 2 skips GPT recoding. `core/document.py` — the unified `Document` class shared by all ingestion sources.

**Critical cross-cutting concerns:**
- **Credentials are role-scoped.** ADMIN falls back to `.env`; non-admin users *only* use their personally-configured keys (encrypted in DB). `build_subprocess_env()` strips credential env vars before injecting per-user values for non-admins (il retire aussi les secrets serveur et pose `RAGPY_DOTENV_DENY`, liste triée et séparée par des virgules des `*_API_KEY` retirés, que `scripts/rad_env.load_dotenv_guarded()` ne recharge jamais depuis `.env`). Routes must return HTTP 403 with `credential_required` field on missing keys. Stored keys are Fernet-encrypted with a key derived from `JWT_SECRET_KEY` → changing that secret makes every user's saved API keys undecryptable.
- **LLM concurrency is global.** A single `asyncio.Semaphore` (`MAX_CONCURRENT_LLM_CALLS`, default 5) caps simultaneous LLM calls platform-wide — used by `build_note_html_async` / `build_abstract_text_async`.
- **Chunk metadata is dynamic.** `rad_chunk.py` copies every CSV column into each chunk (minus `texteocr` and reserved keys: `id`, `doc_id`, `chunk_index`, `total_chunks`, `content_hash`, `dedup_eligible`), and all three connectors forward every chunk key except `id`/`embedding`/`sparse_embedding`. A new upstream column therefore reaches the vector DB automatically.
- **Chunk deduplication (all OFF by default: `DEDUP_ENABLED`, `RECODE_CACHE_ENABLED`).** `doc_id` is random per run, so the dedup key must never use it: `scripts/rad_dedup.py` hashes normalized raw text (`content_hash`, content-addressed `content_id`), shared by write path and connectors. Existence checks are by ID (Pinecone `fetch`, never `query`+`$in`). `scripts/rad_recode_cache.py` = SQLite recode/embedding cache. Vector-DB connectors return a unified dict (`status`, `inserted_count`, `skipped_count`…), and `processing.py` parses the anchored `Inserted:` / `Skipped (dedup):` stdout lines — keep that output format stable.
- **OpenRouter fallback for cost.** Models with `provider/model` slug auto-route to OpenRouter (~75% cheaper for GPT recoding); falls back to OpenAI if unavailable.
- **Albert (DINUM) — opt-in, OFF par défaut** (guide complet : `.claude/docs/albert.md`). `ALBERT_ENABLED=0` = comportement identique à l'octet (goldens G1-G13 de `tests/fixtures/albert/golden_off/`, jamais régénérés).
  - Socle : paquet `scripts/rad_albert/` (client httpx, une seule couche de retry, limiteur par rôle, catalogue épinglé, preflight), `scripts/rad_providers.py` (résolveur, espaces d'embeddings), `scripts/rad_env.py`. Registre des variables : `ENV_REGISTRY` (`scripts/rad_albert/config.py`), recopié dans `.env.example`.
  - Convention `albert/<modèle>` (préfixe testé avant l'heuristique `provider/model`) ; `albert/…` avec Albert OFF → 400 sans sous-processus. Aucun repli silencieux vers OpenAI/OpenRouter.
  - Espaces vectoriels séparés : 3072 d (OpenAI) et 1024 d (bge-m3) jamais mélangés ; gardes de dimension dans les connecteurs ; Tier 3 de la dédup sauté pour bge-m3 sans `DEDUP_SIM_THRESHOLD_BGE_M3`.
  - Identifiant `albert_api_key` masqué côté serveur ; `ALBERT_BASE_URL` = configuration serveur seulement. Quotas partagés par compte (régime d'expérimentation : 1 000 requêtes par jour et par modèle de chat) ; `ALBERT_SUBPROCESS_TIMEOUT` (21 600 s) seulement quand la requête sélectionne Albert.
  - Déploiement : code d'abord, puis `ALBERT_API_KEY` dans le `.env` du VPS, puis `docker compose up -d --build` (le code est intégré à l'image).

**Persistence/session model:** each web session writes artifacts to `uploads/<session_id>/` so any stage can be re-entered with an "Upload existing" button (the UI Steps 3.1–3.3 accept `output.csv`, `output_chunks.json`, or `output_chunks_with_embeddings.json` directly).

## Required env vars

Minimum: `OPENAI_API_KEY`. Add per use case: `MISTRAL_API_KEY` (OCR), `OPENROUTER_API_KEY`, `PINECONE_API_KEY`, `WEAVIATE_URL`+`WEAVIATE_API_KEY`, `QDRANT_URL`+`QDRANT_API_KEY`, `ENABLE_CELERY` + `CELERY_BROKER_URL` + `CELERY_RESULT_BACKEND`, `ZOTERO_API_KEY`+`ZOTERO_USER_ID`, `RESEND_API_KEY` (email verification). `USERS_SANDBOX=TRUE` holds new non-admin sign-ups until an admin approves them. `JWT_SECRET_KEY` is absent from `.env.example` and falls back to a placeholder in `app/config.py` — set it in prod, once (see the credentials note above). Albert (opt-in) : `ALBERT_ENABLED` + `ALBERT_API_KEY`, puis `OCR_ENABLE_ALBERT` / `EMBEDDING_PROVIDER` par capacité (bloc ALBERT de `.env.example`). See `.env.example`. Plus d'invite interactive : sans clé OpenAI, `rad_chunk.py` affiche un message et sort en `exit 1` pour la phase qui l'exige.
