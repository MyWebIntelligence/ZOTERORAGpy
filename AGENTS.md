# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

## Session Initialization

Before working in this repo, also load:
- `docs/GUIDE.md` — comprehensive project guide (commands per agent, env vars, security model, Celery, recent fixes)
- `docs/pipeline_current_architecture.md` — detailed pipeline architecture and roadmap
- `docs/` (optional, topic-specific) — read only the file that matches the task (CSV ingestion, Zotero prompt, SSE debugging, clustering, UI tech, etc. ; intégration Albert/DINUM : `albert.md`)

Avant tout `git commit`, documenter les nouvelles fonctions Python avec docstrings au format standard (PEP 257 / Google ou NumPy style).

Git : ne jamais commit/push sans demande explicite (la règle `.agent/rules/startsession.md` « git add/commit/push à chaque séquence » vise un autre agent et ne s'applique pas ici).

Dépôt public : aucun fichier ni dossier en point n'est versionné (`.*` dans `.gitignore`, exceptions `.gitignore`, `.dockerignore`, `.env.example`, `.github/`) ; la documentation publique va dans `docs/`, jamais sous `.claude/` (un nouveau fichier en point utile au dépôt exige son exception `!nom`).

## Common commands

```bash
# Local setup (lint: ruff.toml, errors only; no formatter config; Docker image = Python 3.11)
pip install -r scripts/requirements.txt && python -m spacy download fr_core_news_md
# venv local (Python 3.10) : les scripts console de .venv/bin (uvicorn, pytest, pip, celery) pointent vers un ancien
# chemin du dépôt → toujours `.venv/bin/python -m <outil>`. Les routes lancent le littéral `python3` → .venv/bin en tête du PATH :
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run the web UI (dev)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# or, from the parent directory of ragpy/ (the script launches ragpy.app.main)
./ragpy/ragpy_cli.sh start | close | kill

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
# async tests use @pytest.mark.anyio (anyio plugin, backend asyncio in conftest.py) — never @pytest.mark.asyncio (pytest-asyncio is not a dependency)
# Static check (errors only, ruff.toml) and CI: .github/workflows/tests.yml (Python 3.11, offline)
.venv/bin/python -m pip install -r scripts/requirements-dev.txt && .venv/bin/python -m ruff check .
# One missing requirement (e.g. ebooklib for tests/test_rad_dataframe_epub.py) aborts the whole collection → reinstall requirements

# .env tooling (registre unique scripts/rad_settings/registry.py ; jamais de valeur affichée)
.venv/bin/python scripts/env_tool.py check              # bilan du .env (inconnues, doublons, invalides)
.venv/bin/python scripts/env_tool.py tidy --apply       # range le .env dans l'ordre du registre (sauvegarde data/config_backups/)
.venv/bin/python scripts/env_tool.py sync --apply       # écrit les variables obligatoires absentes (démarrage strict, sinon refus)
.venv/bin/python scripts/env_tool.py migrate --apply    # sort du mode historique : couples serveur + modèle (acheminement conservé)
.venv/bin/python scripts/env_tool.py prune              # variables historiques sans effet en mode unifié ; --apply [NOMS…] les retire (sauvegarde)
.venv/bin/python scripts/env_tool.py example            # régénère .env.example (fichier généré : ne pas l'éditer à la main)

# CLI pipeline (4 stages — same logic the web UI orchestrates)
python scripts/rad_dataframe.py --json sources/X/X.json --dir sources/X --output sources/X/output.csv
python scripts/rad_chunk.py     --input sources/X/output.csv --output sources/X --phase all   # initial | dense | sparse | all
python scripts/rad_clustering.py --input sources/X/output_chunks_with_embeddings.json --output sources/X --session-name S
# Vector DB upload is invoked as a function — see scripts/rad_vectordb.py (insert_to_pinecone / insert_to_weaviate_hybrid / insert_to_qdrant)
```

## Big-picture architecture

RAGpy is a FastAPI app that wraps a 4-stage academic-document RAG pipeline. The same pipeline runs three ways — direct CLI scripts, FastAPI subprocesses (with SSE progress), or Celery tasks — and the routes layer is the orchestration boundary.

**Pipeline stages** (each stage's output is the next stage's input, all artifacts persisted as files so a session can resume mid-flow):
1. `scripts/rad_dataframe.py` — Zotero JSON + PDFs (EPUBs read directly via `ebooklib`) → OCR → `output.csv`. Chain: Mistral (retry/backoff — a 429 gets a longer ladder, a cooldown shared by the process threads and, for split books, a later pass over the rate-limited parts —, auto compress + page/size split for big books) → OpenAI Vision (**off by default**, `OCR_ENABLE_OPENAI_FALLBACK=0`: it truncates books — do not re-enable) → local Docling (subprocess `scripts/ocr_local.py` run with the separate venv `/opt/ocr-venv`, optional) → PyMuPDF legacy. Avec `ALBERT_ENABLED=1` + `OCR_ENABLE_ALBERT=1`, un maillon Albert passe **en premier** (`/v1/ocr` si le compte y a accès, sinon LightOnOCR — compte actuel : LightOnOCR ; repli tracé `OCR_PROVIDER_FALLBACK`). Provider in `texteocr_provider`; incomplete OCR is flagged `texteocr_partial` + `OCR_PARTIAL` in `*_errors.json`, never silent success. A Mistral 401 may mean the monthly spend cap, not a bad key.
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
- **Session folders have an owner** (audit A02, `app/core/session_access.py`). Every route touching `uploads/<session>` calls `session_refusal(db, path, user, READ|WRITE|STOP, upload_dir=...)` first: project sessions are read by every member but written/stopped only by owner and collaborators (a `viewer` gets 403); uploads made outside a project are recorded in `session_owners` (`SessionOwner`) and reserved to their uploader; a folder with no row at all is admin-only. Storage names are server-generated and every write/cleanup is confined (`app/core/upload_safety.py`). Processing routes then take a job ticket (`app/services/job_control.py`: cross-process `flock` per session+group → 409, `MAX_ACTIVE_JOBS[_PER_USER]` → 429).
- **Credentials are role-scoped.** ADMIN falls back to `.env`; non-admin users *only* use their personally-configured keys (encrypted in DB). `build_subprocess_env()` strips credential env vars before injecting per-user values for non-admins (il retire aussi les secrets serveur et pose `RAGPY_DOTENV_DENY`, liste triée et séparée par des virgules des `*_API_KEY` retirés, que `scripts/rad_env.load_dotenv_guarded()` ne recharge jamais depuis `.env`). Routes must return HTTP 403 with `credential_required` field on missing keys. Stored keys are Fernet-encrypted with a key derived from `JWT_SECRET_KEY` → changing that secret makes every user's saved API keys undecryptable.
- **LLM concurrency is global.** A single `asyncio.Semaphore` (`MAX_CONCURRENT_LLM_CALLS`, default 5) caps simultaneous LLM calls platform-wide — used by `build_note_html_async` / `build_abstract_text_async`.
- **Chunk metadata is dynamic.** `rad_chunk.py` copies every CSV column into each chunk (minus `texteocr` and reserved keys: `id`, `doc_id`, `chunk_index`, `total_chunks`, `content_hash`, `dedup_eligible`), and all three connectors forward every chunk key except `id`/`embedding`/`sparse_embedding`. A new upstream column therefore reaches the vector DB automatically.
- **Chunk deduplication (all OFF by default: `DEDUP_ENABLED`, `RECODE_CACHE_ENABLED`).** `doc_id` is random per run, so the dedup key must never use it: `scripts/rad_dedup.py` hashes normalized raw text (`content_hash`), shared by write path and connectors. Identity = content + stable source (`DEDUP_META_FIELDS`), never the position: v2 id `{hash[:16]}_s{source[:12]}` (audit A05); existence checks are by ID (Pinecone `fetch`, never `query`+`$in`) on the chunk id, the v2 id and the legacy v1 id `{hash[:16]}_{chunk_index}`. `scripts/rad_recode_cache.py` = SQLite recode/embedding cache. Vector-DB connectors return a unified dict (`status`, `inserted_count`, `skipped_count`…), and `processing.py` parses the anchored `Inserted:` / `Skipped (dedup):` stdout lines — keep that output format stable.
- **OpenRouter fallback for cost.** Models with `provider/model` slug auto-route to OpenRouter (~75% cheaper for GPT recoding); falls back to OpenAI if unavailable.
- **Albert (DINUM) — opt-in, OFF par défaut** (guide complet : `docs/albert.md`). `ALBERT_ENABLED=0` = comportement identique à l'octet (goldens G1-G13 de `tests/fixtures/albert/golden_off/`, jamais régénérés hors amendement motivé et accepté explicitement, comme G7 pour la sécurité dans `e812c64`).
  - Socle : paquet `scripts/rad_albert/` (client httpx, une seule couche de retry, limiteur par rôle, catalogue épinglé, preflight), `scripts/rad_providers.py` (résolveur, espaces d'embeddings), `scripts/rad_env.py`. Registre des variables : `ENV_REGISTRY` (`scripts/rad_albert/config.py`), recopié dans `.env.example`.
  - Convention `albert/<modèle>` (préfixe testé avant l'heuristique `provider/model`) ; `albert/…` avec Albert OFF → 400 sans sous-processus. Aucun repli silencieux vers OpenAI/OpenRouter.
  - Espaces vectoriels séparés : 3072 d (OpenAI) et 1024 d (bge-m3) jamais mélangés ; gardes de dimension dans les connecteurs ; Tier 3 de la dédup sauté pour bge-m3 sans `DEDUP_SIM_THRESHOLD_BGE_M3`.
  - Identifiant `albert_api_key` masqué côté serveur ; `ALBERT_BASE_URL` = configuration serveur seulement. Quotas partagés par compte (régime d'expérimentation : 1 000 requêtes par jour et par modèle de chat) ; `ALBERT_SUBPROCESS_TIMEOUT` (21 600 s) seulement quand la requête sélectionne Albert.
  - Déploiement : code d'abord, puis `ALBERT_API_KEY` dans le `.env` du VPS, puis `docker compose up -d --build` (le code est intégré à l'image).
  - Sprint R2 (`.claude/tasks/SPRINT_albert_r2.md`, guide §15-21), tout OFF par défaut : `ALBERT_DATA_POLICY=albert_only` (inférence Albert seule, clés OpenAI/OpenRouter/Mistral retirées des sous-processus ; défaut `compatible`, `app/core/albert_policy.py`) ; **corpus** du registre local (`AlbertCorpus`, `app/services/albert_access.py`, routes `app/routes/albert_corpora.py` : un id de collection venu du navigateur n'autorise rien) ; **recherche, réponses sourcées et rerank retirés le 2026-10-03** (RAGpy prépare les corpus, les questions sont posées par d'autres outils ; variables `LLM_RAG_*`, `RERANK_*`, `ALBERT_RAG_*`, `ALBERT_RERANK_*` obsolètes, `env_tool prune --apply` les retire) ; reprise page par page de LightOnOCR (`ALBERT_OCR_CHECKPOINT`) ; images en pièces jointes ; audio Whisper (`ALBERT_AUDIO_ENABLED`, `scripts/rad_audio.py`, `app/routes/albert_audio.py`) ; usage du compte ; évaluation humaine `scripts/eval/albert/`.
- **Vectors are validated, never faked** (audit A06/A07): dense vectors pass `rad_providers.valid_dense_vector` (finite, non-zero, right dimension) before cache, file and insertion — a failed OpenAI embedding is `None`, never a zero vector, and the dense phase exits 1 above `EMBED_MAX_MISSING_RATIO`; sparse indices come from the stable, versioned `scripts/rad_sparse.py` (blake2b, not the per-process `hash()`).
- **Web subprocesses** (audit A08): `run_subprocess_with_sse` / `run_tracked_subprocess` run each script in its own process group under a single deadline enforced by a supervisor task; a timeout stops the whole group, a client disconnect does NOT (the SSE job keeps running, drained, stoppable, holding its session lock until it ends); server shutdown stops every registered script (`process_manager.stop_all`).

**Persistence/session model:** each web session writes artifacts to `uploads/<session_id>/` so any stage can be re-entered with an "Upload existing" button (the UI Steps 3.1–3.3 accept `output.csv`, `output_chunks.json`, or `output_chunks_with_embeddings.json` directly).

## Required env vars

**Registre unique (sprint « configuration unifiée », `.claude/tasks/SPRINT_config_unifiee.md`).** Toute variable lue par le code est déclarée dans `scripts/rad_settings/registry.py` (bloc, sous-bloc, nature, portée, défaut actuel du code) ; `tests/test_settings_registry.py` le vérifie par analyse AST. `.env.example` est généré depuis ce registre (`env_tool example`, contrôlé par `tests/test_env_tool.py`). Nouvelle variable = l'ajouter au registre puis régénérer. Le `.env` est relu à chaud quand il change (`scripts/rad_settings/access.py` : middleware par requête, avant chaque sous-processus et chaque tâche Celery ; l'environnement réel l'emporte ; variables `deploy` au redémarrage) ; le démarrage refuse un `.env` incomplet (`env_tool sync --apply`), sauf `RAGPY_SETTINGS_STRICT=0` (posé par le conftest). **Modèles (lot 4)** : couple serveur (adresse de l'API, déclarée au bloc 1 avec sa clé) + modèle (envoyé tel quel) par service (`LLM_{DEFAULT,RECODE,NOTES,BOOK,CITATIONS,RAG}_{SERVER,MODEL}`, `scripts/rad_settings/models.py`). `scripts/rad_settings/chat.py::routed_model_for` traduit le couple en modèle routé compris par tout le code : formes historiques pour OpenAI/OpenRouter (adresses officielles) et Albert (`albert/<id>`), forme interne `@<serveur>:<modèle>` sinon, que `rad_providers.resolve_llm_provider` résout en `compat:<serveur>` (client compatible OpenAI sur l'adresse déclarée, clé passée en `server_api_key` dans le web, lue dans l'environnement par `rad_chunk`). Sans `LLM_DEFAULT_SERVER` (« mode historique », D2), rien n'est traduit : comportement d'avant le sprint, goldens inchangés. Les tests sont hermétiques : le conftest pointe `RAGPY_ENV_FILE` vers un fichier absent (aucun `.env` du poste n'est lu). **OCR (lot 5)** : `OCR_SERVER` + `OCR_MODEL` déclarés → `rad_dataframe.configure_explicit_ocr` + `preflight_explicit_ocr` (sinon `exit 2`), un seul moteur (`_extract_text_explicit` : Mistral, Albert `/v1/ocr` ou page par page selon le type du modèle, `local` + `docling`/`pymupdf`, serveur local page par page `local_chat`), aucun repli implicite ; repli seulement s'il est déclaré (`OCR_SERVER_FALLBACK` + `OCR_MODEL_FALLBACK`, autre serveur, contrôlé aussi : document en échec repris par lui, erreur de compte du principal = suite du lot par lui, `fallback_from` + `OCR_PROVIDER_FALLBACK`) ; document en échec ni écrit ni marqué traité ; `OCRAccountError` (sans repli, ou du repli) arrête le lot (`exit 1`) ; colonnes `texteocr_server`/`texteocr_model` du moteur qui a servi. Routes et Celery : clés des serveurs déclarés seulement (`app/services/ocr_target.py`, choix personnels compris côté Celery). Sans `OCR_MODEL` : chaîne historique, goldens G1/G13 inchangés. **Choix personnels (lot 9)** : table `user_settings` (`app/services/user_settings.py`, `GET/PUT /users/me/settings`), priorité champ de l'étape > choix personnel > variable du service > défaut ; clés `anthropic/google/deepseek/qwen/glm_api_key` personnelles, visibles seulement quand l'adresse du serveur est déclarée ; adresses personnelles d'un non-admin contrôlées par `app/core/url_policy.py` (SSRF). Affichage conditionné à `unified_mode` (pages identiques en mode historique). **Embeddings, rerank, audio (lot 6)** : `scripts/rad_settings/capabilities.py` (`declared_choice`, `albert_overlay` lu par `AlbertConfig.from_env`) ; `EmbeddingConfig.from_env` route le couple `EMBEDDING_*` (OpenAI, Albert, ou serveur compatible : `compat_server`, dimension mesurée au premier appel dans `rad_chunk`) ; routes dense et Celery via `app/services/embedding_target.py`. Un modèle déclaré active rerank/transcription (Albert seulement) ; couple vide = règle historique.

Minimum: `OPENAI_API_KEY`. Production: `RAGPY_ENV=production` + a random `JWT_SECRET_KEY` (startup refused otherwise; rotation via `JWT_SECRET_KEY_PREVIOUS` + `scripts/rotate_credentials_key.py`), `CORS_ORIGINS`, `FLOWER_USER`/`FLOWER_PASSWORD`; limits `UPLOAD_MAX_MB`, `UPLOAD_MAX_UNZIPPED_MB`, `UPLOAD_MAX_ZIP_ENTRIES`, `MAX_ACTIVE_JOBS`, `MAX_ACTIVE_JOBS_PER_USER`. Add per use case: `MISTRAL_API_KEY` (OCR), `OPENROUTER_API_KEY`, `PINECONE_API_KEY`, `WEAVIATE_URL`+`WEAVIATE_API_KEY`, `QDRANT_URL`+`QDRANT_API_KEY`, `ENABLE_CELERY` + `CELERY_BROKER_URL` + `CELERY_RESULT_BACKEND`, `ZOTERO_API_KEY`+`ZOTERO_USER_ID`, `RESEND_API_KEY` (email verification). `USERS_SANDBOX=TRUE` holds new non-admin sign-ups until an admin approves them. `JWT_SECRET_KEY` is empty in `.env.example` and falls back to a placeholder in `app/config.py` — set it in prod, once (see the credentials note above). Albert (opt-in) : `ALBERT_ENABLED` + `ALBERT_API_KEY`, puis `OCR_ENABLE_ALBERT` / `EMBEDDING_PROVIDER` par capacité (bloc ALBERT de `.env.example`). See `.env.example`. Plus d'invite interactive : sans clé OpenAI, `rad_chunk.py` affiche un message et sort en `exit 1` pour la phase qui l'exige.
