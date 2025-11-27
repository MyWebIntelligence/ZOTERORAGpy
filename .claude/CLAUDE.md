# RAGpy - Guide d'utilisation et architecture

**Date de création** : 2025-10-21
**Dernière mise à jour** : 2025-11-27 (Phase 3: Architecture Celery pour production)

Ce document constitue le guide de référence pour le projet **RAGpy**, un pipeline sophistiqué de Retrieval-Augmented Generation conçu pour traiter des documents académiques. Il couvre l'utilisation des agents CLI, l'architecture du système et les bonnes pratiques d'implémentation.

> **Note d'architecture** : RAGpy combine une interface web FastAPI moderne avec un pipeline modulaire de traitement. L'application supporte multiple sources d'ingestion (Zotero+PDFs, CSV direct, fichiers manuels) et s'intègre avec diverses bases vectorielles (Pinecone, Weaviate, Qdrant).

> **Astuce interface** : dans l'UI FastAPI, les étapes 3.1 à 3.3 proposent un couple « Upload » / « Generate » pour réinjecter respectivement `output.csv`, `output_chunks.json` ou `output_chunks_with_embeddings.json`. Sans fichier téléversé, l'étape réutilise automatiquement le résultat précédent afin de reprendre un traitement interrompu.

---

## Déploiement Docker (recommandé)

### Démarrage rapide

```bash
# Cloner et configurer
git clone <URL_DU_DEPOT> && cd ragpy
cp .env.example .env
# Éditer .env avec vos clés API

# Lancer
docker compose up -d

# Accéder
open http://localhost:8000
```

### Commandes Docker

```bash
docker compose logs -f ragpy      # Logs temps réel
docker compose down               # Arrêter
docker compose up -d --build      # Reconstruire
docker compose exec ragpy bash    # Shell conteneur
```

### Volumes persistants

- `./data` : Base de données SQLite
- `./uploads` : Sessions de traitement
- `./logs` : Journaux applicatifs
- `./sources` : Fichiers sources (optionnel)

### Qdrant local (optionnel)

Décommentez la section `qdrant` dans `docker-compose.yml` pour une base vectorielle locale.

## Vue d'ensemble des agents

| Agent | Localisation | Rôle principal | Commande de base |
| --- | --- | --- | --- |
| `ragpy_cli.sh` | `ragpy/ragpy_cli.sh` | Gestion du serveur FastAPI (UI) | `./ragpy_cli.sh <start|close|kill>` |
| `rad_dataframe.py` | `ragpy/scripts/rad_dataframe.py` | Extraction Zotero + OCR PDF → CSV | `python scripts/rad_dataframe.py --json ... --dir ... --output ...` |
| `rad_chunk.py` | `ragpy/scripts/rad_chunk.py` | Chunking, recodage GPT, embeddings denses & sparses | `python scripts/rad_chunk.py --input ... --output ... --phase ...` |
| `rad_vectordb.py` | `ragpy/scripts/rad_vectordb.py` | Insertion dans Pinecone / Weaviate / Qdrant | `python - <<'PY' ...` (appel fonctionnel) |
| `crawl.py` | `ragpy/scripts/crawl.py` | Crawler HTML → PDF/Markdown pour constitution de corpus | `python scripts/crawl.py` |

### Pré-requis communs

- Python 3.8 ou plus et accès au dossier `ragpy/`.
- Environnement virtuel recommandé (`python -m venv .venv && source .venv/bin/activate`).
- Dépendances: `pip install -r scripts/requirements.txt` puis `pip install fastapi uvicorn jinja2 python-multipart`.
- Fichier `.env` à la racine contenant au minimum `OPENAI_API_KEY`. Ajouter les clés Pinecone / Weaviate / Qdrant selon les cibles.
- Répertoire `logs/` et `uploads/` existent par défaut; les scripts y écrivent automatiquement.

---

## Agent `ragpy_cli.sh` — Gestion du serveur FastAPI

### Mission
Automatiser le démarrage, l'arrêt et la purge du serveur FastAPI (`uvicorn`). Idéal pour piloter l'interface web lors d'ateliers ou de sessions Vibe Coding.

### Exécution
```bash
# Depuis le dossier parent qui contient `ragpy/`
cd /chemin/vers/__RAG
./ragpy/ragpy_cli.sh start
```

### Sous-commandes
- `start` : lance `uvicorn ragpy.app.main:app` en arrière-plan (`nohup`). Écrit les logs dans `ragpy/ragpy_server.log`.
- `close` : envoie un `SIGTERM` doux au processus `uvicorn` repéré.
- `kill` : kill -9 du serveur et des scripts `python3 scripts/rad_*` résiduels.

### Points d'attention
- Vérifie d'abord si le serveur tourne déjà (affiche les PID détectés).
- Suppose que `uvicorn` est disponible dans l'environnement actif.
- Les journaux sont consultables via `tail -f ragpy/ragpy_server.log`.
- Pour un usage dans `ragpy/` directement, préférez `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`.

---

## Agent `rad_dataframe.py` — Extraction Zotero + OCR PDF

### Mission
Transformer un export Zotero (`.json` + arborescence `files/`) en CSV enrichi avec texte OCR. Première étape du pipeline CLI.

### Dépendances spécifiques
- `mistralai` et `requests` pour l'appel OCR Markdown côté Mistral.
- `openai` (fallback vision) et `fitz` (PyMuPDF) pour les alternatives.
- `pandas` pour l'assemblage du DataFrame.
- Logger configuré vers `logs/pdf_processing.log`.

### Variables d'environnement clés
- `MISTRAL_API_KEY` (obligatoire pour la voie OCR Mistral).
- `MISTRAL_OCR_MODEL` et `MISTRAL_API_BASE_URL` (optionnels selon l'endpoint OCR choisi).
- `OPENAI_API_KEY` et `OPENAI_OCR_MODEL` pour le fallback vision.
- `OPENAI_OCR_MAX_PAGES`, `OPENAI_OCR_MAX_TOKENS`, `OPENAI_OCR_RENDER_SCALE` pour contrôler les appels de secours.

### Paramètres CLI
```bash
python scripts/rad_dataframe.py \
  --json sources/MaBiblio/MaBiblio.json \
  --dir  sources/MaBiblio \
  --output sources/MaBiblio/output.csv
```
- `--json` : chemin de l'export Zotero (UTF-8).
- `--dir` : dossier de base permettant de résoudre les chemins PDF relatifs du JSON.
- `--output` : fichier CSV produit (le script crée le dossier si besoin).

### Comportement remarqué
- En cas de PDF introuvable, tente une recherche fuzzy (normalisation accent, Levenshtein ≤ 2) dans le dossier visé.
- `extract_text_with_ocr` commence par envoyer les PDF à l'endpoint `v1/ocr` de Mistral (upload + document_id), puis bascule sur un fallback OpenAI vision, avant de revenir au flux PyMuPDF historique si aucun service distant n'est disponible.
- Les métadonnées extraites incluent désormais `texteocr_provider` pour tracer l'origine (`mistral`, `openai`, `legacy`).
- Le CSV est encodé en `utf-8-sig` pour compatibilité Excel.

### Journaux & diagnostics
- Trace détaillée dans `logs/pdf_processing.log` (créé si absent).
- En console: progression `tqdm` pour les pages PDF et éléments Zotero.

---

## Agent `rad_chunk.py` — Chunking, recodage GPT, embeddings

### Mission
Enrichir le CSV issu de `rad_dataframe.py` via trois phases successives:
1. Chunking + sauvegarde JSON (`*_chunks.json`).
2. Recodage GPT + embeddings denses OpenAI (`*_chunks_with_embeddings.json`).
3. Embeddings sparses spaCy (`*_chunks_with_embeddings_sparse.json`).

### Dépendances & environnement

**Variables d'environnement obligatoires** :
- `OPENAI_API_KEY` - Embeddings + recodage GPT (saisi via `.env` ou prompt interactif)

**Variables d'optimisation** :
- `OPENROUTER_API_KEY` - Alternative économique (~75% économie sur recodage)
- `OPENROUTER_DEFAULT_MODEL` - Modèle par défaut (ex: `openai/gemini-2.5-flash`)
- `MAX_CONCURRENT_LLM_CALLS` - Limite globale d'appels LLM simultanés (défaut: 5)

**Librairies requises** :
- `langchain_text_splitters` - Chunking intelligent avec RecursiveTextSplitter
- `openai` - API embeddings et completion
- `spacy` (`fr_core_news_md`) - NLP français pour embeddings sparse
- `tqdm`, `pandas` - Utilitaires et manipulation données

**Configuration système** :
- Concurrency: `ThreadPoolExecutor` (par défaut `os.cpu_count() - 1`)
- Sauvegarde thread-safe avec verrou global `SAVE_LOCK`
- Chunking optimisé pour `text-embedding-3-large` (1000 tokens, overlap 150)

**Logique d'optimisation coûts** :
- Si `texteocr_provider` vaut `mistral` ou `csv` → skip recodage GPT automatiquement
- Support OpenRouter pour réduction drastique des coûts API

### Paramètres CLI
```bash
python scripts/rad_chunk.py \
  --input sources/MaBiblio/output.csv \
  --output sources/MaBiblio \
  --phase all
```
- `--input` : CSV (phase `initial`) ou JSON (phases `dense`/`sparse`).
- `--output` : dossier cible des JSON (créé si besoin).
- `--phase` : `initial`, `dense`, `sparse`, ou `all` (enchaîne les trois).

### Détails par phase

- **initial** : lit un CSV, découpe le champ `texteocr` en chunks (~1 000 tokens avec chevauchement 150), recode via GPT (`gpt-4o-mini`) uniquement si l'OCR ne provient pas de Mistral ou CSV direct, puis sauvegarde `output_chunks.json`.
- **dense** : attend un fichier `_chunks.json`, génère les embeddings denses OpenAI (`text-embedding-3-large`), écrit `_chunks_with_embeddings.json`.
- **sparse** : attend `_chunks_with_embeddings.json`, dérive les features spaCy (POS filtrés, lemmas, TF normalisé, hachage mod 100 000), sauvegarde `_chunks_with_embeddings_sparse.json`.
- **all** : enchaîne les trois sous-étapes avec journalisation dans `<output>/chunking.log`.

### Support OpenRouter (économie coûts)

Le script supporte **OpenRouter** comme alternative économique (~75% moins cher) pour le recodage GPT :

```bash
# Utiliser OpenAI (défaut)
python scripts/rad_chunk.py --input data.csv --output ./out --phase initial

# Utiliser OpenRouter (économique)
python scripts/rad_chunk.py --input data.csv --output ./out --phase initial \
  --model openai/gemini-2.5-flash
```

**Auto-détection** : Les modèles avec format `provider/model` utilisent automatiquement OpenRouter. Fallback vers OpenAI si OpenRouter indisponible.

### Comportement complémentaire
- Si la clé OpenAI est absente, le script la demande et propose de la stocker via `python-dotenv`.
- SpaCy : tronque les textes très longs à `nlp.max_length` (ou 50 000 caractères) pour éviter les dépassements.
- Les identifiants de chunk incluent `doc_id`, `chunk_index`, `total_chunks` pour faciliter l'upload.
- Les erreurs d'API GPT sont réessayées séquentiellement (seconde passe) avant fallback sur le texte brut.

### Bonnes pratiques Vibe Coding
1. Vérifier le `.env` avant lancement (`OPENAI_API_KEY`, etc.).
2. Lancer la phase `initial` seule pour valider le découpage, puis `dense`/`sparse` si les coûts OpenAI sont confirmés.
3. Sur de gros corpus, limiter `DEFAULT_MAX_WORKERS` via variable d'environnement pour éviter de saturer l'API.
4. Contrôler les fichiers générés dans `uploads/<session>/` ou `sources/<projet>/` avant ingestion vectorielle.

---

## Agent `rad_vectordb.py` — Insertion dans les bases vectorielles

### Mission
Consommer `*_chunks_with_embeddings_sparse.json` et pousser les vecteurs + métadonnées vers Pinecone, Weaviate (multi-tenants) ou Qdrant.

### Dépendances & configurations
- `pinecone` SDK (>=3.x), `weaviate-client`, `qdrant-client`, `python-dateutil`.
- Variables d'environnement :
  - Pinecone : `PINECONE_API_KEY` (+ `PINECONE_ENV` si nécessaire).
  - Weaviate : `WEAVIATE_URL`, `WEAVIATE_API_KEY`.
  - Qdrant : `QDRANT_URL`, `QDRANT_API_KEY` (optionnelle selon l'instance).
- Tailles de lot par défaut : `PINECONE_BATCH_SIZE = 100`, `WEAVIATE_BATCH_SIZE = 100`, `QDRANT_BATCH_SIZE = 100`.

### Modes d'appel recommandés
```bash
python - <<'PY'
from scripts.rad_vectordb import insert_to_pinecone
import os
res = insert_to_pinecone(
    embeddings_json_file='sources/MaBiblio/output_chunks_with_embeddings_sparse.json',
    index_name='articles-demo',
    pinecone_api_key=os.getenv('PINECONE_API_KEY')
)
print(res)
PY
```
Remplacer `insert_to_pinecone` par `insert_to_weaviate_hybrid` ou `insert_to_qdrant` selon la cible.

### Spécificités par connecteur
- **Pinecone** :
  - Vérifie la présence de l'index dans `pc.list_indexes()`. Aucun auto-create dans ce script; créer l'index en amont avec la bonne dimension (embeddings OpenAI = 3 072).
  - Supporte les vecteurs sparses (`sparse_values`) si fournis.
  - Retry sur les erreurs d'upsert avec délai de 2s.

- **Weaviate (hybride)** :
  - Connexion `weaviate.connect_to_weaviate_cloud` avec auth API key.
  - Vérifie/crée le tenant (`collection.tenants.create`). Paramètre par défaut `tenant_name="alakel"` à modifier selon projet.
  - Cast les ID chunk → UUID v5 stable (`generate_uuid`).
  - Normalise les dates en RFC3339 (`normalize_date_to_rfc3339`).
  - Batching via `collection.with_tenant(...).data.insert_many`.

- **Qdrant** :
  - Tente de récupérer/creer la collection (`client.create_collection`) en inférant la dimension depuis le premier chunk valide.
  - Upsert synchrone avec `wait=True` et vérification du statut `COMPLETED`.
  - Fournit un résumé final (`Total de points insérés`).

### Vérifications avant ingestion
1. Nettoyer les métadonnées dans le JSON d'entrée (titres, dates) pour éviter les conversions invalides.
2. Contrôler l'espace disque: chaque JSON peut peser plusieurs centaines de Mo selon le corpus.
3. Exécuter un lot test (10-20 chunks) avant d'envoyer l'ensemble pour valider credentials et schéma.
4. Sur Weaviate multi-tenant, confirmer que la classe (`class_name`) est déjà définie côté cluster (schema management hors scope du script).

---

## Agent `crawl.py` — Constitution rapide de corpus web

### Mission
Crawler un site (par défaut `https://docs.n8n.io/integrations/`), enregistrer chaque page en PDF (via `wkhtmltopdf` ou Playwright) et Markdown simplifié. Utile pour enrichir un corpus avant passage dans `rad_dataframe.py`.

### Usage
```bash
python scripts/crawl.py
```

### Points clés
- Nécessite `requests`, `beautifulsoup4`, `playwright`. Pour PDF fidèle, installer `wkhtmltopdf` (sinon fallback Playwright headless).
- Enregistre les ressources dans `pages_pdf/` et `pages_md/` créés automatiquement à la racine du script.
- Garde la navigation dans le domaine de départ (`is_internal_link`).
- À adapter avant production: changer `START_URL`, corriger l'oubli de deux-points dans la condition `if response.status_code != 200` si besoin.

---

## Pipeline CLI recommandé

1. **Préparer la source** :
   ```bash
   python scripts/rad_dataframe.py --json sources/MaBiblio/MaBiblio.json --dir sources/MaBiblio --output sources/MaBiblio/output.csv
   ```
2. **Chunk + embeddings** :
   ```bash
   python scripts/rad_chunk.py --input sources/MaBiblio/output.csv --output sources/MaBiblio --phase all
   ```
3. **Upload vectoriel** (ex. Pinecone) :
   ```bash
   python - <<'PY'
from scripts.rad_vectordb import insert_to_pinecone
import os
res = insert_to_pinecone(
    embeddings_json_file='sources/MaBiblio/output_chunks_with_embeddings_sparse.json',
    index_name='ma-collection',
    pinecone_api_key=os.getenv('PINECONE_API_KEY')
)
print(res)
PY
   ```
4. **Lancer l'UI** si nécessaire : `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000` ou `./ragpy_cli.sh start`.

---

## Interface web FastAPI

### Démarrage et gestion
```bash
# Démarrage développement
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Gestion via script CLI
./ragpy_cli.sh start    # Démarrage arrière-plan
./ragpy_cli.sh close    # Arrêt propre
./ragpy_cli.sh kill     # Arrêt forcé + nettoyage processus
```

### Endpoints principaux
- **Upload** : `POST /upload_zip`, `POST /upload_csv` - Ingestion multi-sources
- **Processing** : `POST /process_dataframe`, `/initial_text_chunking`, `/dense_embedding_generation`
- **Vector DB** : `POST /upload_db` - Insertion Pinecone/Weaviate/Qdrant
- **Zotero** : `POST /generate_zotero_notes` - Génération notes académiques automatiques
- **Configuration** : `GET/POST /get_credentials`, `/save_credentials` - Gestion API keys

### Fonctionnalités avancées
- **Server-Sent Events (SSE)** pour suivi temps réel des traitements longs
- **Gestion de sessions** avec répertoires uniques par utilisateur
- **Upload artifacts intermédiaires** pour reprendre traitements interrompus
- **Interface credentials** pour configuration API keys via UI

---

## Ingestion CSV directe

### Module csv_ingestion.py

Permet d'injecter des fichiers CSV dans le pipeline en **contournant l'OCR** :

```python
from ingestion.csv_ingestion import ingest_csv, CSVIngestionConfig

# Configuration flexible
config = CSVIngestionConfig(
    text_column="description",      # Colonne source du texte
    encoding="auto",                # Détection automatique encoding
    skip_empty=True,                # Ignorer lignes vides
    add_row_index=True              # Ajouter métadonnées row_index
)

# Ingestion
documents = ingest_csv("data.csv", config)
df = pd.DataFrame([doc.to_dict() for doc in documents])
df.to_csv("output.csv", index=False)  # Compatible pipeline
```

### Avantages
- **Économie API** : Pas de recodage GPT (texteocr_provider="csv")
- **Métadonnées préservées** : Toutes colonnes CSV conservées
- **Validation robuste** : Gestion encoding, erreurs, lignes vides
- **Flexibilité** : Configuration mapping colonnes via YAML

---

## Architecture et points critiques

### Problème métadonnées hardcodées ⚠️

**Issue majeure** : Les métadonnées sont hardcodées dans 4 emplacements du pipeline :

```python
# rad_chunk.py:250-263 & rad_vectordb.py (3 connecteurs)
chunk_metadata = {
    "title": row_data.get("title", ""),
    "authors": row_data.get("authors", ""),
    # ... 8 autres champs hardcodés
}
# Impact: Colonnes CSV personnalisées perdues → Limitation filtrage vectoriel
```

**Solution recommandée** :
```python
# Injection dynamique de toutes les métadonnées
metadata = {k: v for k, v in chunk.items()
            if k not in ("id", "embedding", "sparse_embedding", "text")}
```

### Forces architecturales ✅

- **Modularité excellente** : Séparation claire des responsabilités
- **Classe Document unifiée** : Abstraction robuste pour toutes sources
- **Pipeline flexible** : Support multi-sources et multi-providers
- **Optimisation coûts** : OpenRouter, skip recodage intelligent
- **Intégration académique** : Zotero bidirectionnel sophistiqué

### Dépendances critiques (épinglées 2025-11-24)

```python
# Pipeline core
pandas>=2.2.2                    # Manipulation données
pymupdf==1.24.2                  # PDF extraction
openai==1.50.2                   # Embeddings + completion
langchain-text-splitters>=0.3.9  # Chunking intelligent (CVE-2025-6985 fix)
spacy==3.7.5                     # NLP français
tiktoken==0.7.0                  # Tokenisation OpenAI

# Vector databases
pinecone-client==5.0.1           # Hybrid search
weaviate-client==4.8.1           # Multi-tenancy
qdrant-client==1.11.1            # Vector similarity

# Web interface
fastapi==0.115.0                 # API moderne
uvicorn==0.30.6                  # ASGI server
jinja2>=3.1.6                    # Templates (CVE-2024-56326 fix)
python-multipart>=0.0.18         # Upload fichiers (CVE-2024-24762 fix)

# Authentication
sqlalchemy==2.0.35               # ORM
python-jose[cryptography]>=3.4.0 # JWT (CVE-2024-33663 fix)
bcrypt==4.0.1                    # Hashing

# Dev & test
pytest==8.3.3                    # Tests
httpx<=0.27.2                    # HTTP client async
chardet==5.2.0                   # Détection encoding
```

---

## Conseils opérationnels et bonnes pratiques

### Développement et debugging
- **Centraliser les clés** dans `.env` et utiliser `source .env` lors des sessions
- **Contrôler les logs** : `logs/app.log`, `logs/pdf_processing.log`, `<output>/chunking.log`
- **Valider par étapes** : Lancer `initial` seul avant `dense`/`sparse` pour valider coûts
- **Optimiser concurrence** : Ajuster `DEFAULT_MAX_WORKERS` selon quotas API

### Production et sécurité
- **Versions épinglées** ✅ : `scripts/requirements.txt` avec versions fixes (2025-11-24)
- **Docker disponible** ✅ : Déploiement simplifié avec `docker compose up -d`
- **Restreindre CORS** : Modifier configuration permissive développement
- **Authentification JWT** ✅ : Implémentée avec vérification email (Resend)
- **Validation stricte** : Implémenter Pydantic models pour validation entrées

### Optimisation performances
- **Session cleanup** : Nettoyer `uploads/` après usage sur machines partagées
- **Batch sizing** : Vérifier tailles lots et quotas API avant traitements massifs
- **Monitoring** : Surveiller usage mémoire et temps traitement par phase
- **Cache embeddings** : Considérer Redis pour éviter recalculs

---

## Contrôle de concurrence LLM (2025-11-25)

### Sémaphore global

Le système utilise un **sémaphore asyncio global** pour limiter les appels LLM concurrents à travers **tous les utilisateurs** de la plateforme.

**Configuration** (`.env`) :
```bash
MAX_CONCURRENT_LLM_CALLS=5      # Limite globale (défaut: 5)
```

**Comportement** :
- Maximum N appels LLM simultanés sur toute la plateforme
- Les requêtes excédentaires attendent qu'un slot se libère
- Protection contre surcharge API et rate limits
- Logs de debug pour tracer acquisition/release des slots

**Fonctions async avec contrôle** :
```python
# app/utils/llm_note_generator.py
await build_note_html_async(...)       # Mode étendu
await build_abstract_text_async(...)   # Mode court
```

### Retry logic LLM

Les appels LLM incluent une logique de retry automatique :
- **1 retry** en cas d'erreur
- **2 secondes** de délai entre tentatives
- Logging détaillé de chaque tentative

---

## Architecture Celery (Phase 3 - Production)

### Vue d'ensemble

RAGpy supporte un mode **dual** pour l'exécution des tâches longues :

- **Mode subprocess** (défaut) : Exécution synchrone via `asyncio.create_subprocess_exec`
- **Mode Celery** (production) : Queue distribuée avec workers dédiés

```text
[User Request] → [FastAPI] → [Celery Queue] → [Worker 1]
                                            → [Worker 2]
                                            → [Worker 3]
                      ↓
                 [Redis Broker]
                      ↓
                 [Result Backend]
```

### Activation Celery

**Configuration** (`.env`) :
```bash
# Activer le mode Celery
ENABLE_CELERY=true

# Configuration Redis
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Monitoring Flower
FLOWER_USER=admin
FLOWER_PASSWORD=changeme
```

### Services Docker

```bash
# Démarrer stack complète avec Celery
docker compose up -d

# Services démarrés:
# - ragpy (FastAPI)
# - redis (broker)
# - celery_worker (4 workers)
# - celery_beat (tâches périodiques)
# - flower (monitoring :5555)
```

### Endpoints API Celery

Les endpoints Celery sont disponibles sous `/api/celery/` :

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/api/celery/process_dataframe` | POST | Soumettre extraction PDF |
| `/api/celery/initial_chunking` | POST | Soumettre chunking |
| `/api/celery/dense_embedding` | POST | Soumettre embeddings denses |
| `/api/celery/sparse_embedding` | POST | Soumettre embeddings sparses |
| `/api/celery/upload_vectordb` | POST | Soumettre upload vector DB |
| `/api/celery/task/{id}/status` | GET | Statut d'une tâche |
| `/api/celery/task/{id}/cancel` | POST | Annuler une tâche |
| `/api/celery/status` | GET | Statut système Celery |
| `/api/celery/workers` | GET | Info workers actifs |

### Tasks Celery

Les tâches sont définies dans `app/tasks/` :

```python
# Extraction PDF
from app.tasks.extraction import process_dataframe_task
task = process_dataframe_task.delay(json_path, base_dir, output_path, session_id)

# Chunking
from app.tasks.chunking import initial_chunking_task
task = initial_chunking_task.delay(input_csv, output_dir, session_id, model)

# Embeddings
from app.tasks.embeddings import dense_embedding_task, sparse_embedding_task

# Vector DB upload
from app.tasks.vectordb import upload_to_vectordb_task

# Cleanup (périodique)
from app.tasks.cleanup import cleanup_sessions_task
```

### Monitoring avec Flower

Accéder à l'interface Flower : `http://localhost:5555`

Fonctionnalités :

- Vue temps réel des workers
- Historique des tâches
- Statistiques de performance
- Retry/revoke de tâches

### Tâches périodiques (Beat)

Celery Beat exécute automatiquement :

- **cleanup-expired-sessions** : Toutes les 6h
- **update-system-metrics** : Chaque minute
- **cleanup-orphaned-processes** : Toutes les heures

### Migration subprocess → Celery

Le système supporte un **dual mode** avec fallback automatique :

```python
# processing.py - Le code existant continue de fonctionner
# celery_tasks.py - Nouveaux endpoints pour mode Celery

# Le frontend peut choisir:
# - POST /process_dataframe (subprocess, SSE)
# - POST /api/celery/process_dataframe (Celery, polling)
```

### Dépendances Celery

```python
# scripts/requirements.txt
celery==5.3.4
redis==5.0.1
flower==2.0.1
```
