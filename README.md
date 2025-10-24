# RAGpy

Pipeline de traitement de documents (PDF, exports Zotero, **CSV**) et interface web pour générer des chunks de texte, produire des embeddings denses et parcimonieux, puis charger ces données dans une base vectorielle (Pinecone, Weaviate ou Qdrant) pour des usages RAG.

**Nouveau** : Support d'ingestion CSV directe (bypass OCR) pour économiser temps et coûts API.

---

## Sommaire

- [A — Usage](#a--usage)
  - [1) Installation (débutant)](#1-installation-débutant)
  - [2) Utilisation de l’interface web](#2-utilisation-de-linterface-web)
  - [3) Utilisation en ligne de commande](#3-utilisation-en-ligne-de-commande)
- [B — Projet](#b--projet)
  - [4) Le projet](#4-le-projet)
  - [5) Architecture de dev](#5-architecture-de-dev)
  - [6) Variables d’environnement (.env)](#6-variables-denvironnement-env)
  - [7) Dépannage (FAQ)](#7-dépannage-faq)
  - [8) Licence](#8-licence)

---

## A — Usage

### 1) Installation (débutant)

Prérequis:
- Python 3.8+
- pip, git

Étapes conseillées (macOS/Linux):
```bash
# 1. Cloner le dépôt
git clone <URL_DU_DEPOT> && cd ragpy

# 2. Créer un environnement virtuel
python3 -m venv .venv
source .venv/bin/activate

# 3. Mettre pip à jour et installer les dépendances
pip install --upgrade pip
pip install -r scripts/requirements.txt
pip install fastapi uvicorn jinja2 python-multipart

# 4. Installer le modèle spaCy FR (si textes FR)
python3 -m spacy download fr_core_news_md

# 5. Créer le fichier .env à la racine
cp scripts/.env.example .env  # si présent, sinon créez-le manuellement
```

Étapes (Windows PowerShell):
```powershell
git clone <URL_DU_DEPOT>
cd ragpy
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r scripts/requirements.txt
pip install fastapi uvicorn jinja2 python-multipart
python -m spacy download fr_core_news_md
copy scripts\.env.example .env
```

Contenu minimal de `.env` (à adapter):
```env
OPENAI_API_KEY=sk-...                      # Obligatoire (embeddings + recodage GPT)
OPENROUTER_API_KEY=sk-or-v1-...            # Optionnel (alternative économique pour recodage)
OPENROUTER_DEFAULT_MODEL=openai/gemini-2.5-flash  # Modèle par défaut OpenRouter
PINECONE_API_KEY=pcsk-...                  # Optionnel si Pinecone
WEAVIATE_URL=https://...                   # Optionnel si Weaviate
WEAVIATE_API_KEY=...                       # Optionnel si Weaviate
QDRANT_URL=https://...                     # Optionnel si Qdrant
QDRANT_API_KEY=...                         # Optionnel (instances publiques sans clé)
```

**Nouveau** : Support OpenRouter pour réduire les coûts de recodage de 2-3x (ex: Gemini 2.5 Flash ~$0.002/1M tokens vs GPT-4o-mini ~$0.15/1M tokens)

Notes:
- Placez `.env` à la racine de `ragpy/`.
- `langchain-text-splitters` est requis pour le découpage; il est listé dans `scripts/requirements.txt`.

### 2) Utilisation de l’interface web

Démarrer le serveur depuis `ragpy/`:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Ensuite, ouvrez http://localhost:8000

**Deux options d'ingestion disponibles** :

#### Option A : ZIP (Zotero + PDFs) - Flux complet avec OCR

- Téléverser un ZIP (export Zotero: JSON + `files/` avec PDFs, ou un dossier de PDFs)
- Lancer « Process dataframe » pour produire `uploads/<session>/output.csv` (OCR Mistral/OpenAI)
- Lancer successivement: « Initial chunking », « Dense embeddings », « Sparse embeddings »
- Dans « Upload to DB », choisir Pinecone / Weaviate / Qdrant et renseigner les infos

#### Option B : CSV (Direct) - **NOUVEAU** - Bypass OCR

- Téléverser un CSV avec une colonne `text` (ou `description`, `content`, etc.)
- **Skip** l'étape « Process dataframe » → passe directement au chunking
- Le reste du flux reste identique (chunking → embeddings → DB)
- **Avantage** : 80% moins de coûts API (pas d'OCR ni de recodage GPT)

**Documentation CSV** : Voir [.claude/task/CSV_INGESTION_GUIDE.md](.claude/task/CSV_INGESTION_GUIDE.md)

Où sont stockés les fichiers?
- Dans `uploads/<session>/` avec les sorties: `output.csv`, `output_chunks.json`, `output_chunks_with_embeddings.json`, `output_chunks_with_embeddings_sparse.json`.

**Note** : Les clés API proviennent de `.env` (réglables via le bouton « Settings ⚙️ » en haut à droite)

**Réduction des coûts avec OpenRouter** : Lors de l'étape "3.1 Initial Text Chunking", vous pouvez spécifier un modèle OpenRouter (ex: `openai/gemini-2.5-flash`) pour le recodage de texte au lieu de GPT-4o-mini. Cela réduit les coûts de ~75% tout en maintenant une qualité comparable. Configurez vos credentials OpenRouter dans Settings.

Astuce: un script shell d’aide `ragpy_cli.sh` existe pour démarrer/arrêter le serveur. Il suppose d’être exécuté depuis le dossier parent contenant `ragpy/`. Si vous êtes déjà dans `ragpy/`, préférez la commande `uvicorn app.main:app ...` ci‑dessus.

### 3) Utilisation en ligne de commande

Traitement complet (hors interface web) à partir d’un export Zotero placé dans `sources/MaBiblio/`:

1) Extraction PDF+Zotero vers CSV
```bash
python scripts/rad_dataframe.py \
  --json sources/MaBiblio/MaBiblio.json \
  --dir  sources/MaBiblio \
  --output sources/MaBiblio/output.csv
```

2) Chunking + embeddings denses + sparses
```bash
# Option A: Utiliser OpenAI GPT-4o-mini (défaut)
python scripts/rad_chunk.py \
  --input sources/MaBiblio/output.csv \
  --output sources/MaBiblio \
  --phase all

# Option B: Utiliser OpenRouter pour économiser sur le recodage (2-3x moins cher)
python scripts/rad_chunk.py \
  --input sources/MaBiblio/output.csv \
  --output sources/MaBiblio \
  --phase all \
  --model openai/gemini-2.5-flash
```
Sorties attendues dans `sources/MaBiblio/`:
- `output_chunks.json`
- `output_chunks_with_embeddings.json`
- `output_chunks_with_embeddings_sparse.json`

3) Chargement en base vectorielle (optionnel, programmatique)

Les fonctions d’insertion sont exposées dans `scripts/rad_vectordb.py` et sont appelées par l’interface web. Pour un usage CLI rapide, lancez‑les depuis Python:

Pinecone
```bash
python - <<'PY'
from scripts.rad_vectordb import insert_to_pinecone
import os
res = insert_to_pinecone(
    embeddings_json_file='sources/MaBiblio/output_chunks_with_embeddings_sparse.json',
    index_name='mon_index',
    pinecone_api_key=os.getenv('PINECONE_API_KEY')
)
print(res)
PY
```

Weaviate (multi‑tenants)
```bash
python - <<'PY'
from scripts.rad_vectordb import insert_to_weaviate_hybrid
import os
count = insert_to_weaviate_hybrid(
    embeddings_json_file='sources/MaBiblio/output_chunks_with_embeddings_sparse.json',
    url=os.getenv('WEAVIATE_URL'),
    api_key=os.getenv('WEAVIATE_API_KEY'),
    class_name='Article',
    tenant_name='default'
)
print('Inserted:', count)
PY
```

Qdrant
```bash
python - <<'PY'
from scripts.rad_vectordb import insert_to_qdrant
import os
count = insert_to_qdrant(
    embeddings_json_file='sources/MaBiblio/output_chunks_with_embeddings_sparse.json',
    collection_name='articles',
    qdrant_url=os.getenv('QDRANT_URL'),
    qdrant_api_key=os.getenv('QDRANT_API_KEY')
)
print('Inserted:', count)
PY
```

---

## B — Projet

### 4) Le projet

Objectif: transformer des documents (PDFs, exports Zotero) en données exploitables pour des systèmes RAG, via un pipeline reproductible et une interface web simple à utiliser.

Grandes étapes:
- Extraction texte + métadonnées depuis Zotero/PDF (`rad_dataframe.py`)
- Découpage en chunks, nettoyage GPT, embeddings denses et sparses (`rad_chunk.py`)
- Insertion dans une base vectorielle (Pinecone, Weaviate, Qdrant) (`rad_vectordb.py` via l’UI)

### 5) Architecture de dev

Arborescence principale:
```
ragpy/
├── app/                  # Application web FastAPI (UI)
│   ├── main.py           # API + orchestration des scripts
│   ├── static/           # Assets UI (CSS/JS/images)
│   └── templates/        # Templates Jinja2 (index.html)
├── scripts/              # Pipeline de traitement
│   ├── rad_dataframe.py  # JSON Zotero + PDFs -> CSV (OCR inclus)
│   ├── rad_chunk.py      # Chunking + recodage GPT + embeddings
│   ├── rad_vectordb.py   # Fonctions d’insertion (Pinecone/Weaviate/Qdrant)
│   └── requirements.txt  # Dépendances
├── uploads/              # Sessions de traitement depuis l’UI
├── logs/                 # Logs (app.log, pdf_processing.log)
├── .env                  # Clés/API (à créer)
└── ragpy_cli.sh          # Aide au démarrage serveur (optionnel)
```

Choix techniques clés:
- FastAPI + Uvicorn pour l’UI backend
- PyMuPDF (OCR léger) pour l’extraction PDF
- OpenAI API pour recodage GPT + embeddings (`text-embedding-3-large`)
- spaCy FR (`fr_core_news_md`) pour le sparse
- Pinecone, Weaviate (multi‑tenants), Qdrant pour le stockage

Journaux et sorties:
- `logs/app.log`, `logs/pdf_processing.log`
- Fichiers de session dans `uploads/<session>/`

### 6) Variables d'environnement (.env)

Clés supportées par l'UI et les scripts:

- `OPENAI_API_KEY` (obligatoire - embeddings + recodage par défaut)
- `OPENROUTER_API_KEY` (optionnel - alternative économique pour recodage)
- `OPENROUTER_DEFAULT_MODEL` (optionnel - ex: `openai/gemini-2.5-flash`)
- `PINECONE_API_KEY`, `PINECONE_ENV` (selon configuration Pinecone)
- `WEAVIATE_URL`, `WEAVIATE_API_KEY`
- `QDRANT_URL`, `QDRANT_API_KEY`

L'UI (« Settings ») permet de lire/écrire `.env` à la racine de `ragpy/`.

**OpenRouter** : Service permettant d'accéder à plusieurs LLM (Gemini, Claude, etc.) via une API unifiée. Particulièrement intéressant pour le recodage de texte grâce à des modèles comme Gemini 2.5 Flash (~75% moins cher que GPT-4o-mini). Les embeddings restent générés via OpenAI `text-embedding-3-large`.

### 7) Dépannage (FAQ)

- Pas de clé API: vérifiez `.env` et la section « Settings » de l’UI.
- Dépendances manquantes: `pip install -r scripts/requirements.txt` puis `pip install fastapi uvicorn jinja2 python-multipart`.
- spaCy manquant: `python -m spacy download fr_core_news_md`.
- Pinecone: créez l’index avec la dimension de votre modèle d’embedding.
- Weaviate: assurez‑vous que la classe existe et que le tenant est correct.
- Qdrant: la collection est créée si absente (dimension déduite du premier chunk).
- Chemins: en CLI, privilégiez des chemins absolus; via l’UI, tout est relatif à `uploads/`.

### 8) Licence

MIT. Voir `LICENSE`.

