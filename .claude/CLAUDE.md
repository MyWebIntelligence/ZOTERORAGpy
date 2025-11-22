# Agents CLI – Vibe Coding

Ce document décrit en détail les agents en ligne de commande disponibles dans le projet `ragpy`. Il s'adresse aux utilisateurs des CLI Vibe Coding qui souhaitent orchestrer et automatiser le pipeline RAG sans passer par l'interface web.

> Astuce interface : dans l'UI FastAPI, les étapes 3.1 à 3.3 proposent un couple « Upload » / « Generate » pour réinjecter respectivement `output.csv`, `output_chunks.json` ou `output_chunks_with_embeddings.json`. Sans fichier téléversé, l'étape réutilise automatiquement le résultat précédent afin de reprendre un traitement interrompu.

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
- `OPENAI_API_KEY` obligatoire (saisi via `.env` ou prompt interactif).
- Librairies: `langchain_text_splitters`, `openai`, `spacy` (`fr_core_news_md` téléchargé si absent), `tqdm`, `pandas`.
- Concurrency: `ThreadPoolExecutor` (par défaut `os.cpu_count() - 1`).
- Sauvegarde thread-safe avec un verrou global `SAVE_LOCK`.
- Si `texteocr_provider` vaut `mistral`, la phase initiale saute le recodage GPT et réutilise les chunks Markdown tels quels pour éviter des appels OpenAI inutiles.

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
- **initial** : lit un CSV, découpe le champ `texteocr` en chunks (~2 500 tokens avec chevauchement 250), recode via GPT (`gpt-4o-mini`) uniquement si l'OCR ne provient pas de Mistral, puis sauvegarde `output_chunks.json`.
- **dense** : attend un fichier `_chunks.json`, génère les embeddings denses OpenAI (`text-embedding-3-large`), écrit `_chunks_with_embeddings.json`.
- **sparse** : attend `_chunks_with_embeddings.json`, dérive les features spaCy (POS filtrés, lemmas, TF normalisé, hachage mod 100 000), sauvegarde `_chunks_with_embeddings_sparse.json`.
- **all** : enchaîne les trois sous-étapes avec journalisation dans `<output>/chunking.log`.

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

## Conseils opérationnels Vibe Coding

- Centraliser les clés dans `.env` et utiliser `source .env` lors des sessions live.
- Sur machine partagée, nettoyer les zip/upload dans `uploads/` après usage.
- Capitaliser les logs (`logs/app.log`, `logs/pdf_processing.log`, `<output>/chunking.log`) pour documenter les ateliers.
- Vérifier systématiquement la taille des lots et les quotas API avant de lancer des traitements massifs.
