# RAGpy

**Solution de veille scientifique intelligente pour chercheurs et doctorants**

RAGpy est un outil avancé de veille scientifique qui optimise et automatise le traitement de votre corpus bibliographique. Conçu pour s'intégrer parfaitement à votre flux de travail académique, il agit à plusieurs niveaux :

- **Optimisation de Publish or Perish** : Import et traitement raffiné des résultats de vos recherches bibliométriques.
- **Enrichissement Zotero** : Mise à jour dynamique de votre bibliothèque avec génération automatique de notes de lecture enrichies et classement structuré des items par clusters thématiques.
- **Gestion de Mémoires RAG** : Création et gestion de bases de connaissances vectorielles (Retrieval-Augmented Generation) permettant à vos LLMs d'exploiter la totalité de votre savoir scientifique avec précision.

RAGpy transforme une collection de documents statique en une base de connaissances vivante et exploitable.

---

## Qu'est-ce que RAGpy peut faire pour vous ?

```
📚 Vos documents                    🎯 Ce que RAGpy produit
─────────────────                   ─────────────────────────

  Zotero + PDFs     ─┐              ┌─► Base vectorielle (RAG)
                     │              │   → Recherche sémantique
  CSV / Excel      ──┼──► RAGpy ───┼─► Fiches de lecture Zotero
                     │              │   → Notes automatiques
  Dossier PDFs     ─┘              └─► Clusters thématiques
                                       → Tags automatiques Zotero
```
Voir le Pipeline complet

[![Pipeline complet](https://claude.ai/public/artifacts/1fab4f46-a04a-4e0f-9387-6d7db7fa37ce)](https://claude.ai/public/artifacts/1fab4f46-a04a-4e0f-9387-6d7db7fa37ce)

### Cas d'usage typiques

| Besoin | Solution RAGpy |
|--------|----------------|
| 🔍 **Recherche sémantique** | Injecter vos articles dans Pinecone/Weaviate/Qdrant pour un RAG personnalisé |
| 📝 **Fiches de lecture automatiques** | Générer des résumés structurés et les pousser dans Zotero |
| 🏷️ **Organisation automatique** | Clusterer vos documents par thème et créer des tags Zotero |
| 📊 **Analyse de corpus** | Identifier les groupes thématiques dans votre bibliographie |

---

## Nouveautés

- **Clustering automatique (UMAP + HDBSCAN)** : Regroupez vos documents par similarité sémantique et générez des tags Zotero automatiquement
- **Authentification utilisateur complète** avec vérification email (Resend) et gestion des rôles
- **Sécurité credentials role-based** : Isolation ADMIN (fallback `.env`) / NON-ADMIN (credentials personnels uniquement)
- **OCR Mistral** pour extraction PDF haute qualité
- **Ingestion CSV directe** (bypass OCR) pour économiser temps et coûts API
- **Génération automatique de fiches de lecture Zotero** via LLM avec push automatique
- **Contrôle de concurrence LLM** : Sémaphore global limitant les appels simultanés

---

## Sommaire

- [A — Installation](#a--installation)
  - [1) Installation Docker (recommandée)](#1-installation-docker-recommandée)
  - [2) Installation manuelle](#2-installation-manuelle)
  - [3) Configuration (.env)](#3-configuration-env)
- [B — Le Pipeline en 5 étapes](#b--le-pipeline-en-5-étapes)
  - [Comprendre le flux de traitement](#comprendre-le-flux-de-traitement)
  - [4) Utilisation de l'interface web](#4-utilisation-de-linterface-web)
  - [5) Clustering automatique et tags Zotero](#5-clustering-automatique-et-tags-zotero)
  - [6) Génération de fiches de lecture Zotero](#6-génération-de-fiches-de-lecture-zotero)
  - [7) Authentification et gestion utilisateurs](#7-authentification-et-gestion-utilisateurs)
  - [8) Utilisation en ligne de commande](#8-utilisation-en-ligne-de-commande)
- [C — Projet](#c--projet)
  - [9) Architecture technique](#9-architecture-technique)
  - [10) Dépannage (FAQ)](#10-dépannage-faq)
  - [11) Licence](#11-licence)

---

## A — Installation

### 1) Installation Docker (recommandée)

**Prérequis** : Docker et Docker Compose installés ([Get Docker](https://docs.docker.com/get-docker/))

```bash
# 1. Cloner le dépôt
git clone <URL_DU_DEPOT> && cd ragpy

# 2. Créer le fichier .env
cp .env.example .env
# Éditer .env avec vos clés API (voir section 3)

# 3. Lancer l'application
docker compose up -d

# 4. Accéder à l'interface
open http://localhost:8000
```

**Commandes utiles** :
```bash
docker compose logs -f ragpy          # Voir les logs
docker compose down                   # Arrêter
docker compose up -d --build          # Reconstruire après modification
docker compose exec ragpy bash        # Accéder au conteneur
```

**Volumes persistants** :
- `./data` : Base de données SQLite
- `./uploads` : Sessions de traitement
- `./logs` : Journaux applicatifs

---

### 2) Installation manuelle

**Prérequis** : Python 3.8+, pip, git

```bash
# 1. Cloner et configurer l'environnement
git clone <URL_DU_DEPOT> && cd ragpy
python3 -m venv .venv
source .venv/bin/activate

# 2. Installer les dépendances
pip install --upgrade pip
pip install -r scripts/requirements.txt
python3 -m spacy download fr_core_news_md

# 3. Configurer et lancer
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

### 3) Configuration (.env)

```env
# ══════════════════════════════════════════════════════════════
# OBLIGATOIRE
# ══════════════════════════════════════════════════════════════
OPENAI_API_KEY=sk-...                      # Embeddings + recodage GPT

# ══════════════════════════════════════════════════════════════
# OCR (recommandé pour PDF)
# ══════════════════════════════════════════════════════════════
MISTRAL_API_KEY=...                        # OCR haute qualité
MISTRAL_OCR_MODEL=mistral-ocr-latest

# ══════════════════════════════════════════════════════════════
# EMAIL / AUTHENTIFICATION
# ══════════════════════════════════════════════════════════════
RESEND_API_KEY=re_...                      # Vérification email
RESEND_FROM_EMAIL=noreply@votredomaine.com
APP_URL=http://localhost:8000

# ══════════════════════════════════════════════════════════════
# ALTERNATIVE ÉCONOMIQUE (2-3x moins cher)
# ══════════════════════════════════════════════════════════════
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_DEFAULT_MODEL=google/gemini-2.5-flash

# ══════════════════════════════════════════════════════════════
# BASES VECTORIELLES (au moins une)
# ══════════════════════════════════════════════════════════════
PINECONE_API_KEY=pcsk-...
WEAVIATE_URL=https://...
WEAVIATE_API_KEY=...
QDRANT_URL=https://...
QDRANT_API_KEY=...

# ══════════════════════════════════════════════════════════════
# ZOTERO (pour fiches et clustering)
# ══════════════════════════════════════════════════════════════
ZOTERO_API_KEY=...
```

---

## B — Le Pipeline en 5 étapes

### Comprendre le flux de traitement

RAGpy fonctionne comme une chaîne de traitement où chaque étape prépare les données pour la suivante :

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE RAGpy                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ÉTAPE 1           ÉTAPE 2              ÉTAPE 3                         │
│  ────────          ────────             ────────                        │
│                                                                          │
│  📥 Import      →  🔪 Chunking       →  🧮 Embeddings                   │
│  (ZIP/CSV)         (découpage)          (vectorisation)                 │
│                                                                          │
│  • Zotero+PDF      • Segments ~1000     • Dense: OpenAI 3072D          │
│  • CSV direct        tokens             • Sparse: spaCy TF-IDF          │
│  • OCR Mistral     • Recodage GPT                                       │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ÉTAPE 4a              ÉTAPE 4b              ÉTAPE 5                    │
│  ─────────             ─────────             ────────                   │
│                                                                          │
│  📊 Vector DB    ou    🏷️ Clustering    ou   📝 Fiches Zotero          │
│  (RAG)                 (organisation)        (résumés)                  │
│                                                                          │
│  • Pinecone            • UMAP + HDBSCAN      • LLM automatique          │
│  • Weaviate            • Tags auto Zotero    • Notes enfants            │
│  • Qdrant              • 8-15 clusters       • Structure académique     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Que se passe-t-il à chaque étape ?

| Étape | Entrée | Traitement | Sortie |
|-------|--------|------------|--------|
| **1. Import** | ZIP (Zotero+PDF) ou CSV | OCR Mistral, extraction métadonnées | `output.csv` |
| **2. Chunking** | `output.csv` | Découpage ~1000 tokens + recodage GPT | `output_chunks.json` |
| **3a. Dense** | `output_chunks.json` | Embeddings OpenAI (3072D) | `..._embeddings.json` |
| **3b. Sparse** | `..._embeddings.json` | Embeddings spaCy (TF-IDF) | `..._sparse.json` |
| **4a. Vector DB** | `..._sparse.json` | Insertion Pinecone/Weaviate/Qdrant | Index RAG |
| **4b. Clustering** | `..._sparse.json` | UMAP → HDBSCAN → Tags | Tags Zotero |
| **5. Fiches** | `output.csv` | Génération LLM | Notes Zotero |

---

### 4) Utilisation de l'interface web

Ouvrez http://localhost:8000 après avoir démarré le serveur.

#### Deux options d'ingestion

**Option A : ZIP (Zotero + PDFs)** — Flux complet avec OCR
1. Téléverser un ZIP (export Zotero: JSON + `files/` avec PDFs)
2. Lancer « Process dataframe » → OCR Mistral
3. Suivre les étapes 2, 3a, 3b

**Option B : CSV (Direct)** — Bypass OCR (80% moins cher)
1. Téléverser un CSV avec colonne `text` (ou `description`, `content`)
2. **Skip** l'étape 1 → passe directement au chunking
3. Pas d'OCR ni de recodage GPT

#### Où sont stockés les fichiers ?

```
uploads/<session>/
├── output.csv                              # Étape 1
├── output_chunks.json                      # Étape 2
├── output_chunks_with_embeddings.json      # Étape 3a
├── output_chunks_with_embeddings_sparse.json  # Étape 3b
└── clustering_results.json                 # Étape 4b
```

---

### 5) Clustering automatique et tags Zotero

**Nouveau !** RAGpy peut automatiquement regrouper vos documents par similarité thématique et créer des tags Zotero.

#### Comment ça marche ?

```
   Documents avec embeddings (3072D)
              │
              ▼
   ┌─────────────────────┐
   │   UMAP (100D)       │  ← Réduction dimensionnelle
   │   Préserve les      │    préservant la structure
   │   relations         │    sémantique locale
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │   HDBSCAN           │  ← Clustering basé sur la densité
   │   Trouve les        │    Pas besoin de spécifier
   │   groupes naturels  │    le nombre de clusters !
   └──────────┬──────────┘
              │
              ▼
   Tags Zotero: _MaBiblio_01, _MaBiblio_02, ...
```

#### Utilisation dans l'interface

1. Complétez les étapes 1 à 3b (jusqu'aux embeddings sparse)
2. Dans **"4.b Cluster Documents to Zotero"** :
   - **Session Name** : Nom pour les tags (ex: `MaBiblio` → `_MaBiblio_01`)
   - **Min Cluster Size** : Taille minimum par cluster (optionnel)
     - Plus petit = plus de clusters
     - Recommandé : 5 pour ~10 clusters, 3 pour ~15 clusters
3. Cliquez **"Cluster Documents"**
4. Cliquez **"Apply Tags to Zotero"** pour synchroniser

#### Paramètres techniques

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| UMAP dimensions | 100D | Préserve plus de nuances sémantiques |
| UMAP min_dist | 0.05 | Clusters plus serrés |
| HDBSCAN method | `leaf` | Trouve plus de clusters granulaires |
| Metric | cosine → euclidean | Cosine pour UMAP, euclidean post-réduction |

#### Exemple de résultat

```json
{
  "n_documents": 160,
  "n_clusters": 12,
  "n_noise": 8,
  "cluster_sizes": {
    "0": 18, "1": 15, "2": 14, "3": 13, "4": 12,
    "5": 11, "6": 10, "7": 10, "8": 9, "9": 8, "10": 7, "11": 6
  }
}
```

#### Ligne de commande

```bash
python scripts/rad_clustering.py \
  --input uploads/session/output_chunks_with_embeddings_sparse.json \
  --output uploads/session \
  --session-name MaBiblio \
  --min-cluster-size 5 \
  --umap-components 100
```

---

### 6) Génération de fiches de lecture Zotero

RAGpy peut générer automatiquement des fiches de lecture académiques et les ajouter comme notes enfants dans Zotero.

#### Configuration

1. **Obtenir une clé API Zotero** sur https://www.zotero.org/settings/keys/new
   - ✅ "Allow library access"
   - ✅ "Allow notes access"

2. **Configurer dans Settings** (⚙️ en haut à droite)

#### Utilisation

Après les étapes 1-3, dans **"Zotero Reading Notes"** :
1. Cochez le mode (Extended Analysis ou Abstract)
2. Sélectionnez le modèle LLM
3. Cliquez "Generate Zotero Notes"

#### Structure des fiches

- **Référence bibliographique** : Titre, auteurs, date, DOI/URL
- **Problématique** : Question(s) de recherche
- **Méthodologie** : Approche, données, méthodes
- **Résultats clés** : Principales conclusions
- **Limites et perspectives** : Points faibles, questions ouvertes

#### Personnalisation

Éditez [app/utils/zotero_prompt.md](app/utils/zotero_prompt.md) pour modifier le template.

Placeholders : `{TITLE}`, `{AUTHORS}`, `{DATE}`, `{DOI}`, `{URL}`, `{ABSTRACT}`, `{TEXT}`, `{LANGUAGE}`

---

### 7) Authentification et gestion utilisateurs

RAGpy intègre un système d'authentification complet.

#### Rôles et credentials

| Rôle | Credentials personnels | Fallback `.env` |
|------|------------------------|-----------------|
| **ADMIN** | ✅ Prioritaire | ✅ Si vide |
| **NON-ADMIN** | ✅ Uniquement | ❌ JAMAIS |

#### Endpoints principaux

| Endpoint | Description |
|----------|-------------|
| `/auth/register` | Inscription + email vérification |
| `/auth/login` | Connexion (retourne JWT) |
| `/auth/verify-email/{token}` | Vérification email |
| `/auth/forgot-password` | Reset password |

---

### 8) Utilisation en ligne de commande

Pipeline complet depuis le terminal :

```bash
# 1. Extraction PDF+Zotero → CSV
python scripts/rad_dataframe.py \
  --json sources/MaBiblio/MaBiblio.json \
  --dir sources/MaBiblio \
  --output sources/MaBiblio/output.csv

# 2. Chunking + embeddings (toutes les phases)
python scripts/rad_chunk.py \
  --input sources/MaBiblio/output.csv \
  --output sources/MaBiblio \
  --phase all

# 3a. Insertion base vectorielle (Pinecone)
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

# 3b. Clustering automatique
python scripts/rad_clustering.py \
  --input sources/MaBiblio/output_chunks_with_embeddings_sparse.json \
  --output sources/MaBiblio \
  --session-name MaBiblio \
  --min-cluster-size 5
```

---

## C — Projet

### 9) Architecture technique

```
ragpy/
├── app/                      # Application web FastAPI
│   ├── main.py               # API + orchestration
│   ├── core/                 # Modules core (security, credentials)
│   ├── routes/               # Routes API (auth, admin, processing)
│   ├── utils/                # Utilitaires (zotero_client, llm_generator)
│   └── templates/            # Templates Jinja2
├── scripts/                  # Pipeline de traitement
│   ├── rad_dataframe.py      # JSON Zotero + PDFs → CSV (OCR)
│   ├── rad_chunk.py          # Chunking + embeddings
│   ├── rad_clustering.py     # UMAP + HDBSCAN clustering
│   ├── rad_vectordb.py       # Insertion vector DB
│   └── requirements.txt      # Dépendances
├── data/                     # Base SQLite
├── uploads/                  # Sessions de traitement
└── logs/                     # Logs applicatifs
```

#### Technologies clés

| Composant | Technologie |
|-----------|-------------|
| Backend | FastAPI + Uvicorn |
| Auth | JWT (python-jose) + bcrypt |
| BDD | SQLAlchemy + SQLite |
| OCR | Mistral OCR (fallback OpenAI Vision) |
| Embeddings | OpenAI `text-embedding-3-large` (3072D) |
| Sparse | spaCy FR `fr_core_news_md` |
| Clustering | UMAP + HDBSCAN |
| Vector DB | Pinecone, Weaviate, Qdrant |

---

### 10) Dépannage (FAQ)

#### Installation

| Problème | Solution |
|----------|----------|
| Port 8000 occupé | `lsof -i :8000` puis `kill <PID>` |
| Dépendances manquantes | `pip install -r scripts/requirements.txt` |
| spaCy manquant | `python -m spacy download fr_core_news_md` |

#### Clustering

| Problème | Solution |
|----------|----------|
| Trop peu de clusters | Réduire `min_cluster_size` (ex: 3 ou 5) |
| Trop de "noise" | Augmenter `min_cluster_size` ou vérifier la qualité des embeddings |
| HDBSCAN crash | Vérifier `scikit-learn<1.6` dans requirements.txt |

#### Zotero

| Problème | Solution |
|----------|----------|
| Clé API invalide | Vérifier permissions "library access" + "notes access" |
| Tags non appliqués | Vérifier `zotero_api_key` dans Settings |
| Erreur 404 | L'itemKey n'existe pas dans votre bibliothèque |

#### OCR

| Problème | Solution |
|----------|----------|
| Mistral OCR échoue | Vérifier `MISTRAL_API_KEY`, fallback auto vers OpenAI |
| Texte mal extrait | Augmenter `OPENAI_OCR_MAX_PAGES` |

---

### 11) Licence

MIT. Voir `LICENSE`.
