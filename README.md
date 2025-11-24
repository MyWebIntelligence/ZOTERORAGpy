# RAGpy

Pipeline de traitement de documents (PDF, exports Zotero, **CSV**) et interface web pour générer des chunks de texte, produire des embeddings denses et parcimonieux, puis charger ces données dans une base vectorielle (Pinecone, Weaviate ou Qdrant) pour des usages RAG.

**Nouveau** :
- **Authentification utilisateur complète** avec vérification email (Resend) et gestion des rôles
- **OCR Mistral** pour extraction PDF haute qualité
- Support d'ingestion CSV directe (bypass OCR) pour économiser temps et coûts API
- **Génération automatique de fiches de lecture Zotero** via LLM avec push automatique vers votre bibliothèque

---

## Sommaire

- [A — Usage](#a--usage)
  - [1) Installation (débutant)](#1-installation-débutant)
  - [2) Utilisation de l'interface web](#2-utilisation-de-linterface-web)
  - [3) Authentification et gestion utilisateurs](#3-authentification-et-gestion-utilisateurs)
  - [4) Génération de fiches de lecture Zotero](#4-génération-de-fiches-de-lecture-zotero)
  - [5) Utilisation en ligne de commande](#5-utilisation-en-ligne-de-commande)
- [B — Projet](#b--projet)
  - [6) Le projet](#6-le-projet)
  - [7) Architecture de dev](#7-architecture-de-dev)
  - [8) Variables d'environnement (.env)](#8-variables-denvironnement-env)
  - [9) Dépannage (FAQ)](#9-dépannage-faq)
  - [10) Licence](#10-licence)

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
# Obligatoire
OPENAI_API_KEY=sk-...                      # Embeddings + recodage GPT

# OCR (recommandé)
MISTRAL_API_KEY=...                        # OCR haute qualité
MISTRAL_OCR_MODEL=mistral-ocr-latest
MISTRAL_API_BASE_URL=https://api.mistral.ai

# Email (recommandé pour auth)
RESEND_API_KEY=re_...                      # Vérification email + reset password
RESEND_FROM_EMAIL=noreply@votredomaine.com
APP_URL=http://localhost:8000              # URL pour les liens dans les emails

# Optionnel - Alternative économique pour recodage
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_DEFAULT_MODEL=google/gemini-2.5-flash

# Optionnel - Bases vectorielles (au moins une)
PINECONE_API_KEY=pcsk-...
WEAVIATE_URL=https://...
WEAVIATE_API_KEY=...
QDRANT_URL=https://...
QDRANT_API_KEY=...

# Optionnel - Zotero
ZOTERO_API_KEY=...
```

**Optimisations coûts** :
- **OpenRouter** : Réduit les coûts de recodage de 2-3x (Gemini 2.5 Flash ~$0.002/1M tokens vs GPT-4o-mini ~$0.15/1M tokens)
- **Mistral OCR** : OCR de haute qualité intégré, avec fallback vers OpenAI Vision si besoin

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

**Réduction des coûts avec OpenRouter** : Lors de l'étape "3.1 Initial Text Chunking", vous pouvez spécifier un modèle OpenRouter (ex: `google/gemini-2.5-flash`) pour le recodage de texte au lieu de GPT-4o-mini. Cela réduit les coûts de ~75% tout en maintenant une qualité comparable. Configurez vos credentials OpenRouter dans Settings.

Astuce: un script shell d'aide `ragpy_cli.sh` existe pour démarrer/arrêter le serveur. Il suppose d'être exécuté depuis le dossier parent contenant `ragpy/`. Si vous êtes déjà dans `ragpy/`, préférez la commande `uvicorn app.main:app ...` ci‑dessus.

### 3) Authentification et gestion utilisateurs

RAGpy intègre un système d'authentification complet avec vérification email.

#### Inscription et connexion

1. **Premier utilisateur** : Automatiquement promu administrateur et vérifié
2. **Utilisateurs suivants** : Doivent vérifier leur email avant d'accéder à l'application

#### Vérification email (Resend)

Le système utilise [Resend](https://resend.com) pour l'envoi d'emails :

- **Email de vérification** : Envoyé à l'inscription, lien valide 24h
- **Reset password** : Lien valide 1h (demande utilisateur ou admin)
- **Blocage automatique** : Utilisateurs non vérifiés ne peuvent pas accéder aux fonctionnalités

Configuration requise dans `.env` :
```env
RESEND_API_KEY=re_...                      # Clé API Resend
RESEND_FROM_EMAIL=noreply@votredomaine.com # Email expéditeur (domaine vérifié)
APP_URL=http://localhost:8000              # URL de base pour les liens
```

> **Note** : Sans configuration Resend, les tokens sont affichés en console (mode développement).

#### Administration utilisateurs

Les administrateurs peuvent :

- Voir la liste des utilisateurs (`/api/admin/users`)
- Activer/désactiver des comptes
- Promouvoir/rétrograder les rôles admin
- Forcer un reset de mot de passe (envoie un email)
- Vérifier manuellement un email

#### Endpoints d'authentification

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/auth/register` | POST | Inscription + envoi email vérification |
| `/auth/login` | POST | Connexion (retourne JWT) |
| `/auth/logout` | POST | Déconnexion |
| `/auth/verify-email/{token}` | GET | Vérification email |
| `/auth/resend-verification` | POST | Renvoyer email de vérification |
| `/auth/forgot-password` | POST | Demander reset password |
| `/auth/reset-password` | POST | Réinitialiser avec token |
| `/auth/me` | GET | Infos utilisateur connecté |

### 4) Génération de fiches de lecture Zotero

**NOUVEAU** : RAGpy peut maintenant générer automatiquement des fiches de lecture académiques et les ajouter comme notes enfants dans votre bibliothèque Zotero.

#### Configuration

1. **Obtenir une clé API Zotero** :
   - Rendez-vous sur https://www.zotero.org/settings/keys/new
   - Créez une nouvelle clé avec les permissions :
     - ✅ "Allow library access"
     - ✅ "Allow notes access"
   - Copiez la clé générée

2. **Configurer dans l'interface** :
   - Cliquez sur l'icône ⚙️ (Settings) en haut à droite
   - Section "Zotero (Optional - for automatic reading notes)"
   - Collez votre clé API
   - User ID et Group ID sont auto-détectés depuis votre export Zotero

#### Utilisation

Après avoir traité un export Zotero (étapes 1-3.3) :

1. Dans l'étape **"4. Choose Output Destination(s)"**, vous avez deux options :
   - **Option A** : Vector Database (flux classique RAG)
   - **Option B** : Zotero Reading Notes (nouveau)

2. Pour générer des fiches de lecture :
   - ☑️ Cochez "Zotero Reading Notes"
   - Sélectionnez le modèle LLM :
     - `gpt-4o-mini` (OpenAI - défaut, bon rapport qualité/prix)
     - `openai/gemini-2.0-flash-exp` (OpenRouter - très économique)
     - `anthropic/claude-3-5-haiku` (OpenRouter - excellent pour textes académiques)
     - `gpt-4o` (OpenAI - meilleure qualité)
   - Cliquez sur "Generate Zotero Notes"

3. Le système va :
   - ✅ Générer une fiche structurée pour chaque article (200-300 mots)
   - ✅ Vérifier si une fiche existe déjà (idempotence)
   - ✅ Créer une note enfant dans Zotero avec les tags `ragpy`, `fiche-lecture`
   - ✅ Afficher un résumé détaillé avec statut par article

#### Structure des fiches générées

Chaque fiche contient :
- **Référence bibliographique** : Titre, auteurs, date, DOI/URL
- **Problématique** : Question(s) de recherche ou objectif principal
- **Méthodologie** : Approche, données, méthodes utilisées
- **Résultats clés** : Principales conclusions ou découvertes
- **Limites et perspectives** : Points faibles, questions ouvertes

#### Fonctionnalités avancées

- **Idempotence** : Relancer la génération ne créera pas de doublons (détection via sentinel unique)
- **Multilingue** : Détection automatique de la langue depuis les métadonnées Zotero
- **Source complète** : Utilise le texte OCR complet + abstract pour une analyse approfondie
- **Parallélisation** : Vous pouvez générer les fiches ET insérer dans la base vectorielle simultanément

#### Exemples de résultats

```
Summary:
✅ Created: 8
ℹ️ Already exists: 2
⏭️ Skipped: 0
❌ Errors: 0

Details:
✅ Machine Learning for NLP (ABC123XY)
   Status: created
   Open in Zotero

ℹ️ Deep Learning Survey (DEF456UV)
   Status: exists
   Note already exists (idempotent)
```

#### Liens Zotero

Les notes créées sont directement accessibles via des liens `zotero://` cliquables dans l'interface, vous permettant d'ouvrir instantanément l'article correspondant dans Zotero Desktop.

#### Personnalisation du Prompt

**NOUVEAU** : Vous pouvez personnaliser le prompt de génération en éditant simplement un fichier texte !

Le fichier [app/utils/zotero_prompt.md](app/utils/zotero_prompt.md) contient le template utilisé pour générer les fiches. Vous pouvez :

- Modifier la structure des fiches (ajouter/supprimer des sections)
- Changer le ton (plus formel, plus technique, etc.)
- Ajuster la longueur (100 mots, 500 mots, etc.)
- Adapter pour votre domaine de recherche

**Placeholders disponibles** : `{TITLE}`, `{AUTHORS}`, `{DATE}`, `{DOI}`, `{URL}`, `{ABSTRACT}`, `{TEXT}`, `{LANGUAGE}`

Exemple de modification rapide :

```markdown
# Fiche minimaliste (100 mots)
Résume {TITLE} par {AUTHORS} en 100 mots en {LANGUAGE}.

Texte : {TEXT}
```

📚 **Guide complet** : Voir [app/utils/README_ZOTERO_PROMPT.md](app/utils/README_ZOTERO_PROMPT.md) pour des exemples détaillés et bonnes pratiques.

**Avantage** : Aucune modification de code nécessaire ! Le fichier est rechargé automatiquement à chaque génération.

### 5) Utilisation en ligne de commande

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

### 6) Le projet

Objectif: transformer des documents (PDFs, exports Zotero) en données exploitables pour des systèmes RAG, via un pipeline reproductible et une interface web simple à utiliser.

Grandes étapes:
- Extraction texte + métadonnées depuis Zotero/PDF (`rad_dataframe.py`)
- Découpage en chunks, nettoyage GPT, embeddings denses et sparses (`rad_chunk.py`)
- Insertion dans une base vectorielle (Pinecone, Weaviate, Qdrant) (`rad_vectordb.py` via l’UI)

### 7) Architecture de dev

Arborescence principale:
```
ragpy/
├── app/                      # Application web FastAPI
│   ├── main.py               # API + orchestration des scripts
│   ├── config.py             # Configuration centralisée
│   ├── core/                 # Modules core
│   │   ├── security.py          # JWT, hashing, tokens
│   │   └── credentials.py       # Gestion clés API utilisateur
│   ├── database/             # Base de données SQLite
│   │   └── session.py           # Session SQLAlchemy
│   ├── middleware/           # Middlewares
│   │   └── auth.py              # Authentification JWT
│   ├── models/               # Modèles SQLAlchemy
│   │   ├── user.py              # Utilisateurs + rôles
│   │   ├── audit.py             # Logs d'audit
│   │   └── project.py           # Projets
│   ├── routes/               # Routes API
│   │   ├── auth.py              # Inscription, login, reset password
│   │   ├── admin.py             # Gestion utilisateurs (admin)
│   │   └── users.py             # Profil utilisateur
│   ├── schemas/              # Schémas Pydantic
│   ├── services/             # Services
│   │   └── email_service.py     # Envoi emails (Resend)
│   ├── utils/                # Modules utilitaires
│   │   ├── zotero_client.py     # Client API Zotero v3
│   │   └── llm_note_generator.py # Générateur de fiches LLM
│   ├── static/               # Assets UI (CSS/JS/images)
│   └── templates/            # Templates Jinja2
│       └── emails/              # Templates emails
├── scripts/                  # Pipeline de traitement
│   ├── rad_dataframe.py      # JSON Zotero + PDFs -> CSV (OCR Mistral)
│   ├── rad_chunk.py          # Chunking + recodage GPT + embeddings
│   ├── rad_vectordb.py       # Insertion (Pinecone/Weaviate/Qdrant)
│   └── requirements.txt      # Dépendances
├── data/                     # Base de données SQLite (ragpy.db)
├── uploads/                  # Sessions de traitement
├── logs/                     # Logs applicatifs
├── .env                      # Variables d'environnement
└── ragpy_cli.sh              # Script démarrage serveur
```

Choix techniques clés:
- FastAPI + Uvicorn pour le backend API
- SQLAlchemy + SQLite pour la persistance utilisateurs
- JWT (python-jose) + bcrypt pour l'authentification
- Resend pour l'envoi d'emails transactionnels
- Mistral OCR (avec fallback OpenAI Vision) pour l'extraction PDF
- OpenAI API pour recodage GPT + embeddings (`text-embedding-3-large`)
- spaCy FR (`fr_core_news_md`) pour les embeddings sparse
- Pinecone, Weaviate (multi-tenants), Qdrant pour le stockage vectoriel

Journaux et sorties:
- `logs/app.log`, `logs/pdf_processing.log`
- Fichiers de session dans `uploads/<session>/`
- Base de données dans `data/ragpy.db`

### 8) Variables d'environnement (.env)

Toutes les variables supportées :

**Obligatoire :**

- `OPENAI_API_KEY` - Embeddings + recodage GPT par défaut

**OCR (recommandé) :**

- `MISTRAL_API_KEY` - OCR haute qualité Mistral
- `MISTRAL_OCR_MODEL` - Modèle OCR (défaut: `mistral-ocr-latest`)
- `MISTRAL_API_BASE_URL` - URL API (défaut: `https://api.mistral.ai`)

**Email / Authentification (recommandé) :**

- `RESEND_API_KEY` - Clé API Resend pour envoi emails
- `RESEND_FROM_EMAIL` - Email expéditeur (défaut: `onboarding@resend.dev`)
- `APP_URL` - URL de base pour liens emails (défaut: `http://localhost:8000`)

**OpenRouter (optionnel - économie coûts) :**

- `OPENROUTER_API_KEY` - Alternative économique pour recodage
- `OPENROUTER_DEFAULT_MODEL` - Modèle par défaut (ex: `google/gemini-2.5-flash`)

**Bases vectorielles (au moins une) :**

- `PINECONE_API_KEY`, `PINECONE_ENV`
- `WEAVIATE_URL`, `WEAVIATE_API_KEY`
- `QDRANT_URL`, `QDRANT_API_KEY`

**Zotero (optionnel) :**

- `ZOTERO_API_KEY` - Génération automatique de fiches de lecture
- `ZOTERO_USER_ID` - Auto-détecté depuis export Zotero
- `ZOTERO_GROUP_ID` - Pour bibliothèques de groupe

L'UI (« Settings ») permet de configurer ces variables via interface graphique.

### 9) Dépannage (FAQ)

**Installation :**

- Dépendances manquantes : `pip install -r scripts/requirements.txt`
- spaCy manquant : `python -m spacy download fr_core_news_md`
- Pas de clé API : vérifiez `.env` et la section « Settings » de l'UI

**Authentification :**

- Email de vérification non reçu : Vérifiez `RESEND_API_KEY` et `RESEND_FROM_EMAIL`
- Erreur 403 "Veuillez vérifier votre email" : Cliquez sur le lien dans l'email ou utilisez `/auth/resend-verification`
- Reset password : Le lien expire après 1 heure
- Sans Resend configuré : Les tokens s'affichent dans la console (mode dev)

**Bases vectorielles :**

- Pinecone : Créez l'index avec la dimension 3072 (text-embedding-3-large)
- Weaviate : Assurez-vous que la classe existe et que le tenant est correct
- Qdrant : La collection est créée automatiquement si absente

**Zotero :**

- Clé API invalide : Vérifiez les permissions ("library access" + "notes access")
- Notes non créées : Vérifiez que l'export ZIP contient bien un JSON Zotero valide
- Doublons : Le système vérifie automatiquement l'existence via sentinel unique
- Erreur 404 : L'itemKey n'existe pas dans votre bibliothèque
- Rate limit (429) : Géré automatiquement avec retry

**OCR :**

- Mistral OCR échoue : Vérifiez `MISTRAL_API_KEY`, fallback automatique vers OpenAI Vision
- Texte mal extrait : Essayez d'augmenter `OPENAI_OCR_MAX_PAGES` pour le fallback

### 10) Licence

MIT. Voir `LICENSE`.

