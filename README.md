# RAGpy

**Solution de veille scientifique intelligente pour chercheurs et doctorants**

RAGpy est un outil avancé de veille scientifique qui optimise et automatise le traitement de votre corpus bibliographique. Conçu pour s'intégrer parfaitement à votre flux de travail académique, il agit à plusieurs niveaux :

- **Optimisation de Publish or Perish** : Import et traitement raffiné des résultats de vos recherches bibliométriques, avec filtrage de pertinence par LLM avant l'import dans Zotero.
- **Enrichissement Zotero** : Mise à jour dynamique de votre bibliothèque avec génération automatique de notes de lecture enrichies (articles et livres) et classement structuré des items par clusters thématiques.
- **Gestion de Mémoires RAG** : Création et gestion de bases de connaissances vectorielles (Retrieval-Augmented Generation) permettant à vos LLMs d'exploiter la totalité de votre savoir scientifique avec précision.

RAGpy transforme une collection de documents statique en une base de connaissances vivante et exploitable.

---

## Qu'est-ce que RAGpy peut faire pour vous ?

```
📚 Vos documents                    🎯 Ce que RAGpy produit
─────────────────                   ─────────────────────────

  Zotero + PDF/EPUB ─┐              ┌─► Base vectorielle (RAG)
                     │              │   → Recherche sémantique
  CSV / Excel      ──┼──► RAGpy ───┼─► Fiches de lecture Zotero
                     │              │   → Notes automatiques
  Publish or Perish ─┘              └─► Clusters thématiques
                                       → Tags automatiques Zotero
```


### Cas d'usage typiques

| Besoin | Solution RAGpy |
|--------|----------------|
| 🔍 **Recherche sémantique** | Injecter vos articles dans Pinecone, Weaviate, Qdrant ou une collection Albert pour un RAG personnalisé |
| 📝 **Fiches de lecture automatiques** | Générer des fiches (article, livre chapitre par chapitre, fiche pédagogique, grille d'évaluation) et les pousser dans Zotero |
| 🏷️ **Organisation automatique** | Clusteriser vos documents par thème et créer des tags Zotero |
| 📥 **Veille bibliographique** | Importer un export Publish or Perish, filtrer les citations pertinentes par LLM, les créer dans Zotero |
| 🇫🇷 **Traitement souverain** | Confier OCR, recodage, embeddings et stockage à l'API Albert (DINUM), hébergée en France |

---

## Nouveautés

- **Intégration Albert (DINUM), optionnelle** : OCR, recodage, notes, embeddings `bge-m3` et collections vectorielles privées servis par l'API publique française ; désactivée par défaut, activable capacité par capacité ([guide](.claude/docs/albert.md)).
- **Sécurité renforcée (audit externe A01–A14)** : imports confinés et limités en taille, droits par session (propriétaire, collaborateur, lecteur), admission des traitements (409 si la session est occupée, 429 au-delà des quotas), démarrage refusé en production sans secret JWT, Redis et Flower publiés en local seulement.
- **OCR résilient** : réessais Mistral avec backoff, gros PDF compressés puis découpés automatiquement, OCR local hors ligne (Docling) pour les scans, et jamais de « succès » silencieux sur un texte tronqué (drapeau `texteocr_partial`).
- **Fiches de lecture de livres** en plusieurs phases (détection des chapitres, analyse chapitre par chapitre, synthèse), lecture directe des **EPUB**.
- **Déduplication des chunks et cache de recodage** (optionnels) : un même texte réinséré n'est plus dupliqué dans la base vectorielle.
- **Projets collaboratifs** avec membres et rôles, reprise d'une session à n'importe quelle étape.
- **Import Publish or Perish → Zotero** avec filtrage de pertinence par LLM et prévisualisation.
- **Clustering automatique (UMAP + HDBSCAN)** et tags Zotero.
- **Mode Celery** (Redis, worker, Beat, Flower) pour la production, à côté du mode sous-processus avec progression en temps réel (SSE).

---

## Sommaire

- [A — Installation et configuration](#a--installation-et-configuration)
  - [1) Installation Docker (recommandée)](#1-installation-docker-recommandée)
  - [2) Installation manuelle](#2-installation-manuelle)
  - [3) Configuration (.env)](#3-configuration-env)
    - [Serveurs d'API et modèles](#serveurs-dapi-et-modèles)
  - [4) Premier démarrage](#4-premier-démarrage)
- [B — Utilisation](#b--utilisation)
  - [5) Le pipeline](#5-le-pipeline)
  - [6) Interface web](#6-interface-web)
  - [7) Clustering automatique et tags Zotero](#7-clustering-automatique-et-tags-zotero)
  - [8) Fiches de lecture Zotero](#8-fiches-de-lecture-zotero)
  - [9) Import Publish or Perish](#9-import-publish-or-perish)
  - [10) Comptes, projets et droits](#10-comptes-projets-et-droits)
  - [11) Ligne de commande](#11-ligne-de-commande)
  - [12) Option souveraine : API Albert (DINUM)](#12-option-souveraine--api-albert-dinum)
- [C — Exploitation](#c--exploitation)
  - [13) Mode Celery](#13-mode-celery)
  - [14) Déploiement en production](#14-déploiement-en-production)
  - [15) Tests](#15-tests)
- [D — Projet](#d--projet)
  - [16) Architecture technique](#16-architecture-technique)
  - [17) Dépannage (FAQ)](#17-dépannage-faq)
  - [18) Licence](#18-licence)

---

## A — Installation et configuration

### 1) Installation Docker (recommandée)

**Prérequis** : Docker et Docker Compose ([Get Docker](https://docs.docker.com/get-docker/)).

```bash
# 1. Cloner le dépôt
git clone <URL_DU_DEPOT> && cd ragpy

# 2. Créer le fichier .env puis le compléter (voir section 3)
cp .env.example .env

# 3. Lancer toute la pile (application, Redis, worker Celery, Beat, Flower)
docker compose up -d

# 4. Ouvrir l'interface
open http://localhost:8000
```

Pour l'application seule (mode sous-processus, sans Celery) : `docker compose up -d ragpy`.

**Services lancés par `docker compose up -d`** :

| Service | Rôle | Accès |
|---------|------|-------|
| `ragpy` | Application FastAPI (Uvicorn, 1 worker) | `http://localhost:8000` |
| `redis` | Broker Celery, limiteur Albert partagé | `127.0.0.1:6379` (`REDIS_BIND`) |
| `celery_worker` | Exécution des tâches du pipeline | — |
| `celery_beat` | Tâches périodiques | — |
| `flower` | Supervision Celery | `127.0.0.1:5555` (`FLOWER_BIND`) |

`flower` refuse de démarrer tant que `FLOWER_USER` et `FLOWER_PASSWORD` ne sont pas définis dans `.env` (ni `admin/admin`, ni `admin/changeme`) ; les autres services fonctionnent sans lui.

**Commandes utiles** :
```bash
docker compose logs -f ragpy          # Journaux en temps réel
docker compose down                   # Arrêter
docker compose up -d --build          # Reconstruire après une mise à jour du code
docker compose exec ragpy bash        # Shell dans le conteneur
```

- **Le code est intégré à l'image** (seuls `data/`, `uploads/`, `logs/` et `sources/` sont montés) : toute mise à jour du code exige `docker compose up -d --build`.
- **Une modification du `.env`** est prise en compte par `docker compose up -d` (recréation des conteneurs), sans reconstruction.
- **OCR local (Docling)** : mettre `INSTALL_LOCAL_OCR=true` dans `.env`, puis `docker compose up -d --build` (image nettement plus lourde : torch CPU, Tesseract FR/EN).

**Volumes persistants** :
- `./data` : base SQLite (`ragpy.db`), verrous de traitement, cache de recodage
- `./uploads` : sessions de traitement
- `./logs` : journaux applicatifs
- `./sources` : fichiers sources (optionnel)

---

### 2) Installation manuelle

**Prérequis** : Python 3.10 ou 3.11 (l'image Docker et la CI utilisent 3.11), pip, git.

```bash
# 1. Cloner et créer l'environnement virtuel
git clone <URL_DU_DEPOT> && cd ragpy
python3 -m venv .venv && source .venv/bin/activate

# 2. Installer les dépendances
python -m pip install --upgrade pip && python -m pip install -r scripts/requirements.txt && python -m spacy download fr_core_news_md

# 3. Configurer puis lancer
cp .env.example .env
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

L'application lance les scripts du pipeline avec la commande `python3` : le venv doit être actif (ou `.venv/bin` en tête du `PATH`), sinon les étapes échouent faute de dépendances. Sans venv activé : `PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`.

Depuis le dossier parent du dépôt, `./ragpy/ragpy_cli.sh start | close | kill` démarre le serveur en arrière-plan, l'arrête ou le tue avec les scripts résiduels.

**OCR local (optionnel)** : installer Docling dans un **venv séparé** (il exige numpy 2 et httpx 0.28, incompatibles avec spaCy et les clients vectoriels du venv principal) avec `scripts/requirements-ocr-local.txt`, puis indiquer son interpréteur dans `LOCAL_OCR_PYTHON`.

---

### 3) Configuration (.env)

Toute la configuration passe par **un seul fichier `.env` à la racine du dépôt**, créé à partir de [`.env.example`](.env.example). Ce modèle liste **toutes les variables lues par le code**, rangées en blocs numérotés, avec leur défaut et leur rôle en commentaire. Il est lu par l'application web, par les scripts du pipeline et par Docker Compose (fichier d'environnement des quatre services et interpolation des `${…}` de `docker-compose.yml`).

**Page « Paramètres » (Administration > Paramètres).** Un administrateur y déclare toutes les variables du `.env`, dans son ordre et par blocs : listes des serveurs déclarés pour les champs « serveur », suggestions de modèles tirées du serveur choisi, bouton « Tester » par serveur, clés en écriture seule (jamais affichées), validation de tout le formulaire avant écriture. L'écriture se fait sur place dans le `.env` (commentaires et ordre conservés, sauvegarde dans `data/config_backups/`, journal d'audit sans valeur). La plupart des variables s'appliquent au traitement suivant (« à chaud ») ; la page signale celles qui demandent un redémarrage et verrouille une vingtaine de variables sensibles (secret JWT, base de données, CORS, broker…), modifiables dans le fichier seulement. Les utilisateurs non administrateurs déclarent leurs propres clés dans **Mes identifiants**.

**Choix personnels (Mon profil > Mes choix de modèles).** Quand le `.env` déclare `LLM_DEFAULT_SERVER`, chaque utilisateur peut choisir, pour le recodage, les notes, les fiches de lecture, le filtre des citations, les réponses sourcées et l'OCR, un serveur (parmi ceux que l'administrateur a déclarés au bloc 1) et un modèle ; un champ vide suit la configuration du serveur, affichée en grisé. Ordre de priorité : champ de l'étape, choix personnel, variable du service, défaut. Les clés des serveurs Anthropic, Google, DeepSeek, Qwen et GLM deviennent des clés personnelles dès que leur adresse est déclarée ; le bouton « Vider » efface une clé enregistrée. Les adresses personnelles (Mistral, Weaviate, Qdrant) d'un non-administrateur doivent être en https vers un hôte public (ni réseau interne, ni nom court de conteneur).

**Embeddings, transcription.** Même règle serveur + modèle : `EMBEDDING_SERVER` + `EMBEDDING_MODEL` (OpenAI `text-embedding-3-large` en 3072 d, Albert `bge-m3` en 1024 d, ou un serveur compatible déclaré au bloc 1 comme Mistral `mistral-embed`, dont la dimension est mesurée au premier appel) et `AUDIO_SERVER` + `AUDIO_MODEL` (Albert seulement). Un modèle déclaré active la fonction ; un couple vide garde la règle historique (`EMBEDDING_PROVIDER`, `ALBERT_AUDIO_ENABLED`). Un index vectoriel n'accepte qu'un espace (modèle et dimension) : changer de modèle d'embeddings exige un autre index.

`.env.example` est **généré** à partir du registre unique des variables, [`scripts/rad_settings/registry.py`](scripts/rad_settings/registry.py) : on modifie le registre, puis on régénère, jamais le fichier à la main. L'outil `scripts/env_tool.py` range aussi votre `.env` dans le même ordre et le contrôle, sans jamais afficher une valeur :

```bash
.venv/bin/python scripts/env_tool.py check                 # bilan : variables inconnues, en double, invalides, obsolètes
.venv/bin/python scripts/env_tool.py tidy                  # aperçu du rangement (noms seulement)
.venv/bin/python scripts/env_tool.py tidy --apply          # range le .env (sauvegarde dans data/config_backups/, valeurs vérifiées identiques)
.venv/bin/python scripts/env_tool.py example               # régénère .env.example depuis le registre
```

#### Configurations minimales

**Poste local (essai)** : une seule clé suffit pour importer un CSV, découper, vectoriser et générer des fiches.
```env
OPENAI_API_KEY=sk-proj-...
```

**PDF, OCR et bases vectorielles** : ajouter selon les besoins.
```env
MISTRAL_API_KEY=...                  # OCR des PDF (premier maillon de la chaîne)
OPENROUTER_API_KEY=sk-or-v1-...      # recodage économique (modèles provider/model)
PINECONE_API_KEY=pcsk_...            # ou WEAVIATE_URL + WEAVIATE_API_KEY, ou QDRANT_URL + QDRANT_API_KEY
ZOTERO_API_KEY=...                   # fiches, tags et import de citations
ZOTERO_USER_ID=1234567               # ou ZOTERO_GROUP_ID pour une bibliothèque de groupe
```

**Serveur de production** : en plus des clés, obligatoire.
```env
RAGPY_ENV=production                 # refuse de démarrer sans vrai secret JWT
JWT_SECRET_KEY=<secret aléatoire>    # python -c "import secrets; print(secrets.token_urlsafe(48))"
APP_URL=https://ragpy.example.org    # liens des e-mails
CORS_ORIGINS=https://ragpy.example.org
RESEND_API_KEY=re_...                # vérification des adresses e-mail
RESEND_FROM_EMAIL=noreply@example.org
USERS_SANDBOX=TRUE                   # nouvelles inscriptions en attente d'approbation
FLOWER_USER=<utilisateur>            # obligatoire pour le conteneur flower
FLOWER_PASSWORD=<mot de passe fort>
```

#### Contenu de `.env.example`, bloc par bloc

| Bloc | Contenu | Remarques |
|------|---------|-----------|
| 1. Serveurs d'API et services externes | un sous-bloc par fournisseur : adresse (`*_API_BASE_URL`) et clé (`*_API_KEY`) d'OpenRouter, Mistral, OpenAI, Albert, Anthropic, Google, DeepSeek, Qwen, GLM, serveur local ; puis Pinecone, Weaviate, Qdrant, Zotero, Resend | OpenRouter et Mistral préremplis, les autres en commentaire ; adresses vérifiées ci-dessous |
| 2. Serveurs et modèles par service | aujourd'hui `OPENROUTER_DEFAULT_MODEL`, `RECODE_MODEL`, `MISTRAL_OCR_MODEL`, `OCR_ENABLE_ALBERT`, `EMBEDDING_PROVIDER`, `ALBERT_ENABLED`, `ALBERT_DATA_POLICY`… | deviendra un couple serveur + modèle par service (voir ci-dessous) |
| 3. OCR : réglages par moteur | Mistral (`MISTRAL_*` : découpage, réessais, échelle 429), Albert (`ALBERT_OCR_*`), local Docling (`LOCAL_OCR_*`), `OCR_MIN_CHARS_PER_PAGE` | OpenAI Vision désactivé (tronque les livres) |
| 4. Découpage, recodage, embeddings | `DEFAULT_*_WORKERS`, `DEFAULT_BATCH_SIZE_GPT`, `RECODE_*` (durcissement, caches), `EMBED_MAX_MISSING_RATIO` | caches et durcissement désactivés par défaut |
| 5. Notes, fiches, citations, audio | `MAX_CONCURRENT_LLM_CALLS`, `BOOK_NOTE_VERSION`, `MAX_CONCURRENT_WEB_FETCHES`, `ALBERT_AUDIO_*` | `MAX_CONCURRENT_LLM_CALLS` est global à la plateforme |
| 6. Déduplication et bases vectorielles | `DEDUP_*`, `*_BATCH_SIZE`, `ALBERT_METADATA_FIELDS` | déduplication désactivée par défaut |
| 7. Albert : quotas, concurrence, délais | `ALBERT_*_RPM`, limiteur, `ALBERT_*_CONCURRENCY`, réessais, délais, journal d'usage | voir [section 12](#12-option-souveraine--api-albert-dinum) |
| 8. Traitements | `MAX_ACTIVE_JOBS[_PER_USER]`, `UPLOAD_MAX_*`, `SESSION_TTL_HOURS`, `CLEANUP_*` | 429 ou 409 au-delà ; aligner nginx |
| 9. Comptes, sécurité, déploiement | `RAGPY_ENV`, `JWT_*`, `CORS_ORIGINS`, `APP_URL`, `DATABASE_URL`, `DEBUG`, `USERS_SANDBOX`, `*_EXPIRE_HOURS` | voir les règles ci-dessous |
| 10. Serveur, Celery, Redis, Flower, image | `UVICORN_*`, `ENABLE_CELERY`, `CELERY_*`, `REDIS_BIND`, `FLOWER_*`, `INSTALL_*` | voir [section 13](#13-mode-celery) |
| 11. Supervision | `ENABLE_METRICS`, `PROMETHEUS_PUSHGATEWAY_URL`, `METRICS_JOB_NAME` | `/metrics` pour Prometheus |

#### Serveurs d'API et modèles

Adresses, documentation et identifiants **vérifiés le 2026-10-03** sur les sources officielles et les catalogues en ligne (`GET /models`). Les identifiants changent vite : revérifier avant de les recopier.

**Règle de choix du modèle (sprint « configuration unifiée »).** On déclare un serveur par défaut (`LLM_DEFAULT_SERVER`, en général OpenRouter) et un modèle par défaut (`LLM_DEFAULT_MODEL`). Chaque service (recodage, notes, fiches, citations, réponses, OCR, embeddings…) peut déclarer son propre couple `…_SERVER` + `…_MODEL` :

- un service qui ne donne qu'un modèle l'envoie au serveur par défaut ;
- un serveur déclaré pour le service prime sur le serveur par défaut ;
- le serveur est l'**adresse de l'API**, qui doit être déclarée au bloc 1 avec sa clé ;
- le nom du modèle est envoyé **tel quel**, dans la nomenclature du serveur (`openai/gpt-4o-mini` chez OpenRouter, `gpt-4o-mini` chez OpenAI).

```env
LLM_DEFAULT_SERVER=https://openrouter.ai/api/v1
LLM_DEFAULT_MODEL=google/gemini-3.8-flash
LLM_RECODE_SERVER=                         # vide = serveur par défaut
LLM_RECODE_MODEL=openai/gpt-4o-mini        # envoyé tel quel au serveur retenu
OCR_SERVER=https://api.mistral.ai/v1       # contrôlé avant le premier document
OCR_MODEL=mistral-ocr-latest
OCR_SERVER_FALLBACK=                       # vides = aucun repli ; remplis = repli vers ce serveur
OCR_MODEL_FALLBACK=                        # (un autre serveur que celui de OCR_SERVER)
```

Repli de l'OCR : vides, un échec du moteur fait échouer le document (il sera refait à la relance). Remplis (par exemple `OCR_SERVER_FALLBACK=https://albert.api.etalab.gouv.fr/v1` et `OCR_MODEL_FALLBACK=lightonocr-2-1b`, ou `local` et `pymupdf`), un document dont l'OCR principal échoue est repris par le repli ; après une erreur de compte du moteur principal (clé, plafond, quota), la suite du lot passe par le repli. Le repli est contrôlé avant le premier document comme le moteur principal, sa clé est exigée, chaque repli est tracé dans `*_errors.json` (`OCR_PROVIDER_FALLBACK`) et les colonnes `texteocr_server` / `texteocr_model` nomment le moteur qui a servi. Il remplace `OCR_ENABLE_OPENAI_FALLBACK` et `OPENAI_OCR_MODEL` (OpenAI Vision ne fait plus d'OCR).

Ces couples valent pour le recodage, les notes, les fiches de lecture et les citations, dès que `LLM_DEFAULT_SERVER` est déclaré. Tant qu'il est vide, l'ancien comportement s'applique (« mode historique » : `OPENROUTER_DEFAULT_MODEL`, recodage `gpt-4o-mini`). Pour passer un `.env` existant à la nouvelle règle, en conservant l'acheminement d'aujourd'hui :

```bash
.venv/bin/python scripts/env_tool.py migrate               # aperçu des couples ajoutés (adresses et modèles, jamais de clé)
.venv/bin/python scripts/env_tool.py migrate --apply       # écriture (sauvegarde dans data/config_backups/)
```

Les étapes du web (recodage, notes) et `rad_chunk.py --server … --model …` acceptent un serveur et un modèle pour une seule exécution. Les clés d'Anthropic, Google, DeepSeek, Qwen, GLM et du serveur local sont pour l'instant réservées aux administrateurs (lues dans le `.env`) ; elles deviendront des identifiants personnels avec la page des paramètres. **OCR (sans repli).** `OCR_SERVER` + `OCR_MODEL` désignent un seul moteur : l'adresse de Mistral (`mistral-ocr-latest`), celle d'Albert (`lightonocr-2-1b` page par page, ou un modèle `/v1/ocr`), celle du serveur local (page par page), ou le mot réservé `local` avec `docling` ou `pymupdf` (couche texte native des PDF nés numériques, sans Tesseract). Le moteur est contrôlé avant le premier document : inaccessible, rien ne démarre. Un document en échec n'est ni écrit ni marqué traité (la relance le refait) ; une erreur de compte (clé refusée, plafond de dépense Mistral, quota Albert) arrête le lot. `OCR_MODEL` vide : ancienne chaîne (Albert si `OCR_ENABLE_ALBERT=1`, Mistral, Docling, PyMuPDF). Les embeddings, le rerank et l'audio suivront la même règle (lot 6).

| Serveur | Adresse de l'API | Documentation | Remarques |
|---|---|---|---|
| OpenRouter | `https://openrouter.ai/api/v1` | [openrouter.ai/docs](https://openrouter.ai/docs/api/reference/overview) | compatible OpenAI ; catalogue public `GET /models` ; embeddings `POST /embeddings` (liste : `GET /embeddings/models`) |
| Mistral AI | `https://api.mistral.ai/v1` | [docs.mistral.ai/api](https://docs.mistral.ai/api/) · [OCR](https://docs.mistral.ai/api/endpoint/ocr) | compatible OpenAI ; OCR par `/files` + `/ocr`. Jusqu'au lot 4, écrire `MISTRAL_API_BASE_URL=https://api.mistral.ai` (le code ajoute `/v1`) |
| OpenAI | `https://api.openai.com/v1` | [developers.openai.com](https://developers.openai.com/api/reference/overview) | API d'origine |
| Albert (DINUM) | `https://albert.api.etalab.gouv.fr/v1` | [guide](https://guides.ia.numerique.gouv.fr/albert-api) · [référence](https://albert.api.etalab.gouv.fr/reference) | compatible OpenAI ; OCR, embeddings, rerank, recherche, transcription ; réservé aux agents publics éligibles |
| Anthropic | `https://api.anthropic.com/v1` | [compatibilité OpenAI](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk) | couche de compatibilité, selon Anthropic pas une solution de production à long terme |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` | [compatibilité OpenAI](https://ai.google.dev/gemini-api/docs/openai) | chat et embeddings |
| DeepSeek | `https://api.deepseek.com` | [api-docs.deepseek.com](https://api-docs.deepseek.com/) | compatible OpenAI, sans `/v1` |
| Qwen (Alibaba Cloud) | `https://<WorkspaceId>.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1` | [Model Studio](https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope) | adresse propre à chaque espace de travail ; États-Unis : `https://dashscope-us.aliyuncs.com/compatible-mode/v1` |
| GLM (Z.ai) | `https://api.z.ai/api/paas/v4` | [docs.z.ai](https://docs.z.ai/api-reference/introduction) | Chine : `https://open.bigmodel.cn/api/paas/v4` |
| Ollama (local) | `http://localhost:11434/v1` | [compatibilité OpenAI](https://docs.ollama.com/api/openai-compatibility) | clé exigée mais ignorée |
| vLLM (local) | `http://localhost:8000/v1` | [serveur en ligne](https://docs.vllm.ai/en/latest/serving/online_serving/) | clé facultative (`--api-key`) |
| LM Studio (local) | `http://localhost:1234/v1` | [compatibilité OpenAI](https://lmstudio.ai/docs/developer/openai-compat) | |

**Modèles de langage réputés** (prix OpenRouter en dollars par million de jetons, entrée / sortie) :

| Modèle | Identifiant chez OpenRouter | Identifiant chez son fournisseur | Contexte | Prix |
|---|---|---|---|---|
| Google Gemini 3.8 Flash | `google/gemini-3.8-flash` | `gemini-3.8-flash` (Google) | 1 048 576 | 0,75 / 3,75 |
| Anthropic Claude Sonnet 5.5 | `anthropic/claude-sonnet-5.5` | `claude-sonnet-5-5` (Anthropic) | 1 000 000 | 2 / 10 |
| Z.ai GLM 5.3 Flash | `z-ai/glm-5.3-flash` | `glm-5.3-flash` (Z.ai) | 1 048 576 | 0,15 / 0,50 |
| OpenAI GPT-5.6 Luna Pro | `openai/gpt-5.6-luna-pro` | `gpt-5.6-luna` avec le mode de raisonnement `pro` (API Responses seulement) | 1 050 000 | 0,20 / 1,20 |
| DeepSeek V4.1 Flash | `deepseek/deepseek-v4.1-flash` | `deepseek-flash` (DeepSeek) | 1 048 576 | 0,30 / 1,20 |
| Qwen3.8 Flash | `qwen/qwen3.8-flash` | `qwen3.8-flash` (Alibaba Cloud) | 1 000 000 | 0,15 / 0,47 |
| Mistral Small 4 | `mistralai/mistral-small-2603` | `mistral-small-2603`, alias `mistral-small-latest` (Mistral) | 262 144 | 0,15 / 0,60 |

Un même modèle n'a pas le même nom d'un serveur à l'autre (`claude-sonnet-5.5` chez OpenRouter, `claude-sonnet-5-5` chez Anthropic). Chez OpenAI, la génération actuelle est GPT-6 (`gpt-6-luna`, `gpt-6.1-sol`…).

**OCR et embeddings :**

| Usage | Serveur | Modèle | Remarques |
|---|---|---|---|
| OCR | Mistral | `mistral-ocr-latest` (OCR 4.x), `mistral-ocr-2512` (OCR 3) | défaut actuel de RAGpy |
| OCR | Albert | `lightonocr-2-1b` | page par page ; `/v1/ocr` (`mistral-ocr-2512`) non ouvert sur le compte actuel |
| Embeddings | OpenAI ou OpenRouter | `text-embedding-3-large` (OpenAI) ou `openai/text-embedding-3-large` (OpenRouter) | 3072 dimensions, espace actuel des index |
| Embeddings | Mistral | `mistral-embed` | 1024 dimensions |
| Embeddings | Albert | `bge-m3` | 1024 dimensions, espace séparé des 3072 d |

Ne jamais mélanger deux espaces d'embeddings dans un même index : changer de modèle d'embeddings impose de revectoriser.

#### Règles à connaître

- **Clés API et rôles.** Les clés du `.env` servent de repli aux **administrateurs** seulement. Un utilisateur non-admin n'y a jamais accès : il saisit ses propres clés dans **Paramètres > Mes Identifiants** (chiffrées en base), faute de quoi l'étape concernée répond 403 avec le champ `credential_required`.
- **`JWT_SECRET_KEY` se définit une fois.** Il signe les sessions et dérive la clé de chiffrement des identifiants enregistrés : le changer sans procédure rend illisibles toutes les clés des utilisateurs. Rotation : nouveau secret dans `JWT_SECRET_KEY`, ancien dans `JWT_SECRET_KEY_PREVIOUS`, redémarrer, `python scripts/rotate_credentials_key.py --dry-run` puis `python scripts/rotate_credentials_key.py`, vider `JWT_SECRET_KEY_PREVIOUS` et redémarrer.
- **Ligne vide ≠ défaut.** Supprimer ou commenter une ligne redonne le défaut du code ; une ligne vide ne le fait que si le commentaire de `.env.example` le précise. En particulier, ne jamais laisser vide `DATABASE_URL`, `JWT_ALGORITHM` ni une valeur numérique.
- **Sous Docker Compose** :
  - `CELERY_BROKER_URL` et `CELERY_RESULT_BACKEND` sont forcés à `redis://redis:6379/0` (réseau interne), quelle que soit leur valeur dans `.env` ;
  - écrire `ALBERT_LIMITER_BACKEND=redis` si Albert est activé : la valeur du `.env` remplace le défaut `redis` de `docker-compose.yml` ;
  - garder `UVICORN_WORKERS=1` : le bouton d'arrêt, les flux SSE et le nettoyage planifié vivent dans le processus ;
  - `REDIS_BIND` et `FLOWER_BIND` restent à `127.0.0.1` sauf pare-feu ;
  - le `.env` est **monté** dans les conteneurs `ragpy`, `celery_worker` et `celery_beat` (`./.env:/app/.env`) au lieu d'être passé en `env_file` : il doit exister sur l'hôte avant `docker compose up`, et une modification (fichier ou page Paramètres) est reprise à chaud, sans recréer les conteneurs, sauf pour les variables de déploiement.
- **Scripts en ligne de commande** : `rad_dataframe.py` et `rad_chunk.py` lisent le `.env` racine ; `rad_vectordb.py` ne lit aucun `.env` (exporter les variables, voir [section 11](#11-ligne-de-commande)). Ne jamais créer de `scripts/.env` : il remplacerait entièrement le `.env` racine pour ces scripts (l'ancien gabarit `scripts/.env.example` a été supprimé).
- **Interrupteur des tests live Albert** : il se pose dans le shell seulement, jamais dans `.env` (voir [section 15](#15-tests)).

---

### 4) Premier démarrage

1. Ouvrir `http://localhost:8000`, puis **S'inscrire**.
2. **Le premier compte créé devient administrateur** (jamais bloqué, même avec `USERS_SANDBOX=TRUE`).
3. Avec `RESEND_API_KEY`, un e-mail de vérification est envoyé (lien valable `EMAIL_VERIFICATION_EXPIRE_HOURS`, 24 h par défaut).
4. Avec `USERS_SANDBOX=TRUE`, les inscriptions suivantes attendent l'approbation d'un administrateur (**Admin > Utilisateurs**).
5. Chaque utilisateur non-admin configure ses clés dans **Paramètres > Mes Identifiants**.

---

## B — Utilisation

### 5) Le pipeline

RAGpy fonctionne comme une chaîne de traitement où chaque étape prépare les données pour la suivante. Chaque étape écrit ses fichiers dans le dossier de la session : un traitement interrompu reprend à l'étape suivante, et chaque étape accepte aussi un fichier téléversé (« Upload »).

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
│  • Zotero+PDF/EPUB • Segments ~1000     • Dense : OpenAI 3072 d         │
│  • CSV direct        tokens               ou bge-m3 1024 d (Albert)     │
│  • OCR en chaîne   • Recodage LLM       • Sparse : spaCy (TF)           │
│                      si OCR brut                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ÉTAPE 4a              ÉTAPE 4b              ÉTAPE 5                    │
│  ─────────             ─────────             ────────                   │
│                                                                          │
│  📊 Vector DB    ou    🏷️ Clustering    ou   📝 Fiches Zotero          │
│  (RAG)                 (organisation)        (résumés)                  │
│                                                                          │
│  • Pinecone            • UMAP + HDBSCAN      • 5 modes, dont livres     │
│  • Weaviate            • Tags auto Zotero    • Notes enfants            │
│  • Qdrant              • 8-15 clusters       • Structure académique     │
│  • Albert (collections)                                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Que se passe-t-il à chaque étape ?

| Étape | Entrée | Traitement | Sortie |
|-------|--------|------------|--------|
| **1. Import** | ZIP (export Zotero + PDF/EPUB) ou CSV | OCR en chaîne, extraction des métadonnées | `output.csv` |
| **2. Chunking** | `output.csv` | Découpage ~1000 tokens (chevauchement 150) + recodage LLM, sauté si le texte vient de Mistral, Docling ou d'un CSV | `output_chunks.json` |
| **3a. Dense** | `output_chunks.json` | Embeddings OpenAI `text-embedding-3-large` (3072 d) ou Albert `bge-m3` (1024 d) | `output_chunks_with_embeddings.json` |
| **3b. Sparse** | `..._with_embeddings.json` | Traits spaCy (lemmes, fréquences, indices stables) | `output_chunks_with_embeddings_sparse.json` |
| **4a. Vector DB** | `..._sparse.json` | Insertion Pinecone / Weaviate / Qdrant / collection Albert | Index RAG |
| **4b. Clustering** | `..._sparse.json` (ou `..._with_embeddings.json`) | UMAP → HDBSCAN → tags | Tags Zotero |
| **5. Fiches** | `output.csv` | Génération LLM | Notes Zotero |

#### Chaîne OCR

```
[Albert, si OCR_ENABLE_ALBERT=1] → Mistral OCR → [OpenAI Vision, désactivé] → OCR local Docling → PyMuPDF (legacy)
```

- Chaque maillon qui échoue passe la main au suivant ; le fournisseur retenu est noté dans la colonne `texteocr_provider`.
- **Mistral** : réessais des erreurs transitoires, gros PDF compressés puis découpés (au-delà de 45 Mo ou 950 pages), sans perdre les parts réussies en cas d'échec partiel. Un 401 peut signifier le plafond de dépense mensuel atteint.
- **OCR local Docling** : lit les PDF scannés sans clé ni réseau quand Mistral est indisponible (s'il est installé).
- **Aucun succès silencieux** : un texte manifestement incomplet est marqué `texteocr_partial` et signalé `OCR_PARTIAL` dans `*_errors.json`.
- Les EPUB sont lus directement, sans OCR.

---

### 6) Interface web

Ouvrez `http://localhost:8000` et connectez-vous. Pages principales : **Mes projets** (`/my-projects`), **Pipeline** (`/pipeline`, lié ou non à un projet), **Profil et identifiants** (`/profile`), **Administration** (`/admin`, administrateurs).

#### Deux options d'ingestion

**Option A : ZIP (Zotero + PDF/EPUB)** — flux complet avec OCR
1. Exporter la collection Zotero (JSON + dossier `files/`) et la zipper.
2. Téléverser le ZIP, puis lancer **Extract Text & Metadata** (OCR).
3. Enchaîner les étapes 3.1 (chunking), 3.2 (dense) et 3.3 (sparse).

**Option B : CSV (direct)** — sans OCR ni recodage, bien moins cher
1. Téléverser un CSV. La colonne de texte est détectée parmi `texteocr`, `text`, `content`, `body`, `description`, `texte`, `contenu` ; toutes les autres colonnes deviennent des métadonnées transmises jusqu'à la base vectorielle.
2. Passer directement au chunking.

Chaque étape 3.x propose **Upload** (réinjecter `output.csv`, `output_chunks.json` ou `output_chunks_with_embeddings.json`) ou **Generate** (réutiliser le résultat précédent). Le bouton d'arrêt stoppe le script en cours ; un traitement lancé continue même si l'onglet est fermé ou rechargé.

#### Où sont stockés les fichiers ?

```
uploads/<session>/
├── output.csv                                   # Étape 1
├── output_errors.json                           # Erreurs et OCR partiels de l'étape 1
├── output_chunks.json                           # Étape 2
├── output_chunks_with_embeddings.json           # Étape 3a
├── output_chunks_with_embeddings_sparse.json    # Étape 3b
├── clustering_results.json                      # Étape 4b
├── dedup_journal.jsonl                          # Refus de la déduplication (si activée)
└── albert_usage.jsonl                           # Journal d'usage Albert (si appelé)
```

Les sessions expirent après `SESSION_TTL_HOURS` (24 h par défaut) et leur dossier est nettoyé automatiquement.

---

### 7) Clustering automatique et tags Zotero

RAGpy peut regrouper vos documents par similarité thématique et créer des tags Zotero.

#### Comment ça marche ?

```
   Documents avec embeddings (3072 d ou 1024 d)
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

1. Complétez les étapes 1 à 3 (embeddings denses au minimum).
2. Dans **« 4.b Cluster Documents to Zotero »** :
   - **Session Name** : nom pour les tags (ex. `MaBiblio` → `_MaBiblio_01`)
   - **Min Cluster Size** : taille minimum par cluster (optionnel) ; plus petit = plus de clusters. Recommandé : 5 pour ~10 clusters, 3 pour ~15 clusters.
3. Cliquez **« Cluster Documents »**, puis **« Apply Tags to Zotero »** pour synchroniser.

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

---

### 8) Fiches de lecture Zotero

RAGpy génère des fiches de lecture et les ajoute comme notes enfants dans Zotero (ou enrichit le champ résumé).

#### Configuration

1. Créer une clé API Zotero sur https://www.zotero.org/settings/keys/new (accès à la bibliothèque, aux notes, et en écriture).
2. La saisir dans **Paramètres > Mes Identifiants** avec l'identifiant de bibliothèque (`zotero_user_id` ou `zotero_group_id`).

#### Modes

| Mode | Contenu | Gabarit de prompt |
|------|---------|-------------------|
| `extended` [FICHE] | Fiche complète (2500-3000 mots), par défaut | `app/utils/zotero_prompt.md` |
| `pedagogique` [CLAIR] | Fiche pour étudiants de L3 | `app/utils/zotero_prompt_pedagogique.md` |
| `evaluation` [EVAL] | Grille d'évaluation par les pairs (SHS) | `app/utils/zotero_prompt_evaluation.md` |
| `book` [LIVRE] | Livre analysé chapitre par chapitre, en plusieurs phases | `app/utils/book_prompt.md` |
| `short` | Résumé court qui enrichit le champ Abstract | `app/utils/zotero_prompt_short.md` |

Le mode `book` détecte les chapitres (table des matières, titres Markdown de l'OCR), analyse chaque chapitre en gardant la mémoire des précédents, puis rédige la synthèse ; `BOOK_NOTE_VERSION=v1` désactive les contrôles qualité de la version 2.

#### Utilisation

Après l'étape 1, dans **« Generate Zotero Reading Notes »** : choisir le mode et le modèle (OpenAI, OpenRouter `provider/model` ou Albert `albert/<id>`), puis lancer. Les appels LLM sont limités sur toute la plateforme par `MAX_CONCURRENT_LLM_CALLS`.

#### Personnalisation

Éditez les gabarits ci-dessus. Placeholders : `{TITLE}`, `{AUTHORS}`, `{DATE}`, `{DOI}`, `{URL}`, `{ABSTRACT}`, `{TEXT}`, `{LANGUAGE}`. Guide : [.claude/docs/README_ZOTERO_PROMPT.md](.claude/docs/README_ZOTERO_PROMPT.md).

---

### 9) Import Publish or Perish

Depuis un projet, **Import Citations (PoP)** :

1. Téléverser l'export JSON de Publish or Perish.
2. Filtrer les citations par pertinence avec un LLM (progression en temps réel ou en tâche de fond).
3. Valider la sélection dans la prévisualisation (citations retenues et écartées).
4. Importer dans Zotero : création ou mise à jour des items, récupération du contenu web en parallèle (`MAX_CONCURRENT_WEB_FETCHES`, lots de `DEFAULT_CITATION_BATCH_SIZE`).

Une session d'import interrompue peut être reprise depuis le projet.

---

### 10) Comptes, projets et droits

#### Rôles et identifiants

| Rôle | Identifiants personnels | Repli sur `.env` |
|------|------------------------|------------------|
| **ADMIN** | ✅ Prioritaires | ✅ Si vides |
| **NON-ADMIN** | ✅ Uniquement | ❌ Jamais |

Les sous-processus d'un non-admin ne reçoivent que ses propres clés ; ils ne peuvent pas relire celles du `.env` serveur.

#### Projets et sessions

| Rôle dans le projet | Lire les sessions | Lancer, écrire, arrêter |
|---------------------|------------------|-------------------------|
| Propriétaire | ✅ | ✅ |
| Collaborateur | ✅ | ✅ |
| Lecteur (`viewer`) | ✅ | ❌ (403) |

Un import fait hors projet est réservé à son auteur. Un dossier de session sans propriétaire enregistré (import antérieur à l'audit de septembre 2026, projet supprimé) est réservé aux administrateurs.

#### Endpoints d'authentification

| Endpoint | Description |
|----------|-------------|
| `POST /auth/register` | Inscription + e-mail de vérification |
| `POST /auth/login` | Connexion (jeton JWT) |
| `POST /auth/refresh` | Renouvellement du jeton |
| `GET /auth/verify-email/{token}` | Vérification de l'adresse |
| `POST /auth/forgot-password` | Demande de réinitialisation |
| `POST /auth/reset-password` | Nouveau mot de passe |
| `GET /auth/me` | Utilisateur courant |

La documentation interactive de l'API est servie sur `/docs`.

---

### 11) Ligne de commande

Le même pipeline se lance sans interface, depuis la racine du dépôt :

```bash
# 1. Export Zotero + PDF/EPUB → CSV (OCR)
python scripts/rad_dataframe.py --json sources/MaBiblio/MaBiblio.json --dir sources/MaBiblio --output sources/MaBiblio/output.csv

# 2. Chunking + embeddings denses + sparses (phases : initial | dense | sparse | all)
python scripts/rad_chunk.py --input sources/MaBiblio/output.csv --output sources/MaBiblio --phase all

# 2 bis. Recodage par OpenRouter, ou chaîne souveraine Albert
python scripts/rad_chunk.py --input sources/MaBiblio/output.csv --output sources/MaBiblio --phase all --model google/gemini-2.5-flash
python scripts/rad_chunk.py --input sources/MaBiblio/output.csv --output sources/MaBiblio --phase all --model albert/ministral-3-8b-instruct-2512 --embedding-provider albert

# 3a. Insertion dans une base vectorielle (rad_vectordb.py ne lit pas le .env : l'exporter d'abord)
set -a && source .env && set +a && python scripts/rad_vectordb.py --input sources/MaBiblio/output_chunks_with_embeddings_sparse.json --db pinecone --index mon-index

# 3b. Clustering automatique
python scripts/rad_clustering.py --input sources/MaBiblio/output_chunks_with_embeddings_sparse.json --output sources/MaBiblio --session-name MaBiblio --min-cluster-size 5
```

- `rad_dataframe.py` ne lit que les pièces jointes situées sous `--dir` (les chemins d'une autre machine sont retrouvés par `CLÉ/fichier`) ; `--allow-outside-dir` lève cette restriction pour un usage local.
- `rad_vectordb.py --db` : `pinecone` (`--index`, `--namespace`), `weaviate` (`--class-name`, `--tenant`), `qdrant` (`--collection`), `albert` (`--albert-collection-name`, `--albert-create-collection`, `--albert-ack-retention` obligatoire).
- Sans clé OpenAI, `rad_chunk.py` s'arrête avec un message (code 1) pour les phases qui l'exigent ; il ne demande jamais de clé de façon interactive.

**Maintenance** : `scripts/rebuild_pinecone_index.py` (reconstruction, `--backfill-hash`), `scripts/backfill_pinecone_inplace.py --id-scheme v2 --dry-run` (migration des identifiants de déduplication), `scripts/rotate_credentials_key.py` (rotation du secret JWT), `scripts/albert_probe.py` (sondes de l'API Albert).

---

### 12) Option souveraine : API Albert (DINUM)

RAGpy peut confier ses quatre usages externes à l'API Albert de la DINUM, hébergée en France : recodage, notes et filtre de citations (modèles `albert/<modèle>`) ; OCR ; embeddings `bge-m3` en 1024 dimensions ; collections vectorielles privées. L'intégration est **désactivée par défaut** (`ALBERT_ENABLED=0` : comportement strictement identique) et s'active **par capacité** :

| Capacité | Activation |
|----------|-----------|
| Chat (recodage, fiches, citations) | choisir un modèle `albert/<modèle>` |
| OCR | `OCR_ENABLE_ALBERT=1` (maillon placé avant Mistral) |
| Embeddings | `EMBEDDING_PROVIDER=albert` ou choix dans l'interface (espace 1024 d séparé, jamais mélangé aux index 3072 d) |
| Base vectorielle | cible `albert` (collections privées, acquittement de rétention obligatoire) |

Minimum dans `.env` : `ALBERT_ENABLED=1` et `ALBERT_API_KEY` (repli réservé aux administrateurs ; les autres utilisateurs saisissent leur clé Albert dans leurs identifiants). Un modèle `albert/…` demandé alors qu'Albert est désactivé est refusé (400), sans repli silencieux vers OpenAI ou OpenRouter. Les quotas sont partagés par compte Albert. Quotas, échéances des modèles, RGPD, sécurité et déploiement : [.claude/docs/albert.md](.claude/docs/albert.md) et le bloc ALBERT de `.env.example`.

---

## C — Exploitation

### 13) Mode Celery

Par défaut, chaque étape tourne en sous-processus de l'application, avec une progression en temps réel (SSE). Le mode Celery confie les étapes à un worker :

1. `ENABLE_CELERY=true` dans `.env`, et `FLOWER_USER` / `FLOWER_PASSWORD`.
2. `docker compose up -d` (Redis, worker, Beat et Flower sont inclus).
3. Endpoints sous `/api/celery/` (`process_dataframe`, `initial_chunking`, `dense_embedding`, `sparse_embedding`, `upload_vectordb`, `task/{id}/status`, `task/{id}/cancel`, `status`, `workers`).

Le worker exécute les mêmes scripts, dans un sous-processus par utilisateur, avec ses seules clés (aucun secret ne transite par Redis). Flower n'est publié que sur `127.0.0.1:5555` : y accéder par un tunnel SSH, `ssh -L 5555:127.0.0.1:5555 utilisateur@serveur`.

---

### 14) Déploiement en production

Liste de contrôle avant `docker compose up -d --build` sur un serveur :

- [ ] `RAGPY_ENV=production` et un `JWT_SECRET_KEY` aléatoire d'au moins 32 caractères (sinon l'application refuse de démarrer). Serveur existant : garder le secret déjà en place, ou suivre la procédure de rotation (section 3).
- [ ] `APP_URL` et `CORS_ORIGINS` sur l'URL publique ; `RESEND_API_KEY` et `RESEND_FROM_EMAIL` pour les e-mails ; `USERS_SANDBOX=TRUE`.
- [ ] `FLOWER_USER` / `FLOWER_PASSWORD` définis ; `REDIS_BIND` et `FLOWER_BIND` à `127.0.0.1`.
- [ ] `UVICORN_WORKERS=1`.
- [ ] Reverse proxy (nginx) : `client_max_body_size` aligné sur `UPLOAD_MAX_MB`, délais de lecture longs et tampon désactivé pour les flux SSE.
- [ ] Albert : déployer le code, puis `ALBERT_API_KEY` et `ALBERT_ENABLED=1`, `ALBERT_LIMITER_BACKEND=redis`, puis `docker compose up -d --build`.
- [ ] Après la mise à jour de septembre 2026 : les vecteurs sparse produits avant ne sont plus comparables (relancer l'étape 3b puis l'envoi pour les index hybrides) ; les sessions importées avant, sans propriétaire enregistré, deviennent réservées aux administrateurs.

---

### 15) Tests

```bash
# Suite principale, entièrement hors ligne
python -m pytest tests/

# Suite complète hors ligne, réseau coupé (utilisée pour valider Albert désactivé)
HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 NO_PROXY=127.0.0.1,localhost,testserver python -m pytest tests/ scripts/test_rad_vectordb.py scripts/test_rad_chunk_initial_phase.py scripts/test_albert_guard_scope.py app/test_main.py -q -p no:cacheprovider

# Analyse statique (erreurs seulement, ruff.toml)
python -m pip install -r scripts/requirements-dev.txt && python -m ruff check .

# Tests live Albert (clé réelle, consomme le quota ; interrupteur dans le shell, jamais dans .env)
ALBERT_LIVE=1 python -m pytest tests/live -m albert_live -v -p no:cacheprovider
```

La CI (`.github/workflows/tests.yml`) exécute la suite hors ligne sous Python 3.11. Les tests asynchrones utilisent `@pytest.mark.anyio`.

---

## D — Projet

### 16) Architecture technique

```
ragpy/
├── app/                        # Application web FastAPI
│   ├── main.py                 # Routeurs, cycle de vie, /health, /metrics
│   ├── config.py               # Réglages (JWT, base, CORS, e-mails)
│   ├── core/                   # credentials, session_access, upload_safety, security, scheduler
│   ├── routes/                 # auth, users, admin, projects, pages, pipeline, ingestion,
│   │                           # processing (SSE), celery_tasks, settings, citations, background_tasks
│   ├── services/               # job_control (admission), process_manager, session_cleanup, e-mails
│   ├── tasks/                  # Tâches Celery (mêmes scripts en sous-processus)
│   ├── utils/                  # Zotero, notes LLM, notes de livres, citations, prompts *.md
│   ├── database/ models/ schemas/   # SQLAlchemy (SQLite) + Pydantic
│   └── templates/ static/      # Interface Jinja2
├── scripts/                    # Pipeline
│   ├── rad_dataframe.py        # Zotero + PDF/EPUB → CSV (chaîne OCR)
│   ├── ocr_local.py            # OCR local Docling (venv séparé)
│   ├── rad_chunk.py            # Chunking, recodage, embeddings denses et sparses
│   ├── rad_clustering.py       # UMAP + HDBSCAN
│   ├── rad_vectordb.py         # Pinecone, Weaviate, Qdrant, collections Albert
│   ├── rad_dedup.py, rad_recode_cache.py, rad_sparse.py, rad_providers.py, rad_env.py
│   ├── rad_albert/             # Client de l'API Albert (DINUM)
│   ├── rebuild_pinecone_index.py, backfill_pinecone_inplace.py, rotate_credentials_key.py
│   └── requirements*.txt       # Dépendances (principal, dev, OCR local)
├── ingestion/                  # Ingestion CSV directe (sans OCR)
├── core/                       # Classe Document commune aux sources
├── tests/                      # Suite pytest (hors ligne) et tests live
├── data/  uploads/  logs/  sources/
├── .env.example                # Référence de configuration
└── docker-compose.yml, Dockerfile
```

Documentation détaillée : [.claude/CLAUDE.md](.claude/CLAUDE.md) (guide complet), [.claude/pipeline_current_architecture.md](.claude/pipeline_current_architecture.md) et [.claude/docs/](.claude/docs/) (ingestion CSV, prompts Zotero, SSE, clustering, interface, Albert).

#### Technologies clés

| Composant | Technologie |
|-----------|-------------|
| Backend | FastAPI + Uvicorn |
| Auth | JWT (python-jose) + bcrypt, identifiants chiffrés (Fernet) |
| BDD | SQLAlchemy + SQLite |
| OCR | Albert (option) → Mistral OCR → Docling local → PyMuPDF ; EPUB via ebooklib |
| LLM | OpenAI, OpenRouter, Albert |
| Embeddings | OpenAI `text-embedding-3-large` (3072 d) ou Albert `bge-m3` (1024 d) |
| Sparse | spaCy FR `fr_core_news_md` |
| Clustering | UMAP + HDBSCAN |
| Vector DB | Pinecone, Weaviate, Qdrant, collections Albert |
| File de tâches | Celery + Redis, Flower |
| Supervision | Prometheus (`/metrics`) |

---

### 17) Dépannage (FAQ)

#### Installation et démarrage

| Problème | Solution |
|----------|----------|
| Port 8000 occupé | `lsof -i :8000` puis `kill <PID>` |
| Dépendances manquantes | `python -m pip install -r scripts/requirements.txt` |
| spaCy manquant | `python -m spacy download fr_core_news_md` |
| « Démarrage refusé (RAGPY_ENV=production) » | Définir un `JWT_SECRET_KEY` aléatoire d'au moins 32 caractères |
| Conteneur `flower` qui redémarre en boucle | Définir `FLOWER_USER` et `FLOWER_PASSWORD` (pas les valeurs par défaut) |
| Une étape échoue en local (module introuvable) | Activer le venv : l'application lance `python3` |
| Clés des utilisateurs illisibles après un changement de `JWT_SECRET_KEY` | Remettre l'ancien secret dans `JWT_SECRET_KEY_PREVIOUS` et lancer `scripts/rotate_credentials_key.py` |
| Code modifié mais comportement inchangé sous Docker | `docker compose up -d --build` (le code est dans l'image) |

#### Traitements

| Code / symptôme | Signification |
|-----------------|---------------|
| 403 avec `credential_required` | Clé manquante : la saisir dans Paramètres > Mes Identifiants |
| 403 sur une session de projet | Rôle lecteur, ou session d'un autre utilisateur |
| 409 | Un traitement du même type tourne déjà sur cette session |
| 429 | `MAX_ACTIVE_JOBS` ou `MAX_ACTIVE_JOBS_PER_USER` atteint |
| 413 | Fichier ou archive au-delà de `UPLOAD_MAX_MB` / `UPLOAD_MAX_UNZIPPED_MB` / `UPLOAD_MAX_ZIP_ENTRIES` |
| 400 sur un modèle `albert/…` | Albert désactivé sur le serveur (`ALBERT_ENABLED=0`) |

#### OCR

| Problème | Solution |
|----------|----------|
| Mistral répond 401 | Clé invalide **ou** plafond de dépense mensuel atteint (console.mistral.ai → Limits) |
| Nombreux 429 Mistral | Baisser `MISTRAL_CONCURRENT_CALLS` et `PDF_EXTRACTION_WORKERS` (garder les deux égaux) |
| Livre scanné quasi vide (`texteocr_partial`) | Mistral a échoué et le texte vient de PyMuPDF : vérifier la clé Mistral, ou installer l'OCR local (`INSTALL_LOCAL_OCR=true`) |
| PDF très volumineux refusé | Laisser `MISTRAL_AUTO_COMPRESS` et `MISTRAL_AUTO_SPLIT` à `true` |

#### Clustering

| Problème | Solution |
|----------|----------|
| Trop peu de clusters | Réduire `min_cluster_size` (ex. 3 ou 5) |
| Trop de « noise » | Augmenter `min_cluster_size` ou vérifier la qualité des embeddings |
| HDBSCAN crash | Vérifier `scikit-learn<1.6` dans requirements.txt |

#### Zotero

| Problème | Solution |
|----------|----------|
| Clé API invalide | Vérifier les droits bibliothèque, notes et écriture |
| Tags non appliqués | Vérifier `zotero_api_key` et l'identifiant de bibliothèque dans les identifiants |
| Erreur 404 | L'itemKey n'existe pas dans votre bibliothèque |

---

### 18) Licence

MIT. Voir `LICENSE`.
