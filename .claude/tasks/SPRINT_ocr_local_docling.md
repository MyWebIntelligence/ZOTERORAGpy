# SPRINT — OCR LOCAL (Docling) : voie d'OCR sans clé, sans cap, hors-ligne

**Créé le** : 2026-06-04
**Statut** : ✅ **POC validé (GO) + intégration livrée (2026-06-04)**. Build image OK (`INSTALL_LOCAL_OCR=true`), isolation venv confirmée (env principal numpy 1.26.4 + spacy intacts ; venv numpy 2.4.4). OCR réel sur *Les Anormaux* (scanné, 0 car. en legacy) → markdown + `<!-- Page N -->` continus, **offline via Tesseract FR/EN** (~16 s/page CPU sur le frontmatter). `_local_ocr_available()=True` dans le process FastAPI. **Moteur OCR = Tesseract CLI (FR/EN), défaut retenu.** Perf ~5 s/page CPU en régime établi (TableFormer coupé + 8 threads ; était ~16 s). **Cold start** : ~60-70 s one-shot au 1er run d'un conteneur recréé = téléchargement des modèles de layout Docling depuis HuggingFace (à pré-bundler pour un vrai offline + cold start rapide — TODO).

**Benchmarks tranchés (2026-06-04, Mac ARM64 / conteneur Linux / CPU, 6 pages *Les Anormaux*) :**
- DPI (`images_scale` 0.5/1.0/2.0) : **aucun effet** (sortie identique 3349 car., temps bruité) → `images_scale` pilote l'export d'images, pas la résolution OCR. Fausse piste.
- GPU : **impossible ici** (torch `+cpu`, CUDA/MPS=False ; Docker Mac n'expose aucun GPU). Réservé à un serveur NVIDIA Linux.
- RapidOCR (onnxruntime) : **8,7 s/page vs 5,1 pour Tesseract** (1,7× plus LENT sur ARM/onnx-CPU) + dépend de modelscope.cn → **abandonné par défaut**. Knob `LOCAL_OCR_ENGINE_OCR=rapidocr` conservé (utile x86/GPU), mais `onnxruntime` retiré de l'image.

C'est un **secours** : Mistral (primaire) OCRise un livre en 1 appel ; le local ne sert que si Mistral est KO. **Reste (TODO offline complet)** : pré-bundler les modèles de layout Docling dans l'image.
**Parent** : extrait du Lot 4 de [SPRINT_ocr_resilience.md](SPRINT_ocr_resilience.md)
**Module principal** : `scripts/rad_dataframe.py` (étape 1) + nouveau `scripts/ocr_local.py`

> **État par lot** : 4-A ✅ (`ocr_local.py`, pagination) · 4-B ✅ (`LocalProvider`, `available()`, sémaphore) · 4-C ✅ (chaîne + skip recodage `rad_chunk`) · 4-D ✅ (`requirements-ocr-local.txt`, `Dockerfile ARG INSTALL_LOCAL_OCR`) · 4-E ✅ (env + `.env.example`) · 4-F ✅ (`tests/test_ocr_local.py` + `TestLocalOcrChain`/`TestLocalOcrSubprocessWrapper`). **Reste** : installer Docling et valider qualité FR + pagination sur un livre scanné réel (*Les Anormaux*).

---

## 1. Contexte & déclencheur

Incident réel (corpus Foucault, 2026-06-02) : le plafond de dépense mensuel Mistral atteint → **401 sur tous les livres** → bascule sur l'extracteur `legacy` (PyMuPDF texte). Résultat observé dans `uploads/41b53169_FoucaultLivre03/` :

| Livres | legacy |
|---|---|
| 7 né-numériques | texte utilisable (435 K–1 M car.) |
| **Sécurité, territoire, population** (scanné) | **`OCR_PARTIAL`** — 9 car./page sur 449 p |
| **L'archéologie du savoir** (scanné) | **`OCR_FAILED`** — 0 car. |
| **Les Anormaux** (scanné) | **`OCR_FAILED`** — 0 car. |

`legacy` **n'est pas un OCR** : sur un PDF scanné (pas de couche texte), il ne sort rien. Les 3 livres scannés sont précisément ceux qui exigent un vrai moteur OCR — et quand Mistral est indisponible (plafond/panne/quota), **il n'existe aucune alternative**. La Phase 1 (Lots 1+2) a rendu ces échecs **visibles** ; ce sprint fournit la **solution** : un OCR **local**, sans clé API, sans cap de pages, exécutable hors-ligne.

**Chaîne actuelle** : `Mistral → (OpenAI opt-in) → legacy`. **Cible de ce sprint** : insérer un moteur local → `Mistral → docling → legacy`.

---

## 2. Objectifs / Non-objectifs

**Objectifs**
1. Un `LocalProvider` (moteur **Docling** par défaut) capable d'OCRiser un **livre scanné entier** sans clé API ni cap.
2. **Aucune régression de l'image FastAPI** : les dépendances lourdes (torch, modèles) sont **isolées en subprocess**, jamais importées dans le process web.
3. **Contrat de sortie respecté** : markdown + marqueurs `<!-- Page N -->` à pagination **continue** (sinon chunking et `book_note_generator` cassent).
4. Sélectionnable dans la chaîne (`OCR_PROVIDER_CHAIN`, Lot 5) ; `available()` se désactive proprement si le moteur n'est pas installé.
5. Comme Mistral, l'OCR local produit un markdown propre → **le recodage GPT doit être sauté** au chunk (cf. `rad_chunk.py`).

**Non-objectifs**
- Finetuner / entraîner un modèle OCR.
- GPU obligatoire (CPU par défaut ; GPU = option).
- Remplacer Mistral (qui reste le primaire quand le plafond le permet).

---

## 3. Choix du moteur

| Moteur | Pourquoi | Réserve |
|---|---|---|
| **Docling (IBM)** ✅ *défaut* | PDF→markdown propre, layout + tableaux, API Python claire, OCR intégré (EasyOCR/Tesseract) pour les scannés, licence MIT | deps moyennes (torch via easyocr) |
| MinerU (OpenDataLab) | excellent académique (formules/tableaux) | deps plus lourdes |
| Marker / Surya (Datalab) | académique multilingue, option API Datalab | API = repasse en mode cloud |

Décision : **Docling** par défaut (`texteocr_provider="docling"`), abstraction permettant d'ajouter MinerU/Marker plus tard via `LOCAL_OCR_ENGINE`. **Spot-check FR obligatoire** (1 article 2 colonnes + 1 livre scanné) avant adoption.

---

## 4. Architecture cible

### 4.1 Isolation par subprocess (`scripts/ocr_local.py`)
Nouveau script CLI dédié, **jamais importé** par `app/` :
```
python scripts/ocr_local.py --input <pdf> --output <md> [--engine docling] [--device cpu|cuda] [--max-pages N]
```
- Charge Docling, convertit, **émet markdown + `<!-- Page N -->`** (voir §4.3), écrit dans `--output` (ou stdout).
- Code de sortie ≠ 0 + message stderr en cas d'échec → le provider le mappe en échec (chaîne continue).
- Raison : torch/easyocr restent hors du process FastAPI ; on garde le modèle de subprocess déjà utilisé par le pipeline (`build_subprocess_env`).

### 4.2 `LocalProvider` dans `scripts/rad_dataframe.py`
- `_extract_text_with_local(pdf_path, max_pages)` : lance `ocr_local.py` via `subprocess.run` (timeout `LOCAL_OCR_TIMEOUT`), lit le markdown produit, renvoie une structure analogue à `_MistralOcr` (`text, partial, pages_done, pages_total, error`).
- `_local_ocr_available()` : vérifie l'import Docling **dans le subprocess** (ex. `python scripts/ocr_local.py --check`) OU la présence du binaire/venv ; si absent → provider **sauté silencieusement** dans la chaîne (pas d'erreur).
- Réutiliser : garde-fou densité `_ocr_density_warning` (un scan illisible reste flaggé `partial`), `_safe_unlink` (fichier md temporaire), et le **sémaphore** (un `LOCAL_OCR_SEMAPHORE` distinct, défaut 1-2 : l'OCR local est CPU/RAM-intensif, ≠ rate-limit réseau).

### 4.3 Pagination `<!-- Page N -->` (point technique clé)
Docling `export_to_markdown()` produit le markdown du document **entier sans bornes de page**. Or le contrat aval exige des marqueurs `<!-- Page N -->` continus. Approche recommandée (à valider sur la version Docling installée) :
- itérer les pages du `DoclingDocument` (`result.document.pages`) et exporter **par page** (ou regrouper les items par `prov.page_no`), en préfixant chaque page de `<!-- Page N -->` ;
- numérotation **1→N continue** ; si on découpe un gros PDF en lots (RAM), réutiliser la **renumérotation cumulative** déjà éprouvée dans `_extract_text_with_mistral` (offset par part).
- Test dédié : un PDF 3 pages → markdown avec `<!-- Page 1/2/3 -->` dans l'ordre.

### 4.4 Intégration chaîne + skip recodage
- **Lot 5** insère `docling` dans `OCR_PROVIDER_CHAIN` (ex. `mistral,docling,legacy`). Sans Lot 5, ajouter d'abord un appel local **avant** `legacy` dans `extract_text_with_ocr` (derrière `_local_ocr_available()`).
- **`rad_chunk.py`** : ajouter `"docling"` (et tout provider local) à l'ensemble des providers qui **sautent le recodage GPT** (aujourd'hui `{"mistral","csv"}`) — l'OCR local sort déjà du markdown propre.

---

## 5. Lots de travail

### 4-A — POC Docling isolé (`scripts/ocr_local.py`)
Script + conversion Docling + émission `<!-- Page N -->`. Vérif manuelle sur 1 livre scanné (ex. *Les Anormaux*) et 1 article 2 colonnes FR. **Effort** : M. **Livrable** : markdown paginé correct, hors process web.

### 4-B — `LocalProvider` + densité + sémaphore (`rad_dataframe.py`)
`_extract_text_with_local`, `_local_ocr_available`, `LOCAL_OCR_SEMAPHORE`, mapping échec/partiel (réutilise `_ocr_density_warning`, `OCRResult`). **Effort** : M.

### 4-C — Branchement dans la chaîne + skip recodage
Insertion avant `legacy` dans `extract_text_with_ocr` (ou via `OCR_PROVIDER_CHAIN` si Lot 5 fait) + ajout `docling` au skip-recodage de `rad_chunk.py`. **Effort** : S.

### 4-D — Packaging deps & Docker
- `scripts/requirements-ocr-local.txt` (docling + easyocr/tesseract) **séparé** du requirements principal.
- `docker-compose.yml` : service optionnel commenté `ocr_local` (sur le modèle de la section `qdrant`) **ou** image multi-stage ; CPU par défaut, `LOCAL_OCR_DEVICE=cuda` si GPU dispo. Documenter l'install. **Effort** : M (packaging + perfs).

### 4-E — Variables d'env & sécurité
- `LOCAL_OCR_ENGINE=docling|mineru|marker` (défaut `docling`), `LOCAL_OCR_DEVICE=cpu|cuda`, `LOCAL_OCR_MAX_PAGES`, `LOCAL_OCR_TIMEOUT`, `LOCAL_OCR_CONCURRENCY`.
- Pas de credential (local) → rien à ajouter au modèle par-rôle ; passer par `build_subprocess_env` comme les autres subprocess. **Effort** : S.

### 4-F — Tests (`tests/test_ocr_providers.py`)
- `_local_ocr_available()` False quand Docling absent → provider sauté, chaîne continue.
- `_extract_text_with_local` : mock subprocess → markdown paginé `<!-- Page N -->` ; échec subprocess → erreur mappée (pas de crash) ; scan illisible → `partial=True` via densité.
- Chaîne : Mistral KO + local OK → `provider="docling"` (pas `legacy`).
- `rad_chunk` saute le recodage pour `texteocr_provider="docling"`.
**Effort** : S-M.

---

## 6. Risques & dépendances
- **Deps lourdes / CONFLIT de versions (résolu)** : Docling exige `numpy 2.x` + `httpx 0.28`, qui **cassent** `spacy`/`thinc` (numpy<2) et `mistralai`/`weaviate` (httpx<0.28) du pipeline → installer Docling dans le **même env casse le build** (constaté : `numpy.dtype size changed` sur `spacy download`). **Solution livrée** : venv dédié `/opt/ocr-venv` (Dockerfile, `INSTALL_LOCAL_OCR=true`), subprocess lancé avec son python (`LOCAL_OCR_PYTHON` / `_local_ocr_python()`), torch **CPU-only** (évite ~5 Go CUDA + le disque plein observé). Perf CPU sur gros livres → `LOCAL_OCR_MAX_PAGES`, GPU optionnel.
- **Pagination Docling** : l'API d'export ne garantit pas les bornes de page → c'est le point à sécuriser en premier (§4.3) ; sinon chunking + `book_note_generator` (détection coarse pagination) cassent.
- **Qualité FR** : valider sur scan réel FR avant d'en faire le secours par défaut (spot-check 4-A).
- **Concurrence** : OCR local = CPU/RAM-bound, pas réseau → sémaphore bas (1-2) distinct de `MISTRAL_SEMAPHORE`, sinon OOM.
- **Contrat de sortie** : markdown + `<!-- Page N -->` continu, identique aux autres providers.

---

## 7. Phasage
1. **POC** : 4-A (valider Docling + pagination + qualité FR sur 1 scanné). Go/No-Go ici.
2. **Intégration** : 4-B + 4-C (provider branché, recodage sauté).
3. **Industrialisation** : 4-D + 4-E + 4-F (packaging Docker, env, tests).

---

## 8. Définition de « terminé » (DoD)
- [ ] Un livre **scanné** (ex. *Les Anormaux*) est OCRisé **entièrement en local**, **sans aucune clé API**, pagination `<!-- Page N -->` continue.
- [ ] Qualité ≥ legacy (et nettement supérieure sur un scanné où legacy = 0).
- [ ] Deps lourdes **hors** image FastAPI (subprocess/service dédié) ; `available()` se désactive proprement si non installé.
- [ ] `docling` inséré dans la chaîne et **recodage GPT sauté** au chunk.
- [ ] Tests verts + doc à jour (`.claude/CLAUDE.md` section OCR + `.env.example` + `docker-compose.yml`).
