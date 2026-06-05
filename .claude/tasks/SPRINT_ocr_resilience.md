# SPRINT — Résilience & économie de l'OCR (multi-fournisseurs, dont OCR local)

**Créé le** : 2026-05-27
**Statut** : Phase 1 livrée (Lot 1 + Lot 2 ✅, 2026-05-27) ; Phases 2-3 à planifier
**Module principal** : `scripts/rad_dataframe.py` (étape 1 du pipeline)

---

## 1. Contexte & déclencheurs

Session du 2026-05-27 — incidents réels observés sur un corpus de livres/articles FR :

| Incident | Symptôme | Cause | Impact |
|---|---|---|---|
| **Plafond mensuel Mistral atteint** | `401` même sur `/v1/models` (gratuit) | Spend cap du workspace → tout le workspace bloqué | Toute la chaîne OCR tombe ; ~heures perdues à croire à une clé invalide |
| **404 « Could not get file »** | `404` sur `/v1/ocr` pour **un** PDF, l'autre passe | Glitch transitoire upload→OCR côté Mistral (pas la clé, pas l'URL) | Bascule injustifiée sur le fallback dégradé |
| **Fallback OpenAI plafonné** | livre de 284 p → CSV de **7 327 car. (~10 pages)** marqué « ✓ success » | `OPENAI_OCR_MAX_PAGES=10`, OCR page-par-page (1 appel vision/page) | **Troncature silencieuse** → CSV inexploitable sans alerte |

**Problème de fond** : l'OCR repose sur un **point de défaillance unique** (Mistral) avec un **seul filet de secours inadapté** aux livres (OpenAI vision, 10 pages). Quand Mistral est indisponible (plafond, incident, quota), il n'existe **aucune alternative économique** capable de traiter un livre entier.

**Chaîne actuelle** (`extract_text_with_ocr`) : Mistral `/v1/ocr` → OpenAI vision (cap 10 p) → PyMuPDF legacy. Provider tracé dans la colonne `texteocr_provider` (`mistral` | `openai` | `legacy` | `csv` | `epub`).

---

## 2. Scénario d'usage cible (cadre du sprint)

- **Corpus académique francophone** ingéré via **Zotero** : mélange d'**articles** (5-30 p, souvent 2 colonnes) et de **livres** (100-460+ p, parfois **scannés**, jusqu'à ~60 MB).
- **Budget contraint** : plafonds de dépense mensuels (cf. incident) → besoin d'options **sans coût marginal par page** et **sans cap** qui bloque tout.
- **Résilience** : un incident d'un fournisseur ne doit jamais dégrader silencieusement la qualité ni stopper le corpus.
- **Confidentialité / autonomie** : intérêt fort pour une voie **100 % locale** (documents sensibles, gros volumes, indépendance vis-à-vis des API).
- **Qualité FR** : markdown structuré, multi-colonnes, notes de bas de page, tableaux ; les benchmarks anglo-centrés (MMLU-Pro) sont de mauvais proxys → privilégier des moteurs solides en français.

---

## 3. Objectifs / Non-objectifs

**Objectifs**
1. **Aucune troncature silencieuse** : un OCR partiel/échoué est signalé explicitement, jamais marqué « success ».
2. **Aucun point de défaillance unique** : ≥ 2 fournisseurs capables de traiter un **livre entier** (un cloud économique + un **local**).
3. **Chaîne de fallback configurable** (ordre par variable d'env), pas codée en dur.
4. **Voie OCR locale** opérationnelle (coût = compute, pas de cap, hors-ligne possible).
5. **Robustesse Mistral** : retries sur erreurs transitoires + diagnostic clair (spend-cap vs clé vs taille).

**Non-objectifs**
- Ré-entraîner/finetuner un modèle OCR.
- Changer le format de sortie (`<!-- Page N -->` + markdown reste le contrat avec l'étape chunking).
- Toucher aux étapes chunking/embeddings/vectordb (hors périmètre).

---

## 4. Architecture cible

### 4.1 Abstraction « OCR provider »
Définir une interface commune dans `scripts/rad_dataframe.py` (ou un sous-module `scripts/ocr/`) :

```python
class OcrProvider(Protocol):
    name: str                       # -> valeur de texteocr_provider
    def available(self) -> bool      # clé/binaire présents ?
    def extract(self, pdf_path: str, *, max_pages: int | None) -> OcrResult
        # OcrResult: text(markdown + <!-- Page N -->), pages_done, pages_total,
        #            partial: bool, error: str | None
```

Réutiliser l'existant comme implémentations : `_extract_text_with_mistral` → `MistralProvider`, `_extract_text_with_openai` → `OpenAIVisionProvider`, flux PyMuPDF → `LegacyProvider`. Conserver `_pdf_size_mb`, `_compress_pdf_for_ocr`, `_split_pdf_for_ocr`, `_safe_unlink`, le sémaphore `MISTRAL_SEMAPHORE`.

### 4.2 Chaîne de fallback configurable
```bash
OCR_PROVIDER_CHAIN=mistral,docling,legacy   # ordre d'essai
```
- Itère la chaîne ; pour chaque provider `available()`, tente `extract()`.
- **Ne descend au provider suivant que sur échec réel** (pas sur un partiel acceptable).
- Trace le provider gagnant dans `texteocr_provider`, et `texteocr_partial=True` + `texteocr_pages_done/total` si dégradé.

---

## 5. Lots de travail

### Lot 1 — Robustesse Mistral (retry transitoire + diagnostic) ✅ LIVRÉ (2026-05-27, durci)
**But** : absorber les glitches transitoires et ne plus confondre les causes d'échec.
**Livré** : `_mistral_upload_and_ocr` = wrapper retry (`MISTRAL_OCR_RETRIES`=4) autour de `_mistral_upload_and_ocr_once` ; backoff **exponentiel** `MISTRAL_OCR_RETRY_BACKOFF * 2**tentative` plafonné à `MISTRAL_OCR_RETRY_MAX_BACKOFF`=60 s, **jitter** anti-thundering-herd ; **`Retry-After` (429)** lu via `_parse_retry_after` et respecté exactement par `_mistral_retry_wait` ; `_classify_mistral_http_error` (401 permanent + message plafond, 404/408/429/5xx transitoires, autres 4xx permanents) ; sémaphore acquis par tentative (sleep hors slot). Tests : `tests/test_ocr_providers.py` (46).

### Politique fournisseurs — OpenAI désactivé par défaut ✅ LIVRÉ (2026-05-27)
**But** (demande explicite) : faire disparaître l'OCR OpenAI, qui tronque les livres (cap 10 p).
**Livré** : `OCR_ENABLE_OPENAI_FALLBACK` (défaut **0**) ; la chaîne par défaut devient **Mistral → legacy**. La branche OpenAI + garde-fous Lot 2 restent dans le code, exécutés seulement si `=1`. Tests : OpenAI jamais appelé par défaut, ré-activable à la demande. **Note** : ne PAS adopter le retry « toute-extraction » proposé ailleurs (re-split/re-OCR de toutes les parts d'un livre sur un blip) — le retry **par part** existant est plus efficace.

### Salvage partiel des livres découpés ✅ LIVRÉ (2026-05-27, review-driven)
**But** : une part en échec ne doit plus jeter tout le livre (revue adversariale, finding HIGH).
**Livré** : `_extract_text_with_mistral` → `_MistralOcr(text, partial, pages_done, pages_total, error)` ; salvage des parts réussies + marqueur `<!-- … OCR ÉCHOUÉ … -->` (renumérotation globale préservée) + flag `partial` ; `_MistralAuthError` (401 compte) abandonne le livre ; toutes-parts-échouées → `OCRExtractionError` (fallback legacy). HTTP-date `Retry-After` géré. Tests : `tests/test_ocr_providers.py` (58).

### Revue adversariale (2026-05-27) — différé (faible ROI / hors périmètre immédiat)
- **Circuit-breaker / concurrence adaptative** sur 429 soutenus (le jitter limite déjà le lockstep) — MEDIUM, à faire si rate-limit récurrent.
- **Timeout adaptatif** par nombre de pages d'une part — MEDIUM, défensif.
- **DELETE `/v1/files` hors sémaphore** — LOW, micro-équité.
- **`OCR_ENABLE_OPENAI_FALLBACK` par-utilisateur** — volontairement **non** fait : c'est un kill-switch admin/.env (OpenAI off pour tous), conforme à la demande.
**Fichiers** : `scripts/rad_dataframe.py` (`_extract_text_with_mistral`, `_mistral_upload_and_ocr`).
**Détails** :
- **Retry** (2-3 tentatives, backoff ~2-4 s) sur erreurs transitoires : `404 "Could not get file"`, `5xx`, timeouts, erreurs réseau. **Pas** de retry sur `401`.
- **Classifier** l'erreur et logger un message explicite :
  - `401` → « clé refusée OU **plafond de dépense mensuel atteint** (console Mistral → Limits) » (cf. mémoire `reference-mistral-401-spend-cap`).
  - `404 Could not get file` → transitoire, retry puis fallback.
  - taille/pages → renvoyer vers compress/split (déjà en place).
- Variable : `MISTRAL_OCR_RETRIES` (défaut 2).
**Acceptation** : un 404 transitoire isolé est rattrapé sans tomber sur OpenAI ; un 401 produit un message d'erreur actionnable.
**Effort** : S.

### Lot 2 — Garde-fou anti-troncature (le plus urgent) ✅ LIVRÉ (2026-05-27)
**But** : ne plus jamais marquer « success » un OCR manifestement incomplet.
**Livré** : `OCRResult` étendu (`partial`/`pages_done`/`pages_total`/`error`) ; `_extract_text_with_openai` renvoie `_PagedOcr` ; `extract_text_with_ocr` tente legacy (non plafonné) avant d'accepter un OpenAI plafonné, sinon flag `partial=True` ; garde-fou densité `_ocr_density_warning` (`OCR_MIN_CHARS_PER_PAGE`=500) sur OpenAI complet + legacy (chemin Mistral non touché) ; colonnes CSV `texteocr_partial/_pages_done/_pages_total` + entrée `OCR_PARTIAL` dans `*_errors.json` + WARNING. Tests : `tests/test_ocr_providers.py`.
**Fichiers** : `scripts/rad_dataframe.py` (`_extract_text_with_openai`, agrégation `extract_text_with_ocr`).
**Détails** :
- Quand un provider plafonné (OpenAI, `OPENAI_OCR_MAX_PAGES=10`) traite un PDF dont `total_pages > pages_done`, marquer le résultat **`partial=True`** et :
  - logger un `WARNING` explicite (« OCR partiel : 10/284 pages »),
  - enregistrer dans `output_errors.json` + colonnes `texteocr_partial` / `texteocr_pages_done` / `texteocr_pages_total`,
  - **continuer la chaîne** vers un provider non plafonné (OCR local) avant d'accepter le partiel.
- Heuristique de sanité générique : si `len(texte)/total_pages` est anormalement bas (ex. < 500 car./page sur un doc texte), flaguer suspect.
**Acceptation** : le cas « livre 284 p → 10 pages » ressort en partiel/échec visible (UI + errors.json), jamais en `success` silencieux.
**Effort** : S-M.

### Lot 4 — OCR LOCAL (cœur de la demande) 🧩 → **sprint dédié : [SPRINT_ocr_local_docling.md](SPRINT_ocr_local_docling.md)**
**But** : voie d'OCR **sans cap, sans coût marginal, hors-ligne**, idéale pour les gros livres scannés et la confidentialité — c'est elle qui sauverait les livres scannés que `legacy` ne sait pas lire quand Mistral est KO.
**Résumé** : nouveau `LocalProvider` (moteur **Docling** par défaut) en **subprocess** isolé, sortie normalisée markdown + `<!-- Page N -->`, sélectionnable via `OCR_PROVIDER_CHAIN`. Effort : M-L.
> Détail complet (Docling, packaging deps, subprocess, Docker, tests, plan en phases) dans **[SPRINT_ocr_local_docling.md](SPRINT_ocr_local_docling.md)**.

### Lot 5 — Chaîne de fallback configurable + abstraction provider
**But** : remplacer la cascade codée en dur par la chaîne pilotable (§4).
**Fichiers** : `scripts/rad_dataframe.py` (`extract_text_with_ocr` → boucle sur la chaîne).
**Détails** :
- `OCR_PROVIDER_CHAIN` (défaut rétro-compatible : `mistral,legacy` ; OpenAI opt-in, OCR local insérable : `mistral,docling,legacy`).
- Respecter le **modèle de sécurité par rôle** (clés par-utilisateur via `build_subprocess_env`).
**Acceptation** : changer l'ordre des fournisseurs = une variable d'env, sans toucher au code ; tests unitaires sur la sélection/fallback.
**Dépendance** : Lot 4 fournit le `LocalProvider` que cette chaîne orchestre.
**Effort** : M.

### Lot 6 — Observabilité OCR par document
**But** : rendre la qualité OCR visible et auditable.
**Fichiers** : `output_errors.json`, SSE progress, éventuel petit récap CSV/JSON.
**Détails** : par document — provider final, `pages_done/total`, `chars`, `partial`, erreurs rencontrées par provider. Surface dans l'UI (badge « OCR partiel »).
**Acceptation** : on sait d'un coup d'œil quel doc est complet/partiel et par quel moteur.
**Effort** : S.

---

## 6. Comparatif fournisseurs (synthèse, à revérifier — prix volatils)

| Fournisseur | Type | Coût indicatif | Livre entier ? | FR | Note |
|---|---|---|---|---|---|
| **Mistral OCR** | Cloud dédié | ~1 $/1000 p | ✅ 1 appel | bon | Primaire ; fragile aux caps/glitches |
| **Docling** | **Local** | compute | ✅ | bon | **Défaut local recommandé** |
| **MinerU** | **Local** | compute | ✅ | bon | Fort académique (formules/tableaux) |
| **Marker/Surya** | Local / API | compute / bas | ✅ | bon | API Datalab si pas d'hébergement |
| **OpenAI vision** | Cloud VLM | élevé (page-par-page) | ❌ cap 10 p | bon | À **reléguer** en dernier recours court |
| **PyMuPDF legacy** | Local (texte) | gratuit | texte natif only | n/a | Pas d'OCR réel (PDF scannés KO) |

---

## 7. Priorisation / phasage

1. **Phase 1 (rapide, anti-régression)** : Lot 2 (garde-fou troncature) + Lot 1 (retry/diagnostic Mistral). ✅ **LIVRÉ**.
2. **Phase 2 (résilience)** : Lot 5 (chaîne configurable) + Lot 4 (**OCR local** → sprint dédié `SPRINT_ocr_local_docling.md`). → plus de point unique, voie sans cap/hors-ligne.
3. **Phase 3 (finitions)** : Lot 6 (observabilité OCR par document).

---

## 8. Risques & dépendances

- **Deps lourdes (Lot 4)** : docling/MinerU tirent torch & co. → **isoler en subprocess/service Docker**, ne pas alourdir l'image FastAPI principale. Risque perf CPU (prévoir GPU optionnel).
- **Qualité FR variable** selon moteur → prévoir un **spot-check FR** (1 article + 1 livre) par fournisseur candidat avant adoption.
- **Contrat de sortie** : tous les providers DOIVENT émettre `<!-- Page N -->` + markdown, pagination **continue** (réutiliser la renumérotation des parts) pour ne pas casser chunking ni `book_note_generator` (qui détecte coarse pagination, etc.).
- **Tests** : ajouter `tests/test_ocr_providers.py` (sélection/chaîne, partial flag, concat lots, normalisation pagination) sur le modèle des tests existants.

---

## 9. Définition de « terminé » (DoD du sprint)

- [ ] Un livre OCRise **complètement** même si Mistral est KO (via **OCR local**), pagination continue.
- [ ] Un OCR incomplet n'est **jamais** `success` silencieux (flag `partial` + errors.json + UI).
- [ ] Ordre des fournisseurs configurable par env, rétro-compatible par défaut.
- [ ] Voie **100 % locale** documentée et testée (sans aucune clé API).
- [ ] 401 Mistral → message actionnable (plafond mensuel vs clé).
- [ ] Tests verts + doc à jour (`.claude/CLAUDE.md` section OCR + `.env.example`).
