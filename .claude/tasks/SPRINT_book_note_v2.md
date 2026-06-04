# SPRINT — Migration `book_note_generator` v2

> **Auteur** : Claude Code (instance locale, repo `ragpy`)
> **Date** : 2026-05-05
> **Sources** : `CLAUDE_CODE_PLAN_book_note_v2.md` (audit 13 écarts) + `PROMPT_Lecture_Livre_Boucle_Automatique.md` v2 (spec produit)
> **Durée estimée** : ~3h30 de code + ~1h30 de tests/QA + 30 min de doc → **1 sprint d'une journée**
> **Objectif unique** : aligner `app/utils/book_note_generator.py` + `app/utils/book_prompt.md` sur la spec v2 du prompt interactif master.

---

## 1. Pourquoi ce sprint

L'exécution réelle de la procédure interactive sur Laclau (*On Populist Reason*, 88 000 mots, 11 unités) a permis d'écrire la version master `PROMPT_Lecture_Livre_Boucle_Automatique.md` v2. Cette version contient **13 améliorations validées par l'usage** que le pipeline orchestré (FastAPI + multi-phase) n'a pas encore intégrées. Ce sprint synchronise le pipeline orchestré sur la spec interactive validée, sans rupture de contrat API.

**Critère de succès global** : pour un même livre de référence, la fiche v2 produite par le pipeline orchestré est de qualité au moins équivalente à celle de la procédure interactive v2, et corrige les régressions stylistiques observées en v1 (formulaire répétitif, marqueurs ordinaux, formules transitoires de clôture).

---

## 2. Périmètre

### Inclus

- `app/utils/book_prompt.md` — refonte intégrale (3 phases).
- `app/utils/book_note_generator.py` — détection format, seuils, exclusion paratextes, troncature intelligente, mémoire bornée, vérifications post-assemblage, drapeau `BOOK_NOTE_VERSION`.
- `tests/utils/test_book_note_generator_v2.py` — nouvelle suite (≥85 % couverture sur `book_note_generator.py`).
- Doc : `.claude/tasks/book_reading_notes.md` (mise à jour) + `docs/book_note_v2_migration.md` (création).

### Exclus (hors sprint, à reporter)

- Estimation de coût LLM (écart 11) → ticket future à raccrocher à la roadmap pricing/pricing.yaml.
- Vérification context-window (écart 12) → ticket future, dépend d'un table central des modèles (`MODEL_CONTEXT_WINDOWS`) à instancier dans `llm_note_generator.py`.
- UI : pas de modification de `templates/index.html` ni des routes `/process_dataframe` (le contrat reste identique).
- Mode Celery : aucune modification de `app/tasks/`.

---

## 3. Architecture cible (rappel)

```
build_book_note_async(metadata, text_content, *, model, ...)
  │
  ├─ Phase 0 (no LLM) — NEW
  │     ├─ _detect_source_format()  → SourceFormat enum
  │     ├─ _normalize_pagination()  → marqueurs unifiés <!-- Page N -->
  │     └─ _split_text_by_pages()
  │
  ├─ Phase 1 (1 LLM call)
  │     ├─ _phase1_detect_structure() — NEW champs is_long_paratext, to_exclude
  │     ├─ _merge_llm_structure() — recalcul is_long_paratext depuis word_count
  │     ├─ Filtrage to_exclude
  │     └─ Vérification seuils (12 chap / 80k mots / 30k chap) → warnings
  │
  ├─ Phase 2 (N LLM calls)
  │     ├─ _smart_truncate_chapter() — NEW (préserve intro + sections + conclusion)
  │     ├─ _build_cumulative_summaries() — NEW (borne à 4000 chars)
  │     ├─ Placeholder CHAPTER_TREATMENT (chapter | paratext_short | paratext_long)
  │     └─ Vérif post-bloc : volume vs target par treatment
  │
  ├─ Phase 3 (1 LLM call)
  │     └─ Section B exige ≥3 tensions/dialogues
  │
  └─ Phase 4 (no LLM) — vérifications renforcées
        ├─ Sentinel + [LIVRE]
        ├─ Présence Section A / Section B / N <h3>Chapitre
        ├─ NEW : absence de « Place dans le livre »
        ├─ NEW : ≥6 références à des chapitres dans Section B
        └─ NEW : interdictions stylistiques (ordinaux, formules transitoires)
```

---

## 4. Découpage en lots (tickets)

Chaque lot est un **commit séparé**. Si un test échoue → on n'enchaîne pas.

| # | Lot | Fichier(s) | LOC ± | Risque | Durée |
|---|-----|-----------|-------|--------|-------|
| L1 | Refonte `book_prompt.md` v2 | `app/utils/book_prompt.md` | +/- 80 | Faible | 20 min |
| L2 | Phase 0 : détection format + normalisation pagination | `book_note_generator.py` | +120 | Faible | 30 min |
| L3 | Phase 1 : seuils + exclusion paratextes + `is_long_paratext` | `book_note_generator.py` | +60 | Faible | 25 min |
| L4 | Troncature intelligente par sections | `book_note_generator.py` | +130 | Moyen | 45 min |
| L5 | Phase 2 : `CHAPTER_TREATMENT` + mémoire bornée | `book_note_generator.py` | +80 | Faible | 30 min |
| L6 | Phase 4 : vérifications post-assemblage v2 | `book_note_generator.py` | +50 | Faible | 20 min |
| L7 | Tests v2 + e2e mock | `tests/utils/test_book_note_generator_v2.py` (nouveau) + `tests/utils/test_book_note_e2e.py` (nouveau) | +400 | Faible | 50 min |
| L8 | Doc : `.claude/tasks/book_reading_notes.md` + `docs/book_note_v2_migration.md` | 2 fichiers MD | +200 | Nul | 30 min |
| L9 | QA manuelle + comparaison v1/v2 sur livre court de référence | (pas de code) | — | — | 30 min |

**Total** : ~1190 LOC + 200 lignes de doc, ~4h30.

---

## 5. Détail des lots

### L1 — Refonte `book_prompt.md` v2

**Modifs Phase 1** (lignes ~99-126) :
- Ajouter `is_long_paratext: bool` et `to_exclude: bool` au schéma JSON de chaque chapitre.
- Ajouter, après les instructions existantes : « Marque `to_exclude=true` pour Notes / Index / Bibliographie / Glossaire / Annexes documentaires sans argumentation propre. Marque `is_long_paratext=true` si paratexte ≥ 5 000 mots. »

**Modifs Phase 2** :
- **Supprimer** la rubrique « Place dans le livre » (lignes ~289-294).
- **Ajouter** placeholder `{CHAPTER_TREATMENT}` à la liste, et conditionnement à 3 branches dans la définition des rubriques.
- **Assouplir** les volumes : « 900-1300 standard ; jusqu'à 1800 chap. théorique central ; 600-900 paratexte court ; 1100-1500 paratexte long ».
- **Renforcer** « INTERDIT ABSOLU » avec deux nouvelles règles :
  - Pas de marqueurs ordinaux internes (« Premier », « Deuxièmement », « D'abord », « Ensuite »).
  - Pas de formule transitoire de clôture (« Le concept articule X, Y et Z », « Le concept fournit le critère »).
- **Renommer** « CHAPITRES DÉJÀ ANALYSÉS » en « CONNAISSANCE CUMULATIVE DU LIVRE » + reformuler le rôle.
- **Ajouter** une mention sur la troncature intelligente (les marqueurs `[... section X intermédiaire omise ...]` doivent être traités correctement par le LLM).

**Modifs Phase 3** :
- Section B : Tensions internes → exiger ≥3 tensions/dialogues (au lieu de ≥1), avec citations explicites « Le ch.X (auteur) défend Y, tandis que ch.Z (auteur) soutient le contraire ».

**Modifs Contraintes globales** (ligne ~487) :
- Supprimer l'obligation de cross-référence dans chaque fiche-chapitre. Préciser que la fonction est portée par la Section B.

**Pièges** : ne pas casser les marqueurs `=== PHASE N ===` (utilisés par `_extract_phase_prompt()`). Vérifier que tous les placeholders existants (`{TOC_RAW}`, `{FIRST_PAGES}`, `{PREVIOUS_SUMMARIES}`, etc.) restent valides.

**DoD** :
- [ ] `_load_book_prompt()` charge sans erreur.
- [ ] `_extract_phase_prompt(1|2|3)` retournent chacune un texte non vide.
- [ ] `grep "Place dans le livre" book_prompt.md` → 0 occurrence.
- [ ] `grep "INTERDIT ABSOLU" book_prompt.md` → bloc présent avec ≥2 nouvelles règles ordinaux/formules.

---

### L2 — Phase 0 : détection format + normalisation pagination

**Ajouts dans `book_note_generator.py`** :

```python
from enum import Enum

class SourceFormat(str, Enum):
    PDF_NATIVE = "pdf_native"
    PDF_OCR = "pdf_ocr"
    EPUB = "epub"
    PLAIN_TEXT = "plain_text"

_EPUB_PAGE_MARKER_RE = re.compile(r'<a\s+id="page_([0-9ivxlcdm]+)"\s*/?>', re.IGNORECASE)
_OCR_DEGRADATION_PATTERNS = [
    re.compile(r"[^\w\s]{5,}"),
    re.compile(r"\b[a-z]\s[a-z]\s[a-z]\b"),
]

def _detect_source_format(text: str) -> SourceFormat: ...
def _normalize_pagination(text: str, source_format: SourceFormat) -> str: ...
```

**Intégration dans `build_book_note_async`** (avant `_split_text_by_pages`) :

```python
source_format = _detect_source_format(text_content)
logger.info("Phase 0: source format detected = %s", source_format.value)
text_content = _normalize_pagination(text_content, source_format)
if progress_cb:
    await progress_cb("format_detection", 1, 1, f"Format : {source_format.value}")
```

**Avertissements assemblage** :
- `PLAIN_TEXT` → warning explicite « pagination absente, références par section ».
- `PDF_OCR` → info « OCR potentiellement dégradé, citations à vérifier ».

**DoD** : tests `test_format_detection_*` (4 cas) + `test_normalize_pagination_epub_to_pipeline` passent.

---

### L3 — Phase 1 : seuils + exclusion paratextes + `is_long_paratext`

**Constantes** :
```python
THRESHOLD_MAX_CHAPTERS = 12
THRESHOLD_MAX_TOTAL_WORDS = 80_000
THRESHOLD_MAX_CHAPTER_WORDS = 30_000
THRESHOLD_LONG_PARATEXT_WORDS = 5_000
MAX_PREVIOUS_SUMMARIES_CHARS = 4_000
```

**Dataclass `Chapter`** :
- Ajouter `is_long_paratext: bool = False` et `to_exclude: bool = False`.
- Ajouter `@property treatment` retournant `"chapter" | "paratext_short" | "paratext_long"`.
- Ajouter `@property word_count`.

**`_merge_llm_structure`** :
- Lire les nouveaux champs avec défaut `False`.
- Recalculer `is_long_paratext` à partir du texte effectif : `is_paratext AND word_count >= THRESHOLD_LONG_PARATEXT_WORDS`.

**Dans `build_book_note_async`** :
- Filtrer `to_exclude=True` avant la boucle Phase 2 (avec log explicite).
- Calculer `total_words`, `max_chapter_words`, `len(chapters)` ; logger un `WARNING` et émettre `progress_cb("threshold_warning", ...)` si seuil franchi.

**DoD** : tests `test_threshold_warning_*`, `test_excluded_paratext_skipped`, `test_long_paratext_recomputed_from_text` passent.

---

### L4 — Troncature intelligente par sections

**Nouveau** : `_smart_truncate_chapter(text, max_chars, target_words=15_000)` qui :
1. Si `word_count <= target_words` → retourne le texte (filet de sécurité `max_chars`).
2. Sinon détecte les sections (regex sous-titres CAPS / numérotés / Markdown / titre-case isolé), avec déduplication < 50 chars.
3. Si ≥2 sections détectées → conserve intro + (titre + ~3 paragraphes) × N + conclusion, avec marqueurs `[... section « X » développement intermédiaire omis ...]`.
4. Sinon fallback début+fin avec marqueur explicite.
5. Filet final si dépassement `max_chars`.

**Conserver** `_truncate_chapter_legacy` (rétrocompatibilité, fallback si exception dans `_smart_truncate_chapter`).

**Substitution** dans `_phase2_analyse_chapter` ligne ~558 :
```python
"CHAPTER_TEXT": _smart_truncate_chapter(chapter.text, PHASE2_CHAPTER_MAX_CHARS),
```

**DoD** : tests `test_smart_truncate_short_chapter_unchanged`, `test_smart_truncate_long_chapter_with_sections`, `test_smart_truncate_long_chapter_without_sections`, `test_smart_truncate_respects_max_chars` passent.

---

### L5 — Phase 2 : `CHAPTER_TREATMENT` + mémoire bornée

**Nouveau** : `_build_cumulative_summaries(summaries, max_chars=MAX_PREVIOUS_SUMMARIES_CHARS)` :
- Si concaténation ≤ `max_chars` → join simple.
- Sinon : 3 premiers + autant des derniers que possible + marqueur d'omission.
- Cas extrême : tronquer le head si même 0 tail ne tient pas.

**Dans `_phase2_analyse_chapter`** :
- Substituer `PREVIOUS_SUMMARIES` par `_build_cumulative_summaries(previous_summaries)`.
- Ajouter `"CHAPTER_TREATMENT": chapter.treatment` dans `_fill_placeholders`.
- Après réception du bloc HTML, compter les mots et logger un warning si > target + 200 (avec target dépendant de `treatment`).

**Vérifier** que `PHASE_MAX_TOKENS["phase2"]` est cohérent avec 1800 mots max → bumper à 8000 si nécessaire.

**DoD** : tests `test_cumulative_summaries_*`, `test_phase2_passes_treatment_to_prompt` passent.

---

### L6 — Phase 4 : vérifications post-assemblage v2

**Dans `_assemble_book_html`**, après les vérifications existantes :

```python
# v2 — Suppression « Place dans le livre »
if "Place dans le livre" in chapters_html:
    warnings_list.append("rubrique « Place dans le livre » résiduelle (devrait être absente en v2)")

# v2 — ≥3 tensions Section B → heuristique ≥6 réfs « ch.X » / « chapitre X »
section_b_str = section_b or ""
total_chapter_refs = len(re.findall(r"\b(?:ch\.|chapitre)\s*\d+", section_b_str, re.IGNORECASE))
if len(structure.chapters) >= 6 and total_chapter_refs < 6:
    warnings_list.append(f"Section B contient {total_chapter_refs} références à des chapitres (seuil v2: ≥6)")

# v2 — Interdictions stylistiques (tolérance 1-2 occurrences)
forbidden_v2 = [
    (r"\b(Premier|Premièrement|Deuxième|Deuxièmement|Troisième|Troisièmement)\s*[—:.]", "marqueur ordinal"),
    (r"\bLe concept articule\b", "formule « Le concept articule »"),
    (r"\bLe concept fournit\b", "formule « Le concept fournit »"),
    (r"\bLe concept opère\b", "formule « Le concept opère »"),
]
for pattern, label in forbidden_v2:
    n = len(re.findall(pattern, chapters_html))
    if n > 2:  # tolérance 2 (citation verbatim possible)
        warnings_list.append(f"{n} occurrences de {label} (interdit v2)")
```

**Drapeau de compatibilité** :
```python
BOOK_NOTE_VERSION = os.getenv("BOOK_NOTE_VERSION", "v2").lower()
```
Conditionner les nouvelles vérifications par `if BOOK_NOTE_VERSION == "v2": ...`.

**DoD** : tests `test_section_b_three_tensions_check`, `test_forbidden_patterns_detected_v2`, `test_v1_flag_skips_v2_checks` passent.

---

### L7 — Tests v2 + e2e mock

Créer `tests/utils/test_book_note_generator_v2.py` avec :

| Test | Vérifie |
|------|---------|
| `test_format_detection_pdf_native` | Marqueurs `<!-- Page N -->` propres → `PDF_NATIVE` |
| `test_format_detection_pdf_ocr_degraded` | Pagination + artefacts → `PDF_OCR` |
| `test_format_detection_epub` | Marqueurs `<a id="page_X"/>` → `EPUB` |
| `test_format_detection_plain_text` | Aucun marqueur → `PLAIN_TEXT` |
| `test_normalize_pagination_epub_to_pipeline` | EPUB → marqueurs unifiés |
| `test_threshold_warning_too_many_chapters` | 15 chapitres → log warning |
| `test_threshold_warning_long_chapter` | Chapitre 35k mots → warning |
| `test_excluded_paratext_skipped` | `to_exclude=True` → absent boucle Phase 2 |
| `test_long_paratext_recomputed_from_text` | Paratexte 7k mots → `is_long_paratext=True` |
| `test_smart_truncate_short_chapter_unchanged` | Chapitre 5k mots → texte intégral |
| `test_smart_truncate_long_chapter_with_sections` | 25k mots + 5 sous-titres → intro + sections + conclusion |
| `test_smart_truncate_long_chapter_without_sections` | 25k mots, aucune section → fallback début+fin |
| `test_smart_truncate_respects_max_chars` | 200k chars → ≤ 80k chars |
| `test_cumulative_summaries_under_limit_unchanged` | 3 résumés courts → join simple |
| `test_cumulative_summaries_over_limit_truncated` | 15 résumés totalisant 6k chars → 3 premiers + N derniers + marqueur |
| `test_phase2_passes_treatment_to_prompt` | `is_long_paratext=True` → `CHAPTER_TREATMENT=paratext_long` dans prompt |
| `test_section_b_three_tensions_check` | 1 tension → warning post-assemblage |
| `test_forbidden_patterns_detected_v2` | 3 « Premier — » dans HTML → warning |
| `test_v1_flag_skips_v2_checks` | `BOOK_NOTE_VERSION=v1` → pas de vérif v2 |

Créer `tests/utils/test_book_note_e2e.py` avec :
- `test_e2e_build_book_note_v2` (marqué `@pytest.mark.slow`) : mock LLM déterministe (fixtures HTML), livre 3-5 chapitres, vérifie sentinel + `[LIVRE]` + N `<h3>Chapitre` + Section A + Section B + 0 warning critique.

**DoD** :
- [ ] `pytest tests/utils/test_book_note_generator_v2.py` → tout passe.
- [ ] `pytest --cov=app.utils.book_note_generator tests/utils/test_book_note_generator_v2.py` → couverture ≥85 %.

---

### L8 — Documentation

**Mise à jour** `.claude/tasks/book_reading_notes.md` :
- Ajouter section « Migration v2 » référençant `CLAUDE_CODE_PLAN_book_note_v2.md`.
- Mettre à jour le diagramme de pipeline avec Phase 0.

**Création** `docs/book_note_v2_migration.md` :
- Tableau v1 vs v2 (les 13 écarts résolus).
- Stratégie de migration (drapeau `BOOK_NOTE_VERSION`, déploiement en 2 étapes).
- Comment lancer le test de régression v1/v2.
- Annexe : changelog par lot.

**DoD** : les deux fichiers existent et sont relus (cohérence avec le code livré).

---

### L9 — QA manuelle + comparaison v1/v2

**Procédure** :
1. Choisir un livre court de référence (3-5 chapitres, ~30 000 mots) déjà traité en v1 (PDF dans `sources/`).
2. Lancer le pipeline avec `BOOK_NOTE_VERSION=v1` → conserver la sortie comme baseline.
3. Lancer le pipeline avec `BOOK_NOTE_VERSION=v2` (défaut).
4. Diff manuel des deux fiches HTML :
   - [ ] v2 a la rubrique « Place dans le livre » absente.
   - [ ] v2 a Section B avec ≥3 tensions explicitement nommées et ≥6 références à des chapitres.
   - [ ] v2 a 0 occurrence des marqueurs ordinaux interdits (« Premier — », « Deuxièmement », etc.).
   - [ ] v2 a 0 formule transitoire de clôture interdite.
   - [ ] Volume des fiches dans la cible (900-1300 standard, 600-900 paratexte court).
   - [ ] Citations verbatim avec pages présentes.
   - [ ] Concepts mobilisés en prose continue, sans labels visibles.

**Bonus si temps** : lancer sur un livre long (10+ chap, 80k+ mots) et vérifier que le `threshold_warning` est bien émis et que la fiche reste utilisable.

**DoD** : checklist ci-dessus tous cochés ; capturer un avant/après dans `docs/book_note_v2_migration.md` (extrait de 2-3 paragraphes-concepts).

---

## 6. Ordre d'exécution chronologique

```
J1 matin  : L1 (prompt) → commit "feat(book): refonte prompt v2"
            L2 (Phase 0) → tests format → commit "feat(book): détection format + normalisation pagination"
            L3 (seuils) → tests seuils → commit "feat(book): seuils + exclusion paratextes"

J1 après-midi : L4 (smart truncate) → tests truncate → commit "feat(book): troncature intelligente par sections"
                L5 (treatment + mémoire) → tests phase2 → commit "feat(book): CHAPTER_TREATMENT + mémoire bornée"
                L6 (vérifs post-assemblage) → tests vérifs → commit "feat(book): vérifications post-assemblage v2"

J1 fin de journée : L7 (e2e mock + couverture) → commit "test(book): suite v2 + e2e mock LLM"
                    L8 (doc) → commit "docs(book): migration v2"
                    L9 (QA manuelle livre court) → si OK → commit "docs(book): comparaison v1/v2 livre court"

J2 matin (optionnel) : QA livre long + vérification SSE/UI inchangés
                       → tag git v2-book-note-released
```

**Règle absolue** : un test rouge bloque le commit. Investiguer avant de continuer.

---

## 7. Risques et mitigation

| Risque | P | I | Mitigation |
|--------|---|---|------------|
| Détection format trop laxe (PDF dégradé classé natif) | M | F | Seuil `degradation_score > 50` ajustable, pas de blocage |
| Détection sections trop laxe (faux positifs) | É | M | Déduplication < 50 chars + test fixture explicite |
| Mémoire bornée perd un résumé crucial | F | M | Stratégie 3-premiers + N-derniers conserve cadre + récent |
| Régression silencieuse qualité fiches | M | É | L9 obligatoire avec diff manuel sur livre référence |
| Le LLM ignore les nouvelles règles du prompt | É | F | Vérifications post-assemblage L6 capturent et signalent |
| Coût LLM accru par contexte | M | F | Bornage à 4k chars (L5) plafonne la croissance |
| Test e2e instable (mock LLM non déterministe) | F | M | Mock retourne fixtures HTML pré-écrites, pas d'appel réseau |
| `BOOK_NOTE_VERSION=v1` casse une route existante | F | É | Tous les ajouts conditionnés ; v1 = comportement actuel inchangé |

---

## 8. Critères d'acceptation finaux (DoD sprint)

1. ✅ Tous les tests unitaires passent (`pytest tests/utils/test_book_note_generator_v2.py`).
2. ✅ Couverture ≥ 85 % sur `app/utils/book_note_generator.py`.
3. ✅ Test e2e mock passe avec sentinel + `[LIVRE]` + N chapitres + Section A + Section B, 0 warning critique.
4. ✅ Sur livre court de référence, fiche v2 manuellement validée (checklist L9).
5. ✅ `grep "Place dans le livre" sur sortie v2` → 0 occurrence.
6. ✅ Section B avec ≥3 tensions identifiées explicitement.
7. ✅ 0 marqueur ordinal interdit, 0 formule transitoire de clôture interdite (tolérance 2).
8. ✅ Paratextes de fin (Notes / Index / Bibliographie) filtrés et absents de la fiche.
9. ✅ `BOOK_NOTE_VERSION=v1` produit toujours la fiche v1 (rétrocompatibilité OK).
10. ✅ Doc à jour : `book_reading_notes.md` + `book_note_v2_migration.md`.
11. ✅ 9 commits Git séparés (un par lot), tags `v2-book-note-released` posé.

---

## 9. Référence rapide — fichiers du chemin [LIVRE]

| Couche | Fichier | Touché par ce sprint |
|--------|---------|---------------------|
| Pipeline livre | `app/utils/book_note_generator.py` | ✅ Lots 2, 3, 4, 5, 6 |
| Pipeline livre | `app/utils/book_prompt.md` | ✅ Lot 1 |
| Pipeline livre | `app/utils/llm_note_generator.py` | ❌ (dispatch déjà OK) |
| Endpoint | `app/routes/processing.py` | ❌ (contrat inchangé) |
| Sécurité | `app/core/credentials.py` | ❌ |
| Push Zotero | `app/utils/zotero_client.py` | ❌ |
| OCR fournisseur | `scripts/rad_dataframe.py` | ❌ |
| UI | `app/templates/index.html` | ❌ |
| Tests | `tests/utils/test_book_note_generator_v2.py` | ✅ Lot 7 (création) |
| Tests | `tests/utils/test_book_note_e2e.py` | ✅ Lot 7 (création) |
| Doc | `.claude/tasks/book_reading_notes.md` | ✅ Lot 8 |
| Doc | `docs/book_note_v2_migration.md` | ✅ Lot 8 (création) |

---

## 10. Hors-sprint (backlog dérivé)

À planifier après ce sprint, dans des tickets séparés :

- **TKT-NEXT-1** : estimation pré-Phase 2 du coût et du temps (écart 11). Dépendance : `app/config/llm_pricing.yaml`.
- **TKT-NEXT-2** : table `MODEL_CONTEXT_WINDOWS` dans `llm_note_generator.py` + `ContextWindowExceededError` (écart 12).
- **TKT-NEXT-3** : génération automatique de `PROMPT_Lecture_Livre_Boucle_Automatique_v2.md` à partir de `book_prompt.md` (écart 10 optionnel).
- **TKT-NEXT-4** : injection dynamique des métadonnées chunk (problème hardcodé identifié dans CLAUDE.md, hors scope livre).
- **TKT-NEXT-5** : exposer `thresholds_exceeded` + estimation à l'UI pour confirmation utilisateur avant run coûteux.

---

**Fin du sprint.**
