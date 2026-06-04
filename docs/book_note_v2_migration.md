# Migration v2 du mode [LIVRE]

**Date** : 2026-05-05
**Auteur** : Claude Code (sprint `.claude/tasks/SPRINT_book_note_v2.md`)
**Sources de référence** :
- Spec produit : `.claude/tasks/PROMPT_Lecture_Livre_Boucle_Automatique.md` v2 (procédure interactive validée par l'usage sur Laclau, *On Populist Reason*).
- Plan d'audit : `.claude/tasks/CLAUDE_CODE_PLAN_book_note_v2.md` (13 écarts identifiés entre v1 déployée et v2 master).

---

## 1. Pourquoi la v2

L'exécution réelle de la procédure interactive sur un livre académique dense (88 000 mots, 11 unités) a fait émerger 13 améliorations de prompt et de pipeline. La v1 du pipeline orchestré (FastAPI + multi-phase) avait dérivé vers des défauts stylistiques (formulaire répétitif, marqueurs ordinaux, formules transitoires de clôture) et ignorait des cas pratiques (paratextes longs, paratextes à exclure, EPUB sans pagination, chapitres très longs). La v2 aligne le pipeline sur la procédure validée.

---

## 2. Tableau v1 vs v2

| Aspect | v1 (avant 2026-05-05) | v2 (à partir de 2026-05-05) |
|--------|------------------------|-------------------------------|
| Détection format source | Implicite, repose sur `<!-- Page N -->` injectés par OCR | `_detect_source_format()` : PDF natif / PDF OCR dégradé / EPUB / texte brut. `_normalize_pagination()` convertit `<a id="page_X"/>` (EPUB) → `<!-- Page X -->` |
| Rubrique « Place dans le livre » | Obligatoire dans chaque fiche-chapitre, ≥1 cross-référence systématique | **Supprimée**. Cross-références portées intégralement par la Section B |
| Section B — Tensions | « ≥1 tension » | **« ≥3 tensions/dialogues »** avec citations explicites. Vérification post-assemblage : ≥6 références à des chapitres |
| Paratextes de fin (Notes, Index, Bibliographie) | Analysés comme chapitres normaux ou inclus selon les hasards de la ToC | Champ `to_exclude` en Phase 1 → filtrés avant la boucle Phase 2, log explicite |
| Paratextes longs (Préface ≥ 5k mots, Concluding Remarks…) | Template léger inadéquat (perd la densité conceptuelle) | Champ `is_long_paratext` + template hybride (1100-1500 mots, conserve Concepts/Résultats/Citations/Limites) |
| Volume cible des fiches-chapitres | 900-1300 mots (rigide) | **900-1300 standard ; jusqu'à 1800 pour chapitre théorique central ; 600-900 paratexte court ; 1100-1500 paratexte long** |
| Marqueurs ordinaux internes (« Premier — », « Deuxièmement ») | Tolérés implicitement | **INTERDIT ABSOLU**. Détection post-assemblage avec tolérance 2 (citations verbatim possibles) |
| Formules transitoires de clôture (« Le concept articule X, Y et Z ») | Tolérées implicitement | **INTERDIT ABSOLU**. Détection post-assemblage avec tolérance 2 |
| Mémoire incrémentale (PREVIOUS_SUMMARIES) | Concaténation simple, croissance linéaire | `_build_cumulative_summaries()` : borné à `MAX_PREVIOUS_SUMMARIES_CHARS=4000` (3 premiers + N derniers + marqueur d'omission) |
| Troncature des chapitres très longs | `_truncate_chapter()` brute (perd la conclusion) | `_smart_truncate_chapter()` : préserve intro + sections nommées + conclusion avec marqueurs explicites. Fallback head+tail si aucune section détectée. Fallback final vers la troncature brute (`_truncate_chapter_legacy`) en cas d'exception |
| Seuils opérationnels | Aucun | `THRESHOLD_MAX_CHAPTERS=12`, `THRESHOLD_MAX_TOTAL_WORDS=80000`, `THRESHOLD_MAX_CHAPTER_WORDS=30000`. Franchissement → log warning + `progress_cb("threshold_warning", …)` |
| Vérifications post-assemblage | Sentinel + `[LIVRE]` + comptage `<h3>Chapitre` + sections obligatoires | + Absence rubrique « Place dans le livre », ≥6 réfs Section B, interdictions stylistiques, détection format source dégradé |
| Drapeau de version | Aucun | `BOOK_NOTE_VERSION=v1\|v2` (défaut `v2`). `v1` désactive les vérifications v2 pour rétrocompatibilité |

---

## 3. Stratégie de drapeau de compatibilité

```bash
# Comportement actuel (défaut)
BOOK_NOTE_VERSION=v2
# Rollback en cas de régression
BOOK_NOTE_VERSION=v1
```

- **`v2` (défaut)** : tout le code v1 reste en place ; les nouvelles vérifications post-assemblage et les nouveaux warnings sont actifs.
- **`v1`** : les vérifications v2 sont skippées (`if BOOK_NOTE_VERSION == "v2": …`). Le prompt source `book_prompt.md` est **toujours** la version v2 — la rétrocompatibilité concerne uniquement le pipeline Python, pas le prompt. Si une régression majeure de qualité est observée, le rollback complet exige aussi de réverter `book_prompt.md` via Git.

---

## 4. Changelog par lot

| Lot | Fichier(s) | Description |
|-----|------------|-------------|
| L1 | `app/utils/book_prompt.md` | Refonte v2 : suppression « Place dans le livre », ajout `is_long_paratext` + `to_exclude` + `CHAPTER_TREATMENT`, volumes assouplis, interdictions stylistiques renforcées, Section B ≥3 tensions, mention troncature intelligente |
| L2 | `app/utils/book_note_generator.py` | `SourceFormat` enum, `_detect_source_format()`, `_normalize_pagination()`, intégration en Phase 0a |
| L3 | `app/utils/book_note_generator.py` | Constantes `THRESHOLD_*`, extension `Chapter` (`is_long_paratext`, `to_exclude`, `treatment`, `word_count`), `_merge_llm_structure` recalcule `is_long_paratext`, filtrage `to_exclude` + check seuils dans `build_book_note_async` |
| L4 | `app/utils/book_note_generator.py` | `_detect_sections()`, `_smart_truncate_chapter()`, conservation `_truncate_chapter_legacy` |
| L5 | `app/utils/book_note_generator.py` | `_build_cumulative_summaries()`, placeholder `CHAPTER_TREATMENT` dans `_phase2_analyse_chapter`, vérification post-volume |
| L6 | `app/utils/book_note_generator.py` | `BOOK_NOTE_VERSION` env, vérifications post-assemblage v2 (Place dans le livre, ≥6 réfs, ordinaux, formules transitoires, hints format source) |
| L7 | `tests/test_book_note_generator_v2.py` | 31 tests unitaires (format detection, normalisation, treatment, smart truncate, cumulative summaries, post-assembly checks, prompt structure) |
| L8 | `.claude/tasks/book_reading_notes.md` + `docs/book_note_v2_migration.md` | Documentation migration |

---

## 5. Hors scope sprint v2 (backlog dérivé)

- **TKT-NEXT-1** : estimation pré-Phase 2 du coût LLM et du temps (écart 11 du plan d'audit). Dépend d'un fichier `app/config/llm_pricing.yaml`.
- **TKT-NEXT-2** : table `MODEL_CONTEXT_WINDOWS` dans `llm_note_generator.py` + `ContextWindowExceededError` (écart 12). Permet une détection proactive des dépassements sur livres très longs.
- **TKT-NEXT-3** : exposer `thresholds_exceeded` + estimation à l'UI pour confirmation utilisateur avant run coûteux.
- **TKT-NEXT-4** : tests e2e avec mock LLM déterministe (livre 3-5 chapitres, fixtures HTML pré-écrites). Reportés hors sprint pour ne pas bloquer la livraison.
- **TKT-NEXT-5** : injection dynamique des métadonnées chunk dans `rad_chunk.py` et les connecteurs vector DB (problème hardcodé identifié dans CLAUDE.md, hors scope livre mais voisin).

---

## 6. Comment vérifier le déploiement v2

```bash
# Tests unitaires v2
docker compose exec ragpy pytest tests/test_book_note_generator_v2.py -v
# → 31 passed

# Smoke test imports
docker compose exec ragpy python -c "
from app.utils.book_note_generator import (
    SourceFormat, _detect_source_format, _smart_truncate_chapter,
    _build_cumulative_summaries, BOOK_NOTE_VERSION,
    THRESHOLD_MAX_CHAPTERS, THRESHOLD_MAX_TOTAL_WORDS,
)
print('v2 OK, BOOK_NOTE_VERSION =', BOOK_NOTE_VERSION)
"

# Test sur livre court de référence (PDF dans sources/)
# Lancer une fiche [LIVRE] depuis l'UI ou via l'API, vérifier :
#   - 0 occurrence de « Place dans le livre » dans les fiches-chapitres
#   - Section B avec ≥3 tensions explicitement nommées et ≥6 références à des chapitres
#   - 0 marqueur ordinal interdit (« Premier — », « Deuxièmement »)
#   - 0 formule transitoire de clôture interdite (« Le concept articule X, Y et Z »)
#   - Volume des fiches dans les bornes (900-1300 standard, 600-900 paratexte court)
#   - Citations verbatim avec pages
#   - Concepts mobilisés en prose continue, sans labels visibles
```

---

## 7. Rollback en cas de régression

```bash
# Rollback rapide du pipeline (prompt v2 conservé)
docker compose exec ragpy bash -c 'echo "BOOK_NOTE_VERSION=v1" >> /app/.env'
docker compose restart ragpy

# Rollback complet (prompt v1 + pipeline v1) — nécessite Git
git revert <commit-sprint-v2>
docker compose up -d --build
```
