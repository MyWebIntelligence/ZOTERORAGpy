# Plan : Fiches de lecture de livres [LIVRE]

**Date** : 2026-05-04
**Statut** : ✅ Implémenté + migré v2 (2026-05-05)
**Inspiration** : `.claude/tasks/prompt_template.md` (modes [FICHE]/[CLAIR]/[EVAL])

> **Migration v2 (2026-05-05)** — Le pipeline orchestré (FastAPI + multi-phase) a été aligné sur la version master `PROMPT_Lecture_Livre_Boucle_Automatique.md` v2 (procédure interactive validée par l'usage). Voir `docs/book_note_v2_migration.md` pour le détail des 13 écarts résolus, la stratégie de drapeau `BOOK_NOTE_VERSION` et le tableau v1 vs v2. Le sprint d'exécution est documenté dans `.claude/tasks/SPRINT_book_note_v2.md`.
>
> **Ajouts v2 du pipeline orchestré** :
> - Phase 0 explicite : `_detect_source_format()` + `_normalize_pagination()` (PDF natif / PDF OCR dégradé / EPUB / texte brut).
> - Champs Phase 1 : `is_long_paratext` et `to_exclude` (Notes / Index / Bibliographie filtrés avant la boucle).
> - Seuils opérationnels : 12 chapitres, 80 000 mots, 30 000 mots/chap → warning + `progress_cb("threshold_warning", …)`.
> - Phase 2 : `_smart_truncate_chapter()` (préserve intro + sections + conclusion), `_build_cumulative_summaries()` (borné à 4 000 chars), placeholder `CHAPTER_TREATMENT` (chapter / paratext_short / paratext_long).
> - Phase 4 : vérifications post-assemblage (absence « Place dans le livre », ≥6 réfs chapitres en Section B, interdictions stylistiques).
> - Drapeau `BOOK_NOTE_VERSION=v1|v2` (défaut v2) pour rollback.

---

## 1. Problématique

Les modes existants ([FICHE], [CLAIR], [EVAL], short) sont calibrés pour des **articles scientifiques** (3 000–15 000 mots, un ou plusieurs auteurs co-signataires d'un texte unique). Les **livres** posent des défis spécifiques que ces prompts ne traitent pas :

### 1.1 Hétérogénéité typologique des livres

| Type | Caractéristique | Difficulté analytique |
|------|-----------------|------------------------|
| **Monographie** | 1 auteur ou auteurs co-signataires d'un argument unifié | Continuité de la voix, progression argumentative |
| **Ouvrage collectif (edited volume)** | Chaque chapitre = auteur(s) différent(s), un directeur d'ouvrage | Auteurs multiples, attribution chapitre-par-chapitre obligatoire |
| **Co-écriture** | 2-3 auteurs principaux pour l'ensemble | Hybride entre les deux précédents |
| **Manuel / handbook** | Plusieurs sections thématiques, contributeurs invités | Structure encyclopédique |

### 1.2 Volume du document

- Un article : 3k–15k mots → tient dans **un seul appel LLM** (contexte ≤200k tokens).
- Un livre : 50k–500k mots → **dépasse souvent le contexte LLM** ou produit des résumés trop superficiels en mono-shot.
- Coût : un livre traité en mono-shot avec GPT-4 = ~5–20€. Une analyse chapitre-par-chapitre rigoureuse impose une stratégie multi-appels.

### 1.3 Structure intrinsèquement chapitrée

Un livre est rarement linéaire : il est **un assemblage de chapitres autonomes** organisés par un projet éditorial. Une fiche de lecture utile doit :

1. Rappeler **à chaque chapitre** : numéro, titre, auteur(s), pages
2. Analyser chaque chapitre **en lui-même** : concepts, méthodes, résultats, limites
3. **Repositionner chaque chapitre** dans la dynamique d'ensemble du livre (ce qu'il apporte au projet, dialogue avec les chapitres voisins, tensions internes)

---

## 2. Objectifs

Créer un nouveau mode **`book` [LIVRE]** dans le système de génération Zotero, caractérisé par :

- **Détection automatique de la typologie** (monographie / collectif / co-écriture)
- **Découpage automatique en chapitres** à partir de l'OCR ou de la ToC
- **Analyse chapitre par chapitre** avec attribution explicite des auteurs
- **Synthèse transversale** identifiant lignes de force, tensions, dialogues entre chapitres
- **Pipeline multi-pass** (détection structure → analyse par chapitre → synthèse → assemblage HTML)

---

## 3. Spécifications

| Aspect | Valeur |
|--------|--------|
| **Mode** | `book` |
| **Préfixe note** | `[LIVRE]` |
| **Longueur cible totale** | 5 000–12 000 mots (proportionnel au nombre de chapitres) |
| **Mots/chapitre** | 350–500 mots (varie selon la longueur du chapitre source) |
| **Format sortie** | HTML compatible Zotero (mêmes balises que les autres modes) |
| **Coût estimé** | 0,50–3 € par livre selon taille (vs ~0,10 € pour un article) |
| **Pipeline** | Multi-pass (3 phases LLM + 1 phase d'assemblage) |
| **Modèle recommandé** | `google/gemini-2.5-flash` (contexte 1M tokens, économique) |
| **Fallback** | `gpt-4o-mini` si OpenRouter indisponible |

---

## 4. Architecture multi-phase

```text
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 0 : Préparation (sans LLM)                                  │
│ - Récupérer OCR complet du livre depuis output.csv                │
│ - Récupérer chunks avec métadonnées de page (output_chunks.json)  │
│ - Détecter signaux de chapitrage (titres en CAPS, "Chapitre X",   │
│   "Chapter X", numéros romains, sauts de page, ToC)               │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 1 : Détection structure (1 appel LLM)                       │
│ Input  : ToC + premières pages (~5k tokens)                       │
│ Prompt : SECTION 1 de book_prompt.md                              │
│ Output : JSON                                                     │
│   {                                                               │
│     "book_type": "monograph|edited_volume|coauthored|handbook",   │
│     "primary_authors": [...],                                     │
│     "editor": "...",                                              │
│     "chapters": [                                                 │
│       {"num": 1, "title": "...", "authors": [...],                │
│        "pages": "1-23", "is_intro": true},                        │
│       ...                                                         │
│     ]                                                             │
│   }                                                               │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 2 : Analyse par chapitre (N appels LLM, parallélisables)    │
│ Pour chaque chapitre :                                            │
│   Input  : texte du chapitre + métadonnées livre + ToC complète   │
│            + résumés courts des chapitres déjà analysés           │
│   Prompt : SECTION 2 de book_prompt.md                            │
│   Output : HTML d'une fiche-chapitre (350-500 mots)               │
│                                                                   │
│ ⚠️ Concurrence limitée par MAX_CONCURRENT_LLM_CALLS                 │
│ ⚠️ Chaque appel produit AUSSI un résumé court (50-80 mots)        │
│   pour servir de contexte aux chapitres suivants                  │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 3 : Synthèse transversale (1 appel LLM)                     │
│ Input  : tous les résumés courts + métadonnées livre              │
│ Prompt : SECTION 3 de book_prompt.md                              │
│ Output : HTML pour les sections globales                          │
│   - Architecture du livre                                         │
│   - Lignes de force transversales                                 │
│   - Tensions et dialogues entre chapitres                         │
│   - Évaluation globale + verdict + score                          │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 4 : Assemblage final (sans LLM)                             │
│ Concatène : préfixe [LIVRE] + identification + architecture       │
│             + N fiches-chapitre + synthèse + évaluation           │
│             + sentinel ragpy-note-id                              │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. Structure HTML de la fiche complète

```html
<!-- ragpy-note-id:uuid -->
<h2>[LIVRE] Smith, J. (dir.) (2024). Titre du livre. Éditeur.</h2>

<h3>1. Identification de l'ouvrage</h3>
<table>
  <thead><tr><th>Champ</th><th>Valeur</th></tr></thead>
  <tbody>
    <tr><td>Type</td><td>Ouvrage collectif (12 contributeurs)</td></tr>
    <tr><td>Direction</td><td>Smith, J.</td></tr>
    <tr><td>Éditeur</td><td>Routledge, coll. Critical Theory</td></tr>
    <tr><td>Pagination</td><td>342 pages — 14 chapitres</td></tr>
    <tr><td>Pertinence pour {PROBLEMATIQUE}</td><td>★★★★☆</td></tr>
  </tbody>
</table>

<h3>2. Architecture de l'ouvrage</h3>
<p>Synthèse 3-5 phrases du projet éditorial...</p>
<p>Logique de structuration : 4 parties (Théorie / Méthode / Études de cas / Perspectives)...</p>

<h3>3. Analyse chapitre par chapitre</h3>

<h3>Chapitre 1 — "Introduction" — J. Smith — pp. 1-15</h3>
<p><strong>Argument central</strong> : ...</p>
<p><strong>Concepts</strong> : ...</p>
<p><strong>Méthode/démarche</strong> : ...</p>
<p><strong>Résultats / thèses</strong> : ...</p>
<p><strong>Limites</strong> : ...</p>
<p><strong>Place dans le livre</strong> : Pose le cadre théorique mobilisé par les chapitres 2-7...</p>

<h3>Chapitre 2 — "Titre" — A. Dupont &amp; B. Martin — pp. 17-42</h3>
<!-- même structure -->

<!-- ... un bloc h3+contenu par chapitre ... -->

<h3>4. Synthèse transversale</h3>
<p><strong>Lignes de force</strong> : ...</p>
<p><strong>Tensions internes</strong> : Le chapitre 5 (X) défend Y, contredit par le chapitre 9 (Z)...</p>
<p><strong>Dialogues entre chapitres</strong> : ...</p>

<h3>5. Évaluation et exploitation</h3>
<p><strong>Forces / faiblesses</strong> : ...</p>
<p><strong>Pertinence pour {PROBLEMATIQUE}</strong> : ...</p>
<p><strong>Citations clés</strong> :</p>
<ol>
  <li>"[Verbatim]" (Smith, ch.1, p.12) — Usage : ...</li>
</ol>
<p><strong>Bibliographie à explorer</strong> : ...</p>
```

---

## 6. Détection des chapitres

### 6.1 Stratégie hiérarchique

Tenter dans l'ordre, basculer sur la stratégie suivante en cas d'échec :

1. **ToC explicite** : Si l'OCR contient "Table des matières" / "Contents" / "Sommaire" en début de document, parser cette section avec un mini-prompt LLM pour extraire la liste structurée.
2. **Métadonnées Zotero** : Vérifier `numPages`, champ `tableOfContents` côté Zotero si disponible.
3. **Détection heuristique sur l'OCR** : Regex sur `^Chapitre \d+`, `^Chapter \d+`, `^[IVX]+\.`, titres en CAPS isolés sur une ligne.
4. **Fallback LLM** : En dernier recours, demander au LLM (Phase 1) de proposer un découpage à partir des 10-15 premières pages OCR + table des matières détectée.

### 6.2 Détection des auteurs par chapitre (cas ouvrage collectif)

Pour les ouvrages collectifs, la ToC contient typiquement le format :

```
Chapitre 5 : Titre du chapitre ........ Auteur, A. & Auteur, B. ........ 87
```

Le prompt Phase 1 doit reconnaître ce pattern et extraire `(num, titre, auteurs[], page_début)`.

### 6.3 Cas limites à gérer

- **Pas de ToC détectable** → traiter le livre comme une monographie avec auteur unique = `{AUTHORS}`, découper artificiellement par tranches de 10-15k mots.
- **Chapitres très courts** (< 1500 mots) → fusionner consécutivement avant analyse pour éviter du bruit.
- **Chapitres très longs** (> 30k mots) → sub-chunker en sections internes traitées séparément puis re-fusionner.
- **Préface / postface / introduction** → traiter comme chapitres mais marquer `is_paratext: true` pour synthèse différenciée.

---

## 7. Modifications du code

### 7.1 Fichiers à créer

```text
app/utils/book_prompt.md                  (À CRÉER — prompt 3-en-1)
app/utils/book_note_generator.py          (À CRÉER — orchestrateur multi-phase)
tests/test_book_note_generator.py         (À CRÉER)
tests/fixtures/sample_book_ocr.txt        (À CRÉER — fixture d'un faux livre 3 chapitres)
```

### 7.2 Fichiers à modifier

| Fichier | Modification |
|---------|--------------|
| `app/utils/llm_note_generator.py` | Ajouter mode `"book"` au mapping `_load_prompt_template()` (ou déléguer à `book_note_generator`) |
| `app/routes/citations.py` | Endpoint `/generate_zotero_notes_sse` accepte `note_mode="book"`, dispatch vers `book_note_generator` |
| `app/templates/citations.html` | Ajouter option `📖 Fiche livre [LIVRE]` au dropdown "Type de fiche" |
| `.claude/docs/README_ZOTERO_PROMPT.md` | Documenter le nouveau mode et son architecture multi-pass |

### 7.3 Module `book_note_generator.py` — squelette

```python
"""
Book reading notes generator.

Multi-phase pipeline producing structured book reading notes for Zotero.
Unlike single-shot article generators, books require chapter-level processing
with shared context.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from app.utils.llm_note_generator import _get_llm_clients, _llm_semaphore


@dataclass
class Chapter:
    num: int
    title: str
    authors: list[str]
    pages: tuple[int, int]
    text: str
    is_paratext: bool = False
    short_summary: Optional[str] = None  # Filled in Phase 2


@dataclass
class BookStructure:
    book_type: str  # "monograph" | "edited_volume" | "coauthored" | "handbook"
    primary_authors: list[str]
    editor: Optional[str]
    chapters: list[Chapter]


async def detect_book_structure(
    text_excerpt: str,
    metadata: dict,
    *,
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    model: str = "google/gemini-2.5-flash",
) -> BookStructure:
    """Phase 1: detect book typology and chapter list via LLM."""
    ...


async def analyse_chapter(
    chapter: Chapter,
    book_meta: dict,
    toc_summary: str,
    previous_summaries: list[str],
    *,
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    model: str = "google/gemini-2.5-flash",
) -> tuple[str, str]:
    """Phase 2: analyse one chapter, return (html_block, short_summary)."""
    async with _llm_semaphore:
        ...


async def synthesise_book(
    structure: BookStructure,
    chapter_summaries: list[str],
    book_meta: dict,
    *,
    problematique: str,
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    model: str = "google/gemini-2.5-flash",
) -> str:
    """Phase 3: cross-cutting synthesis + evaluation HTML."""
    ...


async def build_book_note_async(
    full_text: str,
    chunks_with_pages: list[dict],
    metadata: dict,
    *,
    problematique: str = "",
    language: str = "français",
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    model: str = "google/gemini-2.5-flash",
    progress_cb=None,
) -> str:
    """
    Orchestrate the 4 phases and return the complete HTML note.

    Args:
        full_text: Complete OCR text of the book.
        chunks_with_pages: Chunks with page metadata (for chapter slicing).
        metadata: Title, authors, date, DOI, URL.
        problematique: User research project description.
        language: Output language.
        progress_cb: Optional callback(stage_name, current, total) for SSE.

    Returns:
        Complete HTML reading note prefixed with [LIVRE] and sentinel.
    """
    # Phase 1: detect structure
    structure = await detect_book_structure(...)

    # Phase 2: parallel chapter analysis (limited by global semaphore)
    chapter_blocks: list[str] = []
    summaries: list[str] = []
    for chapter in structure.chapters:
        block, summary = await analyse_chapter(
            chapter, metadata, toc_summary, summaries, ...
        )
        chapter_blocks.append(block)
        summaries.append(summary)

    # Phase 3: synthesis
    synthesis_html = await synthesise_book(structure, summaries, metadata, ...)

    # Phase 4: assemble
    return _assemble_book_html(
        prefix="[LIVRE]",
        metadata=metadata,
        structure=structure,
        chapter_blocks=chapter_blocks,
        synthesis_html=synthesis_html,
    )
```

### 7.4 Routes — paramètre API

```text
POST /generate_zotero_notes_sse
{
    "session": "...",
    "note_mode": "book",   # NEW
    "model": "google/gemini-2.5-flash",
    "max_concurrent_chapters": 3   # NEW (optionnel, défaut = MAX_CONCURRENT_LLM_CALLS)
}
```

Le streaming SSE doit émettre des événements granulaires :

```text
event: phase
data: {"phase": "structure_detection", "status": "started"}

event: phase
data: {"phase": "structure_detection", "status": "completed", "chapters": 14}

event: chapter
data: {"num": 1, "title": "Introduction", "status": "completed"}

event: chapter
data: {"num": 2, "title": "...", "status": "started"}

event: phase
data: {"phase": "synthesis", "status": "started"}

event: completed
data: {"note_id": "uuid", "word_count": 8420}
```

---

## 8. Coûts et performances

### 8.1 Estimation pour un livre type (300 pages, 14 chapitres, 100k mots)

| Phase | Appels LLM | Tokens input | Tokens output | Coût Gemini Flash |
|-------|-----------|--------------|---------------|-------------------|
| 1 — Structure | 1 | ~10k | ~2k | ~0,01 € |
| 2 — Chapitres | 14 | ~14×10k = 140k | ~14×800 = 11k | ~0,15 € |
| 3 — Synthèse | 1 | ~5k | ~3k | ~0,01 € |
| **TOTAL** | **16** | ~155k | ~16k | **~0,17 €** |

Comparaison : un mode mono-shot avec GPT-4 sur 100k tokens = ~3-5 €. **Économie 20-30×.**

### 8.2 Garde-fous

- **Limite haute** : refuser livres > 800k mots (livres saints, encyclopédies) sans flag explicite `--force-large`.
- **Échec partiel** : si le chapitre N échoue, garder les N-1 précédents et marquer `<p><em>⚠️ Analyse du chapitre {N} indisponible</em></p>` plutôt qu'avorter tout.
- **Cache** : après Phase 1, sauvegarder `book_structure.json` pour reprendre si Phase 2 échoue.
- **Concurrence** : respecter le sémaphore global `MAX_CONCURRENT_LLM_CALLS` ; ne PAS lancer 14 appels parallèles si la plateforme a 5 utilisateurs concurrents.

---

## 9. Ordre d'implémentation

### Phase 1 : Prompt et logique de détection (priorité haute)

1. [ ] Rédiger `app/utils/book_prompt.md` avec ses 3 sections (structure / chapitre / synthèse)
2. [ ] Implémenter heuristiques de détection ToC dans `book_note_generator.py`
3. [ ] Tester Phase 1 (détection structure) sur 3 livres réels (1 monographie, 1 collectif, 1 manuel)

### Phase 2 : Orchestration backend

4. [ ] Implémenter `analyse_chapter()` avec gestion du contexte incrémental
5. [ ] Implémenter `synthesise_book()` et `_assemble_book_html()`
6. [ ] Implémenter `build_book_note_async()` (orchestrateur)
7. [ ] Ajouter validation HTML (sentinel, préfixe, sections obligatoires)

### Phase 3 : Routes et SSE

8. [ ] Ajouter dispatch `note_mode="book"` dans `citations.py`
9. [ ] Implémenter événements SSE granulaires (phase / chapter / completed)
10. [ ] Vérifier intégration `build_subprocess_env()` pour les credentials

### Phase 4 : UI

11. [ ] Ajouter option dropdown dans `citations.html`
12. [ ] Affichage barre de progression multi-phase (1 barre par phase)
13. [ ] Avertissement coût/temps si livre détecté > 200 pages

### Phase 5 : Tests et documentation

14. [ ] Tests unitaires détection ToC (regex + LLM-mock)
15. [ ] Test e2e sur fixture de livre 3 chapitres
16. [ ] Documentation `.claude/docs/README_ZOTERO_PROMPT.md` (mode `book`)
17. [ ] Documentation `.claude/docs/BOOK_PROMPT_GUIDE.md` (architecture multi-phase)

---

## 10. Validation qualité

### 10.1 Checklist post-génération

```python
def validate_book_note(html: str, structure: BookStructure) -> dict:
    errors, warnings = [], []

    # 1. Sentinel + préfixe
    if "<!-- ragpy-note-id:" not in html[:200]:
        errors.append("Sentinel manquant")
    if "[LIVRE]" not in html[:500]:
        errors.append("Préfixe [LIVRE] manquant")

    # 2. Tous les chapitres présents
    for ch in structure.chapters:
        marker = f"Chapitre {ch.num}"
        if marker not in html and f"Chapter {ch.num}" not in html:
            warnings.append(f"Chapitre {ch.num} ({ch.title}) absent du rendu")

    # 3. Pour chaque chapitre, vérifier les 5 rubriques
    required_subsections = ["Argument", "Concepts", "Méthode", "Résultats", "Place dans le livre"]
    # ... regex par bloc chapitre

    # 4. Sections globales
    for section in ["Identification", "Architecture", "Synthèse transversale", "Évaluation"]:
        if section not in html:
            warnings.append(f"Section globale '{section}' absente")

    # 5. Longueur proportionnelle au nombre de chapitres
    expected_min = 1500 + 350 * len(structure.chapters)
    expected_max = 3000 + 700 * len(structure.chapters)
    word_count = len(html.split())
    if word_count < expected_min:
        warnings.append(f"Note courte ({word_count} mots < {expected_min} attendus)")

    return {"valid": not errors, "errors": errors, "warnings": warnings,
            "word_count": word_count, "chapters_count": len(structure.chapters)}
```

### 10.2 Critères qualitatifs (à évaluer manuellement sur 5 livres pilotes)

- ✅ Chaque chapitre rappelle bien numéro/titre/auteur(s) en en-tête
- ✅ Concepts/méthode/résultats/limites présents pour chaque chapitre
- ✅ "Place dans le livre" cite explicitement d'autres chapitres (≥1 référence croisée par chapitre)
- ✅ Synthèse transversale identifie ≥2 lignes de force ET ≥1 tension
- ✅ Pertinence pour `{PROBLEMATIQUE}` argumentée

---

## 11. Risques et mitigations

| Risque | Probabilité | Mitigation |
|--------|-------------|------------|
| Détection ToC échoue sur OCR bruité | Élevée | Fallback heuristique + validation manuelle option UI |
| Coût explose sur livres très longs | Moyenne | Garde-fou 800k mots + estimation pré-traitement affichée |
| Chapitres oubliés (saut OCR) | Moyenne | Vérification couverture pages : sum(chapter.pages) ≈ total_pages |
| Synthèse contradictoire avec chapitres | Faible | Synthèse construite à partir des résumés courts (Phase 2) — pas re-lecture |
| Latence : 14 chapitres × N secondes = >5 min | Élevée | Parallélisation limitée (semaphore) + SSE granulaire pour patience UX |
| Hallucinations cross-chapitres | Moyenne | Demander citations explicites avec n° chapitre dans la synthèse |

---

## 12. Notes techniques

- **Rétro-compatibilité** : aucun impact sur les modes existants. Le mode `book` est additif.
- **Sentinel** : format inchangé `<!-- ragpy-note-id:uuid -->`.
- **Détection livre vs article** : ne pas auto-détecter ; l'utilisateur sélectionne explicitement le mode `book` dans le dropdown.
- **Source de vérité OCR** : utiliser `output.csv` colonne `texteocr` ; les chunks sont là pour faciliter le slicing par pages.
- **Métadonnées Zotero** : si `itemType="book"` ou `itemType="bookSection"`, l'UI peut suggérer le mode `book` automatiquement (sans l'imposer).
