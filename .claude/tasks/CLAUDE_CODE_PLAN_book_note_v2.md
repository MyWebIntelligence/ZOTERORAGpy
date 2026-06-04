# Plan de développement Claude Code — Migration `book_note_generator.py` vers v2

> **Auteur du plan** : Claude (Anthropic), à partir de l'audit du code existant et de l'expérience d'exécution réelle de la procédure sur un livre académique dense (Laclau, *On Populist Reason*, 2005, 88 000 mots, 11 unités).
>
> **Destinataire** : Claude Code (instance opérant en local sur le repository)
>
> **Cible** : faire migrer le pipeline existant vers la version v2 du prompt, intégrer les apprentissages issus de l'exécution réelle de la procédure, et corriger les écarts structurels identifiés entre l'architecture actuelle et la procédure optimisée.
>
> **Statut du plan** : opérationnel, prêt à être exécuté de bout en bout. Chaque section indique les fichiers concernés, les opérations à effectuer, les tests à conduire et les pièges à éviter.

---

## 1. Contexte et architecture actuelle

### 1.1 Composants identifiés

L'architecture actuelle repose sur deux modules Python et un fichier de prompts :

| Fichier | Rôle | Lignes |
|---|---|---|
| `app/utils/book_note_generator.py` | Orchestrateur multi-phases (Phase 0 heuristique + Phase 1 LLM + boucle Phase 2 + Phase 3 + assemblage) | 875 |
| `app/utils/llm_note_generator.py` | Couche LLM générique (clients OpenAI/OpenRouter, sémaphore, prompt loader) | 889 |
| `app/utils/book_prompt.md` | Trois sous-prompts délimités par marqueurs `=== PHASE N ===` | 489 |

### 1.2 Flux d'exécution actuel

```
build_book_note_async(metadata, text_content)
   │
   ├──► Phase 0 (no LLM) : _split_text_by_pages() + _build_initial_structure()
   │       Fallback : _split_text_into_chapters_heuristic() ou _split_text_into_fixed_parts()
   │
   ├──► Phase 1 (1 LLM call) : _phase1_detect_structure() → JSON typology + chapters
   │       _merge_llm_structure() : fusion avec Phase 0
   │       Re-slicing par page range avec marge ±3
   │
   ├──► Phase 2 (N LLM calls séquentiels) : _phase2_analyse_chapter()
   │       Mémoire incrémentale via PREVIOUS_SUMMARIES
   │       Sortie : (html_block, short_summary)
   │
   ├──► Phase 3 (1 LLM call) : _phase3_synthesise()
   │       Sortie délimitée par ===SECTION_A=== / ===SECTION_B=== / ===END===
   │
   └──► Phase 4 (no LLM) : _assemble_book_html()
           Sentinel + [LIVRE] header + Section A + chapters + Section B
```

### 1.3 Choix d'implémentation existants à conserver

Plusieurs choix d'implémentation sont solides et doivent être préservés sans modification :

- Split par marqueurs `<!-- Page N -->` injectés en amont par le pipeline OCR (`_split_text_by_pages`).
- Fallback heuristique par regex de titres en cas d'absence de marqueurs (`_split_text_into_chapters_heuristic`).
- Ré-découpage par fenêtre de pages avec marge `PHASE2_PAGE_MARGIN = 3` après Phase 1 LLM.
- Sémaphore global `get_llm_semaphore()` pour le contrôle de concurrence multi-utilisateurs.
- Structure `Chapter` / `BookStructure` (dataclasses).
- Vérifications post-assemblage (sentinel, `[LIVRE]`, comptage des `<h3>Chapitre`, sections obligatoires).
- Tolérance aux erreurs LLM en Phase 2 (block de fallback en cas d'échec d'un chapitre).
- Découpage des trois sous-prompts par marqueurs `=== PHASE N ===` dans `book_prompt.md`.

---

## 2. Diagnostic — écarts entre l'architecture actuelle et la procédure optimisée

L'audit détaillé révèle 13 écarts structurels entre le code existant et la procédure validée par l'exécution réelle. Chaque écart est qualifié par sa nature (prompt / code / les deux) et par sa criticité.

### Écart 1 — Rubrique « Place dans le livre » obsolète (CRITIQUE — prompt + code)

**Constat** : `book_prompt.md` lignes 289-294 impose une rubrique « Place dans le livre » dans chaque fiche-chapitre, avec obligation de citer explicitement au moins un autre chapitre. La contrainte globale ligne 487 le confirme : « *chaque fiche-chapitre doit référencer ≥1 autre chapitre dans la rubrique « Place dans le livre »* ». L'expérience réelle d'exécution sur Laclau a montré que cette rubrique se dégrade en formulaire répétitif : sur les chapitres tardifs, elle re-cite mécaniquement les mêmes chapitres centraux (ch.4, préface) sans ajouter d'information nouvelle.

**Décision v2** : la rubrique est supprimée. Les cross-références sont intégralement portées par la Section B (synthèse transversale), qui doit identifier au moins **3 tensions ou dialogues inter-chapitres** au lieu de l'unique tension actuellement requise.

**Conséquences** :
- `book_prompt.md` : suppression de la rubrique dans le template Phase 2.
- `book_prompt.md` : renforcement de la Section B en Phase 3 (passage de « ≥1 tension » à « ≥3 tensions/dialogues »).
- `book_note_generator.py` : aucune modification de code (la suppression est purement prompt).
- Vérification post-assemblage : ajouter un avertissement si la Section B contient moins de 3 tensions explicitement nommées.

### Écart 2 — Format présupposé OCR uniquement (MAJEUR — code)

**Constat** : le pipeline présuppose que `text_content` est un texte OCR plat avec marqueurs `<!-- Page N -->`. Si l'OCR est dégradé (PDF scanné de mauvaise qualité), si le fichier source est un EPUB avec marqueurs `<a id="page_X"/>` au lieu de `<!-- Page N -->`, ou si le fichier est un texte brut sans pagination, le pipeline tombe en mode dégradé sans avertir explicitement l'utilisateur.

**Décision v2** : Phase 0 explicite de détection du format avec procédure conditionnelle pour PDF natif, PDF scanné/OCR, EPUB, texte brut/Markdown.

**Conséquences** :
- `book_note_generator.py` : ajouter une fonction `_detect_source_format(text_content) -> SourceFormat` retournant un enum `pdf_native | pdf_ocr | epub | plain_text`.
- `book_note_generator.py` : ajouter une fonction `_normalize_pagination(text, source_format) -> str` qui convertit les marqueurs spécifiques au format vers le marqueur unifié `<!-- Page N -->` utilisé par le reste du pipeline.
- `book_note_generator.py` : remonter le `source_format` détecté dans le `progress_cb` et dans les logs.
- `book_note_generator.py` : si `source_format == plain_text`, ajouter un avertissement dans les warnings de l'assemblage final.

### Écart 3 — Absence de seuil de bascule documenté (MAJEUR — code + prompt)

**Constat** : le code accepte sans plafond des livres de N chapitres et des chapitres de M caractères. La constante `PHASE2_CHAPTER_MAX_CHARS = 80000` tronque silencieusement les chapitres très longs (avec un marqueur explicite, ce qui est correct), mais aucun mécanisme ne signale à l'utilisateur ou aux logs qu'un livre dépasse les seuils opérationnels validés (12 chapitres, 80 000 mots utiles, ou un chapitre > 30 000 mots).

**Décision v2** : seuil de bascule documenté avec avertissement explicite et estimation de coût.

**Conséquences** :
- `book_note_generator.py` : ajouter trois constantes `THRESHOLD_MAX_CHAPTERS = 12`, `THRESHOLD_MAX_TOTAL_WORDS = 80000`, `THRESHOLD_MAX_CHAPTER_WORDS = 30000`.
- `book_note_generator.py` : après Phase 1, calculer le volume utile (mots, hors paratextes exclus) et logger un avertissement si l'un des seuils est franchi.
- `book_note_generator.py` : exposer dans le dictionnaire de retour optionnel `{"thresholds_exceeded": [...], "estimated_total_tokens": N, "estimated_cost_usd": X}` pour permettre à la couche UI de proposer une confirmation utilisateur avant exécution.
- `book_prompt.md` : aucun changement nécessaire (la décision est exécutée côté Python, pas par le LLM).

### Écart 4 — Lecture exhaustive sans stratégie pour chapitres longs (MAJEUR — code + prompt)

**Constat** : `_truncate_chapter()` (ligne 513) tronque brutalement à 80 000 caractères sans préserver les sections clés (introduction, sections nommées, conclusion). Sur un chapitre de 150 000 caractères, cette troncature perd potentiellement la conclusion et toutes les sections finales. L'expérience réelle sur le ch.4 de Laclau (62 pages, 23 000 mots, 130 000 caractères) confirme l'enjeu.

**Décision v2** : stratégie de lecture par sections pour les chapitres > 15 000 mots, avec extraction de l'introduction + sections nommées + conclusion.

**Conséquences** :
- `book_note_generator.py` : remplacer `_truncate_chapter()` par `_smart_truncate_chapter(text, max_chars, target_words=15000)`.
- Implémentation : si le chapitre dépasse `target_words`, détecter les sections nommées (regex sur sous-titres : lignes courtes en CAPS, lignes courtes suivies de paragraphe, marqueurs Markdown `##`, etc.). Conserver l'introduction (premier ~10% du texte ou jusqu'à la première section), tous les débuts de section (~3 paragraphes après chaque sous-titre), et la conclusion (dernier ~10% du texte ou à partir de la dernière section). Insérer des marqueurs `[... section X intermédiaire omise ...]` aux jonctions.
- Si aucune section détectable, conserver le comportement actuel (troncature brutale en fin) avec un avertissement dans les logs.
- `book_prompt.md` (Phase 2) : ajouter une mention dans le bloc `### Texte OCR fourni` : « Si tu observes des marqueurs `[... section X intermédiaire omise ...]`, c'est qu'une stratégie d'échantillonnage a été appliquée. Travaille avec ce que tu as et signale dans la rubrique Limites si tu as l'impression qu'une section centrale a été coupée. »

### Écart 5 — Mémoire incrémentale par résumés courts non optimale (MINEUR — prompt + code)

**Constat** : la mémoire incrémentale via `PREVIOUS_SUMMARIES` (résumés de 50-80 mots concaténés) consomme un nombre croissant de tokens à chaque chapitre (sur un livre de 11 chapitres, le 11e appel transporte ~800 mots de résumés). Plus problématique : ces résumés sont produits par le LLM en fin de chaque appel Phase 2 et risquent de dériver (un résumé peu fiable au ch.3 contamine la lecture du ch.4 et au-delà).

**Décision v2** : conserver la mémoire incrémentale (le mécanisme reste valide pour donner du contexte cumulatif) mais reformuler le rôle dans le prompt — il s'agit de « connaissance cumulative pour situer le chapitre courant », pas de « résumés à mémoriser ». Et borner la longueur cumulée.

**Conséquences** :
- `book_note_generator.py` : ajouter `MAX_PREVIOUS_SUMMARIES_CHARS = 4000` (≈ 12-15 chapitres avec résumés de 80 mots). Si dépassement, conserver les 3 premiers (préface + introduction + premier chapitre théorique) et les N derniers, avec un marqueur `[... résumés intermédiaires omis ...]`.
- `book_prompt.md` (Phase 2) : reformuler la section `## CHAPITRES DÉJÀ ANALYSÉS` en `## CONNAISSANCE CUMULATIVE DU LIVRE` et clarifier que ces éléments servent à *situer* le chapitre courant dans l'argument du livre, pas à être paraphrasés.

### Écart 6 — Pas de détection des paratextes de fin (MAJEUR — code + prompt)

**Constat** : le pipeline accepte tels quels les chapitres listés par le LLM en Phase 1, sans distinguer paratextes de début (Préface, Introduction générale) qui doivent être analysés, et paratextes de fin (Notes, Index, Bibliographie, Annexes documentaires) qui doivent être exclus. Sur Laclau, j'ai correctement ignoré `19_Notes.xhtml` et `20_Index.xhtml` mais sans guidance du prompt — un LLM moins prudent les aurait analysés comme des chapitres normaux.

**Décision v2** : Phase 1 distingue trois catégories : chapitre normal, paratexte à analyser, paratexte à exclure.

**Conséquences** :
- `book_prompt.md` (Phase 1) : ajouter dans le schéma JSON deux champs supplémentaires `is_long_paratext: bool` et `to_exclude: bool`. Préciser dans les instructions que `to_exclude=true` doit être positionné pour Notes / Index / Bibliographie / Glossaire / Annexes documentaires sans argumentation propre.
- `book_note_generator.py` : ajouter le champ `to_exclude: bool = False` dans la dataclass `Chapter`. Filtrer les chapitres avec `to_exclude=True` avant la boucle Phase 2.
- `book_note_generator.py` : logger explicitement les paratextes exclus avec leurs pages.

### Écart 7 — Catégorie « paratexte long » manquante (MAJEUR — prompt)

**Constat** : le template Phase 2 distingue chapitre normal (template complet, 900-1300 mots) et paratexte (template léger : Fonction / Forme rhétorique). Mais sur Laclau, les *Concluding Remarks* (28 pages, 10 000 mots) sont un paratexte argumentatif aussi dense qu'un chapitre théorique central. Le template léger est insuffisant ; le template chapitre-normal est mal calibré (les *Concluding Remarks* n'ont pas de « Méthode / démarche » au sens classique mais une « Forme rhétorique »).

**Décision v2** : introduire une troisième catégorie « paratexte long » (≥ 5 000 mots) avec template hybride.

**Conséquences** :
- `book_note_generator.py` : ajouter `is_long_paratext: bool = False` dans `Chapter`. Calculer ce champ après Phase 1 par `is_paratext AND word_count(text) >= 5000`.
- `book_prompt.md` (Phase 2) : ajouter un placeholder `{CHAPTER_TREATMENT}` qui prend les valeurs `chapter | paratext_short | paratext_long`. Dans le prompt Phase 2, conditionner les rubriques en fonction de `{CHAPTER_TREATMENT}` :
  - `chapter` : template actuel complet, volume 900-1300 mots, jusqu'à 1800 si chapitre théorique central.
  - `paratext_short` : « Fonction dans l'économie du livre », « Forme rhétorique », 2-3 concepts annoncés, volume 600-900 mots.
  - `paratext_long` : « Fonction dans l'économie du livre » à la place de « Argument central », « Forme rhétorique » à la place de « Méthode / démarche », mais conserver « Concepts mobilisés », « Résultats », « Citations », « Limites ». Volume 1100-1500 mots.
- `book_note_generator.py` : passer `CHAPTER_TREATMENT` dans `_phase2_analyse_chapter()`.

### Écart 8 — Volumes rigides ne distinguant pas le chapitre théorique central (MINEUR — prompt + code)

**Constat** : le prompt Phase 2 fixe une cible 900-1300 mots pour tous les chapitres normaux. Sur les ouvrages avec un noyau théorique dense (cas Laclau ch.4, mais aussi Bourdieu *La distinction* ch.3, Foucault *Surveiller et punir* ch. « Le panoptisme », etc.), cette borne supérieure est trop basse pour rendre justice à la densité conceptuelle.

**Décision v2** : autoriser jusqu'à 1800 mots pour les chapitres théoriques centraux, identifiés soit par leur position structurale (chapitre central d'une partie théorique), soit par leur densité (nombre de concepts > seuil).

**Conséquences** :
- `book_prompt.md` (Phase 2) : assouplir la cible — « 900-1300 mots standard ; jusqu'à 1800 mots si la densité conceptuelle du chapitre justifie un développement étendu (chapitre théorique central, chapitre fondateur du cadre conceptuel) ; 600-900 paratexte court ; 1100-1500 paratexte long. Ne pas dépasser 1800 mots dans aucun cas. »
- `book_note_generator.py` : ajuster `PHASE_MAX_TOKENS["phase2"]` de 7000 à 8000 (déjà fait par le commentaire ligne 56-58, mais la constante en ligne 62 reste à 7000 — vérifier la cohérence).
- `book_note_generator.py` : ajouter un check dans la vérification post-Phase 2 : compter les mots de chaque bloc HTML et logger un avertissement si > 2000 mots (= dérive).

### Écart 9 — Style des paragraphes-concepts insuffisamment cadré (MAJEUR — prompt)

**Constat** : le prompt actuel interdit les sous-rubriques visibles (« Définition : », « Filiation : ») mais n'interdit pas explicitement les marqueurs ordinaux internes (« Premier », « Deuxième », « Trois », « D'abord ») ni les formules transitoires de clôture (« Le concept articule X, Y et Z »). L'expérience réelle a confirmé que ces deux patterns réapparaissent quasi systématiquement, produisant un effet formulaire malgré la prose continue.

**Décision v2** : interdiction explicite des marqueurs ordinaux internes et des formules transitoires de clôture.

**Conséquences** :
- `book_prompt.md` (Phase 2) : ajouter dans le bloc « INTERDIT ABSOLU » deux nouvelles interdictions :
  1. « Pas de marqueurs ordinaux internes (« Premier », « Deuxième », « Trois », « D'abord », « Ensuite ») dans les paragraphes-concepts. Utiliser des transitions naturelles ou aucune transition. »
  2. « Pas de formules transitoires de clôture systématique du type « Le concept articule X, Y et Z », « Le concept fournit le critère », « Le concept opère analytiquement ». Si une articulation théorique doit être nommée, elle doit l'être dans la prose courante, pas en formule de fin de paragraphe. »
- L'exemple « Habitus numérique » (lignes 251-263 du prompt actuel) reste valide et fait office de référence stylistique.

### Écart 10 — Prompt-source mal aligné avec la version master (MAJEUR — prompt + code)

**Constat** : `book_prompt.md` actuel (version déployée) et `PROMPT_Lecture_Livre_Boucle_Automatique_v2.md` (version master que tu as produite) divergent. Le premier est destiné à un pipeline orchestré (3 sous-prompts), le second à une exécution interactive en boucle. Les deux doivent rester synchronisés sur le contenu des templates même s'ils diffèrent sur la structure (un fichier par phase vs un prompt unifié).

**Décision v2** : maintenir `book_prompt.md` comme source de vérité pour le pipeline orchestré, mais aligner ligne à ligne sur les contraintes de v2.

**Conséquences** :
- `book_prompt.md` : refonte complète selon v2 (toutes les modifications listées dans les écarts 1, 5, 6, 7, 8, 9 ci-dessus).
- Optionnel : générer automatiquement à partir de `book_prompt.md` un fichier `PROMPT_Lecture_Livre_Boucle_Automatique_v2.md` documentant le mode interactif (utile pour la documentation utilisateur, pas pour l'exécution Python).

### Écart 11 — Absence d'estimation préalable du coût et du temps (MINEUR — code)

**Constat** : aucune estimation a priori du coût LLM total ni du temps d'exécution attendu n'est exposée à l'utilisateur ni aux logs. Sur un livre de 88 000 mots, on parle d'environ 1 + N + 1 = 13 appels LLM avec des contextes variant entre 5 000 et 30 000 tokens en entrée et 1 500 à 8 000 en sortie. Le coût peut varier d'un facteur 10 selon le modèle.

**Décision v2** : exposer une estimation après Phase 1 et avant Phase 2.

**Conséquences** :
- `book_note_generator.py` : ajouter une fonction `_estimate_phase2_cost(structure, model) -> Dict` qui calcule, pour chaque chapitre :
  - tokens entrée estimés (longueur prompt + texte chapitre tronqué)
  - tokens sortie estimés (1500 à 2400 selon CHAPTER_TREATMENT)
  - coût en USD selon une table de prix par modèle (à charger depuis `app/config/llm_pricing.yaml` ou env)
  - durée estimée (modèle simple : tokens_total / 50 t/s pour cloud)
- Logger l'estimation après Phase 1.
- Exposer optionnellement via `progress_cb` un signal `estimate` avec la structure d'estimation.

### Écart 12 — Pas de gestion robuste de la fenêtre de contexte du modèle (MINEUR — code)

**Constat** : sur un chapitre central de 25 000 mots tronqué à 80 000 caractères (≈ 25 000 tokens en français + 1 000 tokens de prompt + 800 tokens de TOC + 4 000 tokens de PREVIOUS_SUMMARIES en fin de livre = ~30 800 tokens entrée). Avec un modèle 32k de contexte, on est trop juste si la sortie dépasse 1 500 tokens. Le code actuel ne vérifie pas le `context_window` du modèle utilisé.

**Décision v2** : vérification du contexte disponible avant chaque appel.

**Conséquences** :
- `llm_note_generator.py` : ajouter `MODEL_CONTEXT_WINDOWS: Dict[str, int]` (table à la racine du module) listant les contextes connus pour les modèles supportés (gpt-4o-mini : 128k, gpt-4o : 128k, claude-3-5-sonnet : 200k, gemini-2.5-flash : 1M, etc.).
- `_generate_with_llm()` : avant l'appel, estimer la longueur du prompt en tokens (approximation : `len(prompt) / 4` pour anglais, `len(prompt) / 3` pour français) et vérifier que `tokens_in + max_tokens <= context_window * 0.95`. Si violation, lever `ContextWindowExceededError` avec un message actionnable.
- `book_note_generator.py` : pour Phase 2, attraper cette exception et appliquer la stratégie de section (écart 4) plus agressivement (réduire `target_words` à 10 000 puis 5 000), avec retries.

### Écart 13 — Tests d'intégration manquants pour les cas limites (MAJEUR — qualité)

**Constat** : aucun test automatisé documenté pour vérifier que (a) un EPUB sans pagination produit un fallback gracieux, (b) un livre avec 15 chapitres déclenche l'avertissement de seuil, (c) un paratexte long produit le bon template, (d) la Section B contient bien ≥3 tensions.

**Décision v2** : suite de tests dédiée.

**Conséquences** :
- Créer `tests/utils/test_book_note_generator_v2.py` avec :
  - `test_format_detection_pdf_native()` (fixture : OCR avec marqueurs `<!-- Page N -->`)
  - `test_format_detection_epub_pagebreak()` (fixture : OCR avec marqueurs `<a id="page_X"/>`)
  - `test_format_detection_plain_text()` (fixture : texte sans pagination)
  - `test_threshold_warning_too_many_chapters()` (mock structure avec 15 chapitres → log warning attendu)
  - `test_threshold_warning_long_chapter()` (mock structure avec 1 chapitre de 35 000 mots)
  - `test_smart_truncate_preserves_sections()` (fixture : chapitre avec sous-titres → vérifier que début + fin de chaque section sont conservés)
  - `test_paratext_long_uses_hybrid_template()` (mock chapitre `is_long_paratext=True` → vérifier prompt Phase 2 généré)
  - `test_section_b_three_tensions_check()` (mock Section B avec 1 tension → warning attendu)
  - `test_excluded_paratext_skipped()` (mock structure avec chapitre `to_exclude=True` → pas d'appel Phase 2)
- Couverture cible : 85% sur `book_note_generator.py`.

---

## 3. Plan d'exécution structuré

Le plan d'exécution est découpé en 6 lots indépendants ordonnés par dépendance. Chaque lot est self-contained et testable individuellement.

### Lot 1 — Refonte du prompt source (`book_prompt.md`)

**Objectif** : aligner intégralement `book_prompt.md` sur les décisions v2.

**Fichiers modifiés** : `app/utils/book_prompt.md`

**Opérations** :

1.1. **Phase 1 — modification du schéma JSON** (lignes 99-126) :
- Ajouter `is_long_paratext: bool` et `to_exclude: bool` dans le schéma de chaque chapitre.
- Ajouter dans les instructions (après ligne 97) : « Marque `to_exclude=true` pour les paratextes de fin sans contenu argumentatif propre (Notes, Index, Bibliographie, Glossaire, Annexes documentaires). Ces chapitres seront filtrés et ne feront pas l'objet d'une fiche d'analyse. Marque `is_long_paratext=true` si le paratexte (Préface, Introduction générale, Postface, Concluding Remarks) dépasse 5 000 mots et mérite un traitement substantiel proche d'un chapitre. »

1.2. **Phase 2 — suppression de la rubrique « Place dans le livre »** :
- Supprimer le bloc lignes 289-294.
- Renuméroter implicitement les rubriques suivantes (la « Place dans le livre » disparaît, le bloc « Limites » devient le dernier bloc avant la mention SUMMARY).

1.3. **Phase 2 — ajout du placeholder `{CHAPTER_TREATMENT}`** :
- Dans la liste des placeholders (lignes 132-146), ajouter :
  ```text
  {CHAPTER_TREATMENT}     — chapter | paratext_short | paratext_long
  ```
- Remplacer le bloc de règles (lignes 304-307) qui distingue uniquement « si chapitre est paratexte » par un bloc conditionnel à trois branches selon `{CHAPTER_TREATMENT}`.

1.4. **Phase 2 — assouplissement du volume cible** :
- Remplacer ligne 150 « 900-1300 mots » par « 900-1300 mots standard ; jusqu'à 1800 mots si chapitre théorique central ; 600-900 mots si paratexte court ; 1100-1500 mots si paratexte long ».
- Idem dans le bloc « ### 1. Un bloc HTML structuré » (ligne 208).

1.5. **Phase 2 — renforcement des interdictions stylistiques** :
- Dans le bloc « ⚠️ INTERDIT ABSOLU » (lignes 225-228), ajouter deux nouvelles règles :
  - « Pas de marqueurs ordinaux internes (« Premier », « Deuxième », « Trois », « D'abord », « Ensuite ») dans les paragraphes-concepts. Utiliser des transitions naturelles ou aucune transition. »
  - « Pas de formules transitoires de clôture systématique du type « Le concept articule X, Y et Z », « Le concept fournit le critère », « Le concept opère analytiquement ». Si une articulation théorique doit être nommée, elle doit l'être dans la prose courante, pas en formule de fin de paragraphe. »

1.6. **Phase 2 — reformulation de la mémoire incrémentale** :
- Renommer la section « ## CHAPITRES DÉJÀ ANALYSÉS (mémoire courte) » (ligne 171) en « ## CONNAISSANCE CUMULATIVE DU LIVRE ».
- Ajouter un paragraphe d'introduction : « Ces éléments te donnent la connaissance cumulative du livre à ce stade. Ils servent à situer le chapitre courant dans l'argument global, pas à être paraphrasés ni cités directement. »

1.7. **Phase 2 — ajout de la mention sur la troncature intelligente** :
- Dans le bloc « ### Texte OCR fourni » (ligne 184), après la mention sur la marge ±3 pages, ajouter :
  > « Si tu observes des marqueurs `[... section X intermédiaire omise ...]` dans le texte ci-dessous, c'est qu'une stratégie d'échantillonnage par sections a été appliquée pour respecter la limite de contexte. Travaille avec le texte fourni (introduction, sections nommées, conclusion) et signale dans la rubrique « Limites » si tu identifies qu'une section centrale a probablement été coupée. »

1.8. **Phase 3 — renforcement de la Section B** :
- Modifier la rubrique « Tensions internes » (lignes 403-405) :
  > « **Tensions internes** : identifie au minimum **3 désaccords, contradictions, ou complémentarités fertiles** entre chapitres, avec citations explicites « Le ch.5 (X) défend Y, tandis que le ch.9 (Z) soutient le contraire ». Cette rubrique compense la suppression de la rubrique « Place dans le livre » des fiches-chapitres et doit être substantielle. »

1.9. **Contraintes globales — mise à jour** :
- Modifier la ligne 487 (cross-références) :
  > « **Cross-références** : la fonction de cross-référence est intégralement portée par la Section B (Phase 3), qui doit identifier au moins 3 tensions ou dialogues inter-chapitres. Les fiches-chapitres ne contiennent **plus** de rubrique dédiée aux cross-références ; elles peuvent intégrer naturellement des renvois à d'autres chapitres dans la prose des rubriques existantes lorsque conceptuellement justifié, sans rubrique formelle ni obligation de fréquence. »

**Tests** :
- Test manuel : générer une fiche de lecture sur un livre de référence (corpus de test à constituer).
- Test automatisé : `_load_book_prompt()` charge le fichier sans erreur, `_extract_phase_prompt()` extrait correctement les 3 phases.

**Pièges à éviter** :
- Ne pas supprimer le marqueur `=== PHASE N ===` ni l'ancrage du dernier `code block` dans chaque section : `_extract_phase_prompt()` (ligne 159) repose dessus.
- Vérifier qu'aucun placeholder existant n'est cassé par les ajouts (notamment `{TOC_RAW}`, `{FIRST_PAGES}`, `{PREVIOUS_SUMMARIES}`).

---

### Lot 2 — Détection de format et normalisation de pagination (`book_note_generator.py`)

**Objectif** : ajouter Phase 0 explicite de détection de format et normalisation de la pagination.

**Fichiers modifiés** : `app/utils/book_note_generator.py`

**Opérations** :

2.1. **Ajout de l'enum `SourceFormat`** :
```python
from enum import Enum

class SourceFormat(str, Enum):
    PDF_NATIVE = "pdf_native"   # OCR ou texte natif PDF avec marqueurs <!-- Page N -->
    PDF_OCR = "pdf_ocr"         # OCR de PDF scanné (qualité potentiellement dégradée)
    EPUB = "epub"               # EPUB extrait avec marqueurs <a id="page_X"/>
    PLAIN_TEXT = "plain_text"   # Texte sans pagination
```

2.2. **Fonction `_detect_source_format`** :
```python
# Patterns spécifiques au format
_EPUB_PAGE_MARKER_RE = re.compile(r'<a\s+id="page_([0-9ivxlcdm]+)"\s*/?>', re.IGNORECASE)
_OCR_DEGRADATION_PATTERNS = [
    re.compile(r"[^\w\s]{5,}"),       # séquences de symboles isolés (artefact OCR)
    re.compile(r"\b[a-z]\s[a-z]\s[a-z]\b"),  # caractères isolés séparés par espaces
]

def _detect_source_format(text: str) -> SourceFormat:
    """
    Détecte le format source à partir du contenu textuel.
    
    Returns:
        SourceFormat enum.
    """
    if PAGE_MARKER_RE.search(text):
        # Pagination déjà au format pipeline. Vérifier qualité OCR.
        sample = text[:5000]
        degradation_score = sum(
            len(p.findall(sample)) for p in _OCR_DEGRADATION_PATTERNS
        )
        if degradation_score > 50:  # seuil empirique
            return SourceFormat.PDF_OCR
        return SourceFormat.PDF_NATIVE
    
    if _EPUB_PAGE_MARKER_RE.search(text):
        return SourceFormat.EPUB
    
    return SourceFormat.PLAIN_TEXT
```

2.3. **Fonction `_normalize_pagination`** :
```python
def _normalize_pagination(text: str, source_format: SourceFormat) -> str:
    """
    Convertit les marqueurs de pagination spécifiques au format vers le marqueur 
    unifié `<!-- Page N -->` utilisé par le reste du pipeline.
    
    Pour EPUB, convertit `<a id="page_X"/>` en `<!-- Page X -->`.
    Pour PDF_NATIVE et PDF_OCR, le format est déjà bon.
    Pour PLAIN_TEXT, le texte est retourné tel quel (pas de pagination disponible).
    """
    if source_format == SourceFormat.EPUB:
        # Convertir les marqueurs EPUB vers le format pipeline
        # Préserver les numéros de page imprimés (incluant chiffres romains)
        def replace_marker(m):
            page_id = m.group(1)
            return f"\n<!-- Page {page_id} -->\n"
        normalized = _EPUB_PAGE_MARKER_RE.sub(replace_marker, text)
        return normalized
    
    # PDF_NATIVE, PDF_OCR : déjà au bon format
    # PLAIN_TEXT : pas de pagination
    return text
```

2.4. **Modification de `build_book_note_async`** :
- Avant `_split_text_by_pages(text_content)` (ligne 775), ajouter :
```python
source_format = _detect_source_format(text_content)
logger.info("Phase 0: source format detected = %s", source_format.value)
text_content = _normalize_pagination(text_content, source_format)
if progress_cb:
    await progress_cb("format_detection", 1, 1, f"Format : {source_format.value}")
```

2.5. **Avertissement dans l'assemblage** :
- Dans `_assemble_book_html()`, ajouter un paramètre `source_format` et inclure dans `warnings_list` un avertissement si `source_format == SourceFormat.PLAIN_TEXT` : « pagination absente dans le texte source — références par section au lieu de pages ».
- Si `source_format == SourceFormat.PDF_OCR`, ajouter un warning info-level (pas error) : « OCR potentiellement dégradé, citations à vérifier ».

**Tests** :
- `test_format_detection_pdf_native()` : fixture avec `<!-- Page 1 -->...<!-- Page 2 -->` propre → `PDF_NATIVE`.
- `test_format_detection_pdf_ocr_degraded()` : fixture avec `<!-- Page 1 -->` mais texte avec artefacts → `PDF_OCR`.
- `test_format_detection_epub()` : fixture avec `<a id="page_15"/>` → `EPUB`.
- `test_format_detection_plain_text()` : fixture sans aucun marqueur → `PLAIN_TEXT`.
- `test_normalize_pagination_epub_to_pipeline()` : EPUB → marqueurs `<!-- Page X -->`.

---

### Lot 3 — Phase 1 : seuils de bascule et exclusion des paratextes (`book_note_generator.py`)

**Objectif** : ajouter `is_long_paratext`, `to_exclude`, et le contrôle de seuils.

**Fichiers modifiés** : `app/utils/book_note_generator.py`

**Opérations** :

3.1. **Ajout des constantes de seuil** :
```python
# Seuils opérationnels — au-delà, log un avertissement et expose dans l'API
THRESHOLD_MAX_CHAPTERS = 12
THRESHOLD_MAX_TOTAL_WORDS = 80_000
THRESHOLD_MAX_CHAPTER_WORDS = 30_000
THRESHOLD_LONG_PARATEXT_WORDS = 5_000

# Mémoire incrémentale : limite cumulée pour éviter l'explosion du contexte
MAX_PREVIOUS_SUMMARIES_CHARS = 4_000
```

3.2. **Modification de `Chapter` dataclass** :
```python
@dataclass
class Chapter:
    num: int
    title: str
    authors: List[str]
    pages: Tuple[Optional[int], Optional[int]] = (None, None)
    text: str = ""
    is_paratext: bool = False
    is_long_paratext: bool = False     # NEW
    to_exclude: bool = False           # NEW
    short_summary: Optional[str] = None
    html_block: Optional[str] = None
    
    @property
    def treatment(self) -> str:
        """Retourne le type de traitement à appliquer en Phase 2."""
        if self.is_long_paratext:
            return "paratext_long"
        if self.is_paratext:
            return "paratext_short"
        return "chapter"
    
    @property
    def word_count(self) -> int:
        """Estimation simple du nombre de mots."""
        return len(self.text.split())
```

3.3. **Modification de `_merge_llm_structure`** :
- Lire les nouveaux champs JSON `is_long_paratext` et `to_exclude` avec valeurs par défaut `False`.
- Après merge, recalculer `is_long_paratext` à partir du texte effectivement disponible : si `is_paratext=True` et `word_count >= THRESHOLD_LONG_PARATEXT_WORDS`, alors `is_long_paratext=True`.

3.4. **Filtrage des chapitres exclus** :
- Dans `build_book_note_async`, après le re-slicing (ligne 815), ajouter :
```python
excluded = [ch for ch in structure.chapters if ch.to_exclude]
if excluded:
    logger.info(
        "Phase 0: %d paratextes exclus de l'analyse: %s",
        len(excluded),
        ", ".join(f"ch.{c.num} ({c.title[:40]})" for c in excluded)
    )
structure.chapters = [ch for ch in structure.chapters if not ch.to_exclude]
```

3.5. **Vérification des seuils** :
- Avant la boucle Phase 2, ajouter :
```python
total_words = sum(ch.word_count for ch in structure.chapters)
max_chapter_words = max((ch.word_count for ch in structure.chapters), default=0)

thresholds_exceeded = []
if len(structure.chapters) > THRESHOLD_MAX_CHAPTERS:
    thresholds_exceeded.append(
        f"{len(structure.chapters)} chapitres (seuil: {THRESHOLD_MAX_CHAPTERS})"
    )
if total_words > THRESHOLD_MAX_TOTAL_WORDS:
    thresholds_exceeded.append(
        f"{total_words} mots utiles (seuil: {THRESHOLD_MAX_TOTAL_WORDS})"
    )
if max_chapter_words > THRESHOLD_MAX_CHAPTER_WORDS:
    thresholds_exceeded.append(
        f"chapitre max {max_chapter_words} mots (seuil: {THRESHOLD_MAX_CHAPTER_WORDS})"
    )

if thresholds_exceeded:
    logger.warning(
        "Seuils opérationnels franchis: %s. Qualité dégradée possible sur "
        "les derniers chapitres et la synthèse.",
        " | ".join(thresholds_exceeded)
    )
    if progress_cb:
        await progress_cb(
            "threshold_warning",
            0, 1,
            "Seuils franchis : " + ", ".join(thresholds_exceeded)
        )
```

**Tests** :
- `test_threshold_warning_too_many_chapters` (mock 15 chapitres → warning loggé).
- `test_threshold_warning_long_chapter` (mock 1 chapitre 35k mots → warning loggé).
- `test_excluded_paratext_skipped` (mock chapitre `to_exclude=True` → absent de la boucle Phase 2, présent dans les logs).
- `test_long_paratext_recomputed_from_text` (mock paratexte 7000 mots → `is_long_paratext=True` recalculé).

---

### Lot 4 — Troncature intelligente par sections (`book_note_generator.py`)

**Objectif** : remplacer `_truncate_chapter()` par `_smart_truncate_chapter()` qui préserve l'introduction, les sections nommées et la conclusion.

**Fichiers modifiés** : `app/utils/book_note_generator.py`

**Opérations** :

4.1. **Détection des sections nommées** :
```python
# Patterns pour détecter les sous-titres internes au chapitre
# Lignes courtes (< 100 chars), souvent en CAPS partiel ou suivies d'un saut
# de ligne. Conservatrices pour éviter les faux positifs.
_SECTION_PATTERNS = [
    re.compile(r"^[A-Z][A-Z\s\-:]{4,80}$", re.MULTILINE),     # CAPS lock
    re.compile(r"^\d+(\.\d+)?\s+[A-Z][^\n]{3,80}$", re.MULTILINE),  # 1. Title, 2.3 Subtitle
    re.compile(r"^#{2,4}\s+.+$", re.MULTILINE),               # Markdown headers
    re.compile(r"^[A-Z][a-z][^\n]{5,80}$\n\n", re.MULTILINE), # Title-case isolé suivi de paragraphe
]


def _detect_sections(text: str) -> List[Tuple[int, str]]:
    """
    Détecte les sections nommées dans un chapitre.
    
    Returns:
        Liste de (offset, titre_section) triée par position.
    """
    matches = []
    for pattern in _SECTION_PATTERNS:
        for m in pattern.finditer(text):
            matches.append((m.start(), m.group().strip()))
    matches.sort(key=lambda x: x[0])
    
    # Déduplication : si deux matches sont à moins de 50 chars, garder le premier
    deduped = []
    for offset, title in matches:
        if not deduped or offset - deduped[-1][0] > 50:
            deduped.append((offset, title))
    return deduped
```

4.2. **Fonction `_smart_truncate_chapter`** :
```python
def _smart_truncate_chapter(
    text: str,
    max_chars: int = PHASE2_CHAPTER_MAX_CHARS,
    target_words: int = 15_000,
) -> str:
    """
    Tronque un chapitre long en préservant les sections clés.
    
    Stratégie :
      - Si chapitre <= target_words, retourne le texte intégral (éventuellement
        tronqué brutalement à max_chars en filet de sécurité).
      - Sinon, tente une troncature par sections : conserve l'intro 
        (~10% début), le début de chaque section nommée (~3 paragraphes), 
        et la conclusion (~10% fin).
      - Si aucune section détectée, fallback sur troncature brutale.
    """
    word_count = len(text.split())
    
    # Cas standard : chapitre raisonnablement court
    if word_count <= target_words:
        if len(text) <= max_chars:
            return text
        # Filet de sécurité : si le texte dépasse max_chars en bytes mais pas en mots,
        # tronquer brutalement (cas pathologique : énormément de markup).
        return text[:max_chars - 200] + (
            "\n\n[... troncature : fin du chapitre omise pour respecter la limite ...]"
        )
    
    sections = _detect_sections(text)
    
    if len(sections) < 2:
        # Pas assez de sections détectées : fallback sur troncature brutale
        # avec préservation explicite du début et de la fin.
        head_chars = int(max_chars * 0.7)
        tail_chars = int(max_chars * 0.2)
        head = text[:head_chars]
        tail = text[-tail_chars:]
        return (
            head + 
            "\n\n[... troncature : milieu du chapitre omis pour respecter la limite "
            f"({word_count} mots, sections non détectées) ...]\n\n" +
            tail
        )
    
    # Stratégie par sections : intro + (titre + ~3 paragraphes) × N + conclusion
    intro_end = sections[0][0]
    conclusion_start = sections[-1][0]
    
    parts = [text[:intro_end]]  # Introduction
    
    for i, (offset, title) in enumerate(sections[:-1]):
        # Pour chaque section, conserver le titre + ~3 paragraphes (~1500 chars)
        next_offset = sections[i + 1][0]
        section_text = text[offset:next_offset]
        # Conserver les 1500 premiers chars + la dernière phrase
        if len(section_text) > 2000:
            section_excerpt = (
                section_text[:1500] +
                "\n\n[... section "
                f"« {title[:60]} » développement intermédiaire omis ...]\n\n"
            )
            parts.append(section_excerpt)
        else:
            parts.append(section_text)
    
    # Conclusion : dernière section + texte jusqu'à la fin
    parts.append(text[conclusion_start:])
    
    result = "".join(parts)
    
    # Filet de sécurité final
    if len(result) > max_chars:
        result = result[:max_chars - 200] + (
            "\n\n[... troncature finale : limite de contexte atteinte ...]"
        )
    
    logger.info(
        "Smart truncate: %d → %d chars (%d sections, %d mots originaux)",
        len(text), len(result), len(sections), word_count
    )
    return result
```

4.3. **Substitution dans `_phase2_analyse_chapter`** :
- Remplacer ligne 558 :
```python
"CHAPTER_TEXT": _truncate_chapter(chapter.text, PHASE2_CHAPTER_MAX_CHARS),
```
- Par :
```python
"CHAPTER_TEXT": _smart_truncate_chapter(chapter.text, PHASE2_CHAPTER_MAX_CHARS),
```

4.4. **Conservation de l'ancienne fonction** :
- Renommer l'ancienne `_truncate_chapter` en `_truncate_chapter_legacy` et la conserver pour rétrocompatibilité (utilisable comme fallback en cas d'exception dans `_smart_truncate_chapter`).

**Tests** :
- `test_smart_truncate_short_chapter_unchanged` (chapitre 5000 mots → texte intégral retourné).
- `test_smart_truncate_long_chapter_with_sections` (chapitre 25 000 mots avec 5 sous-titres en CAPS → vérifier que l'intro, chaque début de section, et la conclusion sont présents dans la sortie).
- `test_smart_truncate_long_chapter_without_sections` (chapitre 25 000 mots sans sous-titres → fallback début+fin).
- `test_smart_truncate_respects_max_chars` (chapitre 200 000 chars → sortie ≤ 80 000 chars).

---

### Lot 5 — Phase 2 : traitement différencié et mémoire incrémentale bornée (`book_note_generator.py`)

**Objectif** : passer `CHAPTER_TREATMENT` au prompt et borner la longueur cumulée des résumés précédents.

**Fichiers modifiés** : `app/utils/book_note_generator.py`

**Opérations** :

5.1. **Fonction de bornage des résumés cumulés** :
```python
def _build_cumulative_summaries(
    summaries: List[str],
    max_chars: int = MAX_PREVIOUS_SUMMARIES_CHARS,
) -> str:
    """
    Construit le bloc PREVIOUS_SUMMARIES injecté dans Phase 2, en bornant
    la longueur totale.
    
    Stratégie : si la concaténation dépasse max_chars, conserver les 3 premiers
    résumés (souvent paratextes initiaux + premier chapitre théorique, qui 
    posent le cadre du livre) et les N derniers résumés tels qu'ils tiennent
    dans le budget restant, avec un marqueur d'omission.
    """
    if not summaries:
        return "(aucun chapitre encore analysé)"
    
    full = "\n".join(summaries)
    if len(full) <= max_chars:
        return full
    
    # Bornage : 3 premiers + autant des derniers que possible
    head = summaries[:3]
    head_text = "\n".join(head)
    head_len = len(head_text)
    
    omission_marker = (
        f"\n\n[... {len(summaries) - 3 - 0} résumés intermédiaires omis pour "
        f"respecter la limite de contexte ...]\n\n"
    )
    
    available_for_tail = max_chars - head_len - len(omission_marker)
    
    # Sélectionner les derniers résumés qui tiennent dans available_for_tail
    tail = []
    tail_len = 0
    for s in reversed(summaries[3:]):
        if tail_len + len(s) + 1 > available_for_tail:
            break
        tail.insert(0, s)
        tail_len += len(s) + 1
    
    if not tail:
        # Cas extrême : même 0 tail ne tient pas. Tronquer le head.
        return head_text[:max_chars - 50] + "\n[...]"
    
    n_omitted = len(summaries) - len(head) - len(tail)
    if n_omitted > 0:
        omission_marker = (
            f"\n\n[... {n_omitted} résumés intermédiaires omis pour "
            f"respecter la limite de contexte ...]\n\n"
        )
    else:
        omission_marker = "\n"
    
    return head_text + omission_marker + "\n".join(tail)
```

5.2. **Modification de `_phase2_analyse_chapter`** :
- Remplacer ligne 559 :
```python
"PREVIOUS_SUMMARIES": "\n".join(previous_summaries) if previous_summaries else "(aucun chapitre encore analysé)",
```
- Par :
```python
"PREVIOUS_SUMMARIES": _build_cumulative_summaries(previous_summaries),
```

5.3. **Ajout du placeholder `CHAPTER_TREATMENT`** :
- Dans la liste des placeholders passés à `_fill_placeholders` (lignes 544-561), ajouter :
```python
"CHAPTER_TREATMENT": chapter.treatment,  # chapter | paratext_short | paratext_long
```

5.4. **Vérification post-Phase 2 du volume** :
- Après réception de `block` (ligne 587), ajouter :
```python
block_word_count = len(re.sub(r"<[^>]+>", " ", block).split())
expected_max = {
    "chapter": 1800,
    "paratext_short": 900,
    "paratext_long": 1500,
}.get(chapter.treatment, 1300)

if block_word_count > expected_max + 200:  # tolérance 200 mots
    logger.warning(
        "Phase 2: chapter %d block exceeds target (%d > %d for treatment %s)",
        chapter.num, block_word_count, expected_max, chapter.treatment
    )
```

**Tests** :
- `test_cumulative_summaries_under_limit_unchanged` (3 résumés courts → concaténation simple).
- `test_cumulative_summaries_over_limit_truncated` (15 résumés totalisant 6000 chars → 3 premiers + N derniers + marqueur d'omission, total ≤ 4000 chars).
- `test_phase2_passes_treatment_to_prompt` (mock chapitre `is_long_paratext=True` → vérifier que `CHAPTER_TREATMENT=paratext_long` est dans le prompt généré).

---

### Lot 6 — Vérifications post-assemblage et tests d'intégration

**Objectif** : renforcer les vérifications de qualité de la fiche assemblée et constituer la suite de tests v2.

**Fichiers modifiés** : `app/utils/book_note_generator.py`, nouveau `tests/utils/test_book_note_generator_v2.py`

**Opérations** :

6.1. **Ajout des vérifications dans `_assemble_book_html`** :
- Après la boucle de vérification existante (lignes 698-712), ajouter :

```python
# v2 — Vérification suppression « Place dans le livre »
if "Place dans le livre" in chapters_html:
    warnings_list.append(
        "rubrique « Place dans le livre » présente dans une fiche-chapitre "
        "(devrait être absente en v2 — vérifier book_prompt.md)"
    )

# v2 — Vérification ≥3 tensions dans la Section B
section_b_str = section_b or ""
tension_indicators = [
    re.findall(r"\bch\.\s*\d+", section_b_str, re.IGNORECASE),  # references "ch.X"
    re.findall(r"\bchapitre\s+\d+", section_b_str, re.IGNORECASE),
]
total_chapter_refs = sum(len(refs) for refs in tension_indicators)
# Heuristique: ≥3 tensions/dialogues ≈ ≥6 références à des chapitres distincts
if total_chapter_refs < 6:
    warnings_list.append(
        f"Section B contient {total_chapter_refs} références à des chapitres "
        "(seuil: ≥6 pour 3 tensions/dialogues distincts)"
    )

# v2 — Vérification interdictions stylistiques dans les fiches-chapitres
forbidden_patterns_v2 = [
    (r"\b(Premier|Premièrement|Deuxième|Deuxièmement|Troisième|Troisièmement)\s*[—:.]\s*", "marqueur ordinal"),
    (r"\bLe concept articule\b", "formule transitoire « Le concept articule »"),
    (r"\bLe concept fournit\b", "formule transitoire « Le concept fournit »"),
    (r"\bLe concept opère\b", "formule transitoire « Le concept opère »"),
]
for pattern, label in forbidden_patterns_v2:
    matches = len(re.findall(pattern, chapters_html))
    if matches > 0:
        warnings_list.append(
            f"{matches} occurrence(s) de {label} (interdit en v2)"
        )
```

6.2. **Création du fichier de tests** :
- Créer `tests/utils/test_book_note_generator_v2.py` avec les tests listés dans l'écart 13. Utiliser `pytest` + `unittest.mock` + fixtures de textes courts.
- Couverture cible : 85% sur `book_note_generator.py`.

6.3. **Test end-to-end** :
- Ajouter `tests/utils/test_book_note_e2e.py` avec un test marqué `@pytest.mark.slow` qui appelle `build_book_note_async` sur un livre court de référence (3-5 chapitres, ~30 000 mots) avec un mock LLM déterministe. Vérifier l'absence de warnings critiques dans les logs.

**Pièges à éviter** :
- Les heuristiques de vérification post-assemblage (ex: « ≥6 références à des chapitres ») doivent être suffisamment tolérantes pour ne pas générer de faux positifs sur des livres courts (5 chapitres ou moins, où 3 tensions × 2 références ne sont pas atteignables).
- Tolérer 1-2 occurrences de patterns interdits (peuvent apparaître légitimement dans une citation verbatim de l'auteur du livre).

---

## 4. Plan de migration et compatibilité ascendante

### 4.1 Stratégie de déploiement

La migration se fait en **deux étapes** pour minimiser le risque de régression :

**Étape 1 — Déploiement silencieux (lots 1, 2, 3)** :
- Modifications du prompt (lot 1) et ajouts de Phase 0 explicite (lot 2) et seuils (lot 3).
- Aucun changement du contrat de la fonction `build_book_note_async`.
- Aucun impact UI.
- Tester sur un livre court de référence (3-5 chapitres) en environnement de staging.
- Comparer les fiches générées v1 vs v2 sur le même livre — vérifier que la qualité s'améliore et que les régressions sont absentes.

**Étape 2 — Déploiement avec optimisations (lots 4, 5, 6)** :
- Troncature intelligente, mémoire bornée, vérifications renforcées.
- Tester sur un livre long de référence (10+ chapitres, > 80 000 mots).

### 4.2 Drapeaux de compatibilité

Pour permettre un rollback rapide en cas de régression, exposer une variable d'environnement `BOOK_NOTE_VERSION` :
- `v1` : comportement actuel (rétrocompatibilité).
- `v2` (par défaut) : nouvelles règles.

Dans `book_note_generator.py`, ajouter en tête :
```python
BOOK_NOTE_VERSION = os.getenv("BOOK_NOTE_VERSION", "v2").lower()
```

Et conditionner les nouvelles vérifications :
```python
if BOOK_NOTE_VERSION == "v2":
    # nouvelles vérifications
    ...
```

Les modifications du prompt source (`book_prompt.md`) ne sont pas conditionnées : on s'engage sur v2 comme nouveau standard.

### 4.3 Documentation

- Mettre à jour `.claude/tasks/book_reading_notes.md` (le fichier référence cité ligne 10 du `book_note_generator.py`) pour décrire la procédure v2.
- Ajouter `docs/book_note_v2_migration.md` documentant les écarts v1/v2 et la stratégie de migration.

---

## 5. Ordre d'exécution recommandé pour Claude Code

```
1. Lot 1 (prompt seul — pas de risque code)
   └─ Test : générer fiche sur livre court de référence, comparer à fiche v1
   
2. Lot 6.2 (création fichier de tests vide avec stubs)
   └─ Test : pytest tests/utils/test_book_note_generator_v2.py — 0 passed (stubs only)

3. Lot 2 (détection format)
   └─ Tests : test_format_detection_*, test_normalize_pagination_*

4. Lot 3 (seuils et exclusion paratextes)
   └─ Tests : test_threshold_warning_*, test_excluded_paratext_skipped

5. Lot 4 (troncature intelligente)
   └─ Tests : test_smart_truncate_*

6. Lot 5 (mémoire bornée et CHAPTER_TREATMENT)
   └─ Tests : test_cumulative_summaries_*, test_phase2_passes_treatment_*

7. Lot 6.1 (vérifications post-assemblage)
   └─ Tests : test_section_b_three_tensions, test_forbidden_patterns_detected

8. Lot 6.3 (test end-to-end avec mock LLM déterministe)
   └─ Test : test_e2e_build_book_note_v2

9. Test de régression manuel
   └─ Régénérer une fiche sur un livre déjà traité en v1, comparer le diff
```

À chaque lot, **commit séparé** avec message descriptif (`feat(book): lot N — description`). Si un test échoue, **ne pas continuer** — investiguer la cause avant de passer au lot suivant.

---

## 6. Estimation de complexité et de durée

| Lot | Complexité | LOC modifiées | Risque | Durée estimée Claude Code |
|---|---|---|---|---|
| 1 | Faible | +/- 80 (prompt) | Faible | 20 min |
| 2 | Moyenne | +120 (code) | Faible | 30 min |
| 3 | Moyenne | +60 (code) | Faible | 25 min |
| 4 | Élevée | +130 (code) | Moyen | 45 min |
| 5 | Moyenne | +80 (code) | Faible | 30 min |
| 6 | Moyenne | +200 (tests) | Faible | 50 min |
| **Total** | — | **~670 LOC** | — | **~3h30** |

---

## 7. Critères d'acceptation finaux

La migration est considérée terminée quand toutes les conditions suivantes sont remplies :

1. ✅ Tous les tests unitaires passent (`pytest tests/utils/test_book_note_generator_v2.py`).
2. ✅ Couverture sur `book_note_generator.py` ≥ 85%.
3. ✅ Le test end-to-end sur livre court (3-5 chapitres) génère une fiche conforme : sentinel présent, `[LIVRE]` présent, N blocs de chapitre, Section A et B présentes, **aucune** rubrique « Place dans le livre » dans les fiches-chapitres, **≥ 6 références à des chapitres** dans la Section B.
4. ✅ Le test sur livre long (10+ chapitres, > 80 000 mots) déclenche bien le warning de seuil et produit une fiche dégradée mais utilisable.
5. ✅ Sur un livre EPUB de référence, la pagination est correctement extraite et propagée jusqu'aux pages dans les fiches.
6. ✅ Sur un texte brut sans pagination, le pipeline génère une fiche avec références par section et avertissement explicite.
7. ✅ Aucune occurrence des marqueurs ordinaux interdits (« Premier », « Deuxièmement », etc.) ni des formules transitoires (« Le concept articule », etc.) dans la fiche générée.
8. ✅ Les paratextes de fin (Notes, Index, Bibliographie) sont correctement filtrés et n'apparaissent pas dans la fiche.
9. ✅ Les paratextes longs (Concluding Remarks, Postface substantielle) sont traités avec le template hybride et produisent des fiches denses (1100-1500 mots).
10. ✅ Documentation mise à jour : `.claude/tasks/book_reading_notes.md` et `docs/book_note_v2_migration.md`.

---

## 8. Annexe — Risques identifiés et stratégies de mitigation

| Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|
| Détection de format trop laxe (PDF_OCR classé en PDF_NATIVE alors qu'il est dégradé) | Moyenne | Faible | Le seuil de `degradation_score > 50` est ajustable ; en cas de doute, le pipeline fonctionne quand même, juste sans warning |
| Détection de sections trop laxe (faux positifs) | Élevée | Moyen | La déduplication (matches à < 50 chars d'écart) limite les explosions ; le test `test_smart_truncate_long_chapter_with_sections` doit valider le comportement attendu |
| Mémoire incrémentale tronquée perd un résumé crucial | Faible | Moyen | Stratégie « 3 premiers + N derniers » conserve toujours le cadre théorique initial et le contexte récent ; cas pathologique uniquement pour livres > 30 chapitres (déjà au-delà du seuil de bascule) |
| Régression silencieuse sur la qualité des fiches | Moyenne | Élevé | Comparaison v1/v2 systématique sur livre de référence à l'étape 1 du déploiement |
| Le LLM ignore certaines nouvelles règles du prompt | Élevée | Faible | Vérifications post-assemblage (lot 6.1) détectent et signalent les violations en warnings |
| Coût LLM accru par mémoire cumulative non bornée | Moyenne | Faible | Le bornage à 4000 chars (lot 5) plafonne la croissance |

---

**Fin du plan.**
