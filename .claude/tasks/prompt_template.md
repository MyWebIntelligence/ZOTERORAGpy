# Plan : Création de 2 nouveaux prompts Zotero Notes

**Date** : 2025-12-17
**Statut** : À implémenter

---

## Objectif

Créer **2 nouveaux prompts** pour la génération de fiches de lecture :

1. **Prompt pédagogique L3** — Explication d'article pour étudiants en introduction à la recherche
2. **Prompt grille d'évaluation SHS** — Basé sur les critères des revues internationales top-tier (AJPS, ASR)

---

## Spécifications communes

| Aspect | Valeur |
|--------|--------|
| **Longueur cible** | 1500-3500 mots |
| **Format sortie** | HTML (compatible Zotero) |
| **Usage** | Revue systématique / méta-analyse |
| **max_tokens** | ~10000 |

### Caractéristiques demandées

- Tableau récapitulatif structuré
- Liens explicites avec la problématique de recherche `{PROBLEMATIQUE}`
- Guide de lecture prioritaire (À LIRE / SURVOLER / IGNORER)
- Extraction de citations clés avec numéros de page

### Placeholders disponibles

```
{TITLE}         - Titre de l'article
{AUTHORS}       - Auteurs
{DATE}          - Date publication
{DOI}           - Identifiant DOI
{URL}           - URL de l'article
{PROBLEMATIQUE} - Problématique du projet de recherche
{ABSTRACT}      - Résumé original
{TEXT}          - Texte complet OCR
{LANGUAGE}      - Langue cible (français, English, etc.)
```

---

## Prompt 1 : Pédagogique L3

**Fichier** : `app/utils/zotero_prompt_pedagogique.md`

### Public cible

- Étudiants L3 en introduction à la recherche
- Concepts de base connus (méthodologie, hypothèses, variables)
- Culture disciplinaire acquise
- Jargon technique expliqué systématiquement

### Structure proposée (5 sections)

```
1. FICHE SYNTHÉTIQUE (tableau récapitulatif)
   - Référence APA 7 complète
   - Question de recherche (1 phrase claire)
   - Méthode principale en 3 mots
   - Résultat principal (1 phrase)
   - Pertinence pour {PROBLEMATIQUE} : ★★★★☆/5 avec justification courte

2. CONTEXTE ET ENJEUX (300-400 mots)
   - Le problème sociétal ou scientifique qui motive cette recherche
   - Ce qu'on savait déjà (état de l'art vulgarisé, 2-3 études clés)
   - Le "trou" dans la connaissance que les auteurs veulent combler
   - La question centrale reformulée en langage courant

3. COMMENT ILS ONT FAIT — TUTORIEL MÉTHODOLOGIQUE (1400-1500 mots)

   3.1 Les données (300 mots)
   - Source précise (plateforme, base de données, terrain)
   - Période de collecte et justification
   - Taille de l'échantillon (N=...) et critères de sélection
   - Représentativité : forces et limites

   3.2 Tableau des variables (400 mots)
   | Variable | Définition | Type | Mode de mesure/calcul | Source dans l'article |
   |----------|------------|------|----------------------|----------------------|
   | Ex: Polarisation | Écart entre position et médiane | Continue | Score = |position - médiane| | p.8 |

   Pour CHAQUE variable :
   - Définition accessible (sans jargon)
   - Comment elle est construite/calculée
   - Exemple concret avec valeurs

   3.3 Les méthodes expliquées (500 mots)
   Pour CHAQUE méthode utilisée :

   **[Nom de la méthode]** (ex: Régression logistique)
   - **C'est quoi ?** Explication en 2-3 phrases comme à un L3
   - **À quoi ça sert ici ?** Application concrète dans l'article
   - **Comment ça marche ?** Les grandes étapes (sans formules complexes)
   - **Résultat type** : Comment lire/interpréter l'output (ex: "Un β positif signifie que...")

   3.4 ENCADRÉ : Guide de formation pour reproduire l'étude (300 mots)
   **Prérequis** : Niveau statistique requis, logiciels nécessaires

   **Se former aux méthodes** :
   | Méthode | Ressource gratuite | Livre de référence | Temps estimé |
   |---------|-------------------|-------------------|--------------|
   | Régression | [Cours OpenClassrooms](lien) | Field (2018) | 10h |

   **Étapes de reproduction** :
   1. Obtenir des données similaires via...
   2. Préparer les données avec...
   3. Lancer l'analyse avec [logiciel]...

4. CE QU'ILS ONT TROUVÉ (400-500 mots)
   - Résultats principaux en langage accessible
   - Ce que ça change concrètement dans notre compréhension

   **Tableau synthèse des hypothèses** :
   | Hypothèse | Ce qu'on testait | Résultat | Verdict |
   |-----------|-----------------|----------|---------|
   | H1 | X augmente Y | β=0.34, p<.001 | ✅ Confirmée |
   | H2 | Z modère la relation | Effet non significatif | ❌ Infirmée |

   - Le résultat le plus surprenant ou important (1 paragraphe)

5. POUR ALLER PLUS LOIN (200-300 mots)
   **Études similaires pour approfondir** :
   - [Auteur (Année)](DOI ou lien vérifié) — Utilise la même méthode sur un autre terrain
   - [Auteur (Année)](DOI ou lien vérifié) — Approche complémentaire

   **Concepts à approfondir** :
   - Concept 1 → Voir [ressource pédagogique]
   - Concept 2 → Voir [manuel de référence]
```

### Ton et style

- **Accessible mais rigoureux** : vulgarisation sans perte de précision scientifique
- **"Comme si on expliquait à un.e camarade"** : tutoiement implicite, exemples concrets
- **From scratch** : TOUT définir, AUCUN implicite au-delà du niveau L1
- **Pédagogie active** : questions rhétoriques ("Pourquoi ce choix ?"), analogies du quotidien
- **Vérifiabilité** : chaque affirmation liée à une page ou section de l'article

---

## Prompt 2 : Grille évaluation SHS — Niveau Reviewer International

**Fichier** : `app/utils/zotero_prompt_evaluation.md`

### Philosophie du prompt

Inspiré du **"Manuel de Référence pour l'Évaluation par les Pairs en Sciences Sociales"**, ce prompt adopte la posture d'un **reviewer élite** : auditeur méthodologique, garant de l'éthique, facilitateur de la transparence.

> "L'évaluateur moderne ne peut plus se contenter d'être un lecteur éclairé ; il doit endosser le rôle d'auditeur méthodologique."

### Standards de référence

- **AER** (American Economic Review) — Identification causale
- **ASR** (American Sociological Review) — Cohérence conceptuelle
- **APSR** (American Political Science Review) — Transparence DA-RT
- **Nature Human Behaviour** — Reproductibilité
- **APA JARS-Qual** — Recherche qualitative
- **TOP Guidelines** — Science ouverte
- **COPE** — Éthique de l'évaluation

### Structure proposée (7 sections)

```
1. FICHE D'IDENTIFICATION RAPIDE (tableau synthèse)

   | Champ | Valeur |
   |-------|--------|
   | Référence APA 7 | {AUTHORS} ({DATE}). {TITLE}. DOI: {DOI} |
   | Type d'article | Empirique / Théorique / Méthodologique / Revue |
   | Design | Expérimental / Quasi-expérimental / Observationnel / Qualitatif / Mixte |
   | Données | [Source, N, période, représentativité] |
   | Méthode principale | [Ex: Différences-en-différences avec contrôle synthetique] |
   | Transparence | 🟢 Données + Code / 🟡 Données seules / 🔴 Non disponible |
   | **SCORE GLOBAL** | **X/10** |
   | **VERDICT** | 🟢 À LIRE EN PRIORITÉ / 🟡 À CONSULTER / 🔴 À ÉCARTER |

───────────────────────────────────────────────────────────────

2. ÉVALUATION DE LA CONTRIBUTION (500-600 mots)

   **a) Test du "So What?" — Contribution marginale**
   - Question brutale : "Et alors?" — Pourquoi ce sujet MÉRITE d'être étudié ?
   - Nature de la contribution :
     □ Incrémentale (confirmation/extension)
     □ Substantielle (modification/nuance significative)
     □ Paradigmatique (remise en cause fondamentale)
   - Gap identifié vs Gap combattu : L'auteur démontre-t-il POURQUOI le gap est important ?
   - Comparaison avec 2-3 travaux récents (<5 ans) : déplace-t-il vraiment la frontière ?

   **b) Cohérence conceptuelle (Standard ASR)**
   - Alignement problématique : La question découle-t-elle logiquement de la revue de littérature ?
   - Opérationnalisation : Concepts abstraits → Variables mesurables (justification ?)
   - Architecture logique : Problème → Théorie → Hypothèses → Méthode → Conclusions
   - Contradictions internes détectées ? OUI/NON + détail

   **c) Pertinence pour {PROBLEMATIQUE}**
   - Alignement : DIRECT / INDIRECT (méthode transférable) / TANGENTIEL
   - Apport spécifique pour mon projet : [1-2 phrases]

───────────────────────────────────────────────────────────────

3. AUDIT MÉTHODOLOGIQUE — QUANTITATIF (si applicable, 600-700 mots)

   **a) Stratégie d'identification causale (Standard AER/QJE)**
   - Corrélation vs Causalité : Comment l'endogénéité est-elle traitée ?
   - Variables instrumentales :
     * Restriction d'exclusion justifiée narrativement ? OUI/NON
     * F-statistic rapporté ? Valeur ?
   - DiD (Différences-en-différences) :
     * Tendances parallèles démontrées visuellement ? OUI/NON
     * Test placebo temporel effectué ? OUI/NON
   - Autres designs : RCT / RDD / Matching / Contrôle synthétique

   **b) Détection P-Hacking et HARKing**
   | Red Flag | Présent ? | Commentaire |
   |----------|-----------|-------------|
   | Valeurs p "bumping" (0.049, 0.048) | | Distribution suspecte ? |
   | Degrés de liberté incohérents | | N varie sans explication ? |
   | Analyses multiples non déclarées | | "Garden of forking paths" ? |
   | Arrêt facultatif (stopping rule) | | Échantillon pré-déterminé ? |
   | HARKing suspecté | | Hypothèses post-hoc présentées comme a priori ? |

   **c) Puissance statistique**
   - Justification de N a priori ? OUI (power analysis) / NON
   - Risque d'effet surestimé si étude sous-puissancée
   - Résultats nuls : Intervalle de confiance exclut-il un effet pertinent ?

   **d) Robustesse**
   - Tests de sensibilité effectués ? OUI/NON
   - Formes fonctionnelles alternatives testées ?
   - Outliers traités ?
   - Clustering des erreurs-types approprié ?

───────────────────────────────────────────────────────────────

4. AUDIT MÉTHODOLOGIQUE — QUALITATIF (si applicable, 500-600 mots)

   **a) Congruité méthodologique (Standard JARS-Qual/JBI)**
   - Positionnement épistémologique déclaré ? (constructivisme, réalisme, etc.)
   - Cohérence méthode/épistémologie :
     * Grounded Theory → Codage itératif (ouvert/axial/sélectif) documenté ?
     * Phénoménologie → Épochè (mise entre parenthèses) discutée ?
     * Ethnographie → Réflexivité positionnelle détaillée ?

   **b) Transparence et "Thick Description"**
   - Contexte suffisamment décrit pour juger la transférabilité ?
   - Réflexivité : L'auteur examine-t-il son propre rôle/biais ?
   - Triangulation : Multiples sources (entretiens + observations + documents) ?

   **c) Saturation des données**
   - Critère de saturation explicité ? OUI/NON
   - Preuve narrative (pas juste déclaration "saturation atteinte") ?
   - Cas négatifs/déviants discutés ?

───────────────────────────────────────────────────────────────

5. VÉRIFICATION FORENSIQUE ET TRANSPARENCE (300-400 mots)

   **a) Tests de consistance (si données agrégées)**
   - Test GRIM applicable ? (moyennes sur échelles entières)
   - Incohérences détectées ? OUI/NON
   - Test SPRITE : Distributions plausibles ?

   **b) Science ouverte (TOP Guidelines)**
   | Critère | Statut | Commentaire |
   |---------|--------|-------------|
   | Données disponibles | Lien actif / Sur demande / Non | |
   | Code d'analyse fourni | OUI / Partiel / NON | |
   | Pré-enregistrement | OSF / AsPredicted / Non | |
   | Déviation du protocole déclarée | N/A / OUI / NON | |

   **c) Usage de l'IA**
   - Déclaration d'usage IA ? OUI/NON
   - Signes d'alerte : Hallucinations bibliographiques ? Style générique ?

───────────────────────────────────────────────────────────────

6. FORCES, FAIBLESSES ET NOTATION (400-500 mots)

   **FORCES** (2-3 points majeurs)
   Pour chaque force :
   - Description précise + référence page
   - Standard de la discipline que ça satisfait

   **FAIBLESSES** (2-3 points majeurs)
   Pour chaque faiblesse :
   - Type : Fatale (disqualifiante) / Majeure / Mineure
   - Impact sur validité des conclusions
   - Suggestion d'amélioration

   **TABLEAU DE NOTATION CALIBRÉ** (sévérité AJPS/ASR) :

   | Critère | Note /10 | Justification | Pourquoi pas +1 ? |
   |---------|----------|---------------|-------------------|
   | Originalité/Contribution | | | |
   | Rigueur méthodologique | | | |
   | Clarté argumentative | | | |
   | Validité des conclusions | | | |
   | Transparence/Reproductibilité | | | |
   | Pertinence pour ma recherche | | | |
   | **TOTAL** | **/60** | | |

   Grille de calibration :
   - 1-2 : Disqualifié (erreurs fatales, fraude suspectée)
   - 3-4 : Faible (contribution marginale, failles majeures)
   - **5-6 : Standard** (publiable, majorité des articles)
   - **7-8 : Excellent** (top 20% du champ)
   - 9-10 : Séminal (1-2 par décennie)

───────────────────────────────────────────────────────────────

7. RECOMMANDATIONS ET EXPLOITATION (300-400 mots)

   **Verdict de lecture** :
   - 🟢 **À LIRE EN PRIORITÉ** : Score ≥42/60, contribution directe
   - 🟡 **À CONSULTER** : Score 30-41/60, éléments utiles
   - 🔴 **À ÉCARTER** : Score <30/60, hors sujet ou qualité insuffisante

   **Usages concrets dans ma recherche** :
   - Cadre théorique : Concepts/théories transférables ?
   - Méthodologie : Approche reproductible ? Design adaptable ?
   - Résultats : Hypothèses à tester dans mon contexte ?

   **Citations clés exploitables** (3-5 avec pages) :
   1. "[Verbatim]" (p.X) — Usage : [pour argumenter quoi]
   2. "[Verbatim]" (p.Y) — Usage : [pour justifier quoi]

   **Bibliographie à explorer** (issues de l'article) :
   - Auteur (Année) — Raison : [ex: même méthode, terrain différent]
   - Auteur (Année) — Raison : [ex: critique de cette approche]
```

### Directives spécifiques par discipline

Le prompt doit adapter certains critères selon le champ :

| Discipline | Focus prioritaire | Standard de référence |
|------------|-------------------|----------------------|
| **Économie** | Identification causale, clustering erreurs-types | AER, QJE |
| **Sociologie** | Mécanisme social ("boîte noire"), cas négatifs | ASR, AJS |
| **Science politique** | Transparence DA-RT, validité externe | APSR, IO |
| **Psychologie** | Taille d'effet, puissance, groupe contrôle | APA, Nature HB |

### Ton du prompt

- **Posture** : Reviewer de conférence rang A, pas lecteur bienveillant
- **Question permanente** : "L'article mérite-t-il de modifier mes croyances ?" (perspective bayésienne)
- **Exigence** : Ne JAMAIS présumer l'honnêteté — VÉRIFIER la conformité

### Données empiriques sur l'évaluation par les pairs

Le prompt [EVAL] intègre les résultats de la méta-analyse de 28 revues majeures :

**Fiabilité inter-évaluateurs** :
- ICC moyen = **0.34** (Bornmann, Mutz & Daniel, 2010)
- Kappa de Cohen = **0.17**
- Pour atteindre le seuil acceptable (0.75), il faudrait 6-15 évaluateurs/manuscrit

**Les 5 dimensions consensuelles** (>90% des revues) :
1. **Originalité et contribution** — "transcend subfields" (ASR)
2. **Rigueur méthodologique** — "explicit about research design" (World Politics)
3. **Ancrage littérature** — "engaging the relevant research literature" (AJPS)
4. **Clarté argumentative** — "broader theoretical implications" (EJPR)
5. **Pertinence pour le lectorat**

**Structure optimale du rapport** (consensus des guidelines) :
1. Synthèse de l'argument principal (1-2 phrases)
2. Points forts (même pour manuscrits problématiques)
3. Critiques majeures (numérotées)
4. Critiques mineures (formatage, langue)
5. Recommandation finale

**Biais documentés à expliciter** :
- Biais institutionnel (homophilie de prestige malgré anonymat)
- Biais de genre dans la sélection des reviewers (Helmer et al., 2017)
- "Citation pushing" (suggestions non pertinentes)
- "Cloud of suspicion" (accumulation de critiques mineures)

**Les 15 commandements du reviewer exemplaire** (intégrés dans le prompt) :
1. Évaluer selon l'expertise réelle
2. Déclarer immédiatement tout conflit d'intérêts
3. Maintenir confidentialité absolue
4. Synthétiser l'argument en 1-2 phrases
5. Identifier forces AVANT faiblesses
6. Séparer critiques majeures/mineures (numéroter)
7. Distinguer évaluation vs recommandations
8. Proposer solutions concrètes
9. Évaluer selon les termes du projet (pas imposer sa question)
10. Adapter critères à la méthodologie
11. Éviter le citation pushing
12. Ton professionnel sans langage émotionnel
13. Ne pas corriger la langue (recommander édition si besoin)
14. Respecter les délais ou prévenir
15. Produire rapport rapidement après acceptation

---

## Modifications du code

### Fichiers à créer

```
app/utils/zotero_prompt_pedagogique.md   (À CRÉER)
app/utils/zotero_prompt_evaluation.md    (À CRÉER)
```

### Fichier à modifier : `app/utils/llm_note_generator.py`

**Changement principal** : Refactorer `_load_prompt_template()` pour supporter 5 modes

```python
# Avant (ligne ~137)
def _load_prompt_template(extended_analysis: bool = True) -> str:
    template_filename = "zotero_prompt.md" if extended_analysis else "zotero_prompt_short.md"

# Après
def _load_prompt_template(mode: str = "extended") -> str:
    """
    Load prompt template based on mode.

    Args:
        mode: One of "extended", "short", "pedagogique", "evaluation"

    Returns:
        Prompt template string with placeholders
    """
    TEMPLATE_MAP = {
        "extended": "zotero_prompt.md",
        "short": "zotero_prompt_short.md",
        "pedagogique": "zotero_prompt_pedagogique.md",
        "evaluation": "zotero_prompt_evaluation.md"
    }
    template_filename = TEMPLATE_MAP.get(mode, "zotero_prompt.md")
    # ... reste du code inchangé
```

---

## Interface utilisateur (UI)

### Comportement attendu

Quand l'utilisateur coche **"Extended analysis"** (notes longues), un **menu déroulant** apparaît avec 3 options :

| Option | Code interne | Préfixe en-tête | Description |
|--------|--------------|-----------------|-------------|
| **[FICHE]** | `extended` | `[FICHE]` | Fiche de lecture critique standard (prompt actuel) |
| **[CLAIR]** | `pedagogique` | `[CLAIR]` | Explication pédagogique L3 (tutoriel méthodologique) |
| **[EVAL]** | `evaluation` | `[EVAL]` | Grille d'évaluation reviewer international |

### Préfixes en en-tête des notes

**IMPORTANT** : Chaque note générée doit commencer par son préfixe avant le titre APA.

```html
<!-- ragpy-note-id:uuid -->
<h2>[CLAIR] Smith, J. (2024). Machine Learning for Social Sciences...</h2>
```

Cela permet de :
1. Identifier visuellement le type de fiche dans Zotero
2. Filtrer les notes par type via recherche
3. Distinguer plusieurs fiches pour un même article

### Fichiers UI à modifier

| Fichier | Modification |
|---------|--------------|
| `app/templates/citations.html` | Ajouter menu déroulant conditionnel |
| `app/routes/citations.py` | Récupérer paramètre `note_mode` |
| `app/utils/llm_note_generator.py` | Ajouter préfixe selon mode |

### Logique JavaScript (citations.html)

```javascript
// Quand "Extended analysis" est coché
document.getElementById('extended_analysis').addEventListener('change', function() {
    const dropdown = document.getElementById('note_mode_dropdown');
    dropdown.style.display = this.checked ? 'block' : 'none';
});
```

### Paramètre API

```
POST /generate_zotero_notes
{
    "session_id": 123,
    "extended_analysis": true,
    "note_mode": "pedagogique",  // NEW: "extended" | "pedagogique" | "evaluation"
    "model": "gpt-4o-mini"
}
```

---

**Autres changements code** :

- `_build_prompt()` : Remplacer `extended_analysis: bool` par `mode: str`
- `build_note_html()` : Adapter la signature
- `build_note_html_async()` : Adapter la signature
- Ajuster `max_tokens` selon le mode :
  - `short` : 2000
  - `pedagogique`, `evaluation` : 10000
  - `extended`, `exhaustive` : 16000

---

## Ordre d'implémentation

### Phase 1 : Prompts (priorité haute)

1. [ ] Rédiger `app/utils/zotero_prompt_pedagogique.md`
   - Structure 5 sections avec tutoriel méthodologique
   - Ton L3, from scratch, exemples concrets

2. [ ] Rédiger `app/utils/zotero_prompt_evaluation.md`
   - Grille AJPS/ASR avec notation /50
   - Sévérité calibrée, justifications obligatoires

### Phase 2 : Backend (llm_note_generator.py)

3. [ ] Refactorer `_load_prompt_template(mode: str)`
   - Mapping 4 modes : extended, short, pedagogique, evaluation

4. [ ] Adapter `_build_prompt()` et `build_note_html()`
   - Signature : `extended_analysis: bool` → `mode: str`
   - Ajouter préfixe [FICHE]/[CLAIR]/[EVAL] selon mode

5. [ ] Adapter fonctions async
   - `build_note_html_async()`
   - `build_abstract_text_async()`

6. [ ] Configurer max_tokens par mode :
   - short: 2000
   - pedagogique, evaluation: 10000
   - extended: 16000

### Phase 3 : API et routes (citations.py)

7. [ ] Ajouter paramètre `note_mode` aux endpoints
8. [ ] Gérer rétro-compatibilité `extended_analysis`

### Phase 4 : Interface (optionnel)

9. [ ] Modifier `app/templates/citations.html`
   - Menu déroulant conditionnel [FICHE]/[CLAIR]/[EVAL]
   - JavaScript pour affichage/masquage

### Phase 5 : Tests et documentation

10. [ ] Tester avec un article exemple (les 3 modes)
11. [ ] Mettre à jour `.claude/docs/README_ZOTERO_PROMPT.md`

---

## Notes techniques

- **Rétro-compatibilité** : `extended_analysis=True` → `mode="extended"`, `extended_analysis=False` → `mode="short"`
- **Préfixes** : Ajoutés dans `build_note_html()` APRÈS la génération LLM, dans le HTML final
- **Sentinel** : Format inchangé `<!-- ragpy-note-id:uuid -->`

---

## Structure HTML obligatoire (compatibilité Zotero)

### Balises autorisées

```html
<!-- Tags de structure -->
<h2>...</h2>           <!-- Titre principal (1 seul) -->
<h3>...</h3>           <!-- Sous-sections -->
<p>...</p>             <!-- Paragraphes -->

<!-- Tags de formatage -->
<strong>...</strong>   <!-- Gras (concepts clés) -->
<em>...</em>           <!-- Italique (termes techniques) -->

<!-- Tags de structure complexe -->
<table>
  <thead><tr><th>...</th></tr></thead>
  <tbody><tr><td>...</td></tr></tbody>
</table>
<ul><li>...</li></ul>  <!-- Listes non ordonnées -->
<ol><li>...</li></ol>  <!-- Listes ordonnées -->

<!-- Tags spéciaux -->
<blockquote>...</blockquote>  <!-- Citations verbatim -->
<code>...</code>              <!-- Méthodes/algorithmes -->
```

### Balises INTERDITES

```html
<div>, <span>, <style>, <script>, <img>, <a href="...">
<!-- Zotero les supprime ou les rend mal -->
```

### Structure type d'une note générée

```html
<!-- ragpy-note-id:abc123-def456 -->
<h2>[CLAIR] Smith, J. (2024). Titre de l'article...</h2>

<h3>1. FICHE SYNTHÉTIQUE</h3>
<table>
  <thead><tr><th>Champ</th><th>Valeur</th></tr></thead>
  <tbody>
    <tr><td>Question de recherche</td><td>...</td></tr>
  </tbody>
</table>

<h3>2. CONTEXTE ET ENJEUX</h3>
<p>Premier paragraphe rédigé avec <strong>concepts clés</strong> en gras...</p>
<p>Suite du texte avec <em>termes techniques</em> en italique...</p>

<!-- etc. -->
```

---

## Répartition des mots par section

### [CLAIR] — Prompt pédagogique (Total : 2400-2800 mots)

| Section | Mots | % |
|---------|------|---|
| 1. Fiche synthétique | 100-150 | 5% |
| 2. Contexte et enjeux | 300-400 | 14% |
| 3. Tutoriel méthodologique | 1400-1500 | 54% |
| 4. Résultats | 400-500 | 18% |
| 5. Pour aller plus loin | 200-300 | 9% |
| **TOTAL** | **2400-2850** | 100% |

### [EVAL] — Grille évaluation (Total : 2600-3200 mots, Score /50)

| Section | Mots | % | Contenu clé |
|---------|------|---|-------------|
| 1. Fiche identification | 100-150 | 4% | Tableau récap + score + verdict |
| 2. Contribution | 500-600 | 18% | Test "So What?", cohérence, pertinence |
| 3. Audit quanti | 600-700 | 21% | Identification causale, P-hacking, robustesse |
| 4. Audit quali | 500-600 | 17% | Congruité, thick description, saturation |
| 5. Forensique | 300-400 | 12% | GRIM/SPRITE, TOP Guidelines, IA |
| 6. Synthèse évaluative | 500-600 | 18% | Synthèse argument + Forces + Critiques numérotées + Tableau /50 |
| 7. Recommandations | 300-400 | 12% | Verdict + usages + citations + biblio |
| **TOTAL** | **2800-3450** | 100% | |

**Note** : Sections 3 et 4 mutuellement exclusives (quanti XOR quali), donc total réel ~2200-2750 mots si article mono-méthode.

**Scoring** : 5 dimensions × 10 pts = 50 pts total
- ≥35/50 → 🟢 À LIRE EN PRIORITÉ
- 25-34/50 → 🟡 À CONSULTER
- <25/50 → 🔴 À ÉCARTER

---

## Gestion des cas limites

### Articles courts (<3000 mots source)

```python
# Dans _build_prompt()
if len(text_content) < 3000:
    # Ajouter directive au prompt
    additional_instruction = """
    ATTENTION : Article court détecté.
    - Réduire proportionnellement chaque section de 30%
    - Indiquer clairement les informations MANQUANTES (ex: "Hypothèses : non explicites dans le texte")
    - Ne PAS inventer de contenu absent de l'article
    """
```

### Articles sans méthodologie explicite (essais, tribunes)

Pour [CLAIR] :
- Section 3 devient "L'ARGUMENTATION" (analyse de la structure rhétorique)
- Remplacer "variables" par "concepts mobilisés"

Pour [EVAL] :
- Sections 3-4 fusionnées en "COHÉRENCE ARGUMENTATIVE"
- Notation adaptée (critère "Rigueur méthodologique" → "Rigueur argumentative")

### Articles en langue étrangère

```python
# Détection automatique via {LANGUAGE}
if language != "français":
    # Forcer output en français mais conserver citations en langue originale
    instruction = f"Rédiger la fiche en français. Citations verbatim en {language} suivies de [traduction]."
```

---

## Validation qualité des notes générées

### Checklist automatique (post-génération)

Le système doit vérifier avant envoi à Zotero :

```python
def validate_note_quality(html_content: str, mode: str) -> dict:
    """
    Valide la qualité d'une note générée.

    Returns:
        {"valid": bool, "warnings": list, "errors": list}
    """
    errors = []
    warnings = []

    # 1. Structure HTML valide
    if not html_content.strip().startswith("<!--"):
        errors.append("Sentinel manquant")

    # 2. Préfixe présent
    if mode in ["pedagogique", "evaluation", "extended"]:
        prefix_pattern = r"\[(CLAIR|EVAL|FICHE)\]"
        if not re.search(prefix_pattern, html_content[:500]):
            warnings.append(f"Préfixe [{mode.upper()}] manquant")

    # 3. Sections attendues présentes
    expected_sections = {
        "pedagogique": ["FICHE SYNTHÉTIQUE", "CONTEXTE", "COMMENT", "TROUVÉ", "PLUS LOIN"],
        "evaluation": ["IDENTIFICATION", "CONTRIBUTION", "AUDIT", "FORCES", "RECOMMANDATIONS"]
    }
    if mode in expected_sections:
        for section in expected_sections[mode]:
            if section.upper() not in html_content.upper():
                warnings.append(f"Section '{section}' possiblement absente")

    # 4. Tableaux présents (obligatoires)
    if "<table>" not in html_content:
        if mode in ["pedagogique", "evaluation"]:
            warnings.append("Aucun tableau détecté (attendu dans ce mode)")

    # 5. Longueur raisonnable
    word_count = len(html_content.split())
    min_words = {"short": 300, "pedagogique": 2000, "evaluation": 2200, "extended": 2200}
    max_words = {"short": 800, "pedagogique": 3500, "evaluation": 4000, "extended": 3500}

    if word_count < min_words.get(mode, 1000):
        warnings.append(f"Note courte ({word_count} mots < {min_words[mode]} attendus)")
    if word_count > max_words.get(mode, 5000):
        warnings.append(f"Note longue ({word_count} mots > {max_words[mode]} max)")

    return {
        "valid": len(errors) == 0,
        "word_count": word_count,
        "warnings": warnings,
        "errors": errors
    }
```

---

## Exemples de sortie attendue

### [CLAIR] — Extrait Section 3.2 (Tableau des variables)

```html
<h3>3.2 Tableau des variables</h3>
<table>
  <thead>
    <tr>
      <th>Variable</th>
      <th>Définition</th>
      <th>Type</th>
      <th>Mode de mesure</th>
      <th>Page</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Polarisation affective</strong></td>
      <td>Différence entre l'évaluation de son propre parti et celle du parti adverse, sur une échelle de sympathie 0-100</td>
      <td>Continue</td>
      <td>Score = sentiment_ingroup - sentiment_outgroup. Exemple : si je donne 80 à mon parti et 30 à l'autre, ma polarisation = 50</td>
      <td>p.8</td>
    </tr>
    <tr>
      <td><strong>Exposition médiatique partisane</strong></td>
      <td>Proportion du temps médiatique passé sur des sources alignées avec ses opinions</td>
      <td>Continue [0-1]</td>
      <td>Calculée via tracking navigateur : heures_sites_partisans / heures_total_news</td>
      <td>p.12</td>
    </tr>
  </tbody>
</table>
<p><strong>Exemple concret</strong> : Un participant qui consulte uniquement Fox News aura une exposition = 1.0 (100% partisan), tandis qu'un lecteur équilibré entre Fox et CNN aura ~0.5.</p>
```

### [EVAL] — Extrait Section 3.b (Détection P-Hacking)

```html
<h3>b) Détection P-Hacking et HARKing</h3>
<table>
  <thead>
    <tr>
      <th>Red Flag</th>
      <th>Présent ?</th>
      <th>Commentaire</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Valeurs p "bumping" (0.049, 0.048)</td>
      <td>🔴 OUI</td>
      <td>3 résultats à p=0.047-0.049 (Table 2). Distribution improbable naturellement.</td>
    </tr>
    <tr>
      <td>Degrés de liberté incohérents</td>
      <td>🟢 NON</td>
      <td>N constant à 1,247 dans toutes les analyses.</td>
    </tr>
    <tr>
      <td>Analyses multiples non déclarées</td>
      <td>🟡 POSSIBLE</td>
      <td>6 DVs testées, aucune correction Bonferroni mentionnée.</td>
    </tr>
    <tr>
      <td>HARKing suspecté</td>
      <td>🔴 OUI</td>
      <td>H3 (modération par âge) apparaît uniquement en discussion, absente de l'intro.</td>
    </tr>
  </tbody>
</table>
<p><strong>Verdict P-hacking</strong> : Risque MODÉRÉ à ÉLEVÉ. Recommandation : demander accès aux données pour vérification indépendante.</p>
```

---

## Intégration avec le système existant

### Mapping des noms d'affichage

```python
# Dans citations.py ou llm_note_generator.py
NOTE_MODE_DISPLAY = {
    "extended": "[FICHE] Analyse critique standard",
    "pedagogique": "[CLAIR] Explication pédagogique L3",
    "evaluation": "[EVAL] Grille reviewer international",
    "short": "Résumé court (abstractNote)"
}

NOTE_MODE_PREFIX = {
    "extended": "[FICHE]",
    "pedagogique": "[CLAIR]",
    "evaluation": "[EVAL]",
    "short": ""  # Pas de préfixe pour le mode court
}
```

### Injection du préfixe dans le HTML

```python
def _add_note_prefix(html_content: str, mode: str) -> str:
    """
    Ajoute le préfixe [FICHE]/[CLAIR]/[EVAL] au titre h2.
    """
    prefix = NOTE_MODE_PREFIX.get(mode, "")
    if not prefix:
        return html_content

    # Trouver le premier <h2> et injecter le préfixe
    pattern = r'(<h2>)(.*?)(</h2>)'
    replacement = rf'\1{prefix} \2\3'
    return re.sub(pattern, replacement, html_content, count=1)
```

---

## ANNEXE A : Squelette du prompt [CLAIR] (zotero_prompt_pedagogique.md)

```markdown
# Prompt de génération de fiche pédagogique [CLAIR]

Tu es un enseignant-chercheur expert en vulgarisation scientifique, capable d'expliquer des articles de recherche complexes à des étudiants de Licence 3.

## CONTEXTE

**Titre** : {TITLE}
**Auteurs** : {AUTHORS}
**Date** : {DATE}
**DOI** : {DOI}
**URL** : {URL}
**Ma problématique de recherche** : {PROBLEMATIQUE}

**Résumé original (si disponible)** :
{ABSTRACT}

## TEXTE À ANALYSER

{TEXT}

---

## MISSION

Rédige une **fiche pédagogique explicative** en 5 sections, destinée à un étudiant de L3 qui découvre cet article. Ton objectif : lui permettre de **comprendre** la démarche scientifique ET de **reproduire** l'étude s'il le souhaite.

### Public cible
- Étudiant L3 en sciences humaines/sociales
- Connaît les bases de la méthodologie (hypothèses, variables, échantillon)
- N'est PAS expert du sujet spécifique de l'article
- A besoin que TOUT soit défini et expliqué

### Approche pédagogique
- **Vulgarise sans simplifier à l'excès** : précision scientifique maintenue
- **Définis chaque concept** au premier usage (10-20 mots)
- **Donne des exemples concrets** avec des valeurs réelles
- **Relie à la vie quotidienne** quand possible
- **Cite les pages sources** pour vérifiabilité

---

## STRUCTURE EN 5 SECTIONS

### Section 1 : FICHE SYNTHÉTIQUE (100-150 mots)

Génère un **tableau récapitulatif** avec :

| Champ | Contenu |
|-------|---------|
| **Référence APA 7** | [Format complet] |
| **Question de recherche** | [1 phrase claire, reformulée simplement] |
| **Méthode principale** | [3 mots max : ex. "Enquête longitudinale", "Expérimentation terrain"] |
| **Résultat clé** | [1 phrase : ce que l'étude prouve/montre] |
| **Pertinence pour {PROBLEMATIQUE}** | ★☆☆☆☆ à ★★★★★ + justification en 10 mots |

---

### Section 2 : CONTEXTE ET ENJEUX (300-400 mots)

Réponds à ces questions dans des paragraphes fluides :

1. **Pourquoi cette recherche ?** — Le problème sociétal ou scientifique qui motive l'étude
2. **Ce qu'on savait déjà** — État de l'art simplifié (2-3 études clés citées)
3. **Le "trou" à combler** — Ce que personne n'avait encore étudié/démontré
4. **La question centrale** — Reformulée en langage courant ("En gros, les auteurs se demandent si...")

---

### Section 3 : COMMENT ILS ONT FAIT — TUTORIEL MÉTHODOLOGIQUE (1400-1500 mots)

C'est la section la plus importante. Structure-la ainsi :

#### 3.1 Les données (300 mots)

- **Source** : D'où viennent les données ? (plateforme, base, terrain)
- **Période** : Quand ont-elles été collectées ? Pourquoi ce moment ?
- **Taille** : Combien d'observations/participants ? (N = ...)
- **Sélection** : Comment les cas ont-ils été choisis ? Critères d'inclusion/exclusion
- **Représentativité** : Peut-on généraliser ? Forces et limites

#### 3.2 Tableau des variables (400 mots)

Présente TOUTES les variables importantes sous forme de tableau :

| Variable | Définition | Type | Mode de mesure/calcul | Page |
|----------|------------|------|----------------------|------|
| [Nom] | [Explication accessible] | [Continue/Catégorielle/Binaire] | [Comment c'est calculé + exemple avec valeurs] | p.X |

Pour chaque variable :
- Définition en langage courant (pas de jargon)
- Exemple concret avec des chiffres réels
- Si c'est un indice composite, détaille les composantes

#### 3.3 Les méthodes expliquées (500 mots)

Pour CHAQUE méthode statistique/analytique utilisée, génère un bloc :

**[NOM DE LA MÉTHODE]** (ex: Régression linéaire multiple)

- **C'est quoi ?** [2-3 phrases d'explication comme à un ami]
- **À quoi ça sert ici ?** [Application concrète dans cet article]
- **Comment ça marche ?** [Étapes principales, sans formules complexes]
- **Comment lire le résultat ?** [Ex: "Un coefficient β positif signifie que quand X augmente, Y augmente aussi"]

#### 3.4 ENCADRÉ : Guide pour reproduire l'étude (300 mots)

**Prérequis pour reproduire** :
- Niveau statistique : [Débutant / Intermédiaire / Avancé]
- Logiciels : [R, Python, SPSS, Stata...]
- Données similaires disponibles : [OUI/NON + où les trouver]

**Se former aux méthodes utilisées** :

| Méthode | Ressource gratuite | Manuel de référence | Temps estimé |
|---------|-------------------|---------------------|--------------|
| [Méthode] | [Lien/cours] | [Livre classique] | [Xh] |

**Étapes de reproduction** :
1. Obtenir des données similaires via [source]
2. Préparer les données : [nettoyage, recodage]
3. Lancer l'analyse avec [logiciel] : [code/menu]
4. Interpréter : [ce qu'il faut regarder]

---

### Section 4 : CE QU'ILS ONT TROUVÉ (400-500 mots)

#### Résultats principaux

Résume les découvertes en langage accessible. Pas de jargon statistique brut.

#### Tableau synthèse des hypothèses

| Hypothèse | Ce qu'on testait | Résultat statistique | Verdict |
|-----------|-----------------|---------------------|---------|
| H1 | [Formulation simple] | [β=X, p<.05 ou équivalent] | ✅ Confirmée / ❌ Infirmée / ⚠️ Nuancée |
| H2 | [...] | [...] | [...] |

#### Le résultat le plus important

[1 paragraphe] — Qu'est-ce que cet article change dans notre compréhension du sujet ? Pourquoi c'est intéressant ?

---

### Section 5 : POUR ALLER PLUS LOIN (200-300 mots)

**Concepts clés à retenir** (3 max) :
- **[Concept 1]** : [Définition + pourquoi c'est important]
- **[Concept 2]** : [...]

**Citations utiles pour mes travaux** (2-3 avec pages) :
1. "[Verbatim exact]" (p.X) — Utilisable pour [argumenter quoi]
2. "[...]" (p.Y) — Utilisable pour [...]

**Verdict de lecture** :
- 🟢 **ESSENTIEL** / 🟡 **UTILE** / 🔴 **OPTIONNEL**
- Justification en 1 phrase

**Références pour approfondir** :
- [Auteur (Année)] — [Pourquoi lire cet article]
- [Auteur (Année)] — [...]

---

## CONTRAINTES

- **Longueur totale** : 2400-2800 mots
- **Format** : HTML (balises h2, h3, p, table, strong, em, blockquote)
- **Langue** : {LANGUAGE}
- **Ton** : Pédagogique, bienveillant mais rigoureux
- **Vérifiabilité** : Chaque affirmation renvoie à une page de l'article

## FORMAT DE SORTIE

Génère directement le HTML sans préambule. Commence par :
<h2>{AUTHORS} ({DATE}). {TITLE}...</h2>
```

---

## ANNEXE B : Squelette du prompt [EVAL] (zotero_prompt_evaluation.md)

```markdown
# Prompt de génération de grille d'évaluation [EVAL]

Tu es un **reviewer élite** pour une revue internationale de rang A (AJPS, ASR, AER). Tu adoptes une posture d'**auditeur méthodologique** : exigeant, sceptique, mais constructif.

> "L'évaluateur moderne ne peut plus se contenir d'être un lecteur éclairé ; il doit endosser le rôle d'auditeur méthodologique."

## DONNÉES EMPIRIQUES SUR L'ÉVALUATION PAR LES PAIRS

Cette grille s'appuie sur la méta-analyse de 28 revues majeures en sciences sociales :

**Fiabilité inter-évaluateurs** (Bornmann, Mutz & Daniel, 2010) :
- ICC moyen = 0.34, Kappa de Cohen = 0.17 — fiabilité "quite limited"
- Implication : L'évaluation est un jugement humain imparfait → expliciter TOUJOURS le raisonnement

**Les 5 dimensions consensuelles** (présentes dans >90% des guidelines) :
1. Originalité et contribution — "transcend subfields" (ASR)
2. Rigueur méthodologique — "explicit about research design" (World Politics)
3. Ancrage dans la littérature — "engaging the relevant research literature" (AJPS)
4. Clarté argumentative — "broader theoretical implications" (EJPR)
5. Pertinence pour le lectorat

**Biais documentés à éviter** :
- Ne PAS imposer ta propre question de recherche ("The Real Issue Is...")
- Ne PAS accumuler de critiques mineures créant un "cloud of suspicion"
- Ne PAS pratiquer le "citation pushing" (suggestions non pertinentes)
- Évaluer selon les termes du projet, pas selon tes préférences

## CONTEXTE

**Titre** : {TITLE}
**Auteurs** : {AUTHORS}
**Date** : {DATE}
**DOI** : {DOI}
**URL** : {URL}
**Ma problématique de recherche** : {PROBLEMATIQUE}

**Résumé original (si disponible)** :
{ABSTRACT}

## TEXTE À ANALYSER

{TEXT}

---

## MISSION

Produis une **évaluation critique calibrée** de cet article, selon les standards des meilleures revues internationales. Ton objectif : aider le lecteur à décider si cet article **mérite de modifier ses croyances** sur le sujet.

### Posture d'évaluation (15 commandements du reviewer exemplaire)

1. Synthétiser l'argument en 1-2 phrases (démontrer compréhension)
2. Identifier forces AVANT faiblesses (même pour manuscrits problématiques)
3. Séparer critiques majeures (numérotées) et mineures
4. Distinguer évaluation ("ce qui fonctionne") vs recommandations ("comment améliorer")
5. Proposer des solutions concrètes, pas des critiques abstraites
6. Évaluer selon les termes du projet, sans imposer ta propre question
7. Adapter les critères à la méthodologie (quanti/quali/mixte/théorique)
8. Éviter le citation pushing — ne suggérer que des références vraiment pertinentes
9. Utiliser un ton professionnel sans langage émotionnel ou dénigrant
10. Perspective bayésienne : "Cet article m'apporte-t-il assez d'évidence pour changer d'avis ?"

### Calibration sévère (consensus AJPS/ASR)
- **1-2/10** : Disqualifié (erreurs fatales, fraude suspectée)
- **3-4/10** : Faible (contribution marginale, failles majeures)
- **5-6/10** : Standard (publiable, majorité des articles)
- **7-8/10** : Excellent (top 20% du champ)
- **9-10/10** : Séminal (1-2 par décennie)

### Standards de référence par discipline

| Discipline | Focus prioritaire | Revue de référence |
|------------|-------------------|-------------------|
| Économie | Identification causale, clustering erreurs-types | AER, QJE |
| Sociologie | Mécanisme social ("boîte noire"), cas négatifs | ASR, AJS |
| Science politique | Transparence DA-RT, validité externe | APSR, AJPS |
| Psychologie | Taille d'effet, puissance, groupe contrôle | Nature HB, APA |
| Méthodes quali | Congruité épistémologique, thick description | JARS-Qual, JBI |

---

## STRUCTURE EN 7 SECTIONS

### Section 1 : FICHE D'IDENTIFICATION RAPIDE (tableau)

| Champ | Valeur |
|-------|--------|
| **Référence APA 7** | {AUTHORS} ({DATE}). {TITLE}. DOI: {DOI} |
| **Type d'article** | Empirique / Théorique / Méthodologique / Revue de littérature |
| **Design** | Expérimental / Quasi-expérimental / Observationnel / Qualitatif / Mixte |
| **Données** | [Source, N, période, représentativité en 1 ligne] |
| **Méthode principale** | [Technique analytique centrale] |
| **Transparence** | 🟢 Données + Code / 🟡 Données seules / 🔴 Non disponible |
| **SCORE GLOBAL** | **X/50** (5 dimensions × 10 pts) |
| **VERDICT** | 🟢 À LIRE EN PRIORITÉ (≥35) / 🟡 À CONSULTER (25-34) / 🔴 À ÉCARTER (<25) |

---

### Section 2 : ÉVALUATION DE LA CONTRIBUTION (500-600 mots)

#### a) Test du "So What?" — Contribution marginale

- **Question brutale** : "Et alors ?" — Pourquoi ce sujet MÉRITE d'être étudié ?
- **Nature de la contribution** :
  - □ Incrémentale (confirmation/extension mineure)
  - □ Substantielle (modification/nuance significative)
  - □ Paradigmatique (remise en cause fondamentale)
- **Gap identifié vs Gap combattu** : L'auteur démontre-t-il POURQUOI le gap est important ?
- **Comparaison** : Par rapport à 2-3 travaux récents (<5 ans), déplace-t-il vraiment la frontière ?

#### b) Cohérence conceptuelle (Standard ASR)

- **Alignement problématique** : La question découle-t-elle logiquement de la revue de littérature ?
- **Opérationnalisation** : Passage concepts abstraits → variables mesurables justifié ?
- **Architecture logique** : Problème → Théorie → Hypothèses → Méthode → Conclusions (chaîne cohérente ?)
- **Contradictions internes** : OUI / NON — Si oui, lesquelles ?

#### c) Pertinence pour {PROBLEMATIQUE}

- **Alignement** : DIRECT / INDIRECT (méthode transférable) / TANGENTIEL
- **Apport spécifique** : [1-2 phrases sur l'utilité pour ma recherche]

---

### Section 3 : AUDIT MÉTHODOLOGIQUE — QUANTITATIF (600-700 mots)

*[GÉNÉRER CETTE SECTION UNIQUEMENT SI L'ARTICLE UTILISE DES MÉTHODES QUANTITATIVES]*

#### a) Stratégie d'identification causale (Standard AER/QJE)

- **Corrélation vs Causalité** : Comment l'endogénéité est-elle traitée ?
- **Variables instrumentales** (si utilisées) :
  - Restriction d'exclusion justifiée narrativement ? OUI/NON
  - F-statistic rapporté ? Valeur : ___
  - Instruments faibles suspectés ? OUI/NON
- **Différences-en-différences** (si utilisées) :
  - Tendances parallèles démontrées visuellement ? OUI/NON
  - Test placebo temporel effectué ? OUI/NON
  - Anticipation/spillovers discutés ? OUI/NON
- **Autres designs** : RCT / RDD / Matching / Contrôle synthétique — Hypothèses vérifiées ?

#### b) Détection P-Hacking et HARKing

| Red Flag | Présent ? | Commentaire |
|----------|-----------|-------------|
| Valeurs p "bumping" (0.049, 0.048...) | 🟢/🟡/🔴 | [Distribution suspecte ?] |
| Degrés de liberté incohérents | 🟢/🟡/🔴 | [N varie sans explication ?] |
| Analyses multiples non déclarées | 🟢/🟡/🔴 | [Correction appliquée ?] |
| Arrêt facultatif (stopping rule) | 🟢/🟡/🔴 | [Échantillon pré-déterminé ?] |
| HARKing suspecté | 🟢/🟡/🔴 | [Hypothèses post-hoc présentées comme a priori ?] |

**Verdict P-hacking** : Risque FAIBLE / MODÉRÉ / ÉLEVÉ

#### c) Puissance statistique

- Justification de N a priori (power analysis) ? OUI / NON
- Risque d'effet surestimé si étude sous-puissancée
- Résultats nuls : L'intervalle de confiance exclut-il un effet pertinent ?

#### d) Robustesse

- Tests de sensibilité effectués ? OUI / NON — Lesquels ?
- Formes fonctionnelles alternatives testées ?
- Outliers traités ? Comment ?
- Clustering des erreurs-types approprié au design ?

---

### Section 4 : AUDIT MÉTHODOLOGIQUE — QUALITATIF (500-600 mots)

*[GÉNÉRER CETTE SECTION UNIQUEMENT SI L'ARTICLE UTILISE DES MÉTHODES QUALITATIVES]*

#### a) Congruité méthodologique (Standard JARS-Qual / JBI)

- **Positionnement épistémologique** déclaré ? (constructivisme, réalisme, pragmatisme...)
- **Cohérence méthode/épistémologie** :
  - Grounded Theory → Codage itératif (ouvert/axial/sélectif) documenté ?
  - Phénoménologie → Épochè (mise entre parenthèses) discutée ?
  - Ethnographie → Réflexivité positionnelle détaillée ?
  - Analyse thématique → Approche inductive/déductive clarifiée ?

#### b) Transparence et "Thick Description"

- Contexte suffisamment décrit pour juger la **transférabilité** ?
- **Réflexivité** : L'auteur examine-t-il son propre rôle/biais ?
- **Triangulation** : Multiples sources (entretiens + observations + documents) ?

#### c) Saturation des données

- Critère de **saturation** explicité ? OUI / NON
- Preuve narrative (pas juste déclaration "saturation atteinte") ?
- **Cas négatifs/déviants** discutés ?

---

### Section 5 : VÉRIFICATION FORENSIQUE ET TRANSPARENCE (300-400 mots)

#### a) Tests de consistance (si données agrégées disponibles)

- **Test GRIM** applicable ? (moyennes sur échelles entières)
- Incohérences détectées ? OUI / NON — Détails
- **Test SPRITE** : Distributions plausibles ?

#### b) Science ouverte (TOP Guidelines)

| Critère | Statut | Commentaire |
|---------|--------|-------------|
| Données disponibles | Lien actif / Sur demande / NON | |
| Code d'analyse fourni | OUI / Partiel / NON | |
| Pré-enregistrement | OSF / AsPredicted / NON | |
| Déviation du protocole déclarée | N/A / OUI / NON | |
| Matériaux disponibles | OUI / NON | |

#### c) Usage de l'IA

- Déclaration d'usage IA ? OUI / NON
- Signes d'alerte : Hallucinations bibliographiques ? Style générique ?

---

### Section 6 : SYNTHÈSE ÉVALUATIVE ET NOTATION (500-600 mots)

*Cette section suit la structure optimale du rapport d'évaluation (consensus 28 revues)*

#### a) Synthèse de l'argument (OBLIGATOIRE — 2-3 phrases)

Reformule l'argument principal et la contribution visée pour démontrer une compréhension juste du manuscrit. Cette synthèse aide à contextualiser les critiques qui suivent.

> "[L'article soutient que... en utilisant... pour démontrer que...]"

#### b) FORCES (2-3 points majeurs — même pour manuscrits problématiques)

Selon Brunsma, Prasad & Zuckerman (2013) : "it is better to write the review with the 'forest' in mind".

Pour chaque force :
- Description précise + référence page
- Standard de la discipline que ça satisfait (ex: "Satisfait le critère AER d'identification causale")
- Pourquoi c'est une contribution à retenir

#### c) CRITIQUES MAJEURES (numérotées — maximum 3)

**Règle Ferree (2004)** : "The core of the review should identify whether the research question contributes to larger theory, whether the analysis actually answers the research question, and whether the conclusions flow from the analyses."

Pour chaque critique majeure :

**[C1] Titre de la critique**
- **Nature** : Théorique / Méthodologique / Analytique / Interprétative
- **Gravité** : 🔴 Fatale (disqualifiante) / 🟠 Majeure / 🟡 Mineure
- **Problème** : [Description précise avec page]
- **Impact** : [Conséquence sur la validité des conclusions]
- **Solution proposée** : [Recommandation concrète, pas critique abstraite]

#### d) Critiques mineures (non numérotées — liste brève)

Formatage, jargon, erreurs ponctuelles. NE PAS corriger l'orthographe/grammaire — recommander un service d'édition si nécessaire.

#### e) TABLEAU DE NOTATION CALIBRÉ (5 dimensions consensuelles)

| Dimension | Note /10 | Justification (1 phrase) | Pourquoi pas +1 ? |
|-----------|----------|-------------------------|-------------------|
| **Originalité/Contribution** | | "transcend subfields" ? | |
| **Rigueur méthodologique** | | Design explicite ? | |
| **Ancrage littérature** | | Engaging relevant research ? | |
| **Clarté argumentative** | | Broader implications ? | |
| **Pertinence pour {PROBLEMATIQUE}** | | Apport direct ? | |
| **TOTAL** | **/50** | | |

**Rappel calibration** (temps moyen d'évaluation : 3,4h/manuscrit selon ASA) :
- 1-2/10 : Disqualifié (erreurs fatales, fraude suspectée)
- 3-4/10 : Faible (contribution marginale, failles majeures)
- **5-6/10 : Standard** (publiable, représente la majorité des articles)
- **7-8/10 : Excellent** (top 20% du champ, "one of the few articles from this subfield that others should know about")
- 9-10/10 : Séminal (1-2 par décennie, change le paradigme)

---

### Section 7 : RECOMMANDATIONS ET EXPLOITATION (300-400 mots)

#### Verdict de lecture (basé sur score /50)

- 🟢 **À LIRE EN PRIORITÉ** : Score ≥35/50, contribution directe pour {PROBLEMATIQUE}
- 🟡 **À CONSULTER** : Score 25-34/50, éléments utiles (méthode ou théorie transférable)
- 🔴 **À ÉCARTER** : Score <25/50, hors sujet ou qualité insuffisante

#### Usages concrets dans ma recherche

- **Cadre théorique** : Concepts/théories transférables ?
- **Méthodologie** : Approche reproductible ? Design adaptable ?
- **Résultats** : Hypothèses à tester dans mon contexte ?

#### Citations clés exploitables (3-5 avec pages)

1. "[Verbatim exact]" (p.X) — Usage : [pour argumenter/justifier quoi]
2. "[...]" (p.Y) — Usage : [...]
3. "[...]" (p.Z) — Usage : [...]

#### Bibliographie à explorer (issues de l'article)

- [Auteur (Année)] — Raison : [ex: même méthode, terrain différent]
- [Auteur (Année)] — Raison : [ex: critique de cette approche]

---

## CONTRAINTES

- **Longueur totale** : 2200-2750 mots (sections quanti/quali mutuellement exclusives)
- **Format** : HTML (balises h2, h3, p, table, strong, em, blockquote)
- **Langue** : {LANGUAGE}
- **Ton** : Exigeant, sceptique, constructif (reviewer rang A)
- **Calibration** : Sévérité AJPS/ASR — 5/10 = standard, 7/10 = excellent
- **Scoring** : 5 dimensions × 10 pts = /50 (pas /60)
- **Structure obligatoire** : Synthèse argument → Forces → Critiques majeures numérotées → Critiques mineures → Notation

## RAPPEL : LES 15 COMMANDEMENTS À RESPECTER

1. Synthétiser l'argument en 1-2 phrases
2. Identifier forces AVANT faiblesses
3. Séparer critiques majeures (numérotées) et mineures
4. Distinguer évaluation vs recommandations
5. Proposer solutions concrètes
6. Évaluer selon les termes du projet
7. Adapter critères à la méthodologie
8. Éviter le citation pushing
9. Ton professionnel sans émotionnel
10. Perspective bayésienne ("mérite-t-il de modifier mes croyances ?")

## FORMAT DE SORTIE

Génère directement le HTML sans préambule. Commence par :
<h2>{AUTHORS} ({DATE}). {TITLE}...</h2>
```
