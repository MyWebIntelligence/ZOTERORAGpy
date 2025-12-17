# Système de Génération de Fiches Zotero

## 📝 Vue d'ensemble

RAGpy propose **4 modes de génération** de fiches de lecture académiques via LLM. Chaque mode est optimisé pour un usage spécifique et utilise son propre template de prompt.

## 🎯 Les 4 Modes Disponibles

| Mode | Fichier template | Préfixe | Longueur | Usage |
|------|------------------|---------|----------|-------|
| **extended** | `zotero_prompt.md` | `[FICHE]` | 2500-3000 mots | Méta-analyse méthodologique complète |
| **pedagogique** | `zotero_prompt_pedagogique.md` | `[CLAIR]` | 2400-2800 mots | Tutoriel pour étudiants L3 |
| **evaluation** | `zotero_prompt_evaluation.md` | `[EVAL]` | 2200-2750 mots | Grille d'évaluation peer review |
| **short** | `zotero_prompt_short.md` | (aucun) | 400-600 mots | Enrichit le champ Abstract |

### Mode Extended [FICHE] (défaut)

**Objectif** : Fiche de lecture critique pour méta-analyse.

**Structure** (6 sections) :
1. Référence complète APA 7
2. Pertinence et positionnement (250-350 mots)
3. Architecture théorique et conceptuelle (500-650 mots)
4. Design méthodologique essentiel (450-550 mots)
5. Résultats et validation (450-550 mots)
6. Évaluation critique calibrée avec notation /30

**max_tokens** : 16000

### Mode Pédagogique [CLAIR]

**Objectif** : Fiche explicative pour étudiants L3 en introduction à la recherche.

**Public cible** : Étudiants L3 avec bases méthodologiques mais non experts du sujet.

**Structure** (5 sections) :
1. Fiche synthétique (tableau récapitulatif)
2. Contexte et enjeux (300-400 mots)
3. **Tutoriel méthodologique** (1400-1500 mots) — Section centrale
   - Les données utilisées
   - Tableau des variables avec définitions accessibles
   - Méthodes expliquées simplement
   - Guide pour reproduire l'étude
4. Ce qu'ils ont trouvé (400-500 mots)
5. Pour aller plus loin (200-300 mots)

**max_tokens** : 10000

### Mode Évaluation [EVAL]

**Objectif** : Grille d'évaluation basée sur les standards des revues internationales (AJPS, ASR, AER).

**Posture** : Reviewer élite, auditeur méthodologique, sceptique mais constructif.

**Structure** (7 sections) :
1. Fiche d'identification rapide (tableau)
2. Évaluation de la contribution (500-600 mots)
3. Audit méthodologique quantitatif (600-700 mots) — Si applicable
4. Audit méthodologique qualitatif (500-600 mots) — Si applicable
5. Vérification forensique et transparence (300-400 mots)
6. Synthèse évaluative avec notation /50 (500-600 mots)
7. Recommandations et exploitation (300-400 mots)

**Système de notation** :
| Dimension | Note /10 |
|-----------|----------|
| Originalité/Contribution | |
| Rigueur méthodologique | |
| Ancrage littérature | |
| Clarté argumentative | |
| Pertinence thématique | |
| **TOTAL** | **/50** |

**Calibration** (sévérité AJPS/ASR) :
- 1-2/10 : Disqualifié (erreurs fatales)
- 3-4/10 : Faible (contribution marginale)
- **5-6/10** : Standard (majorité des articles publiables)
- **7-8/10** : Excellent (top 20% du champ)
- 9-10/10 : Séminal (1-2 par décennie)

**max_tokens** : 10000

### Mode Short (Résumé court)

**Objectif** : Enrichir le champ `abstractNote` de Zotero avec un résumé structuré.

**Structure** (4 chapitres) :
1. Cadrage et problématique
2. Cadre théorique et concepts clés
3. Démarche méthodologique
4. Résultats, apports et limites

**Format** : Texte brut (pas de HTML)

**max_tokens** : 2000

## 🔧 Comment ça fonctionne

1. **Sélection** : L'utilisateur choisit le mode via le dropdown "Type de fiche" dans l'interface
2. **Chargement** : Le système charge le template correspondant depuis `app/utils/`
3. **Remplacement** : Les placeholders `{VARIABLE}` sont remplacés par les vraies valeurs
4. **Génération** : Le prompt est envoyé au LLM avec les `max_tokens` appropriés
5. **Préfixage** : Le préfixe (`[FICHE]`, `[CLAIR]`, `[EVAL]`) est ajouté au titre H2

## 📋 Placeholders Disponibles

Tous les templates supportent ces placeholders :

| Placeholder | Description | Exemple |
|------------|-------------|---------|
| `{TITLE}` | Titre de l'article | "Machine Learning for NLP" |
| `{AUTHORS}` | Auteurs | "Smith, J.; Doe, M." |
| `{DATE}` | Date de publication | "2024" |
| `{DOI}` | Digital Object Identifier | "10.1234/example" |
| `{URL}` | URL de l'article | "https://..." |
| `{ABSTRACT}` | Résumé de l'article | "This paper presents..." |
| `{TEXT}` | Texte complet extrait par OCR | Jusqu'à 100k caractères |
| `{LANGUAGE}` | Langue cible | "français", "English", etc. |
| `{PROBLEMATIQUE}` | Problématique de recherche | Titre + description du projet (ex: "Mon Projet — Étude sur...") |

## 🖥️ Interface Utilisateur

Le dropdown "Type de fiche" dans la section "Zotero Notes" permet de sélectionner le mode :

```
📚 Fiche de lecture [FICHE] — Analyse complète (2500-3000 mots)
🎓 Fiche pédagogique [CLAIR] — Pour étudiants L3 (2400-2800 mots)
📊 Grille d'évaluation [EVAL] — Peer review SHS (2200-2750 mots)
📝 Résumé court — Enrichit le champ Abstract (400-600 mots)
```

## 🔄 API et Routes

### Endpoint

```
POST /generate_zotero_notes_sse
```

### Paramètres

| Paramètre | Type | Valeurs | Description |
|-----------|------|---------|-------------|
| `session` | string | - | Nom de la session |
| `note_mode` | string | `extended`, `pedagogique`, `evaluation`, `short` | Mode de génération |
| `model` | string | `gpt-4o-mini`, `google/gemini-2.5-flash`, etc. | Modèle LLM (optionnel) |

### Rétrocompatibilité

L'ancien paramètre `extended_analysis` (`true`/`false`) est toujours supporté :
- `true` → `note_mode=extended`
- `false` → `note_mode=short`

## ✏️ Personnalisation des Templates

Chaque template peut être modifié directement :

| Mode | Fichier |
|------|---------|
| Extended | [app/utils/zotero_prompt.md](../../app/utils/zotero_prompt.md) |
| Pédagogique | [app/utils/zotero_prompt_pedagogique.md](../../app/utils/zotero_prompt_pedagogique.md) |
| Évaluation | [app/utils/zotero_prompt_evaluation.md](../../app/utils/zotero_prompt_evaluation.md) |
| Short | [app/utils/zotero_prompt_short.md](../../app/utils/zotero_prompt_short.md) |

**Note** : Les fichiers sont rechargés à chaque génération. Aucun redémarrage nécessaire.

## 🎯 Bonnes Pratiques

### ✅ À faire

- **Structurer clairement** : Sections bien définies avec longueurs spécifiées
- **Format HTML** : Toujours demander HTML simplifié pour les modes non-short
- **Langue explicite** : Utiliser `{LANGUAGE}` pour multilingue
- **Instructions précises** : Quantifier les attentes (nombre de mots, nombre de points)

### ❌ À éviter

- **Prompts trop longs** : Le LLM peut perdre le fil
- **Instructions contradictoires** : "Sois bref" + "Détaille tout"
- **Placeholders inventés** : Seuls ceux listés ci-dessus fonctionnent
- **HTML complexe** : Éviter CSS, JavaScript, tableaux imbriqués

## 🆘 Dépannage

### Le mode n'est pas pris en compte

- Vérifiez que le fichier template existe dans `app/utils/`
- Consultez les logs : `logs/app.log` pour les erreurs
- Vérifiez le paramètre `note_mode` dans la requête

### Les placeholders ne sont pas remplacés

- Orthographe exacte : `{TITLE}` (majuscules)
- Accolades correctes : `{` et `}` (pas d'espaces)

### Fallback automatique

Si un template est introuvable, le système utilise le mode `extended` par défaut avec un warning dans les logs.

## 📚 Ressources

- [Documentation OpenAI Prompting](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/claude/docs/prompt-engineering)
- [Guidelines AJPS pour reviewers](https://ajps.org/reviewer-guidelines/)
- [Standards ASR pour évaluation](https://www.asanet.org/asr-submission-guidelines)

---

**Mise à jour** : 2025-12-17 — Ajout des modes `pedagogique` [CLAIR] et `evaluation` [EVAL]
