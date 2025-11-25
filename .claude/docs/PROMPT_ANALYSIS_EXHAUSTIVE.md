# Système d'Analyse Académique Exhaustive - Documentation

## Vue d'ensemble

Le système d'analyse académique exhaustive de RAGpy génère des fiches de lecture complètes et structurées à partir d'articles scientifiques en utilisant des LLM (Large Language Models). Contrairement au système précédent limité à 200-300 mots, cette version produit des analyses approfondies sans limite de longueur.

### Caractéristiques principales

- **Analyse exhaustive** : Pas de limite de longueur, le LLM décide de la profondeur nécessaire
- **Structure académique** : 6 sections principales avec sous-sections détaillées
- **Format HTML** : Compatible avec Zotero (balises `<h2>`, `<h3>`, `<p>`, `<table>`, `<ul>`, etc.)
- **Texte intégral** : Analyse de tout le texte OCR sans troncature
- **Problématique personnalisée** : Support du placeholder `{PROBLEMATIQUE}` pour contextualiser l'analyse
- **Multilingue** : Support du français, anglais, espagnol, allemand, italien, portugais

---

## Structure de l'Analyse

### Section 1 : Introduction

**Balise** : `<h2>Introduction</h2>`

**Contenu** :
- `<h3>Auteurs : informations principales</h3>` : Tableau HTML avec prénom, nom, fonction, institution, ville, pays, champs d'expertise
- `<h3>Contexte scientifique de la recherche</h3>` : Positionnement de l'article dans le paysage académique
- `<h3>Motivation et enjeux de l'article</h3>` : Raisons et importance de la recherche
- `<h3>Problématique, hypothèses, objectifs</h3>` : Questions de recherche et cadre théorique
- `<h3>Structure et plan de l'article</h3>` : Résumé synthétique par partie

**Exemple de tableau auteurs** :
```html
<table>
  <thead>
    <tr>
      <th>Prénom</th>
      <th>Nom</th>
      <th>Fonction</th>
      <th>Institution</th>
      <th>Ville</th>
      <th>Pays</th>
      <th>Expertise</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Marie</td>
      <td>Dupont</td>
      <td>Professeure</td>
      <td>Université Paris-Saclay</td>
      <td>Paris</td>
      <td>France</td>
      <td>NLP, Machine Learning</td>
    </tr>
  </tbody>
</table>
```

### Section 2 : État de l'art

**Balise** : `<h2>I. État de l'art</h2>`

**Contenu** :
- Présentation fine et synthétique des différentes approches, écoles, ou modèles mobilisés
- Perspective bayésienne pour évaluer la robustesse des théories (forces, limites, débats actuels)
- Liste claire des concepts, relations, et sources citées
- Rôle, portée et éventuelles controverses de chaque théorie ou référence
- Apports et limites dans la compréhension du problème

**Objectif** : Fournir un panorama critique et structuré des fondements théoriques de l'article.

### Section 3 : Méthodologie, Données, Grille d'analyse

**Balise** : `<h2>II. Méthodologie, Données, Grille d'analyse</h2>`

**Sous-sections** :

#### a) Description méthodologique

- Type de données, origine et procédures d'extraction
- Grille d'analyse, variables/catégories clés, logiques d'articulation
- Outils et algorithmes utilisés (machine learning, NLP, etc.)
- Justification du choix méthodologique
- Lien logique entre méthode employée et hypothèses/problématique
- Reproductibilité : accès aux données, scripts/code, etc.

#### b) Production et traitement des données

- Méthodes d'échantillonnage et définitions opérationnelles
- Techniques de nettoyage et d'annotation
- Étapes-clés de l'analyse
- Principaux résultats intermédiaires et interprétations partielles

### Section 4 : Résultats et conclusions

**Balise** : `<h2>III. Résultats et conclusions</h2>`

**Contenu** :
- Résumé des principales réponses apportées à la problématique
- Hiérarchisation des apports-clefs
- Structure conceptuelle (concepts, paramètres, corrélations, causalités)
- Implications théoriques et pratiques
- Distinction entre ce qui a été confirmé/infirmé
- Apports à l'état de l'art (progrès, nouvelles pistes)

### Section 5 : Notes d'évaluation

**Balise** : `<h2>Notes d'évaluation</h2>`

**Format** : Tableau HTML avec notes (0 à 10) et justification précise

**Critères évalués** :
- **Pertinence** : Qualité de l'éclairage de la problématique
- **Qualité scientifique** : Respect des règles et standards internationaux
- **Légitimité des auteurs** : Niveau d'expertise et rigueur

**Exemple** :
```html
<table>
  <thead>
    <tr>
      <th>Critère</th>
      <th>Note</th>
      <th>Justification</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Pertinence</td>
      <td>9/10</td>
      <td>L'article répond directement aux questions centrales du domaine...</td>
    </tr>
    <tr>
      <td>Qualité scientifique</td>
      <td>8/10</td>
      <td>Méthodologie rigoureuse mais reproductibilité partielle...</td>
    </tr>
    <tr>
      <td>Légitimité des auteurs</td>
      <td>9/10</td>
      <td>Équipe reconnue avec publications de haut niveau...</td>
    </tr>
  </tbody>
</table>
```

### Section 6 : Pistes bibliographiques

**Balise** : `<h2>Pistes bibliographiques</h2>`

**Contenu** :
- Sélection de références citées dans l'article au format APA
- Références indispensables pour approfondir le champ
- Lectures complémentaires pertinentes

---

## Placeholders Disponibles

Le système utilise les placeholders suivants dans le template `app/utils/zotero_prompt.md` :

| Placeholder | Source | Description | Exemple |
|-------------|--------|-------------|---------|
| `{TITLE}` | `metadata.title` | Titre de l'article | "Machine Learning for Social Sciences" |
| `{AUTHORS}` | `metadata.authors` | Auteurs (format libre) | "Smith, J.; Doe, M." |
| `{DATE}` | `metadata.date` | Date de publication | "2024" |
| `{DOI}` | `metadata.doi` | Digital Object Identifier | "10.1234/example.2024" |
| `{URL}` | `metadata.url` | URL de l'article | "https://example.com/article" |
| `{PROBLEMATIQUE}` | `metadata.problematique` | **Problématique de recherche personnalisée** | "Comment les algorithmes de NLP peuvent-ils améliorer l'analyse de corpus sociologiques ?" |
| `{ABSTRACT}` | `metadata.abstract` | Résumé de l'article | "This paper presents..." |
| `{TEXT}` | `text_content` (OCR) | **Texte intégral** (pas de limite) | Tout le contenu OCR |
| `{LANGUAGE}` | Auto-détecté | Langue cible de l'analyse | "français", "English", etc. |

### Placeholder PROBLEMATIQUE (Nouveau)

**Objectif** : Permettre de contextualiser l'analyse par rapport à une problématique de recherche spécifique.

**Utilisation** :

1. **Via CSV** : Ajouter une colonne `problematique` dans votre fichier CSV de métadonnées Zotero

   ```csv
   title,authors,date,problematique
   "Article Title","Smith, J.",2024,"Comment mesurer l'impact des réseaux sociaux sur l'opinion publique ?"
   ```

2. **Valeur par défaut** : Si non spécifiée, le système utilise `"Non spécifiée"`

3. **Dans le prompt** : Le LLM reçoit cette problématique dans le contexte et oriente son analyse en conséquence

**Avantages** :
- Analyse orientée vers vos questions de recherche
- Meilleure pertinence des sections État de l'art et Résultats
- Cohérence avec votre projet de recherche global

---

## Configuration du Système

### Fichiers concernés

1. **`app/utils/zotero_prompt.md`** (124 lignes)
   - Template du prompt avec instructions détaillées
   - Définit la structure académique attendue
   - Spécifie le format HTML et les consignes

2. **`app/utils/llm_note_generator.py`** (440 lignes)
   - Module de génération de notes
   - Gestion des placeholders
   - Appels API OpenAI/OpenRouter
   - **Pas de limite de texte** : ligne 136 (texte intégral)
   - **max_tokens = 16000** : ligne 253 (analyses longues)

3. **`app/main.py`**
   - Endpoint `/generate_zotero_notes`
   - Gestion du modèle par défaut via `OPENROUTER_DEFAULT_MODEL`

### Variables d'environnement (.env)

```bash
# API Keys (au moins une requise)
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-...

# Modèle par défaut (utilisé si l'utilisateur ne spécifie rien)
OPENROUTER_DEFAULT_MODEL=google/gemini-2.5-flash

# Zotero API (pour publier les notes)
ZOTERO_API_KEY=...
ZOTERO_USER_ID=...
```

---

## Choix du Modèle LLM

### Format

- **OpenAI** : `gpt-4o-mini`, `gpt-4o`, `o1-mini`, etc.
- **OpenRouter** : `provider/model`, ex: `openai/gemini-2.5-flash`, `anthropic/claude-3.5-sonnet`

**Détection automatique** : Si le modèle contient `/`, le système utilise OpenRouter, sinon OpenAI.

### Recommandations

#### Pour analyses courtes (< 3000 mots)

| Modèle | Provider | Coût estimé | Qualité | Vitesse |
|--------|----------|-------------|---------|---------|
| `gpt-4o-mini` | OpenAI | $ | Bonne | Rapide |
| `openai/gemini-2.5-flash` | OpenRouter | $ | Bonne | Très rapide |
| `anthropic/claude-3-haiku` | OpenRouter | $ | Très bonne | Rapide |

#### Pour analyses exhaustives (5000-15000 mots)

| Modèle | Provider | Coût estimé | Qualité | Vitesse |
|--------|----------|-------------|---------|---------|
| `gpt-4o` | OpenAI | $$$ | Excellente | Moyenne |
| `anthropic/claude-3.5-sonnet` | OpenRouter | $$$ | Excellente | Moyenne |
| `openai/gemini-2.5-pro` | OpenRouter | $$ | Très bonne | Rapide |
| `openai/o1-mini` | OpenRouter | $$$$ | Excellente | Lente |

#### Pour corpus volumineux (budget limité)

| Modèle | Provider | Coût estimé | Qualité | Vitesse |
|--------|----------|-------------|---------|---------|
| `openai/gemini-2.5-flash` | OpenRouter | $ | Bonne | Très rapide |
| `deepseek/deepseek-chat` | OpenRouter | $ | Correcte | Rapide |
| `meta-llama/llama-3.3-70b` | OpenRouter | $ | Bonne | Rapide |

### Estimation des coûts

**Hypothèses** :
- Article moyen : ~8000 tokens en entrée (texte OCR complet)
- Analyse exhaustive : ~10000 tokens en sortie (prompt demande ~8000-12000 mots)
- Total par article : ~18000 tokens

**Coûts approximatifs par article** (2025) :

| Modèle | Coût/article | Coût/100 articles |
|--------|--------------|-------------------|
| `gpt-4o-mini` | $0.01 | $1 |
| `openai/gemini-2.5-flash` | $0.005 | $0.50 |
| `gpt-4o` | $0.20 | $20 |
| `anthropic/claude-3.5-sonnet` | $0.25 | $25 |
| `openai/o1-mini` | $0.40 | $40 |

**Conseil** : Pour un corpus de 100 articles, utilisez `gemini-2.5-flash` pour tester, puis `claude-3.5-sonnet` pour les articles les plus importants.

---

## Personnalisation du Prompt

### Méthode 1 : Modification directe de zotero_prompt.md

**Fichier** : `app/utils/zotero_prompt.md`

**Avantages** :
- Contrôle total sur le prompt
- Pas besoin de redémarrer le serveur (rechargé à chaque génération)

**Inconvénient** :
- Modifications globales (affecte tous les articles)

**Exemple de personnalisation** :

```markdown
## CONSIGNES SUPPLÉMENTAIRES

- Analyse de manière très pointue chaque variable, méthode et traitement évoqués
- Clarifie et synthétise la logique entre problématique, méthode et résultats
- **AJOUT PERSONNALISÉ** : Évalue systématiquement la validité interne et externe
- **AJOUT PERSONNALISÉ** : Compare avec les standards de votre discipline (sociologie computationnelle)
```

### Méthode 2 : Ajout de contexte via PROBLEMATIQUE

**Plus souple** : Personnalisation article par article via le CSV

**Exemple** :

```csv
title,authors,problematique
"Article A","Smith",Dans quelle mesure les biais algorithmiques reproduisent-ils les inégalités sociales ?
"Article B","Doe",Comment l'analyse de réseaux peut-elle révéler des structures de pouvoir ?
```

Le LLM adaptera l'analyse à chaque problématique spécifique.

### Méthode 3 : Système de prompts multiples (avancé)

**Idée** : Créer plusieurs fichiers de prompt pour différents types d'analyses

**Fichiers possibles** :
- `zotero_prompt_exhaustif.md` (actuel)
- `zotero_prompt_methodologique.md` (focus méthodologie)
- `zotero_prompt_theorique.md` (focus théories)
- `zotero_prompt_rapide.md` (version 500 mots)

**Implémentation** : Modifier `_load_prompt_template()` pour accepter un paramètre `template_name`.

---

## Format HTML Autorisé

### Balises supportées par Zotero

**Zotero accepte uniquement** :
- `<h2>`, `<h3>` : Titres de sections et sous-sections
- `<p>` : Paragraphes
- `<strong>`, `<em>` : Mise en forme (gras, italique)
- `<ul>`, `<li>` : Listes à puces
- `<table>`, `<thead>`, `<tbody>`, `<tr>`, `<th>`, `<td>` : Tableaux

**Balises interdites** :
- `<html>`, `<head>`, `<body>` : Zotero les ajoute automatiquement
- `<div>`, `<span>`, `<a>` : Non supportées dans les notes Zotero
- `<script>`, `<style>` : Interdites pour raisons de sécurité

### Exemple de structure HTML complète

```html
<!-- ragpy-note-id:abc-123-def -->
<h2>Introduction</h2>
<h3>Auteurs : informations principales</h3>
<table>
  <thead>
    <tr><th>Prénom</th><th>Nom</th><th>Institution</th></tr>
  </thead>
  <tbody>
    <tr><td>Marie</td><td>Dupont</td><td>CNRS</td></tr>
  </tbody>
</table>

<h3>Contexte scientifique de la recherche</h3>
<p>L'article s'inscrit dans le courant des <strong>sciences sociales computationnelles</strong>...</p>

<h2>I. État de l'art</h2>
<p>Les auteurs mobilisent trois approches théoriques principales :</p>
<ul>
  <li><strong>Théorie des réseaux sociaux</strong> : Granovetter (1973), Barabási (2016)</li>
  <li><strong>Analyse de contenu automatisée</strong> : Grimmer & Stewart (2013)</li>
  <li><strong>Sociologie computationnelle</strong> : Lazer et al. (2009)</li>
</ul>

<h2>II. Méthodologie, Données, Grille d'analyse</h2>
<h3>a) Description méthodologique</h3>
<p>Les données proviennent de...</p>

<!-- etc. -->
```

---

## Migration depuis l'ancien système

### Différences principales

| Aspect | Ancien système | Nouveau système |
|--------|----------------|-----------------|
| **Longueur** | 200-300 mots max | Pas de limite (8000-12000 mots typique) |
| **Texte analysé** | 8000 caractères max | Texte intégral OCR |
| **Sections** | 5 sections simples | 6 sections avec sous-sections |
| **Tableaux** | Non | Oui (auteurs, évaluation) |
| **Problématique** | Générique | Personnalisable via {PROBLEMATIQUE} |
| **max_tokens** | 2000 | 16000 |
| **Coût/article** | ~$0.005 | ~$0.01 à $0.25 (selon modèle) |

### Checklist de migration

- [x] **Remplacer `zotero_prompt.md`** avec le nouveau template exhaustif
- [x] **Modifier `llm_note_generator.py`** :
  - [x] Ligne 122 : Ajout extraction `problematique`
  - [x] Ligne 136 : Suppression troncature texte `[:8000]`
  - [x] Ligne 149 : Ajout remplacement `{PROBLEMATIQUE}`
  - [x] Ligne 253 : Augmentation `max_tokens` à 16000
  - [x] Ligne 168 : Mise à jour fallback prompt avec PROBLEMATIQUE
- [ ] **Tester avec un article** :
  ```bash
  # 1. Upload un export Zotero JSON
  # 2. Traiter jusqu'à "Sparse Embeddings"
  # 3. Générer les notes Zotero
  # 4. Vérifier : longueur > 3000 mots, tableaux présents
  ```
- [ ] **(Optionnel) Ajouter colonne `problematique` au CSV** si vous souhaitez personnaliser les analyses

### Compatibilité

✅ **Rétrocompatible** : L'ancien système fonctionnait avec le template court. Le nouveau système peut toujours utiliser un template court si vous le souhaitez.

✅ **Pas de changement d'API** : Les endpoints et paramètres restent identiques.

⚠️ **Temps de génération** : Les analyses exhaustives prennent 30-90 secondes vs 10-20 secondes pour le système court.

⚠️ **Coûts** : Multiplication par ~5-50 selon le modèle choisi (voir tableau des coûts).

---

## Exemples d'utilisation

### Cas 1 : Analyse exhaustive standard

**Configuration** :
- Modèle : `anthropic/claude-3.5-sonnet`
- Problématique : Non spécifiée
- Texte : Intégral (OCR)

**Commande** :
1. Interface web : Entrer `anthropic/claude-3.5-sonnet` dans le champ modèle
2. Cliquer "Generate Zotero Notes"

**Résultat attendu** :
- Note de 8000-12000 mots
- Tableaux auteurs et évaluation
- Analyse approfondie de la méthodologie
- Bibliographie structurée

### Cas 2 : Corpus avec problématique personnalisée

**Fichier CSV** (`metadata.csv`) :
```csv
title,authors,date,problematique
"Computational Social Science","Lazer et al.",2009,"Comment les big data transforment-elles les méthodes en sciences sociales ?"
"Network Analysis","Barabási",2016,"Quels modèles mathématiques capturent le mieux la dynamique des réseaux sociaux ?"
```

**Configuration** :
- Modèle : `openai/gemini-2.5-flash` (économique pour corpus)
- Upload CSV avec colonne `problematique`

**Résultat** :
- Chaque analyse orientée vers la problématique spécifique
- Meilleure cohérence avec votre recherche
- Comparaison facilitée entre articles

### Cas 3 : Analyse rapide (test)

**Configuration** :
- Modèle : `gpt-4o-mini`
- Problématique : "Test rapide"

**Objectif** :
- Vérifier que le système fonctionne
- Coût minimal ($0.01)
- Temps rapide (20-30 secondes)

**Note** : Même avec `gpt-4o-mini`, le prompt demande une analyse exhaustive, donc vous obtiendrez quand même ~5000 mots (pas 200-300).

---

## Dépannage

### Problème 1 : "LLM not available or disabled"

**Cause** : API keys manquantes ou mal configurées

**Solution** :
1. Vérifier `.env` :
   ```bash
   OPENAI_API_KEY=sk-...
   # OU
   OPENROUTER_API_KEY=sk-...
   ```
2. Redémarrer le serveur FastAPI
3. Vérifier les logs : `INFO: OpenAI client initialized` ou `INFO: OpenRouter client initialized`

### Problème 2 : "replace() argument 2 must be str, not float"

**Cause** : Métadonnées CSV avec valeurs NaN (pandas)

**Solution** : Déjà corrigée dans `llm_note_generator.py` via fonction `safe_str()` (lignes 105-114, 275-284)

### Problème 3 : Note trop courte (< 1000 mots)

**Causes possibles** :
1. **Texte OCR vide ou court** → Vérifier que `texteocr` est bien rempli dans le CSV
2. **max_tokens insuffisant** → Vérifier ligne 253 : `max_tokens=16000`
3. **Modèle refuse de générer** → Essayer un autre modèle (ex: claude-3.5-sonnet)

**Debug** :
```bash
# Vérifier les logs
grep "Generated note content" logs/app.log
# Devrait afficher : "Generated note content (length: 35000 chars)"
```

### Problème 4 : Timeout API

**Cause** : Analyses très longues (>60 secondes)

**Solution** :
1. Utiliser un modèle plus rapide (gemini-2.5-flash au lieu de o1-mini)
2. Augmenter timeout dans `llm_note_generator.py` :
   ```python
   response = active_client.chat.completions.create(
       ...,
       timeout=180  # 3 minutes
   )
   ```

### Problème 5 : HTML mal formé dans Zotero

**Symptômes** : Tableaux cassés, balises visibles

**Cause** : LLM génère des balises non autorisées

**Solution** :
1. Vérifier le prompt `zotero_prompt.md` ligne 107-119 (instructions HTML claires)
2. Si le problème persiste, post-traiter le HTML :
   ```python
   # Dans llm_note_generator.py, après ligne 254
   import re
   content = re.sub(r'<(?!/?(?:h2|h3|p|strong|em|ul|li|table|thead|tbody|tr|th|td))[^>]+>', '', content)
   ```

---

## Performances et Limites

### Limites actuelles

1. **Coût** : Analyses exhaustives coûtent 5-50× plus cher que le système court
2. **Temps** : 30-90 secondes par article vs 10-20 secondes
3. **Qualité variable** : Dépend fortement du modèle choisi
4. **OCR requis** : Sans texte OCR complet, l'analyse se limite à l'abstract
5. **Langue** : Meilleure qualité en français et anglais

### Recommandations de performance

**Pour 10-20 articles** :
- Modèle : `anthropic/claude-3.5-sonnet`
- Coût : ~$5
- Qualité : Excellente

**Pour 100-500 articles** :
- Modèle : `openai/gemini-2.5-flash`
- Coût : ~$25-50
- Qualité : Très bonne

**Pour 1000+ articles** :
- Stratégie hybride :
  1. Premièr pass avec `gemini-2.5-flash` (tous les articles)
  2. Second pass avec `claude-3.5-sonnet` (top 10% des articles importants)
  3. Coût : ~$100-150

---

## Contribuer et Personnaliser

### Améliorer le prompt

**Fichier** : `app/utils/zotero_prompt.md`

**Idées d'amélioration** :
1. Ajouter des sections spécifiques à votre discipline
2. Demander des métriques quantitatives (p-values, effect sizes)
3. Intégrer des critères de qualité spécifiques (CONSORT, PRISMA, etc.)
4. Ajouter une section "Questions pour la suite"

### Ajouter un nouveau placeholder

**Exemple** : Ajouter `{COLLECTION}` pour la collection Zotero

**Étapes** :

1. **Modifier `llm_note_generator.py`** (fonction `_build_prompt`) :
   ```python
   collection = safe_str(metadata.get("collection"), "N/A")
   ```

2. **Ajouter le remplacement** :
   ```python
   prompt = prompt.replace("{COLLECTION}", collection)
   ```

3. **Utiliser dans `zotero_prompt.md`** :
   ```markdown
   **Collection** : {COLLECTION}
   ```

4. **Mettre à jour le CSV** :
   ```csv
   title,collection
   "Article 1","Sociologie computationnelle"
   ```

### Créer un template personnalisé

**Exemple** : Template pour revues systématiques

**Fichier** : `app/utils/zotero_prompt_revue_systematique.md`

**Structure** :
```markdown
# Prompt de revue systématique

## CONTEXTE
[Métadonnées identiques]

## TÂCHE
Effectue une revue systématique de cet article selon les critères PRISMA.

## STRUCTURE
1. Critères d'inclusion/exclusion
2. Évaluation de la qualité méthodologique
3. Extraction des données clés
4. Synthèse des résultats
5. Niveau de preuve (Grade)

[etc.]
```

**Utilisation** : Modifier `_load_prompt_template()` pour sélectionner ce template.

---

## Changelog

### Version 1.1.0 (2025-10-26) - Système Exhaustif

**Ajouté** :
- ✅ Template exhaustif `zotero_prompt.md` (124 lignes)
- ✅ Placeholder `{PROBLEMATIQUE}` pour personnalisation
- ✅ Support du texte intégral (suppression limite 8000 caractères)
- ✅ max_tokens augmenté à 16000 (analyses longues)
- ✅ 6 sections structurées avec tableaux HTML
- ✅ Documentation complète `PROMPT_ANALYSIS_EXHAUSTIVE.md`

**Modifié** :
- `app/utils/llm_note_generator.py` :
  - Ligne 122 : Extraction `problematique`
  - Ligne 136 : Texte intégral (pas de troncature)
  - Ligne 149 : Remplacement `{PROBLEMATIQUE}`
  - Ligne 253 : `max_tokens=16000`
  - Ligne 168 : Fallback prompt avec PROBLEMATIQUE

**Déprécié** :
- Ancien prompt court (200-300 mots) → Conservé dans fallback uniquement

### Version 1.0.0 (2025-10-25) - Système Court

**Fonctionnalités** :
- Génération de fiches courtes (200-300 mots)
- Support OpenAI et OpenRouter
- Limite 8000 caractères de texte
- 5 sections basiques

---

## Support et Contact

**Documentation** : [BUGFIX_LIBRARY_EXTRACTION.md](BUGFIX_LIBRARY_EXTRACTION.md)

**Tests** :
- `tests/test_prompt_file.py` - Validation du template prompt
- `tests/test_zotero_json_formats.py` - Formats JSON Zotero

**Logs** : Vérifier `logs/app.log` pour le debugging

**Questions fréquentes** : Voir section Dépannage ci-dessus

---

**Dernière mise à jour** : 2025-10-26
**Version** : 1.1.0 (Système d'Analyse Exhaustive)
