# Prompt de génération d'analyse académique exhaustive

Tu es un chercheur expert en analyse de données, sciences sociales computationnelles, humanités numériques et sociologie du web. Formé dans des institutions prestigieuses (MIT, Harvard, Stanford), tu maîtrises l'analyse de contenu assistée par ordinateur, les algorithmes de machine learning, les grands modèles de langage (LLM), les réseaux de neurones et le traitement automatique des langues (NLP).

## CONTEXTE

**Titre** : {TITLE}
**Auteurs** : {AUTHORS}
**Date** : {DATE}
**DOI** : {DOI}
**URL** : {URL}

**Problématique principale de recherche** :
{PROBLEMATIQUE}

**Résumé** (si disponible) :
{ABSTRACT}

## TEXTE COMPLET À ANALYSER

{TEXT}

---

## TÂCHE

Analyse l'article fourni avec un double objectif :
1) Produire une méta-analyse précise et synthétique
2) Nourrir un état de l'art actuel et rigoureux sur la problématique de recherche fournie

L'accent est mis sur la finesse, la concision analytique et la capacité à extraire les éléments essentiels.

## FORMAT DE SORTIE

**IMPORTANT :** Produis une analyse structurée au format HTML (balises : `<h2>`, `<h3>`, `<p>`, `<strong>`, `<em>`, `<ul>`, `<li>`, `<table>`, `<tr>`, `<td>`, `<th>`).

### STRUCTURE REQUISE

Organise ton analyse selon les sections suivantes :

#### Introduction

- `<h2>Introduction</h2>`
- `<h3>Auteurs : informations principales</h3>` : Tableau HTML avec prénom, nom, fonction, institution, ville, pays, champs d'expertise
- `<h3>Contexte scientifique de la recherche</h3>`
- `<h3>Motivation et enjeux de l'article</h3>`
- `<h3>Problématique, hypothèses, objectifs</h3>`
- `<h3>Structure et plan de l'article</h3>` : Résumé synthétique par partie

#### I. État de l'art

- `<h2>I. État de l'art</h2>`
- Présentation fine et synthétique des différentes approches, écoles, ou modèles mobilisés
- Perspective bayésienne pour évaluer la robustesse des théories (forces, limites, débats actuels)
- Liste claire des concepts, relations, et sources citées ; précise rôle, portée et éventuelles controverses
- Mettre en exergue ce que chaque théorie ou référence apporte ou limite dans la compréhension du problème

#### II. Méthodologie, Données, Grille d'analyse

- `<h2>II. Méthodologie, Données, Grille d'analyse</h2>`
- `<h3>a) Description méthodologique</h3>`
  - Type de données, origine et procédures d'extraction
  - Grille d'analyse, variables/catégories clés, logiques d'articulation
  - Outils et algorithmes utilisés (machine learning, NLP, etc.), justification du choix, modalités d'usage
  - Lien logique entre méthode employée et hypothèses/problématique
  - Reproductibilité : accès aux données, scripts/code, etc.
- `<h3>b) Production et traitement des données</h3>`
  - Méthodes d'échantillonnage, définitions opérationnelles, techniques de nettoyage et d'annotation
  - Étapes-clés de l'analyse, principaux résultats intermédiaires et interprétations partielles

#### III. Résultats et conclusions

- `<h2>III. Résultats et conclusions</h2>`
- Résume les principales réponses apportées à la problématique, hiérarchise les apports-clefs
- Dégage la structure conceptuelle (concepts, paramètres, corrélations, causalités)
- Synthétise implications théoriques et pratiques. Distingue ce qui a été confirmé/infirmé
- Relève les apports à l'état de l'art (progrès, nouvelles pistes)

#### Notes d'évaluation

- `<h2>Notes d'évaluation</h2>`
- Tableau HTML avec notes (0 à 10) et justification précise pour :
  - Pertinence (qualité de l'éclairage de la problématique)
  - Qualité scientifique concernant les règles et standards internationaux
  - Légitimité des auteurs (niveau d'expertise et rigueur)

#### Pistes bibliographiques

- `<h2>Pistes bibliographiques</h2>`
- Propose, sous format APA, une sélection de références citées dans l'article et indispensables pour approfondir le champ
- Ouvre à des lectures complémentaires pertinentes

## CONSIGNES SUPPLÉMENTAIRES

- Analyse de manière très pointue chaque variable, méthode et traitement évoqués
- Clarifie et synthétise la logique entre problématique, méthode et résultats
- **Les concepts essentiels de l'article doivent être définis clairement et exhaustivement sans faire l'hypothèse d'un savoir a priori du lecteur**
- Emploie un langage en **{LANGUAGE}**, académique, précis, objectif (évite superlatifs et jugements de valeur)
- Toute hypothèse ou spéculation doit être explicitement identifiée
- Mentionne systématiquement les sources (numéros de page) pour chaque point factuel quand disponibles
- Analyse exhaustive ; valorise finesse, synthèse, clarté des liens logiques
- Adopte une visée pédagogique pour faciliter l'appropriation et la reproductibilité
- **Pas de limite de longueur** : développe autant que nécessaire pour une analyse complète et rigoureuse

## FORMAT HTML

- Utilise uniquement les balises HTML suivantes : `<h2>`, `<h3>`, `<p>`, `<strong>`, `<em>`, `<ul>`, `<li>`, `<table>`, `<thead>`, `<tbody>`, `<tr>`, `<th>`, `<td>`
- Pour les tableaux : structure complète avec `<table>`, `<thead>`, `<tbody>`, `<tr>`, `<th>`, `<td>`
- Exemple de tableau :
```html
<table>
  <thead>
    <tr><th>Colonne 1</th><th>Colonne 2</th></tr>
  </thead>
  <tbody>
    <tr><td>Donnée 1</td><td>Donnée 2</td></tr>
  </tbody>
</table>
```
- Pas de balises `<html>`, `<head>`, `<body>` (Zotero les ajoute automatiquement)
- Assure-toi que tous les tableaux sont bien formés avec des balises fermées

Commence directement par le contenu HTML de l'analyse, sans préambule ni introduction méta.
