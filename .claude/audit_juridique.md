# Audit Juridique RAGpy - Droit Français et Européen

**Date de l'audit** : 2025-12-15
**Version RAGpy auditée** : Commit 045c8b2
**Auditeur** : Claude Code (Anthropic)

---

## Table des Matières

1. [Synthèse Exécutive](#1-synthèse-exécutive)
2. [Cadre Juridique Applicable](#2-cadre-juridique-applicable)
3. [Analyse RGPD](#3-analyse-rgpd)
4. [Analyse Propriété Intellectuelle](#4-analyse-propriété-intellectuelle)
5. [Analyse Sécurité des SI](#5-analyse-sécurité-des-systèmes-dinformation)
6. [Analyse Contractuelle](#6-analyse-contractuelle)
7. [Plan de Développement - Scénario Individuel](#7-plan-de-développement---scénario-individuel)
8. [Plan de Développement - Groupe de Recherche](#8-plan-de-développement---groupe-de-recherche-serveur-privé)
9. [Plan de Développement - Institution Universitaire](#9-plan-de-développement---institution-universitaire)
10. [Annexes - Références Légales Complètes](#10-annexes---références-légales-complètes)

---

## 1. Synthèse Exécutive

### 1.1 Verdict par Scénario

| Scénario | Verdict | Risque Global | Actions Requises |
|----------|---------|---------------|------------------|
| **Chercheur individuel** | ✅ CONFORME | FAIBLE | 2 actions (~5 min) |
| **Groupe de recherche** | ⚠️ CONFORME SOUS CONDITIONS | MOYEN | 8 actions (~1 jour) |
| **Institution universitaire** | ❌ NON CONFORME | ÉLEVÉ | 25 actions (~3 semaines) |

### 1.2 Risques Identifiés par Domaine

| Domaine Juridique | Personnel | Labo Privé | Institutionnel |
|-------------------|-----------|------------|----------------|
| Protection des données (RGPD) | N/A | MOYEN | CRITIQUE |
| Propriété intellectuelle | FAIBLE | FAIBLE | MOYEN |
| Sécurité informatique | FAIBLE | MOYEN | ÉLEVÉ |
| Droit des contrats | FAIBLE | MOYEN | MOYEN |

### 1.3 Points Forts de RAGpy

- Architecture de sécurité solide (chiffrement Fernet, bcrypt 12 rounds)
- Isolation des credentials par rôle (admin/user)
- Audit logging complet
- Support exception TDM pour la recherche

### 1.4 Points Critiques à Corriger

1. Configuration CORS trop permissive (`allow_origins=["*"]`)
2. Absence de Privacy Policy
3. crawl.py ne respecte pas robots.txt
4. Suppression des données incomplète (Art. 17 RGPD)

---

## 2. Cadre Juridique Applicable

### 2.1 Textes Européens

#### Règlement (UE) 2016/679 - RGPD

**Texte officiel** : [EUR-Lex RGPD](https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX%3A32016R0679)

| Article | Intitulé | Applicabilité RAGpy |
|---------|----------|---------------------|
| Art. 2§2(c) | Exception domestique | Exclut usage personnel |
| Art. 5 | Principes du traitement | Base fondamentale |
| Art. 6 | Licéité du traitement | Base légale requise |
| Art. 7 | Conditions du consentement | Si consentement choisi |
| Art. 12-14 | Information des personnes | Privacy Policy |
| Art. 15 | Droit d'accès | Endpoint GET /me |
| Art. 17 | Droit à l'effacement | DELETE incomplet |
| Art. 20 | Portabilité | Non implémenté |
| Art. 28 | Sous-traitants | DPA requis |
| Art. 30 | Registre des traitements | Obligatoire > 250 salariés |
| Art. 32 | Sécurité du traitement | Mesures techniques |
| Art. 33 | Notification de violation | 72h CNIL |
| Art. 35 | Analyse d'impact (DPIA) | Si profilage |
| Art. 89 | Recherche scientifique | Dérogations possibles |

#### Directive (UE) 2019/790 - Droit d'Auteur

**Texte officiel** : [EUR-Lex Directive Copyright](https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX%3A32019L0790)

| Article | Intitulé | Applicabilité RAGpy |
|---------|----------|---------------------|
| Art. 3 | TDM pour recherche | Exception applicable |
| Art. 4 | TDM général (opt-out) | Vérifier robots.txt |

#### Directive (UE) 2022/2555 - NIS2

**Texte officiel** : [EUR-Lex NIS2](https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX%3A32022L2555)

| Article | Applicabilité |
|---------|---------------|
| Art. 2 | Champ d'application | Infrastructures de recherche "importantes" |
| Art. 21 | Mesures de cybersécurité | Si applicable |
| Art. 23 | Notification d'incidents | 24h si applicable |

### 2.2 Textes Français

#### Code de la Propriété Intellectuelle (CPI)

**Texte officiel** : [Légifrance CPI](https://www.legifrance.gouv.fr/codes/id/LEGITEXT000006069414/)

| Article | Intitulé | Applicabilité RAGpy |
|---------|----------|---------------------|
| L.122-4 | Droit de reproduction | Principe général |
| L.122-5 1° | Exception copie privée | Usage personnel |
| L.122-5 3° | Exception courte citation | Fiches de lecture |
| L.122-5-3 | Exception TDM | Recherche scientifique |
| L.335-2 | Contrefaçon | Sanction pénale |
| L.342-1 | Protection des BDD | Extraction illicite |

#### Loi Informatique et Libertés (LIL)

**Texte officiel** : [Légifrance LIL](https://www.legifrance.gouv.fr/loda/id/JORFTEXT000000886460)

| Article | Intitulé | Applicabilité |
|---------|----------|---------------|
| Art. 4 | Recherche scientifique | Régime dérogatoire |
| Art. 78 | Traitement recherche | Conditions spécifiques |

### 2.3 Jurisprudence Pertinente

| Affaire | Juridiction | Principe |
|---------|-------------|----------|
| Google Spain (C-131/12) | CJUE 2014 | Droit à l'oubli |
| Schrems II (C-311/18) | CJUE 2020 | Transferts USA invalides |
| LinkedIn v. hiQ | US 9th Circuit 2022 | Scraping données publiques |

---

## 3. Analyse RGPD

### 3.1 Champ d'Application Territorial

**Article 3 RGPD** : Le règlement s'applique si :
- Le responsable de traitement est établi dans l'UE
- Les personnes concernées sont dans l'UE
- Offre de biens/services ou suivi de comportement dans l'UE

**Application à RAGpy** :
- Serveur en France/UE → RGPD applicable (sauf exception domestique)
- Transferts vers USA (OpenAI, Pinecone) → Problématique Schrems II

### 3.2 Exception Domestique (Art. 2§2(c))

**Texte de l'article** :
> "Le présent règlement ne s'applique pas au traitement de données à caractère personnel effectué par une personne physique dans le cadre d'une activité strictement personnelle ou domestique."

**Interprétation CJUE** (Lindqvist C-101/01) :
- Activité purement personnelle sans lien professionnel
- Pas de diffusion à un nombre indéterminé de personnes
- Pas d'activité économique

**Application à RAGpy - Usage Personnel** :
| Critère | Analyse | Conclusion |
|---------|---------|------------|
| Personne physique | Chercheur individuel | ✅ Oui |
| Activité personnelle | Recherche pour soi | ✅ Oui |
| Pas de diffusion | Stockage local | ✅ Oui |
| Pas d'activité économique | Recherche non commerciale | ✅ Oui |

**Verdict** : Exception domestique applicable pour usage strictement personnel.

### 3.3 Base Légale pour la Recherche (Art. 6 et 89)

**Article 6§1 RGPD - Bases légales** :

| Base | Article | Applicabilité Recherche |
|------|---------|-------------------------|
| Consentement | Art. 6§1(a) | Possible mais contraignant |
| Contrat | Art. 6§1(b) | Non applicable |
| Obligation légale | Art. 6§1(c) | Non applicable |
| Intérêts vitaux | Art. 6§1(d) | Non applicable |
| **Mission d'intérêt public** | **Art. 6§1(e)** | **RECOMMANDÉ** |
| **Intérêt légitime** | **Art. 6§1(f)** | **ALTERNATIVE** |

**Article 89 RGPD - Garanties pour la recherche** :

> "Le traitement à des fins archivistiques dans l'intérêt public, à des fins de recherche scientifique ou historique, ou à des fins statistiques est soumis à des garanties appropriées..."

**Garanties requises** :
- Pseudonymisation si possible
- Minimisation des données
- Mesures techniques de sécurité

### 3.4 Données Collectées par RAGpy

**Fichier** : `app/models/user.py`

| Donnée | Catégorie RGPD | Sensibilité | Chiffrement |
|--------|----------------|-------------|-------------|
| Email | Identifiant | Moyenne | Non |
| Mot de passe | Authentification | Haute | bcrypt 12 rounds |
| Nom/Prénom | Identité | Moyenne | Non |
| Organisation | Professionnelle | Faible | Non |
| Credentials API | Secret | Critique | Fernet AES-128 |
| IP Address | Technique | Moyenne | Non |
| User-Agent | Technique | Faible | Non |
| Timestamps | Métadonnée | Faible | Non |

**Fichier** : `app/models/audit.py`

| Donnée | Finalité | Durée Conservation |
|--------|----------|-------------------|
| Actions utilisateur | Traçabilité | Indéfinie (problème) |
| IP source | Sécurité | Indéfinie (problème) |
| Erreurs | Diagnostic | Indéfinie |

### 3.5 Transferts Internationaux

**Post-Schrems II** : Les transferts vers les USA nécessitent :
- Standard Contractual Clauses (SCCs) nouvelles version
- Mesures supplémentaires si surveillance US possible

| Destinataire | Pays | Mécanisme | Statut RAGpy |
|--------------|------|-----------|--------------|
| OpenAI | USA | DPA + SCCs requis | ❌ Non documenté |
| Mistral | France | N/A (UE) | ✅ Conforme |
| Pinecone | USA | DPA + SCCs requis | ❌ Non documenté |
| Weaviate | Variable | Dépend instance | À vérifier |
| Qdrant | Variable | Dépend instance | À vérifier |

**Note** : OpenAI et Pinecone proposent des DPA sur demande.

### 3.6 Droits des Personnes - État d'Implémentation

| Droit | Article | Endpoint RAGpy | Statut |
|-------|---------|----------------|--------|
| Information | Art. 13-14 | Aucun | ❌ Absent |
| Accès | Art. 15 | GET /users/me | ⚠️ Partiel |
| Rectification | Art. 16 | PUT /users/me | ✅ Implémenté |
| Effacement | Art. 17 | DELETE /users/me | ⚠️ Incomplet |
| Limitation | Art. 18 | Aucun | ❌ Absent |
| Portabilité | Art. 20 | Aucun | ❌ Absent |
| Opposition | Art. 21 | Aucun | ❌ Absent |

---

## 4. Analyse Propriété Intellectuelle

### 4.1 Exception TDM - Cadre Légal Détaillé

#### Article L.122-5-3 CPI (transposition Directive 2019/790)

**Texte** :
> "Lorsque l'oeuvre a été divulguée, l'auteur ne peut interdire :
> [...]
> 10° Les copies ou reproductions numériques réalisées à partir d'une source licite, en vue de l'exploration de textes et de données incluses ou associées aux écrits scientifiques pour les besoins de la recherche publique [...]"

**Conditions cumulatives** :

| Condition | Explication | Vérification RAGpy |
|-----------|-------------|-------------------|
| Source licite | Accès légal au contenu | Abonnement BDD, achat |
| Exploration TDM | Analyse automatisée | OCR, embeddings, clustering |
| Recherche publique | Organisme reconnu | CNRS, Université, INSERM |
| Conservation sécurisée | Accès restreint | Serveur privé |

#### Article 3 Directive 2019/790 - Exception TDM Recherche

**Texte** :
> "Les États membres prévoient une exception aux droits [...] pour les reproductions et extractions effectuées par des organismes de recherche et des institutions du patrimoine culturel en vue de procéder, à des fins de recherche scientifique, à une fouille de textes et de données sur des œuvres ou autres objets protégés auxquels ils ont un accès licite."

**Définition "Organisme de recherche"** (Art. 2§1) :
> "une université, y compris ses bibliothèques, un institut de recherche ou tout autre organisme dont l'objectif premier est de mener des recherches scientifiques ou d'assurer des services d'enseignement impliquant également des activités de recherche"

### 4.2 Application aux Activités de RAGpy

| Activité | Base Légale | Licite ? |
|----------|-------------|----------|
| OCR sur PDF académiques | L.122-5-3 CPI | ✅ Si accès licite |
| Création d'embeddings | L.122-5-3 CPI | ✅ TDM autorisé |
| Stockage base vectorielle | L.122-5-3 CPI | ✅ Conservation recherche |
| Génération fiches lecture | L.122-5 3° CPI | ✅ Courte citation/analyse |
| Partage interne équipe | L.122-5-3 CPI | ✅ Même projet recherche |
| Publication du corpus | L.122-4 CPI | ❌ INTERDIT |
| Crawling web | L.342-1 CPI | ⚠️ Vérifier robots.txt |

### 4.3 Exception Copie Privée (Usage Personnel)

**Article L.122-5 1° CPI** :
> "Lorsque l'oeuvre a été divulguée, l'auteur ne peut interdire :
> 1° Les copies ou reproductions réservées à l'usage privé du copiste et non destinées à une utilisation collective [...]"

**Conditions** :
- Usage strictement privé
- Copiste = utilisateur final
- Pas de diffusion collective

**Application RAGpy usage personnel** : ✅ Exception applicable

### 4.4 Risques Crawl.py

**Fichier analysé** : `scripts/crawl.py`

**Problèmes identifiés** :

1. **Erreur syntaxe ligne 108** :
```python
# ACTUEL (incorrect)
if response.status_code != 200
    return

# CORRIGÉ
if response.status_code != 200:
    return
```

2. **Absence de vérification robots.txt** :
```python
# MANQUANT - À ajouter
import urllib.robotparser

def can_crawl(url):
    rp = urllib.robotparser.RobotFileParser()
    robots_url = urljoin(url, "/robots.txt")
    rp.set_url(robots_url)
    try:
        rp.read()
        return rp.can_fetch("*", url)
    except:
        return True  # Si robots.txt absent, autoriser
```

3. **Article L.342-1 CPI** - Protection des bases de données :
> "Le producteur d'une base de données [...] peut interdire :
> 1° L'extraction [...] de la totalité ou d'une partie qualitativement ou quantitativement substantielle du contenu de la base"

**Risque** : Crawling massif = extraction substantielle potentielle

---

## 5. Analyse Sécurité des Systèmes d'Information

### 5.1 Vulnérabilités Critiques Identifiées

#### V1 : Configuration CORS Permissive

**Fichier** : `app/main.py:141`

```python
# VULNÉRABLE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Accepte TOUTES les origines
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Risque** :
- CSRF (Cross-Site Request Forgery)
- Vol de credentials via site malveillant
- Exfiltration de données

**Article 32 RGPD** :
> "Le responsable du traitement et le sous-traitant mettent en œuvre les mesures techniques et organisationnelles appropriées afin de garantir un niveau de sécurité adapté au risque"

#### V2 : JWT_SECRET_KEY par Défaut

**Fichier** : `app/config.py:59`

```python
JWT_SECRET_KEY: str = os.getenv(
    "JWT_SECRET_KEY",
    "change-this-secret-key-in-production-min-32-chars"  # FAIBLE
)
```

**Risque** : Tokens JWT forgeables si valeur par défaut utilisée

#### V3 : Base SQLite Non Chiffrée

**Fichier** : `data/ragpy.db`

- Données personnelles en clair sur le disque
- Accessible si accès physique au serveur

### 5.2 Points Forts Sécurité

| Mesure | Implémentation | Référence |
|--------|----------------|-----------|
| Hachage mots de passe | bcrypt 12 rounds | `app/core/security.py` |
| Chiffrement credentials | Fernet AES-128 | `app/core/credentials.py` |
| Cookies HttpOnly | Oui | `app/routes/auth.py` |
| Cookies SameSite | Lax | `app/routes/auth.py` |
| Protection brute-force | Lockout 15 min | `app/routes/auth.py` |
| Audit logging | Complet | `app/models/audit.py` |
| Requêtes paramétrées | SQLAlchemy ORM | Global |

### 5.3 Conformité NIS2

**Directive NIS2 Art. 2** - Champ d'application :

Les "infrastructures de recherche" sont potentiellement concernées si :
- Effectif > 50 personnes OU
- CA > 10M EUR OU
- Désignation nationale

**Pour la majorité des laboratoires** : NIS2 non applicable (seuils non atteints)

---

## 6. Analyse Contractuelle

### 6.1 Conditions d'Utilisation des APIs

#### OpenAI

**Document** : [OpenAI Terms of Use](https://openai.com/policies/terms-of-use/)

| Clause | Implication RAGpy |
|--------|-------------------|
| Utilisateur responsable du contenu | Avertissement requis |
| Pas d'entraînement sur données API | ✅ Opt-out par défaut |
| Conformité lois applicables | RGPD à respecter |

**DPA disponible** : [OpenAI DPA](https://openai.com/policies/data-processing-addendum/)

#### Mistral

**Document** : [Mistral Terms](https://mistral.ai/terms/)

| Clause | Implication RAGpy |
|--------|-------------------|
| Suppression après traitement | ✅ `MISTRAL_DELETE_UPLOADED_FILE=true` |
| Pas d'entraînement | ✅ Garanti |
| Conformité RGPD | ✅ Serveurs EU |

#### Pinecone

**Document** : [Pinecone Terms](https://www.pinecone.io/terms/)

| Clause | Implication RAGpy |
|--------|-------------------|
| Ownership données utilisateur | ✅ Utilisateur propriétaire |
| Localisation USA | ⚠️ Transfert international |

### 6.2 Responsabilité RAGpy

**Absence de CGU** = Responsabilité maximale du développeur

**Clauses à ajouter** :
- Limitation de responsabilité
- Propriété intellectuelle des contenus
- Avertissement sur traitement externe
- Consentement éclairé

---

## 7. Plan de Développement - Scénario Individuel

### 7.1 Contexte

- **Utilisateur** : Chercheur individuel
- **Déploiement** : Ordinateur personnel (localhost)
- **Données** : Documents personnels uniquement
- **RGPD** : Non applicable (exception domestique Art. 2§2(c))

### 7.2 Actions Obligatoires

| # | Action | Fichier | Commande/Code | Effort |
|---|--------|---------|---------------|--------|
| 1 | Générer JWT_SECRET_KEY | `.env` | `openssl rand -hex 32` | 2 min |
| 2 | Fix syntaxe crawl.py | `scripts/crawl.py:108` | Ajouter `:` | 1 min |

#### Détail Action 1 : JWT_SECRET_KEY

```bash
# Générer une clé sécurisée
openssl rand -hex 32

# Ajouter dans .env
JWT_SECRET_KEY=<votre_clé_générée>
DEBUG=false
```

#### Détail Action 2 : Fix crawl.py

```python
# Ligne 108 - AVANT
if response.status_code != 200
    return

# Ligne 108 - APRÈS
if response.status_code != 200:
    return
```

### 7.3 Actions Recommandées

| # | Action | Justification | Effort |
|---|--------|---------------|--------|
| 3 | Ajouter robots.txt check | Respect ToS sites | 30 min |
| 4 | Sauvegardes régulières | Perte de données | 10 min |
| 5 | Mettre à jour dépendances | Sécurité CVE | 15 min |

#### Détail Action 3 : Robots.txt

```python
# scripts/crawl.py - Ajouter en début de fichier
import urllib.robotparser
from functools import lru_cache

@lru_cache(maxsize=100)
def get_robot_parser(base_url):
    """Cache le parser robots.txt par domaine."""
    rp = urllib.robotparser.RobotFileParser()
    robots_url = urljoin(base_url, "/robots.txt")
    rp.set_url(robots_url)
    try:
        rp.read()
    except Exception:
        pass  # Si erreur, on autorise par défaut
    return rp

def can_crawl(url):
    """Vérifie si le crawling est autorisé."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    rp = get_robot_parser(base_url)
    return rp.can_fetch("RAGpy-Crawler", url)

# Dans la fonction crawl(), ajouter :
def crawl(url):
    if not can_crawl(url):
        logger.info(f"Crawling interdit par robots.txt : {url}")
        return
    # ... reste du code
```

### 7.4 Checklist Finale - Usage Personnel

```
□ JWT_SECRET_KEY configuré dans .env
□ DEBUG=false dans .env
□ crawl.py ligne 108 corrigé
□ (Optionnel) robots.txt vérifié
□ (Optionnel) Sauvegardes configurées
```

**Temps total estimé** : 5-45 minutes

---

## 8. Plan de Développement - Groupe de Recherche (Serveur Privé)

### 8.1 Contexte

- **Utilisateurs** : Équipe de recherche (2-15 personnes)
- **Déploiement** : Serveur laboratoire privé
- **Données** : Documents recherche, métadonnées utilisateurs
- **RGPD** : Applicable (base légale recherche)

### 8.2 Phase 1 : Corrections Urgentes (Jour 1)

| # | Action | Fichier | Priorité |
|---|--------|---------|----------|
| 1 | JWT_SECRET_KEY | `.env` | CRITIQUE |
| 2 | Corriger CORS | `app/main.py` | CRITIQUE |
| 3 | Fix crawl.py | `scripts/crawl.py` | HAUTE |

#### Détail Action 2 : CORS Restrictif

```python
# app/main.py - Remplacer lignes 138-145

from app.config import settings

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(","),  # Liste explicite
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)
```

```bash
# .env - Ajouter
CORS_ORIGINS=http://localhost:8000,https://ragpy.monlabo.univ.fr
```

### 8.3 Phase 2 : Documentation Légale (Jour 2-3)

| # | Action | Livrable |
|---|--------|----------|
| 4 | Documenter base légale | `LEGAL.md` |
| 5 | Notice utilisateurs | `NOTICE_UTILISATEURS.md` |
| 6 | Registre simplifié | `REGISTRE_TRAITEMENTS.md` |

#### Modèle LEGAL.md

```markdown
# Base Légale du Traitement - RAGpy

## Responsable de Traitement
- Nom : [Nom du laboratoire]
- Adresse : [Adresse]
- Contact DPO : [Email DPO université]

## Finalité du Traitement
Veille scientifique et analyse documentaire pour le projet de recherche [NOM_PROJET].

## Base Légale
Article 6§1(e) RGPD : Mission d'intérêt public (recherche scientifique)

## Catégories de Données
- Identifiants utilisateurs (email professionnel)
- Documents scientifiques (articles, thèses)
- Métadonnées bibliographiques

## Durée de Conservation
- Données utilisateurs : durée du projet + 1 an
- Documents traités : durée du projet
- Logs : 90 jours

## Transferts
- Mistral AI (France) : OCR des documents
- OpenAI (USA) : Embeddings et analyse - DPA signé

## Droits des Personnes
Contact : [email responsable]
```

#### Modèle NOTICE_UTILISATEURS.md

```markdown
# Notice d'Information - RAGpy

## Qui traite vos données ?
Le laboratoire [NOM], sous la responsabilité de [NOM RESPONSABLE].

## Pourquoi ?
Pour permettre la veille scientifique automatisée dans le cadre du projet [NOM].

## Quelles données ?
- Votre email professionnel
- Les documents que vous uploadez
- Vos requêtes de recherche

## Qui y a accès ?
- Les membres du projet de recherche
- Les services techniques (OpenAI, Mistral) pour le traitement

## Combien de temps ?
Vos données sont conservées pendant la durée du projet plus 1 an.

## Vos droits
Vous pouvez demander l'accès, la rectification ou la suppression de vos données
en contactant : [EMAIL]
```

### 8.4 Phase 3 : Sécurité Réseau (Jour 4-5)

| # | Action | Responsable |
|---|--------|-------------|
| 7 | Configurer firewall | Admin système |
| 8 | VPN si accès externe | Admin système |

```bash
# Exemple iptables - Restreindre accès port 8000
sudo iptables -A INPUT -p tcp --dport 8000 -s 192.168.1.0/24 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 8000 -j DROP
```

### 8.5 Checklist Finale - Groupe de Recherche

```
Phase 1 - Urgent
□ JWT_SECRET_KEY sécurisé
□ CORS configuré avec domaines explicites
□ crawl.py corrigé (syntaxe + robots.txt)

Phase 2 - Documentation
□ LEGAL.md créé et validé
□ NOTICE_UTILISATEURS.md affiché aux utilisateurs
□ REGISTRE_TRAITEMENTS.md maintenu

Phase 3 - Sécurité
□ Firewall configuré
□ Accès VPN si nécessaire
□ Sauvegardes automatisées
```

**Temps total estimé** : 1-2 jours

---

## 9. Plan de Développement - Institution Universitaire

### 9.1 Contexte

- **Utilisateurs** : Multi-laboratoires, multi-départements
- **Déploiement** : Infrastructure institutionnelle
- **Données** : Données personnelles multiples, documents sensibles
- **RGPD** : Pleinement applicable, coordination DPO requise

### 9.2 Gouvernance Préalable

| Acteur | Rôle | Actions |
|--------|------|---------|
| DPO Université | Validation juridique | Valider DPIA, Privacy Policy |
| DSI | Infrastructure | Valider architecture, SSO |
| RSSI | Sécurité | Valider PSSI, auditer |
| Porteur projet | Fonctionnel | Définir besoins, former |

### 9.3 Phase 1 : Corrections Critiques (Semaine 1)

| # | Action | Fichier | Responsable | Effort |
|---|--------|---------|-------------|--------|
| 1 | JWT_SECRET_KEY | `.env` | DevOps | 5 min |
| 2 | Corriger CORS | `app/main.py` | Dev | 30 min |
| 3 | Fix crawl.py complet | `scripts/crawl.py` | Dev | 2h |
| 4 | Privacy Policy | `templates/privacy.html` | Juridique + Dev | 4h |
| 5 | Notice d'information | `templates/notice.html` | Juridique + Dev | 2h |

#### Modèle Privacy Policy Institutionnelle

```html
<!-- templates/privacy.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Politique de Confidentialité - RAGpy</title>
</head>
<body>
    <h1>Politique de Confidentialité</h1>

    <h2>1. Responsable de Traitement</h2>
    <p>
        <strong>[Nom de l'Université]</strong><br>
        Adresse : [Adresse complète]<br>
        DPO : [Nom et contact du DPO]
    </p>

    <h2>2. Finalités du Traitement</h2>
    <p>RAGpy traite vos données pour :</p>
    <ul>
        <li>Permettre l'authentification et la gestion de votre compte</li>
        <li>Fournir les services de veille scientifique automatisée</li>
        <li>Améliorer la qualité du service (statistiques anonymisées)</li>
    </ul>

    <h2>3. Base Légale</h2>
    <p>
        Le traitement est fondé sur l'<strong>Article 6§1(e) du RGPD</strong> :
        exécution d'une mission d'intérêt public (recherche scientifique).
    </p>

    <h2>4. Données Collectées</h2>
    <table>
        <tr><th>Donnée</th><th>Finalité</th><th>Durée</th></tr>
        <tr><td>Email institutionnel</td><td>Authentification</td><td>Durée du compte + 1 an</td></tr>
        <tr><td>Nom, Prénom</td><td>Identification</td><td>Durée du compte + 1 an</td></tr>
        <tr><td>Documents uploadés</td><td>Traitement</td><td>24h après traitement</td></tr>
        <tr><td>Logs de connexion</td><td>Sécurité</td><td>90 jours</td></tr>
    </table>

    <h2>5. Destinataires</h2>
    <p>Vos données peuvent être transmises à :</p>
    <ul>
        <li><strong>Mistral AI</strong> (France) : OCR des documents</li>
        <li><strong>OpenAI</strong> (USA) : Génération d'embeddings - Transfert encadré par DPA</li>
        <li><strong>Pinecone</strong> (USA) : Stockage vectoriel - Transfert encadré par SCCs</li>
    </ul>

    <h2>6. Vos Droits</h2>
    <p>Conformément au RGPD, vous disposez des droits suivants :</p>
    <ul>
        <li><strong>Accès</strong> (Art. 15) : Obtenir une copie de vos données</li>
        <li><strong>Rectification</strong> (Art. 16) : Corriger vos données</li>
        <li><strong>Effacement</strong> (Art. 17) : Supprimer vos données</li>
        <li><strong>Portabilité</strong> (Art. 20) : Recevoir vos données dans un format structuré</li>
        <li><strong>Opposition</strong> (Art. 21) : Vous opposer au traitement</li>
    </ul>
    <p>Pour exercer vos droits : <a href="mailto:[DPO_EMAIL]">[DPO_EMAIL]</a></p>

    <h2>7. Réclamation</h2>
    <p>
        Vous pouvez introduire une réclamation auprès de la CNIL :<br>
        <a href="https://www.cnil.fr/fr/plaintes">www.cnil.fr/fr/plaintes</a>
    </p>

    <p><em>Dernière mise à jour : [DATE]</em></p>
</body>
</html>
```

### 9.4 Phase 2 : Conformité RGPD (Semaine 2-3)

| # | Action | Article RGPD | Fichier | Effort |
|---|--------|--------------|---------|--------|
| 6 | Consentement inscription | Art. 7 | `app/models/user.py` | 4h |
| 7 | Suppression complète | Art. 17 | `app/routes/users.py` | 8h |
| 8 | Export données | Art. 20 | `app/routes/users.py` | 6h |
| 9 | Registre traitements | Art. 30 | Documentation | 4h |
| 10 | DPA avec APIs | Art. 28 | Documentation | 2h |

#### Code Action 6 : Champs Consentement

```python
# app/models/user.py - Ajouter colonnes

from sqlalchemy import Column, Boolean, DateTime, String

class User(Base):
    # ... colonnes existantes ...

    # Nouveaux champs consentement
    consent_to_processing = Column(Boolean, default=False, nullable=False)
    consent_date = Column(DateTime(timezone=True), nullable=True)
    consent_version = Column(String(20), nullable=True)  # Ex: "v1.0"
    consent_ip = Column(String(45), nullable=True)  # IP au moment du consentement
```

```python
# app/schemas/auth.py - Modifier RegisterRequest

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    accept_terms: bool  # Existant
    consent_to_processing: bool  # NOUVEAU - Obligatoire

    @field_validator("consent_to_processing")
    @classmethod
    def must_consent(cls, v: bool) -> bool:
        if not v:
            raise ValueError(
                "Vous devez consentir au traitement de vos données pour utiliser ce service"
            )
        return v
```

#### Code Action 7 : Suppression Complète

```python
# app/routes/users.py - Modifier delete_my_account

from app.services.data_deletion import DataDeletionService

@router.delete("/me")
async def delete_my_account(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Supprime complètement le compte utilisateur et toutes ses données.
    Conformité Article 17 RGPD (Droit à l'effacement).
    """
    deletion_service = DataDeletionService(db)

    try:
        # 1. Supprimer des bases vectorielles
        await deletion_service.delete_from_vector_dbs(current_user.id)

        # 2. Supprimer fichiers physiques
        await deletion_service.delete_user_files(current_user.id)

        # 3. Supprimer sessions pipeline
        await deletion_service.delete_pipeline_sessions(current_user.id)

        # 4. Supprimer projets
        await deletion_service.delete_user_projects(current_user.id)

        # 5. Anonymiser audit logs (conservation pour sécurité)
        await deletion_service.anonymize_audit_logs(current_user.id)

        # 6. Supprimer utilisateur
        db.delete(current_user)
        db.commit()

        return {"message": "Compte et données supprimés avec succès"}

    except Exception as e:
        db.rollback()
        logger.error(f"Erreur suppression compte {current_user.id}: {e}")
        raise HTTPException(500, "Erreur lors de la suppression")
```

```python
# app/services/data_deletion.py - NOUVEAU FICHIER

import os
import shutil
from sqlalchemy.orm import Session
from app.models.user import User
from app.models.project import Project
from app.models.pipeline_session import PipelineSession
from app.models.audit import AuditLog

class DataDeletionService:
    """Service de suppression conforme Article 17 RGPD."""

    def __init__(self, db: Session):
        self.db = db

    async def delete_from_vector_dbs(self, user_id: int):
        """Supprime les données des bases vectorielles."""
        # Récupérer les sessions de l'utilisateur
        sessions = self.db.query(PipelineSession).filter(
            PipelineSession.user_id == user_id
        ).all()

        for session in sessions:
            if session.vector_db_type == "pinecone":
                await self._delete_from_pinecone(session)
            elif session.vector_db_type == "weaviate":
                await self._delete_from_weaviate(session)
            elif session.vector_db_type == "qdrant":
                await self._delete_from_qdrant(session)

    async def delete_user_files(self, user_id: int):
        """Supprime les fichiers physiques."""
        sessions = self.db.query(PipelineSession).filter(
            PipelineSession.user_id == user_id
        ).all()

        for session in sessions:
            session_path = os.path.join("uploads", session.session_folder)
            if os.path.exists(session_path):
                shutil.rmtree(session_path)

    async def anonymize_audit_logs(self, user_id: int):
        """Anonymise les logs (conservation pour sécurité)."""
        self.db.query(AuditLog).filter(
            AuditLog.user_id == user_id
        ).update({
            "user_id": None,
            "ip_address": "ANONYMIZED",
            "details": None
        })
```

#### Code Action 8 : Export Données (Portabilité)

```python
# app/routes/users.py - Ajouter endpoint

@router.get("/me/export")
async def export_my_data(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Exporte toutes les données de l'utilisateur.
    Conformité Article 20 RGPD (Droit à la portabilité).

    Returns:
        JSON structuré avec toutes les données personnelles
    """
    from app.services.data_export import DataExportService

    export_service = DataExportService(db)
    data = await export_service.export_user_data(current_user.id)

    return JSONResponse(
        content=data,
        headers={
            "Content-Disposition": f"attachment; filename=ragpy_export_{current_user.id}.json"
        }
    )
```

```python
# app/services/data_export.py - NOUVEAU FICHIER

from datetime import datetime
from sqlalchemy.orm import Session

class DataExportService:
    """Service d'export conforme Article 20 RGPD."""

    def __init__(self, db: Session):
        self.db = db

    async def export_user_data(self, user_id: int) -> dict:
        """Exporte toutes les données de l'utilisateur."""
        user = self.db.query(User).filter(User.id == user_id).first()

        return {
            "export_date": datetime.utcnow().isoformat(),
            "export_version": "1.0",
            "user_profile": {
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "organization": user.organization,
                "title": user.title,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "last_login": user.last_login.isoformat() if user.last_login else None,
            },
            "consent": {
                "consent_to_processing": user.consent_to_processing,
                "consent_date": user.consent_date.isoformat() if user.consent_date else None,
                "consent_version": user.consent_version,
            },
            "projects": await self._export_projects(user_id),
            "pipeline_sessions": await self._export_sessions(user_id),
            "audit_logs": await self._export_audit_logs(user_id),
        }

    async def _export_projects(self, user_id: int) -> list:
        projects = self.db.query(Project).filter(Project.owner_id == user_id).all()
        return [
            {
                "name": p.name,
                "description": p.description,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in projects
        ]

    async def _export_sessions(self, user_id: int) -> list:
        sessions = self.db.query(PipelineSession).filter(
            PipelineSession.user_id == user_id
        ).all()
        return [
            {
                "session_folder": s.session_folder,
                "source_type": s.source_type,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "status": s.status,
            }
            for s in sessions
        ]

    async def _export_audit_logs(self, user_id: int) -> list:
        logs = self.db.query(AuditLog).filter(AuditLog.user_id == user_id).all()
        return [
            {
                "action": l.action,
                "created_at": l.created_at.isoformat() if l.created_at else None,
                "success": l.success,
            }
            for l in logs
        ]
```

### 9.5 Phase 3 : Sécurité Avancée (Semaine 3-4)

| # | Action | Référence | Effort |
|---|--------|-----------|--------|
| 11 | DPIA | Art. 35 RGPD | 8h (avec DPO) |
| 12 | Chiffrement SQLite | Art. 32 RGPD | 4h |
| 13 | Intégration SSO/CAS | PSSI | 16h |
| 14 | Procédure notification | Art. 33 RGPD | 4h |
| 15 | Audit de sécurité | PSSI | 8h |

#### Modèle DPIA Simplifié

```markdown
# Analyse d'Impact sur la Protection des Données (DPIA)
# RAGpy - [NOM UNIVERSITÉ]

## 1. Description du Traitement

### Finalité
Veille scientifique automatisée par analyse de documents académiques.

### Données traitées
- Identifiants utilisateurs (email, nom)
- Documents scientifiques (PDFs, articles)
- Métadonnées bibliographiques
- Logs d'utilisation

### Technologies utilisées
- OCR : Mistral AI, OpenAI Vision
- Embeddings : OpenAI text-embedding-3-large
- Stockage vectoriel : Pinecone/Weaviate/Qdrant
- Base de données : SQLite

## 2. Nécessité et Proportionnalité

### Nécessité
Le traitement est nécessaire pour permettre aux chercheurs d'effectuer
une veille scientifique efficace sur de grands corpus documentaires.

### Proportionnalité
- Minimisation : seules les données nécessaires sont collectées
- Limitation de la conservation : 24h pour les documents traités
- Pseudonymisation : embeddings non réversibles

## 3. Risques Identifiés

| Risque | Probabilité | Gravité | Score |
|--------|-------------|---------|-------|
| Fuite de données personnelles | Moyenne | Élevée | 6/9 |
| Accès non autorisé | Faible | Élevée | 4/9 |
| Perte de données | Faible | Moyenne | 2/9 |
| Transfert USA non encadré | Élevée | Moyenne | 6/9 |

## 4. Mesures d'Atténuation

| Risque | Mesure | Responsable | Délai |
|--------|--------|-------------|-------|
| Fuite données | Chiffrement SQLite | DSI | S+2 |
| Accès non autorisé | SSO/CAS | DSI | S+4 |
| Perte données | Sauvegardes quotidiennes | DSI | S+1 |
| Transfert USA | DPA OpenAI + Pinecone | DPO | S+2 |

## 5. Avis du DPO

[À compléter par le DPO]

## 6. Décision

□ Traitement autorisé sans réserve
□ Traitement autorisé avec mesures ci-dessus
□ Traitement refusé - Consulter CNIL

Date : ____________
Signature DPO : ____________
Signature Responsable : ____________
```

### 9.6 Phase 4 : Fonctionnalités Avancées (Mois 2)

| # | Action | Description | Effort |
|---|--------|-------------|--------|
| 16 | Mode offline | Embeddings locaux | 24h |
| 17 | Anonymisation auto | Hash métadonnées | 8h |
| 18 | Audit transmissions | Log API calls | 8h |
| 19 | Interface DPO | Dashboard conformité | 16h |
| 20 | Tests RGPD | Tests automatisés | 8h |

### 9.7 Checklist Finale - Institution Universitaire

```
GOUVERNANCE
□ DPO informé et impliqué
□ DSI validé architecture
□ RSSI validé sécurité
□ Porteur projet identifié

PHASE 1 - CRITIQUE (Semaine 1)
□ JWT_SECRET_KEY production
□ CORS restrictif configuré
□ crawl.py corrigé complet
□ Privacy Policy publiée
□ Notice d'information visible

PHASE 2 - RGPD (Semaine 2-3)
□ Consentement inscription implémenté
□ Suppression complète (cascade)
□ Export données (portabilité)
□ Registre des traitements
□ DPA signés avec APIs

PHASE 3 - SÉCURITÉ (Semaine 3-4)
□ DPIA réalisée et validée
□ Chiffrement base de données
□ SSO/CAS intégré
□ Procédure notification CNIL
□ Audit sécurité passé

PHASE 4 - AVANCÉ (Mois 2)
□ Mode offline disponible
□ Anonymisation configurable
□ Audit des transmissions
□ Interface DPO
□ Tests conformité automatisés

DOCUMENTATION
□ LEGAL.md
□ PRIVACY.md
□ REGISTRE_TRAITEMENTS.md
□ DPIA.md
□ PROCEDURE_INCIDENT.md
```

**Temps total estimé** : 3-4 semaines (1 développeur + coordination)

---

## 10. Annexes - Références Légales Complètes

### 10.1 RGPD - Articles Clés (Texte Intégral)

#### Article 2§2(c) - Exception Domestique

> "Le présent règlement ne s'applique pas au traitement de données à caractère personnel effectué:
> [...]
> c) par une personne physique dans le cadre d'une activité strictement personnelle ou domestique;"

**Source** : [EUR-Lex Art. 2](https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX%3A32016R0679#d1e1555-1-1)

#### Article 6 - Licéité du Traitement

> "1. Le traitement n'est licite que si, et dans la mesure où, au moins une des conditions suivantes est remplie:
> a) la personne concernée a consenti au traitement de ses données à caractère personnel pour une ou plusieurs finalités spécifiques;
> [...]
> e) le traitement est nécessaire à l'exécution d'une mission d'intérêt public ou relevant de l'exercice de l'autorité publique dont est investi le responsable du traitement;
> f) le traitement est nécessaire aux fins des intérêts légitimes poursuivis par le responsable du traitement ou par un tiers, à moins que ne prévalent les intérêts ou les libertés et droits fondamentaux de la personne concernée [...]"

**Source** : [EUR-Lex Art. 6](https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX%3A32016R0679#d1e1883-1-1)

#### Article 17 - Droit à l'Effacement

> "1. La personne concernée a le droit d'obtenir du responsable du traitement l'effacement, dans les meilleurs délais, de données à caractère personnel la concernant et le responsable du traitement a l'obligation d'effacer ces données à caractère personnel dans les meilleurs délais, lorsque l'un des motifs suivants s'applique:
> a) les données à caractère personnel ne sont plus nécessaires au regard des finalités pour lesquelles elles ont été collectées ou traitées d'une autre manière;
> [...]"

**Source** : [EUR-Lex Art. 17](https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX%3A32016R0679#d1e2589-1-1)

#### Article 89 - Garanties pour la Recherche

> "1. Le traitement à des fins archivistiques dans l'intérêt public, à des fins de recherche scientifique ou historique, ou à des fins statistiques est soumis, conformément au présent règlement, à des garanties appropriées pour les droits et libertés de la personne concernée. Ces garanties garantissent la mise en place de mesures techniques et organisationnelles, en particulier pour assurer le respect du principe de minimisation des données. Ces mesures peuvent comprendre la pseudonymisation, dans la mesure où ces finalités peuvent être atteintes de cette manière."

**Source** : [EUR-Lex Art. 89](https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX%3A32016R0679#d1e6075-1-1)

### 10.2 Code de la Propriété Intellectuelle - Articles Clés

#### Article L.122-5-3 - Exception TDM

> "Lorsque l'oeuvre a été divulguée, l'auteur ne peut interdire :
> [...]
> 10° Les copies ou reproductions numériques réalisées à partir d'une source licite, en vue de l'exploration de textes et de données incluses ou associées aux écrits scientifiques pour les besoins de la recherche publique, à l'exclusion de toute finalité commerciale. Un décret fixe les conditions dans lesquelles l'exploration des textes et des données est mise en œuvre, ainsi que les modalités de conservation et de communication des fichiers produits au terme des activités de recherche pour lesquelles elles ont été produites ;"

**Source** : [Légifrance L.122-5](https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000037388886)

#### Article L.342-1 - Protection des Bases de Données

> "Le producteur d'une base de données, entendu comme la personne qui prend l'initiative et le risque des investissements correspondants, bénéficie d'une protection du contenu de la base lorsque la constitution, la vérification ou la présentation de celui-ci atteste d'un investissement financier, matériel ou humain substantiel.
> Cette protection est indépendante et s'exerce sans préjudice de celles résultant du droit d'auteur ou d'un autre droit sur la base de données ou un de ses éléments constitutifs."

**Source** : [Légifrance L.342-1](https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000006279245)

### 10.3 Directive 2019/790 - Exception TDM

#### Article 3 - Fouille de Textes et de Données à des Fins de Recherche Scientifique

> "1. Les États membres prévoient une exception aux droits visés à l'article 5, point a), et à l'article 7, paragraphe 1, de la directive 96/9/CE, à l'article 2 de la directive 2001/29/CE, et à l'article 15, paragraphe 1, de la présente directive pour les reproductions et extractions effectuées par des organismes de recherche et des institutions du patrimoine culturel en vue de procéder, à des fins de recherche scientifique, à une fouille de textes et de données sur des œuvres ou autres objets protégés auxquels ils ont un accès licite.
>
> 2. Les copies d'œuvres ou d'autres objets protégés réalisées conformément au paragraphe 1 sont stockées avec un niveau de sécurité approprié et peuvent être conservées à des fins de recherche scientifique, y compris pour la vérification des résultats de la recherche."

**Source** : [EUR-Lex Directive 2019/790 Art. 3](https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX%3A32019L0790#d1e1018-92-1)

### 10.4 Liens Utiles

| Ressource | URL |
|-----------|-----|
| CNIL - Guide RGPD | https://www.cnil.fr/fr/rgpd-de-quoi-parle-t-on |
| CNIL - Recherche scientifique | https://www.cnil.fr/fr/recherche-scientifique |
| EUR-Lex RGPD | https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX%3A32016R0679 |
| Légifrance CPI | https://www.legifrance.gouv.fr/codes/id/LEGITEXT000006069414/ |
| OpenAI DPA | https://openai.com/policies/data-processing-addendum/ |
| Mistral Terms | https://mistral.ai/terms/ |
| Pinecone Terms | https://www.pinecone.io/terms/ |

---

## Historique des Révisions

| Version | Date | Auteur | Modifications |
|---------|------|--------|---------------|
| 1.0 | 2025-12-15 | Claude Code | Création initiale |

---

*Document généré par Claude Code (Anthropic) - À valider par un juriste qualifié avant application.*
