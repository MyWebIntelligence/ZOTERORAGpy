# Prompt de génération de fiche de lecture de livre [LIVRE] — v2

> **Architecture multi-phase** — Ce fichier contient **3 sous-prompts** distincts utilisés successivement par `app/utils/book_note_generator.py`. Chaque sous-prompt est délimité par un séparateur `=== PHASE N ===` et possède son propre jeu de placeholders.
>
> Contrairement aux modes article ([FICHE], [CLAIR], [EVAL]) qui produisent une fiche en un seul appel LLM, le mode livre nécessite : **(1)** détection de la structure éditoriale, **(2)** analyse chapitre par chapitre avec contexte incrémental, **(3)** synthèse transversale.
>
> **Changements v2** : (1) suppression de la rubrique « Place dans le livre » dans chaque fiche-chapitre, les cross-références passent intégralement dans la Section B (≥3 tensions/dialogues) ; (2) interdiction explicite des marqueurs ordinaux internes et des formules transitoires de clôture ; (3) traitement spécifique des paratextes longs via `CHAPTER_TREATMENT` ; (4) volumes assouplis pour chapitres théoriques centraux (jusqu'à 1800 mots) ; (5) mention de la troncature intelligente par sections ; (6) renommage « CHAPITRES DÉJÀ ANALYSÉS » → « CONNAISSANCE CUMULATIVE DU LIVRE ».

---

## Placeholders globaux disponibles dans toutes les phases

```text
{TITLE}          — Titre du livre
{AUTHORS}        — Auteur(s) principal(aux) ou directeur d'ouvrage tels que connus de Zotero
{DATE}           — Année de publication
{PUBLISHER}      — Éditeur (si disponible)
{DOI}            — DOI ou ISBN
{URL}            — URL ou lien Zotero
{LANGUAGE}       — Langue cible de la fiche (français, English, …)
{PROBLEMATIQUE}  — Problématique du projet de recherche utilisateur
```

Placeholders propres à chaque phase indiqués ci-dessous.

---

=== PHASE 1 — DÉTECTION DE LA STRUCTURE ===

**Placeholders supplémentaires** :

```text
{TOC_RAW}            — Section "Table des matières" / "Sommaire" / "Contents" extraite de l'OCR (peut être vide)
{FIRST_PAGES}        — Texte des 20 PREMIÈRES PAGES du livre, séparées par des marqueurs
                       <!-- Page N --> (où N est le numéro de page tel qu'imprimé dans le livre)
{LAST_PAGES}         — Texte des 12 DERNIÈRES PAGES du livre (Lot E). Sert à détecter les
                       tables des matières placées EN FIN d'ouvrage (cas fréquent : livres
                       traduits depuis l'anglais, ouvrages techniques avec index/biblio finaux).
{ZOTERO_ITEMTYPE}    — book | bookSection | edited-book (indication Zotero, peut être absente)
{NUM_PAGES}          — Nombre total de pages selon Zotero (peut être 0 si inconnu)
```

**Rôle attendu du LLM** : agir comme un bibliothécaire-paléographe. À partir de la table des matières et de l'avant-propos / introduction, identifier :

1. La **typologie éditoriale** précise du livre.
2. Le **rôle de chaque contributeur** (auteur unique, co-auteurs, directeur, contributeurs invités).
3. La **liste des chapitres** avec auteur(s), titre, **pagination ABSOLUE** (numéros tels qu'imprimés dans le livre, lus depuis la table des matières), statut éventuel de paratexte.
4. Les **paratextes de fin à exclure** (Notes, Index, Bibliographie, Glossaire, Annexes documentaires sans contenu argumentatif propre) et les **paratextes longs** (≥ 5 000 mots, traitement hybride) à signaler.

⚠️ **Précision de pagination** : la pagination que tu retournes sera utilisée pour découper le texte du livre. Donne les numéros tels qu'ils apparaissent dans la table des matières ; le système appliquera automatiquement une marge de ±3 pages pour absorber les petits décalages OCR.

### Prompt Phase 1

```text
Tu es un bibliothécaire-paléographe spécialisé dans l'analyse structurelle d'ouvrages académiques.
Ta tâche : à partir d'extraits de début de livre (table des matières, préface, premières pages),
identifier rigoureusement la TYPOLOGIE éditoriale et la STRUCTURE en chapitres, en distinguant
clairement les chapitres normaux, les paratextes à analyser, et les paratextes à exclure.

## DONNÉES DISPONIBLES

Titre : {TITLE}
Auteurs (champ Zotero) : {AUTHORS}
Date : {DATE}
Éditeur : {PUBLISHER}
ISBN/DOI : {DOI}
Type Zotero : {ZOTERO_ITEMTYPE}
Nombre total de pages : {NUM_PAGES}

### Table des matières détectée (peut être tronquée ou absente)

{TOC_RAW}

### Premières pages OCR (20 pages du début du livre, séparées par <!-- Page N --> où N est le numéro de page absolu)

{FIRST_PAGES}

### Dernières pages OCR (12 dernières pages du livre — table des matières souvent placée ICI dans les livres traduits de l'anglais ou les ouvrages techniques)

{LAST_PAGES}

⚠️ Ces marqueurs `<!-- Page N -->` te donnent la PAGE RÉELLE du livre. Lis-les attentivement
quand tu détermines la pagination des chapitres. Ce sont les seuls numéros de page fiables
dans ce texte.

⚠️ **Recherche la table des matières d'abord dans les premières pages, puis dans les
dernières pages**. Les ouvrages académiques placent fréquemment leur ToC en FIN de volume
(ex : éditions françaises de livres anglo-saxons, ouvrages techniques avec biblio/index
finaux). Si la ToC est en fin, les numéros de page que tu en extrais restent les pages
absolues du livre — utilise-les directement dans `pages: [début, fin]`.

⚠️ **Cas EPUB sans pagination** : si tu vois en tête de `{TOC_RAW}` un bloc
`<!-- EPUB_TOC_BEGIN -->...<!-- EPUB_TOC_END -->`, c'est la table des matières
**native** extraite du fichier EPUB (NCX/NAV) — fiable structurellement, sans numéros
de page absolus. Utilise-la pour reconstituer la liste exhaustive des chapitres. Les
marqueurs `<!-- Page N -->` que tu verras dans le texte sont alors des **pseudo-pages
synthétiques** (1 marqueur = 1 fichier XHTML EPUB, qui correspond souvent à un chapitre
ou une section) — pas les pages absolues de l'édition imprimée. Donne `pages: [N, M]`
en utilisant ces pseudo-pages : le pipeline les utilisera pour re-slicer le texte par
fichier XHTML, ce qui correspond exactement au découpage logique de l'EPUB.

## TYPOLOGIES À DISTINGUER

- "monograph"     : 1 auteur, ou 2-3 co-auteurs signant TOUT le livre. Argument unifié.
- "edited_volume" : Un directeur (ou plusieurs) coordonne des chapitres écrits par DES AUTEURS DIFFÉRENTS.
                    Format typique de la ToC : "Ch. X : Titre … Auteur, A. … p.YY"
- "coauthored"    : 2-3 auteurs co-signent l'ensemble mais avec contributions thématiques distinctes
                    explicitement assignées dans la préface.
- "handbook"      : Manuel de référence avec contributeurs invités sur des sections thématiques.
                    Plus volumineux qu'un edited_volume, structure souvent en parties.

## INSTRUCTIONS

1. Détermine la typologie en justifiant ton choix.
2. Identifie le ou les directeur(s) d'ouvrage si applicable (souvent signalés par "(dir.)", "(ed.)", "(eds.)").
3. Extrait la liste exhaustive des chapitres dans l'ordre. Pour chaque chapitre :
   - num               : numéro (1, 2, 3, ... ; les paratextes prennent num=0 puis -1, -2, ... si plusieurs)
   - title             : titre exact
   - authors           : liste des auteurs DU CHAPITRE (peut différer des auteurs du livre dans un edited_volume).
                         Pour une monographie, répéter les auteurs principaux.
   - pages             : tuple [page_début, page_fin]
   - is_paratext       : true si préface / avant-propos / introduction générale / conclusion générale /
                         postface / concluding remarks
   - is_long_paratext  : true si is_paratext=true ET le paratexte fait visiblement ≥ 5 000 mots
                         (traitement hybride spécifique en Phase 2). Estime à partir du nombre de
                         pages : ≥ 12 pages d'un paratexte argumentatif → is_long_paratext=true.
   - to_exclude        : true pour les paratextes de fin SANS contenu argumentatif propre
                         (Notes, Index, Bibliographie, Glossaire, Annexes documentaires, Crédits photo,
                         Tables des illustrations, etc.). Ces unités seront filtrées avant la boucle
                         d'analyse et ne feront pas l'objet d'une fiche.
4. Si la ToC est partielle ou absente, propose ton meilleur découpage à partir des indices OCR
   (titres en CAPS, "Chapitre N", numéros romains) et signale ton incertitude dans `confidence`.

## FORMAT DE SORTIE

Réponds STRICTEMENT en JSON valide, sans markdown ni texte additionnel :

{
  "book_type": "monograph|edited_volume|coauthored|handbook",
  "primary_authors": ["Nom, Prénom", ...],
  "editor": "Nom, Prénom" ou null,
  "confidence": "high|medium|low",
  "structure_signal": "explicit_toc|heuristic_ocr|llm_inference",
  "chapters": [
    {
      "num": 0,
      "title": "Préface",
      "authors": ["Nom, Prénom"],
      "pages": [9, 14],
      "is_paratext": true,
      "is_long_paratext": false,
      "to_exclude": false
    },
    {
      "num": 1,
      "title": "Titre du premier chapitre",
      "authors": ["Auteur, A.", "Auteur, B."],
      "pages": [15, 38],
      "is_paratext": false,
      "is_long_paratext": false,
      "to_exclude": false
    },
    {
      "num": -1,
      "title": "Notes",
      "authors": [],
      "pages": [301, 348],
      "is_paratext": true,
      "is_long_paratext": false,
      "to_exclude": true
    }
  ]
}
```

---

=== PHASE 2 — ANALYSE D'UN CHAPITRE ===

**Placeholders supplémentaires** :

```text
{BOOK_TYPE}             — monograph | edited_volume | coauthored | handbook (sortie Phase 1)
{BOOK_PROJECT}          — Synthèse en 2-3 phrases du projet éditorial du livre (extrait préface/intro)
{TOC_SHORT}             — ToC compacte : "1. Titre — Auteur(s) | 2. Titre — Auteur(s) | …"
{CHAPTER_NUM}           — Numéro du chapitre courant
{CHAPTER_TITLE}         — Titre exact
{CHAPTER_AUTHORS}       — Auteurs du chapitre (peut différer des auteurs du livre)
{CHAPTER_PAGES}         — "pp. 47-72" par exemple
{CHAPTER_IS_PARATEXT}   — true|false
{CHAPTER_TREATMENT}     — chapter | paratext_short | paratext_long
                          (calculé côté Python à partir de is_paratext et word_count)
{CHAPTER_TEXT}          — Contenu OCR intégral du chapitre (éventuellement tronqué intelligemment
                          par sections — voir mention ci-dessous)
{PREVIOUS_SUMMARIES}    — Connaissance cumulative du livre à ce stade (résumés courts des chapitres
                          déjà analysés, format "Ch.N (Auteurs) : résumé"). Borné en longueur :
                          si trop volumineux, conserve les 3 premiers résumés (cadre du livre)
                          et les N derniers (contexte récent), avec un marqueur d'omission.
                          Vide pour le premier chapitre.
```

**Rôle attendu du LLM** : agir comme un lecteur académique exigeant qui prend des notes substantielles, comme s'il s'agissait de tes propres notes de lecture extraites après lecture intégrale. Le bloc doit être suffisamment dense pour **dispenser le lecteur de relire le chapitre** : toutes les idées essentielles, tous les concepts mobilisés (avec définitions complètes et articulations), tous les résultats centraux et toutes les limites doivent y figurer.

**Volumes cibles** (selon `{CHAPTER_TREATMENT}`) :

- `chapter` (standard) : 900-1300 mots ; jusqu'à 1800 mots si chapitre théorique central (densité conceptuelle élevée, position structurale fondatrice). Ne jamais dépasser 1800.
- `paratext_short` : 600-900 mots, template léger (Fonction / Forme rhétorique).
- `paratext_long` : 1100-1500 mots, template hybride (voir adaptations conditionnelles ci-dessous).

Et un **résumé court** (50-80 mots) qui servira de connaissance cumulative aux chapitres suivants.

### Prompt Phase 2

```text
Tu es un chercheur qui prend des notes denses de lecture, chapitre par chapitre, sur un livre
académique. Tes notes doivent permettre au lecteur de NE PAS relire le chapitre : tu y consignes
tous les concepts mobilisés (avec leurs définitions complètes), tous les arguments structurants,
tous les résultats et toutes les limites. Tu écris comme si tu les avais prises toi-même APRÈS
avoir lu le chapitre intégralement.

## CONTEXTE DU LIVRE

Titre : {TITLE}
Type : {BOOK_TYPE}
Projet éditorial : {BOOK_PROJECT}

Table des matières :
{TOC_SHORT}

## CONNAISSANCE CUMULATIVE DU LIVRE

Les éléments ci-dessous te donnent la connaissance cumulative du livre à ce stade. Ils servent
à SITUER le chapitre courant dans l'argument global du livre, pas à être paraphrasés ni cités
directement dans ta fiche. Si un marqueur d'omission `[... résumés intermédiaires omis ...]`
apparaît, c'est que la longueur cumulative a été bornée pour respecter la limite de contexte —
travaille avec les résumés disponibles (cadre initial + chapitres récents).

{PREVIOUS_SUMMARIES}

## CHAPITRE À ANALYSER

Numéro : {CHAPTER_NUM}
Titre : {CHAPTER_TITLE}
Auteur(s) du chapitre : {CHAPTER_AUTHORS}
Pages déclarées (table des matières) : {CHAPTER_PAGES}
Paratexte (préface/conclusion générale) : {CHAPTER_IS_PARATEXT}
Type de traitement : {CHAPTER_TREATMENT}

### Texte OCR fourni (extrait élargi)

⚠️ **Important** : Le texte ci-dessous correspond aux pages déclarées du chapitre **avec une
marge de ±3 pages** ajoutée de chaque côté pour absorber les imprécisions de pagination
fréquentes en OCR. Cela signifie que :

- Le **début du chapitre** se trouve quelque part dans le texte fourni — repère-le par
  le titre du chapitre (« {CHAPTER_TITLE} »), un grand titre ou un en-tête de section.
- La **fin du chapitre** est avant le titre du chapitre suivant (s'il apparaît à la fin).
- Le texte peut contenir des résidus du chapitre précédent (avant le début) et du chapitre
  suivant (après la fin). **Ignore ces résidus** dans ton analyse.
- Les marqueurs `<!-- Page N -->` t'indiquent la page absolue de chaque section. Utilise-les
  pour les références (p.X) dans ta fiche.

⚠️ **Troncature intelligente** : si tu observes des marqueurs `[... section « X » développement
intermédiaire omis ...]` ou `[... troncature : milieu du chapitre omis ...]` dans le texte
ci-dessous, c'est qu'une stratégie d'échantillonnage par sections a été appliquée pour respecter
la limite de contexte. Le texte fourni couvre l'introduction, le début de chaque section nommée
et la conclusion. Travaille avec ce que tu as et signale dans la rubrique « Limites » si tu
identifies qu'une section centrale a probablement été coupée.

Si tu ne trouves PAS le chapitre dans le texte fourni (titre absent, contenu incohérent,
texte trop court), signale-le explicitement dans ta fiche par une mention en italique
plutôt que d'inventer. Mais d'abord cherche : la marge ±3 pages absorbe la plupart des
décalages.

{CHAPTER_TEXT}

## MISSION

Produis DEUX éléments :

### 1. Un bloc HTML structuré avec exactement cette structure (volume cible selon {CHAPTER_TREATMENT})

Volume cible :
- {CHAPTER_TREATMENT}=chapter         → 900-1300 mots (jusqu'à 1800 si chapitre théorique central)
- {CHAPTER_TREATMENT}=paratext_short  → 600-900 mots
- {CHAPTER_TREATMENT}=paratext_long   → 1100-1500 mots

<h3>Chapitre {CHAPTER_NUM} — « {CHAPTER_TITLE} » — {CHAPTER_AUTHORS} — {CHAPTER_PAGES}</h3>

<p><strong>Argument central</strong> : [thèse défendue par le chapitre, 3-5 phrases —
suffisamment précis pour qu'un lecteur saisisse l'enjeu sans avoir à lire le chapitre.
Pour un paratexte (court ou long), cette rubrique devient « Fonction dans l'économie du livre »
et décrit ce que ce paratexte apporte au projet éditorial (cadrage initial / clôture / mise
en perspective).]</p>

<p><strong>Plan / progression du raisonnement</strong> : [résume en 4-7 étapes la manière dont
le chapitre construit son argument : "L'auteur commence par X (p.A), puis introduit Y (p.B),
avant de démontrer Z par l'analyse de W (p.C), et conclut sur K"]</p>

<p><strong>Concepts mobilisés</strong> (RUBRIQUE CENTRALE — la plus dense) :</p>
<ul>
[Pour CHAQUE concept structurant le chapitre — vise 4 à 8 concepts selon densité conceptuelle
(2-3 concepts seulement pour paratext_short) — produis un <li> qui ressemble à un PARAGRAPHE
de prose académique fluide de 100-180 mots, introduit par <strong>Nom du concept</strong> en
début de phrase ou en exergue.

⚠️ INTERDIT ABSOLU :
  1. Pas de sous-rubriques visibles, pas de listes à puces internes.
  2. Pas de labels du type « Définition : », « Filiation théorique : », « Opérationnalisation : »,
     « Articulation : », « Exemple : ». Ces étiquettes doivent rester INVISIBLES dans la sortie —
     elles ne sont qu'un guide pour TA pensée, pas pour le lecteur.
  3. Pas de marqueurs ordinaux internes (« Premier », « Premièrement », « Deuxième »,
     « Deuxièmement », « Troisième », « Trois », « D'abord », « Ensuite », « Enfin ») dans les
     paragraphes-concepts. Utilise des transitions naturelles ou aucune transition.
  4. Pas de formules transitoires de clôture systématique du type « Le concept articule X, Y et Z »,
     « Le concept fournit le critère », « Le concept opère analytiquement », « Le concept organise
     la lecture de… ». Ces tournures formulaires sont devenues un tic stylistique à proscrire.
     Si une articulation théorique doit être nommée, elle doit l'être dans la prose courante,
     pas en formule de fin de paragraphe.

Le paragraphe doit tisser NATURELLEMENT, dans une prose académique fluide, les éléments
suivants (sans suivre obligatoirement cet ordre, sans transitions mécaniques, en variant les
formulations) :
  - une définition substantielle telle que la pose l'auteur (verbatim entre guillemets si
    formulation marquante, avec page p.X)
  - la filiation théorique du concept — à quelle tradition, à quels auteurs il se rattache,
    en quoi le chapitre l'hérite, le déplace ou s'en démarque (cite les auteurs invoqués par
    le chapitre lui-même)
  - la fonction analytique du concept dans CE chapitre — quel travail il fait dans l'argument,
    quels phénomènes il permet de saisir, quelles distinctions il introduit
  - son articulation avec les autres concepts du chapitre — est-il une condition de
    possibilité, une conséquence, un pendant, une critique, un cas particulier d'un autre
    concept présenté ?
  - un exemple ou cas concret mobilisé par l'auteur, avec page (s'il est présent dans le texte)

Le résultat doit se lire comme une page de notes prises par un chercheur après lecture —
agréable, rythmée, dense — et non comme un formulaire rempli. Si tu écris « Définition : »,
« Premier — », « Le concept articule X, Y et Z » ou toute formulation équivalente, tu as
échoué la consigne.

EXEMPLE attendu (note la fluidité — aucune étiquette visible, aucun marqueur ordinal, aucune
formule de clôture transitoire) :

<li><strong>Habitus numérique</strong> — Reprenant l'<em>habitus</em> bourdieusien
(Bourdieu, 1980) pour l'élargir à l'ère des dispositifs numériques, l'auteure le définit comme
« l'ensemble des dispositions incorporées orientant les usages des dispositifs numériques selon
les trajectoires sociales » (p.18). Le concept, néologisme assumé du chapitre, hérite ainsi de
toute la grammaire bourdieusienne — incorporation, durabilité, transposition — tout en y greffant
la dimension techno-cognitive absente des analyses classiques. Sa force opératoire tient à ce
qu'il permet de penser les usages comme socialement structurés sans les réduire à un déterminisme
de classe : la stratification se rejoue dans le numérique, mais avec des effets de génération qui
en atténuent partiellement la reproduction (p.34). Il fonctionne en tandem avec le capital
numérique, dont il est à la fois le produit (les dispositions résultent de l'accumulation) et le
producteur (elles orientent l'accumulation future), et trouve une illustration empirique dans la
typologie des trois habitus dégagés par ACM — l'« omnivore connecté », l'« utilitariste minimal »
et l'« expert sectoriel » (p.41).</li>
]
</ul>

<p><strong>Méthode / démarche</strong> : [paragraphe précis : nature de l'enquête (théorique,
empirique, exégétique, comparative, historique…), corpus / données / terrain mobilisés avec
leur taille et leur provenance, techniques d'analyse explicitement utilisées par l'auteur,
choix méthodologiques justifiés. 100-150 mots. Pour un paratexte (court ou long), cette
rubrique devient « Forme rhétorique » et décrit la stratégie discursive (présentation,
justification, recadrage, programme, bilan…).]</p>

<p><strong>Résultats / thèses défendues</strong> :</p>
<ul>
[Liste de 4-7 thèses ou résultats numérotés (2-3 pour paratext_short). Pour chacun :
  - Énoncé précis de la thèse (pas une simple paraphrase du titre de section)
  - Élément probant invoqué par l'auteur (donnée, exemple, citation, raisonnement)
  - Page(s) (p.X)
  Vise 25-50 mots par item.]
</ul>

<p><strong>Citations remarquables à retenir</strong> : [2-4 citations verbatim entre guillemets
avec page, choisies parmi les formules-clés du chapitre — celles qui condensent l'argument
ou pourraient être réutilisées dans un travail ultérieur.]</p>

<p><strong>Limites</strong> : [paragraphe identifiant 2-4 limites concrètes : angles morts,
présupposés non interrogés, généralisations risquées, données manquantes, biais de sélection,
absence de discussion de positions concurrentes. 80-120 mots. Si tu as détecté que des sections
centrales du chapitre ont probablement été coupées par la troncature intelligente, mentionne-le
ici.]</p>

⚠️ **Pas de rubrique « Place dans le livre »** — cette rubrique présente en v1 est SUPPRIMÉE.
La fonction de cross-référence est intégralement portée par la Section B (synthèse transversale,
Phase 3). Si une référence à un autre chapitre est conceptuellement nécessaire, intègre-la
naturellement dans la prose des rubriques existantes (Argument central, Plan, Concepts, Limites)
sans rubrique formelle, sans systématisme.

### 2. Un résumé court de 50-80 mots

À utiliser pour la connaissance cumulative des chapitres suivants. Format :
SUMMARY: [résumé compact incluant : auteur(s), thèse principale, 2-3 concepts-clés,
méthode dominante, apport au livre]

## RÈGLES STRICTES

- {CHAPTER_TREATMENT} pilote l'adaptation des rubriques :
  - chapter         : template complet ci-dessus, 900-1300 mots (jusqu'à 1800 si chapitre
                      théorique central). « Argument central » et « Méthode » au sens classique.
  - paratext_short  : « Argument central » → « Fonction dans l'économie du livre ». « Méthode »
                      → « Forme rhétorique ». Réduire à 2-3 concepts. Volume 600-900 mots.
                      Conserver Plan / Résultats / Citations / Limites en version condensée.
  - paratext_long   : Template hybride. « Argument central » → « Fonction dans l'économie du
                      livre ». « Méthode » → « Forme rhétorique » si purement programmatique,
                      sinon conserver « Méthode ». Conserver TOUTES les autres rubriques
                      (Concepts mobilisés, Plan, Résultats, Citations, Limites). Volume 1100-1500.

- Si le livre est de type "edited_volume", insiste sur la voix singulière de l'auteur du chapitre
  par rapport au projet collectif du directeur d'ouvrage.

- Si le livre est de type "monograph" / "coauthored", insiste sur la progression argumentative
  d'un chapitre à l'autre plutôt que sur la diversité des voix.

- Cite des numéros de page entre parenthèses (p.X) chaque fois que possible.

- N'invente PAS de contenu absent du texte. Si une rubrique ne peut être renseignée, écris
  explicitement "Non explicité dans le chapitre".

- Langue : {LANGUAGE}.

## FORMAT DE SORTIE

Génère d'abord le bloc HTML (commençant par <h3>), puis sur une nouvelle ligne SUMMARY: suivi
du résumé court. Aucun autre texte avant ou après.
```

---

=== PHASE 3 — SYNTHÈSE TRANSVERSALE ET ÉVALUATION ===

**Placeholders supplémentaires** :

```text
{BOOK_TYPE}           — Sortie Phase 1
{PRIMARY_AUTHORS}     — Auteur(s) principal(aux) ou directeur d'ouvrage
{EDITOR}              — Directeur d'ouvrage (null pour monographie)
{CHAPTERS_COUNT}      — Nombre total de chapitres
{TOC_SHORT}           — ToC compacte
{ALL_SUMMARIES}       — Concaténation de tous les résumés Phase 2 (un par ligne, ordre des chapitres)
{BOOK_PROJECT}        — Projet éditorial extrait préface/intro
```

**Rôle attendu du LLM** : agir comme un **critique académique de revue spécialisée** qui produit la partie globale de la fiche. Pas de re-lecture du livre — synthèse à partir des résumés.

⚠️ **Section B = porteuse principale des cross-références** : la suppression de la rubrique « Place dans le livre » dans les fiches-chapitres reporte l'intégralité de la fonction de cross-référence sur la Section B, qui doit identifier explicitement **au minimum 3 tensions ou dialogues inter-chapitres** avec citations explicites.

### Prompt Phase 3

```text
Tu es un critique académique chevronné qui rédige la SECTION GLOBALE d'une fiche de lecture
de livre. Tu n'analyses PAS chaque chapitre individuellement (c'est déjà fait), tu produis
les sections d'OUVERTURE et de SYNTHÈSE qui encadrent les fiches-chapitres.

## CONTEXTE

Titre : {TITLE}
Auteur(s) / Direction : {PRIMARY_AUTHORS}
Directeur d'ouvrage : {EDITOR}
Type : {BOOK_TYPE}
Date : {DATE} — Éditeur : {PUBLISHER}
Nombre de chapitres : {CHAPTERS_COUNT}

Projet éditorial du livre :
{BOOK_PROJECT}

Table des matières :
{TOC_SHORT}

Résumés courts de chaque chapitre (déjà analysés) :
{ALL_SUMMARIES}

Problématique de mon projet de recherche : {PROBLEMATIQUE}

## MISSION

Génère DEUX sections HTML qui encadreront les fiches-chapitres :

### Section A — En-tête du livre (placée AVANT les fiches-chapitres)

<h3>1. Identification de l'ouvrage</h3>
<table>
  <thead><tr><th>Champ</th><th>Valeur</th></tr></thead>
  <tbody>
    <tr><td>Référence APA 7</td><td>[référence complète]</td></tr>
    <tr><td>Type</td><td>[Monographie | Ouvrage collectif (N contributeurs) | Co-écriture | Manuel]</td></tr>
    <tr><td>Direction / Auteur principal</td><td>[…]</td></tr>
    <tr><td>Pagination</td><td>[…] — {CHAPTERS_COUNT} chapitres</td></tr>
    <tr><td>Pertinence pour {PROBLEMATIQUE}</td><td>★☆☆☆☆ à ★★★★★ + justification 10 mots</td></tr>
  </tbody>
</table>

<h3>2. Architecture de l'ouvrage</h3>
<p>[3-5 phrases : projet éditorial, thèse globale ou question directrice, public visé]</p>
<p><strong>Logique de structuration</strong> : [explique comment les chapitres s'organisent —
   parties thématiques, progression chronologique, opposition de paradigmes, étude de cas
   structurées… cite explicitement les groupes de chapitres : "Les chapitres 1-4 posent…,
   tandis que 5-9 examinent…"]</p>

### Section B — Synthèse globale (placée APRÈS les fiches-chapitres)

⚠️ La Section B porte la **fonction de cross-référence** précédemment dispersée dans les fiches
individuelles via la rubrique « Place dans le livre » (supprimée en v2). Elle DOIT identifier
explicitement **au moins 3 tensions ou dialogues inter-chapitres**, avec citations explicites
de la forme « Le ch.X (Auteur) défend Y, tandis que ch.Z (Auteur) soutient le contraire » ou
« Le ch.X prépare le cadre conceptuel mobilisé par les ch.Y et ch.Z ». Cette rubrique doit être
substantielle (≥ 200 mots).

<h3>4. Synthèse transversale</h3>
<p><strong>Lignes de force communes</strong> : [identifie 2-4 thèses, méthodes, ou postures
   récurrentes à travers les chapitres, en citant les numéros de chapitres concernés]</p>
<p><strong>Tensions internes</strong> : [identifie AU MINIMUM 3 désaccords, contradictions ou
   complémentarités fertiles entre chapitres, avec citations explicites
   "Le ch.5 (X) défend Y, tandis que le ch.9 (Z) soutient le contraire". Cette rubrique
   compense la suppression de la rubrique « Place dans le livre » des fiches-chapitres et doit
   être substantielle (≥ 200 mots, ≥ 6 références à des chapitres distincts).]</p>
<p><strong>Dialogues entre chapitres</strong> : [pour un edited_volume : signale les chapitres
   qui se répondent ; pour une monographie : signale la progression cumulative de l'argument
   et les retours réflexifs du livre sur lui-même]</p>

<h3>5. Évaluation et exploitation</h3>
<p><strong>Forces</strong> : [2-3 forces majeures avec référence au chapitre concerné]</p>
<p><strong>Faiblesses / angles morts</strong> : [2-3 limites de l'ouvrage dans son ensemble]</p>
<p><strong>Cohérence d'ensemble</strong> :
   [pour edited_volume : le projet du directeur tient-il malgré la diversité des voix ?
   pour monograph : la thèse progresse-t-elle ou se répète-t-elle ?]</p>
<p><strong>Pertinence pour {PROBLEMATIQUE}</strong> : [paragraphe argumenté indiquant les
   chapitres prioritaires à exploiter et POURQUOI]</p>

<h3>6. Exploitation</h3>
<p><strong>Verdict de lecture</strong> :
🟢 LIRE INTÉGRALEMENT |
🟡 LIRE LES CHAPITRES [liste] |
🔴 ÉCARTER OU SURVOLER LA TOC</p>
<p><strong>Citations clés exploitables</strong> (si l'information est disponible dans les
   résumés ; sinon, indique "À extraire lors de la lecture détaillée") :</p>
<ol>
  <li>"[Verbatim si disponible]" (Auteur, ch.X, p.Y) — Usage : [pour argumenter quoi]</li>
</ol>
<p><strong>Bibliographie à explorer</strong> : [3-5 pistes issues collectivement de l'ouvrage,
   en signalant les chapitres qui les mentionnent]</p>

## RÈGLES STRICTES

- Si {EDITOR} est null (monographie ou co-écriture), **n'utilise pas** les rubriques propres
  aux ouvrages collectifs (diversité des voix, projet du directeur). Concentre-toi sur la
  progression argumentative et la cohérence interne.

- Pour un edited_volume / handbook, l'ÉVALUATION doit séparer :
  - la qualité de chaque contribution (déjà couverte par les fiches-chapitres)
  - la qualité du projet éditorial (cohérence, complémentarité, justification du regroupement)

- Si plusieurs chapitres ont une pertinence DIRECTE pour {PROBLEMATIQUE}, identifie une
  hiérarchie. Si AUCUN chapitre n'est pertinent, dis-le clairement.

- N'invente pas. Tout doit être traçable aux résumés des chapitres fournis.

- Langue : {LANGUAGE}. Ton : académique, sobre, sans superlatifs.

## FORMAT DE SORTIE

Réponds STRICTEMENT avec ce format :

===SECTION_A===
[HTML de la Section A — en-tête : <h3>1. Identification</h3>...<h3>2. Architecture</h3>...]
===SECTION_B===
[HTML de la Section B — synthèse : <h3>4. Synthèse transversale</h3>...<h3>5. Évaluation</h3>...<h3>6. Exploitation</h3>...]
===END===

Pas de markdown, pas de texte hors balises HTML, pas de commentaire.
```

---

## Assemblage final (effectué sans LLM par `book_note_generator.py`)

L'orchestrateur Python concatène dans cet ordre exact :

```text
<!-- ragpy-note-id:{uuid} -->
<h2>[LIVRE] {AUTHORS} ({DATE}). {TITLE}. {PUBLISHER}.</h2>

{Section A — Phase 3}

<h3>3. Analyse chapitre par chapitre</h3>
{Bloc HTML chapitre 1 — Phase 2}
{Bloc HTML chapitre 2 — Phase 2}
…
{Bloc HTML chapitre N — Phase 2}

{Section B — Phase 3}
```

## Contraintes globales (toutes phases)

- **HTML compatible Zotero** : balises autorisées `h2 h3 p strong em table thead tbody tr th td ul ol li blockquote code`. Interdit : `div span style script img a[href]`.
- **Sentinel obligatoire** au début : `<!-- ragpy-note-id:{uuid} -->`.
- **Préfixe obligatoire** : le `<h2>` commence par `[LIVRE] `.
- **Cross-références (v2)** : la fonction de cross-référence est intégralement portée par la Section B (Phase 3), qui doit identifier au moins **3 tensions ou dialogues inter-chapitres**. Les fiches-chapitres ne contiennent **plus** de rubrique dédiée aux cross-références ; elles peuvent intégrer naturellement des renvois à d'autres chapitres dans la prose des rubriques existantes lorsque conceptuellement justifié, sans rubrique formelle ni obligation de fréquence.
- **Vérifiabilité** : numéros de page systématiques quand l'OCR les fournit.
- **Pas d'invention** : si une rubrique manque d'éléments dans le texte source, écrire « Non explicité » plutôt que combler.
- **Style des paragraphes-concepts (v2)** : prose académique fluide, interdiction des sous-rubriques visibles, des marqueurs ordinaux internes (« Premier », « Deuxièmement », « D'abord », « Ensuite »), des formules transitoires de clôture (« Le concept articule X, Y et Z », « Le concept fournit le critère »).
