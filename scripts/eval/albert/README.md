# Évaluation humaine des rendus Albert

Ce dossier outille deux décisions par un jugement humain à l'aveugle, sur le modèle du protocole validé de `moisson_shs` (`scripts/eval/` et `docs/EVALUATION.md`) :

1. **D21 : faut-il recoder les textes LightOnOCR ?** Aujourd'hui `ALBERT_OCR_SKIP_RECODE=0` : les textes `albert_lightonocr` sont recodés par le modèle de chat, comme ceux de PyMuPDF.
2. **Transcription audio (Whisper, `scripts/rad_audio.py`) utilisable sans relecture humaine ?** Le jugement porte sur des extraits écoutés, comparés à leur transcription.

L'évaluation de la variante de recherche (fiche de pertinence des passages) a été retirée le 2026-10-03, avec la fonction de réponses sourcées : RAGpy prépare les corpus, les questions sont posées par d'autres outils.

Les scripts produisent les fiches, relisent les fiches remplies et appliquent des critères fixés d'avance (§ 7). Ils ne modifient **aucune configuration** : le changement reste une décision humaine, prise sur le rapport.

| Script | Rôle | Réseau |
|---|---|---|
| `ocr_sheets.py` | fiche de fidélité OCR (image de la page à côté de la transcription), comparaison A/B facultative | non |
| `audio_sheets.py` | fiche de fidélité des transcriptions audio (extrait écouté à côté de sa transcription) | non (ffmpeg local) |
| `metrics.py` | mesures, accord inter-juges, rapport de décision | non |
| `common.py` | CSV, page HTML autonome, kappa, tirages à graine fixe | non |

## 1. Principe

- **Fiche CSV et page HTML.** Chaque fiche existe en CSV UTF-8 avec BOM (séparateur « ; », s'ouvre dans un tableur) et en page HTML autonome : aucun script, style ni police externe, lisible hors ligne. La page se note au clavier, garde les jugements dans le navigateur (`localStorage`) et produit le fichier à rendre avec « Exporter le CSV ».
- **Échelles explicites**, affichées en tête de page (§ 4).
- **À l'aveugle.** Les éléments sont présentés dans un ordre aléatoire à graine fixe (`--seed`, défaut 20261002), sous des identifiants opaques. Côté OCR, ni le fournisseur, ni la colonne comparée, ni la version brute ne sont signalés ; l'ordre A/B est tiré au hasard pour chaque page. Côté audio, le fournisseur n'est pas affiché.
- **Second jugement partiel.** Environ 20 % des éléments (`--second-ratio 0.2`) sont présentés une seconde fois après les autres, dans un autre ordre, sous d'autres identifiants (`r-<n>`). L'accord se mesure par le kappa de Cohen pondéré quadratique.
- **Réglage séparé de la décision.** Les fiches OCR et audio ne servent qu'à décider : on n'y règle rien. Tout paramètre ajusté l'est sur une campagne antérieure.
- **Critères fixés d'avance** (§ 7), écrits ici et repris dans le rapport.

## 2. Où sont les fiches

Toutes les fiches d'une campagne sont dans **`data/albert_eval/<AAAA-MM-JJ>/`** (option `--out` ou `--dir`, défaut : date du jour). Le dossier `data/` est exclu du dépôt git.

| Fichier | Contenu |
|---|---|
| `fiche_ocr.html` / `fiche_ocr.csv` | fiche de fidélité OCR à remplir (juger dans la page HTML : l'image est nécessaire) |
| `fiche_audio.html` / `fiche_audio.csv` | fiche de fidélité audio à remplir (juger dans la page HTML : l'écoute est nécessaire) |
| `fiche_ocr_rempli.csv`, `fiche_audio_rempli.csv` | exports de la page HTML, à déposer dans ce dossier |
| `ocr_items.jsonl`, `audio_items.jsonl` | clés privées (fournisseurs, versions, segments) |
| `rapport_decision.md`, `metriques.json` | résultats de `metrics.py` |

**Ne pas ouvrir les fichiers `*.jsonl` avant d'avoir jugé.** Aucun chemin local absolu n'est écrit dans les fiches : seulement des noms de fichier.

## 3. Commandes

Depuis la racine du dépôt, avec le Python du venv. Chaque commande tient sur une ligne.

| Étape | Réseau | Commande |
|---|---|---|
| 1. Fiche OCR (D21) | non | `.venv/bin/python scripts/eval/albert/ocr_sheets.py --pdf-dir uploads/<session> --csv uploads/<session>/output.csv --provider albert_lightonocr --out data/albert_eval/2026-10-02` |
| 1 bis. Avec comparaison A/B | non | `.venv/bin/python scripts/eval/albert/ocr_sheets.py --pdf-dir uploads/<session> --csv uploads/<session>/output_recode.csv --compare-column texte_recode --provider albert_lightonocr --out data/albert_eval/2026-10-02` |
| 2. Fiche audio | non (ffmpeg local) | `.venv/bin/python scripts/eval/albert/audio_sheets.py --audio-dir sources/<projet>/audio --csv sources/<projet>/output.csv --out data/albert_eval/2026-10-02` |
| 3. Métriques et décision | non | `.venv/bin/python scripts/eval/albert/metrics.py --dir data/albert_eval/2026-10-02` |
| Tests (hors ligne) | non | `HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 NO_PROXY=127.0.0.1,localhost,testserver .venv/bin/python -m pytest tests/test_albert_eval_tools.py -q -p no:cacheprovider` |

Options utiles :

- `ocr_sheets.py` : `--pages-per-doc 5`, `--provider albert_lightonocr` (fournisseurs gardés, séparés par des virgules ; défaut : tous), `--dpi 110`, `--image-format png|jpeg`, `--max-html-mb 120` (au-delà, pages en gris à 72 DPI puis images omises ; avertissement dès 40 Mo), `--seed`, `--second-ratio 0.2`, `--force`.
- `audio_sheets.py` : `--segments` (relevé `<base>_audio_segments.json` écrit par `rad_audio.py` à côté du CSV, pris par défaut), `--clips-per-file 20`, `--max-clip-seconds 30`, `--provider`, `--max-html-mb 120`, `--seed`, `--second-ratio 0.2`, `--force`. Sans ffmpeg, aucun extrait ne peut être découpé : le script s'arrête (code 2) en le disant, sans écrire de fiche.
- `metrics.py` : `--ocr-sheet` / `--audio-sheet` (défaut : `fiche_*_rempli.csv`, sinon `fiche_*.csv` rempli au tableur), `--ocr-items` / `--audio-items`, `--d21-provider albert_lightonocr`, `--out`.

Une fiche déjà remplie n'est jamais écrasée sans `--force` (une copie horodatée est alors gardée dans `sauvegardes/`). Si la nouvelle fiche diffère de l'ancienne (autres pages ou extraits, autre graine), l'export rempli de l'ancienne est rangé dans `sauvegardes/` : `metrics.py` ne le confondra pas avec la nouvelle. Chaque ligne relue est de toute façon confrontée à la clé (document, page ou début de l'extrait).

## 4. Juger

### Échelles

Fidélité d'une transcription OCR à l'image de la page :

- **0 illisible/faux** : texte absent, inutilisable ou sans rapport avec la page ;
- **1 nombreuses erreurs** : le sens est atteint (mots, chiffres ou lignes manquants ou faux) ;
- **2 erreurs mineures** : quelques coquilles, le sens est intact ;
- **3 fidèle** : transcription exacte (la mise en forme ne compte pas).

Comparaison A/B : **A**, **B** ou **égal** (la plus fidèle et la plus lisible).

Fidélité d'une transcription audio à l'extrait écouté :

- **0 inaudible/faux** : extrait inaudible, ou transcription absente, fausse ou sans rapport ;
- **1 nombreuses erreurs** : le sens est atteint (mots manquants, faux ou inventés) ;
- **2 erreurs mineures** : quelques mots ou noms propres erronés, le sens est intact ;
- **3 fidèle** : transcription exacte (ponctuation et hésitations ne comptent pas).

### Avec la page HTML (recommandé)

- Ouvrir `fiche_ocr.html` ou `fiche_audio.html` dans un navigateur : rien n'est chargé depuis le réseau.
- Clavier : `0` à `3` notent et passent à l'élément suivant ; `a`, `b`, `e` pour la comparaison A/B ; `p` écoute l'extrait ou le met en pause, `r` le réécoute depuis le début (fiche audio) ; `←` `→` ou `j` `k` pour naviguer ; `n` pour le prochain élément non jugé ; `c` pour écrire un commentaire, `Échap` pour en sortir. La liste à gauche donne accès à chaque élément et montre la note saisie.
- Les jugements sont gardés dans ce navigateur, sous une clé propre à la fiche et à la graine. Vider les données du site les efface : exporter régulièrement.
- « Exporter le CSV » produit `fiche_ocr_rempli.csv` (ou `fiche_audio_rempli.csv`), en général dans le dossier Téléchargements : **le déposer dans le dossier de campagne**.
- « Importer un CSV » recharge un export pour reprendre ailleurs. « Réinitialiser » efface les jugements du navigateur après une confirmation dans la page.

### Avec un tableur

Ouvrir le CSV (UTF-8 avec BOM, « ; »), remplir `note` (0 à 3), `preference` (A, B ou égal, fiche OCR) et au besoin `commentaire`. Ne modifier ni `item_id` ni les textes. Enregistrer au format **CSV UTF-8**. Le tableur ne montre ni l'image des pages ni le son : les fiches OCR et audio se jugent dans la page HTML.

### Règles

- En cas d'hésitation, prendre la note la plus basse et le dire en commentaire.
- Fidélité : comparer texte, ordre de lecture, chiffres, noms propres et notes ; la mise en forme ne compte pas. Une page en échec affiche « OCR ÉCHOUÉ » : 0 si la page porte du texte, 3 si elle est réellement vide. Un ajout absent de la page est une erreur.
- Audio : juger les mots ; ponctuation, majuscules et hésitations ne comptent pas ; un mot inventé ou un nom propre déformé est une erreur ; ignorer les mots coupés aux bords de l'extrait (marge de 0,25 s).
- Second jugement (`r-…`) : à un autre moment, sans relire le premier.

## 5. Durée estimée

Le script affiche le nombre d'éléments et la durée estimée à la fin de la génération.

- **OCR : environ 30 s par transcription, 45 s par comparaison A/B.** Exemple sans A/B : 10 documents × 5 pages = 50 pages, plus 10 en second jugement, soit 60 × 30 s ≈ 30 min. Avec A/B : 100 transcriptions + 20 en second jugement à 30 s, plus 50 comparaisons à 45 s ≈ 1 h 38.
- **Audio : environ 1,5 × la durée de l'extrait + 10 s par élément** (écoute, réécoute, note). Exemple : 5 enregistrements × 20 extraits de 6 s en moyenne, plus 20 en second jugement, soit 120 × 19 s ≈ 38 min. Le script affiche l'estimation calculée sur les durées réelles.

## 6. Mesures (`metrics.py`)

- **Accord** : kappa de Cohen pondéré quadratique entre premier et second jugement, poids d'accord w = 1 − (i − j)² / 9, κ = (Po − Pe) / (1 − Pe) ; aussi kappa linéaire, accord brut et liste des désaccords de 2 degrés ou plus.
- **OCR** : fidélité moyenne et part des pages notées ≤ 1, par fournisseur, version (brute, comparée) et statut (réussie, en échec) ; préférences A/B rapportées à la version brute ou comparée.
- **Audio** : fidélité moyenne et part des extraits notés ≤ 1, en tout et par enregistrement ; nombre de segments en échec (non jugés).

## 7. Critères de décision (fixés le 2026-10-02, avant tout jugement)

### OCR, D21 (pages `albert_lightonocr` réussies, transcription brute)

Ne plus recoder les textes LightOnOCR (`ALBERT_OCR_SKIP_RECODE=1`) seulement si les trois critères sont tenus :

1. fidélité moyenne ≥ 2,5 ;
2. au plus 5 % des pages notées ≤ 1 ;
3. si une comparaison A/B existe, la version comparée (recodée) est préférée dans moins de 60 % des paires tranchées (A ou B, hors « égal »).

Sinon le recodage est gardé (défaut actuel). La fiche doit être entièrement jugée (sinon « jugements incomplets : pas de décision D21 »). Les pages en échec sont jugées et comptées à part : le recodage ne les corrige pas, et leur taux relève du maillon OCR, pas de D21. Le kappa OCR est donné à titre d'information (sous 0,6, relire les désaccords avant d'appliquer la décision).

### Transcription audio (extraits du premier jugement)

La transcription Whisper est jugée **utilisable sans relecture humaine** seulement si la fidélité moyenne des extraits est ≥ 2,5 **et** si au plus 5 % des extraits sont notés ≤ 1 ; sinon « relecture humaine nécessaire avant usage de la transcription ». La fiche doit être entièrement jugée (sinon « jugements incomplets : pas de décision »). Les segments en échec n'ont pas d'énoncé : ils ne sont pas jugés, leur nombre est donné à part. Le kappa est donné à titre d'information.

## 8. Formats d'entrée

**Session OCR** : `output.csv` de `rad_dataframe.py` avec `filename` (ou `path`, dont seuls les deux derniers composants servent), `texteocr` (marqueurs `<!-- Page N -->`) et `texteocr_provider`. Les PDF sont cherchés dans `--pdf-dir` par nom (aussi sous `<CLÉ>/` et `files/<CLÉ>/`). Les marqueurs `<!-- OCR ÉCHOUÉ … -->` et les parts en échec (`<!-- Part i/N (pages A-B) — OCR ÉCHOUÉ … -->`) donnent des pages en échec, présentées comme les autres. Pour l'A/B, `--compare-column` désigne une seconde transcription **avec les mêmes marqueurs de page** (par exemple le texte recodé page par page) ; elle est présumée être la version recodée dans le rapport.

**Session audio** : `output.csv` de `rad_audio.py` (`filename`, `itemKey`, `texteocr` avec ses marqueurs `<!-- Segment N (hh:mm:ss–hh:mm:ss) -->`, `texteocr_provider`) et le relevé `<base>_audio_segments.json` (segments et énoncés aux horodatages absolus de l'enregistrement). Les énoncés tirés sont ceux, non vides, des segments réussis, de 30 s au plus ; chaque extrait est découpé par ffmpeg (mono, 16 kHz, mp3 32 kb/s, marge de 0,25 s) et intégré en base64 dans la page. Les enregistrements sont cherchés dans `--audio-dir` par nom de fichier.

## 9. Limites

- **Un juge principal.** Le second jugement porte sur 20 % des éléments ; fait par la même personne, il mesure sa constance. Un kappa faible appelle à relire l'échelle et les désaccords.
- **Comparaison A/B.** Le recodage de RAGpy se fait par chunk (`rad_chunk.py`, phase `initial`) : la colonne comparée doit être produite page par page pour garder les marqueurs. Sans elle, D21 se décide sur la fidélité seule.
- **Images.** Rendu à 110 DPI : une page très dense peut demander l'agrandissement (clic sur l'image). Une fiche de plus de 40 Mo peut ralentir le navigateur.
- **Audio.** Seuls les énoncés de 30 s au plus sont tirés (les plus longs sont comptés et écartés) ; l'attribution des locuteurs n'est pas jugée.
- **Portée.** Un résultat sur échantillon ne garantit pas l'absence d'erreur sur tout le corpus ; une modification du pipeline (modèle, prompt, découpage) demande une nouvelle campagne.
