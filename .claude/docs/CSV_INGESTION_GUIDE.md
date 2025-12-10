# Guide d'utilisation - Ingestion CSV dans RAGpy

**Date de création** : 2025-10-21
**Version** : 1.0
**Statut** : Développement terminé, tests à finaliser

---

## Vue d'ensemble

Le pipeline RAGpy supporte désormais l'ingestion directe de fichiers CSV, permettant d'ajouter des documents textuels sans passer par l'OCR PDF. Cette fonctionnalité :

- ✅ Mappe une colonne CSV → variable pivot `texteocr`
- ✅ Conserve **toutes** les colonnes CSV comme métadonnées dynamiques
- ✅ Compatible avec le pipeline existant (chunking, embeddings, vectorisation)
- ✅ Évite les appels API GPT de recodage (économie de coûts)
- ✅ Supporte des métadonnées arbitraires (pas de liste hardcodée)

---

## Installation

### Dépendances

Installer les dépendances Python nécessaires :

```bash
pip install pandas chardet python-dotenv
```

**Note** : `chardet` est optionnel mais recommandé pour la détection automatique d'encodage.

### Structure du projet

Les nouveaux modules ajoutés :

```
ragpy/
├── core/
│   ├── __init__.py
│   └── document.py              # Classe Document unifiée
├── ingestion/
│   ├── __init__.py
│   └── csv_ingestion.py         # Module d'ingestion CSV
├── config/
│   └── csv_config.yaml          # Configuration CSV (exemples)
├── tests/
│   ├── fixtures/
│   │   └── test_documents.csv   # CSV de test
│   └── test_csv_ingestion.py    # Suite de tests
└── scripts/
    ├── rad_chunk.py             # ✅ Refactorisé (métadonnées dynamiques)
    └── rad_vectordb.py          # ✅ Refactorisé (Pinecone, Weaviate, Qdrant)
```

---

## Utilisation rapide

### Méthode 1 : Conversion CSV → DataFrame pour rad_chunk.py

```python
from ingestion import ingest_csv_to_dataframe

# Ingestion CSV → DataFrame compatible
df = ingest_csv_to_dataframe("data/my_documents.csv")

# Sauvegarder pour rad_chunk.py
df.to_csv("output/prepared.csv", index=False, encoding="utf-8-sig")

# Puis lancer le chunking/embeddings comme d'habitude
# python scripts/rad_chunk.py --input output/prepared.csv --output output/ --phase all
```

### Méthode 2 : Manipulation directe des Documents

```python
from ingestion import ingest_csv, CSVIngestionConfig

# Configuration personnalisée
config = CSVIngestionConfig(
    text_column="description",      # Nom de la colonne texte
    encoding="auto",                 # Détection automatique
    meta_columns=["title", "category", "priority"],  # Métadonnées à garder ([] = toutes)
    skip_empty=True,
    add_row_index=True
)

# Ingestion
documents = ingest_csv("data/tickets.csv", config=config)

# Chaque document est une instance de core.Document
for doc in documents:
    print(doc.get_metadata_summary())
    print(doc.meta)  # Toutes les métadonnées CSV
```

---

## Configuration

### Fichier YAML (optionnel)

Modifier [config/csv_config.yaml](../../config/csv_config.yaml) :

```yaml
csv:
  text_column: "text"          # Colonne contenant le texte
  encoding: "auto"              # "auto", "utf-8", "latin-1", etc.
  delimiter: ","                # "," ou ";" ou "\t"
  meta_columns: []              # [] = toutes colonnes (sauf text_column)
  skip_empty: true              # Ignorer lignes avec texte vide
  add_row_index: true           # Ajouter 'row_index' dans meta
```

### Paramètres de CSVIngestionConfig

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `text_column` | str | "text" | Nom de la colonne source de `texteocr` |
| `encoding` | str | "auto" | Encodage du fichier ("auto" utilise chardet) |
| `delimiter` | str | "," | Séparateur CSV |
| `meta_columns` | List[str] | [] | Colonnes à inclure dans meta ([] = toutes) |
| `skip_empty` | bool | True | Ignorer lignes avec texte vide |
| `add_row_index` | bool | True | Ajouter `row_index` dans meta |

---

## Format CSV requis

### Minimal

```csv
text
"Mon premier document avec du texte."
"Un second document avec plus de contenu."
```

### Complet (avec métadonnées)

```csv
text,title,category,date,author,tags
"Introduction to NLP...","NLP Guide","Technology","2023-05-15","John Smith","nlp,ai"
"Data quality matters...","Data Quality","Data Science","2023-06-20","Jane Doe","data,ml"
```

### Colonnes automatiques ajoutées

Quelle que soit la structure du CSV, ces champs sont ajoutés automatiquement :

| Champ | Valeur | Description |
|-------|--------|-------------|
| `source_type` | "csv" | Type de source d'ingestion |
| `texteocr_provider` | "csv" | Fournisseur du texte (évite recodage GPT) |
| `ingested_at` | ISO timestamp | Date/heure d'ingestion |
| `row_index` | int | Index de la ligne (si `add_row_index=True`) |

---

## Pipeline complet : CSV → Vectorisation

### Étape 1 : Préparer le CSV

Créer un fichier `data/my_kb.csv` :

```csv
text,title,category,priority
"This is knowledge base entry 1...","Entry 1","Support","High"
"This is knowledge base entry 2...","Entry 2","Sales","Medium"
```

### Étape 2 : Ingestion et conversion

```python
from ingestion import ingest_csv_to_dataframe

df = ingest_csv_to_dataframe("data/my_kb.csv")
df.to_csv("output/kb_prepared.csv", index=False, encoding="utf-8-sig")
```

### Étape 3 : Chunking + Embeddings

```bash
python scripts/rad_chunk.py \
  --input output/kb_prepared.csv \
  --output output/ \
  --phase all
```

Cela génère :
- `output/output_chunks.json`
- `output/output_chunks_with_embeddings.json`
- `output/output_chunks_with_embeddings_sparse.json`

### Étape 4 : Vectorisation (Pinecone)

```python
from scripts.rad_vectordb import insert_to_pinecone
import os

result = insert_to_pinecone(
    embeddings_json_file='output/output_chunks_with_embeddings_sparse.json',
    index_name='my-kb-index',
    pinecone_api_key=os.getenv('PINECONE_API_KEY')
)
print(result)
```

**Important** : Les métadonnées CSV (`title`, `category`, `priority`, etc.) seront **automatiquement injectées** dans Pinecone grâce à la refactorisation !

---

## Tests

### Lancer la suite de tests

```bash
# Installer les dépendances
pip install pandas chardet

# Exécuter les tests
python3 tests/test_csv_ingestion.py
```

### Tests inclus

1. **Test 1** : Ingestion CSV basique
2. **Test 2** : Configuration personnalisée (`meta_columns` sélectifs)
3. **Test 3** : Conversion en DataFrame compatible `rad_chunk.py`
4. **Test 4** : Validation de la classe `Document`
5. **Test 5** : Sanitization des métadonnées

---

## Différences clés : CSV vs PDF/OCR

| Aspect | PDF/OCR | CSV |
|--------|---------|-----|
| **Source de texte** | Extraction OCR (Mistral/OpenAI/PyMuPDF) | Colonne CSV directe |
| **Recodage GPT** | Oui (sauf Mistral) | ❌ Non (économie API) |
| **Métadonnées** | Zotero fixes (title, authors, date, etc.) | ✅ Dynamiques (toutes colonnes) |
| **Provider** | "mistral", "openai", "legacy" | "csv" |
| **Coût API** | OCR + recodage GPT | ✅ Seulement embeddings |

---

## Exemples de cas d'usage

### 1. Base de tickets support

```python
config = CSVIngestionConfig(
    text_column="description",
    meta_columns=["ticket_id", "title", "category", "priority", "status"]
)
docs = ingest_csv("support_tickets.csv", config=config)
```

### 2. Articles de blog

```python
config = CSVIngestionConfig(
    text_column="content",
    meta_columns=["title", "author", "publish_date", "tags", "url"]
)
docs = ingest_csv("blog_posts.csv", config=config)
```

### 3. Produits e-commerce

```python
config = CSVIngestionConfig(
    text_column="description_longue",
    meta_columns=["ref_produit", "nom", "categorie", "prix", "stock"],
    encoding="latin-1",  # Ancien export Excel
    delimiter=";"
)
docs = ingest_csv("catalogue.csv", config=config)
```

---

## Bonnes pratiques

### 1. Encodage

- Toujours préférer `encoding="auto"` pour la détection automatique
- Si échec, essayer manuellement : `"utf-8"`, `"latin-1"`, `"cp1252"`

### 2. Colonne texte

- Choisir la colonne avec le contenu le plus riche (description, contenu, texte_complet)
- Éviter les colonnes avec texte < 50 caractères (titres seuls)

### 3. Métadonnées

- Laisser `meta_columns=[]` pour ingérer toutes les colonnes (recommandé)
- Utiliser `meta_columns=[...]` seulement si vous avez beaucoup de colonnes inutiles

### 4. Validation

- Toujours tester avec un petit CSV (10 lignes) avant de traiter des milliers de lignes
- Vérifier les métadonnées dans le JSON généré : `output_chunks.json`

---

## Dépannage

### Erreur : "Colonne 'text' absente du CSV"

**Cause** : La colonne par défaut `"text"` n'existe pas dans votre CSV.

**Solution** :
```python
config = CSVIngestionConfig(text_column="votre_colonne")
docs = ingest_csv("file.csv", config=config)
```

### Erreur : "UnicodeDecodeError"

**Cause** : Encodage incorrect.

**Solution** :
```python
config = CSVIngestionConfig(encoding="latin-1")  # ou "cp1252"
docs = ingest_csv("file.csv", config=config)
```

### Avertissement : "Texte très court"

**Cause** : Certaines lignes ont moins de 10 caractères de texte.

**Solution** :
- Vérifier que vous utilisez la bonne colonne (`text_column`)
- Activer `skip_empty=True` pour ignorer ces lignes

### Métadonnées manquantes dans Pinecone

**Cause** : Ancienne version de `rad_vectordb.py` avec métadonnées hardcodées.

**Solution** : Vérifier que vous utilisez la version refactorisée (commit du 2025-10-21).

---

## Modifications du code existant

### Fichiers modifiés

1. **[scripts/rad_chunk.py](../../scripts/rad_chunk.py)**
   - Ligne 209 : Ajout de `"csv"` dans la liste des providers sans recodage
   - Lignes 251-269 : Construction **dynamique** des métadonnées (injection de `row_data`)

2. **[scripts/rad_vectordb.py](../../scripts/rad_vectordb.py)**
   - Fonction `prepare_vectors_for_pinecone()` : Métadonnées dynamiques
   - Fonction `insert_to_weaviate_hybrid()` : Properties dynamiques + normalisation dates
   - Fonction `prepare_points_for_qdrant()` : Payload dynamique

### Backward compatibility

✅ Les modifications sont **rétrocompatibles** :
- Le pipeline PDF/OCR continue de fonctionner normalement
- Les métadonnées Zotero (title, authors, date, etc.) sont toujours injectées
- Aucune API publique n'a été modifiée

---

## Prochaines étapes (post-MVP)

- [ ] Support de fichiers Excel (`.xlsx`, `.xls`)
- [ ] Support de JSON/JSONL
- [ ] Interface web pour upload CSV dans FastAPI
- [ ] Validation de schéma avancée (types, contraintes)
- [ ] Preview des données avant ingestion
- [ ] Support de CSV compressés (`.csv.gz`)
- [ ] Mapping de colonnes via UI (glisser-déposer)

---

## Ressources

- **Documentation architecture** : [pipeline_current_architecture.md](../pipeline_current_architecture.md)

- **Tests** : [tests/test_csv_ingestion.py](../../tests/test_csv_ingestion.py)
- **Configuration** : [config/csv_config.yaml](config/csv_config.yaml)

---

## Support

Pour toute question ou problème :
1. Consulter les logs : `logs/app.log`, `output/chunking.log`
2. Vérifier la structure du CSV (encodage, séparateur, colonnes)
3. Tester avec le CSV de test : `tests/fixtures/test_documents.csv`

---

**Développé le** : 2025-10-21
**Contributeur** : Claude + Amar Lakel
**Version pipeline** : RAGpy 2.0 (ingestion multi-source)
