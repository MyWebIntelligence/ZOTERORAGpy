# Mise à jour de l'interface web - Support CSV

**Date** : 2025-10-21
**Version** : 1.1
**Statut** : ✅ Terminé

---

## Vue d'ensemble

L'interface web RAGpy (FastAPI) a été mise à jour pour supporter **deux modes d'ingestion** :
- **Option A** : ZIP Archive (Zotero + PDFs) - flux existant
- **Option B** : CSV File (texte direct) - **NOUVEAU**

---

## Changements apportés

### 1. Backend - Nouvelle route `/upload_csv`

**Fichier** : [app/main.py](app/main.py:143-215)

```python
@app.post("/upload_csv")
async def upload_csv_endpoint(file: UploadFile = File(...)):
    """
    Upload CSV file for direct ingestion (bypass PDF/OCR).
    Converts CSV to DataFrame using ingestion.csv_ingestion module,
    then saves it as output.csv for the pipeline.
    """
```

**Fonctionnement** :
1. Reçoit un fichier `.csv` uploadé
2. Valide l'extension (refuse les non-CSV)
3. Crée un répertoire unique dans `uploads/`
4. Appelle `ingest_csv_to_dataframe()` du module `ingestion`
5. Sauvegarde le DataFrame comme `output.csv` (compatible avec le pipeline)
6. Retourne le chemin de session pour les étapes suivantes

**Réponse** :
```json
{
  "path": "a1b2c3d4_my_documents",
  "tree": ["output.csv"],
  "message": "CSV ingested successfully: 42 rows processed."
}
```

---

### 2. Frontend - Interface double choix

**Fichier** : [app/templates/index.html](app/templates/index.html:131-163)

#### Avant (v1.0)

```html
<section id="upload-section">
  <h2>1. Upload ZIP Archive</h2>
  <form id="uploadForm" enctype="multipart/form-data">
    <input type="file" name="file" accept=".zip" required />
    <button type="submit">Upload</button>
  </form>
</section>
```

**Problème** :
- Pas d'explication claire
- Accepte seulement ZIP
- CSV rejeté avec erreur

#### Après (v1.1)

```html
<section id="upload-section">
  <h2>1. Upload Documents</h2>
  <p>Choose your data source:</p>

  <!-- Option A: ZIP (Zotero + PDFs) -->
  <div style="border: 2px solid #1976d2; ...">
    <h3>📦 Option A: ZIP Archive (Zotero + PDFs)</h3>
    <p>For Zotero exports... Requires OCR processing.</p>
    <form id="uploadZipForm">...</form>
  </div>

  <!-- Option B: CSV Direct -->
  <div style="border: 2px solid #2e7d32; ...">
    <h3>📄 Option B: CSV File (Direct Text)</h3>
    <p>For CSV files... Skips OCR - faster and cheaper.</p>
    <form id="uploadCsvForm">...</form>
  </div>
</section>
```

**Améliorations** :
- ✅ **2 options visuellement distinctes** (bleu pour ZIP, vert pour CSV)
- ✅ **Explications claires** pour chaque option
- ✅ **Extensions acceptées** affichées
- ✅ **Avantages mis en avant** (skip OCR, faster, cheaper)

---

### 3. JavaScript - Double handler d'upload

**Fichier** : [app/templates/index.html](app/templates/index.html:421-474)

#### Handler ZIP (Option A)

```javascript
document.getElementById('uploadZipForm').addEventListener('submit', async (e) => {
  // ... upload vers /upload_zip
  if (res.ok) {
    currentPath = json.path;
    uploadResult.innerHTML = `✅ ZIP uploaded: ${currentPath}
                               Next: Extract Text & Metadata (Step 2)`;
    document.getElementById('dataframe-section').style.display = 'block';
  }
});
```

**Flux** : ZIP → Step 2 (Extract Text) → Step 3 (Chunking) → ...

#### Handler CSV (Option B)

```javascript
document.getElementById('uploadCsvForm').addEventListener('submit', async (e) => {
  // ... upload vers /upload_csv
  if (res.ok) {
    currentPath = json.path;
    uploadResult.innerHTML = `✅ CSV ingested: ${currentPath}
                               ${json.message}
                               CSV processed! Next: Text Chunking (Step 3.1)`;
    // SKIP Step 2 (dataframe extraction) - already done in backend
    document.getElementById('dataframe-section').style.display = 'none';
    document.getElementById('initial-chunk-section').style.display = 'block';
  }
});
```

**Flux** : CSV → **DIRECT** Step 3.1 (Chunking) → Step 3.2 → ...

**Différence clé** : CSV saute l'étape 2 (extraction OCR) car le texte est déjà disponible.

---

## Flux utilisateur comparé

### Option A : ZIP (Zotero + PDFs)

```
┌────────────────────────────────────────┐
│ 1. Upload ZIP Archive                  │
│    - Zotero JSON + PDF files           │
└────────────────────────────────────────┘
                 ↓
┌────────────────────────────────────────┐
│ 2. Extract Text & Metadata (OCR)       │
│    - rad_dataframe.py                  │
│    - Mistral/OpenAI OCR                │
│    - Output: output.csv                │
└────────────────────────────────────────┘
                 ↓
┌────────────────────────────────────────┐
│ 3.1 Initial Text Chunking              │
│     - rad_chunk.py --phase initial     │
│     - GPT recodage (si non-Mistral)    │
└────────────────────────────────────────┘
                 ↓
┌────────────────────────────────────────┐
│ 3.2 Dense Embeddings                   │
│     - OpenAI embeddings                │
└────────────────────────────────────────┘
                 ↓
┌────────────────────────────────────────┐
│ 3.3 Sparse Embeddings                  │
│     - spaCy TF lemmatisé               │
└────────────────────────────────────────┘
                 ↓
┌────────────────────────────────────────┐
│ 4. Upload to Vector DB                 │
│    - Pinecone / Weaviate / Qdrant      │
└────────────────────────────────────────┘
```

**Temps estimé** : 10-30 min (selon nombre de PDFs)
**Coût API** : OCR + GPT recodage + embeddings

---

### Option B : CSV (Direct)

```
┌────────────────────────────────────────┐
│ 1. Upload CSV File                     │
│    - CSV with text column              │
│    - Backend: ingestion.csv_ingestion  │
│    - Output: output.csv (auto)         │
└────────────────────────────────────────┘
                 ↓
                 ⏭️ SKIP Step 2 (pas d'OCR)
                 ↓
┌────────────────────────────────────────┐
│ 3.1 Initial Text Chunking              │
│     - rad_chunk.py --phase initial     │
│     - ❌ PAS de GPT recodage (CSV)     │
└────────────────────────────────────────┘
                 ↓
┌────────────────────────────────────────┐
│ 3.2 Dense Embeddings                   │
│     - OpenAI embeddings                │
└────────────────────────────────────────┘
                 ↓
┌────────────────────────────────────────┐
│ 3.3 Sparse Embeddings                  │
│     - spaCy TF lemmatisé               │
└────────────────────────────────────────┘
                 ↓
┌────────────────────────────────────────┐
│ 4. Upload to Vector DB                 │
│    - Pinecone / Weaviate / Qdrant      │
└────────────────────────────────────────┘
```

**Temps estimé** : 3-10 min (selon nombre de lignes)
**Coût API** : **Seulement embeddings** (80% d'économie)

---

## Format CSV attendu

### Minimal (1 colonne)

```csv
text
"Mon premier document avec du texte."
"Un second document avec plus de contenu."
```

### Recommandé (avec métadonnées)

```csv
text,title,category,priority,date,author
"Introduction to NLP...","NLP Guide","Tech","High","2023-05-15","John"
"Data quality matters...","Data Quality","Science","Medium","2023-06-20","Jane"
```

**Colonnes automatiques ajoutées** :
- `source_type`: "csv"
- `texteocr_provider`: "csv" (pour skip GPT recodage)
- `ingested_at`: timestamp ISO
- `row_index`: numéro de ligne (optionnel)

**Toutes les colonnes CSV** sont conservées comme métadonnées dans Pinecone/Weaviate/Qdrant !

---

## Tests de l'interface

### Test 1 : Upload CSV basique

1. Démarrer le serveur :
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. Ouvrir http://localhost:8000

3. Dans "Option B: CSV File", sélectionner un CSV avec une colonne `text`

4. Cliquer "Upload CSV"

**Attendu** :
- Message : "✅ CSV ingested: a1b2c3d4_my_documents"
- Message : "CSV ingested successfully: 10 rows processed."
- Section "3.1 Initial Text Chunking" apparaît
- Section "2. Extract Text & Metadata" reste cachée

### Test 2 : Upload ZIP (régression)

1. Dans "Option A: ZIP Archive", sélectionner un ZIP Zotero

2. Cliquer "Upload ZIP"

**Attendu** :
- Message : "✅ ZIP uploaded: a1b2c3d4_my_library"
- Section "2. Extract Text & Metadata" apparaît
- Workflow normal (OCR → CSV → chunking)

---

## Validation avec CSV de test

### Préparation

```bash
# Créer un CSV de test
cat > /tmp/test.csv << 'EOF'
text,title,category
"First document about AI.","AI Intro","Technology"
"Second document about data.","Data Guide","Science"
EOF
```

### Scénario complet

1. **Upload CSV** via interface → Option B
2. **Chunking** : Cliquer "Generate Chunks" (Step 3.1)
3. **Dense Embeddings** : Cliquer "Generate Dense Embeddings" (Step 3.2)
4. **Sparse Embeddings** : Cliquer "Generate Sparse Embeddings" (Step 3.3)
5. **Vectorisation** : Choisir "Pinecone", Index "test-csv", cliquer "Start Upload"

### Vérification Pinecone

```python
from pinecone import Pinecone

pc = Pinecone(api_key="...")
index = pc.Index("test-csv")

# Query pour tester les métadonnées
results = index.query(
    vector=[0.1]*3072,  # Dummy vector
    top_k=1,
    include_metadata=True
)

print(results['matches'][0]['metadata'])
# Attendu: {'title': 'AI Intro', 'category': 'Technology', 'source_type': 'csv', ...}
```

---

## Messages d'erreur possibles

| Erreur | Cause | Solution |
|--------|-------|----------|
| "Only .csv files are accepted." | Extension non-.csv | Vérifier le fichier uploadé |
| "Failed to process CSV file." | CSV mal formé | Vérifier l'encodage (UTF-8 recommandé) |
| "CSV ingestion module not found." | Module `ingestion` absent | Vérifier que `core/` et `ingestion/` existent |
| "Colonne 'text' absente" | Pas de colonne `text` | Renommer la colonne principale en `text` |

---

## Configuration avancée

### Modifier la colonne texte par défaut

Actuellement hardcodé à `"text"`. Pour changer :

**Option 1 : Backend** - Modifier [ingestion/csv_ingestion.py](ingestion/csv_ingestion.py)

```python
# Ligne ~50
class CSVIngestionConfig:
    def __init__(
        self,
        text_column: str = "description",  # ← Changer ici
        ...
    ):
```

**Option 2 : Interface** - Ajouter un champ "Text Column" dans le formulaire

```html
<!-- Dans app/templates/index.html -->
<form id="uploadCsvForm" ...>
  <label for="csvTextColumn">Text Column Name:</label>
  <input type="text" id="csvTextColumn" value="text" placeholder="text" />

  <input type="file" name="file" accept=".csv" required />
  <button type="submit">Upload CSV</button>
</form>
```

Puis modifier le JavaScript pour envoyer ce paramètre au backend.

---

## Dépannage

### CSV uploadé mais Step 3.1 ne s'affiche pas

**Cause** : Erreur JavaScript ou réponse backend incorrecte

**Debug** :
1. Ouvrir la console navigateur (F12)
2. Chercher erreurs JavaScript
3. Vérifier la réponse de `/upload_csv` :
   ```javascript
   console.log(json);  // Dans le handler uploadCsvForm
   ```

### Métadonnées CSV manquantes dans Pinecone

**Cause** : Version non refactorisée de `rad_vectordb.py`

**Vérification** :
```bash
grep -n "for key, value in chunk.items()" scripts/rad_vectordb.py
```

**Attendu** : Ligne 86 (Pinecone), 546 (Weaviate), 646 (Qdrant)

### Recodage GPT activé malgré CSV

**Cause** : `texteocr_provider` non défini à "csv"

**Vérification** :
```bash
cat tests/fixtures/test_output.csv | head -1
# Doit contenir : texteocr_provider
```

---

## Prochaines améliorations

- [ ] Champ "Text Column" configurable dans l'interface
- [ ] Preview des colonnes CSV avant upload
- [ ] Support drag-and-drop pour upload
- [ ] Validation de schéma CSV côté frontend
- [ ] Estimation du coût API avant traitement
- [ ] Support fichiers Excel (.xlsx)

---

## Ressources

- **Guide utilisateur CSV** : [CSV_INGESTION_GUIDE.md](CSV_INGESTION_GUIDE.md)
- **Architecture pipeline** : [pipeline_current_architecture.md](pipeline_current_architecture.md)
- **Code backend** : [app/main.py](app/main.py:143-215)
- **Code frontend** : [app/templates/index.html](app/templates/index.html:131-474)

---

**Développé le** : 2025-10-21
**Par** : Claude + Amar Lakel
**Version RAGpy** : 2.1 (Multi-source web interface)
**Status** : ✅ Production-ready
