# Feature Clustering RAGpy - Plan de Développement

**Date de création** : 2025-12-10
**Statut** : À implémenter
**Priorité** : Feature majeure

---

## 📋 Résumé Exécutif

Ajouter une étape **4.b** au pipeline RAGpy pour :
1. Clusterer les documents basé sur leurs embeddings denses (UMAP + HDBSCAN)
2. Utiliser les embeddings sparse pour validation/affinage
3. Générer des tags Zotero au format `_NomSession_01`, `_NomSession_02`...
4. Appliquer automatiquement ces tags via l'API Zotero

### Cas d'usage principal
Permettre aux chercheurs de **trier automatiquement leurs articles en sous-collections** dans Zotero basé sur la similarité sémantique.

---

## 🏗️ Architecture

### Pipeline avec Clustering

```
Pipeline existant:
[Zotero JSON] → [rad_dataframe] → [rad_chunk] → [rad_vectordb] → FIN
                    (1)              (2-4)           (5)

Pipeline étendu:
[Zotero JSON] → [rad_dataframe] → [rad_chunk] → [rad_vectordb]
                    (1)              (2-4)           (5)
                                                       ↓
                                              [rad_clustering] (4.b)
                                                       ↓
                                              [Tags Zotero API]
```

### Flux de données

```
Input:  output_chunks_with_embeddings_sparse.json
            │
            ▼
    ┌───────────────────┐
    │ Agrégation chunks │  (mean pooling par doc_id)
    │ → Doc embeddings  │
    └───────────────────┘
            │
            ▼
    ┌───────────────────┐
    │     UMAP          │  (3072D → 50D, métrique cosine)
    │   Réduction       │
    └───────────────────┘
            │
            ▼
    ┌───────────────────┐
    │    HDBSCAN        │  (min_cluster_size=3, auto clusters)
    │   Clustering      │
    └───────────────────┘
            │
            ▼
Output: clustering_results.json
            │
            ▼
    ┌───────────────────┐
    │  Zotero API       │  (PATCH items avec tags)
    │  Tag Application  │
    └───────────────────┘
```

---

## 📁 Structure des Fichiers

### Fichiers à Créer

| Fichier | Description | Lignes estimées |
|---------|-------------|-----------------|
| `scripts/rad_clustering.py` | Script CLI principal (UMAP + HDBSCAN) | ~350 |
| `tests/test_clustering.py` | Tests unitaires et intégration | ~200 |
| `tests/fixtures/test_embeddings_cluster.json` | Fixture de test | ~50 docs simulés |

### Fichiers à Modifier

| Fichier | Modifications |
|---------|---------------|
| `scripts/requirements.txt` | Ajouter `umap-learn>=0.5.4`, `hdbscan>=0.8.33` |
| `app/routes/processing.py` | Ajouter 3 endpoints clustering |
| `app/utils/zotero_client.py` | Ajouter `add_tags_to_items()` |
| `.claude/CLAUDE.md` | Documenter la nouvelle étape 4.b |

---

## 🔧 Implémentation Détaillée

### 1. Script `scripts/rad_clustering.py`

#### Structure du fichier

```python
#!/usr/bin/env python3
"""
rad_clustering.py - Document Clustering Agent
==============================================

Clusters documents based on their embeddings and generates Zotero tags.

Usage:
    python scripts/rad_clustering.py \
        --input uploads/session/output_chunks_with_embeddings_sparse.json \
        --output uploads/session \
        --session-name MaBiblio

Output:
    clustering_results.json with cluster assignments and tags
"""

# Configuration
DEFAULT_UMAP_N_COMPONENTS = 50      # Dimensions après UMAP
DEFAULT_UMAP_N_NEIGHBORS = 15       # Voisins UMAP
DEFAULT_UMAP_MIN_DIST = 0.1         # Distance min UMAP
DEFAULT_MIN_CLUSTER_SIZE = 3        # Min docs par cluster
RANDOM_SEED = 42                    # Reproductibilité
```

#### Fonctions principales

```python
def aggregate_chunk_embeddings(
    chunks: List[Dict],
    method: str = "mean"
) -> Dict[str, np.ndarray]:
    """
    Agrège les embeddings de chunks au niveau document.

    Args:
        chunks: Liste de chunks avec 'doc_id' et 'embedding'
        method: Méthode d'agrégation ('mean', 'max', 'first')

    Returns:
        Dict mapping doc_id → embedding agrégé (3072D)

    Example:
        >>> chunks = [
        ...     {"doc_id": "doc1", "embedding": [0.1, 0.2]},
        ...     {"doc_id": "doc1", "embedding": [0.3, 0.4]},
        ... ]
        >>> result = aggregate_chunk_embeddings(chunks)
        >>> result["doc1"]  # [0.2, 0.3] (mean)
    """
```

```python
def reduce_dimensions_umap(
    embeddings: np.ndarray,
    n_components: int = 50
) -> np.ndarray:
    """
    Réduit la dimensionnalité avec UMAP.

    Args:
        embeddings: Matrice (n_docs, 3072)
        n_components: Dimensions cibles

    Returns:
        Matrice réduite (n_docs, n_components)

    Note:
        Utilise métrique 'cosine' pour embeddings textuels.
    """
```

```python
def cluster_documents_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 3
) -> np.ndarray:
    """
    Clustering HDBSCAN avec détection automatique.

    Args:
        embeddings: Embeddings réduits (n_docs, 50)
        min_cluster_size: Taille minimum de cluster

    Returns:
        Array de labels (-1 = noise/outlier)

    Note:
        Nombre de clusters calculé automatiquement.
        Contrainte ~N/10 clusters respectée via min_cluster_size.
    """
```

```python
def generate_cluster_tags(
    doc_ids: List[str],
    labels: np.ndarray,
    session_name: str
) -> Dict[str, str]:
    """
    Génère les tags Zotero.

    Args:
        doc_ids: Liste des IDs documents
        labels: Labels de cluster
        session_name: Nom de session (ex: 'MaBiblio')

    Returns:
        Dict mapping doc_id → tag
        Example: {"doc1": "_MaBiblio_01", "doc2": "_MaBiblio_noise"}

    Format:
        - Clusters: _SessionName_01, _SessionName_02, ...
        - Outliers: _SessionName_noise
    """
```

```python
def run_clustering_pipeline(
    embeddings_json_path: str,
    output_dir: str,
    session_name: str,
    min_cluster_size: Optional[int] = None
) -> Dict:
    """
    Pipeline complet de clustering.

    Args:
        embeddings_json_path: Chemin vers les embeddings
        output_dir: Répertoire de sortie
        session_name: Nom pour les tags
        min_cluster_size: Override taille min (auto si None)

    Returns:
        Dict avec statistiques et mappings

    Progress:
        Émet PROGRESS|row|current/total|message pour SSE
    """
```

#### CLI Arguments

```bash
python scripts/rad_clustering.py \
    --input <path>              # Requis: JSON embeddings
    --output <dir>              # Requis: Répertoire sortie
    --session-name <name>       # Requis: Nom pour tags
    --min-cluster-size <int>    # Optionnel: Override auto
    --aggregation <method>      # Optionnel: mean|max|first
    --umap-components <int>     # Optionnel: Dim UMAP (défaut 50)
```

#### Format de sortie `clustering_results.json`

```json
{
  "session_name": "MaBiblio",
  "n_documents": 45,
  "n_clusters": 5,
  "n_noise": 2,
  "cluster_sizes": {
    "1": 12,
    "2": 10,
    "3": 8,
    "4": 8,
    "5": 5
  },
  "min_cluster_size_used": 3,
  "aggregation_method": "mean",
  "umap_components": 50,
  "documents": [
    {
      "doc_id": "851598844422",
      "cluster_id": 1,
      "cluster_tag": "_MaBiblio_01",
      "title": "Article Title Here",
      "authors": "Smith, John",
      "itemKey": "ABC12345"
    },
    {
      "doc_id": "851598844423",
      "cluster_id": -1,
      "cluster_tag": "_MaBiblio_noise",
      "title": "Outlier Article",
      "authors": "Doe, Jane",
      "itemKey": "DEF67890"
    }
  ]
}
```

---

### 2. Endpoints API (`app/routes/processing.py`)

#### Endpoint synchrone

```python
@router.post("/cluster_documents")
async def cluster_documents(
    session_folder: str = Form(...),
    session_name: str = Form(...),
    min_cluster_size: Optional[int] = Form(None),
    aggregation: str = Form("mean"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Cluster documents based on embeddings.

    Args:
        session_folder: Chemin relatif session (uploads/)
        session_name: Nom pour tags (ex: 'MaBiblio')
        min_cluster_size: Override taille min cluster
        aggregation: Méthode agrégation chunks

    Returns:
        {
            "success": true,
            "n_documents": 45,
            "n_clusters": 5,
            "n_noise": 2,
            "cluster_sizes": {...},
            "output_path": "session/clustering_results.json"
        }

    Errors:
        400: Embeddings file not found
        500: Clustering failed
    """
```

#### Endpoint SSE

```python
@router.post("/cluster_documents_sse")
async def cluster_documents_sse(
    session_folder: str = Form(...),
    session_name: str = Form(...),
    min_cluster_size: Optional[int] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Cluster documents with SSE progress streaming.

    Events:
        - init: {"type": "init", "total": 5, "message": "Starting clustering"}
        - progress: {"type": "progress", "current": 2, "total": 5, "percent": 40}
        - complete: {"type": "complete", "n_clusters": 5, "n_noise": 2}
        - error: {"type": "error", "message": "..."}
    """
```

#### Endpoint Zotero Tags

```python
@router.post("/apply_cluster_tags_zotero")
async def apply_cluster_tags_zotero(
    session_folder: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Apply cluster tags to Zotero items.

    Requires:
        - clustering_results.json in session folder
        - Zotero credentials configured

    Returns:
        {
            "success": true,
            "tagged_count": 43,
            "failed_count": 2,
            "errors": [...]
        }

    Errors:
        400: clustering_results.json not found
        403: Zotero credentials missing
    """
```

---

### 3. Fonction Zotero (`app/utils/zotero_client.py`)

```python
def add_tags_to_items(
    library_type: str,
    library_id: str,
    tag_mapping: Dict[str, str],
    api_key: str
) -> Dict:
    """
    Add tags to multiple Zotero items.

    Args:
        library_type: "users" or "groups"
        library_id: ID de la bibliothèque
        tag_mapping: {itemKey: tag} mapping
                    Ex: {"ABC123": "_MaBiblio_01"}
        api_key: Clé API Zotero

    Returns:
        {
            "success_count": 43,
            "failed_count": 2,
            "errors": [{"item_key": "...", "error": "..."}]
        }

    Notes:
        - Vérifie si tag existe déjà (skip si oui)
        - Gère conflits de version (If-Unmodified-Since-Version)
        - Retry avec backoff sur rate limit (429)
    """
```

---

## 🧪 Plan de Tests

### Tests Unitaires (`tests/test_clustering.py`)

```python
class TestEmbeddingAggregation:
    """Tests pour l'agrégation des embeddings."""

    def test_aggregate_mean(self):
        """Agrégation moyenne de 2 chunks → 1 doc."""

    def test_aggregate_max(self):
        """Agrégation max pooling."""

    def test_aggregate_single_chunk(self):
        """Document avec un seul chunk."""

    def test_aggregate_missing_embedding(self):
        """Chunk sans embedding ignoré."""


class TestTagGeneration:
    """Tests pour la génération de tags."""

    def test_generate_tags_format(self):
        """Format _SessionName_01 respecté."""

    def test_generate_tags_noise(self):
        """Outliers reçoivent _SessionName_noise."""

    def test_generate_tags_special_chars(self):
        """Caractères spéciaux nettoyés du nom."""

    def test_generate_tags_padding(self):
        """Numéros paddés à 2 chiffres (01, 02...)."""


class TestClusteringPipeline:
    """Tests d'intégration du pipeline."""

    @pytest.fixture
    def sample_embeddings(self, tmp_path):
        """Fixture: 20 docs avec embeddings 3072D."""

    def test_full_pipeline(self, sample_embeddings):
        """Pipeline complet génère clustering_results.json."""

    def test_min_documents_error(self, tmp_path):
        """Erreur si < 3 documents."""

    def test_output_structure(self, sample_embeddings):
        """Structure JSON de sortie valide."""


class TestZoteroTagging:
    """Tests pour l'intégration Zotero (mocked)."""

    @patch("requests.patch")
    def test_add_tags_success(self, mock_patch):
        """Tags ajoutés avec succès."""

    @patch("requests.patch")
    def test_add_tags_version_conflict(self, mock_patch):
        """Retry sur conflit de version 412."""

    @patch("requests.patch")
    def test_add_tags_rate_limit(self, mock_patch):
        """Backoff sur rate limit 429."""
```

### Fixture de Test

Créer `tests/fixtures/test_embeddings_cluster.json` :
- 20 documents simulés
- 3 chunks par document
- Embeddings aléatoires 3072D
- Métadonnées réalistes (title, authors, itemKey)

---

## 📦 Dépendances

### Ajouts à `scripts/requirements.txt`

```
# Clustering (Step 4.b)
umap-learn>=0.5.4          # Réduction dimensionnelle
hdbscan>=0.8.33            # Clustering densité
```

### Notes d'installation

- **umap-learn** : Utilise numba pour accélération
- **hdbscan** : Compatible avec numba
- Temps d'installation : ~2-3 minutes (compilation numba)
- Taille totale : ~50 MB supplémentaires

---

## 📅 Roadmap d'Implémentation

### Phase 1 : Script CLI (2h)

- [ ] Créer `scripts/rad_clustering.py`
  - [ ] Fonction `aggregate_chunk_embeddings()`
  - [ ] Fonction `reduce_dimensions_umap()`
  - [ ] Fonction `cluster_documents_hdbscan()`
  - [ ] Fonction `generate_cluster_tags()`
  - [ ] Fonction `run_clustering_pipeline()`
  - [ ] CLI argument parsing
  - [ ] Progress logging (PROGRESS|row|...)

- [ ] Ajouter dépendances à `requirements.txt`

- [ ] Test manuel CLI

### Phase 2 : API Endpoints (1h)

- [ ] Ajouter à `app/routes/processing.py` :
  - [ ] `POST /cluster_documents`
  - [ ] `POST /cluster_documents_sse`

- [ ] Test endpoints via curl

### Phase 3 : Intégration Zotero (1h)

- [ ] Ajouter `add_tags_to_items()` à `zotero_client.py`
- [ ] Ajouter `POST /apply_cluster_tags_zotero`
- [ ] Test ajout de tags sur bibliothèque test

### Phase 4 : Tests & Documentation (1h)

- [ ] Créer `tests/test_clustering.py`
- [ ] Créer fixture `tests/fixtures/test_embeddings_cluster.json`
- [ ] Exécuter `pytest tests/test_clustering.py -v`
- [ ] Mettre à jour `.claude/CLAUDE.md` (section 4.b)

---

## ⚠️ Risques et Mitigations

| Risque | Impact | Mitigation |
|--------|--------|------------|
| Trop d'outliers (>30%) | Perte d'info | Documenter, ajuster `min_samples` si besoin |
| Installation UMAP lente | UX | Utiliser wheels pré-compilés, avertir user |
| Conflits version Zotero | Échecs API | Retry existant avec backoff |
| Petit corpus (<10 docs) | Clustering pauvre | Message d'erreur explicite |
| Clusters déséquilibrés | Analyse biaisée | Afficher distribution dans résultats |

---

## ✅ Critères de Succès

1. **Fonctionnel** : CLI exécutable standalone
2. **Intégré** : API endpoints avec SSE streaming
3. **Zotero** : Tags automatiquement appliqués
4. **Testé** : Tests unitaires passent à 100%
5. **Documenté** : Guide utilisateur dans CLAUDE.md

---

## 🔗 Références

- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [HDBSCAN Documentation](https://hdbscan.readthedocs.io/)
- [Zotero Web API v3](https://www.zotero.org/support/dev/web_api/v3/basics)
- [RAGpy Pipeline Architecture](.claude/pipeline_current_architecture.md)
