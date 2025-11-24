# Résolution des erreurs 404 du pipeline - 24 novembre 2025

## Problème identifié

Le pipeline RAGpy générait de multiples erreurs 404 lors de l'exécution car plusieurs endpoints appelés par le frontend (`index.html`) n'existaient pas dans le backend.

## Endpoints manquants détectés

Après analyse complète du frontend, les endpoints suivants étaient manquants :

1. ❌ `/initial_text_chunking` - Génération des chunks de texte
2. ❌ `/process_dataframe_sse` - Version SSE du traitement DataFrame
3. ❌ `/dense_embedding_generation` - Génération des embeddings denses
4. ❌ `/sparse_embedding_generation` - Génération des embeddings sparses
5. ❌ `/upload_db` - Upload vers base de données vectorielle
6. ❌ `/generate_zotero_notes_sse` - Génération de notes Zotero

## Solution implémentée

Tous les endpoints manquants ont été ajoutés à `/app/routes/processing.py` :

### 1. `/initial_text_chunking` (POST)
- **Fonction** : Génère les chunks de texte initiaux depuis output.csv
- **Script appelé** : `scripts/rad_chunk.py --phase initial`
- **Paramètres** :
  - `path` : chemin de session relatif (required)
  - `model` : modèle LLM optionnel (défaut: gpt-4o-mini)
- **Timeout** : 30 minutes
- **Sortie** : `output_chunks.json`

### 2. `/process_dataframe_sse` (POST)
- **Fonction** : Version SSE du traitement DataFrame pour mises à jour en temps réel
- **Paramètres** :
  - `path` : chemin de session relatif
- **Format** : Server-Sent Events (text/event-stream)
- **Événements** : init, progress, complete, error

### 3. `/dense_embedding_generation` (POST)
- **Fonction** : Génère les embeddings denses (OpenAI)
- **Script appelé** : `scripts/rad_chunk.py --phase dense`
- **Entrée** : `output_chunks.json`
- **Sortie** : `output_chunks_with_embeddings.json`
- **Timeout** : 30 minutes

### 4. `/sparse_embedding_generation` (POST)
- **Fonction** : Génère les embeddings sparses (spaCy)
- **Script appelé** : `scripts/rad_chunk.py --phase sparse`
- **Entrée** : `output_chunks_with_embeddings.json`
- **Sortie** : `output_chunks_with_embeddings_sparse.json`
- **Timeout** : 30 minutes

### 5. `/upload_db` (POST)
- **Fonction** : Upload des embeddings vers une base vectorielle
- **Script appelé** : `scripts/rad_vectordb.py`
- **Paramètres** :
  - `path` : chemin de session
  - `db_choice` : pinecone|weaviate|qdrant
  - Paramètres spécifiques par DB (index, namespace, class, tenant, collection)
- **Entrée** : `output_chunks_with_embeddings_sparse.json`
- **Timeout** : 1 heure

### 6. `/generate_zotero_notes_sse` (POST)
- **Fonction** : Génération de notes Zotero (placeholder)
- **Paramètres** :
  - `path` : chemin de session
  - `extended_analysis` : true|false
  - `model` : modèle LLM optionnel
- **Format** : SSE
- **Statut** : Feature à implémenter

## Modifications des imports

Ajout des imports nécessaires dans `processing.py` :
```python
import asyncio
from fastapi.responses import StreamingResponse
```

## Flux du pipeline complet

1. **Upload** → `/upload_zip` ou `/upload_csv` (déjà existants)
2. **Extraction** → `/process_dataframe` ou `/process_dataframe_sse` (ajouté)
3. **Chunking** → `/initial_text_chunking` (ajouté)
4. **Dense Embeddings** → `/dense_embedding_generation` (ajouté)
5. **Sparse Embeddings** → `/sparse_embedding_generation` (ajouté)
6. **Vector DB** → `/upload_db` (ajouté)
7. **Optionnel: Zotero** → `/generate_zotero_notes_sse` (placeholder)

## Tests requis

Avant de valider en production :

1. ✅ Vérifier que tous les endpoints répondent (pas de 404)
2. ⚠️ Tester le workflow complet avec un petit dataset
3. ⚠️ Valider les timeouts (ajuster si nécessaire)
4. ⚠️ Vérifier les permissions des fichiers générés
5. ⚠️ Tester la gestion d'erreurs de chaque endpoint
6. ⚠️ Vérifier que les scripts Python appelés existent et fonctionnent

## Actions post-déploiement

1. **Redémarrer le serveur FastAPI** pour charger les nouveaux endpoints
2. Tester le pipeline end-to-end
3. Monitorer les logs pour détecter d'éventuelles erreurs
4. Adresser les 8 vulnérabilités signalées par GitHub Dependabot

## Fichiers modifiés

- `/app/routes/processing.py` : +292 lignes, -1 ligne
  - Ajout de 6 nouveaux endpoints
  - Support SSE pour progression en temps réel

## Commits

1. `f8b762e` - Fix: Add missing /initial_text_chunking endpoint
2. `8023d4e` - Add all missing pipeline endpoints to prevent 404 errors

## Notes techniques

- **SSE Implementation** : Les endpoints SSE utilisent `StreamingResponse` et le format `text/event-stream`
- **Timeouts** : Configurés selon la complexité (30 min pour embeddings, 1h pour DB)
- **Error Handling** : Tous les endpoints incluent gestion d'erreurs et logging détaillé
- **Scripts Python** : Les chemins sont construits avec `RAGPY_DIR` pour portabilité

## Prochaines étapes recommandées

1. Implémenter la génération réelle de notes Zotero
2. Améliorer le SSE de `/process_dataframe_sse` avec progression réelle
3. Ajouter des tests unitaires pour chaque endpoint
4. Documenter l'API avec Swagger/OpenAPI
5. Corriger les vulnérabilités de dépendances
