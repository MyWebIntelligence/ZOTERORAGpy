# Performance Baseline RAGpy - Phase 1

**Date** : 2025-11-25
**Version** : 1.0.0

## Configuration Actuelle

| Paramètre | Valeur |
|-----------|--------|
| Uvicorn workers | 7 |
| Docker RAM limit | 8GB |
| Docker CPU limit | 4 cores |
| DEFAULT_MAX_WORKERS | 8 |
| DEFAULT_DOC_WORKERS | 6 |
| MAX_CONCURRENT_LLM_CALLS | 6 |
| EMBEDDING_BATCH_SIZE | 32 |

## Métriques Système (Idle)

Données collectées via `/health/detailed` :

| Métrique | Valeur |
|----------|--------|
| CPU % | ~0% |
| Memory % | 23.2% |
| Memory disponible | 6.31 GB |
| Disk libre | 31.31 GB |
| Disk % utilisé | 47.3% |

## Capacités Théoriques

### Requêtes HTTP
- **Workers Uvicorn** : 7 processus
- **Concurrency limit** : 100 req/worker = 700 req simultanées max
- **Backlog TCP** : 2048 connexions en queue

### Traitement LLM
- **Appels LLM simultanés** : 6 (sémaphore global)
- **Retry automatique** : 1 retry, 2s délai

### Pipeline Processing
- **Documents parallèles** : 6 (DEFAULT_DOC_WORKERS)
- **Embeddings batch** : 32 chunks/batch

## Résultats Baseline (2025-11-25)

### Test Locust : 20 utilisateurs, 3 minutes

| Métrique | Résultat | Target |
|----------|----------|--------|
| Requests total | **1754** | > 1000 ✅ |
| Failures | **0%** | 0% ✅ |
| Requests/sec | **9.76** | > 3 ✅ |
| Latence moyenne | **38ms** | < 300ms ✅ |
| Latence P50 | **5ms** | - |
| Latence P95 | **120ms** | < 1000ms ✅ |
| Latence P99 | **160ms** | - |

### Détail par endpoint

| Endpoint | Reqs | Avg | P95 | P99 |
|----------|------|-----|-----|-----|
| `/health` | 920 | 7ms | 17ms | 98ms |
| `/health/detailed` | 497 | 116ms | 140ms | 240ms |
| `/` | 260 | 7ms | 19ms | 82ms |
| `/get_credentials` | 77 | 9ms | 16ms | 71ms |

## Tests de Régression

### Commande de test baseline
```bash
docker exec ragpy locust -f /app/locust_clean.py \
    --host=http://localhost:8000 \
    --users 20 \
    --spawn-rate 5 \
    --run-time 3m \
    --headless \
    --only-summary
```

### Test 3 : SSE Streaming
```bash
# Vérifier que les SSE fonctionnent sous charge
curl -N http://localhost:8000/api/processing/session/{id}/stream
# Devrait recevoir des events sans timeout
```

## Comparaison Phase 1 vs Pre-Phase 1

| Métrique | Avant | Après Phase 1 | Amélioration |
|----------|-------|---------------|--------------|
| Workers | 1 | 7 | 7x |
| Concurrent LLM | illimité | 6 (contrôlé) | Stabilité |
| RAM limit | non défini | 8GB | Prévisibilité |
| Healthcheck | basique | détaillé | Observabilité |

## Phase 2 - T1 : ThreadPoolExecutor OCR (Implémenté 2025-11-25)

### Configuration ajoutée

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| PDF_EXTRACTION_WORKERS | 4 | Workers parallèles pour OCR |
| MISTRAL_CONCURRENT_CALLS | 3 | Limite API Mistral simultanés |

### Fonctionnalités implémentées

1. **ThreadPoolExecutor** pour traitement parallèle des PDFs
   - Mode parallèle activé si `PDF_EXTRACTION_WORKERS > 1`
   - Mode séquentiel conservé pour compatibilité (workers=1)

2. **Rate limiting Mistral API**
   - Semaphore limitant les appels API concurrents
   - Protection contre rate limits

3. **Thread-safe operations**
   - Locks pour écriture CSV (`_CSV_LOCK`)
   - Locks pour sauvegarde progress (`_PROGRESS_LOCK`)
   - Locks additionnels pour compteurs et erreurs

4. **Fonction `_process_single_zotero_item`**
   - Traitement thread-safe d'un item
   - Retourne `ItemProcessingResult` avec records et erreurs

### Gains attendus

| Métrique | Phase 1 | Phase 2-T1 | Gain |
|----------|---------|------------|------|
| Extraction OCR | séquentiel | parallèle (4 workers) | 2-3x |
| Throughput docs/min | ~5-10 | ~15-20 | 2-3x |
| Rate limit hits | N/A | contrôlé | Stabilité |

### Validation requise

```bash
# Test avec petit corpus (10 PDFs)
time python scripts/rad_dataframe.py \
  --json tests/fixtures/10_docs.json \
  --dir tests/fixtures \
  --output /tmp/test_parallel.csv

# Comparer avec baseline séquentiel
# Vérifier logs pour workers actifs
grep "PARALLEL mode" logs/pdf_processing.log
```

## Objectifs Phase 2 restants

| Tâche | Status | Description |
|-------|--------|-------------|
| P2-T1 | ✅ Fait | ThreadPoolExecutor extraction PDF |
| P2-T2 | En attente | Optimisation batching OpenAI |
| P2-T3 | En attente | Cleanup automatique sessions |
| P2-T4 | En attente | Monitoring Prometheus |
| P2-T5 | En attente | Index database sessions |

## Notes Opérationnelles

- OpenAI rate limit : Géré par batching + retry
- Mistral OCR timeout : Surveillé via logs
- Database locks : SQLite WAL mode activé
- OOM events : Protégé par Docker limits

## Commandes de Monitoring

```bash
# Status container
docker stats ragpy --no-stream

# Health détaillé
curl -s http://localhost:8000/health/detailed | jq .

# Logs temps réel
docker compose logs -f ragpy

# Sessions actives
curl -s http://localhost:8000/health/detailed | jq '.database.active_sessions'
```

---

**Prochaine mise à jour** : Après exécution des tests Locust baseline
