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

## Tests de Régression

### Test 1 : Health Check Performance
```bash
# 1000 requêtes, 50 concurrentes
hey -n 1000 -c 50 http://localhost:8000/health/detailed

# Baseline attendu:
# - Requests/sec: > 100
# - Latence P95: < 500ms
# - Erreurs: 0%
```

### Test 2 : Load Test Locust (10 users, 5 min)
```bash
locust -f tests/load/locust_baseline.py \
       --host=http://localhost:8000 \
       --users 10 \
       --spawn-rate 2 \
       --run-time 5m \
       --headless \
       --csv=baseline_phase1
```

| Métrique | Target |
|----------|--------|
| Requests total | > 1000 |
| Failures | 0% |
| Median response time | < 300ms |
| 95th percentile | < 1000ms |
| Requests/sec | > 3 |

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

## Objectifs Phase 2

Si Phase 2 implémentée, les améliorations attendues :

| Métrique | Phase 1 | Target Phase 2 | Gain |
|----------|---------|----------------|------|
| Extraction OCR | séquentiel | parallèle | 2-3x |
| Concurrent users | 10-20 | 40-60 | 2-3x |
| Embedding throughput | baseline | +50% | 1.5x |

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
