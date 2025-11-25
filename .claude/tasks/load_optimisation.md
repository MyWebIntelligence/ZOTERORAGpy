# Plan d'Optimisation Load RAGpy - Guide Équipe Dev

**Créé** : 2025-11-25
**Objectif** : Améliorer la capacité de traitement parallèle de 10-20 utilisateurs → 40-60 utilisateurs
**Gain attendu** : 2.5-3x amélioration des performances
**Durée totale** : 3-7 jours (selon scope retenu)

---

## Vue d'Ensemble des Phases

| Phase | Durée | Complexité | Gain Perf | Risque |
|-------|-------|------------|-----------|--------|
| **Phase 1 : Quick Wins** | 2-4h | ⭐ Faible | 2-3x | Faible |
| **Phase 2 : Performance** | 1-2 jours | ⭐⭐ Modéré | 1.5x additionnel | Modéré |
| **Phase 3 : Production** | 2-5 jours | ⭐⭐⭐ Élevé | 2x additionnel | Élevé |

**Recommandation** : Implémenter Phase 1 → Mesurer → Décider Phase 2/3 selon besoins réels

---

# PHASE 1 : QUICK WINS (2-4h)

> **Objectif** : Gains rapides sans modification d'architecture
> **Prérequis** : Accès Docker, connaissance Docker Compose basique
> **Validation** : Load test Locust avec 10-20 utilisateurs simultanés

## P1-T1 : Configuration Uvicorn Multi-Workers

### Objectif
Passer de 1 worker à 4 workers pour gérer les requêtes HTTP concurrentes.

**Gain attendu** : 3-4x capacité de requêtes simultanées

### Fichiers à modifier

#### 1. `Dockerfile` (ligne 49)

**Avant** :
```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Après** :
```dockerfile
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--timeout-keep-alive", "120", \
     "--limit-concurrency", "100", \
     "--backlog", "2048"]
```

**Explication paramètres** :
- `--workers 4` : 4 processus worker (1 par CPU core recommandé)
- `--timeout-keep-alive 120` : Connexions keep-alive 2min (processus longs)
- `--limit-concurrency 100` : Max 100 requêtes simultanées par worker
- `--backlog 2048` : Queue TCP socket pour pics de charge

#### 2. `.env` (nouvelles variables)

**Ajouter** :
```bash
# Uvicorn Configuration
UVICORN_WORKERS=4
UVICORN_TIMEOUT_KEEP_ALIVE=120
UVICORN_LIMIT_CONCURRENCY=100
```

#### 3. `Dockerfile` (utiliser variables env)

**Version dynamique** :
```dockerfile
CMD uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers ${UVICORN_WORKERS:-4} \
    --timeout-keep-alive ${UVICORN_TIMEOUT_KEEP_ALIVE:-120} \
    --limit-concurrency ${UVICORN_LIMIT_CONCURRENCY:-100} \
    --backlog 2048
```

### Tests de validation

```bash
# 1. Rebuild container
docker compose down
docker compose build --no-cache
docker compose up -d

# 2. Vérifier workers actifs
docker compose exec ragpy ps aux | grep uvicorn
# Devrait montrer 1 master + 4 workers

# 3. Test concurrence basique
for i in {1..20}; do
  curl -s http://localhost:8000/health &
done
wait
# Toutes les requêtes doivent répondre < 200ms

# 4. Load test simple (installer hey)
brew install hey  # macOS
hey -n 1000 -c 50 http://localhost:8000/health
# Target: >100 req/s sans erreurs
```

### Critères de succès
- ✅ 4+ processus uvicorn visibles dans `ps aux`
- ✅ 50 requêtes concurrentes sans timeout
- ✅ Latence P95 < 500ms sous charge

### Estimation temps
- **Dev** : 30 min
- **Test** : 30 min
- **Total** : 1h

### Risques et mitigations
- **Risque** : Consommation RAM x4
  - **Mitigation** : Voir P1-T2 (limites Docker)
- **Risque** : Workers partagent même DB SQLite
  - **Mitigation** : SQLite supporte lectures concurrentes (WAL mode vérifié app/core/database.py)

---

## P1-T2 : Limites Ressources Docker

### Objectif
Prévenir Out Of Memory (OOM) et garantir performances prévisibles.

**Gain attendu** : Stabilité système sous charge élevée

### Fichiers à modifier

#### `docker-compose.yml` (service ragpy, après ligne 10)

**Avant** :
```yaml
services:
  ragpy:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ragpy
    restart: unless-stopped
```

**Après** :
```yaml
services:
  ragpy:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ragpy
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
    ulimits:
      nofile:
        soft: 65536
        hard: 65536
```

**Explication** :
- **limits** : Maximum autorisé (hard cap)
- **reservations** : Minimum garanti (scheduling)
- **ulimits.nofile** : Max fichiers ouverts (uploads multiples)

### Configuration adaptative (selon machine hôte)

**Machine 8GB RAM** :
```yaml
limits:
  cpus: '2.0'
  memory: 4G
reservations:
  cpus: '1.0'
  memory: 2G
```

**Machine 16GB+ RAM** :
```yaml
limits:
  cpus: '8.0'
  memory: 12G
reservations:
  cpus: '4.0'
  memory: 6G
```

### Tests de validation

```bash
# 1. Rebuild avec nouvelles limites
docker compose down
docker compose up -d

# 2. Vérifier limites actives
docker stats ragpy --no-stream
# MEM USAGE doit être < limite définie

# 3. Test stress mémoire
docker compose exec ragpy python3 -c "
import numpy as np
# Allouer 3GB (doit réussir si limite >= 4GB)
arr = np.zeros((400000000,), dtype=np.uint8)
print(f'Allocated {arr.nbytes / 1e9:.2f} GB')
"

# 4. Monitoring continu (30s)
docker stats ragpy
# Surveiller MEM % ne dépasse pas 80% de la limite
```

### Critères de succès
- ✅ `docker inspect ragpy` montre les limites configurées
- ✅ Aucun OOM kill pendant load test
- ✅ Utilisation RAM stable < 80% de la limite

### Estimation temps
- **Dev** : 15 min
- **Test** : 15 min
- **Total** : 30 min

### Dépendances
- **Bloquant** : Aucune (peut être fait en parallèle de P1-T1)

### Risques et mitigations
- **Risque** : Limites trop basses → crashes
  - **Mitigation** : Commencer avec limites généreuses, monitorer, réduire progressivement
- **Risque** : Machine hôte insuffisante
  - **Mitigation** : Documenter requirements minimaux (8GB RAM, 4 CPU cores)

---

## P1-T3 : Variables Environnement Concurrence

### Objectif
Centraliser le contrôle du parallélisme dans `.env` pour tuning rapide.

**Gain attendu** : Ajustement dynamique sans rebuild code

### Fichiers à modifier

#### 1. `.env` (nouvelles variables)

**Ajouter** :
```bash
# ===== CONCURRENCY CONTROL =====

# Script workers (CPU-bound tasks)
DEFAULT_MAX_WORKERS=8           # ThreadPoolExecutor global
DEFAULT_DOC_WORKERS=6           # Documents en parallèle (chunking)
PDF_EXTRACTION_WORKERS=4        # PDFs en parallèle (OCR)

# API batching (rate limits)
DEFAULT_BATCH_SIZE_GPT=5        # Chunks/batch recodage GPT
DEFAULT_EMBEDDING_BATCH_SIZE=32 # Chunks/batch embeddings OpenAI

# Vector DB upsert
PINECONE_BATCH_SIZE=100
WEAVIATE_BATCH_SIZE=100
QDRANT_BATCH_SIZE=100

# Session management
MAX_ACTIVE_SESSIONS=50          # Limite sessions simultanées
SESSION_TTL_HOURS=24            # Auto-cleanup après 24h
```

#### 2. `scripts/rad_chunk.py` (lignes 111-115)

**Avant** :
```python
DEFAULT_MAX_WORKERS = os.cpu_count() - 1
DEFAULT_BATCH_SIZE_GPT = 5
DEFAULT_EMBEDDING_BATCH_SIZE = 32
```

**Après** :
```python
DEFAULT_MAX_WORKERS = int(os.getenv('DEFAULT_MAX_WORKERS', os.cpu_count() - 1))
DEFAULT_BATCH_SIZE_GPT = int(os.getenv('DEFAULT_BATCH_SIZE_GPT', 5))
DEFAULT_EMBEDDING_BATCH_SIZE = int(os.getenv('DEFAULT_EMBEDDING_BATCH_SIZE', 32))
DEFAULT_DOC_WORKERS = int(os.getenv('DEFAULT_DOC_WORKERS', 3))
```

#### 3. `scripts/rad_chunk.py` (ligne 321)

**Avant** :
```python
num_doc_workers = min(3, DEFAULT_MAX_WORKERS)
```

**Après** :
```python
num_doc_workers = min(DEFAULT_DOC_WORKERS, DEFAULT_MAX_WORKERS)
```

#### 4. `scripts/rad_vectordb.py` (lignes 25-27)

**Avant** :
```python
PINECONE_BATCH_SIZE = 100
WEAVIATE_BATCH_SIZE = 100
QDRANT_BATCH_SIZE = 100
```

**Après** :
```python
PINECONE_BATCH_SIZE = int(os.getenv('PINECONE_BATCH_SIZE', 100))
WEAVIATE_BATCH_SIZE = int(os.getenv('WEAVIATE_BATCH_SIZE', 100))
QDRANT_BATCH_SIZE = int(os.getenv('QDRANT_BATCH_SIZE', 100))
```

#### 5. `scripts/rad_dataframe.py` (nouvelle variable au début)

**Ajouter après imports** :
```python
PDF_EXTRACTION_WORKERS = int(os.getenv('PDF_EXTRACTION_WORKERS', 1))
# Note: Sera utilisé en Phase 2 (ThreadPoolExecutor)
```

### Tests de validation

```bash
# 1. Tester avec valeurs custom
export DEFAULT_MAX_WORKERS=16
export DEFAULT_DOC_WORKERS=8

# 2. Lancer chunking test
python scripts/rad_chunk.py \
  --input tests/fixtures/small_corpus.csv \
  --output /tmp/test_chunk \
  --phase initial

# 3. Vérifier logs montrent bonnes valeurs
grep "max_workers" logs/chunking.log
# Devrait montrer 16 workers

# 4. Test via Docker
docker compose down
docker compose up -d
docker compose exec ragpy env | grep WORKERS
# Doit montrer toutes les variables
```

### Critères de succès
- ✅ Variables `.env` surchargent valeurs hardcodées
- ✅ Logs scripts montrent valeurs configurées
- ✅ Changement `.env` + restart change comportement (sans rebuild)

### Estimation temps
- **Dev** : 45 min
- **Test** : 15 min
- **Total** : 1h

### Dépendances
- **Bloquant** : Aucune

### Risques et mitigations
- **Risque** : Valeurs invalides (négatifs, strings)
  - **Mitigation** : Ajouter validation avec fallback

```python
def get_env_int(key: str, default: int, min_val: int = 1) -> int:
    try:
        value = int(os.getenv(key, default))
        return max(min_val, value)
    except (ValueError, TypeError):
        logger.warning(f"Invalid {key}, using default {default}")
        return default

DEFAULT_MAX_WORKERS = get_env_int('DEFAULT_MAX_WORKERS', os.cpu_count() - 1)
```

---

## P1-T4 : Healthcheck Amélioré

### Objectif
Endpoint `/health` robuste pour détecter problèmes avant qu'ils impactent utilisateurs.

**Gain attendu** : Détection proactive de dégradations

### Fichiers à modifier

#### 1. `app/main.py` (nouveau endpoint détaillé)

**Ajouter après endpoint `/health` existant** :
```python
from datetime import datetime
import psutil
import sqlite3

@app.get("/health/detailed")
async def health_detailed():
    """Health check avec métriques système détaillées"""

    # CPU et RAM
    cpu_percent = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()

    # Database connectivity
    db_status = "unknown"
    try:
        conn = sqlite3.connect("data/ragpy.db", timeout=5)
        conn.execute("SELECT 1")
        conn.close()
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    # Disk space
    disk = psutil.disk_usage('/app')

    # Active sessions (from database)
    active_sessions = 0
    try:
        from app.models.pipeline_session import PipelineSession, SessionStatus
        from app.core.database import get_db
        db = next(get_db())
        active_sessions = db.query(PipelineSession).filter(
            PipelineSession.status == SessionStatus.PROCESSING
        ).count()
    except:
        pass

    health_data = {
        "status": "healthy" if db_status == "healthy" and cpu_percent < 90 and mem.percent < 90 else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "system": {
            "cpu_percent": cpu_percent,
            "memory_percent": mem.percent,
            "memory_available_gb": mem.available / 1e9,
            "disk_free_gb": disk.free / 1e9,
            "disk_percent": disk.percent
        },
        "database": {
            "status": db_status,
            "active_sessions": active_sessions
        },
        "workers": {
            "uvicorn_workers": int(os.getenv('UVICORN_WORKERS', 1)),
            "max_workers_configured": int(os.getenv('DEFAULT_MAX_WORKERS', os.cpu_count() - 1))
        }
    }

    status_code = 200 if health_data["status"] == "healthy" else 503
    return JSONResponse(content=health_data, status_code=status_code)
```

#### 2. `scripts/requirements.txt` (ajouter dépendance)

**Ajouter** :
```
psutil==5.9.8
```

#### 3. `docker-compose.yml` (améliorer healthcheck)

**Avant** :
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 10s
```

**Après** :
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health/detailed"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 10s
```

### Tests de validation

```bash
# 1. Installer psutil
docker compose exec ragpy pip install psutil==5.9.8

# 2. Tester endpoint
curl http://localhost:8000/health/detailed | jq .

# Exemple réponse attendue:
# {
#   "status": "healthy",
#   "timestamp": "2025-11-25T10:30:00",
#   "system": {
#     "cpu_percent": 15.2,
#     "memory_percent": 42.1,
#     "memory_available_gb": 4.8,
#     "disk_free_gb": 50.3,
#     "disk_percent": 35.2
#   },
#   "database": {
#     "status": "healthy",
#     "active_sessions": 3
#   },
#   "workers": {
#     "uvicorn_workers": 4,
#     "max_workers_configured": 8
#   }
# }

# 3. Test healthcheck Docker
docker compose ps
# STATUS doit montrer "healthy"

# 4. Simuler dégradation (optionnel)
# Lancer stress test → CPU > 90% → status passe "degraded"
```

### Critères de succès
- ✅ `/health/detailed` retourne JSON complet
- ✅ Status code 503 si système dégradé
- ✅ Docker healthcheck passe (vert dans `docker compose ps`)

### Estimation temps
- **Dev** : 30 min
- **Test** : 15 min
- **Total** : 45 min

### Dépendances
- **Recommandé après** : P1-T1, P1-T2 (pour voir impact dans métriques)

---

## P1-T5 : Documentation et Baseline Metrics

### Objectif
Documenter la config actuelle et établir baseline de performance pour comparer après Phase 2/3.

**Gain attendu** : Mesure objective des améliorations

### Fichiers à créer

#### 1. `.claude/docs/performance_baseline.md`

```markdown
# Performance Baseline RAGpy - Phase 1

**Date** : 2025-11-25
**Configuration** :
- Uvicorn workers: 4
- Docker RAM limit: 8GB
- Docker CPU limit: 4 cores
- DEFAULT_MAX_WORKERS: 8

## Metrics Collectées

### Load Test Locust (10 utilisateurs, 5 min)

| Métrique | Valeur |
|----------|--------|
| Requests total | 1,234 |
| Failures | 0 (0%) |
| Median response time | 245 ms |
| 95th percentile | 890 ms |
| Requests/sec | 4.11 |
| Max concurrent users | 10 |

### Processing Pipeline (100 documents)

| Phase | Temps Total | Docs/Min | Notes |
|-------|-------------|----------|-------|
| Extraction OCR | 18 min | 5.5 | Mistral API |
| Chunking initial | 4 min | 25 | Recodage GPT skippé |
| Embeddings dense | 2.5 min | 40 | OpenAI batching |
| Embeddings sparse | 1 min | 100 | spaCy local |
| Upsert Pinecone | 30 sec | 200 | Batching 100 |

### Resource Utilization (moyenne sous charge)

| Ressource | Utilisation |
|-----------|-------------|
| CPU | 45% |
| RAM | 3.2 GB / 8 GB (40%) |
| Disk I/O | 15 MB/s |
| Network | 8 Mbps |

## Tests de Régression

### Test 1 : Upload + Process 10 PDFs
```bash
time curl -X POST http://localhost:8000/api/pipeline/upload_zip \
  -F "file=@tests/fixtures/10_docs.zip"
# Baseline: 15.2 seconds
```

### Test 2 : Concurrent Sessions
```bash
for i in {1..5}; do
  curl -X POST http://localhost:8000/api/pipeline/projects/1/sessions &
done
wait
# Baseline: Toutes sessions créées < 2 seconds
```

### Test 3 : SSE Processing
```bash
time python tests/load/test_sse_chunking.py
# Baseline: 10 chunks processed in 8.5 seconds
```

## Notes

- OpenAI rate limit hit: 0 fois
- Mistral OCR timeout: 0 fois
- Database lock errors: 0
- OOM events: 0

## Prochaines Étapes

Phase 2 devrait améliorer:
- [ ] Extraction OCR: 5.5 → 15 docs/min (2.7x)
- [ ] Chunking: 25 → 60 docs/min (2.4x)
- [ ] Concurrent users: 10 → 25 (2.5x)
```

#### 2. `tests/load/locust_baseline.py` (script load testing)

```python
"""
Locust load testing script - Baseline Phase 1
Usage: locust -f tests/load/locust_baseline.py --host=http://localhost:8000
"""

from locust import HttpUser, task, between
import random

class RAGpyUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        """Login et setup session"""
        # Créer un projet test
        response = self.client.post("/api/pipeline/projects/", json={
            "name": f"Load Test {random.randint(1000, 9999)}",
            "description": "Automated load test"
        })
        self.project_id = response.json()["id"]

    @task(5)
    def health_check(self):
        """Test le plus fréquent - health monitoring"""
        self.client.get("/health/detailed")

    @task(2)
    def list_sessions(self):
        """Lister les sessions existantes"""
        self.client.get(f"/api/pipeline/projects/{self.project_id}/sessions")

    @task(1)
    def create_session(self):
        """Créer une nouvelle session (moins fréquent)"""
        self.client.post(
            f"/api/pipeline/projects/{self.project_id}/sessions",
            json={"name": f"Session {random.randint(1, 999)}"}
        )

# Commandes de test:
# 1. Baseline (10 users, 5 min)
#    locust -f tests/load/locust_baseline.py --host=http://localhost:8000 \
#           --users 10 --spawn-rate 2 --run-time 5m --headless

# 2. Stress test (50 users, 10 min)
#    locust -f tests/load/locust_baseline.py --host=http://localhost:8000 \
#           --users 50 --spawn-rate 5 --run-time 10m --headless

# 3. Peak test (100 users, spike)
#    locust -f tests/load/locust_baseline.py --host=http://localhost:8000 \
#           --users 100 --spawn-rate 10 --run-time 2m --headless
```

### Tests de validation

```bash
# 1. Installer Locust
pip install locust

# 2. Créer dossier tests
mkdir -p tests/load tests/fixtures

# 3. Lancer baseline test
locust -f tests/load/locust_baseline.py \
       --host=http://localhost:8000 \
       --users 10 \
       --spawn-rate 2 \
       --run-time 5m \
       --headless \
       --csv=baseline_phase1

# 4. Analyser résultats
cat baseline_phase1_stats.csv
# Sauvegarder dans .claude/docs/performance_baseline.md

# 5. Test pipeline complet
time python scripts/rad_dataframe.py \
  --json tests/fixtures/small_zotero.json \
  --dir tests/fixtures \
  --output /tmp/baseline_test.csv
# Noter le temps dans baseline.md
```

### Critères de succès
- ✅ Baseline metrics documentées dans `.claude/docs/performance_baseline.md`
- ✅ Script Locust fonctionnel
- ✅ Au moins 1 test pipeline end-to-end chronométré
- ✅ Aucun échec pendant baseline collection

### Estimation temps
- **Dev** : 1h
- **Test + doc** : 1h
- **Total** : 2h

### Dépendances
- **Requis** : P1-T1, P1-T2, P1-T4 (pour metrics complètes)

---

# PHASE 2 : OPTIMISATIONS PERFORMANCE (1-2 jours)

> **Objectif** : Optimiser goulots code sans changer architecture
> **Prérequis** : Phase 1 complétée et baseline établie
> **Validation** : Comparaison metrics Phase 1 vs Phase 2

## P2-T1 : ThreadPoolExecutor Extraction PDF ✅ COMPLÉTÉ (2025-11-25)

### Objectif
Paralléliser l'extraction OCR des PDFs pour réduire le goulot d'étranglement principal.

**Gain attendu** : 2-3x vitesse extraction (5-10 → 15-20 docs/min)

### Implémentation réalisée

**Fichiers modifiés** :

- `scripts/rad_dataframe.py` : ThreadPoolExecutor + rate limiting + thread-safe operations
- `.env` : Ajout `MISTRAL_CONCURRENT_CALLS=3`

**Fonctionnalités ajoutées** :

1. Classe `ItemProcessingResult` pour retour thread-safe
2. Fonction `_process_single_zotero_item` pour traitement parallèle
3. Semaphore `MISTRAL_SEMAPHORE` pour rate limiting API
4. Locks thread-safe : `_CSV_LOCK`, `_PROGRESS_LOCK`
5. Mode parallèle/séquentiel automatique selon `PDF_EXTRACTION_WORKERS`

---

### Spécification originale (référence)

### Fichiers à modifier

#### `scripts/rad_dataframe.py` (refactor fonction process_items)

**Avant** (ligne 350-380, traitement séquentiel) :
```python
for idx, item in enumerate(tqdm(items, desc="Processing Zotero items")):
    # Extraction séquentielle
    pdf_path = resolve_pdf_path(item)
    text = extract_text_with_ocr(pdf_path)
    documents.append(text)
```

**Après** (utiliser ThreadPoolExecutor) :
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_single_item(item_data):
    """Process un item Zotero (thread-safe)"""
    idx, item = item_data
    try:
        pdf_path = resolve_pdf_path(item, base_dir)
        if not pdf_path:
            logger.warning(f"[{idx}] PDF not found for {item.get('title', 'Unknown')}")
            return None

        logger.info(f"[{idx}] Processing: {pdf_path}")
        text = extract_text_with_ocr(pdf_path)

        # Construire document
        doc = Document(
            title=item.get('title', ''),
            authors=format_authors(item.get('creators', [])),
            # ... autres champs
            texteocr=text,
            texteocr_provider=getattr(extract_text_with_ocr, 'last_provider', 'unknown')
        )
        return doc
    except Exception as e:
        logger.error(f"[{idx}] Error processing item: {e}")
        return None

# Main processing loop
documents = []
with ThreadPoolExecutor(max_workers=PDF_EXTRACTION_WORKERS) as executor:
    # Soumettre tous les items
    futures = {
        executor.submit(process_single_item, (idx, item)): idx
        for idx, item in enumerate(items)
    }

    # Collecter résultats avec progress bar
    for future in tqdm(as_completed(futures), total=len(items), desc="Extracting PDFs"):
        doc = future.result()
        if doc:
            documents.append(doc)

logger.info(f"Successfully processed {len(documents)}/{len(items)} documents")
```

**Ajustements nécessaires** :

1. **Thread-safety logging** (début fichier) :
```python
import logging
from logging.handlers import QueueHandler, QueueListener
import queue

# Setup thread-safe logging
log_queue = queue.Queue()
queue_handler = QueueHandler(log_queue)
logger = logging.getLogger(__name__)
logger.addHandler(queue_handler)

# Listener dans main thread
listener = QueueListener(log_queue, *logger.handlers)
listener.start()
```

2. **Rate limiting API** (wrapper OCR) :
```python
import threading
from functools import wraps
import time

# Semaphore pour limiter appels API concurrents
MISTRAL_SEMAPHORE = threading.Semaphore(3)  # Max 3 appels Mistral simultanés

def rate_limited_ocr(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with MISTRAL_SEMAPHORE:
            return func(*args, **kwargs)
    return wrapper

@rate_limited_ocr
def extract_text_with_ocr(pdf_path):
    # ... code existant
```

### Tests de validation

```bash
# 1. Test avec petit corpus (10 PDFs)
time python scripts/rad_dataframe.py \
  --json tests/fixtures/10_docs.json \
  --dir tests/fixtures \
  --output /tmp/test_parallel.csv

# Comparer avec baseline:
# Baseline: ~120 secondes (séquentiel)
# Target: ~40-50 secondes (parallèle, 3-4 workers)

# 2. Test avec corpus moyen (100 PDFs)
time python scripts/rad_dataframe.py \
  --json tests/fixtures/100_docs.json \
  --dir tests/fixtures \
  --output /tmp/test_parallel_100.csv

# 3. Vérifier logs thread-safe (pas de mélange)
tail -f logs/pdf_processing.log
# Les lignes doivent être complètes, pas entrelacées

# 4. Test stabilité (5 runs consécutifs)
for i in {1..5}; do
  python scripts/rad_dataframe.py \
    --json tests/fixtures/10_docs.json \
    --dir tests/fixtures \
    --output /tmp/run_$i.csv
  echo "Run $i completed"
done
# Tous doivent réussir avec même nombre de documents
```

### Critères de succès
- ✅ Extraction 2-3x plus rapide vs baseline
- ✅ Aucune perte de données (même nb documents)
- ✅ Logs lisibles (pas d'entrelacement)
- ✅ Aucun timeout API Mistral (semaphore fonctionne)

### Estimation temps
- **Dev** : 3h
- **Test** : 1h
- **Debug** : 1h (probable)
- **Total** : 5h

### Dépendances
- **Requis** : P1-T3 (variable PDF_EXTRACTION_WORKERS dans .env)

### Risques et mitigations
- **Risque** : Race conditions dans logging
  - **Mitigation** : QueueHandler thread-safe
- **Risque** : Rate limits Mistral API
  - **Mitigation** : Semaphore limite concurrence
- **Risque** : Consommation RAM excessive (N PDFs en mémoire)
  - **Mitigation** : Limiter PDF_EXTRACTION_WORKERS à 3-4

---

## P2-T2 : Optimisation Batching API OpenAI

### Objectif
Améliorer le throughput embeddings et recodage GPT via batching intelligent.

**Gain attendu** : 1.5-2x vitesse embeddings

### Fichiers à modifier

#### `scripts/rad_chunk.py` (fonction generate_embeddings_batch)

**Avant** (ligne 352-369, batching simple) :
```python
def generate_embeddings_batch(chunks_batch):
    texts = [chunk["text"] for chunk in chunks_batch]
    response = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL
    )
    return [emb.embedding for emb in response.data]
```

**Après** (batching adaptatif avec retry) :
```python
import time
from openai import RateLimitError

def generate_embeddings_batch(chunks_batch, retry_count=0, max_retries=3):
    """
    Generate embeddings avec retry exponentiel et adaptive batching
    """
    texts = [chunk["text"] for chunk in chunks_batch]
    batch_size = len(texts)

    try:
        response = client.embeddings.create(
            input=texts,
            model=EMBEDDING_MODEL,
            timeout=60.0  # Timeout explicite
        )
        return [emb.embedding for emb in response.data]

    except RateLimitError as e:
        if retry_count >= max_retries:
            logger.error(f"Rate limit after {max_retries} retries, batch size {batch_size}")
            raise

        # Exponential backoff
        wait_time = 2 ** retry_count
        logger.warning(f"Rate limit hit, waiting {wait_time}s (retry {retry_count + 1}/{max_retries})")
        time.sleep(wait_time)

        # Réduire batch size si échec répété
        if retry_count > 0 and batch_size > 10:
            # Split batch en deux
            mid = batch_size // 2
            logger.info(f"Splitting batch {batch_size} → {mid} + {batch_size - mid}")

            emb1 = generate_embeddings_batch(chunks_batch[:mid], retry_count + 1, max_retries)
            emb2 = generate_embeddings_batch(chunks_batch[mid:], retry_count + 1, max_retries)
            return emb1 + emb2
        else:
            return generate_embeddings_batch(chunks_batch, retry_count + 1, max_retries)

    except Exception as e:
        logger.error(f"Embedding error (batch {batch_size}): {e}")
        # Fallback: process un par un
        if batch_size > 1:
            logger.info(f"Fallback: processing batch individually")
            embeddings = []
            for chunk in chunks_batch:
                try:
                    resp = client.embeddings.create(input=[chunk["text"]], model=EMBEDDING_MODEL)
                    embeddings.append(resp.data[0].embedding)
                except Exception as e2:
                    logger.error(f"Failed individual embedding: {e2}")
                    embeddings.append([0.0] * 3072)  # Zero vector fallback
            return embeddings
        else:
            raise
```

**Amélioration parallélisation** (ligne 371-390) :

**Avant** :
```python
with ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS) as executor:
    futures = [executor.submit(generate_embeddings_batch, batch) for batch in batches]
    for future in tqdm(as_completed(futures), total=len(batches)):
        embeddings.extend(future.result())
```

**Après** (avec monitoring) :
```python
import time

batch_start = time.time()
embeddings_generated = 0
rate_limit_hits = 0

with ThreadPoolExecutor(max_workers=min(DEFAULT_MAX_WORKERS, 4)) as executor:
    # Max 4 workers pour éviter rate limits
    futures = {executor.submit(generate_embeddings_batch, batch): i for i, batch in enumerate(batches)}

    for future in tqdm(as_completed(futures), total=len(batches), desc="Generating embeddings"):
        try:
            batch_embeddings = future.result()
            embeddings.extend(batch_embeddings)
            embeddings_generated += len(batch_embeddings)
        except RateLimitError:
            rate_limit_hits += 1
            logger.error(f"Batch failed after retries")

elapsed = time.time() - batch_start
logger.info(f"Embeddings: {embeddings_generated} in {elapsed:.1f}s ({embeddings_generated/elapsed:.1f}/s)")
logger.info(f"Rate limit hits: {rate_limit_hits}")
```

### Tests de validation

```bash
# 1. Test petit batch (100 chunks)
python scripts/rad_chunk.py \
  --input tests/fixtures/100_chunks.json \
  --output /tmp/test_embed \
  --phase dense

# Vérifier logs:
# - "Embeddings: 100 in XXs (YY/s)"
# - Rate limit hits: 0 ou faible

# 2. Test gros batch (1000 chunks)
python scripts/rad_chunk.py \
  --input tests/fixtures/1000_chunks.json \
  --output /tmp/test_embed_1k \
  --phase dense

# 3. Test rate limit intentionnel
# Réduire DEFAULT_EMBEDDING_BATCH_SIZE à 100
# Lancer avec 5000 chunks
# Vérifier retry mechanism fonctionne

# 4. Comparaison performance
# Baseline: 1000 chunks en ~60s (16.7/s)
# Target: 1000 chunks en ~30-40s (25-33/s)
```

### Critères de succès
- ✅ Vitesse embeddings 1.5-2x baseline
- ✅ Rate limit hits < 5% des batches
- ✅ Aucune perte de chunks (count identique avant/après)
- ✅ Logs montrent throughput (embeddings/sec)

### Estimation temps
- **Dev** : 2h
- **Test** : 1h
- **Total** : 3h

### Dépendances
- **Requis** : P1-T3 (variables env configurées)
- **Recommandé** : P2-T1 (pour pipeline complet)

---

## P2-T3 : Cleanup Automatique Sessions

### Objectif
Implémenter TTL (Time-To-Live) et nettoyage automatique des sessions inactives.

**Gain attendu** : Prévention fuite disque, amélioration stabilité long-terme

### Fichiers à modifier

#### 1. `app/models/pipeline_session.py` (ajouter champ TTL)

**Ajouter** :
```python
from datetime import datetime, timedelta

class PipelineSession(Base):
    # ... champs existants

    expires_at = Column(DateTime, nullable=True)
    cleaned_up = Column(Boolean, default=False)

    def is_expired(self) -> bool:
        """Check si la session a expiré"""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at

    def set_expiry(self, hours: int = None):
        """Définir expiration (défaut depuis .env)"""
        ttl_hours = hours or int(os.getenv('SESSION_TTL_HOURS', 24))
        self.expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)
```

#### 2. `app/services/session_cleanup.py` (nouveau service)

```python
"""
Service de nettoyage automatique des sessions expirées
"""
import os
import shutil
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from app.models.pipeline_session import PipelineSession, SessionStatus
from app.core.database import SessionLocal

logger = logging.getLogger(__name__)

def cleanup_expired_sessions(dry_run: bool = False) -> dict:
    """
    Nettoie les sessions expirées (fichiers + DB)

    Args:
        dry_run: Si True, simule sans supprimer

    Returns:
        {"cleaned": int, "freed_mb": float, "errors": int}
    """
    db: Session = SessionLocal()
    stats = {"cleaned": 0, "freed_mb": 0.0, "errors": 0}

    try:
        # Trouver sessions expirées
        expired_sessions = db.query(PipelineSession).filter(
            PipelineSession.expires_at < datetime.utcnow(),
            PipelineSession.cleaned_up == False
        ).all()

        logger.info(f"Found {len(expired_sessions)} expired sessions")

        for session in expired_sessions:
            try:
                # Calculer taille avant suppression
                session_path = f"uploads/{session.session_folder}"
                if os.path.exists(session_path):
                    folder_size = get_folder_size(session_path)
                    stats["freed_mb"] += folder_size / 1e6

                    if not dry_run:
                        shutil.rmtree(session_path)
                        logger.info(f"Deleted {session_path} ({folder_size/1e6:.2f} MB)")

                # Marquer comme nettoyé
                if not dry_run:
                    session.cleaned_up = True
                    db.commit()

                stats["cleaned"] += 1

            except Exception as e:
                logger.error(f"Error cleaning session {session.id}: {e}")
                stats["errors"] += 1

        # Nettoyer aussi sessions orphelines (plus de 7 jours, aucune activité)
        old_sessions = db.query(PipelineSession).filter(
            PipelineSession.updated_at < datetime.utcnow() - timedelta(days=7),
            PipelineSession.status == SessionStatus.PROCESSING,  # Bloqué
            PipelineSession.cleaned_up == False
        ).all()

        logger.info(f"Found {len(old_sessions)} stale sessions (>7 days)")

        for session in old_sessions:
            try:
                session_path = f"uploads/{session.session_folder}"
                if os.path.exists(session_path) and not dry_run:
                    folder_size = get_folder_size(session_path)
                    shutil.rmtree(session_path)
                    stats["freed_mb"] += folder_size / 1e6

                if not dry_run:
                    session.status = SessionStatus.FAILED
                    session.cleaned_up = True
                    db.commit()

                stats["cleaned"] += 1
            except Exception as e:
                logger.error(f"Error cleaning stale session {session.id}: {e}")
                stats["errors"] += 1

    finally:
        db.close()

    return stats

def get_folder_size(folder_path: str) -> int:
    """Calcule taille totale dossier (bytes)"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total += os.path.getsize(filepath)
    return total
```

#### 3. `app/core/scheduler.py` (nouveau - tâches périodiques)

```python
"""
APScheduler pour tâches périodiques (cleanup, monitoring, etc.)
"""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import logging
from app.services.session_cleanup import cleanup_expired_sessions

logger = logging.getLogger(__name__)
scheduler = BackgroundScheduler()

def start_scheduler():
    """Démarre le scheduler avec jobs configurés"""

    # Job 1: Cleanup sessions (chaque nuit à 2h)
    scheduler.add_job(
        func=cleanup_expired_sessions,
        trigger=CronTrigger(hour=2, minute=0),
        id='cleanup_sessions',
        name='Cleanup expired sessions',
        replace_existing=True
    )

    # Job 2: Cleanup sessions (aussi toutes les 6h pendant journée)
    scheduler.add_job(
        func=cleanup_expired_sessions,
        trigger='interval',
        hours=6,
        id='cleanup_sessions_frequent',
        name='Cleanup sessions (frequent)',
        replace_existing=True
    )

    scheduler.start()
    logger.info("Scheduler started with cleanup jobs")

def stop_scheduler():
    """Arrête proprement le scheduler"""
    scheduler.shutdown()
    logger.info("Scheduler stopped")
```

#### 4. `app/main.py` (intégrer scheduler)

**Ajouter après création app** :
```python
from app.core.scheduler import start_scheduler, stop_scheduler

@app.on_event("startup")
async def startup_event():
    logger.info("Starting RAGpy application...")
    start_scheduler()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down RAGpy application...")
    stop_scheduler()
```

#### 5. `app/routes/admin.py` (nouveau endpoint admin)

```python
from fastapi import APIRouter, Depends
from app.services.session_cleanup import cleanup_expired_sessions

router = APIRouter(prefix="/admin", tags=["admin"])

@router.post("/cleanup/sessions")
async def trigger_cleanup(dry_run: bool = False):
    """Trigger manuel du cleanup (admin only)"""
    stats = cleanup_expired_sessions(dry_run=dry_run)
    return {
        "message": "Cleanup completed" if not dry_run else "Cleanup simulation",
        "stats": stats
    }
```

#### 6. `scripts/requirements.txt` (ajouter dépendance)

```
APScheduler==3.10.4
```

### Migration database

```bash
# Créer migration Alembic
alembic revision -m "Add session TTL and cleanup fields"

# Fichier migration (alembic/versions/XXX_add_session_ttl.py):
def upgrade():
    op.add_column('pipeline_sessions', sa.Column('expires_at', sa.DateTime(), nullable=True))
    op.add_column('pipeline_sessions', sa.Column('cleaned_up', sa.Boolean(), default=False))

def downgrade():
    op.drop_column('pipeline_sessions', 'cleaned_up')
    op.drop_column('pipeline_sessions', 'expires_at')

# Appliquer migration
alembic upgrade head
```

### Tests de validation

```bash
# 1. Test dry-run cleanup
curl -X POST http://localhost:8000/admin/cleanup/sessions?dry_run=true
# Devrait retourner: {"cleaned": 0, "freed_mb": 0, "errors": 0}

# 2. Créer sessions de test avec TTL court
curl -X POST http://localhost:8000/api/pipeline/projects/1/sessions \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Session", "ttl_hours": 0.01}'  # Expire en 36s

# Attendre 1 minute, puis cleanup
sleep 60
curl -X POST http://localhost:8000/admin/cleanup/sessions

# 3. Vérifier fichiers supprimés
ls uploads/
# Le dossier de la session de test ne doit plus exister

# 4. Test scheduler (vérifier logs)
docker compose logs -f ragpy | grep "Scheduler"
# Devrait montrer: "Scheduler started with cleanup jobs"

# 5. Simuler cleanup manuel (cron imité)
docker compose exec ragpy python -c "
from app.services.session_cleanup import cleanup_expired_sessions
stats = cleanup_expired_sessions(dry_run=False)
print(stats)
"
```

### Critères de succès
- ✅ Sessions expirées nettoyées automatiquement
- ✅ Scheduler démarre avec l'application
- ✅ Endpoint admin `/admin/cleanup/sessions` fonctionnel
- ✅ Logs montrent cleanup périodique (toutes les 6h)
- ✅ Migration database appliquée sans erreur

### Estimation temps
- **Dev** : 3h
- **Migration DB** : 30 min
- **Test** : 1h
- **Total** : 4.5h

### Dépendances
- **Requis** : Alembic configuré (vérifier `alembic.ini` existe)

### Risques et mitigations
- **Risque** : Suppression accidentelle sessions actives
  - **Mitigation** : Vérifier `status != PROCESSING` avant cleanup
- **Risque** : Scheduler crash app au startup
  - **Mitigation** : Try/except autour `start_scheduler()`, logs détaillés

---

## P2-T4 : Monitoring Basique Prometheus

### Objectif
Exposer métriques Prometheus pour suivre performance temps réel.

**Gain attendu** : Visibilité problèmes avant qu'ils impactent users

### Fichiers à modifier

#### 1. `app/main.py` (instrumenter avec prometheus_fastapi_instrumentator)

**Ajouter** :
```python
from prometheus_fastapi_instrumentator import Instrumentator

# Après création app
instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="fastapi_inprogress_requests",
    inprogress_labels=True,
)

instrumentator.instrument(app).expose(app, endpoint="/metrics")
```

#### 2. `app/utils/metrics.py` (métriques custom)

```python
"""
Métriques Prometheus custom pour RAGpy
"""
from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps

# Counters
pdf_processed_total = Counter(
    'ragpy_pdf_processed_total',
    'Total PDFs processed',
    ['provider']  # mistral, openai, legacy
)

chunks_generated_total = Counter(
    'ragpy_chunks_generated_total',
    'Total chunks generated',
    ['phase']  # initial, dense, sparse
)

vectordb_upsert_total = Counter(
    'ragpy_vectordb_upsert_total',
    'Total vectors upserted',
    ['database']  # pinecone, weaviate, qdrant
)

api_errors_total = Counter(
    'ragpy_api_errors_total',
    'Total API errors',
    ['provider', 'error_type']  # openai/rate_limit, mistral/timeout, etc.
)

# Histograms (latence)
pdf_extraction_duration = Histogram(
    'ragpy_pdf_extraction_duration_seconds',
    'PDF extraction duration',
    ['provider'],
    buckets=[1, 5, 10, 30, 60, 120, 300]  # 1s, 5s, 10s, ..., 5min
)

chunking_duration = Histogram(
    'ragpy_chunking_duration_seconds',
    'Chunking duration per document',
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30]
)

embedding_batch_duration = Histogram(
    'ragpy_embedding_batch_duration_seconds',
    'Embedding batch generation duration',
    ['model'],
    buckets=[0.5, 1, 2, 5, 10, 20, 60]
)

# Gauges (état actuel)
active_sessions = Gauge(
    'ragpy_active_sessions',
    'Number of active processing sessions'
)

disk_usage_gb = Gauge(
    'ragpy_disk_usage_gb',
    'Disk usage in GB',
    ['folder']  # uploads, logs, data
)

# Decorators utilitaires
def track_duration(histogram):
    """Decorator pour tracker durée avec histogram"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                # Extraire labels depuis kwargs si présent
                labels = {}
                if 'provider' in kwargs:
                    labels['provider'] = kwargs['provider']
                if 'model' in kwargs:
                    labels['model'] = kwargs['model']

                if labels:
                    histogram.labels(**labels).observe(duration)
                else:
                    histogram.observe(duration)
        return wrapper
    return decorator
```

#### 3. `scripts/rad_dataframe.py` (instrumenter extraction)

**Ajouter début fichier** :
```python
from app.utils.metrics import (
    pdf_processed_total,
    pdf_extraction_duration,
    api_errors_total
)
```

**Modifier fonction extract_text_with_ocr** :
```python
@track_duration(pdf_extraction_duration)
def extract_text_with_ocr(pdf_path, provider='mistral'):
    """Extract text with metrics tracking"""
    try:
        # ... code existant

        # À la fin, incrémenter counter
        pdf_processed_total.labels(provider=provider).inc()

        return text
    except Exception as e:
        # Tracker erreurs
        error_type = type(e).__name__
        api_errors_total.labels(provider=provider, error_type=error_type).inc()
        raise
```

#### 4. `docker-compose.yml` (ajouter Prometheus service)

**Ajouter** :
```yaml
services:
  # ... ragpy service existant

  prometheus:
    image: prom/prometheus:latest
    container_name: ragpy-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'

volumes:
  prometheus_data:
    driver: local
```

#### 5. `monitoring/prometheus.yml` (config Prometheus)

**Créer fichier** :
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ragpy'
    static_configs:
      - targets: ['ragpy:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
```

#### 6. `scripts/requirements.txt` (dépendances)

**Ajouter** :
```
prometheus-client==0.19.0
prometheus-fastapi-instrumentator==6.1.0
```

### Tests de validation

```bash
# 1. Rebuild avec instrumentation
docker compose down
docker compose up -d

# 2. Vérifier endpoint metrics
curl http://localhost:8000/metrics | grep ragpy_
# Devrait montrer métriques custom:
# ragpy_pdf_processed_total{provider="mistral"} 0
# ragpy_chunks_generated_total{phase="initial"} 0
# ...

# 3. Générer activité
python scripts/rad_dataframe.py \
  --json tests/fixtures/10_docs.json \
  --dir tests/fixtures \
  --output /tmp/test_metrics.csv

# 4. Vérifier métriques mises à jour
curl http://localhost:8000/metrics | grep ragpy_pdf_processed_total
# ragpy_pdf_processed_total{provider="mistral"} 10

# 5. Accéder Prometheus UI
open http://localhost:9090
# Query: rate(ragpy_pdf_processed_total[5m])
# Devrait montrer graphique activité

# 6. Tester queries utiles
# - PDF processing rate: rate(ragpy_pdf_processed_total[1m])
# - Average extraction time: rate(ragpy_pdf_extraction_duration_seconds_sum[5m]) / rate(ragpy_pdf_extraction_duration_seconds_count[5m])
# - Error rate: rate(ragpy_api_errors_total[5m])
```

### Critères de succès
- ✅ Endpoint `/metrics` expose métriques Prometheus
- ✅ Prometheus scrape avec succès (targets UP)
- ✅ Métriques custom incrémentées correctement
- ✅ Histograms montrent distribution latences

### Estimation temps
- **Dev** : 3h
- **Config Prometheus** : 1h
- **Test** : 1h
- **Total** : 5h

### Dépendances
- **Optionnel** : P2-T3 (pour métriques cleanup)

---

## P2-T5 : Index Database Sessions

### Objectif
Optimiser requêtes DB fréquentes avec indexes appropriés.

**Gain attendu** : 5-10x vitesse queries sessions

### Fichiers à modifier

#### `alembic/versions/XXX_add_session_indexes.py` (nouvelle migration)

```python
"""Add indexes to pipeline_sessions table

Revision ID: add_session_indexes
"""
from alembic import op

def upgrade():
    # Index sur status (filtrage sessions actives)
    op.create_index(
        'ix_pipeline_sessions_status',
        'pipeline_sessions',
        ['status']
    )

    # Index composite status + updated_at (cleanup queries)
    op.create_index(
        'ix_pipeline_sessions_status_updated',
        'pipeline_sessions',
        ['status', 'updated_at']
    )

    # Index sur expires_at (cleanup queries)
    op.create_index(
        'ix_pipeline_sessions_expires_at',
        'pipeline_sessions',
        ['expires_at']
    )

    # Index sur project_id (filtrage par projet)
    op.create_index(
        'ix_pipeline_sessions_project_id',
        'pipeline_sessions',
        ['project_id']
    )

def downgrade():
    op.drop_index('ix_pipeline_sessions_project_id')
    op.drop_index('ix_pipeline_sessions_expires_at')
    op.drop_index('ix_pipeline_sessions_status_updated')
    op.drop_index('ix_pipeline_sessions_status')
```

### Tests de validation

```bash
# 1. Appliquer migration
alembic upgrade head

# 2. Vérifier indexes créés (SQLite)
sqlite3 data/ragpy.db "SELECT * FROM sqlite_master WHERE type='index' AND tbl_name='pipeline_sessions';"

# 3. Test performance queries
# Avant indexes (baseline):
time sqlite3 data/ragpy.db "SELECT * FROM pipeline_sessions WHERE status='processing';"

# 4. Créer 1000 sessions de test
python -c "
from app.core.database import SessionLocal
from app.models.pipeline_session import PipelineSession, SessionStatus
import random

db = SessionLocal()
for i in range(1000):
    session = PipelineSession(
        project_id=1,
        session_folder=f'test_{i}',
        status=random.choice([SessionStatus.COMPLETED, SessionStatus.PROCESSING, SessionStatus.FAILED])
    )
    db.add(session)
db.commit()
print('Created 1000 test sessions')
"

# 5. Benchmark requêtes
time sqlite3 data/ragpy.db "SELECT COUNT(*) FROM pipeline_sessions WHERE status='processing';"
# Target: < 10ms avec index vs ~100ms sans

# 6. Cleanup sessions de test
sqlite3 data/ragpy.db "DELETE FROM pipeline_sessions WHERE session_folder LIKE 'test_%';"
```

### Critères de succès
- ✅ 4 indexes créés dans table pipeline_sessions
- ✅ Query `WHERE status=` 5-10x plus rapide
- ✅ Aucune erreur migration

### Estimation temps
- **Dev** : 30 min
- **Test** : 30 min
- **Total** : 1h

---

# PHASE 3 : ARCHITECTURE PRODUCTION (2-5 jours)

> **Objectif** : Transformation architecture pour scale horizontal
> **Prérequis** : Phases 1 + 2 complétées et validées
> **Validation** : Load test 50-100 utilisateurs simultanés

## P3-T1 : Queue Système Celery

### Objectif
Déléguer traitements longs (OCR, chunking, embeddings) à workers Celery pour libérer workers HTTP.

**Gain attendu** : 3-5x capacité utilisateurs simultanés

### Architecture cible

```
[User Request] → [FastAPI] → [Celery Queue] → [Worker 1]
                                            → [Worker 2]
                                            → [Worker 3]
                      ↓
                 [Redis Broker]
                      ↓
                 [Result Backend]
```

### Fichiers à modifier

#### 1. `docker-compose.yml` (ajouter services)

**Ajouter** :
```yaml
services:
  # ... ragpy existant

  redis:
    image: redis:7-alpine
    container_name: ragpy-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  celery_worker:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ragpy-celery-worker
    restart: unless-stopped
    command: celery -A app.celery_app worker --loglevel=info --concurrency=4
    volumes:
      - ./data:/app/data
      - ./uploads:/app/uploads
      - ./logs:/app/logs
      - ./sources:/app/sources
    env_file:
      - .env
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
    depends_on:
      - redis
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G

  celery_beat:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ragpy-celery-beat
    restart: unless-stopped
    command: celery -A app.celery_app beat --loglevel=info
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    env_file:
      - .env
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
    depends_on:
      - redis

  flower:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ragpy-flower
    restart: unless-stopped
    command: celery -A app.celery_app flower --port=5555
    ports:
      - "5555:5555"
    env_file:
      - .env
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
    depends_on:
      - redis

volumes:
  redis_data:
    driver: local
```

#### 2. `app/celery_app.py` (nouveau - config Celery)

```python
"""
Configuration Celery pour RAGpy
"""
from celery import Celery
import os

# Configuration broker et backend
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Créer instance Celery
celery_app = Celery(
    'ragpy',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        'app.tasks.extraction',
        'app.tasks.chunking',
        'app.tasks.embeddings',
        'app.tasks.vectordb',
        'app.tasks.cleanup'
    ]
)

# Configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,

    # Retry policy
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Timeouts
    task_soft_time_limit=3600,  # 1h soft limit
    task_time_limit=7200,        # 2h hard limit

    # Result expiration
    result_expires=86400,  # 24h

    # Concurrency
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,

    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# Beat schedule (tâches périodiques)
celery_app.conf.beat_schedule = {
    'cleanup-expired-sessions': {
        'task': 'app.tasks.cleanup.cleanup_sessions_task',
        'schedule': 21600.0,  # Toutes les 6h
    },
    'update-system-metrics': {
        'task': 'app.tasks.monitoring.update_metrics_task',
        'schedule': 60.0,  # Chaque minute
    },
}
```

#### 3. `app/tasks/extraction.py` (nouveau - task OCR)

```python
"""
Celery task pour extraction PDF
"""
from celery import Task
from app.celery_app import celery_app
import logging
import sys
import os

# Ajouter scripts/ au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts'))

from rad_dataframe import process_zotero_to_csv
from app.utils.metrics import pdf_processed_total

logger = logging.getLogger(__name__)

class ExtractionTask(Task):
    """Base task avec retry automatique"""
    autoretry_for = (Exception,)
    retry_kwargs = {'max_retries': 3}
    retry_backoff = True

@celery_app.task(base=ExtractionTask, bind=True, name='extraction.process_dataframe')
def process_dataframe_task(
    self,
    json_path: str,
    base_dir: str,
    output_path: str,
    session_id: int
):
    """
    Process Zotero JSON + PDFs → CSV

    Args:
        self: Task instance (bind=True)
        json_path: Path to Zotero export JSON
        base_dir: Base directory for PDF resolution
        output_path: Output CSV path
        session_id: Database session ID

    Returns:
        {"status": "success", "row_count": int, "output": str}
    """
    try:
        # Update session status
        from app.core.database import SessionLocal
        from app.models.pipeline_session import PipelineSession, SessionStatus

        db = SessionLocal()
        session = db.query(PipelineSession).filter(PipelineSession.id == session_id).first()
        if session:
            session.status = SessionStatus.PROCESSING
            db.commit()
        db.close()

        # Progress callback
        def progress_callback(current, total, item_title):
            """Update task state avec progress"""
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': current,
                    'total': total,
                    'item': item_title,
                    'percent': int((current / total) * 100)
                }
            )

        # Execute extraction
        logger.info(f"Starting extraction: {json_path}")
        result = process_zotero_to_csv(
            json_path=json_path,
            base_dir=base_dir,
            output_csv=output_path,
            progress_callback=progress_callback
        )

        # Update session avec résultats
        db = SessionLocal()
        session = db.query(PipelineSession).filter(PipelineSession.id == session_id).first()
        if session:
            session.status = SessionStatus.COMPLETED
            session.row_count = result['row_count']
            db.commit()
        db.close()

        # Update metrics
        pdf_processed_total.labels(provider='celery').inc(result['row_count'])

        logger.info(f"Extraction completed: {result['row_count']} documents")
        return {
            "status": "success",
            "row_count": result['row_count'],
            "output": output_path
        }

    except Exception as e:
        logger.error(f"Extraction failed: {e}")

        # Update session status
        db = SessionLocal()
        session = db.query(PipelineSession).filter(PipelineSession.id == session_id).first()
        if session:
            session.status = SessionStatus.FAILED
            session.error_message = str(e)
            db.commit()
        db.close()

        raise
```

#### 4. `app/routes/processing.py` (adapter pour Celery)

**Remplacer subprocess par task Celery** :

**Avant** :
```python
@router.post("/process_dataframe")
async def process_dataframe(path: str):
    cmd = ["python", "scripts/rad_dataframe.py", ...]
    process = await asyncio.create_subprocess_exec(*cmd)
    await process.wait()
```

**Après** :
```python
from app.tasks.extraction import process_dataframe_task

@router.post("/process_dataframe")
async def process_dataframe(
    path: str,
    session_id: int,
    background_tasks: BackgroundTasks
):
    """
    Lance extraction PDF en background via Celery
    Retourne task_id pour polling status
    """
    # Résoudre chemins
    session_folder = f"uploads/{path}"
    json_path = f"{session_folder}/input.json"
    output_path = f"{session_folder}/output.csv"

    # Lancer task Celery
    task = process_dataframe_task.delay(
        json_path=json_path,
        base_dir=session_folder,
        output_path=output_path,
        session_id=session_id
    )

    return {
        "task_id": task.id,
        "status": "queued",
        "message": "Extraction task queued"
    }

@router.get("/task/{task_id}/status")
async def get_task_status(task_id: str):
    """Poll task status (remplace SSE)"""
    from app.celery_app import celery_app

    task = celery_app.AsyncResult(task_id)

    if task.state == 'PENDING':
        response = {
            "state": task.state,
            "status": "Task queued, waiting for worker"
        }
    elif task.state == 'PROGRESS':
        response = {
            "state": task.state,
            "current": task.info.get('current', 0),
            "total": task.info.get('total', 1),
            "percent": task.info.get('percent', 0),
            "item": task.info.get('item', '')
        }
    elif task.state == 'SUCCESS':
        response = {
            "state": task.state,
            "result": task.result
        }
    elif task.state == 'FAILURE':
        response = {
            "state": task.state,
            "error": str(task.info)
        }
    else:
        response = {
            "state": task.state,
            "status": "Unknown state"
        }

    return response
```

#### 5. `scripts/requirements.txt` (dépendances)

**Ajouter** :
```
celery==5.3.4
redis==5.0.1
flower==2.0.1
```

### Migration progressive

**Stratégie recommandée** : Dual mode (subprocess + Celery) avec feature flag

```python
# .env
ENABLE_CELERY=true

# processing.py
USE_CELERY = os.getenv('ENABLE_CELERY', 'false').lower() == 'true'

@router.post("/process_dataframe")
async def process_dataframe(...):
    if USE_CELERY:
        # Nouvelle voie Celery
        task = process_dataframe_task.delay(...)
        return {"task_id": task.id}
    else:
        # Ancienne voie subprocess (fallback)
        process = await asyncio.create_subprocess_exec(...)
```

### Tests de validation

```bash
# 1. Démarrer stack complète
docker compose up -d

# 2. Vérifier services
docker compose ps
# Doit montrer: ragpy, redis, celery_worker, celery_beat, flower

# 3. Vérifier worker Celery actif
docker compose logs celery_worker | grep "ready"
# Devrait montrer: "celery@worker ready"

# 4. Accéder Flower monitoring
open http://localhost:5555
# Doit montrer 1 worker actif, 0 tasks

# 5. Lancer task test
curl -X POST http://localhost:8000/api/pipeline/process_dataframe \
  -H "Content-Type: application/json" \
  -d '{"path": "test_session", "session_id": 1}'

# Réponse: {"task_id": "abc-123-def", "status": "queued"}

# 6. Poll task status
curl http://localhost:8000/api/pipeline/task/abc-123-def/status

# Devrait évoluer:
# PENDING → PROGRESS (avec percent) → SUCCESS

# 7. Vérifier dans Flower
# Task doit apparaître dans "Tasks" tab

# 8. Test concurrent (10 tasks simultanés)
for i in {1..10}; do
  curl -X POST http://localhost:8000/api/pipeline/process_dataframe \
    -H "Content-Type: application/json" \
    -d "{\"path\": \"test_$i\", \"session_id\": $i}" &
done
wait

# 9. Vérifier queue dans Flower
# Les 10 tasks doivent être visibles et processer en parallèle (4 workers)
```

### Critères de succès
- ✅ Worker Celery démarre et connecte à Redis
- ✅ Tasks apparaissent dans Flower
- ✅ Polling `/task/{id}/status` retourne progress
- ✅ 4 tasks concurrent s'exécutent en parallèle
- ✅ Aucune perte de données vs subprocess

### Estimation temps
- **Dev Celery** : 1 jour
- **Migration routes** : 4h
- **Tests** : 4h
- **Total** : 2 jours

### Dépendances
- **Requis** : Phase 1 + Phase 2 (surtout P2-T1 ThreadPoolExecutor)
- **Bloquant** : Redis disponible

### Risques et mitigations
- **Risque** : Complexité debugging (workers séparés)
  - **Mitigation** : Logs centralisés, Flower monitoring
- **Risque** : Redis SPOF (Single Point Of Failure)
  - **Mitigation** : Redis persistence (AOF), backup quotidien
- **Risque** : Breaking change pour frontend (SSE → polling)
  - **Mitigation** : Dual mode avec feature flag

---

## P3-T2 : Load Balancer Nginx

### Objectif
Distribuer charge HTTP entre plusieurs instances FastAPI.

**Gain attendu** : Scale horizontal, haute disponibilité

### Architecture cible

```
[Users] → [Nginx :80] → [FastAPI Instance 1 :8001]
                      → [FastAPI Instance 2 :8002]
                      → [FastAPI Instance 3 :8003]
```

### Fichiers à modifier

#### 1. `docker-compose.yml` (scale FastAPI instances)

**Modifier** :
```yaml
services:
  ragpy:
    # Renommer → ragpy-api
    # Retirer 'container_name' pour permettre scale
    build:
      context: .
      dockerfile: Dockerfile
    restart: unless-stopped
    # Retirer ports (Nginx sera le frontend)
    volumes:
      - ./data:/app/data
      - ./uploads:/app/uploads
      - ./logs:/app/logs
      - ./sources:/app/sources
    env_file:
      - .env
    environment:
      - APP_URL=${APP_URL:-http://localhost:8000}
    deploy:
      replicas: 3  # 3 instances
      resources:
        limits:
          cpus: '2.0'
          memory: 4G

  nginx:
    image: nginx:alpine
    container_name: ragpy-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - ragpy
```

**Note** : Docker Compose v3 `replicas` nécessite Swarm mode OU utiliser scale manuel :
```bash
docker compose up -d --scale ragpy=3
```

#### 2. `nginx/nginx.conf` (config load balancing)

```nginx
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for" '
                    'rt=$request_time uct="$upstream_connect_time" '
                    'uht="$upstream_header_time" urt="$upstream_response_time"';

    access_log /var/log/nginx/access.log main;

    # Performance tuning
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 500M;  # Upload gros ZIPs

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml text/javascript
               application/json application/javascript application/xml+rss;

    # Upstream (pool de backends FastAPI)
    upstream ragpy_backend {
        least_conn;  # Load balancing: connexions minimales

        # Docker Compose résolution DNS
        # Si scale manuel: ragpy-1, ragpy-2, ragpy-3
        # Si Swarm: tasks.ragpy
        server ragpy-1:8000 max_fails=3 fail_timeout=30s;
        server ragpy-2:8000 max_fails=3 fail_timeout=30s;
        server ragpy-3:8000 max_fails=3 fail_timeout=30s;

        keepalive 32;
    }

    # Server block
    server {
        listen 80;
        server_name localhost;  # Remplacer par domaine en prod

        # Security headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;

        # Root location
        location / {
            proxy_pass http://ragpy_backend;

            # Headers proxy
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # Timeouts (processus longs)
            proxy_connect_timeout 60s;
            proxy_send_timeout 3600s;     # 1h pour uploads
            proxy_read_timeout 3600s;     # 1h pour processing

            # Buffering
            proxy_buffering off;  # Important pour SSE
            proxy_request_buffering off;

            # WebSocket support (si besoin futur)
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # Health check endpoint (bypass load balancing)
        location /health {
            access_log off;
            proxy_pass http://ragpy_backend/health;
        }

        # Static files (si utilisés)
        location /static/ {
            alias /app/static/;
            expires 30d;
            add_header Cache-Control "public, immutable";
        }

        # Metrics (Prometheus)
        location /metrics {
            # Restreindre accès (IP whitelist)
            allow 172.16.0.0/12;  # Docker network
            allow 127.0.0.1;
            deny all;

            proxy_pass http://ragpy_backend/metrics;
        }
    }

    # HTTPS server (optionnel, nécessite certificats)
    # server {
    #     listen 443 ssl http2;
    #     server_name yourdomain.com;
    #
    #     ssl_certificate /etc/nginx/ssl/cert.pem;
    #     ssl_certificate_key /etc/nginx/ssl/key.pem;
    #     ssl_protocols TLSv1.2 TLSv1.3;
    #     ssl_ciphers HIGH:!aNULL:!MD5;
    #
    #     # ... même config que HTTP
    # }
}
```

#### 3. `Dockerfile` (health check interne)

**Ajouter** :
```dockerfile
# Installer curl pour Nginx health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

### Tests de validation

```bash
# 1. Scale instances manuellement
docker compose up -d --scale ragpy=3 --no-recreate

# 2. Vérifier 3 instances actives
docker compose ps | grep ragpy
# Devrait montrer: ragpy-1, ragpy-2, ragpy-3

# 3. Tester load balancing
for i in {1..30}; do
  curl -s http://localhost/health | jq -r '.timestamp' &
done
wait

# Vérifier logs Nginx
docker compose logs nginx | grep "GET /health"
# Devrait montrer distribution entre upstream servers

# 4. Test failover (kill une instance)
docker compose stop ragpy-1

# Requêtes doivent continuer sur ragpy-2 et ragpy-3
curl http://localhost/health
# Doit réussir

# 5. Redémarrer instance
docker compose start ragpy-1

# Attendre health check passe (~30s)
# Trafic doit redistribuer sur les 3

# 6. Load test avec distribution
hey -n 1000 -c 50 http://localhost/health

# Analyser logs Nginx
docker compose exec nginx cat /var/log/nginx/access.log | \
  awk '{print $11}' | sort | uniq -c
# Devrait montrer distribution équitable entre upstreams

# 7. Test session affinity (si nécessaire)
# Vérifier cookies/headers conservés après proxy
```

### Critères de succès
- ✅ 3 instances FastAPI actives
- ✅ Nginx distribue charge équitablement
- ✅ Failover automatique si instance down
- ✅ Aucune perte requête pendant redémarrage instance
- ✅ Timeout adapté aux traitements longs (3600s)

### Estimation temps
- **Config Nginx** : 3h
- **Docker Compose scale** : 2h
- **Tests failover** : 2h
- **Total** : 7h

### Dépendances
- **Recommandé** : P3-T1 (Celery pour libérer workers HTTP)

### Risques et mitigations
- **Risque** : Session stickiness nécessaire (SQLite local)
  - **Mitigation** : Migrer vers PostgreSQL partagé (voir P3-T4)
- **Risque** : Uploads interrompus si instance redémarre
  - **Mitigation** : Volume partagé `/uploads` (NFS, S3)
- **Risque** : SSL termination complexe
  - **Mitigation** : Utiliser Let's Encrypt + certbot

---

## P3-T3 : Monitoring Avancé Grafana

### Objectif
Dashboards visuels pour monitoring temps réel.

**Gain attendu** : Détection proactive anomalies, analyse tendances

### Fichiers à modifier

#### 1. `docker-compose.yml` (ajouter Grafana)

**Ajouter** :
```yaml
services:
  # ... services existants

  grafana:
    image: grafana/grafana:latest
    container_name: ragpy-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_SERVER_ROOT_URL=http://localhost:3000
    depends_on:
      - prometheus

volumes:
  grafana_data:
    driver: local
```

#### 2. `monitoring/grafana/provisioning/datasources/prometheus.yml`

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
```

#### 3. `monitoring/grafana/dashboards/ragpy_overview.json`

Dashboard JSON complet (excerpt clé) :
```json
{
  "dashboard": {
    "title": "RAGpy Overview",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [{
          "expr": "rate(http_requests_total[5m])"
        }]
      },
      {
        "title": "PDF Processing Rate",
        "targets": [{
          "expr": "rate(ragpy_pdf_processed_total[5m])"
        }]
      },
      {
        "title": "Active Sessions",
        "targets": [{
          "expr": "ragpy_active_sessions"
        }]
      },
      {
        "title": "API Error Rate",
        "targets": [{
          "expr": "rate(ragpy_api_errors_total[5m])"
        }]
      },
      {
        "title": "P95 Latency",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
        }]
      }
    ]
  }
}
```

*(Fichier complet disponible : générer via Grafana UI puis exporter)*

### Tests de validation

```bash
# 1. Démarrer Grafana
docker compose up -d grafana

# 2. Accéder UI
open http://localhost:3000
# Login: admin / admin (ou GRAFANA_PASSWORD)

# 3. Vérifier datasource
# Settings → Data Sources → Prometheus
# Status: "Data source is working"

# 4. Importer dashboard
# + → Import → Upload JSON
# Sélectionner monitoring/grafana/dashboards/ragpy_overview.json

# 5. Générer activité pour visualiser
python scripts/rad_dataframe.py \
  --json tests/fixtures/100_docs.json \
  --dir tests/fixtures \
  --output /tmp/test_grafana.csv

# 6. Vérifier dashboard rafraîchit
# Panels doivent montrer:
# - Request rate augmente
# - PDF processing rate active
# - Active sessions = 1
# - P95 latency visible

# 7. Configurer alertes (optionnel)
# Alerting → New alert rule
# Condition: ragpy_api_errors_total > 10
# Notification: Email / Slack
```

### Critères de succès
- ✅ Grafana accessible sur :3000
- ✅ Datasource Prometheus connecté
- ✅ Dashboard affiche métriques temps réel
- ✅ Panels rafraîchissent automatiquement

### Estimation temps
- **Setup Grafana** : 2h
- **Dashboard design** : 3h
- **Alertes** : 2h
- **Total** : 7h

### Dépendances
- **Requis** : P2-T4 (Prometheus configuré)

---

## P3-T4 : Migration PostgreSQL Partagé

### Objectif
Remplacer SQLite par PostgreSQL pour support multi-instances.

**Gain attendu** : Vrai scale horizontal, concurrence DB robuste

### Fichiers à modifier

#### 1. `docker-compose.yml` (ajouter PostgreSQL)

```yaml
services:
  postgres:
    image: postgres:15-alpine
    container_name: ragpy-postgres
    restart: unless-stopped
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=ragpy
      - POSTGRES_USER=ragpy
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-changeme}
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ragpy"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
    driver: local
```

#### 2. `.env` (nouvelle variable)

```bash
DATABASE_URL=postgresql://ragpy:changeme@postgres:5432/ragpy
```

#### 3. `app/core/database.py` (adapter config)

**Avant** :
```python
SQLALCHEMY_DATABASE_URL = "sqlite:///./data/ragpy.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
```

**Après** :
```python
SQLALCHEMY_DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'sqlite:///./data/ragpy.db'  # Fallback SQLite
)

# Config adaptée par DB
if SQLALCHEMY_DATABASE_URL.startswith('postgresql'):
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        pool_size=20,
        max_overflow=40,
        pool_pre_ping=True,
        pool_recycle=3600
    )
else:
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args={"check_same_thread": False}
    )
```

#### 4. Migration données (script one-time)

```python
# scripts/migrate_sqlite_to_postgres.py
"""
One-time migration SQLite → PostgreSQL
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.pipeline_session import Base, PipelineSession
from app.models.user import User
# ... autres models

def migrate():
    # Source (SQLite)
    sqlite_url = "sqlite:///./data/ragpy.db"
    sqlite_engine = create_engine(sqlite_url)
    SQLiteSession = sessionmaker(bind=sqlite_engine)

    # Target (PostgreSQL)
    postgres_url = os.getenv('DATABASE_URL')
    postgres_engine = create_engine(postgres_url)
    PostgresSession = sessionmaker(bind=postgres_engine)

    # Créer schema PostgreSQL
    Base.metadata.create_all(postgres_engine)

    # Migrate data
    sqlite_session = SQLiteSession()
    postgres_session = PostgresSession()

    try:
        # Users
        users = sqlite_session.query(User).all()
        for user in users:
            postgres_session.merge(user)

        # Sessions
        sessions = sqlite_session.query(PipelineSession).all()
        for session in sessions:
            postgres_session.merge(session)

        postgres_session.commit()
        print(f"Migrated {len(users)} users, {len(sessions)} sessions")

    except Exception as e:
        postgres_session.rollback()
        print(f"Migration failed: {e}")
        raise
    finally:
        sqlite_session.close()
        postgres_session.close()

if __name__ == '__main__':
    migrate()
```

### Tests de validation

```bash
# 1. Démarrer PostgreSQL
docker compose up -d postgres

# 2. Vérifier DB accessible
docker compose exec postgres psql -U ragpy -d ragpy -c "SELECT version();"

# 3. Migrer données existantes
docker compose exec ragpy python scripts/migrate_sqlite_to_postgres.py

# 4. Vérifier migration
docker compose exec postgres psql -U ragpy -d ragpy -c "SELECT COUNT(*) FROM pipeline_sessions;"

# 5. Redémarrer RAGpy avec PostgreSQL
docker compose restart ragpy

# 6. Test CRUD
curl -X POST http://localhost:8000/api/pipeline/projects/ \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Project", "description": "PostgreSQL test"}'

# 7. Vérifier insertion
docker compose exec postgres psql -U ragpy -d ragpy -c "SELECT * FROM projects WHERE name='Test Project';"

# 8. Test concurrence (10 writes simultanés)
for i in {1..10}; do
  curl -X POST http://localhost:8000/api/pipeline/projects/1/sessions \
    -H "Content-Type: application/json" \
    -d "{\"name\": \"Concurrent Session $i\"}" &
done
wait

# Vérifier aucun deadlock
docker compose logs postgres | grep -i "deadlock"
# Devrait être vide
```

### Critères de succès
- ✅ PostgreSQL démarre et passe health check
- ✅ Migration SQLite → PostgreSQL sans perte données
- ✅ Toutes requêtes CRUD fonctionnent
- ✅ Concurrence 10+ writes simultanés sans erreur
- ✅ Connexion pool fonctionne (logs "idle connections")

### Estimation temps
- **Setup PostgreSQL** : 2h
- **Migration script** : 3h
- **Tests concurrence** : 2h
- **Total** : 7h

### Dépendances
- **Critique** : Backup SQLite avant migration
- **Recommandé** : P3-T2 (multi-instances bénéficient de PostgreSQL)

---

## P3-T5 : Auto-Scaling et Orchestration

### Objectif
Implémenter auto-scaling basé sur charge (Docker Swarm ou Kubernetes).

**Gain attendu** : Élasticité, optimisation coûts

*(Détail complet omis pour concision - scope avancé)*

**Estimation temps** : 3-5 jours (selon choix Swarm vs K8s)

---

# ANNEXES

## A. Checklist Déploiement Production

```markdown
### Pré-déploiement
- [ ] Variables `.env` production configurées (pas de defaults)
- [ ] Secrets stockés via secrets manager (AWS SSM, Vault, etc.)
- [ ] SSL certificats générés (Let's Encrypt)
- [ ] Backup database configuré (PostgreSQL dump quotidien)
- [ ] Logs centralisés (CloudWatch, Datadog, etc.)
- [ ] Monitoring alertes configurées (PagerDuty, OpsGenie)

### Sécurité
- [ ] CORS restreint aux domaines autorisés
- [ ] Rate limiting activé (Nginx limit_req)
- [ ] Authentication JWT rotée régulièrement
- [ ] Dépendances auditées (pip-audit, safety)
- [ ] Container images scannées (Trivy, Snyk)

### Performance
- [ ] Load test validé (100+ users simultanés)
- [ ] Database indexes optimisés
- [ ] CDN configuré (static assets)
- [ ] Caching activé (Redis pour sessions)

### Opérations
- [ ] Runbook incidents documenté
- [ ] Rollback plan testé
- [ ] Health checks tous endpoints critiques
- [ ] Métriques SLO définies (99.5% uptime, P95 < 500ms)
```

## B. Scripts Utilitaires

### `scripts/load_test.sh` (quick load testing)

```bash
#!/bin/bash
# Quick load test sans Locust

ENDPOINT="${1:-http://localhost:8000/health}"
REQUESTS="${2:-1000}"
CONCURRENCY="${3:-50}"

echo "Load testing: $ENDPOINT"
echo "Requests: $REQUESTS, Concurrency: $CONCURRENCY"

hey -n $REQUESTS -c $CONCURRENCY $ENDPOINT

# Analyser résultats
echo ""
echo "Summary:"
echo "- Target throughput: > 100 req/s"
echo "- Target P95 latency: < 500ms"
echo "- Target error rate: < 1%"
```

### `scripts/cleanup_old_sessions.sh` (manuel cleanup)

```bash
#!/bin/bash
# Nettoyage manuel sessions > 30 jours

DAYS_OLD="${1:-30}"

echo "Cleaning sessions older than $DAYS_OLD days..."

# Via API
curl -X POST http://localhost:8000/admin/cleanup/sessions?dry_run=false

# Ou direct disk
find uploads/ -type d -mtime +$DAYS_OLD -exec du -sh {} \; -exec rm -rf {} \;

echo "Cleanup completed"
```

## C. Métriques Cibles (SLO)

| Métrique | Objectif | Critique |
|----------|----------|----------|
| **Uptime** | 99.5% | 99.0% |
| **Latency P50** | < 200ms | < 500ms |
| **Latency P95** | < 500ms | < 2s |
| **Latency P99** | < 2s | < 5s |
| **Error rate** | < 0.5% | < 2% |
| **Throughput** | > 100 req/s | > 50 req/s |
| **PDF extraction** | > 15 docs/min | > 5 docs/min |
| **Concurrent users** | 50+ | 20+ |

---

**FIN DU PLAN D'OPTIMISATION LOAD**

Pour questions ou support :
- Créer issue GitHub avec tag `performance`
- Consulter `.claude/CLAUDE.md` section "Optimisation performances"
- Monitoring Grafana : http://localhost:3000
- Celery Flower : http://localhost:5555
