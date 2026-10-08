# Diagnostic et corrections des barres de progression SSE

## 🐛 Problèmes identifiés et corrigés

### Problème 1 : Lecture séquentielle au lieu de concurrente (CRITIQUE)

**Bug** : Dans `sse_helpers.py`, la fonction `merge_streams()` itérait sur les tasks de manière **séquentielle** :

```python
# ❌ AVANT (BUGGÉ)
for task in tasks:
    async for event in task:  # Attend que stdout finisse AVANT de lire stderr
        yield event
```

**Impact** : Le script attendait que stdout se termine complètement avant de lire stderr. Comme `tqdm` écrit sur stderr, **aucune barre de progression n'était jamais affichée** pendant l'exécution, seulement à la toute fin.

**Correction** : Utilisation d'une `asyncio.Queue` pour merger stdout et stderr **en temps réel** :

```python
# ✅ APRÈS (CORRIGÉ)
event_queue = asyncio.Queue()

async def read_and_queue(stream, stream_name):
    async for event in read_stream(stream, stream_name):
        await event_queue.put(event)

readers = [
    asyncio.create_task(read_and_queue(process.stdout, "stdout")),
    asyncio.create_task(read_and_queue(process.stderr, "stderr"))
]

# Stream events as they come from EITHER stream
while not (all(r.done() for r in readers) and event_queue.empty()):
    event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
    yield f"data: {json.dumps(event)}\n\n"
```

### Problème 2 : Double backslash dans format SSE (CRITIQUE)

**Bug** : Les chaînes SSE utilisaient `\\n\\n` au lieu de `\n\n` :

```python
# ❌ AVANT
yield f"data: {json.dumps(event)}\\\\n\\\\n"  # Produit: data: {...}\\n\\n (littéral)
```

**Impact** : Le navigateur recevait les chaînes `\n` littérales au lieu de vraies nouvelles lignes, cassant complètement le parsing SSE côté client.

**Correction** :

```python
# ✅ APRÈS
yield f"data: {json.dumps(event)}\n\n"  # Produit: data: {...}\n\n (newlines réelles)
```

## ✅ Correctifs appliqués

**Commit** : `edf7460` - fix: Critical SSE bugs

**Fichiers modifiés** :
- `app/utils/sse_helpers.py` : +41 lines, -17 lines

**Changements** :
1. Remplacé boucle `for` par `asyncio.create_task` + `asyncio.Queue`
2. Lecture concurrente vraie de stdout et stderr
3. Streaming événements dès qu'ils arrivent (polling avec timeout)
4. Correction tous les `\\n\\n` → `\n\n`

## 🧪 Comment tester

### Étape 1 : Redémarrer FastAPI

**IMPORTANT** : Le serveur doit être redémarré pour charger les corrections.

Si le serveur utilise `--reload` :
- Il recharge automatiquement après chaque modification de fichier
- Vérifier les logs : `INFO: Application startup complete`

Si lancé manuellement :
```bash
# Arrêter (Ctrl+C)
# Relancer
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Étape 2 : Tester avec curl

```bash
# Tester l'endpoint SSE (remplacer SESSION_FOLDER par un vrai dossier)
curl -N -X POST "http://localhost:8000/process_dataframe_sse" \
  -F "path=SESSION_FOLDER" \
  2>&1 | head -50
```

**Résultat attendu** :
```
data: {"type": "init", "total": 150, "message": "Found 150 items to process"}

data: {"type": "progress", "current": 1, "total": 150, "percent": 0, "message": "Processing Zotero items: 1/150"}

data: {"type": "progress", "current": 2, "message": "Processed 2 items"}

... (événements continus)

data: {"type": "complete", "message": "Process completed successfully"}
```

### Étape 3 : Tester dans l'interface

1. Ouvrir `http://localhost:8000/pipeline`
2. Uploader un ZIP Zotero (ou CSV)
3. Cliquer sur "Extract & Process"
4. **Vérifier** :
   - ✅ Barre de progression anime en temps réel
   - ✅ Compteur "X/Y" s'incrémente
   - ✅ Status message se met à jour
   - ✅ Pourcentage augmente progressivement

## 🔍 Debugging si ça ne marche toujours pas

### Check 1 : Le serveur a-t-il rechargé ?

```bash
# Vérifier les logs du serveur
# Devrait afficher :
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

Si `--reload` ne marche pas, redémarrer manuellement.

### Check 2 : FastAPI charge-t-il le bon code ?

```bash
# Vérifier qu'il n'y a pas d'erreur d'import
curl http://localhost:8000/docs

# Devrait afficher le Swagger UI
```

### Check 3 : Les logs de subprocess sont-ils capturés ?

Ajouter temporairement dans `sse_helpers.py` ligne 66 :

```python
logger.info(f"[{stream_name}] {decoded}")  # Changer debug → info
```

Puis relancer et vérifier les logs du serveur.

### Check 4 : Le frontend appelle-t-il le bon endpoint ?

Ouvrir DevTools (F12) → Network → Filter "Fetch/XHR"

Cliquer sur "Extract & Process" et vérifier :
- URL appelée : `/process_dataframe_sse` (pas `/process_dataframe`)
- Type : `text/event-stream`
- Status : `200`
- Response devrait streamer progressivement

### Check 5 : Parser fonctionne-t-il ?

Tester les parsers manuellement :

```python
from app.utils.sse_helpers import parse_tqdm_progress, parse_dataframe_logs

# Test tqdm
line = "Processing Zotero items: 45%|████▌     | 45/100 [00:23<00:28,  1.98it/s]"
result = parse_tqdm_progress(line)
print(result)  # Devrait retourner dict avec current=45, total=100

# Test dataframe
line = "INFO - Detected Zotero JSON format: direct array with 150 items"
result = parse_dataframe_logs(line)
print(result)  # Devrait retourner dict avec type="init", total=150
```

## 📝 Logs attendus côté serveur

Quand un utilisateur lance l'extraction, les logs serveur devraient montrer :

```
INFO:app.routes.processing:SSE dataframe processing for path: 'session_xyz'
INFO:app.utils.sse_helpers:Started subprocess: python3 /path/to/rad_dataframe.py ...
DEBUG:app.utils.sse_helpers:[stderr] Processing Zotero items:   0%|          | 0/150 [00:00<?, ?it/s]
DEBUG:app.utils.sse_helpers:[stderr] Processing Zotero items:   1%|▏         | 1/150 [00:02<05:30,  2.22s/it]
DEBUG:app.utils.sse_helpers:[stdout] INFO - ✓ Item ABC123 saved (1 total)
...
```

## ⚠️ Si le problème persiste

Symptômes possibles et causes :

| Symptôme | Cause probable | Solution |
|----------|---------------|----------|
| Pas de barre du tout | Frontend appelle endpoint sync | Vérifier dans DevTools quel endpoint est appelé |
| Barre figée à 0% | Parsers ne matchent pas les logs | Activer logger.info et vérifier format des logs |
| Barre saute à 100% d'un coup | subprocess.run() au lieu de SSE | Vérifier que l'endpoint utilise bien `run_subprocess_with_sse` |
| Erreur 500 | Bug dans sse_helpers | Vérifier stack trace serveur |
| Timeout | Script Python trop lent | Augmenter timeout dans endpoint |

## 🎯 Checklist de validation

- [ ] Serveur redémarré après commit `edf7460`
- [ ] `curl` test retourne des événements SSE progressifs
- [ ] Logs serveur montrent lignes stderr capturées
- [ ] Frontend affiche barre qui anime
- [ ] Compteur X/Y s'incrémente en temps réel
- [ ] Pas d'erreur dans Console DevTools
- [ ] `/docs` fonctionne (pas d'erreur import)

## 📚 Références

- [Server-Sent Events (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
- [asyncio.Queue](https://docs.python.org/3/library/asyncio-queue.html)
- [FastAPI StreamingResponse](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)
