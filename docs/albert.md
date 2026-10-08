# Albert (DINUM) dans RAGpy

**Dernière mise à jour** : 2026-10-02 (sprint Albert R2 : politique `albert_only`, recherche et réponses sourcées, rerank, corpus, reprises OCR, images, audio, usage, évaluation ; branche `feat/albert`)

Albert est l'API d'inférence souveraine de la DINUM (OpenGateLLM, hébergement SecNumCloud). RAGpy peut s'en servir pour ses quatre usages qui envoyaient jusqu'ici des données hors UE :

| Capacité | Remplace | Modèle Albert par défaut | Sélection |
|---|---|---|---|
| Chat : recodage des chunks, notes Zotero, fiches de lecture, filtre de citations | OpenAI / OpenRouter | `ministral-3-8b-instruct-2512` (volume), `gpt-oss-120b` (notes) | modèle `albert/<modèle>` |
| OCR des PDF | Mistral en tête de chaîne | `lightonocr-2-1b` (ou `mistral-ocr-2512` via `/v1/ocr` si le compte y a accès) | `OCR_ENABLE_ALBERT=1` |
| Embeddings denses | `text-embedding-3-large` (3072 d) | `bge-m3` (1024 d, espace séparé) | `EMBEDDING_PROVIDER=albert` ou champ `embedding_provider` |
| Base vectorielle | Pinecone / Weaviate / Qdrant | collections Albert privées (4e cible) | `db_choice=albert` ou `--db albert` |
| Recherche et réponses sourcées (R2) | — | — | **retirées le 2026-10-03** (section 16) |
| Transcription audio (R2) | — (nouveau) | `whisper-large-v3` | `ALBERT_AUDIO_ENABLED=1`, import « Enregistrements audio » (section 19) |
| Politique d'inférence (R2) | — | Albert seul, traitements locaux autorisés | `ALBERT_DATA_POLICY=albert_only` (section 15) |

**Opt-in, désactivé par défaut.** Avec `ALBERT_ENABLED=0` (défaut), RAGpy se comporte exactement comme avant, à l'octet près : sorties, clés de cache, lignes stdout, réponses HTTP et SSE, HTML. C'est la même convention que `DEDUP_ENABLED=0`. Les références figées de ce comportement (goldens G1 à G13, `tests/fixtures/albert/golden_off/`) sont rejouées à chaque lot et ne sont jamais régénérées.

Références : spécification `docs/SPRINT_albert.md`, sondes `.claude/tasks/albert_probe_report.md`, décisions mesurées `tests/fixtures/albert/decisions.json`, registre des variables `scripts/rad_albert/config.py` (`ENV_REGISTRY`).

---

## 1. Activation par capacité

### 1.1 Interrupteur maître et clé

1. `ALBERT_ENABLED=1` dans le `.env` du serveur (configuration serveur, réservée à l'admin). À 0 : aucun appel, `/api/albert/*` répond 404, aucune interface ni clé JSON Albert.
2. Une clé Albert (identifiant `albert_api_key`, variable `ALBERT_API_KEY`) :
   - **admin** : clé personnelle, sinon repli sur `ALBERT_API_KEY` du `.env` ;
   - **non-admin** : clé personnelle **uniquement**, saisie dans Paramètres > Mes Identifiants (jamais celle du `.env`, même présente).
   - Création sur le playground Albert (`https://albert.playground.etalab.gouv.fr/`, page « Clés API » ; la valeur n'est affichée qu'une fois) ; l'API est réservée au secteur public éligible (administrations et opérateurs publics).
   - **Rotation annuelle des clés** : une clé expire au plus tard un an après sa création. Avant l'échéance, créer une nouvelle clé sur le playground, l'enregistrer dans RAGpy (Paramètres > Mes Identifiants ; pour la clé admin du `.env`, remplacer `ALBERT_API_KEY` puis redémarrer le serveur), vérifier par « Tester la connexion », puis seulement révoquer l'ancienne (révocation irréversible). Le preflight ne lit pas l'échéance des clés.
   - **Expiration du compte**, distincte de celle des clés : le preflight (`ALBERT_PREFLIGHT=1`) lit la date d'expiration du **compte** (`expires` de `/v1/me`) et avertit 30 jours avant ; le renouvellement du compte se demande à albert.api@numerique.gouv.fr. Un compte expiré donne un 403 qui arrête le job (section 3.5).
3. Vérification : modale admin (carte Albert) > « Tester la connexion », qui appelle `GET /api/albert/status` avec la clé enregistrée.

Un sélecteur par capacité complète l'interrupteur maître (sections suivantes). Les bibliothèques et les routes relisent la configuration à chaque appel (`AlbertConfig.from_env()`). Les scripts CLI `rad_dataframe` et `rad_chunk` chargent le `.env` (`load_dotenv_guarded()`) puis lisent la configuration à l'import ; `rad_vectordb` la lit à l'exécution (`--db albert`) **sans charger le `.env`** : ses variables doivent être présentes dans l'environnement du shell (section 1.5).

### 1.2 Chat : convention `albert/<modèle>`

- Tout nom de modèle préfixé par `albert/` (casse ignorée) est routé vers Albert, avant l'heuristique historique `provider/model` → OpenRouter. La découpe se fait au premier `/` : `albert/openai/gpt-oss-120b` envoie `openai/gpt-oss-120b`.
- Toute chaîne sans préfixe `albert/` est routée exactement comme avant.
- `albert/…` avec Albert désactivé donne un **400** (ou un événement SSE d'erreur), sans sous-processus ni envoi de texte (chunking, notes, citations, `upload_pop_json`).
- Un alias (`albert/openweight-small`) est résolu en id épinglé par `/v1/models`, en CLI comme sur le web (section 3.6) : c'est l'id qui est envoyé et qui entre dans les clés de cache et dans `recode_model` (l'API renvoie l'alias dans `response.model`, d'où cette résolution).
- Usages :
  - recodage : champ modèle de l'étape chunking, ou CLI `--model albert/ministral-3-8b-instruct-2512` ;
  - notes Zotero et fiches de lecture : `albert/gpt-oss-120b` (rôle `notes`) ;
  - filtre de citations : `albert/ministral-3-8b-instruct-2512` (rôle `citation`).
- Aucun repli silencieux vers OpenAI ou OpenRouter : après un échec définitif, le recodage garde le texte brut, une note retombe sur le gabarit, le pré-filtre des citations garde la citation ; une erreur de compte ou de quota arrête le job (section 3.5).

### 1.3 OCR : `OCR_ENABLE_ALBERT`

- `OCR_ENABLE_ALBERT=1` (avec `ALBERT_ENABLED=1` et une clé) place le maillon Albert **en tête** de la chaîne OCR. Réglage serveur, pas de choix par requête.
- Un non-admin qui n'a qu'une clé Albert peut lancer l'OCR : quand le maillon Albert est actif, la route accepte une clé Mistral, sinon OpenAI, sinon la seule clé Albert, avant de répondre 403.
- Détails en section 4.

### 1.4 Embeddings : `EMBEDDING_PROVIDER`, champ `embedding_provider`

- Serveur : `EMBEDDING_PROVIDER=openai` (défaut) ou `albert`. Avec `ALBERT_ENABLED=0`, la valeur `albert` n'est pas ignorée : la phase dense la refuse (`exit 2`).
- Par requête : champ de formulaire `embedding_provider` de l'étape dense (sélecteur de l'interface) ; il l'emporte sur la valeur serveur. Albert désactivé : champ ignoré, sauf `albert` → 400. Valeur hors {openai, albert} → 400.
- CLI : `ALBERT_ENABLED=1 .venv/bin/python scripts/rad_chunk.py --input <session>/output_chunks.json --output <session> --phase dense --embedding-provider albert` (`albert` avec Albert désactivé : `exit 2`).
- Détails en section 5.

### 1.5 Collections : `db_choice=albert`

- Interface : option « Albert » de l'étape base vectorielle, champs de collection (id, ou nom exact avec création), case de confirmation RGPD obligatoire.
- Route : `POST /upload_db` avec `db_choice=albert`, `albert_collection_id` ou `albert_collection_name`, `albert_create_collection`, `albert_gdpr_ack=true` (sinon 400 `gdpr_ack_required`).
- CLI : `ALBERT_ENABLED=1 .venv/bin/python scripts/rad_vectordb.py --input <session>/output_chunks.json --db albert --albert-collection-name ma-collection --albert-create-collection --albert-ack-retention`
  - **`ALBERT_API_KEY` doit être exportée dans le shell** avant cette commande : contrairement à `rad_chunk.py` et `rad_dataframe.py`, `rad_vectordb.py` ne charge pas le `.env` (sinon `ERROR: ALBERT_API_KEY environment variable not set`, `exit 1`). Il en va de même pour les autres réglages `ALBERT_*` du `.env` (`ALBERT_BASE_URL`, `ALBERT_METADATA_FIELDS`…), ignorés s'ils ne sont pas dans l'environnement.
  - Par la route `/upload_db`, le serveur injecte lui-même la clé de l'utilisateur dans l'environnement du sous-processus.
- Détails en section 6.

### 1.6 Parcours entièrement souverain

Pour qu'aucun texte ne quitte Albert :

1. `ALBERT_ENABLED=1`, `OCR_ENABLE_ALBERT=1` ;
2. recodage avec un modèle `albert/…` (sinon le texte LightOnOCR, recodé par défaut, partirait vers le modèle de recodage historique) ;
3. embeddings `albert` et cible `albert` ;
4. notes et citations avec un modèle `albert/…` ;
5. surveiller les entrées `OCR_PROVIDER_FALLBACK` de `*_errors.json` : elles signalent un document passé au maillon suivant (Mistral, UE) après un échec Albert.

Depuis le sprint R2, `ALBERT_DATA_POLICY=albert_only` impose ce parcours côté serveur (section 15) : les points 2 à 4 deviennent automatiques (défauts Albert, refus des autres fournisseurs) et le maillon Mistral n'est plus jamais appelé (point 5 : repli vers l'OCR local, puis legacy).

---

## 2. Modèles par rôle et cycle de vie

Catalogue épinglé dans `scripts/rad_albert/catalog.py` (ids, alias, contexte, raisonnement, dimension, date de retrait). La chaîne de repli d'un rôle est filtrée par la date du jour : un modèle retiré quitte la chaîne le jour de son retrait.

| Rôle | Primaire (alias) | Repli | Remarques |
|---|---|---|---|
| `recode`, `citation` | `ministral-3-8b-instruct-2512` (`openweight-small`, 262 144 tokens) | `mistral-small-3-2-24b-instruct-2506` jusqu'au 2026-11-30, puis `gemma-4-31b-it` à partir du 2026-12-01 | modèle de volume |
| `notes` (notes étendues, pédagogiques, d'évaluation, synthèse de livre) | `gpt-oss-120b` (`openweight-large`, 131 072 tokens, raisonnement) | `ministral-3-8b-instruct-2512` | marge `ALBERT_REASONING_HEADROOM`, effort `ALBERT_REASONING_EFFORT` ; le filtre de citations force l'effort `low` |
| `book_structure` (phase 1 des fiches), `long_context` (invite > 0,9 × 131 072 tokens) | `ministral-3-8b-instruct-2512` | comme `recode` | — |
| `ocr_doc` | `mistral-ocr-2512` (`/v1/ocr`, accès restreint) | `lightonocr-2-1b` | compte actuel : pas d'accès |
| `ocr_chat` | `lightonocr-2-1b` (`openweight-ocr`, 16 384 tokens) | aucun repli Albert (maillon Mistral ensuite) | statut à revérifier après le 2026-10-01 |
| `embed` | `bge-m3` (`openweight-embeddings`, 1024 d, lots de 64) | **aucun, jamais** | un repli mélangerait les espaces |

- Repli de rôle : seulement après des 503 « Model is too busy » répétés (`ALBERT_BUSY_RETRIES`, `ALBERT_MODEL_FALLBACK=1`) ; jamais sur 404, jamais pour les embeddings. Le modèle réellement servi est journalisé et utilisé dans la clé de cache. Un repli absent de la liste `/v1/models` du compte n'est jamais envoyé (section 3.6).
- Id épinglé disparu : 404 explicite (message renvoyant à `/v1/models`), jamais de changement silencieux de modèle.
- Exclus des défauts et des replis : `qwen3-coder-30b-a3b-instruct` (retiré le 2026-10-01), `deepseek-v4-flash` (fin de test le 2026-10-01, risque de censure), `qwen3-vl-embedding-8b` (4096 d, fin de test le 2026-10-01), `mistral-medium-2508` (accès restreint).

**Échéances.**

- **2026-10-01** : fin de la phase de test de LightOnOCR et d'`openweight-ocr` (statut annoncé incohérent), retrait de qwen3-coder, fin de test de deepseek et de qwen3-vl-embedding.
- **2026-12-01** : retrait de `mistral-small-3-2-24b-instruct-2506` (l'alias `openweight-medium` passe à `gemma-4-31b-it`), qui entre automatiquement dans les chaînes de repli.

**Re-sonde après chaque échéance** (les fixtures commitées ne sont jamais réécrites ; le diff est produit dans `data/albert_probe/`) :

```bash
ALBERT_LIVE=1 .venv/bin/python scripts/albert_probe.py --only P2,P5,P21 --out data/albert_probe --diff-against tests/fixtures/albert
```

Si le diff impose un changement : mettre à jour `catalog.py`, les fixtures concernées et `decisions.json` dans un commit dédié, puis rejouer `tests/test_albert_config.py`, `tests/test_albert_catalog.py`, `tests/test_albert_client.py`, `tests/test_albert_recode.py` et `tests/test_albert_notes.py`.

---

## 3. Quotas, débit et délais

### 3.1 Quotas partagés par compte ; régime d'expérimentation

Les quotas Albert sont attachés au **compte** (à la clé) : toutes les sessions, les workers Celery, les tests live et tout autre usage de la même clé les partagent. Chaque non-admin qui apporte sa clé a ses propres quotas ; la clé admin du `.env` est commune à tous les admins.

Le compte sondé le 2026-09-26 est en **régime d'expérimentation** (`/v1/me`, sonde P1) :

| Modèles | Requêtes par minute | Requêtes par jour | Tokens par minute |
|---|---|---|---|
| `gpt-oss-120b` | 10 | 1 000 | 128 000 |
| autres modèles de chat (ministral, LightOnOCR…) | 50 | **1 000 par modèle** | 128 000 |
| `bge-m3`, rerank | 500 | 50 000 | — |

La correspondance routeur → modèle n'est pas exposée par l'API : l'attribution ci-dessus est déduite. Le preflight (`ALBERT_PREFLIGHT=1`) détecte le régime et avertit : un recodage de masse ou l'OCR d'un gros livre peut atteindre le plafond de 1 000 requêtes par jour. **Le passage en régime de production se demande à albert.api@numerique.gouv.fr** ; reporter ensuite les nouveaux quotas dans `ALBERT_*_RPM`.

### 3.2 Limiteur et concurrence

- Limiteur proactif par rôle, réglé à environ 90 % des quotas du compte : `ALBERT_RECODE_RPM=45`, `ALBERT_NOTES_RPM=9`, `ALBERT_OCR_RPM=45`, `ALBERT_EMBED_RPM=450` (envois de chunks compris), `ALBERT_CHAT_TPM=115000` (tokens d'entrée estimés, `ALBERT_TOKEN_ESTIMATOR=chars` = longueur / 3). Tous sont multipliés par `ALBERT_PROCESS_SHARE`.
- **Le budget suit le modèle envoyé** : un modèle à raisonnement (gpt-oss, 10 requêtes par minute mesurées, D1) consomme le budget `notes` (`ALBERT_NOTES_RPM`) même sous un rôle du budget `recode` (`recode`, `citation`, `book_structure`, `long_context`), en CLI (`rad_chunk.py --model albert/gpt-oss-120b`) comme sur le web (notes, phase 1 des fiches de lecture, filtre de citations) ; sur une chaîne de repli, le budget est choisi pour chaque modèle essayé (un repli ministral revient au budget `recode`). Le journal d'usage garde le rôle demandé ; un repli ministral du rôle `notes` reste sur `notes` ; OCR et embeddings ne changent jamais de budget.
- **Rafale du seau local** (`ALBERT_LIMITER_BACKEND=local`, et seau de repli du backend Redis) : la rafale de départ vaut la concurrence configurée du rôle (`ALBERT_RECODE_CONCURRENCY`, `ALBERT_NOTES_CONCURRENCY`, `ALBERT_OCR_CONCURRENCY`, `ALBERT_EMBED_CONCURRENCY` pour les embeddings et les envois vers une collection), plafonnée à une minute de débit : ces envois parallèles partent ensemble, sans attente. La rafale de départ est prise sur le débit de la première minute (jamais plus de `RPM × part` requêtes sur la première minute d'un processus), puis le débit régulier s'applique. La rafale plafonne aussi le niveau accumulé pendant une inactivité, d'où la borne exacte : au plus RPM × part + rafale − 1 requêtes sur toute minute glissante, la rafale valant au plus la concurrence du rôle (46 requêtes à 45 par minute avec la rafale de 2 du recodage, atteintes juste après une inactivité). De même pour les tokens d'entrée estimés : au plus TPM × part + rafale de tokens − 1 tokens sur toute minute glissante, la rafale de tokens étant la même fraction de minute que la rafale de requêtes (rafale × TPM / RPM : environ 5 100 tokens pour le budget `recode` et 25 600 pour le budget `notes` avec les valeurs par défaut) ; une requête plus grosse que cette rafale part dès que le seau la contient et ne dépasse la borne que de son propre excédent. Avant le lot 9, le seau démarrait plein et se remplissait d'une minute entière pendant une inactivité : près du double du débit réglé sur une minute (89 requêtes à 45 par minute).
- `ALBERT_LIMITER_BACKEND=local` (défaut) : un budget par processus. Sous Celery ou avec plusieurs sessions simultanées, passer à `redis` (`ALBERT_LIMITER_REDIS_URL`, sinon `CELERY_BROKER_URL`, sinon `redis://localhost:6379/0`). Le budget Redis est commun à toute l'instance, par rôle et non par clé : c'est prudent quand plusieurs clés coexistent.
- **Redis injoignable** : repli sur un seau local (un seul WARNING par panne), puis Redis est retenté toutes les 30 s (`REDIS_RETRY_SECONDS`) ; dès qu'il répond, la fenêtre partagée reprend et le seau local est abandonné (un INFO le signale). Jamais de repli définitif ni d'échec du job pour une panne Redis.
- En asynchrone (notes, fiches, citations), l'attente du limiteur n'occupe aucun fil de l'exécuteur par défaut de la boucle : un seau local est réservé puis attendu par `asyncio.sleep` ; l'acquisition d'un limiteur Redis et la pause posée après un 429 passent par un exécuteur dédié (`albert-limiter`).
- Plafonds de concurrence par processus, valeurs retenues dans `tests/fixtures/albert/decisions.json` : `ALBERT_RECODE_CONCURRENCY=2`, `ALBERT_NOTES_CONCURRENCY=2` et `ALBERT_EMBED_CONCURRENCY=4` (D22), `ALBERT_OCR_CONCURRENCY=1` (D5), `ALBERT_PUSH_CONCURRENCY=1` (D19). D22 part de la formule `ceil(0.9 × RPM × p50 / 60)` : elle donne 2 pour les notes (9 requêtes par minute, p50 9,6 s), mais 1 pour le recodage (45 par minute, p50 0,94 s) ; le recodage est gardé à 2 parce que le limiteur par minute reste la garde principale (note de D22).
- Sur le web, le sémaphore Albert (`ALBERT_NOTES_CONCURRENCY`) est pris **avant** le sémaphore global (`MAX_CONCURRENT_LLM_CALLS`) : une tâche Albert en attente ne retient aucune place globale et ne retarde donc pas les appels OpenAI ou OpenRouter de la plateforme.
- Une seule couche de retry (client Albert) : `ALBERT_MAX_RETRIES=4`, backoff `ALBERT_RETRY_BACKOFF=2.0` plafonné à `ALBERT_RETRY_MAX_BACKOFF=60`, sommeil hors sémaphores. Un 429 avec `Retry-After` ≤ `ALBERT_RETRY_AFTER_MAX` (120 s) est attendu ; au-delà, ou s'il persiste après les réessais, c'est un quota épuisé (arrêt du job).
- **429 sans `Retry-After`** (en-tête facultatif chez Albert) : réessais normaux ; s'ils sont épuisés avant qu'une fenêtre complète d'une minute (60 s, plus 1 s de marge) se soit écoulée depuis le premier 429, une attente complémentaire complète la fenêtre, puis un dernier essai a lieu. Seul un 429 qui persiste après cette fenêtre complète compte comme quota épuisé : un dépassement par minute n'est jamais pris pour le plafond journalier. Sans réessai permis (`ALBERT_MAX_RETRIES=0`), aucune attente n'est ajoutée.
- Délais HTTP par type d'appel : `ALBERT_TIMEOUT_CHAT`, `ALBERT_TIMEOUT_NOTES`, `ALBERT_TIMEOUT_OCR_PAGE`, `ALBERT_TIMEOUT_OCR_DOC`, `ALBERT_TIMEOUT_EMBED`, `ALBERT_TIMEOUT_COLLECTIONS`.

### 3.3 Calcul de débit

Durée d'une phase (minutes) ≈ max( N / (RPM × part), N × p50 / (60 × concurrence) ), où N est le nombre de requêtes :

| Phase | Requêtes | Débit plafond (défauts) | Exemple | Plafond journalier (expérimentation) |
|---|---|---|---|---|
| OCR LightOnOCR | 1 par page | 45 pages/min (p50 1,15 s, concurrence 1) | livre de 300 pages ≈ 7 min | 1 000 pages par jour |
| Recodage | 1 par chunk | 45 chunks/min (p50 0,94 s) | 1 000 chunks ≈ 22 min | 1 000 chunks par jour et par modèle (filtre de citations et phase 1 des fiches compris) |
| Notes gpt-oss | 1 par note | 9 notes/min (p50 9,6 s, concurrence 2) | 100 notes ≈ 11 min | 1 000 par jour |
| Fiche de livre | 1 (structure) + 1 par chapitre + 1 (synthèse) | idem notes | livre de 17 chapitres ≈ 19 requêtes | idem |
| Embeddings bge-m3 | 1 par lot de `min(DEFAULT_EMBEDDING_BATCH_SIZE, 64)` textes | 450 requêtes/min | 10 000 chunks en lots de 32 ≈ 313 requêtes | 50 000 par jour |
| Envoi vers une collection | 1 POST par tranche de 64 chunks, environ 2 requêtes bge-m3 par POST (D19) | limiteur `embed` | 5 000 chunks ≈ 79 POST | quota bge-m3 |

En régime d'expérimentation, le plafond journalier est atteint bien avant le délai des sous-processus. En régime de production, vérifier que la durée estimée reste sous `ALBERT_SUBPROCESS_TIMEOUT`.

### 3.4 `ALBERT_SUBPROCESS_TIMEOUT`

- Défaut 21 600 s (6 h), soit au plus environ 16 200 requêtes à 45 par minute.
- Appliqué **seulement** quand la requête sélectionne Albert : modèle `albert/…`, `embedding_provider=albert`, `db_choice=albert` ou OCR Albert actif. Sinon, les délais historiques (1 800 s, 3 600 s pour `/upload_db`) sont inchangés.
- Même règle sous Celery (`runner.run_script`), avec des limites Celery relevées pour les seules tâches Albert (`runner.albert_time_limits()` : `soft_time_limit` = délai + 300 s, `time_limit` = délai + 600 s ; les limites globales restent 3 600 / 7 200 s).
- Corpus plus long : relever `ALBERT_SUBPROCESS_TIMEOUT` ou découper le corpus en plusieurs sessions.

### 3.5 Erreurs de compte et de quota

`AlbertAuthError` (401 clé invalide, 403 compte expiré, 400 budget épuisé) et `AlbertQuotaExhausted` arrêtent le job entier. `AlbertQuotaExhausted` signale un 429 avec `Retry-After` > 120 s, ou un 429 qui persiste après les réessais ; pour un 429 **sans `Retry-After`**, seulement après une fenêtre complète d'une minute depuis le premier 429 (attente complémentaire puis dernier essai, section 3.2), pour qu'un simple dépassement du débit par minute n'arrête jamais un job.

- OCR : l'erreur est mémorisée pour le processus (un seul WARNING), les documents suivants sautent le maillon Albert sans appel et passent à Mistral (repli tracé) ;
- recodage : lots restants en texte brut sans appel, puis `exit 2` ;
- fiches de livre et citations : l'exception remonte ;
- routes SSE : événement `{type: error, message, credential_required: albert_api_key}`, puis arrêt.

### 3.6 Liste `/v1/models` : alias et replis (CLI et web)

La liste `/v1/models` du compte sert à résoudre les alias et à filtrer les chaînes de repli. Elle est lue au plus une fois par période de 600 s et par portée (empreinte de la clé, URL de base), puis installée dans le client Albert de la requête (`AlbertClient.set_model_listing`), sans nouvelle requête :

- **CLI** (`rad_chunk.py` : recodage et embeddings) : par le preflight (`ALBERT_PREFLIGHT=1`), dont le résultat est mis en cache 600 s (clé, URL de base, rôles, date) ;
- **web** (notes Zotero, fiches de lecture, filtre de citations), sans preflight : la liste est chargée à la résolution d'un alias ou, quand aucune liste n'est en cache et que la chaîne a un repli, juste avant le premier repli (après les 503 répétés du modèle en tête ; aucune requête si ce modèle répond), en chemin synchrone comme asynchrone (fil de travail, la boucle n'est jamais bloquée) ; elle est gardée 600 s par le processus serveur et installée dans chaque nouveau client de la même clé, y compris quand le modèle demandé est un id épinglé ; la liste d'une clé n'est jamais installée dans le client d'une autre.

**Alias.** Un alias (`openweight-small`, `openai/gpt-oss-120b`…) est résolu en id servi par `/v1/models`, jamais par la table statique du catalogue, qui ne suit pas un alias re-pointé par Albert (`openweight-medium` passe à `gemma-4-31b-it` le 2026-12-01). C'est l'id qui est envoyé, journalisé et utilisé dans les clés de cache. Un id épinglé ne déclenche aucune requête ; un alias absent de la liste du compte donne une erreur explicite, sans envoi.

**Replis.** Quand la liste est connue, un modèle de repli absent de la liste du compte est retiré de la chaîne du rôle et n'est jamais envoyé : un 503 répété du modèle en tête ne se transforme plus en 404 permanent (qui arrêterait le job) sur un repli que le compte n'offre pas. Le modèle en tête est toujours gardé : son absence reste une erreur explicite (404 renvoyant à `/v1/models`), jamais un repli silencieux. Sans liste connue en CLI (par exemple `ALBERT_PREFLIGHT=0`), la chaîne du catalogue est gardée telle quelle ; sur le web, la liste est chargée juste avant le premier repli, au plus une fois par clé et par 600 s, et si cette requête échoue (hors erreur de compte, qui remonte), la chaîne du catalogue est gardée telle quelle.

---

## 4. OCR souverain

**Ordre de la chaîne quand `OCR_ENABLE_ALBERT=1`** : Albert (`/v1/ocr` si le compte y a accès, sinon LightOnOCR) → Mistral → OpenAI Vision (opt-in, désactivé par défaut) → OCR local Docling → legacy PyMuPDF. Désactivé, la chaîne, les textes, les logs et les messages sont inchangés.

- `ALBERT_OCR_MODE=auto` (défaut) : `/v1/ocr` si l'accès est établi (sonde mise en cache pour le processus), sinon LightOnOCR ; `chat` = LightOnOCR seul ; `ocr` = `/v1/ocr` seul.
- **Compte actuel : pas d'accès à `/v1/ocr`** (404 « Model mistral-ocr-2512 not found », décision D3) → l'OCR Albert passe par **LightOnOCR**.
- **LightOnOCR** (`texteocr_provider=albert_lightonocr`) : chaque page est rastérisée par PyMuPDF (`ALBERT_OCR_DPI=200`, plus grand côté `ALBERT_OCR_MAX_SIDE=1540`) et envoyée seule en chat (`ALBERT_OCR_MAX_TOKENS`, `ALBERT_OCR_TEMPERATURE`, `ALBERT_OCR_TOP_P`).
  - Un marqueur `<!-- Page N -->` par page, y compris vide ; une page en échec reçoit `<!-- OCR ÉCHOUÉ (albert_lightonocr) : raison -->`, qui n'est pas pris pour un marqueur de page.
  - `finish_reason=length` rend le résultat partiel (`texteocr_partial`, entrée `OCR_PARTIAL`) ; au-delà de `ALBERT_OCR_MAX_FAILED_RATIO` (0.5) de pages en échec, le maillon est abandonné au profit du suivant ; `ALBERT_OCR_MAX_PAGES` plafonne le nombre de pages (0 = aucun).
  - Le garde-fou de densité (`OCR_MIN_CHARS_PER_PAGE`) s'applique à LightOnOCR.
- **`/v1/ocr`** (`texteocr_provider=albert_mistral_ocr`) : PDF envoyé en URI de données, pages indexées à partir de 0 (marqueurs N = index + 1), découpé en parts de `ALBERT_OCR_PART_MB` Mo et `ALBERT_OCR_PART_PAGES` pages ; 403 ou 404 → absence d'accès mémorisée, bascule sur LightOnOCR ; 413 → un seul redécoupage plus fin, puis LightOnOCR.
- **Repli tracé, jamais silencieux** : quand Albert échoue et qu'un autre maillon produit le texte, `OCRResult.fallback_from` est renseigné et une entrée `OCR_PROVIDER_FALLBACK` est ajoutée à `*_errors.json`.

**Recodage des textes OCR Albert.**

- `albert_mistral_ocr` est ajouté à `RECODE_SKIP_PROVIDERS` (markdown propre, comme Mistral).
- `albert_lightonocr` est **recodé par défaut** (`ALBERT_OCR_SKIP_RECODE=0`) : la qualité de LightOnOCR sur des ouvrages SHS scannés n'est pas évaluée. `ALBERT_OCR_SKIP_RECODE=1` l'exclut du recodage.
- **Décision D21 en attente de confirmation par l'utilisateur**, d'après la qualité observée sur un PDF scanné réel. Rappel : pour rester souverain, recoder avec un modèle `albert/…`.

---

## 5. Embeddings bge-m3 : un espace séparé

- **Deux espaces, jamais mélangés** : OpenAI `text-embedding-3-large` (3072 d, défaut) et Albert `bge-m3` (1024 d, normalisé L2, `ALBERT_EMBED_L2_NORMALIZE=1`). Aucun repli d'un espace vers l'autre.
- Les chunks Albert portent `embedding_provider=albert`, `embedding_model=bge-m3`, `embedding_dim=1024`. Ces champs ne sont écrits **que hors défaut** : le fichier OpenAI est identique à l'octet.
- Côté Albert, jamais de vecteur nul : un texte vide n'est pas envoyé (vecteur absent), un échec laisse le vecteur absent. `ALBERT_EMBED_MAX_MISSING_RATIO=0.0` : au moindre manque, le fichier est écrit puis la phase sort en `exit 1` avec un message ; une erreur de compte donne `exit 2`.
- Lots plafonnés à 64 textes (`ALBERT_EMBED_BATCH`) ; requêtes triées par index, sans paramètre `dimensions`.
- Cache d'embeddings cloisonné (`RECODE_EMBED_CACHE_ENABLED`) : clé `albert:bge-m3` avec `{model, provider, dim, norm}` ; un vecteur en cache de mauvaise dimension compte comme absent ; les clés OpenAI sont inchangées.

**Index dédiés et gardes des connecteurs** (contrôles avant toute dédup et tout envoi) :

| Cible | Contrôle | Conseil |
|---|---|---|
| Pinecone | `.dimension` de l'index (lue sur l'appel `describe_index` existant) | créer un index dédié de dimension 1024 (métrique `dotproduct` si les vecteurs sparse sont envoyés) |
| Qdrant | taille de la collection existante ; nouvelle collection créée à la dimension uniforme du fichier | collection dédiée |
| Weaviate | lecture d'un objet avec son vecteur, **seulement** pour un espace hors défaut | classe ou tenant dédié |
| Toutes | fichier mélangeant les espaces refusé localement | — |

- `scripts/rebuild_pinecone_index.py` contrôle les dimensions **avant** de recréer l'index (`exit 2` en cas d'incohérence).
- `scripts/rad_clustering.py` refuse des dimensions mélangées avec un message clair.
- **Dédup Tier 3 désactivée pour bge-m3** : le seuil de 0.97 est calibré pour `text-embedding-3-large`. Tant que `DEDUP_SIM_THRESHOLD_BGE_M3` est vide, le Tier 3 est sauté pour un fichier bge-m3 (WARNING unique) ; s'il est défini, ce seuil s'applique. Les fichiers sans champs d'espace ne changent pas.

---

## 6. Collections Albert (4e cible vectorielle)

Les embeddings sont calculés **côté serveur** : `embedding` et `sparse_embedding` sont retirés, seuls le texte et une liste blanche de métadonnées partent. Entrée d'une session (`resolve_albert_input`) : fichier sparse, sinon `output_chunks_with_embeddings.json`, sinon `output_chunks.json`.

- **Collection privée forcée** (une collection publique est refusée). Par id (vérifiée privée) ou par nom **exact** : création si `albert_create_collection` / `--albert-create-collection` ; plusieurs homonymes → erreur listant les ids, jamais de choix implicite.
- **Métadonnées** : 10 champs scalaires au plus (`ALBERT_METADATA_FIELDS`, défaut `content_id,content_hash,chunk_index,total_chunks,title,authors,year,doi,item_key,filename`) ; `content_id`, `content_hash`, `chunk_index` et `DEDUP_META_FIELDS` y sont obligatoires (sinon refus avant tout appel) ; `path` n'est **jamais** envoyé ; valeurs vides, NaN et infinies écartées, textes tronqués à 255 caractères ; `doi`, sinon `url` sous la même clé.
- **Un document Albert par source** : `ragpy:{itemKey}:{nom du fichier}` (repli sur le titre).
- **Idempotence append-skip, toujours active** : chaque chunk a un `content_id` stable (adressé par contenu, même avec `DEDUP_ENABLED=0`) ; les chunks déjà présents sont relus (préchargement paginé) et sautés. Une relance donne `Inserted: 0` et `Skipped (existing): N`. Limite : un nouveau chunking recodé produit d'autres textes, donc d'autres ids.
- **Envoi** : tranches de 64 chunks sous `ALBERT_PUSH_CONCURRENCY` et le limiteur `embed`.
- **Rollback** : l'échec d'une tranche d'un document créé pendant le run supprime ce document (`partial_error`). Une création à l'issue incertaine (`AlbertUncertainWriteError`) n'est jamais renvoyée : elle est rapprochée par relecture, et une relance reste sûre.
- **Manifestes** : `albert_manifest.jsonl` à côté du fichier d'entrée, une ligne par tranche réussie (ids de chunks renvoyés par l'API), écrite au fil de l'eau ; copie (même partielle) dans **`data/albert_manifests/<user_id>/<session>-<horodatage>.jsonl`**, hors `uploads/`, pour survivre au nettoyage des sessions. Ces manifestes permettent de retrouver et supprimer ce qui a été déposé.
- **Sortie** : bloc `=== Result ===` inchangé, plus `Skipped (existing): N` et `Albert manifest: <chemin>` pour la seule cible albert.
- **Acquittement RGPD exigé partout** : case de l'interface, `albert_gdpr_ack=true` côté route (400 `gdpr_ack_required` sinon), `--albert-ack-retention` en CLI.
- **Journal d'audit** (table d'audit de la base) : `ALBERT_COLLECTION_UPLOAD` à chaque envoi (collection, session, acquittement) ; `ALBERT_COLLECTION_DELETE` à chaque suppression, écrite avant l'appel puis complétée du résultat.
- **Suppression** : bouton de l'interface, ou `DELETE /api/albert/collections/{id}?confirm=true` (sans `confirm=true` : 400 `confirm_required`) ; seul le propriétaire de la collection peut la supprimer. `GET /api/albert/collections` liste les collections privées (paginé).

---

## 7. RGPD et conformité

- **Chemin d'inférence (chat, OCR, embeddings)** : aucun contenu conservé par la DINUM. Seules des métadonnées techniques sont gardées **24 mois** : compte, adresse IP, tokens, code HTTP, latence.
- La DINUM agit en **sous-traitant au sens de l'art. 28 du RGPD**. Hébergement SecNumCloud, aucun appel à un fournisseur externe.
- **Exception : les collections stockent** textes et métadonnées **jusqu'à leur suppression**. L'interface le dit, exige un acquittement journalisé et propose la suppression.
- Restent à la charge de l'utilisateur : droits d'auteur des PDF déposés, données personnelles éventuellement contenues, **éligibilité** au service (secteur public) des non-admins qui apportent leur clé.
- **Supprimer les collections avant de supprimer un compte RAGpy** : la suppression d'un compte ne supprime pas ses collections Albert. Procédure : lister les collections (interface ou `GET /api/albert/collections`), s'aider des manifestes de `data/albert_manifests/<user_id>/`, supprimer chaque collection, puis supprimer le compte.
- Rotation annuelle des clés (section 1.1).

---

## 8. Modèle de sécurité

- **Identifiant unique `albert_api_key` → `ALBERT_API_KEY`**, inscrit dans les trois registres de `app/core/credentials.py` (la purge non-admin le couvre).
- **Isolation des non-admins** : `build_subprocess_env` retire toutes les variables d'identifiants et les secrets serveur (`FLOWER_PASSWORD`, `JWT_SECRET_KEY`, `RESEND_API_KEY`), réinjecte les seuls identifiants personnels et pose **`RAGPY_DOTENV_DENY`** : la liste triée, séparée par des virgules, sans espace, des `*_API_KEY` retirés et non réinjectés. `scripts/rad_env.load_dotenv_guarded()` retire ces noms après le chargement du `.env` : un sous-processus non-admin ne peut plus recharger la clé de l'admin. L'environnement admin est inchangé et la configuration non secrète reste relue.
- `/api/albert/status` répond 403 `credential_required` à un non-admin sans clé personnelle, même si le `.env` en contient une.
- **URL de base serveur uniquement** (`ALBERT_BASE_URL`) : https imposé, `/v1` garanti, hôte en liste blanche (`albert.api.etalab.gouv.fr` ; `api.albert.etalab.gouv.fr` est réécrit avec un WARNING), identifiants dans l'URL refusés, redirections jamais suivies. `ALBERT_ALLOW_CUSTOM_HOST=1` autorise un OpenGateLLM auto-hébergé. Les listings Pinecone, Weaviate et Qdrant n'acceptent plus `url` ni `api_key` en paramètre de requête (anti-SSRF).
- **Clé masquée** : `GET /get_credentials` renvoie `••••` suivi des 4 derniers caractères ; `POST /save_credentials` ignore une valeur égale au masque (une valeur vide efface la clé) ; `GET /users/me/credentials` masque la clé quand Albert est actif et ne contient aucun champ albert sinon.
- **Valeurs du formulaire admin** : `POST /save_credentials` (qui écrit le `.env`) refuse seulement une valeur contenant un séparateur de ligne (au sens de `str.splitlines` : `\n`, `\r`, VT, FF, NEL, U+2028, U+2029…) ou NUL, qui couperait la ligne `NOM=valeur` et injecterait d'autres variables : réponse 400 `{error, invalid_keys}` (noms seulement, jamais les valeurs), rien n'est écrit. Une tabulation ou tout autre caractère est accepté comme avant, Albert activé ou non.
- La clé n'apparaît jamais dans une URL, un log, le stdout, le HTML ni une réponse. La bibliothèque `scripts/rad_albert` ne lit jamais la clé dans l'environnement : elle lui est passée explicitement. Aucun appel du navigateur vers l'hôte Albert : tout passe par le serveur.
- Albert désactivé : `/api/albert/*` répond 404 comme une route inconnue, aucune clé albert dans les réponses (même si l'admin en a une en base), HTML inchangé.

---

## 9. Celery

- Routes `/api/celery/*` **authentifiées** ; la soumission vérifie l'accès à la session ; statut et annulation réservés au propriétaire de la tâche ou à un admin (propriétaire enregistré dans Redis ; sans Redis, admins seulement) ; `/api/celery/status` et `/api/celery/workers` réservés aux admins.
- Chaque tâche lance **la même CLI que la route, en sous-processus**, avec `build_subprocess_env(utilisateur chargé par user_id)` : aucun secret ne transite par Redis, l'isolation non-admin s'applique comme en mode web.
- `autoretry` limité aux erreurs d'infrastructure ; la branche albert de l'envoi vectoriel n'est jamais réessayée automatiquement.
- Limites de temps Albert : voir section 3.4. Avec Celery, utiliser `ALBERT_LIMITER_BACKEND=redis`.
- Le worker doit partager la base SQLite de `data/` et le même `JWT_SECRET_KEY` que l'application (déchiffrement Fernet des identifiants) : c'est le cas avec `docker-compose.yml`. Le processus worker garde en mémoire les clés admin du `.env` ; seuls ses sous-processus sont isolés.

---

## 10. Tests

**Suite mockée (hors ligne, derrière un proxy mort)** :

```bash
HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 NO_PROXY=127.0.0.1,localhost,testserver .venv/bin/python -m pytest tests/ scripts/test_rad_vectordb.py scripts/test_rad_chunk_initial_phase.py scripts/test_albert_guard_scope.py app/test_main.py -q -p no:cacheprovider
```

- `conftest.py` (racine) : hors live, pose `ALBERT_ENABLED=0`, `OCR_ENABLE_ALBERT=0`, `ALBERT_API_KEY=''`, `EMBEDDING_PROVIDER=''` et `ALBERT_LIMITER_BACKEND=local` avant tout import de l'application (la clé réelle n'entre jamais dans la session de test) ; installe une garde réseau (tout appel vers `*.etalab.gouv.fr` fait échouer le test) ; force `anyio_backend` à asyncio.
- Faux serveur `tests/albert_fakes.py` (`FakeAlbert`, `httpx.MockTransport`) ; clé factice `fake-albert-key-0001`.
- **Cache tiktoken persistant** : sans `TIKTOKEN_CACHE_DIR` (ni `DATA_GYM_CACHE_DIR`) dans le shell, `conftest.py` le pose dès son import sur `data/tiktoken_cache` (dossier du dépôt ignoré par git) ; le cache par défaut, sous le dossier temporaire du système, peut être purgé par macOS, et les tests qui chargent un encodage échouent alors (garde réseau ou proxy mort). Rien n'est téléchargé pendant les tests : préchauffer le cache une fois, avec le réseau, par `TIKTOKEN_CACHE_DIR=$PWD/data/tiktoken_cache .venv/bin/python -c "import tiktoken; tiktoken.encoding_for_model('text-embedding-3-large'); tiktoken.get_encoding('o200k_base')"`
- Tests de régression du lot 9 : `tests/test_albert_review_regressions_f1.py` à `_f4.py` et `_g1.py`, `tests/test_session_ownership.py`, `tests/test_albert_blank_chunks.py`, `tests/test_albert_web_ledger.py`.
- Goldens du comportement désactivé (jamais régénérés) : `.venv/bin/python -m pytest tests/test_albert_off_golden.py tests/test_albert_off_golden_routes.py -q -p no:cacheprovider`
- Synchronisation variables et documentation : `.venv/bin/python -m pytest tests/test_albert_docs_sync.py -q -p no:cacheprovider`

**Suite live (clé réelle, consomme le quota du compte)** : `ALBERT_LIVE=1` se pose **uniquement dans le shell**, jamais dans un fichier `.env` (une session dont le `.env` contient cet interrupteur est refusée). Sans lui, tous les tests live sont ignorés, même si la clé est présente.

```bash
ALBERT_LIVE=1 .venv/bin/python -m pytest tests/live -m albert_live -v -p no:cacheprovider
```

Fichiers : `tests/live/test_albert_live_core.py`, `test_albert_live_llm.py`, `test_albert_live_ocr.py` (le test `/v1/ocr` est ignoré sans accès), `test_albert_live_embed.py`, `test_albert_live_collections.py`, `test_albert_live_e2e.py`.

**Test de bout en bout souverain** (vraies routes, authentification et sous-processus ; 2 PDF texte et 1 PDF scanné ; un non-admin qui n'a qu'une clé Albert ; nettoyage final) :

```bash
ALBERT_LIVE=1 .venv/bin/python scripts/albert_e2e_smoke.py --corpus-size 3 --cleanup
```

Résultat visé : `E2E ALBERT OK (ON 7/7, OFF 3/3, isolation 2/2)`. Depuis le lot 9, le contrôle on6 (`generate_zotero_notes_sse` avec `albert/gpt-oss-120b`) est vérifié en entier : la route des notes écrit `<session>/albert_usage.jsonl` (section 14, « Traçabilité de l'usage »), et on6 exige un enregistrement du rôle `notes` servi par l'id épinglé de gpt-oss avec un coût numérique (`NOTES_LEDGER_REQUIRED = True` dans `scripts/albert_e2e_smoke.py`).

- Sans ce journal, on6 échoue : le meilleur résultat serait alors `E2E ALBERT ÉCHEC (ON 6/7, OFF 3/3, isolation 2/2)`.
- Au lot 8, l'E2E tournait avec ce contrôle assoupli (absence d'enregistrement seulement consignée) : un `E2E ALBERT OK` de cette époque ne prouve pas la traçabilité de l'usage des notes.

Ce que le run modifie ou force :

- **Environnement du serveur** (processus seulement, jamais le `.env`) : `ALBERT_ENABLED=1` et `OCR_ENABLE_ALBERT=1`, puis `ALBERT_ENABLED=0` au redémarrage des contrôles OFF ; `DEDUP_ENABLED`, `RECODE_CACHE_ENABLED` et `RECODE_EMBED_CACHE_ENABLED` forcées à `0` (la relance de l'envoi compte des chunks « déjà présents » et aucune trace ne reste dans les caches partagés) ; `ALBERT_OCR_SKIP_RECODE=0` (le texte LightOnOCR passe par le recodage contrôlé) et `ALBERT_USAGE_LOG=1` (journal d'usage toujours écrit) ; `.venv/bin` en tête du `PATH`, vérifié par `shutil.which('python3')`.
- **Base SQLite temporaire** (`DATABASE_URL` vers un dossier temporaire du système, jamais `data/ragpy.db`), avec trois utilisateurs temporaires aux ids fixes : 990001 (admin), 990002 (A, non-admin avec sa clé Albert) et 990003 (B, non-admin sans clé).
- **Nettoyage final** (bloc `finally`, chaque étape isolée : une étape en échec devient une anomalie sans empêcher les suivantes) : arrêt des serveurs, collection du run `ragpy-probe-e2e-<AAAAMMJJ-HHMMSS>`, dossiers de session et archives créés sous `uploads/`, copies de manifestes sous `data/albert_manifests/990002/`, base temporaire. Un Ctrl-C, SIGTERM ou SIGHUP interrompt le run et déclenche ce nettoyage.
- **`--cleanup`** : supprime aussi au démarrage les collections `ragpy-probe-e2e-AAAAMMJJ-HHMMSS` laissées par un run précédent de ce script, jamais celles de la suite live (`ragpy-probe-e2e-AAAAMMJJTHHMMSS-<pid>`), qui peut tourner en même temps. Sans l'option, seule la collection du run est supprimée.
- **Seule sortie durable** : `data/e2e_albert/summary-<AAAAMMJJ-HHMMSS>.json` (clé masquée). Comme l'application, le serveur ajoute ses lignes aux journaux de `logs/` (`logs/app.log`…), où le run recherche ensuite la clé (une fuite compte comme anomalie).
- **Codes de sortie** : 0 (OK), 1 (au moins un contrôle en échec ou une anomalie), 2 (refus : `ALBERT_LIVE=1` absent du shell ou présent dans le `.env`, clé absente du `.env`, `python3` du `PATH` hors de `.venv/bin`).

**Sondes et nettoyage** (script `scripts/albert_probe.py`, httpx seul, clé lue par `dotenv_values` et jamais affichée ; collections temporaires `ragpy-probe-*` supprimées en fin de sonde). Un nouveau passage complet écrit sous `data/albert_probe/<horodatage>/` et se compare aux fixtures commitées, qui ne sont jamais réécrites :

```bash
ALBERT_LIVE=1 .venv/bin/python scripts/albert_probe.py --all --out data/albert_probe --diff-against tests/fixtures/albert
```

```bash
ALBERT_LIVE=1 .venv/bin/python scripts/albert_probe.py --cleanup --dry-run
```

Attendu : `ragpy-probe restantes: 0` ; sinon relancer sans `--dry-run` pour supprimer les collections `ragpy-probe-*` restantes (E2E compris).

---

## 11. Exécution locale

- Toujours passer par `.venv/bin/python -m <outil>` : les scripts console de `.venv/bin` (`uvicorn`, `pytest`, `pip`, `celery`) pointent vers un ancien chemin du dépôt.
- Les routes lancent les scripts avec le littéral `python3` : `.venv/bin` doit être **en tête du PATH** du serveur, sinon un `python3` système sans les dépendances est utilisé.

```bash
PATH="$PWD/.venv/bin:$PATH" ALBERT_ENABLED=1 OCR_ENABLE_ALBERT=1 .venv/bin/python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Vérification rapide : `PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -c "import shutil; print(shutil.which('python3'))"` doit afficher un chemin dans `.venv/bin`.
- Pipeline CLI avec Albert (clé dans le `.env` admin ou dans l'environnement) : `ALBERT_ENABLED=1 .venv/bin/python scripts/rad_chunk.py --input sources/X/output.csv --output sources/X --phase initial --model albert/ministral-3-8b-instruct-2512`

---

## 12. Docker et déploiement sur le VPS

- Le code (`app/`, `scripts/`) est **intégré à l'image**, pas monté : tout changement de code exige une reconstruction, `docker compose up -d --build`.
- **Ordre de déploiement sur le VPS** (à respecter) :
  1. déployer d'abord le code (isolation `.env` des sous-processus non-admin comprise) ;
  2. puis ajouter `ALBERT_API_KEY` (clé admin) au `.env` du VPS, avec `ALBERT_ENABLED=1`, les sélecteurs de capacité voulus et `ALBERT_LIMITER_BACKEND=redis` si Celery est actif ;
  3. puis `docker compose up -d --build`.

  Raison : avec un code antérieur à l'isolation, une clé présente dans le `.env` serait rechargée par les sous-processus des non-admins.
- Contrôle : `docker compose logs -f ragpy`, puis « Tester la connexion » dans la modale admin.
- Python 3.10 en local, 3.11 dans l'image : `docker compose build ragpy` vérifie la construction.

---

## 13. Risques résiduels et points ouverts

- **Sessions d'autrui** (corrigé au lot 9, décision utilisateur du 2026-09-27, goldens G7 amendés de façon motivée dans `e812c64`) : les routes pipeline de `app/routes/processing.py` et `/upload_stage_file` contrôlent le dossier de session avant tout travail. Une session enregistrée d'un projet auquel un non-admin n'a pas accès donne 403 (événement SSE unique sur les routes SSE, corps JSON sur `/cluster_documents_sse`) ; un dossier qui ne se résout pas strictement sous `uploads/` donne 400 pour tous. Depuis l'audit externe du 2026-09-27 (A02, `app/core/session_access.py`), un import hors projet est réservé à son auteur (`SessionOwner`), un lecteur (`viewer`) ne modifie ni n'arrête rien (403), et un dossier sans aucune ligne en base (ancien flux `/upload_zip`, projet supprimé) est réservé aux administrateurs : le « comportement historique » ne s'applique plus. Changements visibles énumérés dans `SPRINT_albert.md` (dérogations acceptées).
- **Bouton d'arrêt** (vérifié au lot 9) : `index.html` appelle `/stop_all_scripts` sans en-tête `Authorization`, mais l'authentification accepte le cookie `access_token` envoyé par la page ; `/upload_stage_file` repose sur le même cookie.
- **Journal d'usage des routes web** (corrigé au lot 9) : les routes des notes Zotero (fiches de lecture comprises) et du filtre de citations ajoutent les appels Albert de la requête à `<session>/albert_usage.jsonl` (section 14, « Traçabilité de l'usage ») ; l'E2E l'exige pour on6 (`NOTES_LEDGER_REQUIRED = True`, section 10). `upload_pop_json` ne fait que valider le modèle, sans appel Albert.
- **Chunks au texte vide** (corrigé au lot 9, option A) : un chunk au texte vide ou blanc n'est jamais envoyé aux embeddings bge-m3 et ne compte plus comme embedding manquant (phase dense, garde des connecteurs, `rebuild_pinecone_index --all`) ; une ligne unique donne le nombre de chunks vides ignorés. Un vrai vecteur manquant fait toujours échouer la phase.
- **Colonnes d'espace héritées** : un CSV produit avant le lot 6 et possédant des colonnes nommées `embedding_provider`, `embedding_model` ou `embedding_dim` peut être lu comme une déclaration d'espace ; la phase dense les réécrit, mais un ancien fichier d'embeddings peut rester ambigu.
- **Qualité de LightOnOCR non évaluée** sur des ouvrages SHS scannés ; D21 (recodage) à confirmer ; accès `/v1/ocr` absent du compte actuel.
- **Échéances des modèles** (2026-10-01, 2026-12-01) : re-sonder après chaque date (section 2).
- **Quotas** : 1 000 requêtes par jour et par modèle de chat en expérimentation, partagés par tous les usages de la clé ; le limiteur local ne voit qu'un processus ; quotas OCR non documentés par l'API, correspondance routeur → modèle déduite.
- **Tier 3 bge-m3** : seuil `DEDUP_SIM_THRESHOLD_BGE_M3` à calibrer avant tout usage.
- **Raisonnement de gpt-oss** : du raisonnement pourrait passer dans le contenu ; parades : marge de raisonnement, garde de ratio au recodage, ministral par défaut pour le volume, lecture par jeton exact au pré-filtre des citations.
- **RGPD des collections** : textes stockés jusqu'à suppression ; droits d'auteur, données personnelles et éligibilité à la charge de l'utilisateur ; supprimer les collections avant un compte.
- **Celery** : worker partageant `data/` et `JWT_SECRET_KEY` ; accès SQLite concurrents entre web et worker ; clés admin en mémoire du worker.
- **Clé Pinecone historique** : présente dans l'historique git (commit `d1f5875`, déjà poussé) ; sa révocation côté console Pinecone reste **à confirmer par l'utilisateur**.
- **Changements visibles acceptés** : `albert/…` avec Albert désactivé donne un 400 au lieu d'un envoi à OpenRouter ; les non-admins n'héritent plus des clés du `.env` ; plus d'invite interactive dans `rad_chunk` ; `/openapi.json` gagne des composants Albert même désactivé ; au lot 9, contrôle des sessions et des chemins sous `uploads/`, authentification de `/upload_stage_file`, `/upload_zip` et `/upload_csv`, archives ZIP dont une entrée sort du dossier refusées (liste complète dans `SPRINT_albert.md`).

---

## 14. Référence des variables

Générée depuis `ENV_REGISTRY` (`scripts/rad_albert/config.py`) ; le bloc `# ===== ALBERT API (DINUM) — OPT-IN, OFF PAR DÉFAUT =====` du `.env.example` racine reprend ces défauts, ce que vérifie `tests/test_albert_docs_sync.py`. L'interrupteur des tests live ne figure volontairement dans aucun fichier d'environnement.

| Variable | Défaut | Sens |
|---|---|---|
| `ALBERT_ENABLED` | `0` | Interrupteur maître. 0 = comportement strictement identique : ni appel, ni route, ni interface, ni clé JSON Albert. |
| `ALBERT_API_KEY` | vide | Clé Bearer Albert (identifiant albert_api_key ; repli .env réservé aux admins). Jamais lue par la bibliothèque, masquée dans /get_credentials. |
| `ALBERT_BASE_URL` | `https://albert.api.etalab.gouv.fr/v1` | URL de base de l'API, configuration serveur uniquement : https, /v1 garanti, hôte en liste blanche ; vide = défaut. |
| `ALBERT_ALLOW_CUSTOM_HOST` | `0` | 1 = autorise un OpenGateLLM auto-hébergé hors liste blanche (https reste imposé). |
| `OCR_ENABLE_ALBERT` | `0` | 1 = maillon OCR Albert avant Mistral (exige ALBERT_ENABLED=1 et une clé). |
| `ALBERT_OCR_MODE` | `auto` | auto = /v1/ocr si l'accès est établi, sinon LightOnOCR ; chat = LightOnOCR seul ; ocr = /v1/ocr seul. |
| `ALBERT_OCR_CHAT_MODEL` | `lightonocr-2-1b` | Modèle OCR par chat sur pages rastérisées (id épinglé). |
| `ALBERT_OCR_DOC_MODEL` | `mistral-ocr-2512` | Modèle de /v1/ocr, accès restreint (id épinglé). |
| `ALBERT_OCR_DPI` | `200` | Résolution de rastérisation des pages envoyées à LightOnOCR. |
| `ALBERT_OCR_MAX_SIDE` | `1540` | Plus grand côté d'une page rastérisée, en pixels. |
| `ALBERT_OCR_MAX_TOKENS` | `4096` | max_tokens d'une requête OCR par chat. |
| `ALBERT_OCR_TEMPERATURE` | `0.2` | Température d'une requête OCR par chat. |
| `ALBERT_OCR_TOP_P` | `0.9` | top_p d'une requête OCR par chat. |
| `ALBERT_OCR_MAX_PAGES` | `0` | Plafond de pages par document (0 = aucun ; au-delà, résultat partiel). |
| `ALBERT_OCR_MAX_FAILED_RATIO` | `0.5` | Part de pages en échec au-delà de laquelle le maillon Albert est abandonné. |
| `ALBERT_OCR_PART_MB` | `15` | Taille maximale (Mo) d'une part envoyée en base64 à /v1/ocr. |
| `ALBERT_OCR_PART_PAGES` | `100` | Nombre maximal de pages d'une part envoyée à /v1/ocr. |
| `ALBERT_OCR_CONCURRENCY` | `1` | Requêtes OCR Albert simultanées par processus. |
| `ALBERT_OCR_SKIP_RECODE` | `0` | 1 = texte albert_lightonocr exclu du recodage (0 = recodé, défaut prudent). |
| `EMBEDDING_PROVIDER` | `openai` | Fournisseur d'embeddings : openai (text-embedding-3-large, 3072 d) ou albert (bge-m3, 1024 d, exige ALBERT_ENABLED=1). |
| `ALBERT_EMBED_MODEL` | `bge-m3` | Modèle d'embeddings Albert (id épinglé, jamais de repli). |
| `ALBERT_EMBED_BATCH` | `64` | Textes par requête d'embeddings (plafonné à 64). |
| `ALBERT_EMBED_MAX_MISSING_RATIO` | `0.0` | Part d'embeddings manquants tolérée avant l'échec de la phase (0.0 = aucun manque). |
| `ALBERT_EMBED_L2_NORMALIZE` | `1` | 1 = vecteurs normalisés L2 (norme enregistrée dans l'espace vectoriel). |
| `ALBERT_REASONING_EFFORT` | `medium` | Effort de raisonnement des modèles gpt-oss (low, medium, high). |
| `ALBERT_REASONING_HEADROOM` | `2048` | Tokens ajoutés à max_tokens pour le raisonnement des modèles gpt-oss. |
| `ALBERT_MODEL_FALLBACK` | `1` | 1 = repli sur le modèle suivant du rôle après des 503 répétés (jamais sur 404, jamais pour les embeddings). |
| `ALBERT_BUSY_RETRIES` | `2` | Essais courts sur 503 avant le repli de rôle. |
| `ALBERT_RECODE_RPM` | `45` | Requêtes par minute du rôle recode (limiteur proactif, environ 90 % du quota). |
| `ALBERT_NOTES_RPM` | `9` | Requêtes par minute du rôle notes. |
| `ALBERT_OCR_RPM` | `45` | Requêtes par minute du rôle OCR. |
| `ALBERT_EMBED_RPM` | `450` | Requêtes par minute du rôle embed (envois de chunks compris). |
| `ALBERT_CHAT_TPM` | `115000` | Tokens d'entrée par minute, estimés, pour les rôles de chat. |
| `ALBERT_PROCESS_SHARE` | `1.0` | Part des quotas attribuée à ce processus (0 < part <= 1). |
| `ALBERT_LIMITER_BACKEND` | `local` | local = limiteur par processus ; redis = partagé (recommandé avec Celery ou plusieurs sessions). |
| `ALBERT_LIMITER_REDIS_URL` | vide | URL Redis du limiteur ; vide = CELERY_BROKER_URL, sinon redis://localhost:6379/0. |
| `ALBERT_RECODE_CONCURRENCY` | `2` | Recodages Albert simultanés par processus. |
| `ALBERT_NOTES_CONCURRENCY` | `2` | Appels Albert simultanés pour les notes et les fiches, par processus. |
| `ALBERT_EMBED_CONCURRENCY` | `4` | Requêtes d'embeddings Albert simultanées. |
| `ALBERT_PUSH_CONCURRENCY` | `1` | Envois de chunks simultanés vers une collection Albert. |
| `ALBERT_MAX_RETRIES` | `4` | Réessais au plus sur erreur transitoire. |
| `ALBERT_RETRY_BACKOFF` | `2.0` | Base (secondes) du backoff exponentiel. |
| `ALBERT_RETRY_MAX_BACKOFF` | `60` | Plafond (secondes) du backoff. |
| `ALBERT_RETRY_AFTER_MAX` | `120` | Retry-After au-delà duquel un 429 signifie un quota épuisé (abandon). |
| `ALBERT_TIMEOUT_CHAT` | `120` | Délai HTTP (secondes) d'une requête de chat (recodage, filtre de citations). |
| `ALBERT_TIMEOUT_NOTES` | `300` | Délai HTTP (secondes) d'une requête de note ou de fiche. |
| `ALBERT_TIMEOUT_OCR_PAGE` | `120` | Délai HTTP (secondes) de l'OCR d'une page par chat. |
| `ALBERT_TIMEOUT_OCR_DOC` | `300` | Délai HTTP (secondes) d'une part envoyée à /v1/ocr. |
| `ALBERT_TIMEOUT_EMBED` | `60` | Délai HTTP (secondes) d'une requête d'embeddings. |
| `ALBERT_TIMEOUT_COLLECTIONS` | `120` | Délai HTTP (secondes) des appels aux collections et documents. |
| `ALBERT_SUBPROCESS_TIMEOUT` | `21600` | Délai (secondes) des sous-processus quand la requête sélectionne Albert. |
| `ALBERT_METADATA_FIELDS` | `content_id,content_hash,chunk_index,total_chunks,title,authors,year,doi,item_key,filename` | Métadonnées envoyées aux collections : 10 au plus, avec content_id, content_hash, chunk_index et DEDUP_META_FIELDS ; jamais path. |
| `DEDUP_SIM_THRESHOLD_BGE_M3` | vide | Seuil Tier 3 de la dédup pour bge-m3 ; vide = Tier 3 sauté pour bge-m3. |
| `ALBERT_USAGE_LOG` | `1` | 1 = ledger albert_usage.jsonl écrit quand Albert a été appelé. |
| `ALBERT_PREFLIGHT` | `1` | 1 = contrôle du compte et des modèles (/v1/me, /v1/models) avant un traitement Albert. |
| `ALBERT_TOKEN_ESTIMATOR` | `chars` | Estimation des tokens d'entrée du limiteur (chars = longueur / 3). |

Sprint R2 (sections 15 à 21) :

| Variable | Défaut | Sens |
|---|---|---|
| `ALBERT_DATA_POLICY` | `compatible` | compatible = fournisseurs historiques conservés ; albert_only = inférence Albert et traitement local seulement (OpenAI, OpenRouter et Mistral direct refusés) ; valeur inconnue = albert_only. |
| `ALBERT_OCR_CHECKPOINT` | `1` | 1 = reprise page par page de l'OCR LightOnOCR (pages validées gardées dans le dossier de sortie, jamais renvoyées). |
| `ALBERT_LIMITER_REQUIRE_REDIS` | `0` | 1 = avec ALBERT_LIMITER_BACKEND=redis, refuse tout nouvel appel tant que Redis est injoignable (au lieu du seau local). |
| `ALBERT_AUDIO_ENABLED` | `0` | 1 = import et transcription d'enregistrements audio par Albert (whisper-large-v3). |
| `ALBERT_AUDIO_MODEL` | `whisper-large-v3` | Modèle de transcription (type automatic-speech-recognition, id épinglé). |
| `ALBERT_AUDIO_RPM` | `45` | Requêtes par minute du rôle audio. |
| `ALBERT_AUDIO_CONCURRENCY` | `1` | Segments audio transcrits simultanément par processus. |
| `ALBERT_AUDIO_LANGUAGE` | `fr` | Langue des enregistrements (code ISO 639-1 ; auto = détection par le modèle). |
| `ALBERT_AUDIO_SEGMENT_SECONDS` | `600` | Durée (secondes) des segments découpés par ffmpeg pour les fichiers longs (30 à 3600). |
| `ALBERT_TIMEOUT_AUDIO` | `300` | Délai HTTP (secondes) de la transcription d'un segment. |

Variables internes, jamais à écrire dans un `.env` : `RAGPY_DOTENV_DENY` (posée par le serveur pour les sous-processus non-admin) et `RAGPY_UPDATE_GOLDEN` (régénération des goldens, interdite après le lot 0).

**Traçabilité de l'usage** : les scripts `rad_dataframe` (OCR), `rad_chunk` (recodage, embeddings) et `rad_vectordb` (envoi vers une collection) ajoutent à `albert_usage.jsonl` (modèle servi `response.model`, tokens, `cost`, `impacts`), dans le dossier de la phase, les appels Albert qu'ils ont faits, et affichent une ligne de synthèse (`ALBERT_USAGE_LOG=0` : synthèse seule). Aucun fichier n'est écrit si Albert n'a pas servi. Depuis le lot 9, les routes web qui appellent Albert dans le processus du serveur (notes Zotero et fiches de lecture : `generate_zotero_notes_sse` ; filtre de citations : `filter_citations_sse`, `filter_citations_bg`, `batch_import_citations_sse`) font de même : un journal par requête, remis aux clients Albert, puis ajouté à `<session>/albert_usage.jsonl` en fin de traitement, seulement si Albert a été appelé (id épinglé envoyé, modèle servi, tokens, coût ; `ALBERT_USAGE_LOG=0` : ligne de synthèse seule dans le journal du serveur).

---

## 15. Politique d'inférence : `ALBERT_DATA_POLICY` (sprint R2)

Réglage **serveur**, jamais assoupli par une requête (`scripts/rad_albert/policy.py`, couche web `app/core/albert_policy.py`).

| Valeur | Effet |
|---|---|
| `compatible` (défaut) | Comportement historique inchangé : OpenAI, OpenRouter et Mistral restent disponibles à côté d'Albert. |
| `albert_only` | Inférence par Albert seulement ; traitements locaux (OCR Docling, extraction PyMuPDF, spaCy, clustering) autorisés. |
| valeur inconnue | **Échec fermé** : traitée comme `albert_only` (variantes `albert-only`, `Albert Only`, `strict` reconnues ; WARNING unique pour une valeur inconnue). |

En `albert_only` :

- **Chat** (recodage, notes, fiches, citations, réponses) : un modèle vide prend le défaut Albert du rôle (`albert/ministral-3-8b-instruct-2512` pour le recodage, `albert/gpt-oss-120b` pour les notes) ; tout modèle sans préfixe `albert/` est refusé (400 `policy_violation`, ou un événement SSE d'erreur). Les configurations de citations stockées (modèle réutilisé tel quel) exigent un modèle `albert/…` explicite.
- **Embeddings** : `embedding_provider` vide devient `albert` (bge-m3), `openai` est refusé.
- **OCR** : maillons Mistral et OpenAI sautés ; chaîne Albert → OCR local → legacy. Les routes d'OCR n'exigent plus de clé Mistral (clé Albert si le maillon Albert est actif, sinon aucune clé).
- **Sous-processus** (toutes les étapes, routes et Celery : OCR, découpage, embeddings denses et sparse, clustering, envoi vers une base, audio) : `OPENAI_API_KEY`, `OPENROUTER_API_KEY` et `MISTRAL_API_KEY` sont retirées de l'environnement et ajoutées à `RAGPY_DOTENV_DENY` (admins compris) : aucun script ne peut les recharger depuis `.env`.
- **Scripts** (défense en profondeur) : `rad_chunk.py` refuse un `--model` non Albert ou des embeddings OpenAI (`Albert abort: kind=config reason=policy_violation`, code 2) ; `rad_dataframe.py` saute Mistral et OpenAI.
- **Albert désactivé** : tout lancement est refusé (400 `policy_requires_albert`, `Albert abort: kind=config reason=policy_requires_albert`) ; jamais de retour silencieux vers les fournisseurs historiques.

Les **exports** vers Zotero ou une base vectorielle externe (Pinecone, Weaviate, Qdrant) ne relèvent pas de cette politique : ce sont des destinations de stockage, distinctes de l'inférence. L'interface le rappelle (`GET /api/albert/capabilities`, champ `policy.note`).

## 16. Recherche et réponses sourcées — retirées le 2026-10-03

La recherche (`/api/albert/search`), les réponses sourcées (`/api/albert/rag`, flux SSE, annulation), les collections publiques, le rerank et la page « Interroger un corpus » (`/albert/rag`) du sprint R2 ont été **retirés le 2026-10-03** à la demande d'Amar : RAGpy prépare les corpus, les questions sont posées par d'autres outils. Supprimés : `scripts/rad_albert/rag.py`, `app/routes/albert_rag.py`, la page et son script, les méthodes `search`, `rerank`, `chat_stream` et `list_public_collections` du client, les réglages `rag_*`/`rerank_*` d'`AlbertConfig` et les variables `LLM_RAG_*`, `RERANK_*`, `ALBERT_RAG_*`, `ALBERT_RERANK_*` (désormais obsolètes : `env_tool prune --apply` les retire du `.env`). Archive de l'état antérieur : `data/code_backups/avant_retrait_rag.patch` et `avant_retrait_rag_untracked.tgz`. La gestion des corpus (section 17) et l'usage du compte (section 20) restent, dans `app/routes/albert_corpora.py` (`GET /api/albert/capabilities` y reste aussi, sans les champs de recherche).

## 17. Corpus Albert dans RAGpy (registre local et droits, sprint R2)

Un identifiant de collection envoyé par le navigateur n'est jamais une autorisation : les routes n'acceptent que des **corpus** du registre local (`app/models/albert_corpus.py`, tables `albert_corpora` et `albert_corpus_sources`, service `app/services/albert_access.py`), puis Albert vérifie de nouveau l'accès avec la clé de l'appelant.

- **Enregistrement automatique** après chaque envoi `db_choice=albert`, par la route comme par la tâche Celery (succès ou échec partiel), depuis les seules lignes du manifeste écrites par **cet** envoi (le manifeste de session s'allonge d'un envoi et d'un utilisateur à l'autre) : collection, documents (hors documents annulés par rollback), nombre de chunks, champs bibliographiques lus dans le fichier de chunks (titre, auteurs, année, DOI/URL, nom de fichier seul, jamais de chemin). La session d'un projet partage le corpus avec ce projet.
- **Rattachement** d'une collection privée existante : `POST /api/albert/corpora/bind` (`collection_id`, `project_id` facultatif), lue avec la clé de l'appelant ; une collection publique est refusée.
- **Droits** : lecture (liste des documents) pour le propriétaire et tout membre du projet ; écriture (suppression d'un document distant) pour le propriétaire et les collaborateurs, jamais un lecteur ; détachement local pour le propriétaire (ou un admin). Une même clé Albert partagée entre deux comptes RAGpy n'ouvre **aucun** accès croisé implicite. Le partage d'un projet ne prête jamais la clé du propriétaire : chaque membre lit avec sa propre clé (une collection privée d'un autre compte répond 404 : corpus indisponible).
- **Documents** : `GET /api/albert/corpora/{id}/documents` (liste distante jointe au catalogue local) ; `DELETE /api/albert/corpora/{id}/documents/{document_id}?confirm=true` (appartenance à la collection vérifiée avant la suppression, audit `ALBERT_DOCUMENT_DELETE` écrit avant l'appel puis complété).
- **Détachement** : `DELETE /api/albert/corpora/{id}` retire le corpus du registre, **sans** supprimer la collection distante (audit `ALBERT_CORPUS_DETACH`). La suppression d'une collection reste `DELETE /api/albert/collections/{id}?confirm=true` (section 6).
- **Suppression d'un compte** : refusée (409, inventaire des collections) tant que l'utilisateur possède des corpus Albert, sauf confirmation explicite `?albert_data=keep` ; l'inventaire est alors consigné dans l'audit `USER_DELETE` (`albert_collections_kept`), pour un effacement ultérieur, et les corpus deviennent `orphaned` (invisibles : SQLite réutilise les identifiants et n'applique pas les clés étrangères).
- **Suppression d'un projet** : ses corpus cessent d'être partagés (projet retiré), pour qu'un futur projet du même identifiant n'en hérite pas.
- **Suppression d'une collection** (`DELETE /api/albert/collections/{id}?confirm=true`) : les corpus correspondants passent à `deleted` (réponse 410 ensuite).
- Migration : tables nouvelles créées par `create_all()` au démarrage (aucune colonne ajoutée à une table existante).

## 18. Reprise page par page de l'OCR et pièces jointes images (sprint R2)

- **Reprise** (`ALBERT_OCR_CHECKPOINT=1`, défaut) : chaque page LightOnOCR réussie et non tronquée est écrite dans `<dossier de sortie>/albert_ocr_checkpoints/<empreinte du PDF>/<empreinte des paramètres>/page_NNNNN.json` (écriture atomique, chemin confiné). Une relance (le lendemain d'un quota journalier atteint, après un arrêt) ne renvoie que les pages manquantes, en échec ou tronquées. Une page n'est reprise que si le document (sha256), le modèle épinglé, le DPI, le plus grand côté, `max_tokens`, la température et `top_p` sont identiques ; un fichier altéré est ignoré. Les points de reprise appartiennent à la session (rien n'est partagé entre utilisateurs) et disparaissent avec elle. Module `scripts/rad_albert/ocr_checkpoint.py`.
- **Images** (PNG, JPEG, TIFF, WebP) en pièces jointes Zotero : transcrites seulement quand le maillon OCR Albert est utilisable (`OCR_ENABLE_ALBERT=1` et une clé Albert) ; l'image est convertie en PDF temporaire (PyMuPDF), puis suit la chaîne OCR habituelle. Albert désactivé : extension non supportée, comme avant.

## 19. Transcription audio (sprint R2)

Activation : `ALBERT_ENABLED=1` et `ALBERT_AUDIO_ENABLED=1`. Modèle `whisper-large-v3` (P26 : `verbose_json` renvoie `text`, `segments`, `language`, `duration`, `usage`). Fichiers de 20 Mo au plus en mp3 ou wav ; les autres formats et les fichiers longs sont convertis et découpés par **ffmpeg** (mono 16 kHz, segments de `ALBERT_AUDIO_SEGMENT_SECONDS`).

- **Interface** : étape 1, option « Enregistrements audio (Albert) » : import de plusieurs fichiers, langue, vocabulaire facultatif, puis transcription en direct ; l'étape suivante est le découpage (le texte transcrit n'est pas recodé : `albert_whisper` figure dans `RECODE_SKIP_PROVIDERS`).
- **Routes** : `POST /upload_audio` (noms de stockage générés côté serveur, `<session>/audio/`, `UPLOAD_MAX_MB` par fichier et `UPLOAD_MAX_UNZIPPED_MB` pour l'ensemble) et `POST /process_audio_sse` (contrôle de session, verrou d'étape, clé Albert, politique, délai `ALBERT_SUBPROCESS_TIMEOUT`).
- **CLI** : `ALBERT_ENABLED=1 ALBERT_AUDIO_ENABLED=1 .venv/bin/python scripts/rad_audio.py --input sources/X/audio --output sources/X/output.csv --language fr --prompt "Latour, ANT, STS"` (option `--max-files`) : `output.csv` compatible avec `rad_chunk.py` (`itemKey` stable `audio-<sha16>`, `title`, `date`, `filename` et `path` réduits au nom de fichier, `texteocr` avec marqueurs `<!-- Segment N (hh:mm:ss–hh:mm:ss) -->`, `texteocr_provider=albert_whisper`, `texteocr_partial`, segments réussis et totaux, `language`, `duration_s`, `segments`) ; à côté : `<base>_errors.json`, `<base>_audio_segments.json` (sous-segments horodatés sur l'enregistrement entier) et `albert_usage.jsonl`. Codes : 0 (au moins un fichier transcrit), 1 (aucun), 2 (refus ou erreur de compte, ligne `Albert abort:`).
- **Confidentialité** : Albert reçoit un nom neutre (`audio-<sha16>.wav`, `audio-<sha16>-segNNNN.mp3`), jamais le nom local, qui peut contenir le nom d'une personne interrogée ; aucun chemin local dans les sorties.
- **Reprise** : `<sortie>/albert_audio_checkpoints/<empreinte du fichier>/<empreinte des paramètres>/seg_NNNN.json` (segments réussis seulement, écriture atomique, chemin confiné) ; une relance ne renvoie ni ne redécoupe ce qui est déjà transcrit. Fichiers identiques transcrits une seule fois.
- **ffmpeg** : sans lui, seuls les mp3 et wav de 20 Mo au plus sont transcrits (un fichier long est alors envoyé entier, avec une note) ; image Docker : `INSTALL_FFMPEG=true` dans `.env` puis `docker compose up -d --build`. Segments plafonnés à 3 000 s (48 kb/s : moins de 20 Mo).
- Les étiquettes de locuteurs (format `diarized_json`) ne sont pas cohérentes d'un segment à l'autre : aucune identité de locuteur n'est supposée entre segments.

## 20. Usage du compte (sprint R2)

`GET /api/albert/usage?days=N` (1 à 90) : seaux journaliers UTC de `/v1/usage` (dates `start_time`/`end_time` toujours explicites : P24, 422 sans elles), totaux, coût dans l'**unité interne d'Albert** (aucune devise supposée) ; un champ absent reste inconnu. L'usage du compte couvre **tous** les usages de la clé, pas seulement RAGpy. L'interface affiche les jours en heure de Paris.

`ALBERT_LIMITER_REQUIRE_REDIS=1` (avec `ALBERT_LIMITER_BACKEND=redis`) : une panne Redis arrête les appels (erreur de quota explicite, `Albert abort: kind=account`) au lieu du seau local par processus, qui ne garantit pas le plafond du compte.

## 21. Évaluation humaine des rendus (sprint R2)

Outils dans `scripts/eval/albert/` (voir son `README.md`) : fiches CSV (UTF-8, « ; ») et pages HTML autonomes hors ligne, à l'aveugle (ordre à graine fixe, ni variante, ni score, ni rang, ni fournisseur), notation au clavier (`0` à `3`), jugements gardés dans le navigateur, export CSV, second jugement sur environ 20 % des éléments (kappa pondéré quadratique), puis `metrics.py` (moyennes, accord) et `rapport_decision.md` aux seuils fixés d'avance. L'évaluation de la variante de recherche a été retirée le 2026-10-03 avec les réponses sourcées.

| Fiche | Objet | Échelle | Décision (seuils fixés d'avance) |
|---|---|---|---|
| `ocr_sheets.py` | **D21** : faut-il recoder les textes LightOnOCR ? (page rendue à côté de sa transcription ; comparaison A/B facultative) | 0 illisible/faux · 1 nombreuses erreurs · 2 erreurs mineures · 3 fidèle | `ALBERT_OCR_SKIP_RECODE=1` seulement si moyenne ≥ 2,5, au plus 5 % de pages ≤ 1 et, en A/B, version recodée préférée dans moins de 60 % des paires |
| `audio_sheets.py` | Qualité des transcriptions Whisper (extraits audio intégrés, ffmpeg requis) | même échelle que l'OCR | transcription utilisable sans relecture si moyenne ≥ 2,5 et au plus 5 % d'extraits ≤ 1 |

Toute fiche incomplète ou modifiée après génération bloque la décision. Fiches et clés privées dans `data/albert_eval/<AAAA-MM-JJ>/` (exclu de git) ; l'export rempli s'appelle `fiche_*_rempli.csv`. Durées estimées (affichées par chaque script) : 30 s par page OCR (50 pages ≈ 30 min), 1,5 × la durée de l'extrait + 10 s pour l'audio.

```bash
.venv/bin/python scripts/eval/albert/ocr_sheets.py --pdf-dir uploads/<session> --csv uploads/<session>/output.csv --provider albert_lightonocr --out data/albert_eval/2026-10-02
```

```bash
.venv/bin/python scripts/eval/albert/audio_sheets.py --audio-dir sources/<projet>/audio --csv sources/<projet>/output.csv --out data/albert_eval/2026-10-02
```

```bash
.venv/bin/python scripts/eval/albert/metrics.py --dir data/albert_eval/2026-10-02
```

**Sondes R2** (`scripts/albert_probe.py`, fixtures `tests/fixtures/albert/P22_*` à `P26_*`, relevé du 2026-10-02) : P22 rerank, P23 chat streamé, P24 `/v1/usage`, P25 collections publiques et modèles rerank/audio, P26 transcription. Catalogue inchangé après l'échéance du 2026-10-01 (LightOnOCR, qwen3-coder, deepseek et qwen3-vl-embedding toujours listés ; à revérifier) :

```bash
ALBERT_LIVE=1 .venv/bin/python scripts/albert_probe.py --only P2,P22,P23,P24,P25,P26 --out data/albert_probe --diff-against tests/fixtures/albert
```

La recette live du parcours de recherche (`scripts/albert_rag_smoke.py`) a été retirée avec la fonction le 2026-10-03.

Tests hors ligne R2 : `tests/test_albert_client_r2.py`, `test_albert_corpora_routes.py`, `test_albert_review_r2.py`, `test_albert_policy.py`, `test_albert_ocr_checkpoint.py`, `test_albert_audio_routes.py`, `test_albert_audio_pipeline.py`, `test_albert_eval_tools.py`.
