# Sprint Albert : intégration de l'API Albert (DINUM) dans RAGpy

> **Statut** : spécification validée le 2026-09-26, exécutée lot par lot sur la branche `feat/albert`. Les `path:line` renvoient au commit de base `<sha_base>` ; on se repère par symbole. Lots 0 à 7 commités le 2026-09-26, lot 8 en cours le 2026-09-27 : voir « Journal d'exécution » en fin de document.

## Context

> **Environnement local** : les scripts console du `.venv` (`pip`, `pytest`, `uvicorn`, `celery`) pointent vers un ancien chemin du dépôt ; toujours utiliser `.venv/bin/python -m <outil>`. La suite de référence (`data/albert_baseline/baseline.xml`, commit `5e40aed`) donne 399 verts, 35 échecs et 2 erreurs dans 10 fichiers, tous attribués à la remise au vert du lot 2.

**Pourquoi ce chantier.**
- L'audit `.claude/audit_juridique.md` (§3.5, l. 215-229) classe les transferts vers les États-Unis comme non conformes depuis Schrems II : OpenAI pour les embeddings et le recodage, Pinecone pour le stockage. Aucun DPA ni aucune SCC n'est documenté (« ❌ Non documenté »). Seul Mistral (UE) est conforme.
- Albert (DINUM, OpenGateLLM 0.8.0 bêta) fournit une inférence souveraine. Il est hébergé chez Outscale SecNumCloud (tenant DINUM) et n'appelle aucun fournisseur externe (`.claude/albertai.md` l. 32 et §16).
- Le chemin d'inférence (chat, OCR, embeddings) ne conserve aucun contenu. Seules des métadonnées sont gardées 24 mois : compte, IP, tokens, code HTTP, latence. La DINUM est sous-traitante au sens de l'art. 28 du RGPD.
- **Exception** : les collections Albert **stockent** textes et métadonnées jusqu'à leur suppression. L'interface doit le dire et proposer la suppression.
- Albert couvre les quatre usages de RAGpy qui envoient aujourd'hui des données hors UE : LLM, OCR, embeddings et base vectorielle.

**Décisions utilisateur figées (non rediscutées).**
1. **Périmètre : les 4 capacités.**
   - (a) Chat comme 3e fournisseur : recodage GPT de `rad_chunk`, notes Zotero, fiches de lecture, pré-filtre et filtre de citations.
   - (b) OCR : LightOnOCR via `/v1/chat/completions` sur des pages rastérisées par PyMuPDF, et `/v1/ocr` `mistral-ocr-2512` si le compte y a accès.
   - (c) Embeddings `bge-m3` en 1024 d, dans un espace séparé, jamais mélangé avec `text-embedding-3-large` (3072 d).
   - (d) Collections Albert comme 4e cible vectorielle (`insert_to_albert`).
2. **Opt-in, désactivé par défaut.** Albert désactivé, tout reste identique à l'octet : sorties, clés de cache, lignes stdout, formes HTTP et SSE. C'est la même convention que `DEDUP_ENABLED=0`.
3. **Chaîne OCR** quand `OCR_ENABLE_ALBERT=1` : Albert (`/v1/ocr` si le compte y a accès, sinon LightOnOCR), puis Mistral, puis OpenAI (opt-in), puis Docling, puis legacy. Désactivé, la chaîne est inchangée.
4. **Git.**
   - Commit de la baseline dédup non commitée sur `main` (autorisé par l'utilisateur).
   - Puis branche `feat/albert`, avec un commit par lot validé.
   - Messages factuels, signés par l'utilisateur seul (`Amar LAKEL <amar@lakel.net>`, identité git vérifiée). Aucune ligne d'attribution, aucun co-auteur, aucune mention d'outil d'assistance, ni dans les messages ni dans les fichiers du dépôt (contrôlé par `tests/tools/check_attribution.py`).
   - Docstrings obligatoires sur toute nouvelle fonction (règle `CLAUDE.md:12`).
5. **Dette dans le périmètre.**
   - (i) Le `input()` exécuté à l'import de `rad_chunk`, le rechargement du `.env` admin par les sous-processus non-admin, les tests de référence cassés.
   - (ii) Le repli `.env` de `_get_llm_clients`.
   - (iii) Celery : `run_initial_phase`, `run_dense_phase` et `run_sparse_phase` absents, signature d'extraction fausse, routes sans authentification, identifiants lus dans l'env du worker.
   - (iv) Le littéral de clé Pinecone (`scripts/rad_vectordb.py:411`, présent depuis `d1f5875`, déjà poussé sur `origin/main`) et l'affichage `pinecone_api_key[:10]` (`:1242`). **L'utilisateur doit révoquer puis remplacer cette clé.**
6. **Sondes live autorisées** avec la clé de PRODUCTION : lecture, inférence, et une collection privée temporaire `ragpy-probe-*` supprimée à la fin.
   - `ALBERT_API_KEY` doit être ajoutée au `.env` avant le L0.
   - Les tests sont mockés par défaut (`httpx.MockTransport`).
   - La suite `albert_live` ne s'active qu'avec `ALBERT_LIVE=1` explicite dans le shell.
7. **Style.** Commandes shell sur une seule ligne. Plan rédigé en français.

**Faits vérifiés dans le dépôt (lecture seule).**

*Import et chargement de l'environnement*
- Pas de `scripts/__init__.py`. Motif d'import double en `rad_chunk.py:35-44`.
- `tests/__init__.py` existe. Aucun `conftest.py` nulle part. Aucun fichier de configuration pytest (ni `pytest.ini`, ni `pyproject.toml`, ni `setup.cfg`).
- `load_dotenv()` nus aux lignes `rad_chunk.py:84`, `rad_dataframe.py:165` et `:205`.
- `app/config.py:22` exécute `load_dotenv()` à l'import. `load_dotenv` sans `override` ne remplace jamais une variable déjà présente, même vide.
- Il n'existe pas de `scripts/.env`. Depuis un module de `scripts/`, `find_dotenv()` remonte au `.env` racine. En mode `python -c`, il part du répertoire courant.

*Chemin de recodage (`rad_chunk`)*
- `input()` à l'import en `rad_chunk.py:87-97`. La garde `client is None` est en `:1007`.
- `DEFAULT_DOC_WORKERS=3` (`rad_chunk.py:156`) et `DEFAULT_MAX_WORKERS` (`:153`) : les documents sont déjà traités en parallèle.
- Les sites de routage LLM par `"/" in` sont `rad_chunk.py:215/217`, `:312` (`"/" in eff_model`), `llm_note_generator.py:412`, `citation_filter.py:488`, `processing.py:458/999/1304` (`'/' in model`) et `citations.py:328/1063/1591` (`"/" in model_name`).

*Identifiants et configuration*
- `build_subprocess_env` est en `credentials.py:378-453`. Son mapping mêle secrets et configuration.
- `GET /get_credentials` renvoie **en clair** les valeurs du `.env` et de la base (`settings.py:92-126`). La superposition base parcourt tout `CREDENTIAL_ENV_MAPPING` (`:117-123`). Le masquage n'est fait que côté JS (`index.html:1608-1610`).
- `GET /users/me/credentials` utilise `response_model=UserCredentialsResponse`, dont tous les champs sont obligatoires (`users.py:162`, `schemas/user.py:160`).
- `DEFAULT_LLM_MODEL` est figé à l'import dans `pages.py:26`, `citations.py:26`, `citation_filter.py:54` et `parallel_citation_processor.py:51`.
- `DEDUP_META_FIELDS` est une variable d'env (`rad_dedup.py:177`, défaut `title`). `content_hash` et `content_id` sont en `rad_dedup.py:70/75`.

*Routes et sous-processus*
- Les routes lancent les scripts avec le littéral `"python3"` (`processing.py:318, 477, 600, 664, 738, 834, 1318, 1382, 1445, 1634`). En local, `python3` sur le PATH est un Python Homebrew sans les dépendances du projet.
- Délais des sous-processus : 1800 s (`processing.py:324, 484, 609, 670, 744, 1332, 1396, 1459`), 3600 s pour `/upload_db` (`:854`), 600 s pour le clustering SSE (`:1651`).
- `/upload_db` vérifie l'existence du fichier sparse (`processing.py:800-805`) **avant** d'examiner `db_choice`. La regex `^Dedup journal:\s*(\S+)` est en `:889`.
- Les routes SSE importent `run_subprocess_with_sse` localement (`processing.py:546, 1281, 1359, 1426, 1611`). Il faut donc patcher `app.utils.sse_helpers.run_subprocess_with_sse`.
- `celery_tasks.py:306-312` exige le fichier sparse avant la liste blanche des cibles. `is_celery_available()` pingue Redis (`:63`).

*Utilitaires LLM et OCR*
- Filtre de citations :
  - `citation_filter.py:585-597` et `:692-707` tiennent le sémaphore global autour de `run_in_executor` ;
  - le réessai `asyncio.sleep` se fait sémaphore tenu ;
  - `citations.py:1173` acquiert le même sémaphore avant d'appeler `filter_citation_with_llm` ;
  - `citation_filter.py:604` accepte ou rejette selon `'NA' not in upper`.
- `_generate_with_llm` fait `max_attempts=2` (`llm_note_generator.py:434`). `book_note_generator.py:883/1705/1727` avalent toutes les exceptions.
- `OCRResult` est un `NamedTuple` à 6 champs : text, provider, partial, pages_done, pages_total, error (`rad_dataframe.py:568-582`).

*Tests et environnement*
- `pytest-asyncio` est absent. Le plugin anyio 4.11 paramètre `anyio_backend` sur `("asyncio", "trio")`, et `trio` n'est pas installé.
- `ebooklib` manque dans le `.venv` alors qu'il est épinglé en `scripts/requirements.txt:23`.
- Versions : httpx 0.27.0, openai 1.50.2, Python 3.10.16 en local (3.11 dans Docker), redis installé.
- Le `lastfailed` actuel est incomplet. Par exemple, `tests/test_integration_api.py::test_get_credentials` appelle une route admin sans authentification.
- `UPLOAD_DIR` est fixé à `uploads/` dans le dépôt (`app/core/config.py:23`). `init_db` est en `app/database/init_db.py:104` et `create_access_token` en `app/core/security.py:54`.

*État git*
- Trois `.DS_Store` sont suivis (`.DS_Store`, `app/.DS_Store`, `.claude/.DS_Store`), dont deux déjà modifiés.
- `scripts/.env.example` (suivi) contient les gabarits `sk-proj-XXXX`, `sk-or-v1-XXXX` et `pcsk_XXXX`, comme `.env.example`.
- `.claude/albertai.md` (non suivi) contient des mentions d'outils tiers (l. 1, 9, 14 et 1186).

**Points à confirmer par l'utilisateur avant l'exécution** (aucun ne bloque la conception) :
- (a) `albert/…` avec Albert OFF renvoie un 400, au lieu d'un envoi à OpenRouter.
- (b) Politique de docstrings : les fonctions `test_*` sous `tests/` et les closures imbriquées en sont exemptées. Sinon, lancer `check_docstrings.py --strict`.
- (c) Le commit de baseline ajoutera environ 45 docstrings au travail de dédup.
- (d) `.claude/albertai.md` reste non suivi, parce qu'il contient des mentions d'outils tiers.
- (e) Retrait optionnel des `.DS_Store` de l'index (`git rm --cached`), une décision qui revient à l'utilisateur.
- (f) D21 : sauter ou non le recodage des textes LightOnOCR. Le défaut prudent est de recoder.
- (g) `ALBERT_SUBPROCESS_TIMEOUT=21600`.

## Décisions d'architecture

1. **Socle partagé : le paquet `scripts/rad_albert/`.**
   - Modules : `config`, `errors`, `retry`, `catalog`, `limiter`, `usage`, `client` et `preflight`, puis `ocr` (L4) et `collections` (L5).
   - Il est importable en `scripts.rad_albert` (app, Celery, tests) et en `rad_albert` (CLI), selon le motif de `rad_chunk.py:35-44`.
   - `__init__` ne réexporte que `config` et `errors` (stdlib seulement). Sur le chemin OFF, ni httpx ni openai ne sont chargés.
   - Les tests importent toujours `scripts.rad_albert`, et un test d'identité le vérifie.
2. **Deux modules stdlib indépendants du fournisseur.**
   - `scripts/rad_providers.py` : le résolveur LLM, `EmbeddingSpace`, `EmbeddingConfig`, `check_uniform_space` et `target_mismatch_message`.
   - `scripts/rad_env.py` : `load_dotenv_guarded`. Il est placé dans `scripts/` pour que `find_dotenv()` résolve le même fichier qu'aujourd'hui.
   - Routes et scripts en ont besoin même Albert OFF, et ces modules n'ont aucun effet de bord.
3. **Résolveur unique `resolve_llm_provider(model, *, albert_enabled) -> ProviderResolution(provider, wire_model, credential_key)`.**
   - Le préfixe `albert/` est testé en premier, sans tenir compte de la casse (`model[:7].lower()`), avant l'heuristique `/`.
   - La découpe se fait au premier `/` : `albert/openai/gpt-oss-120b` donne le wire `openai/gpt-oss-120b`.
   - Un wire vide lève `ValueError`. Si Albert est désactivé, `AlbertDisabledError(ValueError)` donne un 400 ou un événement SSE d'erreur.
   - `legacy_provider()` et `legacy_cache_provider_label()` répliquent exactement `rad_chunk.py:215/217/312`. Le résolveur remplace les 11 sites listés plus haut.
   - Deux fonctions complémentaires :
     - `rad_chunk.effective_recode_model(cli_model, cfg)` (L3) est l'unique source du modèle de recodage, pour le routage comme pour la clé de cache ;
     - `llm_note_generator.resolve_default_llm_model(openrouter_model=None)` (L2) est l'unique source du modèle par défaut côté web.
4. **Client HTTP unique en httpx**, sans SDK openai sur le chemin Albert.
   - `follow_redirects=False`. En-têtes : `Authorization` et `User-Agent` uniquement. Jamais de champ `user`.
   - `api_key` est obligatoire et jamais lu dans `os.environ`. Le transport est injectable (`MockTransport`).
5. **Une seule couche de retry**, dans `rad_albert/retry.py`.
   - `parse_retry_after` est porté de `rad_dataframe.py:1069-1105` (secondes et date HTTP), avec un plafond. `backoff_seconds` reprend `:1108-1127`, paramétré. La gigue ne s'applique qu'en l'absence de `Retry-After`. Le sommeil a lieu **hors** sémaphore.
   - Sur le chemin Albert, les boucles de réessai existantes sont **contournées** : `_generate_with_llm` `max_attempts=2`, `filter_citation_with_llm` `max_retries`, et le 2e passage séquentiel de `rad_chunk.py:275-284`.
   - Le chemin Mistral n'est pas touché. Un test de parité prouve la même sémantique.
6. **Catalogue = données** (`rad_albert/catalog.py`) : ids épinglés, alias, type, contexte, raisonnement, dimension, `deprecated_on`.
   - `fallback_chain(role, today)` filtre par date. Les embeddings n'ont jamais de repli.
   - Les alias sont résolus en ids par `/v1/models` au preflight, et c'est l'id résolu qui entre dans les clés de cache.
   - Les défauts sont cohérents avec `tests/fixtures/albert/decisions.json`, ce qu'un test vérifie.
7. **Limiteur proactif par rôle.**
   - Deux budgets : les RPM, et les tokens d'entrée par minute estimés à `len/3`, multipliés par `ALBERT_PROCESS_SHARE`. `pause(retry_after)` sur un 429.
   - En asynchrone, l'acquisition passe par `await asyncio.to_thread(limiter.acquire, …)` et le sommeil par `asyncio.sleep`. La boucle d'événements n'est jamais bloquée.
   - Le backend Redis est optionnel (`ALBERT_LIMITER_REDIS_URL`, sinon `CELERY_BROKER_URL`, sinon `redis://localhost:6379/0`).
8. **Ledger d'usage.** Il enregistre `response.model`, les tokens, `cost` et `impacts`. `albert_usage.jsonl` et une ligne de synthèse ne sont écrits **que si Albert a été appelé**.
9. **Identifiant unique `albert_api_key` → `ALBERT_API_KEY`**, toujours inscrit dans les 3 registres (`credentials.py:76-137`) pour que la purge non-admin (`:417-420`) le couvre.
   - Il est masqué des réponses JSON et des formulaires quand Albert est OFF.
   - `ALBERT_BASE_URL` est une configuration serveur uniquement.
10. **Fermeture du rechargement `.env`.**
    - Pour les non-admins, `build_subprocess_env` pose `RAGPY_DOTENV_DENY` : la liste **triée, séparée par des virgules, sans espace** des noms `*_API_KEY` retirés et non réinjectés.
    - `rad_env.load_dotenv_guarded()` retire ces noms après le chargement s'ils étaient absents avant. L'env admin ne change pas, et la configuration non secrète reste relue.
    - Ajout livré en W1 (L1) : les secrets serveur `SERVER_SECRET_ENV_VARS` (`FLOWER_PASSWORD`, `JWT_SECRET_KEY`, `RESEND_API_KEY`) sont aussi retirés de l'env des sous-processus non-admin. Ils ne sont jamais nécessaires aux scripts et n'entrent pas dans la liste DENY.
11. **Opt-in à deux niveaux.** `ALBERT_ENABLED` est l'interrupteur maître, complété d'un sélecteur par capacité.
    - **Bibliothèques** (`rad_albert`, `rad_providers`, `app/utils`, routes) : lecture à l'appel via `AlbertConfig.from_env()`.
    - **Scripts CLI** (`rad_dataframe`, `rad_chunk`, `rad_vectordb`) : lecture à l'import en constantes de module patchables, selon le motif existant. Le conftest les force à OFF.
    - L'interface est gardée par des blocs Jinja en ligne. `/api/albert/*` répond 404 `{"detail":"Not Found"}` quand Albert est OFF.
12. **OCR : le maillon Albert passe avant Mistral** (décision 3).
    - Il vit dans `rad_albert/ocr.py`, qui reçoit par injection les helpers de découpe, de parsing et de suppression.
    - La seule refacto du chemin Mistral est l'extraction pure du parseur `pages[]` (`rad_dataframe.py:1274-1332`), sous golden.
    - Une erreur de compte ou de quota est mémorisée pour tout le processus (`_ALBERT_ACCOUNT_DISABLED`).
13. **Espaces vectoriels séparés.**
    - Les champs `embedding_provider`, `embedding_model` et `embedding_dim` ne sont écrits que hors défaut (motif `rad_chunk.py:509-513`).
    - Le cache est cloisonné : clé `albert:bge-m3`, paramètres `{model, provider, dim, norm}`.
    - Jamais de vecteur nul côté Albert.
    - Les gardes de dimension sont dans les connecteurs, sans appel réseau supplémentaire sur le chemin par défaut.
    - Le Tier 3 de la dédup est désactivé pour bge-m3, sauf seuil explicite.
14. **Collections : la 4e cible.** CLI `--db albert` dans `rad_vectordb` et branche dédiée dans `/upload_db` et dans Celery. Les helpers purs sont dans `rad_albert/collections.py`, y compris `resolve_albert_input`.
    - L'idempotence est toujours active. Le content_id est stable même sans champs dédup.
    - La collection est forcée en privé.
    - Les métadonnées passent par une liste blanche de 10 champs au plus. `content_id` et `DEDUP_META_FIELDS` y sont imposés, `path` n'est jamais envoyé.
    - Rollback par document. Manifeste écrit par tranche, puis copié hors `uploads/`.
    - Un acquittement RGPD est exigé (interface, backend, CLI) et journalisé dans l'audit.
15. **Web async.** `run_llm_slot(semaphore, thunk, *, albert_policy=None)` est utilisé par les notes, les fiches de livre **et les deux sites du filtre de citations**.
    - Pour OpenAI et OpenRouter, c'est strictement le code actuel.
    - Pour Albert, dans l'ordre :
      1. acquisition du limiteur hors sémaphore ;
      2. un seul essai (`single_attempt`) sous le sémaphore global, puis sous le sémaphore Albert (`ALBERT_NOTES_CONCURRENCY`), toujours acquis dans le même ordre ;
      3. sommeil hors sémaphores, puis nouvel essai.
16. **Celery = exécuteur de sous-processus.** Il lance les mêmes CLI que les routes, avec `build_subprocess_env(utilisateur chargé par user_id)`. Les routes sont authentifiées, aucun secret ne passe par Redis, et `autoretry` est limité aux erreurs d'infrastructure.
17. **Byte-identité OFF prouvée.**
    - Les goldens G1 à G13 sont capturés au L0 sur le code de base, avec des identifiants **factices**, et ne sont jamais régénérés.
    - Leurs sorties sont restreintes par des **listes littérales** : les 6 champs d'`OCRResult`, les 15 noms d'env de la baseline, le bloc `=== Result ===` seul.
18. **Git.** Baseline sur `main` avec une liste explicite, puis `feat/albert`, puis un commit par lot. Seul l'coordinateur exécute git. Chaque message est contrôlé **avant** le commit. Jamais de push ni de merge sans demande.
19. **Délai des sous-processus Albert.** `ALBERT_SUBPROCESS_TIMEOUT` (21600 s) ne s'applique que si la requête sélectionne Albert :
    - modèle `albert/…` ;
    - `embedding_provider=albert` ;
    - `db_choice=albert` ;
    - OCR Albert actif.
    Sinon, le kwarg `timeout` est identique à aujourd'hui (capturé par G7). La même règle vaut dans `runner.run_script` sous Celery.
20. **Clé Albert jamais renvoyée en clair.** `/get_credentials` renvoie `••••` suivi des 4 derniers caractères, et `/save_credentials` ignore une valeur égale au masque. Quand Albert est OFF, la superposition base est filtrée par `_admin_form_env_keys()`.
21. **Portée des tests : `conftest.py` à la racine du dépôt.** Ce n'est pas un fichier de configuration pytest, et il couvre `tests/`, `scripts/` et `app/`.
    - `ALBERT_LIVE` est capturé à l'import du conftest. Il est refusé s'il figure dans `.env`.
    - Au `pytest_configure`, hors live : `ALBERT_ENABLED=0`, `OCR_ENABLE_ALBERT=0`, `ALBERT_API_KEY=''` et `EMBEDDING_PROVIDER=''`. Comme `load_dotenv` n'écrase pas, la clé de production n'entre jamais dans la session de test.
    - Une garde réseau est installée pour la session.
    - `anyio_backend` vaut `asyncio`.
22. **Erreurs de compte : arrêt du job entier.** `AlbertAuthError` et `AlbertQuotaExhausted` :
    - sont mémorisées pour le processus (OCR) ;
    - arrêtent le recodage (`exit 2`) ;
    - sont relancées par les 3 phases des fiches de livre et par `process_citations_parallel` ;
    - remontent aux routes sous forme d'un événement SSE `{type:error, credential_required}`.

### Arbitrages entre les trois conceptions

| Sujet | Options | Retenu | Pourquoi (une ligne) |
|---|---|---|---|
| Emplacement du socle | paquet `rad_albert/` (archi) ; modules plats (risque, livraison) | paquet, plus 2 modules plats stdlib | sous-modules isolés ; un `__init__` léger garantit zéro httpx quand OFF |
| SDK openai ou httpx | SDK + httpx (archi) ; httpx seul (risque, livraison) | httpx seul | une seule boucle de retry et de classification, MockTransport natif |
| `ALBERT_BASE_URL` | identifiant utilisateur en liste blanche (livraison) ; config serveur (archi, risque) | config serveur | ferme la classe SSRF sans ajouter de champ |
| Hôte `api.albert…` | rejet (archi) ; réécriture (risque, livraison) | réécriture + WARNING | hôte équivalent documenté par un guide officiel ; URL réservée à l'admin |
| Fuite dotenv | `RAGPY_ENV_BLOCKLIST` / `RAGPY_STRIPPED_ENV` / `RAGPY_DOTENV_DENY` ; toutes clés ou secrets seuls | `RAGPY_DOTENV_DENY`, secrets `*_API_KEY` seuls, format trié et séparé par des virgules | ferme la fuite, la config reste relue, un seul mécanisme couvert par un test AST |
| `input()` de rad_chunk | invite dans `__main__` sur TTY (archi) ; suppression (livraison) | suppression : message et `exit 1` | aucun chemin bloquant caché |
| Celery | admin seul et wrappers en processus (livraison) ; sous-processus par utilisateur (archi, risque) | sous-processus par utilisateur | corrige les 4 dettes, zéro secret dans Redis |
| Limiteur | local + Redis (archi) ; SQLite partagé (risque) ; local (livraison) | local + Redis optionnel, acquisition async via `to_thread` | pas de verrou SQLite sur Google Drive ; boucle asyncio jamais bloquée |
| Cible collections | `--db albert` (archi, livraison) ; module et route séparés (risque) | `--db albert` + helpers séparés + ack RGPD | un seul contrat stdout et un seul flux UI |
| OCR | dans `rad_dataframe` (archi, livraison) ; module séparé (risque) | `rad_albert/ocr.py` + insertion dans la chaîne | édition minimale du fichier chaud, sans import circulaire |
| `albert/` avec Albert OFF | 400 (archi, livraison) ; routage historique (risque) | 400 (y compris `upload_pop_json`) | l'entrée échoue déjà aujourd'hui après envoi du texte ; seule dérogation, à confirmer |
| Sondes | L0 (livraison) ; lot dédié avant ou après le client | L0, script httpx autonome + `decisions.json` | fixtures et défauts disponibles dès le gel des contrats |
| Tests async existants | `asyncio.run` ; `@pytest.mark.anyio` | anyio + fixture `anyio_backend='asyncio'` ; `asyncio.run` pour le neuf | marqueur minimal, sans variante trio |
| Plafond Retry-After | 60 / 120 / 900 s | 120 s ; au-delà, `AlbertQuotaExhausted` | le backoff ne résout pas un plafond journalier |
| Embeddings manquants | ratio 0.0 / 0.02 / booléen | `ALBERT_EMBED_MAX_MISSING_RATIO=0.0` | jamais d'index silencieusement incomplet |
| Repli OCR | silencieux ou tracé ; mode STRICT | tracé (`OCRResult.fallback_from` + `OCR_PROVIDER_FALLBACK`) | décision 3 respectée, souveraineté auditée |
| Détection des régressions | `comm` / `regress_check` / JUnit | `tests/tools/junit_diff.py` (failure + error + tests disparus) | attribution claire des nouveaux échecs |
| Délai des sous-processus | inchangé ; global ; propre à Albert | propre à Albert (`ALBERT_SUBPROCESS_TIMEOUT`) | débit Albert plus bas ; chemin OFF identique |
| Clé Albert dans `/get_credentials` | en clair (comme les autres) ; masquée côté serveur | masquée côté serveur | clé de production soumise à quota, valable un an |
| Portée du conftest | `tests/conftest.py` ; racine | racine | GC3 lance aussi `scripts/` et `app/` |
| Recodage des textes LightOnOCR | sauter ; recoder | recoder par défaut (`ALBERT_OCR_SKIP_RECODE=0`) | qualité non évaluée ; D21 à confirmer |

### Drapeaux de configuration (tous OFF par défaut)

| Variable | Défaut | Portée | Sens |
|---|---|---|---|
| `ALBERT_ENABLED` | `0` | serveur | Interrupteur maître. À 0 : ni appel, ni route, ni UI, ni clé JSON Albert |
| `ALBERT_API_KEY` | vide | identifiant `albert_api_key` (repli `.env` pour les admins seulement) | Clé Bearer. Retirée et bloquée pour les non-admins ; jamais lue par la bibliothèque ; masquée dans `/get_credentials` |
| `ALBERT_BASE_URL` | `https://albert.api.etalab.gouv.fr/v1` | serveur | https, `/v1` garanti, hôte en liste blanche ; vide = défaut |
| `ALBERT_ALLOW_CUSTOM_HOST` | `0` | serveur | Autorise un OpenGateLLM auto-hébergé |
| `OCR_ENABLE_ALBERT` | `0` | serveur | Maillon Albert avant Mistral (exige `ALBERT_ENABLED=1` et une clé) |
| `ALBERT_OCR_MODE` | `auto` (D3) | serveur | `auto` = `/v1/ocr` si l'accès est mémorisé, sinon LightOnOCR ; `chat` ; `ocr` |
| `ALBERT_OCR_CHAT_MODEL` / `ALBERT_OCR_DOC_MODEL` | `lightonocr-2-1b` / `mistral-ocr-2512` | serveur | Ids épinglés |
| `ALBERT_OCR_DPI` / `_MAX_SIDE` / `_MAX_TOKENS` / `_TEMPERATURE` / `_TOP_P` | 200 / 1540 / 4096 / 0.2 / 0.9 | serveur | Rastérisation et chat §9.1 |
| `ALBERT_OCR_MAX_PAGES` / `ALBERT_OCR_MAX_FAILED_RATIO` | 0 / 0.5 | serveur | Plafond (au-delà : partial) et seuil d'abandon du maillon |
| `ALBERT_OCR_PART_MB` / `_PART_PAGES` / `ALBERT_OCR_CONCURRENCY` | 15 / 100 (D4) / 2 (D5) | serveur | Découpage base64 de `/v1/ocr` et parallélisme HTTP |
| `ALBERT_OCR_SKIP_RECODE` | `0` (D21) | serveur | 1 = texte `albert_lightonocr` exclu du recodage |
| `EMBEDDING_PROVIDER` | `openai` | serveur ; champ par requête et `--embedding-provider` | `albert` = `bge-m3`, 1024 d ; `albert` avec `ALBERT_ENABLED=0` : `exit 2` explicite |
| `ALBERT_EMBED_MODEL` / `_BATCH` / `_MAX_MISSING_RATIO` / `_L2_NORMALIZE` | `bge-m3` / 64 / 0.0 / 1 (D12) | serveur | Lot plafonné à 64 ; tout manque donne `exit 1` ; normalisation enregistrée dans l'espace |
| `ALBERT_REASONING_EFFORT` / `_HEADROOM` | `medium` / 2048 | serveur | gpt-oss uniquement ; le filtre de citations force `low` |
| `ALBERT_MODEL_FALLBACK` / `ALBERT_BUSY_RETRIES` | `1` / 2 | serveur | Repli de rôle après N réponses 503 ; jamais sur 404, jamais pour les embeddings |
| `ALBERT_RECODE_RPM` / `_NOTES_RPM` / `_OCR_RPM` / `_EMBED_RPM` / `ALBERT_CHAT_TPM` | D1 (repli 60 / 30 / 30 / 600 / 200000) | serveur | Limiteur proactif, environ 90 % du quota × part |
| `ALBERT_PROCESS_SHARE` / `ALBERT_LIMITER_BACKEND` / `ALBERT_LIMITER_REDIS_URL` | 1.0 / `local` / vide (repli `CELERY_BROKER_URL`) | serveur | `redis` recommandé avec Celery ou plusieurs sessions |
| `ALBERT_RECODE_CONCURRENCY` / `ALBERT_NOTES_CONCURRENCY` | 4 / 4 (D22) | serveur | Plafonds par processus : sémaphore de module dans `rad_chunk`, sémaphore async Albert dans `run_llm_slot` |
| `ALBERT_EMBED_CONCURRENCY` / `ALBERT_PUSH_CONCURRENCY` | 4 (D22) / 1 (D19) | serveur | Parallélisme des embeddings et des envois de chunks |
| `ALBERT_MAX_RETRIES` / `_RETRY_BACKOFF` / `_RETRY_MAX_BACKOFF` / `ALBERT_RETRY_AFTER_MAX` | 4 / 2.0 / 60 / 120 | serveur | Retry-After au-delà de 120 s : quota épuisé, abandon |
| `ALBERT_TIMEOUT_CHAT` / `_NOTES` / `_OCR_PAGE` / `_OCR_DOC` / `_EMBED` / `_COLLECTIONS` | 120 / 300 / 120 / 300 (D4) / 60 / 120 | serveur | Délais HTTP par type d'appel |
| `ALBERT_SUBPROCESS_TIMEOUT` | 21600 | serveur | Délai des sous-processus quand la requête sélectionne Albert |
| `ALBERT_METADATA_FIELDS` | `content_id,content_hash,chunk_index,total_chunks,title,authors,year,doi,item_key,filename` | serveur | 10 au plus ; doit contenir `content_id`, `content_hash`, `chunk_index` et `DEDUP_META_FIELDS` (sinon refus au démarrage) ; jamais `path` |
| `DEDUP_SIM_THRESHOLD_BGE_M3` | vide | serveur | Vide : Tier 3 sauté pour bge-m3 (WARNING unique) ; défini : seuil appliqué |
| `ALBERT_USAGE_LOG` / `ALBERT_PREFLIGHT` / `ALBERT_TOKEN_ESTIMATOR` | 1 / 1 / `chars` | serveur | Effectifs seulement si Albert est réellement utilisé |
| `model=albert/<id>`, `db_choice=albert` (+ `albert_collection_id`, `_name`, `albert_create_collection`, `albert_gdpr_ack`), `embedding_provider` | aucun | par requête | Sélection explicite ; 400 si Albert est OFF |
| `ALBERT_LIVE` | absent | shell uniquement, **jamais** dans `.env` ni dans les `.env.example` | Active `-m albert_live` (valeur capturée à l'import du conftest) |
| `RAGPY_DOTENV_DENY` | absent | interne, pour les non-admins | Noms triés, séparés par des virgules, que le `.env` ne doit pas recharger |
| `RAGPY_UPDATE_GOLDEN` | absent | tests, L0 seulement | Régénère les goldens (interdit après L0) |

### Taxonomie des erreurs (`rad_albert/errors.py`, classification sur le statut HTTP)

| Statut / condition | Classe | Retry ? | Effet |
|---|---|---|---|
| 400 contenant `InsufficientBudget` | `AlbertAuthError(budget_exhausted)` | non | Arrêt du job (décision 22), message « budget Albert épuisé » |
| 400 autre | `AlbertPermanentError(bad_request)` | non | Extrait du corps journalisé (300 caractères, clé masquée) |
| 401 | `AlbertAuthError(invalid_key)` | non | `credential_required=albert_api_key`, arrêt du job |
| 403 contenant `account has expired` | `AlbertAuthError(account_expired)` | non | « compte expiré, écrire à albert.api@numerique.gouv.fr », arrêt du job |
| 403 autre | `AlbertPermanentError(no_access)` | non | OCR : absence d'accès à `/v1/ocr` mémorisée, bascule sur LightOnOCR ; collections : « droit d'écriture manquant » (D1) |
| 404 | `AlbertPermanentError(not_found)` | non (≠ Mistral `:1054`) | Message renvoyant à `/v1/models`, pas de repli silencieux |
| 408, 500, 502, 504, réseau (Connect, Read/Write/PoolTimeout, RemoteProtocol) | `AlbertTransientError` | oui, 4 au plus, backoff plafonné à 60 s avec gigue | — |
| 409 / 413 / 422 (dont `Wrong model type`) | `AlbertPermanentError` | non (413 : un seul redécoupage OCR) | Erreur de configuration |
| 429, Retry-After ≤ 120 s | `AlbertTransientError(retry_after)` | oui, en respectant le délai | `limiter.pause()` |
| 429 persistant ou Retry-After > 120 s | `AlbertQuotaExhausted` (sous-classe d'Auth) | non | « quota journalier atteint », arrêt du job |
| 503 « Model is too busy » | `AlbertModelBusy` | 2 essais courts, puis repli de rôle | Modèle servi journalisé, puis utilisé dans la clé de cache |
| `finish_reason=length` avec contenu vide | `AlbertTruncatedError` | notes : 1 réessai (budget ×1.5, effort `low`) ; recodage : `fallback_truncated` | Jamais traité comme une réponse vide valide |

### Modèles par rôle (épinglés ; `today` injecté dans `fallback_chain`)

| Rôle | Primaire (alias) | Chaîne de repli | Échéances |
|---|---|---|---|
| `recode`, `citation` | `ministral-3-8b-instruct-2512` (`openweight-small`, 262 144 tokens, T 0.1) | `mistral-small-3-2-24b-instruct-2506` jusqu'au 2026-11-30, puis `gemma-4-31b-it` | mistral-small retiré le **2026-12-01** |
| `notes` (extended, pedagogique, evaluation, synthèse de livre) | `gpt-oss-120b` (`openweight-large`, 131 072 tokens, raisonnement, +2048, effort `medium`) | `ministral-3-8b-instruct-2512` | quota 50 RPM / 5 000 RPD |
| `book_structure` (phase 1 JSON), `long_context` | `ministral-3-8b-instruct-2512` | mistral-small, puis gemma (selon la date) | idem |
| `ocr_doc` | `mistral-ocr-2512` (`/v1/ocr`, accès restreint) | `lightonocr-2-1b` | — |
| `ocr_chat` | `lightonocr-2-1b` (`openweight-ocr`, 16 384 tokens) | aucun repli Albert ; maillon Mistral ensuite | fin de phase de test le **2026-10-01** (statut incohérent) |
| `embed` | `bge-m3` (`openweight-embeddings`, 1024 d, lots de 64) | **aucun, jamais** | — |

Exclus des défauts et des replis :
- `qwen3-coder-30b-a3b-instruct` (retiré le 2026-10-01) ;
- `deepseek-v4-flash` (fin de test le 2026-10-01, risque de censure) ;
- `qwen3-vl-embedding-8b` (4096 d, fin de test le 2026-10-01) ;
- `mistral-medium-2508` (accès restreint).

## Invariants et leurs preuves

| # | Invariant | Test qui le prouve | Lot |
|---|---|---|---|
| 1 | Albert OFF : sorties identiques à l'octet. Couvre le texte OCR et les 6 champs d'`OCRResult`, les logs OCR, le JSON des chunks, le stdout des phases, le fichier dense et les kwargs, les clés de cache, le bloc Result, l'env mappé, l'argv, le kwarg `timeout`, les 403 et le SSE des routes (pipeline, citations, clustering compris), le JSON d'identifiants et le HTML | `tests/test_albert_off_golden.py::test_g1…test_g13`, rejoués à chaque gate | L0 (toutes) |
| 2 | OFF : aucun appel réseau Albert, même avec une clé dans l'env, sur les 3 racines de tests | `tests/test_albert_test_infra.py::test_network_guard_blocks_and_records_etalab` ; `scripts/test_albert_guard_scope.py::test_network_guard_active_outside_tests_dir` ; `tests/test_albert_ocr.py::test_off_no_albert_call_even_with_key` | L1, L4 |
| 3 | OFF : `/api/albert/*` en 404, aucune clé albert dans le JSON (même si l'admin en a une en base), HTML inchangé | `tests/test_albert_credentials.py::test_status_404_when_off`, `::test_me_credentials_off_equals_golden`, `::test_get_credentials_off_ignores_db_albert_key` ; `tests/test_albert_ui.py::test_off_html_equals_golden` | L1, L7 |
| 4 | Toute chaîne sans `albert/` est routée comme aujourd'hui | `tests/test_rad_providers.py::test_resolver_matches_legacy_matrix` (30 chaînes ou plus) ; `tests/test_albert_recode.py::test_off_keys_match_golden` | L1, L3 |
| 5 | `albert/…` avec Albert OFF : 400, sans sous-processus ni envoi (seule dérogation) | `tests/test_rad_providers.py::test_albert_prefix_disabled_raises` ; `tests/test_albert_routes.py::test_albert_prefix_disabled_400_no_subprocess`, `::test_upload_pop_json_albert_off_400` | L1, L7 |
| 6 | Un non-admin n'obtient jamais les clés du `.env` admin (construction de l'env et rechargement dotenv) | `tests/test_albert_credentials.py::test_nonadmin_env_strips_albert_key_and_sets_deny` ; `tests/test_env_isolation.py::test_subprocess_cannot_refill_secrets_from_dotenv` | L1, L2 |
| 7 | La garde dotenv ne touche que les secrets refusés ; la config est relue ; l'admin est inchangé | `tests/test_env_isolation.py::test_guard_is_plain_load_dotenv_without_marker`, `::test_config_vars_still_reloaded` | L2 |
| 8 | Plus aucun `load_dotenv()` nu dans les scripts lancés en sous-processus | `tests/test_env_isolation.py::test_no_bare_load_dotenv_in_subprocess_scripts` (AST) | L2 |
| 9 | `_get_llm_clients` n'a plus de repli `.env` ; tous ses appelants passent des clés explicites | `tests/test_debt_regressions.py::test_get_llm_clients_has_no_env_fallback`, `::test_get_llm_clients_callers_pass_explicit_keys` | L2 |
| 10 | L'import de `rad_chunk` ne demande jamais de saisie ; stdin des sous-processus = DEVNULL | `tests/test_debt_regressions.py::test_import_without_openai_key_never_calls_input`, `::test_route_subprocesses_use_devnull_stdin` | L2 |
| 11 | La bibliothèque ne lit jamais la clé dans l'env ; la clé est obligatoire | `tests/test_albert_client.py::test_constructor_requires_key_never_reads_environ` + grep de gate | L1 |
| 12 | Clé jamais dans une URL, un corps de **réponse**, un log, sur stdout ni dans le HTML | `tests/test_albert_client.py::test_key_only_in_authorization_header`, `::test_errors_and_logs_redact_key` ; `tests/test_debt_regressions.py::test_vectordb_cli_stdout_has_no_key` ; `tests/test_albert_credentials.py::test_status_no_query_api_key_and_no_key_in_response`, `::test_get_credentials_masks_albert_key` ; `tests/test_albert_ui.py::test_key_never_rendered` | L1, L2, L7 |
| 13 | SSRF : URL serveur seulement, https, liste blanche, pas de redirection | `tests/test_albert_config.py::test_normalise_*` ; `tests/test_albert_client.py::test_no_redirect_follow` ; `tests/test_albert_credentials.py::test_base_url_not_a_user_credential` | L1 |
| 14 | Littéral Pinecone absent du dépôt | `tests/test_debt_regressions.py::test_no_secret_literals_in_repo` (parcourt `git ls-files`, ignore les gabarits en X) | L2 |
| 15 | Celery : authentification, accès à la session, aucun secret dans Redis, env par utilisateur, pas de retry sur les erreurs déterministes | `tests/test_celery_tasks.py::test_every_route_requires_auth`, `::test_foreign_session_denied`, `::test_delay_payload_has_no_secret`, `::test_task_env_is_per_user`, `::test_nonzero_exit_not_retried` ; `tests/test_albert_routes.py::test_celery_albert_gated_and_never_autoretried` | L2, L7 |
| 16 | Erreurs classées sur le statut ; erreurs de compte jamais retentées ; 404 permanent ; Retry-After (secondes et date HTTP) plafonné ; quota épuisé = abandon | `tests/test_albert_client.py::test_classification_matrix`, `::test_permanent_single_attempt`, `::test_retry_after_seconds_and_http_date_capped`, `::test_quota_exhausted_abort` | L1 |
| 17 | Backoff hors sémaphore (code synchrone et web, filtre de citations compris) | `tests/test_albert_client.py::test_sleep_outside_semaphore` ; `tests/test_albert_notes.py::test_semaphore_released_during_backoff` ; `tests/test_albert_citation_filter.py::test_semaphore_released_during_backoff` ; `tests/test_albert_ocr.py::test_semaphore_released_during_backoff` | L1, L3, L4 |
| 18 | Quotas : buckets RPM et TPM, pause sur 429, part par processus, backend Redis, boucle asyncio jamais bloquée | `tests/test_albert_limiter.py::test_bucket_rpm_tpm_fake_clock`, `::test_pause_on_429`, `::test_process_share`, `::test_redis_window_fake_redis`, `::test_limiter_does_not_block_event_loop` | L1 |
| 19 | Souveraineté : aucun repli silencieux vers OpenAI ou OpenRouter | `tests/test_albert_recode.py::test_no_silent_openai_fallback` ; `tests/test_albert_notes.py::test_final_failure_raises_no_gpt4o_mini` ; `tests/test_albert_citation_filter.py::test_albert_never_calls_openai` ; `tests/test_albert_embeddings.py::test_albert_never_calls_openai_embeddings` | L3, L6 |
| 20 | Chaîne OCR : Albert d'abord quand ON ; repli tracé, jamais silencieux | `tests/test_albert_ocr.py::test_on_albert_runs_before_mistral`, `::test_albert_fail_then_mistral_recorded_fallback` | L4 |
| 21 | `/v1/ocr` en pages indexées à partir de 0 ; marqueurs N=index+1 continus ; marqueur d'échec ≠ `PAGE_MARKER_RE` | `tests/test_albert_ocr.py::test_doc_ocr_pages_zero_based`, `::test_page_markers_continuous_with_failed_page` | L4 |
| 22 | Chemin Mistral identique après l'extraction du parseur ; `OCRResult` jamais déballé par position | `tests/test_albert_ocr.py::test_mistral_payload_parser_bytes_unchanged`, `::test_ocrresult_never_unpacked_positionally` + G1 + `TestSplitPartialSalvage` | L4 |
| 23 | Cache jamais pollué : repli, tronqué ou suspect jamais mis en cache ; clé Albert = provider `albert` + id résolu ; clés OpenAI inchangées ; routage et clé toujours cohérents | `tests/test_albert_recode.py::test_length_is_fallback_truncated_not_cached`, `::test_suspicious_ratio_not_cached`, `::test_cache_key_uses_albert_provider_and_resolved_id`, `::test_harden_cli_albert_routes_and_keys_consistent` ; G4 | L3 |
| 24 | gpt-oss : marge de raisonnement ; vide avec `length` ≠ réponse valide ; le raisonnement n'est jamais le contenu | `tests/test_albert_client.py::test_chat_reasoning_headroom`, `::test_truncated_empty_raises` ; `tests/test_albert_recode.py::test_reasoning_content_never_used` ; `tests/test_albert_citation_filter.py::test_none_content_no_attributeerror` | L1, L3 |
| 25 | Contrat embeddings : 64 au plus, tri par index, pas de `dimensions`, pas de `''` | `tests/test_albert_client.py::test_embed_slices_64_sorts_checks_dim_rejects_empty` ; `tests/test_albert_embeddings.py::test_albert_130_texts_three_posts` | L1, L6 |
| 26 | Espaces jamais mélangés : champs hors défaut seulement, pas de zéros Albert, HIT de mauvaise dimension = MISS, gardes avant écriture, rebuild, clustering | `tests/test_albert_embeddings.py::test_openai_request_and_output_identical`, `::test_albert_never_writes_zero_vectors`, `::test_wrong_dim_hit_is_miss` ; `tests/test_vector_space_guards.py::test_pinecone_dim_mismatch_before_upsert`, `::test_qdrant_existing_size_mismatch`, `::test_weaviate_fetch_only_non_default`, `::test_mixed_file_refused_offline`, `::test_rebuild_preflight_exit_2`, `::test_clustering_mixed_dims_clear_error` | L6 |
| 27 | Collections : privée forcée, 64 chunks au plus par POST, 10 scalaires au plus, jamais `path`, multipart, relance idempotente (même sans champs dédup), rollback, aucun vecteur envoyé | `tests/test_albert_vectordb.py::test_public_refused_private_forced`, `::test_chunks_sliced_64`, `::test_sanitize_rules`, `::test_document_multipart_no_metadata`, `::test_rerun_idempotent_zero_post`, `::test_rerun_idempotent_without_dedup_fields`, `::test_metadata_override_must_keep_content_id`, `::test_slice_failure_deletes_new_document`, `::test_no_vectors_in_payloads` ; écritures à l'issue incertaine jamais rejouées : `tests/test_albert_client_writes.py::test_create_collection_never_resent_when_outcome_unknown`, `::test_create_document_never_resent_on_500`, `::test_add_chunks_never_resent_on_502` | L1, L5 |
| 28 | Acquittement RGPD (UI, backend, CLI) journalisé dans l'audit ; manifeste écrit par tranche et copié hors `uploads/` ; action de suppression | `tests/test_albert_vectordb.py::test_cli_requires_ack_retention`, `::test_manifest_written_per_slice` ; `tests/test_albert_routes.py::test_upload_db_albert_requires_gdpr_ack`, `::test_upload_db_albert_audit_and_manifest_copy` ; `tests/test_albert_ui.py::test_on_gdpr_confirm_present` ; `tests/test_albert_settings_routes.py::test_delete_collection` | L5, L7 |
| 29 | Bloc Result inchangé ; lignes Albert propres à albert | `tests/test_albert_vectordb.py::test_stdout_extra_lines_albert_only` + G5 | L5 |
| 30 | Cycle de vie : modèles dépréciés retirés des replis après leur date ; exclus jamais par défaut ; id épinglé disparu = erreur explicite | `tests/test_albert_catalog.py::test_fallback_chain_drops_mistral_small_after_2026_12_01`, `::test_excluded_never_default`, `::test_resolve_alias_to_id`, `::test_embeddings_have_no_fallback` (livrés dans `test_albert_catalog.py` et non `test_albert_config.py`) ; `tests/test_albert_recode.py::test_404_permanent_abort_message` | L1, L3 |
| 31 | Reproductibilité : `response.model` journalisé ; ledger écrit seulement si Albert est appelé ; `recode_status`/`recode_model` toujours écrits pour Albert | `tests/test_albert_client.py::test_ledger_records_response_model_cost` ; `tests/test_albert_embeddings.py::test_no_usage_file_when_off` ; `tests/test_albert_recode.py::test_albert_run_always_emits_recode_model` | L1, L3, L6 |
| 32 | Les tests live ne s'activent jamais sur la seule présence de la clé ; `ALBERT_LIVE` absent des fichiers env | `tests/test_albert_test_infra.py::test_live_skipped_without_ALBERT_LIVE`, `::test_live_refused_if_env_file_has_ALBERT_LIVE` ; `tests/test_albert_docs_sync.py::test_albert_live_absent_from_env_examples` | L1, L8 |
| 33 | Toute variable `ALBERT_*` lue par le code est documentée, OFF par défaut | `tests/test_albert_docs_sync.py::test_every_registered_var_in_env_example` | L8 |
| 34 | Import OFF léger (ni httpx ni openai via `rad_albert`) | `tests/test_albert_config.py::test_package_import_is_light` | L1 |
| 35 | Docstrings sur les nouvelles fonctions (politique figée) | GC7 ; `tests/test_gate_tools.py::test_check_docstrings_policy` | tous |
| 36 | Filtre de citations Albert : aucune acquisition imbriquée du sémaphore | `tests/test_albert_citation_filter.py::test_no_nested_acquire_deadlock` ; `tests/test_albert_routes.py::test_citations_albert_no_outer_semaphore` | L3, L7 |
| 37 | Une seule couche de retry sur le chemin Albert | `tests/test_albert_notes.py::test_single_retry_layer_notes` ; `tests/test_albert_citation_filter.py::test_single_retry_layer_citations` ; `tests/test_albert_recode.py::test_single_retry_layer_recode` | L3 |
| 38 | Erreur de compte ou de quota = arrêt du job entier (OCR mémorisé, recodage, fiches, citations) | `tests/test_albert_ocr.py::test_account_error_memoized_skips_next_documents` ; `tests/test_albert_recode.py::test_account_error_abort_exit_2` ; `tests/test_albert_notes.py::test_book_account_error_propagates` ; `tests/test_albert_citation_filter.py::test_account_error_aborts_job` | L3, L4 |
| 39 | Délai de sous-processus étendu seulement quand Albert est sélectionné | G7 (kwarg `timeout`) ; `tests/test_albert_routes.py::test_albert_timeout_only_when_selected` | L7 |
| 40 | Concurrence de recodage plafonnée par processus | `tests/test_albert_recode.py::test_recode_concurrency_is_process_wide` | L3 |
| 41 | Tier 3 de la dédup sauté pour bge-m3 sans seuil explicite | `tests/test_vector_space_guards.py::test_tier3_disabled_for_bge_m3_by_default` | L6 |
| 42 | Aucun test paramétré en trio ; `anyio_backend` = asyncio | `tests/test_albert_test_infra.py::test_anyio_backend_is_asyncio` + hook de collection | L1 |
| 43 | Aucune valeur secrète réelle dans les goldens, les fixtures, le diff indexé ni les XML JUnit | GC6 ; `tests/test_albert_probe_sanitize.py` ; `tests/test_albert_off_golden.py::test_golden_harness_uses_fake_credentials` | tous |
| 44 | Aucune ligne d'attribution ni mention d'outil d'assistance dans les messages et le contenu ajouté | GC8, GC9 (`check_attribution.py`) | tous |

## Lots

**Références.** Les `path:line` du plan désignent le commit de baseline (`<sha_base>`). Les consignes des exécutants citent le **symbole** (fonction ou bloc) et le sha. Les exécutants se repèrent par symbole (`git show <sha_base>:<chemin>`), jamais par numéro de ligne, puisque les lots précédents décalent les lignes.

**Gates communes (GC), exécutées par l'coordinateur.**

Notations : `<L>` = lot, `<W>` = vague. Fichiers de travail dans `data/albert_gates/` (ignoré par git) :

| Fichier | Contenu |
|---|---|
| `owned_<L>.txt` | Un chemin relatif littéral par ligne, sans commentaire ni ligne vide. Une ligne terminée par `/` désigne un dossier entier (réservé à `tests/fixtures/albert/`). |
| `owned_<W>.txt` | L'union des `owned_<L>.txt` de la vague. |
| `restrict_<L>.txt` | Lignes `chemin=symbole1,symbole2` (seuls ces symboles de premier niveau peuvent différer, par comparaison AST HEAD/index) ou `chemin=@docstrings` (seules les docstrings peuvent différer). |
| `preexisting.txt` | Les entrées `git status` d'avant L0, ignorées. |
| `msg_<L>.txt` | Le message de commit. |

**Ordre d'exécution :** GC4, puis les tests du lot, puis GC1 et GC2, puis GC3 (fin de vague), puis GC5, GC6, GC7, GC9, puis GC8.

- **GC1** (goldens rejoués) : `.venv/bin/python -m pytest tests/test_albert_off_golden.py tests/test_albert_off_golden_routes.py -q -p no:cacheprovider` → 0 failed (goldens répartis en `golden_off/scripts/` et `golden_off/routes/`)
- **GC2** (goldens intacts) : `git status --porcelain --untracked-files=all -- tests/fixtures/albert/golden_off | wc -l` → 0 ; `git log --format=%h -- tests/fixtures/albert/golden_off | wc -l` → 1
- **GC3** (suite complète, une fois par vague ; `;` et non `&&`, pour que le diff tourne même en cas d'échec) :
  - Commande (derrière un proxy mort : aucun appel externe accidentel possible) : `HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 NO_PROXY=127.0.0.1,localhost,testserver .venv/bin/python -m pytest tests/ scripts/test_rad_vectordb.py scripts/test_rad_chunk_initial_phase.py scripts/test_albert_guard_scope.py app/test_main.py -q -p no:cacheprovider --junitxml=data/albert_gates/<W>.xml; .venv/bin/python tests/tools/junit_diff.py --baseline data/albert_baseline/green.xml --current data/albert_gates/<W>.xml --max-failures 0`
  - Attendu : `NEW_FAILURES=0 FAILURES=0 ERRORS=0 MISSING=0`. `PASSED_DELTA` et `NET_LEGACY` sont informatifs.
  - En W1, la référence est `baseline.xml`.
  - `junit_diff` compte failure et error. Il sort en non-zéro si un test vert à la référence disparaît ou passe en skip sans figurer dans `data/albert_gates/allowed_removals.txt`.
- **GC4** (périmètre de l'arbre de travail) : `.venv/bin/python tests/tools/check_scope.py --owned data/albert_gates/owned_<W>.txt --preexisting data/albert_gates/preexisting.txt` → `SCOPE_OK`
  - `check_scope` utilise `git status --porcelain --untracked-files=all` et compare des chemins exacts.
  - Il ignore `(^|/)\.DS_Store$`, `Icon\r` et `._*`.
  - Il signale à part les copies de conflit Drive (`* (1).*`, `CONFLIT`).
- **GC5** (indexation contrôlée) : `.venv/bin/python tests/tools/check_scope.py --owned data/albert_gates/owned_<L>.txt --preexisting data/albert_gates/preexisting.txt --emit-add-list data/albert_gates/add_<L>.txt && git add --pathspec-from-file=data/albert_gates/add_<L>.txt && .venv/bin/python tests/tools/check_scope.py --owned data/albert_gates/owned_<L>.txt --restrict data/albert_gates/restrict_<L>.txt --staged` → `SCOPE_OK`
  - La liste émise ne contient que des fichiers existants ou supprimés-suivis. Jamais de `.DS_Store`.
- **GC6** (secrets) : `.venv/bin/python tests/tools/check_secrets.py --staged --paths data/albert_gates tests/fixtures/albert` → `PATTERN_HITS=0 REAL_VALUE_HITS=0`
  - Motifs `sk-`, `pcsk_` et `Bearer` de 20 caractères ou plus, en ignorant les gabarits composés de X.
  - Valeurs réelles lues par `dotenv_values('.env')` pour les noms contenant KEY, SECRET, TOKEN ou PASSWORD (12 caractères au moins).
  - Rien n'est jamais affiché, hormis des compteurs et des noms de fichiers.
- **GC7** (docstrings) : `.venv/bin/python tests/tools/check_docstrings.py --staged` → `MISSING=0`
- **GC9** (attribution dans le contenu ajouté) : `git diff --cached -U0 --text | .venv/bin/python tests/tools/check_attribution.py --diff` → `ATTRIBUTION_HITS=0`. L'outil ne lit que les lignes ajoutées (`+`, hors en-têtes `+++`), ignore les noms de fichiers existants du projet (`CLAUDE.md`, dossier `.claude/`) et construit ses motifs par concaténation, pour ne jamais contenir les chaînes qu'il recherche.
- **GC8** (commit ; le message est contrôlé AVANT le commit) :
  1. `.venv/bin/python tests/tools/check_attribution.py --file data/albert_gates/msg_<L>.txt` → `ATTRIBUTION_HITS=0`
  2. `git interpret-trailers --parse data/albert_gates/msg_<L>.txt | wc -l` → 0
  3. `git commit -F data/albert_gates/msg_<L>.txt`
  4. `git log -1 --format='%B%n%(trailers)' | .venv/bin/python tests/tools/check_attribution.py --stdin` → `ATTRIBUTION_HITS=0`
  5. `git log -1 --format='%an <%ae>'` → `Amar LAKEL <amar@lakel.net>`

  Si le contrôle a posteriori échoue : `git commit --amend -F data/albert_gates/msg_<L>.txt` immédiatement, avant tout autre commit. Un message ne cite jamais `CLAUDE.md` ; on écrit « documentation projet ».

### Lot 0 — Baseline, branche, sondes live, fixtures assainies, golden OFF, outils de gate

- **Objectif** : figer le point de départ, mesurer la dette de tests, établir les faits de l'API (sondes P1 à P21 et décisions D1 à D22), puis capturer sur le code de base l'oracle d'identité OFF, avec des identifiants factices.
- **Dépend de** : prérequis utilisateur. **Vague** : W0 (séquencée, voir l'organisation).
- **Fichiers possédés** :
  - outils : `tests/tools/junit_diff.py`, `tests/tools/check_docstrings.py`, `tests/tools/check_scope.py`, `tests/tools/check_secrets.py`, `tests/tools/check_attribution.py`, `tests/test_gate_tools.py` ;
  - sondes : `scripts/albert_probe.py`, `tests/test_albert_probe_sanitize.py`, dossier `tests/fixtures/albert/` (fixtures P*, `decisions.json`, fixtures synthétiques) ;
  - golden : `tests/test_albert_off_golden.py`, `tests/fixtures/albert/golden_off/` ;
  - suivi : `.claude/tasks/albert_probe_report.md`, `docs/SPRINT_albert.md` ;
  - commit baseline sur `main` : liste explicite, avec des docstrings seulement si nécessaire ;
  - hors git : `data/albert_baseline/`, `data/albert_probe/`, `data/albert_gates/`.
- **Tâches** :
  1. **[Utilisateur, bloquant]**
     - Ajouter `ALBERT_API_KEY` dans le `.env` local uniquement ; jamais `ALBERT_LIVE`.
     - Révoquer et remplacer la clé Pinecone du commit initial `d1f5875` (préfixe `pcsk_`, déjà poussée sur `origin/main`).
     - Suspendre la synchronisation Google Drive du dépôt (recommandé).
     - Répondre aux points à confirmer (a) à (g).
  2. **Outils de gate (exécutant A)**, écrits sans être indexés. Chacun a ses docstrings et `tests/test_gate_tools.py` les teste.
     - `check_docstrings.py` : comparaison AST entre `git show HEAD:f` et `git show :f`.
       - Docstring exigée pour toute fonction ou méthode nouvelle de premier niveau ou de classe, `__init__` compris.
       - Exemptions : closures imbriquées, lambdas, fonctions `test_*` des fichiers de test. `--strict` supprime ces exemptions.
     - `check_scope.py` : voir GC4 et GC5.
     - `junit_diff.py` : voir GC3. `--summary` liste les fichiers en échec.
     - `check_secrets.py` : voir GC6.
  3. **Baseline (exécutant D, puis coordinateur).**
     - Lancer `check_docstrings --staged` sur la liste de baseline indexée.
     - Ajouter UNIQUEMENT des docstrings aux fonctions signalées : environ 45 dans `rad_dedup`, `rad_recode_cache`, `bench_dedup`, `backfill`, `rebuild`, `rad_vectordb` et les helpers non-test de `test_dedup`.
     - Signaler à l'utilisateur ces ajouts dans son travail de dédup.
     - `.claude/albertai.md` est **exclu** de la baseline : il reste non suivi et entre dans `preexisting.txt`.
  4. **Environnement (coordinateur)**, après le commit de baseline et le passage sur `feat/albert`.
     - `.venv/bin/pip install ebooklib==0.18` : dépendance déjà épinglée, changement d'environnement seulement.
     - Préchauffer les caches tiktoken et spaCy.
     - Écrire `preexisting.txt`.
     - Produire le JUnit de référence. `junit_diff --summary` donne la liste réelle des fichiers en échec, qui alimente la partie « tests obsolètes » d'`owned_L2.txt`.
  5. **`scripts/albert_probe.py` (exécutant B).** httpx seul. La clé est lue par `dotenv_values` et jamais affichée. Le script refuse de tourner sans `ALBERT_LIVE=1`.
     - Ordre : P1 à P21 selon `digest_albert_api.txt`.
       - P3 : PDF fitz `PAGE_ONE_7Q` / `PAGE_TWO_9Z`, `pages:[0]` puis `[1]`.
       - P4 seulement si P3 donne l'accès.
       - P13 passif, jamais de rafale.
       - P14 à P19 sur `ragpy-probe-<ts>` (privée), supprimée dans un `finally`.
       - P21 en concurrence 1 puis 8.
     - Options : `--all`, `--only`, `--cleanup [--dry-run]` (préfixe `ragpy-probe-`, qui couvre `ragpy-probe-e2e-`), `--out <dossier>` et `--diff-against <dossier de fixtures>` pour les re-sondes.
     - Sorties :
       - bruts dans `data/albert_probe/<ts>/raw/` ;
       - fixtures `{request, status, headers utiles, body}` passées par `sanitize_payload` (clé, e-mails, noms et ids de compte retirés) ;
       - `tests/fixtures/albert/decisions.json` : pour chaque Dn, la valeur mesurée, le défaut proposé, la source (`probe` ou `default`) et le consommateur.
     - Pour tout endpoint sans réponse réelle de succès (par exemple `/v1/ocr` sans accès), écrire une fixture marquée `"synthetic": true`, rédigée d'après `.claude/albertai.md`.
  6. **Harnais golden (exécutant C, sur le code de baseline déjà commité).** Régénérable uniquement avec `RAGPY_UPDATE_GOLDEN=1`.
     - **Hygiène** :
       - `monkeypatch.delenv` des 15 noms d'env de la baseline (liste littérale), de `ALBERT_*` et de `EMBEDDING_PROVIDER`, puis `setenv` de valeurs factices (`fake-openai-0001`, etc.) ;
       - `app.routes.settings.RAGPY_DIR` redirigé vers `tmp_path`, avec un `.env` factice (motif de `tests/test_credentials_ui_sync.py`) ;
       - valeurs stockées sous forme d'empreintes `sha256[:12]` ;
       - normalisation des chemins (`<RAGPY>`, `<TMP>`, `<SESSION>`) et des durées (`<T>`) ;
       - `random.randint` patché (doc_id, `rad_chunk.py:424`), `DEFAULT_MAX_WORKERS=1`.
     - **Listes littérales** : les 6 champs d'`OCRResult` lus par `getattr`, les 15 noms d'env. `_asdict()` et `CREDENTIAL_ENV_MAPPING.values()` sont proscrits.
     - **Points de patch** : `app.routes.processing.run_tracked_subprocess` et `app.utils.sse_helpers.run_subprocess_with_sse`. La fixture `force_split` est recopiée depuis `tests/test_ocr_providers.py:503`, avec une docstring.
     - **Goldens** :
       - **G1 OCR** : legacy avec `MISTRAL_API_KEY=None`, et Mistral mocké par `_mistral_upload_and_ocr_once` avec split/salvage. Capture : texte, 6 champs, message final (`rad_dataframe.py:1802-1806`), ordre de la chaîne (espions).
       - **G2** : octets du JSON de `process_document_chunks` et kwargs de `create`, en OpenAI et en OpenRouter (`_resp`, `tests/test_dedup.py:336`).
       - **G3** : octets du fichier de `generate_and_save_embeddings` (`:716`) et kwargs exacts `{input, model, timeout=60.0}`.
       - **G4** : `recode_key` et `embed_key` sur la matrice {gpt-4o-mini, google/gemini-2.5-flash, openai/gpt-oss-120b} × prefer_openai × harden.
       - **G5** : le bloc `=== Result ===` seul, pour Pinecone mocké.
       - **G6** : `build_subprocess_env`, admin et non-admin, restreint aux 15 noms.
       - **G7** : argv, env (15 noms), kwarg `timeout`, corps 403 et 400, premiers événements SSE pour :
         - `process_dataframe(_sse)`, `initial_text_chunking(_sse)` (gpt-4o-mini, gemini), `dense(_sse)`, `sparse(_sse)` ;
         - `upload_db` pinecone, weaviate et qdrant, et `db_choice=albert` (400 actuel) ;
         - `generate_zotero_notes_sse` ;
         - `filter_citations_sse`, `batch_import_citations_sse` et `filter_citations_background`, sans clés et avec un slug OpenRouter sans clé ;
         - `cluster_documents(_sse)` ;
         - le JSON d'état du pipeline (`pipeline.py:737-746`).
       - **G8** : JSON de `GET /users/me/credentials` et de `GET /get_credentials` (ordre des clés).
       - **G9** : sha256 normalisé de `index.html`, `user/profile.html` et `user/project_detail.html`, en admin et en non-admin.
       - **G10** : stdout de l'import de `rad_chunk`, avec une clé factice, en sous-processus.
       - **G11** : kwargs de `_generate_with_llm` et de `_call_llm_api` (`max_tokens=1500`).
       - **G12** : stdout (capsys) de `process_document_chunks` et de `generate_and_save_embeddings` avec clients mockés : lignes `Using OpenAI with model: …`, `Traitement de '…'`, `PROGRESS|…`.
       - **G13** : logs INFO et WARNING (caplog) d'`extract_text_with_ocr` sur le chemin OFF (Mistral mocké, legacy).
       - Pas de golden sparse (hash salé).
  7. **Sondes (coordinateur)** : lancement, puis nettoyage. Un analyste en lecture seule rédige les décisions dans le rapport ; chacune a un défaut prudent si la sonde n'est pas concluante.

     | Décision | Sonde | Objet | Consommateur |
     |---|---|---|---|
     | D1 | P1 | RPM/RPD par modèle, budget, expiration, permissions (écriture des collections) | L1 config, preflight, message 403 des collections |
     | D2 | P2 | ids, alias, types, contexte | L1 catalog |
     | D3 | P3 | accès à `/v1/ocr`, pages à partir de 0 | défaut `ALBERT_OCR_MODE` (L4), saut du test live |
     | D4 | P4 | taille des parts et latence | `ALBERT_OCR_PART_MB`, `_PART_PAGES`, `ALBERT_TIMEOUT_OCR_DOC` |
     | D5 | P5 | LightOnOCR accepté en chat, latence | `ALBERT_OCR_CONCURRENCY`, `ALBERT_TIMEOUT_OCR_PAGE` |
     | D6 | P6 | forme des erreurs | `errors.py` |
     | D7 | P7 | présence de cost et impacts | champs du ledger |
     | D8 | P8 | `max_tokens` ou `max_completion_tokens` | `client.chat` |
     | D9 | P9 | champ de raisonnement, budget | marge de raisonnement, raisonnement jamais pris comme contenu |
     | D10 | P10 | modes JSON | filtre de citations, phase 1 des fiches |
     | D11 | P11 | déterminisme de `seed` | mode harden (L3) : seed envoyé seulement si déterministe ; défaut : non envoyé, `seed` absent de `dp` |
     | D12 | P12 | plafond de lot, dimension, norme | `ALBERT_EMBED_L2_NORMALIZE`, `norm` dans les paramètres d'espace et la clé de cache |
     | D13 | P13 | Retry-After (secondes ou date), corps distinguant RPM, RPD et TPM | `retry.py`, quota épuisé |
     | D14 | P14 | filtre de nom, noms en double | L5 : correspondance exacte ; plusieurs correspondances = erreur listant les ids, jamais de choix implicite |
     | D15 | P15 | multipart ou urlencoded | `create_document` |
     | D16 | P16 | limites d'envoi, fusion des métadonnées, ids de chunks renvoyés ou non, atomicité d'un POST de 64 | contenu du manifeste (ids s'ils sont renvoyés, sinon document, tranche et nombre) et rollback (suppression du document dans tous les cas) |
     | D17 | P17 | `query=null`, `rff_k`, seuil | `client.search` |
     | D18 | P18 | latence d'indexation | attentes des tests live, note de cohérence du preload |
     | D19 | P19 | quota bge-m3 consommé par les envois | `ALBERT_PUSH_CONCURRENCY`, rôle `embed` du limiteur pendant l'envoi |
     | D20 | P20 | `/health` et `/health/models` | **informative** : le preflight utilise `/v1/models`, le repli 503 reste sur les statuts HTTP |
     | D21 | qualité OCR mesurée en live | recodage des textes LightOnOCR | `ALBERT_OCR_SKIP_RECODE` (défaut 0), confirmé au L8 avec l'utilisateur |
     | D22 | P21 | débit p50/p95, formule `ceil(0.9 × RPM × p50 / 60)` | `ALBERT_RECODE_CONCURRENCY`, `ALBERT_NOTES_CONCURRENCY`, `ALBERT_EMBED_CONCURRENCY` |
  8. **`docs/SPRINT_albert.md`**, rédigé en termes de lots et de contrats, sans ligne d'attribution ni mention d'outil d'assistance. Il fige :
     - les signatures du socle ;
     - le format de `RAGPY_DOTENV_DENY` ;
     - les libellés OCR ;
     - `EMBEDDING_PROVIDER` ;
     - `resolve_default_llm_model` et `effective_recode_model` ;
     - la politique de docstrings ;
     - la convention de clé factice (`FAKE_ALBERT_KEY = 'fake-albert-key-0001'`, jamais de préfixe `sk-` ni `pcsk_` sous `tests/`) ;
     - la règle d'assertion : n'asserter que des booléens et des noms, jamais une valeur de clé ;
     - le repérage par symbole et le `<sha_base>`.
- **Tests** :
  - `tests/test_albert_off_golden.py` :: `test_g1…test_g13` verts sur la baseline, puis sur 2 passes sans régénération ; `test_golden_harness_uses_literal_lists` (AST : ni `_asdict`, ni `CREDENTIAL_ENV_MAPPING.values`) ; `test_golden_harness_uses_fake_credentials`.
  - `tests/test_albert_probe_sanitize.py` :: suppression de Authorization, Bearer, e-mails et ids ; idempotence ; aucune fixture ne ressemble à une clé.
  - `tests/test_gate_tools.py` :: `test_junit_diff_counts_errors_and_missing`, `test_check_scope_ignores_ds_store_flags_conflicts`, `test_check_scope_restrict_symbols`, `test_check_secrets_never_prints_values`, `test_check_docstrings_policy`, `test_check_attribution_flags_trailers_ignores_project_doc_names` (le fichier de l'outil ne contient pas lui-même les chaînes recherchées).
- **Gate** :
  1. **Prérequis utilisateur** :
     - `.venv/bin/python -c "from dotenv import dotenv_values; v=dotenv_values('.env').get('ALBERT_API_KEY') or ''; print('CLE OK' if len(v)>20 else 'CLE ABSENTE')"` → `CLE OK`
     - `grep -c '^ALBERT_LIVE' .env` → 0
  2. **Commit de baseline sur `main`** :
     - `git branch --show-current` → `main`
     - `git add -- docs/GUIDE.md .env.example CLAUDE.md app/routes/processing.py app/tasks/vectordb.py scripts/rad_chunk.py scripts/rad_vectordb.py scripts/test_rad_vectordb.py .claude/tasks/SPRINT_chunk_dedup.md .claude/tasks/sprint-deduplicate.md scripts/backfill_pinecone_inplace.py scripts/rad_dedup.py scripts/rad_recode_cache.py scripts/rebuild_pinecone_index.py tests/load/bench_dedup.py tests/test_dedup.py && .venv/bin/python tests/tools/check_docstrings.py --staged` → `MISSING=0`
     - `git diff --cached --name-only | grep -cE 'DS_Store|__pycache__|(^|/)\.env$|\.log$|albertai'` → 0
     - GC6 → `PATTERN_HITS=0 REAL_VALUE_HITS=0`, puis GC9 → 0
     - Message dans `data/albert_gates/msg_baseline.txt` : `Sprint déduplication : cascade Tier 1-3, cache de recodage, adaptateurs de connecteurs, outils de maintenance Pinecone`. Contrôles 1 et 2 de GC8.
     - `git commit -F data/albert_gates/msg_baseline.txt && git switch -c feat/albert && git branch --show-current` → `feat/albert`, puis contrôles 4 et 5 de GC8
  3. **Environnement et JUnit de référence** :
     - `.venv/bin/pip install ebooklib==0.18 && .venv/bin/python -c "import tiktoken, spacy; tiktoken.encoding_for_model('text-embedding-3-large'); spacy.load('fr_core_news_md'); print('CACHES OK')"` → `CACHES OK`
     - `git status --porcelain --untracked-files=all > data/albert_gates/preexisting.txt && grep -c albertai data/albert_gates/preexisting.txt` → 1
     - `.venv/bin/python -m pytest tests/ scripts/test_rad_vectordb.py scripts/test_rad_chunk_initial_phase.py app/test_main.py -q -p no:cacheprovider --junitxml=data/albert_baseline/baseline.xml; .venv/bin/python tests/tools/junit_diff.py --summary data/albert_baseline/baseline.xml` → termine ; affiche `FAILURES=… ERRORS=… FAILING_FILES=…`, qui sert à compléter `owned_L2.txt`
  4. **Sondes** :
     - `ALBERT_LIVE=1 .venv/bin/python scripts/albert_probe.py --all --report .claude/tasks/albert_probe_report.md --fixtures-dir tests/fixtures/albert` → dernière ligne `SONDES: 21 exécutées, 0 erreur de script`
     - `ALBERT_LIVE=1 .venv/bin/python scripts/albert_probe.py --cleanup --dry-run` → `ragpy-probe restantes: 0`
     - `grep -rlE 'sk-[A-Za-z0-9_-]{16,}|Bearer [A-Za-z0-9._-]{20,}|lakel' tests/fixtures/albert .claude/tasks/albert_probe_report.md` → aucune sortie
  5. **Goldens et outils** : `.venv/bin/python -m pytest tests/test_albert_off_golden.py tests/test_albert_probe_sanitize.py tests/test_gate_tools.py -q -p no:cacheprovider` → 0 failed (2 passes)
  6. **Commit L0** : GC4, GC5, GC6, GC7, GC9, GC8.
- **Commit** : `Tests : sondes de l'API Albert, fixtures assainies, références golden du comportement par défaut, outils de contrôle`.
- **Rollback** : `git revert --no-edit <sha L0>` sur `feat/albert`. Le commit baseline de `main` (travail de l'utilisateur) est conservé. `--cleanup` supprime les collections restantes.

### Lot 1 — Socle Albert : paquet client, résolveur, identifiants, infrastructure de test

- **Objectif** : livrer et figer tous les contrats utilisés par les capacités, sans un octet de différence quand Albert est OFF, et avec une infrastructure de test couvrant les 3 racines.
- **Dépend de** : L0. **Vague** : W1, en deux phases. W1a (contrats) passe avant le reste de L1, qui tourne en parallèle avec L2.
- **Fichiers possédés** :
  - socle : `scripts/rad_albert/__init__.py`, `scripts/rad_albert/config.py`, `scripts/rad_albert/catalog.py`, `scripts/rad_albert/errors.py`, `scripts/rad_albert/retry.py`, `scripts/rad_albert/limiter.py`, `scripts/rad_albert/usage.py`, `scripts/rad_albert/client.py`, `scripts/rad_albert/preflight.py`, `scripts/rad_providers.py`, `scripts/requirements.txt` ;
  - infrastructure de test : `conftest.py` (racine), `tests/albert_fakes.py`, `scripts/test_albert_guard_scope.py` ;
  - tests : `tests/test_albert_config.py`, `tests/test_albert_client.py`, `tests/test_albert_limiter.py`, `tests/test_rad_providers.py`, `tests/test_albert_test_infra.py`, `tests/test_albert_credentials.py`, `tests/test_credentials_ui_sync.py`, `tests/live/__init__.py`, `tests/live/test_albert_live_core.py` ;
  - app : `app/core/credentials.py`, `app/schemas/user.py`, `app/routes/users.py`, `app/routes/settings.py`, `app/routes/pages.py` ;
  - templates : `app/templates/user/profile.html`, `app/templates/index.html` (carte de la modale admin seulement).
- **Tâches** :
  1. **W1a, contrats (livrés avant L2)** :
     - `config.py` et `errors.py` (exécutant 1A) ;
     - `rad_providers.py`, `conftest.py` et `albert_fakes.py` (exécutant 1B) ;
     - dans `credentials.py`, la pose de `RAGPY_DOTENV_DENY` (exécutant 1C).
  2. **`config.py`.**
     - Parseurs repris de `rad_dedup.py:139-156` (`''` = non défini).
     - `AlbertConfig.from_env(env=None)` avec **tous** les champs du tableau des drapeaux, figés. Il ne lit jamais `ALBERT_API_KEY`.
     - `normalise_base_url` : https ; `/v1` conservé, contrairement à `credentials.py:344` ; réécriture de `api.albert` avec WARNING ; liste blanche ; pas d'userinfo. Plus `root_url`.
     - `ENV_REGISTRY` : nom, défaut, sens en français.
     - `validate_metadata_fields(fields, dedup_meta_fields)` : refuse une liste de plus de 10 champs, avec `path`, ou sans `content_id`, `content_hash`, `chunk_index` et `DEDUP_META_FIELDS`.
  3. **`catalog.py`** : `MODELS`, `ROLES`, `EXCLUDED` ; `fallback_chain(role, *, today)`, `resolve_model(requested, listing)`, `check_endpoint_type`, `is_reasoning`, `embedding_dim`, `canonical_id`.
  4. **`errors.py`, `retry.py` et `limiter.py`.**
     - Erreurs : taxonomie ci-dessus et `MESSAGES_FR`.
     - `call_with_retry(fn, *, policy, semaphore, limiter, tokens, sleep)` : acquisition du limiteur hors sémaphore, sémaphore autour de l'envoi seulement, sommeil après libération.
     - `acall_with_retry` : `await asyncio.to_thread(limiter.acquire, tokens)` et `asyncio.sleep`.
     - Limiteur : `TokenBucket` thread-safe (horloge et sommeil injectables) ; `RedisWindowLimiter` (import paresseux, URL `ALBERT_LIMITER_REDIS_URL`, sinon `CELERY_BROKER_URL`, sinon `redis://localhost:6379/0`) ; `get_limiter(role, cfg)` ; `estimate_input_tokens`.
  5. **`usage.py`** : `UsageLedger` (enregistre, `summary_line`, `write_jsonl`) et `extract_usage`.
  6. **`client.py`** : `AlbertClient(cfg, api_key, *, transport=None, ledger=None, sleep=time.sleep)`.
     - `chat()` : `max_tokens` (D8). Pour un modèle de raisonnement, ajout de la marge et de `reasoning_effort`. Ne renvoie que `content`. Option `single_attempt`.
     - `embed()` : tranches de 64, tri par index, contrôle de dimension, `''` refusé, jamais `dimensions`, normalisation L2 selon D12.
     - `ocr_image()` (message image seule) et `ocr_document()` (data URI, pages transmises telles quelles).
     - `me`, `models`, `health` (à la racine).
     - Collections : `list_collections`, `create_collection` (private forcée), `delete_collection`, `list_documents`, `create_document` (multipart `files={'name':(None,n),'collection_id':(None,str(cid))}`), `delete_document`, `list_chunks` (`_paginate` limit 100), `add_chunks` (1 à 64).
     - `search` : `method` explicite, `rff_k`, `score_threshold` seulement en semantic.
  7. **`preflight.py`** : `run_preflight(client, *, roles, today, margin_days=7)`.
     - `/v1/me` : compte expiré → `AlbertAuthError` ; avertissement à moins de 30 jours ; permissions (D1).
     - `/v1/models` : résolution et types.
     - Cache par `sha256(clé)[:12]`, TTL 600 s.
     - `__init__.py` : réexports légers seulement.
  8. **`rad_providers.py`.**
     - Décision 3.
     - `EmbeddingSpace(provider, model, dim, batch_max, is_default, norm)` ; `OPENAI_DEFAULT` (3072) ; `ALBERT_BGE_M3` (1024, 64, `l2`).
     - `EmbeddingConfig.from_env` :
       - `albert` avec `ALBERT_ENABLED≠1` lève `AlbertDisabledError` ;
       - une valeur inconnue lève `ValueError` ;
       - `embed_params_json()` vaut **exactement** `json.dumps({"model": "text-embedding-3-large"}, sort_keys=True)` pour OpenAI (`rad_chunk.py:667`).
     - `check_uniform_space` et `target_mismatch_message` (ignore une dimension non entière).
  9. **`conftest.py` à la racine.**
     - **À l'import**, avant tout import de `app` : capture `LIVE = os.environ.get('ALBERT_LIVE') == '1'`. Si `dotenv_values('.env')` contient `ALBERT_LIVE`, `pytest.UsageError` avec un message.
     - **`pytest_configure`** :
       - enregistre le marqueur `albert_live` ;
       - hors live, pose `ALBERT_ENABLED=0`, `OCR_ENABLE_ALBERT=0`, `ALBERT_API_KEY=''`, `EMBEDDING_PROVIDER=''` et `ALBERT_LIMITER_BACKEND=local`. Le `load_dotenv()` non-override d'`app/config.py:22` n'injecte donc jamais la clé de production ;
       - installe la garde réseau de session (voir ci-dessous).
     - **Garde réseau** : patch de `httpx.HTTPTransport.handle_request`, `httpx.AsyncHTTPTransport.handle_async_request` et `requests.adapters.HTTPAdapter.send`.

       | Accès | Test live, en mode live | Test nouveau | Test historique |
       |---|---|---|---|
       | Hôte `*.etalab.gouv.fr` | autorisé | enregistré et levé | enregistré et levé |
       | Autre hôte non local | — | enregistré et levé | enregistré seulement (`data/albert_gates/net_legacy.json`, compteur `NET_LEGACY`) |

       Les tests nouveaux sont `test_albert_*`, `test_env_isolation`, `test_debt_regressions`, `test_celery_tasks` et `test_vector_space_guards`.
     - **Fixture autouse** :
       - `delenv` de `ALBERT_*`, `OCR_ENABLE_ALBERT` et `EMBEDDING_PROVIDER` (monkeypatch) ;
       - constantes Albert des modules déjà importés forcées à OFF ;
       - échec au teardown si un enregistrement bloquant existe, même quand le code a avalé l'exception.
     - **Fixture de session `anyio_backend`**, qui renvoie `'asyncio'` (avec docstring). `pytest_collection_modifyitems` lève `UsageError` si un id contient `[trio]`.
     - Les tests `albert_live` sont sautés sauf si `LIVE`.
  10. **`tests/albert_fakes.py`, complet dès L1.**
      - `FakeAlbert(fixtures_dir)` : MockTransport pour **tous** les endpoints de `client.py`.
        - Réponses scriptées par (méthode, chemin).
        - Injection de statuts et de Retry-After.
        - Stockage en mémoire des collections, documents et chunks, avec pagination.
        - Réponses OCR et chat-image tirées des fixtures, réelles ou synthétiques.
        - Journal des appels.
      - `FakeRedis` : incr, expire, get, set, avec TTL et horloge injectable.
      - `FAKE_ALBERT_KEY`.
      - Le fichier est gelé après W1 et n'est modifié ensuite que par le gardien, entre deux vagues.
  11. **`credentials.py`.**
      - `albert_api_key` ajouté aux registres `:76-99`, `:102-118` et `:121-137`, avec le message « Clé API Albert (DINUM) requise. Configurez-la dans Paramètres > Mes Identifiants. ».
      - `visible_credential_keys()`.
      - Dans `build_subprocess_env`, pour les non-admins seulement : `env['RAGPY_DOTENV_DENY'] = ','.join(sorted(noms *_API_KEY retirés et non réinjectés))`. Signature inchangée.
  12. **`schemas/user.py:160-212` et `users.py`.**
      - `albert_api_key: Optional[CredentialValue] = None` et `Optional[str]`.
      - `users.py:162` : `albert_api_key` ignoré quand OFF, plus `response_model_exclude_none=True`.
      - `users.py:182` (PUT) : `albert_api_key` ignoré quand OFF.
  13. **`settings.py`.**
      - `_admin_form_env_keys()` remplace les 3 listes (`:81-87`, `:107-114`, `:180-187`), dans le même ordre, avec `ALBERT_API_KEY` seulement si Albert est ON.
      - `GET /get_credentials` :
        - superposition base filtrée par `_admin_form_env_keys()` ;
        - `ALBERT_API_KEY` renvoyée masquée côté serveur (`••••` + 4 derniers caractères, ou `''`).
      - `POST /save_credentials` : une valeur `ALBERT_API_KEY` qui commence par `••••` est ignorée ; `''` efface.
      - `GET /api/albert/status` et `GET /api/albert/models`, sur le squelette de `list_qdrant_collections` (`:398-466`) :
        - 404 si OFF ;
        - 403 `{error, credential_required, configure_url}` si la clé manque ;
        - 401 amont → 400 `credential_invalid` ; 429 ou 5xx → 502 ;
        - `run_in_threadpool` ; **aucun** `Query(api_key)` ; ni e-mail ni nom dans la réponse.
  14. **Interface.**
      - `pages.py:36` : `albert_enabled` ajouté au contexte.
      - `profile.html` : section Albert après Mistral (`:142-170`) et entrée dans `credentialKeys` (`:720-727`).
      - `index.html` : carte Albert entre Mistral (`:912`) et Pinecone (`:931`), avec :
        - « Tester la connexion », qui appelle `/api/albert/status` avec le JWT de l'application tiré de localStorage ;
        - un lien vers le playground ;
        - la mention de l'expiration à un an et de l'éligibilité.
      - Blocs Jinja **en fin de ligne existante**. Le `do` parasite en tête de `index.html` est conservé.
  15. **`scripts/requirements.txt:61`** : déplacer `httpx<=0.27.2` dans la section runtime, même version.
- **Tests** :
  - `tests/test_albert_config.py` :: `test_defaults_off_never_reads_key` ; `test_normalise_*` (vide, `/`, http, `api.albert` réécrit, hôte étranger rejeté, drapeau custom) ; `test_root_url_strips_v1` ; `test_fallback_chain_drops_mistral_small_after_2026_12_01` ; `test_excluded_never_default` ; `test_resolve_alias_to_id` ; `test_embeddings_have_no_fallback` ; `test_package_import_is_light` (sous-processus) ; `test_defaults_consistent_with_decisions` ; `test_metadata_fields_validation`.
    - Livraison : les quatre tests du catalogue (`test_fallback_chain_drops_mistral_small_after_2026_12_01`, `test_excluded_never_default`, `test_resolve_alias_to_id`, `test_embeddings_have_no_fallback`) sont dans `tests/test_albert_catalog.py` ; les écritures jamais rejouées après un échec ambigu sont testées dans `tests/test_albert_client_writes.py`.
  - `tests/test_albert_client.py` :: `test_classification_matrix` (fixtures P6) ; `test_permanent_single_attempt` ; `test_retry_after_seconds_and_http_date_capped` ; `test_quota_exhausted_abort` ; `test_busy_then_role_fallback` ; `test_sleep_outside_semaphore` ; `test_parse_retry_after_parity_with_rad_dataframe` ; `test_constructor_requires_key_never_reads_environ` ; `test_key_only_in_authorization_header` ; `test_errors_and_logs_redact_key` ; `test_no_user_field` ; `test_no_redirect_follow` ; `test_chat_reasoning_headroom` ; `test_truncated_empty_raises` ; `test_embed_slices_64_sorts_checks_dim_rejects_empty` ; `test_add_chunks_1_to_64` ; `test_create_document_multipart` ; `test_create_collection_private_forced` ; `test_search_rejects_threshold_non_semantic` ; `test_ocr_document_pages_passthrough` ; `test_ledger_records_response_model_cost` ; `test_preflight_expired_raises` ; `test_fake_albert_covers_all_client_endpoints`.
  - `tests/test_albert_limiter.py` :: `test_bucket_rpm_tpm_fake_clock`, `test_pause_on_429`, `test_process_share`, `test_redis_window_fake_redis` (FakeRedis), `test_limiter_does_not_block_event_loop` (`asyncio.run` et compteur de ticks), `test_redis_url_resolution`.
  - `tests/test_rad_providers.py` :: `test_resolver_matches_legacy_matrix`, `test_albert_prefix_split_first_slash`, `test_albert_prefix_case_insensitive`, `test_albert_prefix_disabled_raises`, `test_empty_wire_model_raises`, `test_legacy_cache_label_matches_rad_chunk_312`, `test_embed_params_json_openai_exact`, `test_embedding_config_albert_when_disabled_raises`, `test_check_uniform_space_mixed_zero_none`, `test_target_mismatch_ignores_non_int`.
  - `tests/test_albert_test_infra.py` :: `test_marker_registered`, `test_live_skipped_without_ALBERT_LIVE` (même avec une clé), `test_live_refused_if_env_file_has_ALBERT_LIVE`, `test_env_neutralised_before_app_import`, `test_network_guard_blocks_and_records_etalab`, `test_guard_fails_test_even_if_exception_swallowed`, `test_mock_transport_unaffected`, `test_anyio_backend_is_asyncio`.
  - `scripts/test_albert_guard_scope.py` :: `test_network_guard_active_outside_tests_dir` (consomme l'enregistrement via la fixture dédiée).
  - `tests/test_albert_credentials.py` :: `test_registries_equal_and_bijective`, `test_deny_format_sorted_comma`, `test_nonadmin_env_strips_albert_key_and_sets_deny`, `test_admin_env_has_no_deny`, `test_me_credentials_off_equals_golden`, `test_me_credentials_on_masked`, `test_get_credentials_masks_albert_key`, `test_save_credentials_ignores_mask_roundtrip`, `test_get_credentials_off_ignores_db_albert_key`, `test_base_url_not_a_user_credential`, `test_status_404_when_off`, `test_status_403_nonadmin_without_key_even_if_env_key`, `test_status_success_fake`, `test_status_upstream_401_to_400`, `test_status_no_query_api_key_and_no_key_in_response`.
  - `tests/test_credentials_ui_sync.py` :: tests existants verts ; aller-retour de `ALBERT_API_KEY` quand ON (le masque n'écrase pas la clé).
  - `tests/live/test_albert_live_core.py` :: `/v1/me` valide ; modèles requis présents et bien typés ; embed de 3 textes (1024 d, triés) ; `/health` à la racine et 404 sous `/v1` ; clé invalide → `AlbertAuthError`.
- **Gate** :
  - `.venv/bin/python -m pytest tests/test_albert_config.py tests/test_albert_client.py tests/test_albert_limiter.py tests/test_rad_providers.py tests/test_albert_test_infra.py tests/test_albert_credentials.py tests/test_credentials_ui_sync.py scripts/test_albert_guard_scope.py -q -p no:cacheprovider` → 0 failed
  - `.venv/bin/python -c "import sys, scripts.rad_albert, scripts.rad_providers; print('httpx' in sys.modules, 'openai' in sys.modules)"` → `False False`
  - `.venv/bin/python -c "import sys; sys.path.insert(0, 'scripts'); import rad_albert, rad_providers; print('cli-import-ok')"` → `cli-import-ok`
  - `grep -nE 'os\.(environ|getenv)' scripts/rad_albert/client.py scripts/rad_albert/preflight.py | wc -l` → 0
  - `ALBERT_LIVE=1 .venv/bin/python -m pytest tests/live/test_albert_live_core.py -m albert_live -v -p no:cacheprovider` → passed
  - GC4, GC1, GC2 ; GC3 une fois pour W1, après L2 ; puis GC5, GC6, GC7, GC9, GC8
- **Commit** : `Albert lot 1 : paquet client (configuration, erreurs, retry, catalogue, limiteur, usage, preflight), résolveur de fournisseurs, identifiant albert_api_key, route de statut, infrastructure de test`.
- **Rollback** : `git revert` du commit L1, **après** celui de L2 (ordre inverse). Si des utilisateurs ont déjà saisi une clé Albert, retirer `albert_api_key` des registres la purge à leur prochaine sauvegarde : il faut les prévenir.

### Lot 2 — Dette : fuite `.env`, `_get_llm_clients`, `input()`, clé Pinecone, Celery, tests obsolètes

- **Objectif** : corriger toute la dette que l'utilisateur a placée dans le périmètre, et obtenir une suite 100 % verte comme référence (`green.xml`).
- **Dépend de** : L0 et W1a (`RAGPY_DOTENV_DENY` dans `credentials.py`, `legacy_provider` dans `rad_providers.py`, `conftest.py` avec `anyio_backend`). **Vague** : W1, phase W1b.
- **Fichiers possédés** :
  - dette de base : `scripts/rad_env.py`, `scripts/rad_chunk.py`, `scripts/rad_dataframe.py`, `scripts/rad_vectordb.py`, `app/utils/llm_note_generator.py`, `app/routes/processing.py` et `app/utils/sse_helpers.py` ;
    - restrictions : `app/routes/processing.py=run_tracked_subprocess` et `app/utils/sse_helpers.py=run_subprocess_with_sse`, dans `restrict_L2.txt` ;
  - Celery : `app/tasks/runner.py`, `app/tasks/extraction.py`, `app/tasks/chunking.py`, `app/tasks/embeddings.py`, `app/tasks/vectordb.py`, `app/routes/celery_tasks.py` ;
  - tests : `tests/test_env_isolation.py`, `tests/test_debt_regressions.py`, `tests/test_celery_tasks.py` ;
  - tests obsolètes : **liste dérivée de `baseline.xml`** (chaque fichier en failure ou error est attribué à l'exécutant 2C). Liste attendue au minimum :
    - `tests/test_llm_note_generator.py`, `tests/test_citations_routes.py`, `tests/test_citation_fetcher.py`, `tests/test_citation_filter.py`, `tests/test_integration_api.py`, `tests/test_rad_dataframe_epub.py` ;
    - `scripts/test_rad_vectordb.py`, `scripts/test_rad_chunk_initial_phase.py`, `app/test_main.py`.
  - Un vrai bug produit (20 lignes au plus) : l'coordinateur ajoute le fichier source à `owned_L2.txt` avant correction. Si ce fichier appartient à L1, la correction est confiée à l'exécutant 1C.
- **Tâches** :
  1. **`rad_env.load_dotenv_guarded(*a, **kw)`**, dans `scripts/`.
     - Lit `RAGPY_DOTENV_DENY` (virgules, sans espace), note les noms présents, appelle `dotenv.load_dotenv`, puis retire les noms refusés qui étaient absents avant.
     - Sans la variable, le comportement est strictement identique à `load_dotenv`.
     - Remplace `rad_chunk.py:84` et `rad_dataframe.py:165/205`, avec l'import double.
  2. **`rad_chunk.py:87-99`.**
     - Supprimer les `input()`. `update_env_file` n'est supprimé que s'il n'est plus référencé (grep).
     - `client = OpenAI(...)` si la clé existe, sinon `None`. La ligne `print` `:89` est conservée.
     - La garde `:1007` est adaptée à la phase, avec le même message et `exit 1` :
       - sparse n'exige rien ;
       - initial exige un client pour le modèle choisi (`legacy_provider`) ;
       - dense exige OpenAI (L6 étendra).
  3. **`rad_vectordb.py`** : supprimer le bloc commenté `:409-418` contenant le littéral, et l'affichage de la clé `:1242`.
  4. **`llm_note_generator.py:131-138`.**
     - Supprimer `load_dotenv(override=True)` et les replis `os.getenv` des clés.
     - Nouvelle fonction `resolve_default_llm_model(openrouter_model=None)` : `openrouter_model`, sinon `dotenv_values(find_dotenv())` lu sans muter l'env, sinon `os.getenv`, sinon `gpt-4o-mini`. `_get_llm_clients` l'utilise.
     - Docstrings mises à jour.
  5. **`processing.py:75` et `sse_helpers.py:65`** : `stdin=asyncio.subprocess.DEVNULL`.
  6. **`app/tasks/runner.py`.**
     - `load_user(user_id)`.
     - `run_script(cmd, env, *, on_progress, timeout)` : Popen, stdin DEVNULL, lignes `PROGRESS` passées à `parse_multilevel_progress` (`sse_helpers.py:321`), kill du groupe sur timeout ou révocation.
     - `parse_vectordb_stdout` : regex identiques à `processing.py:883-891`.
     - Propriétaire de tâche `ragpy:celery_owner:<id>` dans Redis (TTL 7 jours).
  7. **Tâches réécrites** : `extraction.py:108-134` (lance `rad_dataframe.py --json --dir --output` ; clés mistral puis openai, comme `processing.py:285-306`), `chunking.py:111`, `embeddings.py:106/:221` et `vectordb.py:225-315`, sans aucun `os.getenv` d'identifiant.
     - Chaque tâche lance la CLI via `run_script` avec `build_subprocess_env(user, required_keys=…)`.
     - argv identiques aux routes, `--class` compris ; délais identiques aux routes.
     - Clés des dictionnaires de retour conservées.
     - `autoretry_for` limité aux erreurs d'infrastructure.
  8. **`celery_tasks.py`** : `Depends(get_current_active_user)` sur toutes les routes (`:73` à `:426`).
     - Soumission : `verify_session_access` (`pipeline.py:109`) ; 403 d'identifiants de même forme ; `.delay(..., user_id=…)` sans secret.
     - Statut et annulation : réservés au propriétaire ou à un admin.
     - `/status` et `/workers` : `require_admin`.
  9. **Tests obsolètes** :
     - `test_llm_note_generator` : patcher `_get_llm_clients` ou `OpenAI` avec des clés explicites.
     - `test_citations_routes` (4) : utilisateur chiffré, patcher les bons points.
     - `test_citation_fetcher` et tests async de `test_citation_filter` : `@pytest.mark.anyio` (la fixture `anyio_backend='asyncio'` du conftest évite les variantes trio).
     - `test_integration_api` : `dependency_overrides` admin.
     - `scripts/test_rad_vectordb` : `namespace` `:221`, et `:454-455`.
     - `app/test_main` (7) : `run_tracked_subprocess` en AsyncMock, `dependency_overrides`.
     - `test_rad_chunk_initial_phase` : `tests/fixtures/test_documents.csv`, sortie dans `tmp_path` (plus d'artefact `scripts/test_output_chunks.json`).
     - Règle de repli : `xfail(strict=True)` motivé et signalé à l'utilisateur.
- **Tests** :
  - `tests/test_env_isolation.py` :
    - `test_subprocess_cannot_refill_secrets_from_dotenv` : sous-processus `sys.executable -c` avec `cwd=tmp_path` (le `.env` factice y est trouvé) et `scripts/` sur `sys.path`. Le `.env` factice contient `OPENAI_API_KEY`, `ALBERT_API_KEY` et `MISTRAL_OCR_MODEL`, avec un env non-admin. On vérifie la présence des noms et des booléens, jamais les valeurs.
    - `test_guard_is_plain_load_dotenv_without_marker`, `test_config_vars_still_reloaded`, `test_deny_literal_equals_credentials`, `test_no_bare_load_dotenv_in_subprocess_scripts`.
  - `tests/test_debt_regressions.py` :: `test_import_without_openai_key_never_calls_input` ; `test_route_subprocesses_use_devnull_stdin` ; `test_no_secret_literals_in_repo` (regex construites par concaténation, `git ls-files`, gabarits en X ignorés) ; `test_vectordb_cli_stdout_has_no_key` ; `test_get_llm_clients_has_no_env_fallback` ; `test_get_llm_clients_callers_pass_explicit_keys` (AST sur les 4 sites) ; `test_default_model_read_fresh_from_dotenv` ; `test_import_stdout_with_key_equals_golden` (G10).
  - `tests/test_celery_tasks.py` (`is_celery_available` et `.delay` patchés, FakeRedis) :: `test_every_route_requires_auth` ; `test_foreign_session_denied` ; `test_missing_credential_403_same_shape` ; `test_delay_payload_has_no_secret` ; `test_task_env_is_per_user` (un `OPENAI_API_KEY` admin dans le worker ne fuit pas) ; `test_argv_matches_http_routes` ; `test_nonzero_exit_not_retried` ; `test_parse_vectordb_stdout_parity` ; `test_progress_lines_update_state` ; `test_no_missing_symbol_imports`.
  - Tests réactivés (tâche 9) : 0 failed.
- **Gate** :
  - `git grep -nE 'pcsk_[A-WYZa-z0-9]' -- scripts app | wc -l` → 0 ; `grep -cE '\binput\(' scripts/rad_chunk.py` → 0 ; `grep -c 'override=True' app/utils/llm_note_generator.py` → 0
  - `grep -nE 'run_(initial|dense|sparse)_phase|getenv[(].(PINECONE|WEAVIATE|QDRANT)_' app/tasks/*.py | wc -l` → 0
  - `env -u OPENAI_API_KEY -u OPENROUTER_API_KEY RAGPY_DOTENV_DENY=OPENAI_API_KEY,OPENROUTER_API_KEY .venv/bin/python -c "import sys; sys.path.insert(0, 'scripts'); import rad_chunk; print('client', rad_chunk.client)" < /dev/null` → dernière ligne `client None`, sans blocage
  - `.venv/bin/python -m pytest tests/test_env_isolation.py tests/test_debt_regressions.py tests/test_celery_tasks.py tests/test_llm_note_generator.py tests/test_citations_routes.py tests/test_citation_fetcher.py tests/test_citation_filter.py tests/test_integration_api.py tests/test_rad_dataframe_epub.py scripts/test_rad_vectordb.py scripts/test_rad_chunk_initial_phase.py app/test_main.py -q -p no:cacheprovider -rs` → 0 failed (skip ou xfail seulement s'ils sont listés) ; aucun id `[trio]`
  - GC3 pour W1 avec `--baseline data/albert_baseline/baseline.xml` → `NEW_FAILURES=0 FAILURES=0 ERRORS=0 MISSING=0`, puis copie du XML de W1 en `data/albert_baseline/green.xml`
  - GC1, GC2, GC4 à GC9 (commit de L1 d'abord, puis L2)
- **Commit** : `Dette : isolation .env des sous-processus non-admin, rad_chunk non interactif, retrait de la clé Pinecone, _get_llm_clients sans repli .env, Celery authentifié en sous-processus, tests de référence remis au vert`.
- **Rollback** : `git revert` du commit L2, ce qui rouvre les fuites. Préférer une correction en avant. Ne jamais réintroduire le littéral.

### Lot 3 — Chat Albert : recodage, notes Zotero, fiches de lecture, filtre de citations

- **Objectif** : router `albert/<id>` vers Albert sur tous les chemins LLM :
  - sans repli silencieux ;
  - avec une seule couche de retry ;
  - avec le backoff hors sémaphore ;
  - avec arrêt du job sur erreur de compte ;
  - en gérant le raisonnement de gpt-oss et des clés de cache épinglées.

  OpenAI et OpenRouter restent identiques.
- **Dépend de** : L1, L2. **Vague** : W2.
- **Fichiers possédés** : `scripts/rad_chunk.py`, `app/utils/llm_note_generator.py`, `app/utils/book_note_generator.py`, `app/utils/citation_filter.py`, `app/utils/parallel_citation_processor.py`, `tests/test_albert_recode.py`, `tests/test_albert_notes.py`, `tests/test_albert_citation_filter.py`, `tests/live/test_albert_live_llm.py`.
- **Tâches** :
  1. **`effective_recode_model(cli_model, cfg)`**, source unique utilisée par `gpt_recode_batch` et `recode_batch_cached`. Un `--model albert/…` l'emporte et `prefer_openai` est ignoré ; sinon, la sémantique harden actuelle (`:206`, `:311`) s'applique à l'identique.
  2. **`gpt_recode_batch` (`:183`).**
     - Résolution sur `effective_recode_model`, puis retour anticipé vers `_albert_recode_batch`.
     - Le chemin historique `:205-227` reste intact, 2e passage séquentiel `:275-284` compris.
     - `"/" in model` devient `legacy_provider(model) == "openrouter"`.
  3. **`_albert_recode_batch`.**
     - `ALBERT_RECODE_SEMAPHORE = threading.Semaphore(ALBERT_RECODE_CONCURRENCY)` au niveau du module, partagé par tous les workers de documents (motif `MISTRAL_SEMAPHORE`), tenu pendant l'envoi seulement.
     - `AlbertClient.chat` avec le rôle `recode`. Pas de 2e passage séquentiel.
     - Correspondance de `_extract` (`:240`) :
       - `stop` → `recoded` ;
       - `length` ou `content_filter` → `fallback_truncated` ;
       - vide → `fallback_raw` ;
       - garde de ratio de longueur [0.3, 1.5], Albert seulement → `fallback_raw`.
     - Erreurs :
       - 503 : repli de rôle, modèle servi enregistré ;
       - 404 : message renvoyant à `/v1/models`, sans repli ;
       - `AlbertAuthError` : `_ALBERT_ABORT`, lots restants en `fallback_raw` sans appel, `__main__` sort en `exit 2`. **Jamais** `client` ni `openrouter_client`.
  4. **`recode_batch_cached` (`:296-343`).**
     - Albert : provider `albert`, modèle `albert/<id résolu>`, `dp` = `decode_params_json()` enrichi des champs Albert. `seed` n'y figure que si D11 est déterministe.
     - Autres fournisseurs : `:312` via `legacy_cache_provider_label`, à l'identique.
     - Seul `recoded` est mis en cache. Les prompts ne changent pas.
  5. **Traçabilité et replis.**
     - Quand le fournisseur est `albert`, `recode_status` et `recode_model=albert/<id servi>` sont **toujours** écrits, même sans cache ni harden.
     - `RECODE_SKIP_PROVIDERS` (`:163`) reçoit `albert_mistral_ocr`, et `albert_lightonocr` seulement si `ALBERT_OCR_SKIP_RECODE=1`.
     - Preflight du rôle au début de la phase initial. Aide de `--model` (`:945`). Ledger seulement si Albert a été appelé.
  6. **`llm_note_generator.py`.**
     - `_get_albert_client(key)`, sans env.
     - `run_llm_slot` (décision 15), avec un sémaphore async Albert créé paresseusement (`ALBERT_NOTES_CONCURRENCY`) et le limiteur via `to_thread`.
     - `_generate_with_llm(..., albert_api_key=None)` : résolveur avant `:412`. Branche Albert :
       - **hors** de la boucle `max_attempts` ;
       - `NOTE_MODE_MAX_TOKENS[mode]` plus la marge pour un modèle de raisonnement, avec un plancher en mode short ;
       - vide avec `length` → 1 réessai (×1.5, effort `low`), puis erreur.
     - Le repli gpt-4o-mini (`:414-420`) est réservé à OpenAI et OpenRouter.
     - `albert_api_key` en dernier kwarg (défaut `None`) de `build_note_html` (`:561`), `build_abstract_text` (`:668`) et de leurs versions async (`:799`, `:852`). Disponibilité (`:631`, `:721`) élargie.
     - `AlbertAuthError` et `AlbertQuotaExhausted` remontent à la route. Un transitoire épuisé retombe sur le template (sémantique `:651-653`).
  7. **`book_note_generator.py`.**
     - `albert_api_key` en keyword-only dans `_phase1_detect_structure` (`:824`), `_phase2_analyse_chapter` (`:1195`), `_phase3_synthesise` (`:1295`) et `build_book_note_async` (`:1482`), propagé (`:1554-1560`, `:1699-1704`, `:1721-1726`).
     - Les 3 sites de sémaphore passent par `run_llm_slot`.
     - Dans les `try/except` de `:883`, `:1705` et `:1727`, `except (AlbertAuthError, AlbertQuotaExhausted): raise` est placé **avant** `except Exception`. Ces classes n'apparaissent jamais sur le chemin OFF.
     - Estimation `len/3.2` : au-delà de 0.9 × le contexte, rôle `long_context`. Phase 1 en `json_object` si D10 le permet.
  8. **`citation_filter.py`.**
     - `_call_llm_api` (`:459`) : résolveur à `:488`. Albert : `max_tokens` 1500, ou 1500+2048 et effort `low` pour un modèle de raisonnement ; `single_attempt` ; `content None` → `AlbertTruncatedError`.
     - `pre_filter_citation` (`:587`) et `filter_citation_with_llm` (`:694`) : branche Albert via `run_llm_slot` (aucun sommeil sémaphore tenu), **hors** de la boucle `max_retries` et de l'`asyncio.sleep(2)`.
     - Le chemin OpenAI et OpenRouter reste identique.
     - Pré-filtre, branche Albert seulement : réponse lue comme un jeton exact (premier mot normalisé dans {RELEVANT, NA}, sinon la citation est gardée) ; fail-open conservé, sauf pour `AlbertAuthError` et `AlbertQuotaExhausted`, qui remontent.
     - `albert_api_key` en dernier kwarg de `pre_filter_citation` (`:517`), `filter_citation_with_llm` (`:619`) et `process_citations_parallel` (`:75`, `:347`), transmis (`:153-161`, `:185-194`).
  9. **`parallel_citation_processor.py`** : à la première `AlbertAuthError` ou `AlbertQuotaExhausted`, annulation des tâches restantes et relance de l'exception vers la route (L7 émet l'événement SSE).
- **Tests** :
  - `tests/test_albert_recode.py` :: `test_off_keys_match_golden` (G4) ; `test_off_create_kwargs_match_golden` (G2) ; `test_off_stdout_matches_golden` (G12) ; `test_albert_prefix_routes_wire_model` ; `test_no_silent_openai_fallback` ; `test_length_is_fallback_truncated_not_cached` ; `test_suspicious_ratio_not_cached` ; `test_reasoning_content_never_used` ; `test_503_twice_fallback_recorded_in_key` ; `test_404_permanent_abort_message` ; `test_account_error_abort_exit_2` ; `test_cache_key_uses_albert_provider_and_resolved_id` ; `test_harden_albert_ignores_prefer_openai` ; `test_harden_cli_albert_routes_and_keys_consistent` ; `test_harden_recode_model_albert_with_cli_gpt4omini` ; `test_skip_providers_mistral_ocr_only_by_default` ; `test_recode_concurrency_is_process_wide` ; `test_albert_run_always_emits_recode_model` ; `test_single_retry_layer_recode` ; `test_seed_only_if_deterministic`.
  - `tests/test_albert_notes.py` :: `test_off_generate_kwargs_unchanged` (G11) ; `test_run_llm_slot_without_policy_equals_executor_call` ; `test_gptoss_budget_headroom_effort` ; `test_empty_length_retry_then_error` ; `test_final_failure_raises_no_gpt4o_mini` ; `test_no_env_key_fallback` ; `test_account_error_propagates` ; `test_book_account_error_propagates` ; `test_semaphore_released_during_backoff` (`asyncio.run`) ; `test_single_retry_layer_notes` ; `test_book_three_phases_forward_key` ; `test_long_prompt_uses_long_context_model`.
  - `tests/test_albert_citation_filter.py` :: `test_none_content_no_attributeerror` ; `test_reasoning_budget_low_effort` ; `test_albert_never_calls_openai` ; `test_parallel_positional_call_without_key_still_works` ; `test_semaphore_released_during_backoff` ; `test_no_nested_acquire_deadlock` (sémaphore de taille 1, 2 citations, `asyncio.wait_for`) ; `test_single_retry_layer_citations` ; `test_account_error_aborts_job` ; `test_albert_prefilter_exact_token_parse` ; `test_off_prefilter_parse_unchanged`.
  - `tests/live/test_albert_live_llm.py` :: 3 chunks recodés avec ministral (`recoded`, `response.model` = id) ; note courte gpt-oss non vide ; filtre en JSON valide.
- **Gate** :
  - `.venv/bin/python -m pytest tests/test_albert_recode.py tests/test_albert_notes.py tests/test_albert_citation_filter.py tests/test_dedup.py tests/test_llm_note_generator.py tests/test_citation_filter.py tests/test_citations_routes.py tests/test_book_note_generator_v2.py -q -p no:cacheprovider` → 0 failed
  - `grep -nE '"/" in (eff_)?model' scripts/rad_chunk.py app/utils/llm_note_generator.py app/utils/citation_filter.py | wc -l` → 0
  - `grep -rnE "getenv[(].ALBERT_API_KEY|environ[[].ALBERT_API_KEY" app/ | wc -l` → 0
  - `ALBERT_LIVE=1 .venv/bin/python -m pytest tests/live/test_albert_live_llm.py -m albert_live -v -p no:cacheprovider` → passed
  - GC4, GC1, GC2 ; GC3 une fois pour W2 ; GC5 à GC9
- **Commit** : `Albert lot 3 : fournisseur de chat pour le recodage, les notes et le filtre de citations (sans repli silencieux, une seule couche de retry, cache indexé sur le modèle résolu)`.
- **Rollback** : `git revert` du commit L3. Sans risque tant qu'Albert est OFF. Les entrées de cache Albert restent isolées (provider `albert`).

### Lot 4 — OCR souverain en tête de chaîne (`/v1/ocr` puis LightOnOCR)

- **Objectif** : quand `OCR_ENABLE_ALBERT=1`, tenter Albert avant Mistral, avec :
  - des pages indexées à partir de 0 et des marqueurs continus ;
  - une récupération page par page ;
  - un repli tracé ;
  - une erreur de compte mémorisée pour le processus.

  Désactivé : chaîne, textes, logs et messages identiques.
- **Dépend de** : L1, L2. **Vague** : W2.
- **Fichiers possédés** : `scripts/rad_dataframe.py`, `scripts/rad_albert/ocr.py`, `tests/test_albert_ocr.py`, `tests/live/test_albert_live_ocr.py`.
- **Tâches** :
  1. **Constantes de module** après `load_dotenv_guarded` (vers `:349`). Ce sont des constantes de script CLI, patchables (décision 11), et aucun appel réseau n'a lieu à l'import.
     - `OCR_ENABLE_ALBERT = cfg.enabled and cfg.ocr_enabled`.
     - `ALBERT_API_KEY = os.getenv(...) or None` (frontière du script).
     - `ALBERT_OCR_SEMAPHORE`, distinct de `MISTRAL_SEMAPHORE` (`:49`).
     - `_ALBERT_V1_OCR_AVAILABLE: Optional[bool] = None`.
     - `_ALBERT_ACCOUNT_DISABLED: Optional[str] = None`.
  2. **Refacto pure**, sous G1, G13, `TestSplitPartialSalvage` et `TestMistralPartialFlowsToOcrResult` : extraire `_ocr_pages_payload_to_markdown(payload, page_offset=0)` de `:1274-1332` et la renumérotation de `:956-966`, puis les appeler depuis le chemin Mistral. `:1253`, `_parse_retry_after` et `_mistral_retry_wait` ne sont **pas** touchés.
  3. **`rad_albert/ocr.py`.**
     - `render_page_png_b64(page, dpi, max_side)` : motif de `:1391-1404`, avec `zoom = min(dpi/72, max_side/max(w,h))`.
     - `ocr_pdf_lightonocr(...)`, rendu :
       - un seul `fitz.Document`, rendu séquentiel ;
       - HTTP dans un pool à fenêtre bornée, sous le sémaphore seulement pendant l'envoi.
     - `ocr_pdf_lightonocr(...)`, assemblage et échecs :
       - `<!-- Page N -->` pour chaque page, y compris vide ;
       - une page en échec reçoit son marqueur suivi de `<!-- OCR ÉCHOUÉ (albert_lightonocr) : raison -->`, qui ne correspond pas à `PAGE_MARKER_RE` (`book_note_generator.py:107`) ;
       - `finish_reason=length` rend le résultat partial ;
       - au-delà de `ALBERT_OCR_MAX_FAILED_RATIO`, exception et passage au maillon suivant ;
       - `AlbertAuthError` et `AlbertQuotaExhausted` abandonnent le maillon et sont remontées pour mémorisation.
     - `ocr_pdf_v1(pdf_path, client, *, split_fn, unlink_fn, parse_fn, …)` :
       - data URI, `pages = list(range(0, n))` (ne jamais copier `:1253`) ;
       - découpage via `_split_pdf_for_ocr` (`:709`) et `_compress_pdf_for_ocr` (`:666`), injectés ;
       - 403 ou 404 : indisponibilité mémorisée, bascule sur LightOnOCR ;
       - 413 : un seul redécoupage plus fin, puis LightOnOCR.
  4. **`_try_albert_ocr`**, inséré **avant** le bloc Mistral (`:1637`), sous `if OCR_ENABLE_ALBERT and ALBERT_API_KEY and not _ALBERT_ACCOUNT_DISABLED:`, avec import paresseux.
     - Une erreur de compte ou de quota pose `_ALBERT_ACCOUNT_DISABLED` avec un seul log WARNING. Les documents suivants sautent le maillon sans appel.
     - `_finalize_ocr_result` (`:1525`) ; `_ocr_density_warning` (`:1562`) pour LightOnOCR seulement ; plafond (`:1679-1711`) si `ALBERT_OCR_MAX_PAGES>0`.
     - `provider_used`, `last_error` et `track_error` comme en `:1652-1661`.
     - Le message final (`:1802-1806`) n'est complété que si le maillon est actif.
     - Sonde d'accès mise en cache sur le motif de `_local_ocr_available` (`:1438-1463`).
  5. **`OCRResult` (`:568`)** : nouveau dernier champ `fallback_from: Optional[str] = None`. G1 ne compare que les 6 champs de base via `getattr`. `_process_single_zotero_item` (`:2085-2106`) ajoute une entrée `OCR_PROVIDER_FALLBACK` quand ce champ est posé, ce qui n'arrive jamais quand Albert est OFF.
- **Tests** :
  - `tests/test_albert_ocr.py`, chaîne et byte-identité : `test_off_no_albert_call_even_with_key`, `test_off_final_error_message_identical`, `test_off_logs_match_golden` (G13), `test_mistral_payload_parser_bytes_unchanged`, `test_ocrresult_never_unpacked_positionally` (AST sur `scripts/` et `app/`), `test_on_albert_runs_before_mistral`, `test_albert_fail_then_mistral_recorded_fallback`, `test_auth_error_falls_through_without_retry`, `test_account_error_memoized_skips_next_documents`.
  - `tests/test_albert_ocr.py`, `/v1/ocr` (fixtures réelles ou synthétiques) : `test_doc_ocr_pages_zero_based`, `test_doc_ocr_markers_index_plus_one`, `test_doc_ocr_403_memoized_no_second_probe`, `test_413_resplit_then_lightonocr`.
  - `tests/test_albert_ocr.py`, LightOnOCR : `test_lightonocr_message_image_only_params`, `test_rasterization_a4_max_side_le_1541`, `test_page_markers_continuous_with_failed_page`, `test_finish_length_marks_partial`, `test_failed_ratio_raises_chain_continues`, `test_404_single_request`, `test_semaphore_released_during_backoff`, `test_density_only_lightonocr`, `test_partial_creates_ocr_partial_entry`.
  - `tests/live/test_albert_live_ocr.py` :: LightOnOCR sur un PDF de 2 pages (`PAGE_ONE_7Q`) ; `/v1/ocr` `pages:[0]` (ignoré si `decisions.json` indique D3 = pas d'accès) ; chaîne complète avec provider `albert_*`.
- **Gate** :
  - `.venv/bin/python -m pytest tests/test_albert_ocr.py tests/test_ocr_providers.py tests/test_rad_dataframe_pages_split.py tests/test_rad_dataframe_epub.py tests/test_ocr_local.py tests/test_book_note_generator_v2.py -q -p no:cacheprovider` → 0 failed
  - `grep -c 'range(1, max_pages + 1)' scripts/rad_dataframe.py` → 1 (la ligne Mistral seule)
  - `ALBERT_LIVE=1 .venv/bin/python -m pytest tests/live/test_albert_live_ocr.py -m albert_live -v -p no:cacheprovider` → passed. Un 404 sur `lightonocr-2-1b` (fin de test prévue le 2026-10-01) déclenche l'arrêt avec escalade, conformément à la règle d'arrêt.
  - GC4, GC1, GC2, GC5 à GC9 (GC3 partagée par W2)
- **Commit** : `Albert lot 4 : OCR souverain en tête de chaîne (Mistral OCR via /v1/ocr si accès, sinon LightOnOCR), repli tracé, arrêt sur erreur de compte`.
- **Rollback** : `git revert` du commit L4. Le chemin Mistral revient exactement à l'état L2.

### Lot 5 — Collections Albert, 4e cible vectorielle

- **Objectif** : un envoi idempotent vers une collection privée (embeddings côté serveur), conforme aux limites de l'API, tracé par un manifeste écrit au fil de l'eau et intégré à la dédup Tier 2. Le contrat stdout et les 3 connecteurs existants restent inchangés.
- **Dépend de** : L1, L2. **Vague** : W2.
- **Fichiers possédés** : `scripts/rad_vectordb.py`, `scripts/rad_albert/collections.py`, `scripts/rad_dedup.py` (`scripts/rad_dedup.py=@docstrings` dans `restrict_L5.txt`), `tests/test_albert_vectordb.py`, `tests/load/bench_albert.py`, `tests/live/test_albert_live_collections.py`.
- **Tâches** :
  1. **`collections.py`.**
     - `DEFAULT_METADATA_FIELDS` (10 champs) et `effective_metadata_fields(env)`, qui appelle `config.validate_metadata_fields` avec `DedupConfig.from_env().meta_fields`. Toute surcharge incomplète est refusée avant tout appel réseau.
     - `albert_scalar` : écarte `''`, `None`, NaN et inf ; float entier converti en int ; `|v|>1e16` converti en chaîne ; listes jointes ; troncature à 255.
     - `sanitize_metadata` : jamais `path` ; `doi`, sinon `url` sous la même clé.
     - `document_name_for` = `ragpy:{itemKey}:{basename(filename)}`, repli sur le titre, hash au-delà de 255 caractères.
     - `content_id_for` : `chunk['id']` s'il est adressé par contenu. Sinon `rad_dedup.content_id(chunk['content_hash'], chunk_index)` si le hash existe. Sinon `rad_dedup.content_id(rad_dedup.content_hash(chunk['text']), chunk_index)`.
       - Ce dernier cas couvre `DEDUP_ENABLED=0`, où l'id vaut `{doc_id aléatoire}_{idx}`.
       - Limite documentée : un nouveau chunking recodé produit d'autres textes, donc d'autres ids.
     - `manifest_row`.
     - `resolve_albert_input(session_dir)` : sparse, puis `with_embeddings`, puis `output_chunks.json`. Réutilisé par la route et par Celery (L7).
  2. **`_AlbertDedupAdapter`**, à côté de `_PineconeDedupAdapter` (`:467-501`).
     - `preload()` est appelé **avant** `_run_dedup` (`:624`), hors de `dedup_filter` (qui avale les exceptions).
     - Correspondance exacte du nom côté client. Plusieurs collections du même nom → erreur listant les ids (D14).
     - `list_chunks` paginé. `existing()` réindexe par `content_hash` et réhydrate `DEDUP_META_FIELDS`.
     - Pas de `nearest`, donc le Tier 3 est sauté (`rad_dedup.py:515`). Un échec du preload fait échouer l'envoi bruyamment.
  3. **`insert_to_albert(embeddings_json_file, collection_id=None, collection_name=None, albert_api_key=None, *, create_collection=False, ack_retention=False, transport=None) -> dict`**, préparation :
     - erreurs renvoyées en dict `_vectordb_result('error')` (`:711`), jamais d'exception ;
     - get-or-create d'une collection privée ;
     - `embedding` et `sparse_embedding` retirés ;
     - preload, puis `_run_dedup`, puis append-skip par `content_id_for`, **toujours actif**.
  4. **`insert_to_albert`**, envoi, rollback et retour :
     - regroupement par document, puis `add_chunks` par 64 sous `ALBERT_PUSH_CONCURRENCY` et le limiteur de rôle `embed` (D19) ;
     - échec d'une tranche d'un document créé pendant le run → `delete_document`, puis `partial_error` ;
     - `albert_manifest.jsonl` : une ligne par tranche réussie, écrite en **append + flush** (ids de chunks si D16 les renvoie, sinon document, tranche et nombre) ;
     - retour enrichi de `existing_count` et `manifest_path`.
     - écritures non idempotentes (point reporté de W1) : `create_collection`, `create_document` et `add_chunks` ne sont jamais rejoués après un échec ambigu (`AlbertUncertainWriteError`) ; l'envoi réconcilie par relecture (préchargement des chunks existants) avant toute nouvelle tentative, et une relance reste sûre (append-skip).
  5. **CLI (`:1176-1309`).**
     - `albert` ajouté aux choix (`:1189`), plus `--albert-collection-id`, `--albert-collection-name`, `--albert-create-collection` et `--albert-ack-retention`. Aucune de ces options ne partage de préfixe avec `--class`, `--tenant`, `--index` ou `--namespace` (piège d'abréviation, `processing.py:843`).
     - La clé est lue dans l'env.
     - Bloc Result inchangé, plus `Skipped (existing): N` et `Albert manifest: <chemin>` **pour albert seulement**.
- **Tests** :
  - `tests/test_albert_vectordb.py` :: `test_sanitize_rules` ; `test_document_name_stable_unique` ; `test_content_id_reuses_rad_dedup` ; `test_rerun_idempotent_without_dedup_fields` ; `test_metadata_override_must_keep_content_id` ; `test_collection_get_or_create_exact_name_paginated` ; `test_duplicate_collection_names_error` ; `test_public_refused_private_forced` ; `test_document_multipart_no_metadata` ; `test_chunks_sliced_64` (130 chunks → 3 POST) ; `test_no_vectors_in_payloads` ; `test_rerun_idempotent_zero_post` ; `test_slice_failure_deletes_new_document` ; `test_manifest_written_per_slice` (échec simulé à la 2e tranche : la 1re ligne est présente) ; `test_preload_failure_is_loud` ; `test_tier2_rehydrates_truncated_title` ; `test_stdout_extra_lines_albert_only` ; `test_cli_requires_ack_retention` ; `test_argparse_class_prefix_still_resolves` ; `test_resolve_albert_input_order` ; `test_pinecone_result_block_equals_golden` (G5).
  - `tests/load/bench_albert.py` :: POST de chunks = `ceil(n/64)` par document ; 0 appel HTTP en Tier 2 après le preload ; 0 appel à `search` ou `embeddings`.
  - `tests/live/test_albert_live_collections.py` :: envoi vers `ragpy-probe-<ts>` ; relance → `Inserted: 0` et `Skipped (existing): 5` ; recherche lexicale après l'attente D18 ; suppression par finaliseur.
- **Gate** :
  - `.venv/bin/python -m pytest tests/test_albert_vectordb.py tests/test_dedup.py scripts/test_rad_vectordb.py tests/load/bench_albert.py tests/load/bench_dedup.py -q -p no:cacheprovider` → 0 failed
  - `.venv/bin/python scripts/rad_vectordb.py --help | grep -cE '^  --albert-'` → 4
  - `ALBERT_LIVE=1 .venv/bin/python -m pytest tests/live/test_albert_live_collections.py -m albert_live -v -p no:cacheprovider && ALBERT_LIVE=1 .venv/bin/python scripts/albert_probe.py --cleanup --dry-run` → passed, puis `ragpy-probe restantes: 0`
  - GC4, GC1, GC2, GC5 à GC9
- **Commit** : `Albert lot 5 : collections Albert comme cible vectorielle (idempotence par content_id, métadonnées en liste blanche, rollback, manifeste incrémental, acquittement de rétention)`.
- **Rollback** : `git revert` du commit L5. Les collections de test sont supprimées par `--cleanup`. Les collections réelles des utilisateurs ne sont jamais touchées.

### Lot 6 — Embeddings bge-m3 (1024 d) et gardes d'espace vectoriel

- **Objectif** : `EMBEDDING_PROVIDER=albert` sans jamais mélanger les espaces. Le fichier et la requête OpenAI restent identiques à l'octet. Le Tier 3 n'est jamais appliqué à bge-m3 avec un seuil non calibré.
- **Dépend de** : L3 (`rad_chunk` transmis), L5 (`rad_vectordb` transmis). **Vague** : W3.
- **Fichiers possédés** : `scripts/rad_chunk.py`, `scripts/rad_vectordb.py`, `scripts/rad_clustering.py`, `scripts/rebuild_pinecone_index.py`, `tests/test_albert_embeddings.py`, `tests/test_vector_space_guards.py`, `tests/live/test_albert_live_embed.py`.
- **Tâches** :
  1. **`get_embeddings_batch` (symbole ; `:591` en baseline)** reçoit un kwarg `space=None`.
     - Chemin OpenAI **strictement** `create(input, model, timeout=60.0)`, sans `encoding_format`, zéros à 3072 conservés.
     - Chemin Albert : `AlbertClient.embed`. Texte vide → `None` ; échec → `None`. Jamais de zéros, jamais OpenAI.
  2. **`_embed_with_cache`** : `space` transmis depuis `process_chunks_for_embedding`. Clé Albert = `embed_key(t, "albert:bge-m3", {model, provider, dim, norm})`. Un HIT de mauvaise dimension compte comme MISS, et `None` n'est jamais mis en cache. Chemin OpenAI inchangé.
  3. **`generate_and_save_embeddings`.**
     - `EmbeddingConfig.from_env` résolu une fois. `albert` avec `ALBERT_ENABLED≠1` ou une valeur inconnue donne un message français et `exit 2`.
     - Preflight `embed`. Lot = `min(DEFAULT_EMBEDDING_BATCH_SIZE, 64)`.
     - Zéros seulement pour OpenAI. Libellé des métriques = `space.model`. Champs d'espace seulement hors défaut.
     - Ratio de manques au-dessus du seuil : fichier écrit, puis `exit 1` avec un message français. Erreur de compte : `exit 2`.
     - Ledger si Albert a été appelé.
     - CLI `--embedding-provider {openai,albert}`. La garde de phase dense n'exige OpenAI que pour l'espace OpenAI.
  4. **`rad_vectordb`.**
     - `_check_vector_space` (via `check_uniform_space`) est appelé **avant** dédup et upsert :
       - Pinecone : `.dimension` lue sur l'appel `describe_index` déjà existant, seulement si c'est un int ;
       - Qdrant : collection existante → `config.params.vectors.size` (motif `settings.py:452`) ; nouvelle collection → dimension uniforme hors vecteurs nuls ;
       - Weaviate : `fetch_objects(limit=1, include_vector=True)` **seulement** pour un espace hors défaut ;
       - fichier mixte refusé localement.
     - `_run_dedup` : si l'espace du fichier est `albert` et que `DEDUP_SIM_THRESHOLD_BGE_M3` est vide, le Tier 3 est désactivé pour ce run, avec un WARNING unique. S'il est défini, ce seuil s'applique. Pour un fichier sans champs d'espace (défaut), rien ne change.
  5. **`rad_clustering.py`** (`:99-131`, `:567`) : message clair sur des dimensions mélangées ; vecteurs nuls ou `None` ignorés seulement hors défaut. **`rebuild_pinecone_index.py`** (`:160/:197`) : contrôle préalable des dimensions, `exit 2` **avant** `recreate_index`.
- **Tests** :
  - `tests/test_albert_embeddings.py` :: `test_openai_request_and_output_identical` (G3) ; `test_off_stdout_matches_golden` (G12) ; `test_albert_130_texts_three_posts` ; `test_albert_sorted_by_index` ; `test_request_float_no_dimensions` ; `test_empty_text_not_sent` ; `test_albert_never_writes_zero_vectors` ; `test_missing_ratio_exit_1` ; `test_account_error_exit_2` ; `test_albert_provider_when_disabled_exit_2` ; `test_space_fields_only_non_default` ; `test_albert_cache_namespaced_with_norm` ; `test_wrong_dim_hit_is_miss` ; `test_openai_embed_key_unchanged` (G4) ; `test_albert_never_calls_openai_embeddings` ; `test_batch_clamped_64` ; `test_cli_flag_sets_env` ; `test_no_usage_file_when_off`.
  - `tests/test_vector_space_guards.py` :: `test_pinecone_dim_mismatch_before_upsert` ; `test_pinecone_non_int_dimension_proceeds` ; `test_qdrant_existing_size_mismatch` ; `test_qdrant_new_collection_uniform_dim` ; `test_weaviate_fetch_only_non_default` ; `test_mixed_file_refused_offline` ; `test_rebuild_preflight_exit_2` ; `test_clustering_mixed_dims_clear_error` ; `test_clustering_legacy_zero_unchanged` ; `test_tier3_disabled_for_bge_m3_by_default` ; `test_tier3_bge_m3_explicit_threshold`.
  - `tests/live/test_albert_live_embed.py` :: 3 textes → 1024 d, norme conforme à D12 ; phase dense sur 5 chunks.
- **Gate** :
  - `.venv/bin/python -m pytest tests/test_albert_embeddings.py tests/test_vector_space_guards.py tests/test_dedup.py tests/test_clustering.py scripts/test_rad_vectordb.py tests/test_albert_vectordb.py tests/test_albert_recode.py -q -p no:cacheprovider` → 0 failed
  - `.venv/bin/python scripts/rad_chunk.py --help | grep -cE '^  --embedding-provider'` → 1
  - `ALBERT_LIVE=1 .venv/bin/python -m pytest tests/live/test_albert_live_embed.py -m albert_live -v -p no:cacheprovider` → passed
  - GC4, GC1, GC2 ; GC3 une fois pour W3 ; GC5 à GC9
- **Commit** : `Albert lot 6 : embeddings bge-m3 dans un espace de 1024 dimensions séparé, gardes d'espace vectoriel des connecteurs, Tier 3 désactivé pour bge-m3`.
- **Rollback** : `git revert` du commit L6. Les fichiers bge-m3 déjà produits restent marqués par leur espace.

### Lot 7 — Câblage : routes, SSE, endpoints de collections, UI, Celery

- **Objectif** : exposer les 4 capacités dans les routes et l'interface quand Albert est ON. OFF : réponses HTTP et SSE, argv, env, délais et HTML identiques.
- **Dépend de** : L3, L4, L5 (et L1, L2). **Vague** : W3.
- **Fichiers possédés** :
  - routes : `app/routes/processing.py`, `app/routes/citations.py`, `app/routes/settings.py`, `app/routes/pipeline.py`, `app/routes/celery_tasks.py` ;
  - templates : `app/templates/index.html`, `app/templates/user/project_detail.html` ;
  - Celery : `app/tasks/extraction.py`, `app/tasks/chunking.py`, `app/tasks/embeddings.py`, `app/tasks/vectordb.py`, `app/tasks/runner.py` ;
  - tests : `tests/test_albert_routes.py`, `tests/test_albert_settings_routes.py`, `tests/test_albert_ui.py`.
- **Tâches** :
  0. **Défaut préexistant à corriger (découvert par les goldens du lot 0).** Les routes SSE `initial_text_chunking_sse` et `dense_embedding_generation_sse` lèvent un `NameError` au lieu d'émettre l'événement d'erreur d'identifiant pour un non-admin sans clé (et pour un slug OpenRouter sans clé). Le golden `g7_sse_credential_error_known_defect.json` accepte exactement deux issues : le crash figé, ou un unique événement `{type:error, message, credential_required}` identique au 403 de la route JSON jumelle, sans sous-processus. Corriger ici pour obtenir la seconde issue.
  1. **Délai.** `_subprocess_timeout(default, albert_selected)` renvoie `ALBERT_SUBPROCESS_TIMEOUT` si Albert est sélectionné, sinon la valeur actuelle littérale (1800 ou 3600). Il est appliqué aux sites de `run_tracked_subprocess` et de `run_subprocess_with_sse` concernés, et le même calcul vaut dans `runner.run_script`.
  2. **OCR** (`processing.py:285-306` et `:579-596`) : si `ocr_active` (Albert ON, `OCR_ENABLE_ALBERT=1`, clé disponible), 3e essai `required_keys=['albert_api_key']` avant le 403, avec le délai Albert. Sinon, payload et délai identiques.
  3. **Chunking** (`:446-472`, `:1301-1315`) : `resolve_llm_provider`, puis `required_keys=[credential_key]`. `AlbertDisabledError` → 400 JSON ou événement SSE d'erreur, **sans** sous-processus.
  4. **Dense** (`:624-660`, `:1349-1380`) : `embedding_provider: str = Form(None)`.
     - OFF : le champ est ignoré, sauf la valeur `albert`, qui donne un 400.
     - ON : champ du formulaire, sinon `EMBEDDING_PROVIDER` du serveur. Une valeur hors {openai, albert} donne un 400. `required_keys` est déduit du fournisseur, et `subprocess_env['EMBEDDING_PROVIDER']` est posé explicitement (openai comme albert).
  5. **Notes SSE** (`:995-1006`, `:1106-1188`) : modèle **effectif** = champ du formulaire, sinon `resolve_default_llm_model()`. Il est validé par le résolveur, et la clé est récupérée par `get_credential_or_env`, puis transmise. `AlbertAuthError` ou `AlbertQuotaExhausted` → événement `{type:error, message, credential_required}`, puis arrêt.
  6. **upload_db** (`:780-904`) : la branche albert est traitée **avant** la vérification du fichier sparse (`:800-805`), **seulement si Albert est ON**. Sinon, code actuel et même 400 (G7).
     - Entrée via `resolve_albert_input`.
     - `albert_gdpr_ack=true` exigé, sinon 400 `gdpr_ack_required`.
     - Champs de collection, `required_keys=['albert_api_key']`, flags CLI, délai Albert.
     - Lecture de `Skipped (existing)`, du manifeste (ligne entière) et, **pour la branche albert seulement**, de `^Dedup journal:\s*(.+)$` (ligne entière). Les 3 connecteurs existants gardent la regex `(\S+)`.
     - Dans un `finally` : copie du manifeste (même partiel) vers `data/albert_manifests/<user_id>/<session>-<ts>.jsonl`.
     - `create_audit_log(db, action="ALBERT_COLLECTION_UPLOAD", user_id=…, resource_type="albert_collection", details={collection_id, collection_name, session, gdpr_ack: true})`.
  7. **Citations** (`citations.py:323-335`, `:1056-1067`, `:1588-1599`, formes conservées, y compris le 400 de `:1592`).
     - Résolveur, puis clé transmise (`:369-381`, `:1174-1185`, `:1670-1682`).
     - Sur le chemin Albert, le `async with semaphore` externe de `:1173` est retiré, puisque `filter_citation_with_llm` acquiert en interne. Le chemin OFF est inchangé.
     - `AlbertAuthError` levée par `process_citations_parallel` → événement SSE d'erreur et arrêt du job.
     - `upload_pop_json` (`:84`) valide le modèle par le résolveur avant d'écrire `config.json` (`:165-172`). `albert/` avec Albert OFF donne un 400.
  8. **`settings.py`** : `GET /api/albert/collections` (privées, paginées) et `DELETE /api/albert/collections/{id}?confirm=true`, en 404 si OFF. La suppression est journalisée par `create_audit_log`. **`pipeline.py`** (`:737-746`) : `embedding_*` exposés seulement s'ils sont présents.
  9. **Celery.**
     - `submit_vectordb_upload_task` : si `db_choice == 'albert'` et Albert ON, `resolve_albert_input` est appelé **avant** la vérification sparse (`:306-312`) et l'ack est exigé. Sinon, le code actuel (sparse, puis liste blanche) est inchangé.
     - Branche albert dans `tasks/vectordb.py`, **sans autoretry**.
     - `required_keys` via le résolveur (chunking) ; 3e essai albert (extraction) ; `EMBEDDING_PROVIDER` résolu comme dans la route (dense) ; délais selon la tâche 1.
  10. **Templates** (gardes Jinja en ligne). Dans `index.html` :
      - aides de modèle (`:1167-1171`, `:1407-1411`) et `<datalist>` alimentée par `/api/albert/models` ;
      - `<select embedding_provider>` ajouté au `FormData` (`:2132`) s'il existe ;
      - `<option value="albert">` (`:1272`) et `#albertParams` (clone de `fetchPineconeIndexes`, `:2310-2357`) ;
      - confirmation RGPD : rétention jusqu'à suppression, collection privée, DINUM art. 28, droits d'auteur et données personnelles ;
      - bouton de suppression de collection ;
      - déverrouillage anticipé de la section base réservé à albert (`:1721`, `:2267`, `:2297`).

      `project_detail.html:268-274` : aide `albert/<modèle>`. Aucun `fetch` vers l'hôte Albert (CORS).
- **Tests** :
  - `tests/test_albert_routes.py` :: `test_off_payloads_equal_golden` (G7 étendu) ; `test_albert_prefix_disabled_400_no_subprocess` ; `test_upload_pop_json_albert_off_400` ; `test_albert_timeout_only_when_selected` ; `test_chunking_albert_nonadmin_403_even_with_env_key` ; `test_chunking_albert_env_has_user_key` ; `test_ocr_albert_only_user_allowed_when_active` ; `test_dense_albert_env_and_keys` ; `test_dense_server_default_albert_requires_albert_key` ; `test_dense_invalid_provider_400` ; `test_notes_sse_albert_key_forwarded_and_account_error_event` ; `test_notes_empty_model_uses_effective_default_for_credential` ; `test_upload_db_albert_input_order_and_args` ; `test_upload_db_albert_requires_gdpr_ack` ; `test_upload_db_albert_parses_existing_manifest_and_journal_with_spaces` ; `test_upload_db_albert_audit_and_manifest_copy` ; `test_upload_db_other_dbs_unchanged` ; `test_citations_three_sites_forward_key` ; `test_citations_albert_no_outer_semaphore` ; `test_citations_sse_account_error_event` ; `test_celery_albert_gated_and_never_autoretried` ; `test_celery_albert_accepts_output_chunks`.
  - `tests/test_albert_settings_routes.py` :: `test_collections_404_when_off` ; `test_collections_403_without_key` ; `test_collections_private_paginated` ; `test_delete_collection` (confirm exigé, audit) ; `test_key_not_in_urls_or_logs`.
  - `tests/test_albert_ui.py` :: `test_off_html_equals_golden` (G9) ; `test_on_blocks_present` ; `test_on_gdpr_confirm_present` ; `test_key_never_rendered` ; `test_no_albert_host_in_templates`.
- **Gate** :
  - `.venv/bin/python -m pytest tests/test_albert_routes.py tests/test_albert_settings_routes.py tests/test_albert_ui.py tests/test_celery_tasks.py tests/test_citations_routes.py tests/test_credentials_ui_sync.py app/test_main.py -q -p no:cacheprovider` → 0 failed
  - `grep -nE "'/' in model|\"/\" in model_name" app/routes/processing.py app/routes/citations.py | wc -l` → 0
  - `grep -n 'Query(' app/routes/settings.py | grep -ci albert` → 0
  - GC4, GC1, GC2, GC5 à GC9 (GC3 partagée par W3)
- **Commit** : `Albert lot 7 : routes et interface (sélection du fournisseur, clé Albert selon la capacité, délais dédiés, collections avec confirmation de rétention, audit et suppression, branche Celery)`.
- **Rollback** : `git revert` du commit L7. Les capacités restent utilisables en CLI.

### Lot 8 — Documentation, `.env.example`, suite live, E2E souverain

- **Objectif** : documenter l'opt-in, la sécurité, les quotas, le RGPD et le cycle de vie des modèles. Prouver le fonctionnement réel de bout en bout par les vraies routes, puis nettoyer.
- **Dépend de** : L6, L7. **Vague** : W4.
- **Fichiers possédés** :
  - `.env.example`, `scripts/.env.example`, `CLAUDE.md`, `docs/GUIDE.md`, `docs/albert.md`, `README.md` ;
  - `tests/test_albert_docs_sync.py`, `tests/live/test_albert_live_e2e.py`, `scripts/albert_e2e_smoke.py` ;
  - `.claude/tasks/albert_probe_report.md`, `docs/SPRINT_albert.md` ;
  - **si la re-sonde l'exige** (ajoutés explicitement à `owned_L8.txt` par l'coordinateur, et appliqués par le gardien du socle) : `scripts/rad_albert/catalog.py`, les `tests/fixtures/albert/P*.json` concernés, `tests/fixtures/albert/decisions.json`, `tests/test_albert_config.py`, et `scripts/rad_chunk.py` si D21 conclut à sauter le recodage.
- **Tâches** :
  1. **`.env.example`.**
     - Bloc `# ===== ALBERT API (DINUM) — OPT-IN, OFF PAR DÉFAUT =====` placé **avant** `# Flower monitoring credentials` (`:151`). Il est généré depuis `ENV_REGISTRY`, avec des commentaires français (« 0 = comportement strictement identique »).
     - `ALBERT_API_KEY` vide. **Jamais** `ALBERT_LIVE`. Ne pas réordonner ; garder l'absence de saut de ligne final.
     - `scripts/.env.example` : ajout d'une seule ligne de commentaire en tête, qui renvoie à `.env.example` racine pour les variables Albert. Aucune suppression.
  2. **Documentation**, sans ligne d'attribution ni mention d'outil d'assistance.
     - `CLAUDE.md` et `docs/GUIDE.md` : chaîne OCR, `RECODE_SKIP_PROVIDERS`, espaces 3072 et 1024, 4e cible, convention `albert/`, `RAGPY_DOTENV_DENY`, Celery authentifié en sous-processus, commandes live, rebuild Docker nécessaire.
     - `docs/albert.md` :
       - activation par capacité ;
       - quotas partagés par compte, calcul de débit et `ALBERT_SUBPROCESS_TIMEOUT` ;
       - échéances du 2026-10-01 et du 2026-12-01 ;
       - RGPD : 24 mois, collections conservées jusqu'à suppression, art. 28, éligibilité, suppression des collections **avant** la suppression d'un compte utilisateur ;
       - rotation annuelle des clés ;
       - suppression à partir des manifestes de `data/albert_manifests/` ;
       - Tier 3 désactivé pour bge-m3 ;
       - lancement local avec `.venv/bin` en tête du PATH (les routes lancent `python3`).
     - `README.md` : un paragraphe.
  3. **`scripts/albert_e2e_smoke.py`, préparation.**
     - Corpus fitz : 2 PDF texte, 1 PDF image seule, plus un JSON Zotero zippé.
     - Base SQLite temporaire (`DATABASE_URL`). Schéma via `init_db` (`app/database/init_db.py:104`). Insertion SQLAlchemy **avant** le démarrage :
       - un admin ;
       - deux non-admins A et B, actifs et vérifiés ;
       - un projet appartenant à A.
     - Tokens frappés par `create_access_token` (`app/core/security.py:54`), avec le même `JWT_SECRET_KEY`.
     - Serveur `uvicorn` sur un port libre. Son env contient :
       - `PATH=<repo>/.venv/bin` + `os.pathsep` + `PATH` ;
       - `DATABASE_URL` ;
       - `ALBERT_ENABLED=1` et `OCR_ENABLE_ALBERT=1`, dans l'env du processus seulement.
     - Au démarrage : `shutil.which('python3', path=<PATH du serveur>)` doit pointer dans `.venv/bin`, sinon `exit 2` avec un message.
     - A enregistre sa clé par `PUT /users/me/credentials` (clé lue par `dotenv_values`, jamais affichée). B n'en a pas.
     - Upload du corpus par `POST /projects/{project_id}/upload_zip` (`pipeline.py:215`). Le dossier de session renvoyé est retenu pour le nettoyage.
  4. **`albert_e2e_smoke.py`, contrôles**, avec la route et l'assertion de chacun :
     - **ON (7), en tant que A** :

       | # | Route | Assertion |
       |---|---|---|
       | 1 | `process_dataframe_sse` | `texteocr_provider` du scanné commence par `albert_` ; marqueurs `<!-- Page N -->` continus |
       | 2 | `initial_text_chunking_sse` avec `albert/ministral-3-8b-instruct-2512` | `recode_model` commence par `albert/` avec l'id résolu ; `recode_status` présent |
       | 3 | `dense_embedding_generation_sse` avec `embedding_provider=albert` | `embedding_dim=1024` partout, aucun vecteur nul |
       | 4 | `upload_db` albert (création de `ragpy-probe-e2e-<ts>`, ack) | `Inserted: N>0` ; manifeste copié dans `data/albert_manifests/` ; entrée d'audit |
       | 5 | `upload_db` relancé | `Inserted: 0` et `Skipped (existing): N` |
       | 6 | `generate_zotero_notes_sse` avec `albert/gpt-oss-120b`, mode short | la note n'est pas le template ; `albert_usage.jsonl` contient le modèle servi et le coût |
       | 7 | `GET /api/albert/collections`, puis `DELETE …?confirm=true` | la collection est listée, puis absente |

       La partie « journal » du contrôle 6 n'est pas vérifiable avant la correction du lot 9 (tâche 0) : la route des notes n'écrit pas `<session>/albert_usage.jsonl` (seuls `rad_dataframe`, `rad_chunk` et `rad_vectordb` écrivent ce journal ; le ledger en mémoire de l'`AlbertClient` des notes est perdu). Assouplissement provisoire dans `scripts/albert_e2e_smoke.py`, à confirmer par l'utilisateur : `NOTES_LEDGER_REQUIRED = False` rend l'absence d'enregistrement de notes informative (consignée dans le détail du contrôle 6, sans échec) ; des enregistrements de notes sans modèle servi ni coût font toujours échouer le contrôle. Le lot 9 repasse la constante à `True` avec la correction.

     - **Isolation (2)** :
       1. B lance le chunking `albert/…` → 403 `credential_required=albert_api_key`, alors que le `.env` admin contient la clé.
       2. `GET /users/me/credentials` de A renvoie une valeur masquée, sans la clé brute ; celui de B ne contient aucune valeur albert.
     - **OFF (3)**, après redémarrage avec `ALBERT_ENABLED=0` :
       1. Aucune occurrence de `albert` (insensible à la casse) dans le HTML d'index et de profil.
       2. `/api/albert/status` en 404, et chunking `albert/…` en 400.
       3. `GET /get_credentials` (admin) sans clé `ALBERT_API_KEY`.
     - **Fin, dans un `finally`** : suppression des collections `ragpy-probe-e2e-*`, des dossiers de session créés sous `uploads/`, de la base temporaire et des copies de manifestes E2E. Résumé JSON dans `data/e2e_albert/`.
  5. **Re-sondes.** Si la date est au moins le 2026-10-01 : `albert_probe.py --only P2,P5,P21 --out data/albert_probe --diff-against tests/fixtures/albert`. Les fixtures commitées ne sont jamais écrasées directement.
     - Si le diff impose un changement : le gardien du socle met à jour `catalog.py`, les fixtures concernées et `decisions.json`, dans les fichiers ajoutés à `owned_L8`. Les tests L1 et L3 sont rejoués dans la gate.
     - D21 est confirmé avec l'utilisateur d'après la qualité OCR du PDF scanné.
- **Tests** :
  - `tests/test_albert_docs_sync.py` :: `test_every_registered_var_in_env_example` ; `test_every_albert_env_read_in_code_is_registered` (AST) ; `test_albert_live_absent_from_env_examples` (les 2 fichiers) ; `test_albert_defaults_off`.
  - `tests/live/test_albert_live_e2e.py` :: CLI lancées en sous-processus avec `sys.executable`, jamais `"python3"` (OCR, recodage, dense, envoi, relance).
- **Gate** :
  - `.venv/bin/python -m pytest tests/test_albert_docs_sync.py -q -p no:cacheprovider` → passed ; `grep -c ALBERT_LIVE .env.example scripts/.env.example` → `.env.example:0` et `scripts/.env.example:0` ; `git diff HEAD -- .env.example scripts/.env.example | grep -c '^-[^-]'` → 0
  - Si la re-sonde a modifié le catalogue : `.venv/bin/python -m pytest tests/test_albert_config.py tests/test_albert_client.py tests/test_albert_recode.py tests/test_albert_notes.py -q -p no:cacheprovider` → 0 failed
  - `ALBERT_LIVE=1 .venv/bin/python -m pytest tests/live -m albert_live -v -p no:cacheprovider` → tous passed (au plus 1 skip motivé pour `/v1/ocr`)
  - `ALBERT_LIVE=1 .venv/bin/python scripts/albert_e2e_smoke.py --corpus-size 3 --cleanup` → `E2E ALBERT OK (ON 7/7, OFF 3/3, isolation 2/2)` avec `NOTES_LEDGER_REQUIRED = False` : la partie « journal » du contrôle 6 reste informative, donc cette gate n'est **pas** atteignable au lot 8 telle que spécifiée (contrôle 6 complet). La version complète (`NOTES_LEDGER_REQUIRED = True`, journal exigé) relève de la gate du lot 9, après la correction de la tâche 0 ; sans cette correction, elle donnerait au mieux `E2E ALBERT ÉCHEC (ON 6/7, OFF 3/3, isolation 2/2)`
  - `ALBERT_LIVE=1 .venv/bin/python scripts/albert_probe.py --cleanup --dry-run` → `ragpy-probe restantes: 0`
  - `docker compose build ragpy` → code de sortie 0
  - GC4, GC1, GC2, GC3, GC5 à GC9
- **Commit** : `Albert lot 8 : documentation (activation, sécurité, quotas, RGPD, cycle de vie des modèles), variables d'environnement d'exemple, suite live et test de bout en bout`.
- **Rollback** : `git revert` du commit L8 (documentation, outils et éventuelles données de catalogue).

### Lot 9 — Revue adversariale finale et corrections

- **Objectif** : trouver et confirmer par un test les défauts restants sur `main..feat/albert`, puis livrer une branche propre et non poussée.
- **Dépend de** : L8. **Vague** : W5.
- **Fichiers possédés** : `tests/test_albert_review_regressions.py`, plus les fichiers des constats confirmés, attribués sans chevauchement (`owned_L9.txt` rédigé après regroupement). Pour les points reportés (tâche 0) : `app/routes/processing.py`, `app/utils/llm_note_generator.py` et `scripts/albert_e2e_smoke.py` (journal d'usage des notes, `NOTES_LEDGER_REQUIRED`), `docs/albert.md` (sections 10 et 13), `app/templates/index.html` (si le bouton d'arrêt l'exige), `scripts/rad_chunk.py`, `scripts/rad_vectordb.py`, et les goldens G7 concernés sous `tests/fixtures/albert/golden_off/routes/` (commit d'amendement dédié).
- **Tâches** :
  0. **Points reportés des vagues précédentes** (`data/albert_gates/deferred.md`), traités avant la revue :
     - **Appartenance des sessions (décision utilisateur du 2026-09-27).** Les routes pipeline de `processing.py` n'appellent pas `verify_session_access` : une session d'un autre utilisateur reste accessible par son chemin (défaut préexistant, repéré par V-sécu en W1). Correction au lot 9, dans un **commit dédié qui amende les goldens G7 de façon motivée** :
       - seuls les cas « session d'autrui » passent de 200 à 403 ;
       - les dossiers sans `PipelineSession` (ancien flux `/upload_zip`) restent accessibles à leur créateur ;
       - tout le reste reste identique à l'octet ;
       - le message du commit cite la décision ; GC2 attend alors 2 commits sur `golden_off` (L0, puis cet amendement), et GC1 rejoue les goldens amendés.
       - Déjà livré au lot 7 : `stop_all_scripts` authentifiée et refus d'une session d'un projet inaccessible (même contrôle que les branches Albert).
     - **Bouton d'arrêt** : `index.html` appelle `fetch('/stop_all_scripts')` sans en-tête `Authorization` (repli sur le cookie `access_token`) ; vérifier que l'arrêt fonctionne avec la route désormais authentifiée, sinon ajouter l'en-tête (bloc hors chemin OFF, G9 à préserver).
     - **Chunks au texte vide (option A)** : ne plus les compter comme embeddings manquants ; modifier ensemble `rad_chunk._finish_albert_embeddings` et `rad_vectordb._missing_vectors_error`, avec des tests (phase dense bge-m3 et garde des connecteurs).
     - **Journal d'usage de la route des notes (constaté au lot 8 ; condition du contrôle 6 complet de l'E2E).** `generate_zotero_notes_sse` n'écrit pas `<session>/albert_usage.jsonl` : `_get_albert_client` (`app/utils/llm_note_generator.py`) crée un `AlbertClient` par appel, dont le `UsageLedger` en mémoire est perdu, et la route n'écrit que `generated_notes.json`.
       - Correction : la route des notes ajoute les enregistrements du ledger de ses clients Albert (rôle `notes` ou `long_context`, modèle servi, coût) à `<session>/albert_usage.jsonl`, seulement si Albert a été appelé et selon `ALBERT_USAGE_LOG` (même contrat que les scripts) ; chemin OFF identique à l'octet (G7).
       - Propriétaire : le correcteur du lot 9 qui possède `app/routes/processing.py` (déjà chargé de l'appartenance des sessions) reçoit aussi `app/utils/llm_note_generator.py` ; un seul propriétaire pour les deux fichiers, sans chevauchement.
       - Test de régression rouge puis vert dans `tests/test_albert_review_regressions.py` (faux serveur `FakeAlbert`, notes courtes avec un modèle `albert/…` : le journal existe et porte le modèle servi et le coût). Dans le même commit, `NOTES_LEDGER_REQUIRED = True` dans `scripts/albert_e2e_smoke.py` ; gate : l'E2E atteint `E2E ALBERT OK (ON 7/7, OFF 3/3, isolation 2/2)` avec le journal des notes exigé.
       - Documentation dans le même commit : `docs/albert.md` (sections 10 et 13, paragraphe « Traçabilité de l'usage ») et la puce correspondante des risques résiduels ; ces fichiers sont ajoutés à `owned_L9.txt` avec ce point.
       - Même lacune, hors contrôle E2E : fiches de lecture (`app/utils/book_note_generator.py`) et filtre de citations (`app/routes/citations.py`, `app/utils/citation_filter.py`, `upload_pop_json` compris). À corriger dans le même mouvement si les fichiers sont attribués au lot 9, sinon à consigner dans les risques résiduels.
     - Chaque point donne un test dans `tests/test_albert_review_regressions.py`, rouge avant la correction et vert après.
  1. Six relecteurs en lecture seule :
     - R1 : identifiants, SSRF, stdout, Celery, `/get_credentials` ;
     - R2 : identité OFF (goldens, délais, logs) ;
     - R3 : contrat de l'API (pages à partir de 0, `rff_k`, 64, multipart, 404, Retry-After) ;
     - R4 : quotas, concurrence, sémaphores imbriqués, une seule couche de retry ;
     - R5 : souveraineté, RGPD (manifeste, audit), reproductibilité ;
     - R6 : qualité (docstrings, code mort, tests affaiblis, attribution dans les fichiers via `check_attribution.py`).

     Chaque constat donne `path:line`, un scénario et un test.
  2. Deux vérificateurs reproduisent chaque constat (CONFIRMÉ ou REJETÉ). Les constats confirmés sont regroupés par fichier, avec au plus 4 correcteurs disjoints et 1 rédacteur de tests de régression. Au plus 2 tours. Les constats PLAUSIBLE sont rapportés sans correction.
  3. Rapport final à l'utilisateur : commits, tests, live, E2E, actions restantes (révocation Pinecone, confirmations en attente).
- **Tests** : `tests/test_albert_review_regressions.py` :: un test par constat confirmé, rouge avant la correction et vert après.
  - **Réalisé** : au lieu d'un fichier unique, les tests de régression du lot 9 sont répartis par correcteur et par tâche, chacun rouge avant la correction et vert après (ou `xfail` strict motivé quand la correction est bloquée) :
    - revue adversariale : `tests/test_albert_review_regressions_f1.py`, `tests/test_albert_review_regressions_f2.py`, `tests/test_albert_review_regressions_f3.py`, `tests/test_albert_review_regressions_f4.py` (premier tour), `tests/test_albert_review_regressions_g1.py` (second tour) ;
    - tâche 0 : `tests/test_session_ownership.py` (appartenance des sessions, amendement des goldens G7), `tests/test_albert_blank_chunks.py` (chunks au texte vide, option A), `tests/test_albert_web_ledger.py` (journal d'usage des routes web) ;
    - synchronisation de la documentation (cache tiktoken, limiteur, `/v1/models`, `/save_credentials`, changements visibles) : `tests/test_albert_docs_sync.py`.
- **Gate** :
  - GC3 sur W5 → `NEW_FAILURES=0 FAILURES=0 ERRORS=0 MISSING=0` ; `.venv/bin/python -m pytest tests/test_albert_off_golden.py tests/load/bench_dedup.py tests/load/bench_albert.py -q -p no:cacheprovider` → 0 failed
  - `ALBERT_LIVE=1 .venv/bin/python scripts/albert_e2e_smoke.py --corpus-size 3 --cleanup` → `E2E ALBERT OK (ON 7/7, OFF 3/3, isolation 2/2)`
  - `git log main..feat/albert --format='%B%n%(trailers)' | .venv/bin/python tests/tools/check_attribution.py --stdin` → `ATTRIBUTION_HITS=0` ; `git log main..feat/albert --format='%an <%ae>' | sort -u` → `Amar LAKEL <amar@lakel.net>`
  - `git diff main..feat/albert -U0 | .venv/bin/python tests/tools/check_attribution.py --diff` → `ATTRIBUTION_HITS=0` ; `git diff main..feat/albert --name-only | grep -c DS_Store` → 0
  - `git status -sb | head -1` → `## feat/albert` (sans branche distante)
  - GC2 après l'amendement motivé des goldens G7 : `git log --format=%h -- tests/fixtures/albert/golden_off | wc -l` → 2 ; `git status --porcelain --untracked-files=all -- tests/fixtures/albert/golden_off | wc -l` → 0
  - GC5 à GC9, seulement s'il y a des corrections
- **Commit** : `Albert lot 9 : corrections issues de la revue finale` (seulement s'il y a des corrections).
- **Rollback** : `git revert` du commit L9, ou abandon de la branche. `main` n'a pas bougé depuis la baseline.

## Organisation de l'exécution par vagues

**Règles globales**
- Seul l'coordinateur exécute git, les sondes, les suites live, l'E2E et les commits. Il rédige lui-même `msg_<L>.txt`.
- Les exécutants :
  - écrivent **uniquement** dans leurs fichiers possédés, et ne lisent ni n'affichent `.env` ;
  - ne régénèrent jamais les goldens et n'ajoutent aucune dépendance ;
  - écrivent du code compatible Python 3.10, avec une docstring sur chaque nouvelle fonction (politique figée) ;
  - n'ajoutent **aucune ligne d'attribution ni mention d'outil d'assistance dans les fichiers produits** (code, commentaires, docstrings, documentation, `SPRINT_albert.md`, rapport) ;
  - n'assertent que des booléens et des noms, jamais une valeur de clé ;
  - utilisent `FAKE_ALBERT_KEY` (jamais de préfixe `sk-` ni `pcsk_` sous `tests/`) ;
  - écrivent leurs commandes sur une seule ligne ;
  - se repèrent par symbole et par `<sha_base>`, jamais par numéro de ligne.
- Au début de chaque vague, l'coordinateur écrit `owned_<L>.txt`, `owned_<W>.txt` et `restrict_<L>.txt` : chemins littéraux, une ligne chacun, sans commentaire.
- **Gel des contrats** après W1 : signatures de `rad_albert`, `rad_providers`, `rad_env`, `app/tasks/runner`, `resolve_default_llm_model`, et le format de `RAGPY_DOTENV_DENY`. `conftest.py` et `tests/albert_fakes.py` sont aussi gelés.
  - Un besoin de changement passe par l'coordinateur. Un exécutant unique, le « gardien du socle », l'applique **entre deux vagues** (jamais pendant un fan-out), et le fichier est ajouté au lot demandeur.
  - En W2 et W3, chaque lot écrit ses helpers de test dans son propre fichier.
  - Les sondes ne peuvent ajuster que des valeurs par défaut et des données de catalogue.
- **Après le 2026-10-01**, au début de chaque vague : re-sonde P2/P5 vers `data/albert_probe/<ts>/` avec `--diff-against`. Si un id épinglé a disparu : arrêt et escalade.
- **Forme d'une vague d'exécution** :
  1. Répartition entre les exécutants d'implémentation (fichiers disjoints), avec une consigne écrite : tâches, symboles, `<sha_base>`, contrats de `SPRINT_albert.md`, interdits.
  2. En parallèle, des exécutants de test écrivent les tests à partir des contrats et de `FakeAlbert`.
  3. Barrière, puis GC4.
  4. L'coordinateur lance les tests de chaque lot. Chaque échec est renvoyé au propriétaire du fichier, 3 itérations au plus.
  5. Vérificateurs en lecture seule, en parallèle :
     - V-OFF : goldens, aucun `print` ni log INFO hors branche Albert, aucun réseau par défaut, délais inchangés ;
     - V-sécu : flux de la clé, non-admins, SSRF, stdout, RGPD, `/get_credentials` ;
     - V-contrat : API Albert, réutilisation des helpers cités, une seule couche de retry.
  6. Un exécutant gate-runner exécute séquentiellement les gates mockées (`-p no:cacheprovider`) et GC3 une fois par vague.
  7. Commits séquentiels par l'coordinateur, dans l'ordre des dépendances (GC5, GC6, GC7, GC9, GC8).
- Une gate rouge n'aboutit jamais à un commit. Après 3 échecs, escalade à l'utilisateur. **Arrêt obligatoire** sur :
  - toute erreur de compte Albert en live (401, 403 compte expiré, 400 budget, quota journalier) ;
  - un 404 sur un id épinglé en live ;
  - une sonde qui contredit un contrat au-delà de ce que les défauts de configuration peuvent absorber ;
  - un golden modifié ;
  - un `LEAK` ou `REAL_VALUE_HITS>0`.

| Vague | Lots | Exécutants d'implémentation (fichiers possédés) | Vérification | Gate et commits |
|---|---|---|---|---|
| W0 | L0 | Séquence : A (outils + `test_gate_tools`) → D (docstrings de la baseline, avec l'outil de A) → coordinateur (indexation baseline, GC6, GC9, commit `main`, passage sur `feat/albert`, ebooklib, caches, `preexisting.txt`, JUnit de référence) → B (`albert_probe.py`, sanitize, `decisions.json`) ∥ C (harnais golden G1-G13) → coordinateur (sondes, `--cleanup`) → analyste en lecture seule (rapport D1-D22) | V-OFF (déterminisme sur 2 passes, identifiants factices) | Commit baseline sur `main`, puis L0 |
| W1 | L1 ∥ L2 | **W1a (contrats)** : 1A `config`, `errors` ; 1B `conftest.py`, `albert_fakes.py`, `rad_providers` ; 1C DENY dans `credentials.py`. **W1b** : 1A reste de `rad_albert/*` + requirements ; 1C schemas, users, settings (masquage), pages, templates + tests ; 1D tests client, config, limiteur, infra ; 2A `rad_env`, `rad_chunk`, `rad_dataframe`, `rad_vectordb`, `llm_note_generator`, `processing` (restreint), `sse_helpers` (restreint) + tests ; 2B runner, tâches, `celery_tasks` + test ; 2C tests obsolètes (liste issue de `baseline.xml`) | V-sécu, V-OFF, V-contrat (L1) | GC3 → `green.xml` ; commits L1 puis L2 |
| W2 | L3 ∥ L4 ∥ L5 | 3A `rad_chunk` ; 3B `llm_note`, `book_note`, `citation_filter`, `parallel` ; 3C tests L3 ; 4A `rad_dataframe`, `rad_albert/ocr.py` ; 4B tests L4 ; 5A `rad_vectordb`, `rad_albert/collections.py`, docstring `rad_dedup` ; 5B tests L5 + bench | 3 × V-contrat, V-OFF | Live L3, L4 et L5 par l'coordinateur ; commits L3, L4, L5 |
| W3 | L6 ∥ L7 | 6A `rad_chunk`, `rad_vectordb`, `rad_clustering`, `rebuild` ; 6B tests L6 ; 7A routes, pipeline, Celery ; 7B templates ; 7C tests L7 | V-OFF, V-sécu | Live L6 ; commits L6, L7 |
| W4 | L8 | 8A documentation et `.env.example` ; 8B live e2e ; 8C `albert_e2e_smoke.py` ; 8D vérificateur documentation/code ; gardien du socle si la re-sonde l'exige | V-contrat | Suite live, E2E, `--cleanup`, re-sonde P2/P5/P21 si la date est au moins le 2026-10-01, `docker compose build` ; commit L8 |
| W5 | L9 | 6 relecteurs → 2 vérificateurs → au plus 4 correcteurs disjoints + 1 rédacteur de tests de régression | — | Gates complètes, E2E, hygiène de la branche ; commit L9 éventuel |

Soit environ 50 lancements d'exécutants et 11 commits : la baseline sur `main`, puis L0 à L9 sur `feat/albert`. Aucun push. Limite connue : dans une vague, les commits intermédiaires sont vérifiés sur l'union des lots ; le point vert garanti est le dernier commit de chaque vague.

## Vérification de bout en bout

1. **Suite mockée complète (hors ligne).**
   - Commande (derrière un proxy mort : aucun appel externe accidentel possible) : `HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 NO_PROXY=127.0.0.1,localhost,testserver .venv/bin/python -m pytest tests/ scripts/test_rad_vectordb.py scripts/test_rad_chunk_initial_phase.py scripts/test_albert_guard_scope.py app/test_main.py -q -p no:cacheprovider --junitxml=data/albert_gates/final.xml; .venv/bin/python tests/tools/junit_diff.py --baseline data/albert_baseline/green.xml --current data/albert_gates/final.xml --max-failures 0` → `NEW_FAILURES=0 FAILURES=0 ERRORS=0 MISSING=0`.
   - La garde réseau prouve qu'aucun appel etalab n'a eu lieu, sur les 3 racines. `NET_LEGACY` donne le nombre d'accès réseau des tests historiques (informatif).
2. **Goldens et bancs de mesure.**
   - `.venv/bin/python -m pytest tests/test_albert_off_golden.py tests/load/bench_dedup.py tests/load/bench_albert.py -q -p no:cacheprovider` → 0 failed.
   - `git log --format=%h -- tests/fixtures/albert/golden_off | wc -l` → 1 (2 après l'amendement motivé des goldens G7 au lot 9).
3. **Suite live (manuelle, clé de production).**
   - `ALBERT_LIVE=1 .venv/bin/python -m pytest tests/live -m albert_live -v -p no:cacheprovider`, puis `ALBERT_LIVE=1 .venv/bin/python scripts/albert_probe.py --cleanup --dry-run` → `ragpy-probe restantes: 0`.
   - Sans `ALBERT_LIVE=1` dans le shell, tous les tests live sont ignorés, même si la clé est présente. Un `ALBERT_LIVE` dans `.env` fait refuser la session.
4. **E2E souverain sur 3 PDF** (2 texte, 1 scanné), par les vraies routes, l'authentification et les sous-processus, pour un non-admin qui n'a qu'une clé Albert.
   - Commande : `ALBERT_LIVE=1 .venv/bin/python scripts/albert_e2e_smoke.py --corpus-size 3 --cleanup` → `E2E ALBERT OK (ON 7/7, OFF 3/3, isolation 2/2)`.
   - Ce résultat ne vaut preuve complète qu'avec `NOTES_LEDGER_REQUIRED = True`, donc après la correction du lot 9 (journal d'usage de la route des notes). Avant elle, la partie « journal » du contrôle 6 est informative (`NOTES_LEDGER_REQUIRED = False`).
   - Les 12 contrôles sont détaillés au L8. Aucune clé n'apparaît dans le stdout ni dans les logs, et aucune collection ne reste.
5. **Contrôle visuel (optionnel)** : `PATH="$PWD/.venv/bin:$PATH" ALBERT_ENABLED=1 OCR_ENABLE_ALBERT=1 .venv/bin/python -m uvicorn app.main:app --port 8811`.
   - Modale admin : carte Albert, « Tester la connexion », clé affichée masquée.
   - Datalist des modèles, choix d'embeddings, option Albert et confirmation RGPD.
   - Suppression de la collection de test.
6. **Build de l'image** : `docker compose build ragpy` (Python 3.11). Pas de `up` automatique ; le déploiement revient à l'utilisateur.

## Hors périmètre et risques résiduels

- **Clé Pinecone compromise.** Elle est dans `d1f5875`, poussé sur `github.com/MyWebIntelligence/ragpy`. **L'utilisateur doit la révoquer puis la remplacer** : le L2 ne la retire que de l'arbre. Réécrire l'historique (`git filter-repo` puis push forcé) est destructif ; c'est la décision de l'utilisateur, et le plan ne le fait pas. **La révocation côté console Pinecone reste à confirmer par l'utilisateur** (point reporté, 2026-09-27).
- **Ordre de déploiement sur le VPS.** Déployer le code (L2 inclus), **puis** ajouter `ALBERT_API_KEY` au `.env` du VPS, puis `docker compose up -d --build`. Avant le L2, une clé présente dans le `.env` serait rechargée par les sous-processus non-admin.
- **Réserves sur Celery.** Aucun secret ne transite par Redis.
  - Le worker doit partager la base SQLite `data/` et le même `JWT_SECRET_KEY` pour déchiffrer les identifiants Fernet. C'est le cas avec docker-compose (volume et `env_file`), à vérifier au déploiement. Accès concurrents SQLite entre web et worker.
  - Le processus worker garde en mémoire les clés admin du `.env`. Seuls les sous-processus sont isolés.
  - Le contrôle du propriétaire de tâche exige Redis ; sans lui, statut et annulation sont réservés aux admins.
  - Les clients externes non authentifiés de `/api/celery` cessent de fonctionner (l'interface ne les appelle pas).
- **Cycle de vie des modèles.**
  - Le 2026-10-01 (pendant W1 ou W2) : fin de test de LightOnOCR et d'`openweight-ocr` (statut incohérent), retrait de qwen3-coder, fin de test de deepseek et de qwen3-vl-embedding.
  - Le 2026-12-01 : retrait de mistral-small, remplacé par `gemma-4-31b-it`.
  - Commande à relancer après chaque échéance : `ALBERT_LIVE=1 .venv/bin/python scripts/albert_probe.py --only P2,P5,P21 --out data/albert_probe --diff-against tests/fixtures/albert`, puis ajuster le catalogue par un commit dédié.
  - Un id épinglé disparu donne un 404 explicite, jamais un changement silencieux de modèle.
- **Quotas partagés par compte** (sessions, workers, tests live, autres usages de la clé).
  - Le limiteur local ne voit qu'un processus ; sous Celery, utiliser `ALBERT_LIMITER_BACKEND=redis`.
  - gpt-oss plafonne à 5 000 requêtes par jour, d'où ministral recommandé pour le recodage de masse.
  - Les quotas OCR ne sont pas documentés (lus via `/v1/me`), et la correspondance `router_id` → modèle est déduite.
  - Les très gros corpus peuvent dépasser même `ALBERT_SUBPROCESS_TIMEOUT` : la documentation en donne le calcul.
- **Accès à `/v1/ocr` inconnu.** Sans accès, l'OCR souverain repose sur LightOnOCR, dont la qualité sur des ouvrages SHS scannés n'est pas évaluée. D21 reste à confirmer ; le défaut est de recoder.
- **Dérogations et changements visibles acceptés.**
  - `albert/…` avec Albert OFF donne un 400 (chunking, notes, citations, `upload_pop_json`) au lieu d'un envoi à OpenRouter : **à confirmer par l'utilisateur**.
  - Les non-admins n'héritent plus des clés du `.env` admin (sous-processus et `_get_llm_clients`).
  - Plus d'invite interactive dans `rad_chunk`, et plus de ligne `API Key:` dans le stdout Pinecone.
  - `rebuild_pinecone_index` refuse une dimension incohérente.
  - `/openapi.json` gagne des composants Albert, même quand Albert est OFF.
  - Les tests async historiques tournent sur asyncio seulement.
  - Les listings Pinecone, Weaviate et Qdrant de `settings.py` n'acceptent plus `url` ni `api_key` en paramètre de requête (anti-SSRF, W1).
  - Les secrets serveur `SERVER_SECRET_ENV_VARS` (`FLOWER_PASSWORD`, `JWT_SECRET_KEY`, `RESEND_API_KEY`) sont retirés de l'env des sous-processus non-admin (W1) ; au lot 9, `load_dotenv_guarded` empêche aussi ces sous-processus de les recharger depuis le `.env`.
  - `/save_credentials` (formulaire admin, écrit le `.env`) refuse, avant toute écriture, une valeur contenant un séparateur de ligne (au sens de `str.splitlines`) ou NUL : 400 `{error, invalid_keys}` (noms seulement), au lieu d'une écriture qui aurait injecté d'autres variables dans le `.env` ; une tabulation ou tout autre caractère reste accepté comme sur `main`.
  - **Lot 9, chemin désactivé** (décision utilisateur du 2026-09-27 et revue finale) :
    - routes pipeline de `processing.py` : une session enregistrée (`PipelineSession`) d'un projet auquel un non-admin n'a pas accès donne 403 (JSON, ou un unique événement SSE `{type: error}` avec le statut 403 sur les routes SSE), sans sous-processus ni lecture ni écriture ; ce contrôle passe avant ceux des identifiants. Goldens G7 amendés de façon motivée dans `ca9561c` (16 cas `member_keys` sur `gsess-full`) ; les dossiers sans `PipelineSession` (ancien flux `/upload_zip`) gardent le comportement historique ;
    - `/cluster_documents_sse` refuse en JSON (403 `{"error": …}`), et non en flux SSE, car la page lit du JSON sur un statut de refus ;
    - un dossier de session qui ne se résout pas strictement sous `uploads/` (`..`, chemin absolu, `.`, lien symbolique sortant, octet NUL) donne 400 pour tous, administrateurs compris ;
    - `/upload_stage_file/{stage}` exige un utilisateur authentifié (401 sans jeton ; en-tête `Authorization` ou cookie `access_token`, que la page envoie), refuse la session enregistrée d'un projet inaccessible (403) et un chemin hors de `uploads/` (400), avant toute écriture ;
    - `/upload_zip` et `/upload_csv` exigent un utilisateur authentifié (401 sans jeton ; en-tête `Authorization` ou cookie `access_token`) et, quand un `project_id` est fourni, le même contrôle que les routes `/api/pipeline/projects/{id}/upload_*`, avant toute écriture : 404 pour un projet inconnu, 403 pour un projet inaccessible ou en lecture seule (propriétaire, collaborateur et administrateur passent) ; sans `project_id`, comportement historique pour l'utilisateur authentifié ;
    - une archive ZIP dont une entrée sortirait du dossier d'extraction (`..`, nom absolu) est refusée en entier avant toute écriture : `/upload_zip` répond 400 `Uploaded file is not a valid ZIP archive.`, comme pour une archive invalide.
- **Dédup Tier 3.** Le seuil de 0.97 est calibré pour `text-embedding-3-large`. Il est désactivé pour bge-m3 tant que `DEDUP_SIM_THRESHOLD_BGE_M3` n'est pas défini, et le seuil adapté reste à calibrer.
- **Cas limite des colonnes d'espace.** Un CSV produit avant le lot 6 et possédant des colonnes nommées `embedding_provider`, `embedding_model` ou `embedding_dim` peut être lu comme une déclaration d'espace. La phase dense les réécrit ; seul un ancien fichier d'embeddings peut rester ambigu (documenté dans `docs/albert.md`).
- **Journal d'usage des traitements en processus.** Corrigé au lot 9 (tâche 0, `c7314aa` et `ca9561c`) : les routes des notes Zotero (fiches de lecture comprises) et du filtre de citations ajoutent leurs appels Albert à `<session>/albert_usage.jsonl`, seulement si Albert a été appelé ; `upload_pop_json` ne fait que valider le modèle. Le contrôle 6 de l'E2E exige ce journal (`NOTES_LEDGER_REQUIRED = True`, `ffa8e11`).
- **Chunks au texte vide (bge-m3).** Corrigé au lot 9 (option A, `c7314aa`) : ils ne sont ni envoyés ni comptés comme embeddings manquants (phase dense, garde des connecteurs, `rebuild_pinecone_index --all`).
- **Rafale initiale du seau local.** Corrigé au lot 9 : le seau démarrait plein et laissait passer près du double du débit réglé sur la première minute d'un processus ; il se remplissait de nouveau d'une minute entière après une inactivité (revue finale). La rafale vaut désormais la concurrence du rôle (plafonnée à une minute de débit) : elle est prise sur le débit de la première minute (au plus RPM × part requêtes sur la première minute) et plafonne le niveau accumulé, d'où la borne exacte : au plus RPM × part + rafale − 1 requêtes sur toute minute glissante, la rafale valant au plus la concurrence du rôle. De même au plus TPM × part + rafale de tokens − 1 tokens, la rafale de tokens valant rafale × TPM / RPM (une requête plus grosse ne dépasse la borne que de son propre excédent). Les tests épinglés du lot 1 (`tests/test_albert_limiter.py`) sont mis à jour en conséquence ; la borne après inactivité est couverte par `tests/test_albert_review_regressions_f3.py`.
- **Fuite de raisonnement de gpt-oss.** Du raisonnement pourrait passer dans `content`. Parades : D9, garde de ratio au recodage, ministral par défaut, analyse par jeton exact dans le pré-filtre.
- **RGPD des collections.** Les textes sont stockés chez la DINUM jusqu'à leur suppression. Droits d'auteur des PDF, données personnelles et éligibilité au secteur public des non-admins qui apportent leur clé restent à la charge de l'utilisateur. Sont fournis : acquittement journalisé, collection privée, manifestes incrémentaux hors `uploads/`, suppression. La suppression d'un compte RAGpy ne supprime pas ses collections (documenté).
- **Dette existante hors périmètre, signalée et non corrigée.**
  - Sémaphores imbriqués du filtre de citations sur le **chemin OFF** (`citations.py:1173`, puis `citation_filter.py:694`). Corrigés seulement sur le chemin Albert, pour préserver l'identité OFF.
  - Regex `^Dedup journal:\s*(\S+)` (`processing.py:889`), qui tronque les chemins avec espaces pour les 3 connecteurs historiques. Corrigée pour la seule branche albert.
  - Faux rejets `'NA' in upper` du pré-filtre sur le chemin OFF (`citation_filter.py:603-604`).
  - `configure_url` `/settings/credentials`, qui pointe vers une route inexistante.
  - Chemins non confinés `os.path.join(UPLOAD_DIR, path)` dans `processing.py` : dossiers de session des routes pipeline et de `/upload_stage_file` confinés au lot 9 (400 hors de `uploads/`).
  - Hash sparse non reproductible (`PYTHONHASHSEED`).
  - Littéral `"python3"` des routes : conservé pour ne pas changer l'argv. En local, uvicorn doit être lancé avec `.venv/bin` en tête du PATH.
  - Contrôle d'appartenance des sessions absent des routes pipeline de `processing.py` (`verify_session_access`) : **sorti de cette liste par la décision utilisateur du 2026-09-27**, corrigé au lot 9 avec amendement motivé des goldens G7 (voir Lot 9, tâche 0).
- **Environnement.**
  - Python 3.10 en local contre 3.11 dans Docker (vérifié par `docker compose build`).
  - Synchronisation Google Drive pendant des écritures parallèles : copies de conflit (signalées par `check_scope`) et verrous `.git` possibles. Suspendre la synchronisation.
  - `.DS_Store` suivis et modifiés : ignorés par `check_scope`, jamais indexés. Leur retrait de l'index revient à l'utilisateur.
  - Désaccord existant entre le pin `redis==5.0.1` et `redis` 7.1.0 installé.
  - Les caches tiktoken et spaCy doivent être préchauffés (L0). S'ils sont purgés, les tests nouveaux échouent explicitement en nommant l'hôte. Depuis le lot 9, `conftest.py` place le cache tiktoken dans `data/tiktoken_cache` (persistant, ignoré par git) quand `TIKTOKEN_CACHE_DIR` n'est pas défini : le cache temporaire par défaut avait été purgé par macOS. Préchauffage : `docs/albert.md`, section 10.

### Fichiers critiques
- app/core/credentials.py
- scripts/rad_chunk.py
- scripts/rad_dataframe.py
- scripts/rad_vectordb.py
- app/routes/processing.py
- nouveau : scripts/rad_albert/client.py
- nouveau : conftest.py

## Journal des révisions

1. **[major] Sémaphores et backoff du filtre de citations** — Corrigé.
   - Décision 15 et L3 tâche 8 : les deux sites passent par `run_llm_slot`, en `single_attempt`, hors de la boucle `max_retries`.
   - L7 tâche 7 : l'acquisition externe de `citations.py:1173` est retirée sur le chemin Albert.
   - Tests `test_semaphore_released_during_backoff` et `test_no_nested_acquire_deadlock` ; invariants 17 et 36.
   - Le chemin OFF garde la dette, signalée dans les risques.
2. **[major] Délais des sous-processus** — Corrigé. `ALBERT_SUBPROCESS_TIMEOUT` (décision 19, L7 tâche 1, runner Celery), kwarg `timeout` capturé dans G7, `test_albert_timeout_only_when_selected`, documentation du calcul au L8.
3. **[major] content_id quand `DEDUP_ENABLED=0`** — Corrigé.
   - Repli `content_id(content_hash(text), chunk_index)`, avec sa limite documentée.
   - Liste blanche forcée par `validate_metadata_fields`.
   - Tests `test_rerun_idempotent_without_dedup_fields` et `test_metadata_override_must_keep_content_id`.
4. **[major] Portée de la garde réseau du conftest** — Corrigé. `conftest.py` à la racine (décision 21, L1 tâche 9), `scripts/test_albert_guard_scope.py`, neutralisation de `ALBERT_API_KEY` avant l'import d'`app.config`.
5. **[major] Couverture stdout et logs OFF** — Corrigé. G12 (capsys des phases `rad_chunk`) et G13 (caplog OCR) ajoutés au L0 et rejoués par GC1.
6. **[major] Routes modifiées sans oracle OFF** — Corrigé. G7 étendu aux 3 routes de citations, à l'état du pipeline et au clustering. Validation d'`upload_pop_json` au L7, avec son test.
7. **[major] Clé Albert en clair via `/get_credentials`** — Corrigé. Masquage côté serveur et masque ignoré par `/save_credentials` (décision 20, L1 tâche 13). Tests associés ; invariant 12 reformulé (corps de **réponse**).
8. **[major] Commandes de gate fausses** — Corrigé.
   - `git grep 'pcsk_[A-WYZa-z0-9]'`.
   - `grep -cE '^  --…'` pour argparse.
   - `"/" in (eff_)?model`.
   - Gabarits en X ignorés par `test_no_secret_literals_in_repo` et `check_secrets`.
9. **[major] `.DS_Store` suivis (GC4, GC5)** — Corrigé. `check_scope` les ignore et signale les copies de conflit. Le retrait de l'index reste une décision de l'utilisateur.
10. **[major] GC8 contrôlait après le commit** — Corrigé. Message dans `msg_<L>.txt`, grep et `interpret-trailers` avant `commit -F`, contrôle a posteriori avec `--amend` immédiat. Interdiction de citer `CLAUDE.md` dans un message.
11. **[major] Attribution dans les fichiers du dépôt** — Corrigé. GC9 sur le contenu ajouté via `check_attribution.py`, consigne dans chaque répartition, `SPRINT_albert.md` rédigé en termes de lots et de contrats.
12. **[major] P21 non raccordée ; concurrence de recodage** — Corrigé. D22 = P21 (`ALBERT_RECODE_CONCURRENCY`, `ALBERT_NOTES_CONCURRENCY`, `ALBERT_EMBED_CONCURRENCY`), `_PUSH_CONCURRENCY` rattaché à D19, `ALBERT_RECODE_SEMAPHORE` de module, `test_recode_concurrency_is_process_wide`.
13. **[minor] Autres sondes sans consommateur** — Corrigé. Tableau D1-D22 avec un consommateur par décision : D11 seed, D12 normalisation, D14 noms en double, D16 ids et atomicité, D1 permissions. D20 est marquée informative.
14. **[minor] `RECODE_SKIP_PROVIDERS` et propriété au L8** — Corrigé. Seul `albert_mistral_ocr` est ajouté d'office ; `albert_lightonocr` passe derrière `ALBERT_OCR_SKIP_RECODE=0`. `catalog.py`, les fixtures et `rad_chunk.py` sont ajoutés à `owned_L8` si nécessaire.
15. **[minor] Priorité entre harden et `--model albert/`** — Corrigé. `effective_recode_model`, source unique du routage et de la clé, avec deux tests.
16. **[minor] Résolution d'`EMBEDDING_PROVIDER`** — Corrigé. Formulaire, puis serveur, seulement si Albert est ON ; `required_keys` déduit ; env posé explicitement ; `exit 2` si `albert` est demandé avec Albert OFF. Tests au L6 et au L7.
17. **[minor] Erreurs de compte avalées (fiches, citations)** — Corrigé. Relance dans les 3 phases et dans `process_citations_parallel`, événement SSE au L7 (décision 22). Tests `test_book_account_error_propagates` et `test_account_error_aborts_job`.
18. **[minor] Réessais empilés** — Corrigé. Boucles contournées sur le chemin Albert (décision 5). Tests `test_single_retry_layer_*` ; invariant 37.
19. **[minor] Erreurs de compte OCR non mémorisées** — Corrigé. `_ALBERT_ACCOUNT_DISABLED` et `test_account_error_memoized_skips_next_documents`.
20. **[minor] Manifeste et traçabilité RGPD** — Corrigé. Append et flush par tranche, copie dans `data/albert_manifests/` dans un `finally`, `create_audit_log` sur l'envoi et la suppression, documentation sur la suppression de compte. Test `test_manifest_written_per_slice`.
21. **[minor] Regex « Dedup journal » et espaces** — Corrigé pour la seule branche albert (ligne entière). Test étendu.
22. **[minor] Décision « NA » du pré-filtre** — Corrigé sur la branche Albert (jeton exact). Chemin OFF inchangé et testé.
23. **[minor] Propriété des tests en échec inconnus** — Corrigé. `owned_L2` dérivé de `baseline.xml` ; règle pour les correctifs produit ; `xfail(strict=True)` motivé en repli.
24. **[minor] Fuite de la clé par les assertions** — Corrigé. Règle « booléens et noms seulement », `FAKE_ALBERT_KEY`, `ALBERT_API_KEY=''` en session, GC6 étendue aux XML JUnit.
25. **[minor] Décision 11 contre L4 ; champs d'`OCRResult`** — Corrigé. Décision 11 reformulée (bibliothèques et scripts CLI), G1 limité aux 6 champs via `getattr`, `test_ocrresult_never_unpacked_positionally`.
26. **[minor] Notes : validation sur le modèle effectif** — Corrigé. `resolve_default_llm_model()` (L2) utilisé par la route (L7), avec `test_notes_empty_model_uses_effective_default_for_credential`.
27. **[minor] Limiteur bloquant dans la boucle asyncio** — Corrigé. `asyncio.to_thread` et `asyncio.sleep`, URL Redis précisée, `test_limiter_does_not_block_event_loop`.
28. **[minor] Seuil Tier 3 pour bge-m3** — Corrigé. Tier 3 sauté sauf `DEDUP_SIM_THRESHOLD_BGE_M3` (L6 tâche 4), avec deux tests.
29. **[minor] Reproductibilité par chunk** — Corrigé. `recode_status` et `recode_model` toujours écrits pour Albert, avec `test_albert_run_always_emits_recode_model`.
30. **[minor] Branche albert de Celery et fichier sparse** — Corrigé. `resolve_albert_input` partagé, avant la vérification sparse seulement si Albert est ON, avec `test_celery_albert_accepts_output_chunks`.
31. **[minor] E2E et `scripts/.env.example`** — Corrigé. Utilisateurs vérifiés insérés en base temporaire, dossiers de session supprimés en `finally`, `scripts/.env.example` couvert par le test et par la gate.
32. **[blocker] Littéral `python3` sur le PATH local** — Corrigé. PATH de l'E2E avec `.venv/bin` en tête et vérification `shutil.which`, commande de contrôle visuel avec `PATH=…`, `sys.executable` dans l'E2E live, documentation. L'argv des routes est inchangé.
33. **[blocker] Variantes trio d'anyio** — Corrigé. Fixture de session `anyio_backend='asyncio'` dans le conftest racine (W1a, avant 2C), hook qui refuse les ids `[trio]`, `test_anyio_backend_is_asyncio`.
34. **[blocker] Goldens capturant de vrais secrets** — Corrigé. `delenv` et `setenv` factices, `RAGPY_DIR` redirigé, empreintes plutôt que valeurs, GC6 enrichie des valeurs réelles (`check_secrets.py`, sans affichage) sur le diff, les fixtures et les XML, `test_golden_harness_uses_fake_credentials`.
35. **[major] Comptage argparse (L5, L6)** — Corrigé (même correction que le point 8). `grep -cE '^  --albert-'` → 4 et `'^  --embedding-provider'` → 1.
36. **[major] `check_scope` : `.DS_Store`, fichiers non suivis, artefacts** — Corrigé. Ignorés : `.DS_Store`, `Icon\r`, `._*`. Options `--untracked-files=all`, chemins exacts, `preexisting.txt`. GC4 passe avant GC3, et `test_rad_chunk_initial_phase` écrit désormais dans `tmp_path`.
37. **[major] Politique de docstrings et séquence de W0** — Corrigé. Politique figée (exemption de `test_*` et des closures, `--strict` possible, à confirmer), environ 45 docstrings signalées dans la baseline, séquence A → D → commit `main` → B ∥ C.
38. **[major] `albertai.md` dans la baseline** — Corrigé. Retiré de la liste et laissé non suivi (`preexisting`). Contrôle du nom de fichier et GC9 sur la baseline, dont le dry-run donne 0 correspondance.
39. **[major] `&&` dans GC3** — Corrigé. `;`, et `junit_diff` compte failure et error, `MISSING` et `PASSED_DELTA`, avec liste d'exceptions `allowed_removals.txt`.
40. **[major] Liste figée des tests obsolètes** — Corrigé (même correction que le point 23). `test_integration_api.py` est ajouté explicitement.
41. **[major] FakeAlbert partagé en W2** — Corrigé. FakeAlbert complet dès L1 (tous les endpoints, états, pagination), gelé ; modifications en série par le gardien entre les vagues ; helpers dans chaque fichier de test.
42. **[major] Portée du conftest et déclenchement live** — Corrigé (même correction que le point 4). `ALBERT_LIVE` capturé à l'import du conftest, refus s'il figure dans `.env`, `test_live_refused_if_env_file_has_ALBERT_LIVE`. Le `.dockerignore` n'est pas modifié, jugé inutile.
43. **[major] `--pathspec-from-file` et listes annotées** — Corrigé. `owned_<L>.txt` littéral, `--emit-add-list` qui ne garde que les fichiers existants, restrictions par symbole dans `restrict_<L>.txt`, vérifiées par AST.
44. **[major] G1 et G6 figés contre L1 et L4** — Corrigé. Listes littérales (6 champs, 15 noms), `_asdict` et `MAPPING.values()` proscrits, `test_golden_harness_uses_literal_lists`.
45. **[major] Re-sondes du L8 et propriété ; LightOnOCR pendant W2** — Corrigé. Re-sondes vers `data/albert_probe/` avec `--diff-against`, fichiers de catalogue ajoutés à `owned_L8` et tests L1/L3 rejoués si besoin. Un 404 sur l'id épinglé arrête la gate live de L4 avec escalade, et une re-sonde a lieu au début de chaque vague après le 2026-10-01.
46. **[major] E2E sous-spécifié** — Corrigé. Les 7, 3 et 2 contrôles sont énumérés avec leur route et leur assertion ; insertion SQLAlchemy et `create_access_token` ; collections `ragpy-probe-e2e-<ts>` ; nettoyage des dossiers `uploads/` ; contrôle OFF de `/get_credentials`.
47. **[minor] Décisions seulement dans un markdown** — Corrigé. `decisions.json` produit par la sonde, `test_defaults_consistent_with_decisions`, fixtures synthétiques marquées `"synthetic": true`.
48. **[minor] Dépendances internes à W1, séparateur DENY, appelant de `_get_llm_clients`** — Corrigé. Phase W1a (config, `rad_providers`, conftest et DENY avant L2), format DENY figé (trié, virgules, sans espace), `test_get_llm_clients_callers_pass_explicit_keys` au lieu de posséder `citation_filter.py` au L2.
49. **[minor] Contrôle GC8 avant le commit** — Corrigé (même correction que le point 10).
50. **[minor] Fausses clés contre GC6 ; test qui se détecte lui-même ; prose « Bearer »** — Corrigé. `FAKE_ALBERT_KEY` sans `sk-`, regex construites par concaténation et limitées à `git ls-files`, motif `Bearer [A-Za-z0-9._-]{20,}`.
51. **[minor] Grep de L3 et `eff_model`** — Corrigé (même correction que le point 8).
52. **[minor] `find_dotenv` dans le test d'isolation** — Corrigé. Sous-processus `sys.executable -c` avec `cwd=tmp_path`, assertions sur les noms et les booléens seulement. Fait vérifié : aucun `scripts/.env`, et `rad_env.py` placé dans `scripts/`.
53. **[minor] Points de patch SSE, `force_split`, chemins** — Corrigé. Patch de `app.utils.sse_helpers.run_subprocess_with_sse`, `force_split` recopiée avec une docstring, normalisation `<RAGPY>`, `<TMP>` et `<SESSION>`.
54. **[minor] Faux Redis et `is_celery_available`** — Corrigé. `FakeRedis` écrit à la main dans `albert_fakes.py`, patch de `is_celery_available` et `.delay`, aucune nouvelle dépendance.
55. **[minor] Superposition base de `/get_credentials`** — Corrigé. Filtrée par `_admin_form_env_keys()`, avec `test_get_credentials_off_ignores_db_albert_key`.
56. **[minor] Décalage des `path:line`** — Corrigé. Repérage par symbole et `<sha_base>` dans tous les consignes (en-tête des Lots et règles globales).
57. **[minor] Garde réseau limitée à etalab** — Corrigé en partie.
    - Tout hôte non local est bloqué dans les tests nouveaux et seulement journalisé dans les tests historiques (`NET_LEGACY`, informatif), avec préchauffage des caches tiktoken et spaCy au L0.
    - Motif de la restriction : un blocage total pourrait casser des tests historiques qui accèdent légitimement au réseau.

## Journal d'exécution

> Branche `feat/albert`, aucun push. Suite complète en fin de vague (GC3, derrière un proxy mort) : W1 987 verts et 26 ignorés, W2 1 206 verts et 35 ignorés, W3 1 358 verts et 37 ignorés, 0 échec et 0 erreur à chaque vague. Goldens G1-G13 rejoués à chaque gate, jamais régénérés (un seul commit sur `tests/fixtures/albert/golden_off`).

- **Base, `5e40aed` (sur `main`, 2026-09-26)** — sprint déduplication commité comme point de départ : cascade Tier 1-3, cache de recodage, adaptateurs de connecteurs, outils de maintenance Pinecone, docstrings complétées. JUnit de référence (`baseline.xml`) : 399 verts, 35 échecs, 2 erreurs dans 10 fichiers.
- **L0, `f4b2930`** — outils de gate (`junit_diff`, `check_docstrings`, `check_scope`, `check_secrets`, `check_attribution`) et `tests/test_gate_tools.py` ; `scripts/albert_probe.py` : 21 sondes exécutées, 0 erreur de script ; compte en **régime d'expérimentation** (1 000 requêtes par jour et par modèle de chat, gpt-oss à 10 requêtes par minute), **pas d'accès à `/v1/ocr`** (OCR Albert par LightOnOCR) ; `response.model` renvoie l'alias quand on appelle par alias ; fixtures assainies et `decisions.json` (D1-D22) ; rapport `albert_probe_report.md` ; harnais golden G1-G13 capturé sur le code de base avec des identifiants factices.
- **L1, `82be764`** — paquet `scripts/rad_albert/` (config et `ENV_REGISTRY`, errors, retry, catalog, limiter, usage, client httpx, preflight) et `scripts/rad_providers.py` ; identifiant `albert_api_key` dans les trois registres, masqué côté serveur, `/api/albert/status` et `/api/albert/models` (404 si OFF) ; carte Albert de la modale admin et section du profil ; `conftest.py` racine (neutralisation de l'env, garde réseau, marqueur `albert_live`, anyio asyncio) et `tests/albert_fakes.py` ; suite live du socle. En W1 également : secrets serveur retirés de l'env des sous-processus non-admin, listings vectoriels sans `url`/`api_key` en paramètre de requête (anti-SSRF). Tests du catalogue livrés dans `tests/test_albert_catalog.py`, écritures jamais rejouées dans `tests/test_albert_client_writes.py`.
- **L2, `cdefd1d`** — `scripts/rad_env.py` et `RAGPY_DOTENV_DENY` ; `rad_chunk` sans `input()` ; littéral Pinecone et affichage de clé retirés ; `_get_llm_clients` sans repli `.env` et `resolve_default_llm_model` ; stdin des sous-processus en DEVNULL ; Celery réécrit en exécuteur de sous-processus par utilisateur (`app/tasks/runner.py`, routes authentifiées, aucun secret dans Redis) ; tests de référence remis au vert : 987 verts, 0 échec (`green.xml`).
- **L3, `3311432`** — chat Albert pour le recodage (`effective_recode_model`, `ALBERT_RECODE_SEMAPHORE`, clé de cache sur l'id résolu, `recode_status`/`recode_model` toujours écrits), les notes et fiches (`run_llm_slot`, marge de raisonnement de gpt-oss) et le filtre de citations (aucun sémaphore imbriqué) ; une seule couche de retry ; aucun repli silencieux vers OpenAI ou OpenRouter ; erreur de compte = arrêt du job ; `albert_mistral_ocr` ajouté à `RECODE_SKIP_PROVIDERS`, `albert_lightonocr` derrière `ALBERT_OCR_SKIP_RECODE`.
- **L4, `0d93830`** — `scripts/rad_albert/ocr.py` et maillon Albert en tête de chaîne OCR (`/v1/ocr` si accès, sinon LightOnOCR page par page) ; marqueurs de page continus et marqueur d'échec distinct ; repli tracé (`OCRResult.fallback_from`, `OCR_PROVIDER_FALLBACK`) ; erreur de compte mémorisée pour le processus ; parseur Mistral extrait sans changement d'octet.
- **L5, `43bce7e`** — collections Albert comme 4e cible (`insert_to_albert`, `--db albert` et ses 4 options) : collection privée forcée, métadonnées en liste blanche (10 au plus, jamais `path`), idempotence append-skip par `content_id`, rollback par document, manifeste écrit par tranche, acquittement de rétention ; créations à l'issue incertaine jamais rejouées, réconciliées par relecture (point reporté de W1).
- **L6, `b9b7a88`** — embeddings `bge-m3` (1024 d) dans un espace séparé (`--embedding-provider`, champs d'espace hors défaut seulement, jamais de vecteur nul, cache cloisonné) ; gardes de dimension de Pinecone, Qdrant et Weaviate, contrôle préalable de `rebuild_pinecone_index.py`, message du clustering ; Tier 3 sauté pour bge-m3 sans `DEDUP_SIM_THRESHOLD_BGE_M3`.
- **L7, `b31b97e`** — routes et interface : sélection du fournisseur (chunking, dense, notes, citations, `upload_pop_json`), clé Albert exigée selon la capacité, `ALBERT_SUBPROCESS_TIMEOUT` seulement quand Albert est sélectionné ; `/upload_db` albert (entrée `resolve_albert_input`, acquittement RGPD, audit `ALBERT_COLLECTION_UPLOAD`, copie des manifestes dans `data/albert_manifests/<user_id>/`) ; `GET /api/albert/collections` et `DELETE /api/albert/collections/{id}?confirm=true` (audit `ALBERT_COLLECTION_DELETE`) ; branche Celery (jamais réessayée automatiquement, limites de temps relevées par `runner.albert_time_limits`) ; correction du `NameError` des routes SSE de chunking et d'embeddings pour un utilisateur sans clé ; `stop_all_scripts` authentifiée, avec refus d'une session d'un projet inaccessible.
- **L8, `3790a64` (2026-09-27)** — bloc `# ===== ALBERT API (DINUM) — OPT-IN, OFF PAR DÉFAUT =====` du `.env.example` généré depuis `ENV_REGISTRY` (56 variables) et renvoi en tête de `scripts/.env.example` ; guide `docs/albert.md` ; sections Albert de la documentation projet et du `README.md` ; `tests/test_albert_docs_sync.py` (invariants 32 et 33) ; intégration des points reportés ci-dessous. Constat : la partie « journal » du contrôle 6 de l'E2E n'était pas vérifiable avant le lot 9 (la route des notes n'écrivait pas `albert_usage.jsonl`) ; assouplissement provisoire `NOTES_LEDGER_REQUIRED = False`, levé au lot 9. Gate : suite live 20 tests verts (1 ignoré faute d'accès à `/v1/ocr`), E2E souverain `ON 7/7, OFF 3/3, isolation 2/2` (journal des notes informatif), image Docker construite.
- **L9 (1/3), `c7314aa`** — tâche 0 : journal d'usage Albert des appels web côté utilitaires (notes courtes et étendues, fiches de livre, filtre de citations : dernier argument `albert_usage_ledger`) ; chunks au texte vide non comptés comme embeddings manquants (option A : phase dense bge-m3, garde des connecteurs, `rebuild_pinecone_index --all`). Tests : `tests/test_albert_web_ledger.py`, `tests/test_albert_blank_chunks.py`.
- **Sécurité, `ca9561c`** — amendement motivé des goldens G7 (décision utilisateur du 2026-09-27) : contrôle d'appartenance des sessions sur 12 routes pipeline de `processing.py` ; seuls 16 cas changent (persona `member_keys` sur `gsess-full`, détail dans `data/albert_gates/golden_amendment_L9.md`) ; écart délibéré au-delà de la lettre de la décision : 400 pour un dossier hors de `uploads/`, sous-dossiers, dossiers parents et autres graphies d'une session enregistrée refusés à un non-membre, refus en JSON sur `/cluster_documents_sse` ; branchement du journal d'usage des routes web (`<session>/albert_usage.jsonl`, seulement si Albert a été appelé). Tests : `tests/test_session_ownership.py`. GC2 : 2 commits sur `tests/fixtures/albert/golden_off`.
- **L9 (3/3), `ffa8e11`** — le contrôle on6 de l'E2E souverain exige le journal d'usage des notes (`NOTES_LEDGER_REQUIRED = True`).
- **L9, revue finale et corrections (commit final du lot 9 en attente)** — premier tour de corrections (`tests/test_albert_review_regressions_f1.py` à `_f4.py`) : un 429 sans `Retry-After` n'est un quota épuisé qu'après une fenêtre complète d'une minute ; attente du limiteur hors de l'exécuteur par défaut et pause Redis hors de la boucle ; trace de repli OCR et journal d'usage OCR écrits document par document ; chemin complet du journal de dédup sous Celery ; garde de la CLI alignée sur le routage effectif du recodage ; alias du chemin web résolus par `/v1/models` ; sémaphore Albert pris avant le sémaphore global ; secrets serveur jamais rechargés du `.env` par un sous-processus non-admin ; budget du limiteur selon le modèle envoyé (gpt-oss sur `notes`) ; replis absents de `/v1/models` retirés de la chaîne ; Redis retenté après une panne ; `/upload_stage_file` authentifiée et contrôlée ; archives ZIP à entrée sortante refusées. Second tour (`tests/test_albert_review_regressions_g1.py`, `tests/test_albert_review_regressions_f3.py` révisé, `tests/test_albert_docs_sync.py`) : budget du limiteur et liste `/v1/models` sur le chemin web (`AlbertClient.set_model_listing`), authentification et contrôle de projet de `/upload_zip` et `/upload_csv`, rafale de départ du seau local bornée par la concurrence du rôle et prise sur la première minute, `/save_credentials` limité aux séparateurs de ligne et NUL (tabulation acceptée comme sur `main`), cache tiktoken persistant (`data/tiktoken_cache`) posé par `conftest.py`, documentation (`docs/albert.md`, sections 3.2, 3.5, 3.6, 8, 10, 13 et 14 ; dérogations ci-dessus). Suite mockée complète, goldens OFF et gates au commit final.

**Points reportés (`data/albert_gates/deferred.md`) et leur place dans ce document.**

| Point | Traitement |
|---|---|
| Routes pipeline de `processing.py` sans `verify_session_access` | Décision utilisateur du 2026-09-27 : corrigé au lot 9 (`ca9561c`, amendement motivé des goldens G7, `tests/test_session_ownership.py`) ; retiré de la dette hors périmètre |
| `stop_all_scripts` sans authentification | Livré au lot 7 (authentification et refus d'une session d'autrui) |
| Bouton d'arrêt de `index.html` sans en-tête `Authorization` | Vérifié au lot 9 : l'authentification accepte le cookie `access_token` envoyé par la page (même mécanisme pour `/upload_stage_file`) |
| `NameError` des routes SSE chunking et dense (golden `g7_sse_credential_error_known_defect.json`) | Corrigé au lot 7 (tâche 0) |
| Invariant 30 : tests dans `tests/test_albert_catalog.py`, écritures non rejouées dans `tests/test_albert_client_writes.py` | Tableau des invariants (30 et 27) et tests du lot 1 mis à jour |
| Écritures Albert non idempotentes (`AlbertUncertainWriteError`) | Livré au lot 5 (réconciliation par relecture) ; lot 5, tâche 4 |
| Listings Pinecone/Weaviate/Qdrant sans `url`/`api_key` en paramètre de requête | Dérogations et changements visibles acceptés |
| `SERVER_SECRET_ENV_VARS` retirés de l'env non-admin | Décision 10 et dérogations |
| Clé Pinecone dans l'historique (`d1f5875`) | Risques résiduels : révocation à confirmer par l'utilisateur |
| Chunks au texte vide comptés comme embeddings manquants | Corrigé au lot 9 (option A, `c7314aa`, `tests/test_albert_blank_chunks.py`) |
| Colonnes `embedding_provider` / `embedding_model` / `embedding_dim` d'un CSV antérieur au lot 6 | Risques résiduels (cas limite documenté) |
| Journal d'audit (`ALBERT_COLLECTION_UPLOAD`, suppression) et manifestes `data/albert_manifests/<user_id>/` | Documentés dans `docs/albert.md` (section 6) |
| Limites de temps Celery des tâches Albert (`runner.albert_time_limits`) | Documentées dans `docs/albert.md` (sections 3.4 et 9) |
| Journal d'usage `albert_usage.jsonl` non écrit par la route des notes (contrôle 6 de l'E2E), ni par les fiches de lecture et le filtre de citations (constaté au lot 8) | Corrigé au lot 9 (`c7314aa` pour les utilitaires, `ca9561c` pour les routes, `tests/test_albert_web_ledger.py`) ; contrôle 6 complet de l'E2E (`NOTES_LEDGER_REQUIRED = True`, `ffa8e11`) ; documenté dans `docs/albert.md` (sections 10, 13 et 14) |
