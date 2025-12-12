# Audit de Code RAGpy

**Date:** 12 Décembre 2025
**Version analysée:** 1.0.0
**Auteur:** Antigravity (Agent AI)

## 1. Synthèse Exécutive

L'application RAGpy présente une architecture modulaire intéressante avec une séparation claire entre le frontend (FastAPI) et les tâches de traitement. Cependant, elle souffre d'une dette technique majeure liée à son héritage "script-based". Le cœur du traitement repose sur l'exécution de scripts Python autonomes via `subprocess`, ce qui rend l'application fragile, difficile à tester et à maintenir.

L'architecture actuelle est fonctionnelle mais nécessite une refonte significative pour passer d'un "orchestrateur de scripts" à une véritable application web intégrée.

## 2. Analyse Architecturale

### 2.1. Pattern "Subprocess Orchestrator" (Critique)
Le problème architectural le plus important est l'utilisation intensive de `subprocess` pour exécuter la logique métier située dans le dossier `scripts/` (`rad_dataframe.py`, `rad_chunk.py`, etc.).

*   **Problème:** Les routes API (`app/routes/processing.py`) construisent des lignes de commande et lancent des processus.
*   **Conséquences:**
    *   **Performance:** Overhead de démarrage de l'interpréteur Python à chaque tâche.
    *   **Fragilité:** Dépendance aux chemins de fichiers et aux environnements d'exécution.
    *   **Debug:** Difficile de tracer les erreurs qui se produisent dans les sous-processus (parsing de stderr nécessaire).
    *   **Testing:** Quasiment impossible de faire des tests d'intégration propres sans mocker `subprocess`.

### 2.2. Structure des Dossiers
*   **Module `ingestion` orphelin:** Le dossier `ingestion/` se trouve à la racine du projet, hors de `app/`. Il contient pourtant de la logique métier (`csv_ingestion.py`) utilisée par l'API. Il devrait être intégré dans `app/services` ou `app/ingestion`.
*   **Dossier `scripts/`:** Contient la logique "cœur" du pipeline mais sous forme de scripts CLI. Cette logique devrait être encapsulée dans des classes de service dans `app/services`.

### 2.3. Gestion de la Configuration
La configuration est fragmentée :
*   `app/config.py`: Gère les variables d'environnement et settings globaux (style `pydantic`).
*   `app/core/config.py`: Gère les constantes de chemins (`APP_DIR`, `LOG_DIR`).
*   **Incohérence:** `app/main.py` importe des deux. Il serait préférable d'unifier la configuration en un seul point de vérité (ex: `app/core/config.py` utilisant `pydantic-settings`).

## 3. Qualité du Code et Maintenabilité

### 3.1. "Fat Routes" (Routes Obèses)
Les fichiers de routes, notamment `app/routes/processing.py` (1700+ lignes) et `app/routes/ingestion.py`, contiennent beaucoup trop de logique.
*   **Violations:** Mélange de gestion HTTP, manipulation de fichiers, construction de commandes shell, et logique métier.
*   **Recommandation:** Déplacer toute la logique de traitement dans des services dédiés (`ProcessingService`, `IngestionService`). Les routes ne devraient faire que valider les entrées et appeler les services.

### 3.2. Gestion des Erreurs
*   Beaucoup de blocs `try...except Exception as e` qui capturent toutes les erreurs et renvoient une 500 générique.
*   Cela masque les erreurs spécifiques (ex: `FileNotFound`, `PermissionError`) et rend le débogage client difficile.
*   L'utilisation de `JSONResponse(status_code=500, ...)` est omniprésente au lieu d'utiliser les mécanismes d'exception FastAPI (`HTTPException`).

### 3.3. Duplication de Code
*   La logique de lecture/vérification des CSV semble dupliquée ou dispersée entre `app/routes/ingestion.py` et `app/routes/processing.py`.
*   La construction de l'environnement des sous-processus (`build_subprocess_env`) est répétée dans plusieurs endpoints.

## 4. Sécurité

### 4.1. Points Positifs
*   **Gestion des Credentials:** Le système `app/core/credentials.py` avec isolation ADMIN/NON-ADMIN et chiffrement en base est robuste.
*   **Isolation des Sessions:** Le `ProcessManager` assure que les processus sont bien liés aux sessions utilisateurs.

### 4.2. Points de Vigilance
*   **Injection de Commandes:** Bien que `subprocess.run` avec une liste d'arguments soit sûr, la construction dynamique des chemins et arguments doit être rigoureusement validée.
*   **Environnement Subprocess:** Passer des secrets via les variables d'environnement des sous-processus est une pratique acceptable mais nécessite une vigilance constante pour ne pas les leaker dans les logs (ce qui semble être géré, mais reste un risque).

## 5. Plan de Refactoring Complet et Optimisation

Ce plan détaille les étapes concrètes pour transformer RAGpy en une application de production robuste, performante et maintenable.

### Phase 1 : Assainissement des Fondations (Semaine 1)

**Objectif :** Stabiliser la base de code et unifier la configuration sans changer la logique métier.

1.  **Unification de la Configuration**
    *   **Action :** Fusionner `app/config.py` et `app/core/config.py`.
    *   **Implémentation :** Utiliser `pydantic-settings` pour valider toutes les variables d'environnement au démarrage.
    *   **Bénéfice :** Fail-fast en cas de configuration manquante, typage fort des settings.

2.  **Restructuration des Dossiers**
    *   **Action :** Déplacer le module `ingestion/` vers `app/services/ingestion/`.
    *   **Action :** Créer un package `app/services/` clair avec des sous-modules : `dataframe`, `chunking`, `embedding`, `vectordb`.
    *   **Bénéfice :** Structure standardisée, fin des imports relatifs douteux.

3.  **Typage et Linting**
    *   **Action :** Ajouter `mypy` et `ruff` au pipeline CI.
    *   **Action :** Typer strictement les modèles de données (Pydantic schemas pour toutes les entrées/sorties API).

### Phase 2 : Migration vers une Architecture de Services (Semaine 2-3)

**Objectif :** Supprimer la dépendance à `subprocess` pour le cœur du métier.

1.  **Extraction de la Logique des Scripts**
    *   **Action :** Refactoriser `scripts/rad_dataframe.py` en une classe `DataframeService`.
    *   **Action :** Refactoriser `scripts/rad_chunk.py` en une classe `ChunkingService`.
    *   **Action :** Refactoriser `scripts/rad_vectordb.py` en une classe `VectorDBService`.
    *   **Détail :** Ces services doivent être appelables directement en Python (plus de CLI args parsing à l'intérieur).

2.  **Suppression de `subprocess`**
    *   **Action :** Remplacer les appels `run_tracked_subprocess` dans les routes par des appels directs aux méthodes des services (ex: `await dataframe_service.process(...)`).
    *   **Gestion Async :** Pour les tâches longues, exécuter ces méthodes dans un threadpool (`run_in_executor`) ou via Celery (voir Phase 4).

### Phase 3 : Refactoring API et Injection de Dépendances (Semaine 4)

**Objectif :** Nettoyer les contrôleurs (routes) et améliorer la testabilité.

1.  **Injection de Dépendances (DI)**
    *   **Action :** Utiliser `Depends` de FastAPI pour injecter les services dans les routes.
    *   **Exemple :** `def process_dataframe(service: DataframeService = Depends(get_dataframe_service))`
    *   **Bénéfice :** Facilite le mock des services pour les tests.

2.  **Nettoyage des Routes ("Slim Controllers")**
    *   **Action :** Déplacer toute la logique de validation, de gestion de fichiers et de formatage de réponse des routes vers les services.
    *   **Cible :** Les fonctions de route ne doivent pas dépasser 20-30 lignes.

### Phase 4 : Performance et Asynchronisme (Semaine 5)

**Objectif :** Optimiser la réactivité et la scalabilité.

1.  **Intégration Celery Complète**
    *   **Action :** Utiliser Celery pour *toutes* les tâches de traitement (OCR, Chunking, Embedding).
    *   **Architecture :** Le endpoint HTTP renvoie immédiatement un `task_id`. Le client polle le statut ou utilise SSE pour les mises à jour.
    *   **Bénéfice :** Non-blocage du serveur web, gestion de file d'attente, retries automatiques.

2.  **Mise en Cache**
    *   **Action :** Implémenter un cache Redis pour les embeddings coûteux (éviter de re-calculer l'embedding d'un texte déjà traité).

### Phase 5 : Tests et Qualité (Continu)

1.  **Tests d'Intégration**
    *   **Action :** Créer une suite de tests utilisant `TestClient` de FastAPI et une base de données de test (SQLite in-memory).
    *   **Couverture :** Tester le flux complet : Upload -> Processing -> Result.

2.  **Tests Unitaires**
    *   **Action :** Tester chaque méthode de service isolément avec des mocks pour les appels externes (OpenAI, Pinecone).

## 6. Conclusion

Ce plan transforme RAGpy d'un prototype fonctionnel en une solution industrielle. L'investissement prioritaire est la **Phase 2 (Suppression de subprocess)**, qui éliminera la majorité de la fragilité actuelle et débloquera la capacité à tester correctement l'application.
