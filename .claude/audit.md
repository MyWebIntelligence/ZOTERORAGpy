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

## 5. Plan de Refactoring Recommandé

Pour transformer RAGpy en une application robuste, voici les étapes recommandées :

### Phase 1 : Consolidation (Court Terme)
1.  **Déplacer `ingestion/`** dans `app/ingestion` ou `app/services/ingestion`.
2.  **Unifier la config** dans `app/core/config.py` avec `pydantic-settings`.
3.  **Nettoyer les imports** circulaires ou inutiles détectés lors du déplacement.

### Phase 2 : Service Layer (Moyen Terme)
1.  **Transformer les Scripts en Services:**
    *   Refactoriser `scripts/rad_dataframe.py` -> `app/services/dataframe_service.py`.
    *   Refactoriser `scripts/rad_chunk.py` -> `app/services/chunking_service.py`.
    *   Ces services doivent exposer des méthodes Python (`process_dataframe(...)`) au lieu d'être appelés via CLI.
2.  **Supprimer `subprocess`:** Remplacer les appels `run_tracked_subprocess` par des appels de fonctions asynchrones (via `asyncio` ou `Celery` pour les tâches longues).

### Phase 3 : Nettoyage des Routes (Long Terme)
1.  **Alléger les contrôleurs:** `app/routes/processing.py` ne doit contenir que la logique HTTP.
2.  **Injection de Dépendances:** Utiliser le système de DI de FastAPI pour injecter les services dans les routes.
3.  **Tests:** Écrire des tests unitaires pour les nouveaux services (maintenant testables sans subprocess).

## 6. Conclusion

RAGpy a une base solide en termes de fonctionnalités et de modèle de données. L'effort principal doit porter sur la suppression de la couche "scripting" au profit d'une architecture orientée services. Cela améliorera drastiquement la performance, la testabilité et la maintenabilité du projet.
