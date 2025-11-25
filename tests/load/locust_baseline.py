"""
Locust load testing script - Baseline Phase 1
Usage: locust -f tests/load/locust_baseline.py --host=http://localhost:8000

Tests disponibles:
1. Baseline (10 users, 5 min):
   locust -f tests/load/locust_baseline.py --host=http://localhost:8000 \
          --users 10 --spawn-rate 2 --run-time 5m --headless --csv=baseline

2. Stress test (30 users, 10 min):
   locust -f tests/load/locust_baseline.py --host=http://localhost:8000 \
          --users 30 --spawn-rate 5 --run-time 10m --headless --csv=stress

3. Peak test (50 users, 2 min):
   locust -f tests/load/locust_baseline.py --host=http://localhost:8000 \
          --users 50 --spawn-rate 10 --run-time 2m --headless --csv=peak
"""

from locust import HttpUser, task, between
import random
import json


class RAGpyUser(HttpUser):
    """Simule un utilisateur typique de RAGpy"""

    # Temps d'attente entre les requêtes (1-3 secondes)
    wait_time = between(1, 3)

    def on_start(self):
        """Setup initial - appelé une fois par utilisateur simulé"""
        self.session_id = None
        self.project_id = None

    @task(10)
    def health_check_simple(self):
        """Test le plus fréquent - health basique"""
        self.client.get("/health")

    @task(5)
    def health_check_detailed(self):
        """Health check détaillé avec métriques système"""
        with self.client.get("/health/detailed", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "degraded":
                    response.failure("System degraded")
            elif response.status_code == 503:
                response.failure("System unhealthy (503)")

    @task(3)
    def homepage(self):
        """Charger la page d'accueil"""
        self.client.get("/")

    @task(2)
    def static_assets(self):
        """Charger des assets statiques (CSS/JS)"""
        # Simule le chargement d'assets communs
        self.client.get("/static/css/style.css", name="/static/css/*")

    @task(1)
    def api_credentials_check(self):
        """Vérifier l'endpoint credentials (sans auth)"""
        # Cet endpoint devrait retourner 401 sans auth
        with self.client.get("/get_credentials", catch_response=True) as response:
            # 401 ou 403 sont attendus sans authentification
            if response.status_code in [401, 403, 307]:
                response.success()


class RAGpyAuthenticatedUser(HttpUser):
    """
    Utilisateur authentifié pour tests plus avancés.
    Note: Nécessite un compte test configuré.
    """

    wait_time = between(2, 5)
    weight = 1  # Moins fréquent que l'utilisateur non-auth

    # Désactivé par défaut - activer si compte test disponible
    abstract = True

    def on_start(self):
        """Login et setup"""
        # TODO: Implémenter login si nécessaire
        pass

    @task(5)
    def list_projects(self):
        """Lister les projets (nécessite auth)"""
        self.client.get("/api/pipeline/projects/")

    @task(2)
    def create_session(self):
        """Créer une session de traitement"""
        if self.project_id:
            self.client.post(
                f"/api/pipeline/projects/{self.project_id}/sessions",
                json={"name": f"Load Test Session {random.randint(1000, 9999)}"}
            )


# Configuration pour tests headless
if __name__ == "__main__":
    import os
    print("""
    RAGpy Load Testing - Baseline Phase 1
    =====================================

    Commandes disponibles:

    1. Test baseline (recommandé pour première mesure):
       locust -f tests/load/locust_baseline.py --host=http://localhost:8000 \\
              --users 10 --spawn-rate 2 --run-time 5m --headless --csv=baseline

    2. Interface web (pour exploration):
       locust -f tests/load/locust_baseline.py --host=http://localhost:8000
       Puis ouvrir http://localhost:8089

    3. Stress test:
       locust -f tests/load/locust_baseline.py --host=http://localhost:8000 \\
              --users 30 --spawn-rate 5 --run-time 10m --headless --csv=stress
    """)
