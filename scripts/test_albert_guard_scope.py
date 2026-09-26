"""Garde réseau active hors de ``tests/`` : le ``conftest.py`` racine couvre aussi ``scripts/``.

Une requête vers l'API Albert depuis ce fichier est refusée et enregistrée ;
l'enregistrement est acquitté par la fixture ``network_guard`` (sinon le test
échouerait au teardown). Aucun appel réseau n'a lieu.
"""

import httpx
import pytest
import requests


def test_network_guard_active_outside_tests_dir(network_guard, monkeypatch):
    monkeypatch.setenv("ALBERT_API_KEY", "fake-albert-key-0001")
    with pytest.raises(network_guard.error_class):
        httpx.get("https://albert.api.etalab.gouv.fr/health", timeout=1.0)
    with pytest.raises(network_guard.error_class):
        requests.get("https://albert.api.etalab.gouv.fr/v1/models", timeout=1.0)
    blocked = network_guard.consume()
    assert [(r.host, r.reason) for r in blocked] == [("albert.api.etalab.gouv.fr", "albert")] * 2
    assert network_guard.blocked() == []
