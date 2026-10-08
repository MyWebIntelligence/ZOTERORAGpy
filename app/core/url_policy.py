"""Contrôle des adresses personnelles (sprint « configuration unifiée », lot L9).

Un compte non administrateur peut enregistrer des adresses de service
(``weaviate_url``, ``qdrant_url``, ``mistral_url``) que le serveur contacte
ensuite en son nom : sans contrôle, ce serait une porte vers le réseau interne
(SSRF : Redis, Flower, métadonnées du nuage, conteneurs voisins). Règles :

* ``https`` obligatoire, sans identifiants dans l'adresse ;
* hôte nommé avec au moins un point (un nom court comme ``redis`` ou ``qdrant``
  désigne un conteneur voisin), hors suffixes réservés au réseau local ;
* adresse IP littérale publique seulement (ni privée, ni boucle locale, ni
  lien local, ni réservée) ; formes numériques non standard refusées.

Contrôle sans résolution DNS (hors ligne, déterministe) : un nom public qui
résoudrait vers une adresse privée n'est pas détecté ici. Les administrateurs
ne sont pas concernés (ils configurent le serveur lui-même).
"""

from __future__ import annotations

import ipaddress
import re
from typing import Optional
from urllib.parse import urlsplit

PERSONAL_URL_KEYS = ("mistral_url", "weaviate_url", "qdrant_url")
"""Identifiants personnels qui portent une adresse contactée par le serveur."""

_LOCAL_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".home.arpa", ".intranet", ".corp")
_NUMERIC_LABEL = re.compile(r"^(0x[0-9a-f]+|[0-9]+)$")


def personal_url_error(value: str) -> Optional[str]:
    """Raison du refus d'une adresse personnelle, ou ``None`` si elle est admise.

    Args:
        value: Adresse saisie (non vide).

    Returns:
        Message en français, ou ``None``.
    """
    try:
        parts = urlsplit(value.strip())
        host = (parts.hostname or "").lower()
        _port = parts.port  # ValueError si le port est invalide
    except ValueError:
        return "adresse invalide"
    if parts.scheme != "https":
        return "adresse en https obligatoire"
    if parts.username or parts.password:
        return "identifiants interdits dans l'adresse"
    if not host:
        return "hôte absent"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None:
        return None if ip.is_global else "adresse IP non publique refusée"
    if host == "localhost" or host.endswith(_LOCAL_SUFFIXES):
        return "nom réservé au réseau local refusé"
    labels = host.rstrip(".").split(".")
    if len(labels) < 2:
        return "nom d'hôte complet exigé (un nom court désigne une machine du réseau interne)"
    if all(_NUMERIC_LABEL.match(label) for label in labels):
        return "adresse IP non standard refusée"
    return None


__all__ = ["PERSONAL_URL_KEYS", "personal_url_error"]
