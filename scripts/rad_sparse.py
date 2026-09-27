"""scripts/rad_sparse.py — Encodage sparse stable et versionné des lemmes (audit A06).

Module **stdlib uniquement**, à partager avec les consommateurs des vecteurs
sparse (requêtes hybrides) : la même fonction doit produire les indices à
l'indexation (``rad_chunk.extract_sparse_features``) et à la requête.

Avant l'audit du 2026-09-27, l'indice d'un lemme valait ``hash(lemma) % 100000`` :
``hash`` des chaînes Python est salé par processus (``PYTHONHASHSEED``), donc
``recherche`` donnait 31606 dans un processus et 17079 dans un autre, et une
collision écrasait le poids déjà posé. Les vecteurs sparse produits ainsi ne
sont pas comparables d'un run à l'autre : les index hybrides construits avant
cette version doivent être ré-encodés (phase sparse relancée, puis envoi).

Encodage ``SPARSE_ENCODING`` (version 1) :

* indice = ``int.from_bytes(blake2b(lemme UTF-8, 8 octets), "big") % 100000`` ;
* poids = fréquence du lemme dans le chunk / nombre total de lemmes retenus (TF) ;
* collision de deux lemmes sur un même indice : poids **additionnés** ;
* indices triés par ordre croissant, sérialisés en chaînes (format historique
  des fichiers ``*_sparse.json`` ; les connecteurs les convertissent en entiers).

La sélection des lemmes (spaCy ``fr_core_news_md``, catégories NOUN, PROPN, ADJ,
VERB, sans mots vides ni ponctuation, lemmes de plus d'un caractère, en
minuscules) reste dans ``rad_chunk`` ; un consommateur doit l'appliquer à
l'identique.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Dict, Iterable, List

SPARSE_DIM = 100000
SPARSE_ENCODING = "lemma-tf/blake2b64-mod100000/v1"


def sparse_index(lemma: str) -> int:
    """Indice stable d'un lemme dans l'espace sparse (identique dans tout processus).

    Args:
        lemma: lemme déjà normalisé (minuscules).

    Returns:
        Un entier dans ``[0, SPARSE_DIM)``.
    """
    digest = hashlib.blake2b(str(lemma).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % SPARSE_DIM


def sparse_vector_from_lemmas(lemmas: Iterable[str]) -> Dict[str, List]:
    """Vecteur sparse TF d'une liste de lemmes (encodage ``SPARSE_ENCODING``).

    Args:
        lemmas: lemmes retenus du texte, répétitions comprises.

    Returns:
        ``{"indices": [str, ...], "values": [float, ...]}``, indices triés ;
        listes vides si aucun lemme.
    """
    counts = Counter(lemmas)
    total = sum(counts.values())
    if total <= 0:
        return {"indices": [], "values": []}
    weights: Dict[int, float] = {}
    for lemma, count in counts.items():
        index = sparse_index(lemma)
        weights[index] = weights.get(index, 0.0) + count / total
    ordered = sorted(weights)
    return {"indices": [str(index) for index in ordered], "values": [weights[index] for index in ordered]}
