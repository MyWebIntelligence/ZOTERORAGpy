"""
Listes JSON lues et écrites en flux, en mémoire bornée (bibliothèque standard seulement).

Un fichier ``output_chunks_with_embeddings.json`` peut peser plusieurs Go
(livre de 1 800 pages : 1,8 Go, 21 147 chunks de 3 072 nombres). ``json.load``
le transforme en objets Python plusieurs fois plus gros : dans un conteneur
limité à 8 Go, le processus était tué (code -9). Ici, les éléments sont
décodés un par un (``JSONDecoder.raw_decode`` sur un tampon glissant) et
écrits un par un, octets identiques à ``json.dump(liste, indent=2,
ensure_ascii=False)`` : la mémoire reste de l'ordre d'un chunk.

Partagé par les scripts du pipeline (``rad_chunk.py``…) et l'application
(``app/utils/json_stream.py``).
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Iterable, Iterator

BLOCK_SIZE = 1 << 20
"""Taille des lectures (caractères)."""

LOOKAHEAD = 4096
"""Caractères disponibles au moins avant chaque décodage (nombres jamais coupés)."""

_WHITESPACE = " \t\r\n"


def iter_json_array(path: str, block_size: int = BLOCK_SIZE) -> Iterator[Any]:
    """
    Éléments d'un fichier contenant une liste JSON, un par un.

    Args:
        path: Fichier JSON (UTF-8).
        block_size: Taille des lectures.

    Yields:
        Chaque élément de la liste, dans l'ordre.

    Raises:
        json.JSONDecodeError: Le fichier ne contient pas une liste JSON valide
            (sous-classe de ``ValueError``).
    """
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as fh:
        state = {"buf": "", "idx": 0, "eof": False}

        def fill() -> bool:
            """Ajoute un bloc au tampon (en retirant la partie déjà lue) ; faux en fin de fichier."""
            data = fh.read(block_size)
            if not data:
                state["eof"] = True
                return False
            state["buf"] = state["buf"][state["idx"]:] + data
            state["idx"] = 0
            return True

        def skip_ws() -> None:
            """Avance au prochain caractère significatif (lit d'autres blocs si besoin)."""
            while True:
                buf, idx = state["buf"], state["idx"]
                while idx < len(buf) and buf[idx] in _WHITESPACE:
                    idx += 1
                state["idx"] = idx
                if idx < len(buf) or not fill():
                    return

        fill()
        skip_ws()
        if state["idx"] >= len(state["buf"]) or state["buf"][state["idx"]] != "[":
            raise json.JSONDecodeError(f"{os.path.basename(path)} : liste JSON attendue", state["buf"], state["idx"])
        state["idx"] += 1
        skip_ws()
        if state["idx"] < len(state["buf"]) and state["buf"][state["idx"]] == "]":
            return
        while True:
            # Marge de lecture : un nombre coupé en fin de tampon (« -1. » avant « 5e10 »)
            # se décoderait à tort ; objets et chaînes incomplets lèvent une erreur, relue.
            while not state["eof"] and len(state["buf"]) - state["idx"] < LOOKAHEAD and fill():
                pass
            while True:
                try:
                    obj, end = decoder.raw_decode(state["buf"], state["idx"])
                except json.JSONDecodeError:
                    if not fill():
                        raise
                    continue
                break
            yield obj
            state["idx"] = end
            skip_ws()
            if state["idx"] >= len(state["buf"]):
                raise json.JSONDecodeError(f"{os.path.basename(path)} : liste JSON tronquée", state["buf"], state["idx"])
            char = state["buf"][state["idx"]]
            state["idx"] += 1
            if char == "]":
                return
            if char != ",":
                raise json.JSONDecodeError(f"{os.path.basename(path)} : séparateur inattendu « {char} »",
                                           state["buf"], state["idx"] - 1)
            skip_ws()


def count_json_array(path: str) -> int:
    """
    Nombre d'éléments d'une liste JSON, en mémoire bornée.

    Args:
        path: Fichier JSON.

    Returns:
        Le nombre d'éléments.

    Raises:
        ValueError: Pas une liste JSON valide.
    """
    return sum(1 for _ in iter_json_array(path))


def load_json(path: str) -> Any:
    """
    Comme ``json.load`` sur ``path``, sans jamais lire tout le fichier dans une chaîne.

    Une liste est décodée élément par élément (``iter_json_array``) : même
    liste, mêmes objets, mais la mémoire de pointe est celle des objets seuls
    (``json.load`` lisait d'abord le fichier entier : 1,8 Go de texte, jusqu'à
    3,6 Go en mémoire avec des accents, avant de créer les objets). Tout autre
    contenu passe par ``json.load``.

    Args:
        path: Fichier JSON (UTF-8).

    Returns:
        L'objet décodé.

    Raises:
        json.JSONDecodeError: JSON invalide.
        OSError: Fichier illisible.
    """
    with open(path, "r", encoding="utf-8") as fh:
        head = fh.read(4096).lstrip()
    if head.startswith("["):
        return list(iter_json_array(path))
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json_array_atomic(items: Iterable[Any], path: str, *, indent: int = 2, ensure_ascii: bool = False) -> int:
    """
    Écrit une liste JSON élément par élément, de façon atomique.

    Octets identiques à ``json.dump(list(items), f, indent=indent,
    ensure_ascii=ensure_ascii)`` : chaque élément est sérialisé seul puis
    décalé d'un niveau. Fichier temporaire du même dossier, vidé sur disque,
    puis ``os.replace`` : une interruption laisse l'ancien fichier intact.

    Args:
        items: Éléments (itérable, consommé une seule fois).
        path: Fichier cible.
        indent: Indentation (comme ``json.dump``).
        ensure_ascii: Comme ``json.dump``.

    Returns:
        Le nombre d'éléments écrits.
    """
    directory = os.path.dirname(os.path.abspath(path))
    temporary = os.path.join(directory, f".{os.path.basename(path)}.{os.getpid()}.{threading.get_ident()}.tmp")
    pad = " " * indent
    count = 0
    try:
        with open(temporary, "w", encoding="utf-8") as fh:
            for item in items:
                text = json.dumps(item, ensure_ascii=ensure_ascii, indent=indent).replace("\n", "\n" + pad)
                fh.write(("[\n" if count == 0 else ",\n") + pad + text)
                count += 1
            fh.write("\n]" if count else "[]")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            try:
                os.remove(temporary)
            except OSError:
                pass
    return count


__all__ = ["count_json_array", "iter_json_array", "load_json", "write_json_array_atomic"]
