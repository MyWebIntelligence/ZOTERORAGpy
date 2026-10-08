"""Reprise page par page de l'OCR LightOnOCR (sprint Albert R2, lot L3) — stdlib seulement.

Une page transcrite avec succès (``finish_reason`` autre que ``length``) est
écrite dans le dossier de sortie du traitement (dossier de session pour
l'application web) ; une relance réutilise ces pages et n'envoie à Albert que
les pages manquantes, en échec ou tronquées. Utile en régime d'expérimentation
(1 000 requêtes par jour et par modèle) : un livre interrompu par le quota
journalier reprend le lendemain là où il s'était arrêté.

Une page n'est réutilisée que si **tout** correspond :

* le document : empreinte ``sha256`` du fichier (16 caractères) ;
* les paramètres : modèle épinglé, DPI, plus grand côté, ``max_tokens``,
  température, ``top_p`` et version du format (16 caractères) ;
* l'indice de page.

Disposition : ``<racine>/albert_ocr_checkpoints/<document>/<paramètres>/page_NNNNN.json``,
écriture atomique (fichier temporaire puis ``os.replace``), chemin confiné à
la racine. Rien n'est partagé entre sessions ni entre utilisateurs. Un
fichier illisible ou incohérent est ignoré (la page est retranscrite).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from typing import Any, Dict, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1
DIRNAME = "albert_ocr_checkpoints"
"""Nom du dossier des reprises, dans le dossier de sortie."""

_READ_CHUNK = 1024 * 1024


def file_digest(path: str) -> str:
    """Empreinte ``sha256`` d'un fichier (16 premiers caractères hexadécimaux).

    Args:
        path: chemin du fichier.

    Returns:
        L'empreinte.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_READ_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def params_digest(params: Mapping[str, Any]) -> str:
    """Empreinte des paramètres de transcription (version du format comprise).

    Args:
        params: paramètres (modèle, DPI, plus grand côté, ``max_tokens``…).

    Returns:
        16 caractères hexadécimaux.
    """
    payload = dict(params)
    payload["checkpoint_version"] = CHECKPOINT_VERSION
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def lightonocr_params(cfg: Any, model: str) -> Dict[str, Any]:
    """Paramètres qui changent le texte d'une page LightOnOCR.

    Args:
        cfg: configuration Albert (``ocr_dpi``, ``ocr_max_side``, ``ocr_max_tokens``,
            ``ocr_temperature``, ``ocr_top_p``).
        model: id épinglé du modèle envoyé.

    Returns:
        Le dictionnaire des paramètres.
    """
    return {
        "model": str(model),
        "dpi": int(getattr(cfg, "ocr_dpi", 200)),
        "max_side": int(getattr(cfg, "ocr_max_side", 1540)),
        "max_tokens": int(getattr(cfg, "ocr_max_tokens", 4096)),
        "temperature": float(getattr(cfg, "ocr_temperature", 0.2)),
        "top_p": float(getattr(cfg, "ocr_top_p", 0.9)),
    }


class PageCheckpoint:
    """Pages LightOnOCR validées d'un document, pour une configuration donnée.

    Attributes:
        root: dossier de sortie (racine de confinement).
        directory: dossier des pages de ce document et de ces paramètres.
        doc_key: empreinte du document.
        param_key: empreinte des paramètres.
        reused: nombre de pages relues lors du dernier ``load``.
        saved: nombre de pages écrites.
    """

    def __init__(self, root: str, pdf_path: str, params: Mapping[str, Any]) -> None:
        """Prépare le point de reprise (aucune écriture avant ``save``).

        Args:
            root: dossier de sortie du traitement.
            pdf_path: document transcrit.
            params: paramètres de transcription (``lightonocr_params``).

        Raises:
            ValueError: dossier calculé hors de la racine.
            OSError: document illisible.
        """
        self.root = os.path.realpath(root)
        self.doc_key = file_digest(pdf_path)
        self.param_key = params_digest(params)
        directory = os.path.realpath(os.path.join(self.root, DIRNAME, self.doc_key, self.param_key))
        if os.path.commonpath([directory, self.root]) != self.root:
            raise ValueError("Point de reprise OCR hors du dossier de sortie.")
        self.directory = directory
        self.reused = 0
        self.saved = 0

    def _path(self, index: int) -> str:
        """Fichier d'une page (indice à partir de 0)."""
        return os.path.join(self.directory, f"page_{int(index):05d}.json")

    def load(self, indices: Iterable[int]) -> Dict[int, str]:
        """Textes des pages déjà validées parmi ``indices``.

        Args:
            indices: indices de pages (à partir de 0).

        Returns:
            ``{indice: texte}`` des pages réutilisables.
        """
        out: Dict[int, str] = {}
        if not os.path.isdir(self.directory):
            self.reused = 0
            return out
        for index in indices:
            path = self._path(index)
            try:
                with open(path, encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, ValueError):
                continue
            if (
                not isinstance(data, dict)
                or data.get("doc") != self.doc_key
                or data.get("params") != self.param_key
                or data.get("index") != int(index)
                or not isinstance(data.get("text"), str)
            ):
                continue
            out[int(index)] = data["text"]
        self.reused = len(out)
        return out

    def save(self, index: int, text: str) -> bool:
        """Écrit une page validée (écriture atomique ; une erreur n'interrompt jamais l'OCR).

        Args:
            index: indice de la page (à partir de 0).
            text: texte transcrit (page non tronquée).

        Returns:
            Vrai si la page a été écrite.
        """
        try:
            os.makedirs(self.directory, exist_ok=True)
            payload = {"doc": self.doc_key, "params": self.param_key, "index": int(index), "text": str(text)}
            fd, tmp = tempfile.mkstemp(prefix=".page_", suffix=".tmp", dir=self.directory)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False)
                os.replace(tmp, self._path(index))
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except OSError as exc:
            logger.warning("Point de reprise OCR non écrit (page %d) : %s", int(index) + 1, type(exc).__name__)
            return False
        self.saved += 1
        return True


def open_checkpoint(root: Optional[str], pdf_path: str, cfg: Any, model: str) -> Optional[PageCheckpoint]:
    """Point de reprise d'un document, ou ``None`` (désactivé, racine absente, erreur).

    Args:
        root: dossier de sortie (``None`` : pas de reprise).
        pdf_path: document transcrit.
        cfg: configuration Albert (``ocr_checkpoint`` et paramètres OCR).
        model: id épinglé du modèle LightOnOCR.

    Returns:
        Le point de reprise, ou ``None``.
    """
    if not root or not bool(getattr(cfg, "ocr_checkpoint", True)):
        return None
    try:
        return PageCheckpoint(root, pdf_path, lightonocr_params(cfg, model))
    except (OSError, ValueError) as exc:
        logger.warning("Reprise OCR désactivée pour ce document : %s", exc)
        return None


__all__ = [
    "CHECKPOINT_VERSION",
    "DIRNAME",
    "PageCheckpoint",
    "file_digest",
    "lightonocr_params",
    "open_checkpoint",
    "params_digest",
]
