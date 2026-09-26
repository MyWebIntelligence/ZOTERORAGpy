"""Ledger d'usage des appels Albert — stdlib seulement.

Chaque réponse d'inférence (chat, OCR, embeddings, recherche) est enregistrée,
ainsi que chaque envoi de chunks (rôle ``push`` : vectorisation côté serveur
qui consomme le quota bge-m3, D19 ; sans modèle ni tokens, la réponse n'en
porte pas) ; les appels de gestion (collections, documents, listes) ne le
sont pas. Chaque enregistrement porte le modèle **envoyé** (``model`` : id
épinglé après résolution, modèle de repli compris), le champ ``model``
**renvoyé** par l'API (``response_model``,
qui recopie le nom demandé, alias compris : D5, D12), les tokens,
``usage.cost`` et ``usage.impacts`` (D7). Le ledger ne produit rien tant
qu'Albert n'a pas été appelé : ``write_jsonl`` n'écrit pas de fichier vide et
``summary_line`` n'est destinée qu'aux traitements qui ont appelé Albert.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional

USAGE_FILENAME = "albert_usage.jsonl"
"""Nom conventionnel du fichier de ledger dans le dossier de session."""

_TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _number(value: Any) -> Optional[float]:
    """Nombre fini (``float``), ou ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int(value: Any) -> Optional[int]:
    """Entier (arrondi), ou ``None``."""
    number = _number(value)
    return int(round(number)) if number is not None else None


def _impact_value(value: Any) -> Optional[float]:
    """Valeur d'impact : nombre, ou ``{"min", "max"}`` (moyenne des bornes)."""
    if isinstance(value, Mapping):
        low, high = _number(value.get("min")), _number(value.get("max"))
        if low is not None and high is not None:
            return (low + high) / 2.0
        return low if low is not None else high
    return _number(value)


def extract_usage(payload: Any) -> Dict[str, Any]:
    """Extrait l'objet ``usage`` d'une réponse Albert, champs manquants tolérés.

    Args:
        payload: corps JSON d'une réponse (``{"usage": {...}}``) ou l'objet
            ``usage`` lui-même ; ``None`` donne un usage vide.

    Returns:
        ``{"prompt_tokens", "completion_tokens", "total_tokens", "cost",
        "impacts": {"kWh", "kgCO2eq"}, "requests"}`` (valeurs ``None`` si
        absentes). ``impacts`` est lu dans ``usage.impacts``, sinon dans
        ``usage.carbon`` (bornes min/max moyennées).
    """
    usage: Any = None
    if isinstance(payload, Mapping):
        if "usage" in payload:
            usage = payload.get("usage")
        elif any(name in payload for name in _TOKEN_FIELDS + ("cost", "impacts", "carbon")):
            usage = payload
    if not isinstance(usage, Mapping):
        usage = {}
    out: Dict[str, Any] = {name: _int(usage.get(name)) for name in _TOKEN_FIELDS}
    if out["total_tokens"] is None and (out["prompt_tokens"] is not None or out["completion_tokens"] is not None):
        out["total_tokens"] = (out["prompt_tokens"] or 0) + (out["completion_tokens"] or 0)
    out["cost"] = _number(usage.get("cost"))
    impacts_src = usage.get("impacts")
    if not isinstance(impacts_src, Mapping):
        impacts_src = usage.get("carbon") if isinstance(usage.get("carbon"), Mapping) else {}
    out["impacts"] = {
        "kWh": _impact_value(impacts_src.get("kWh")),
        "kgCO2eq": _impact_value(impacts_src.get("kgCO2eq")),
    }
    out["requests"] = _int(usage.get("requests"))
    return out


class UsageRecord(dict):
    """Enregistrement du ledger : dictionnaire JSON dont les clés se lisent aussi en attributs."""

    def __getattr__(self, name: str) -> Any:
        """Lit la clé ``name`` (``AttributeError`` si elle est absente)."""
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


class UsageLedger:
    """Journal thread-safe des appels Albert (tokens, coût, impacts, modèles).

    Attributes:
        records: enregistrements (``UsageRecord``), dans l'ordre d'arrivée.
        errors: nombre d'appels en échec signalés par ``record_error``.
    """

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        """Crée un ledger vide.

        Args:
            clock: horloge murale (horodatage ``ts`` des enregistrements).
        """
        self._clock = clock
        self._lock = threading.Lock()
        self.records: List[UsageRecord] = []
        self.errors = 0
        self._written = 0

    def record(
        self,
        *,
        endpoint: str,
        model: Optional[str] = None,
        response_model: Optional[str] = None,
        usage: Any = None,
        role: Optional[str] = None,
        latency_s: Optional[float] = None,
        request_id: Optional[str] = None,
        status: Optional[int] = None,
        fallback_from: Optional[str] = None,
        items: Optional[int] = None,
        finish_reason: Optional[str] = None,
    ) -> UsageRecord:
        """Enregistre une réponse Albert.

        Args:
            endpoint: chemin appelé (``/v1/chat/completions``…).
            model: nom de modèle envoyé (id épinglé après résolution ; modèle
                qui a servi la réponse, repli compris).
            response_model: champ ``model`` de la réponse (``response.model``).
            usage: objet ``usage`` ou corps complet de la réponse.
            role: rôle applicatif.
            latency_s: durée de l'appel, en secondes.
            request_id: identifiant de la réponse (``id``).
            status: statut HTTP.
            fallback_from: modèle primaire quand un repli a servi la réponse.
            items: nombre d'éléments traités (textes, pages, chunks).
            finish_reason: motif de fin (chat).

        Returns:
            L'enregistrement ajouté.
        """
        parsed = extract_usage(usage)
        entry = UsageRecord({
            "ts": round(float(self._clock()), 3),
            "endpoint": endpoint,
            "role": role,
            "model": model if model is not None else response_model,
            "response_model": response_model,
            "prompt_tokens": parsed["prompt_tokens"],
            "completion_tokens": parsed["completion_tokens"],
            "total_tokens": parsed["total_tokens"],
            "cost": parsed["cost"],
            "impacts": parsed["impacts"],
            "requests": parsed["requests"],
            "latency_s": round(float(latency_s), 3) if latency_s is not None else None,
            "request_id": request_id,
            "status": status,
            "fallback_from": fallback_from,
            "items": items,
            "finish_reason": finish_reason,
        })
        with self._lock:
            self.records.append(entry)
        return entry

    def record_error(self, *, endpoint: str, status: Optional[int] = None, model: Optional[str] = None) -> None:
        """Compte un appel en échec (aucun enregistrement détaillé)."""
        with self._lock:
            self.errors += 1

    def __len__(self) -> int:
        """Nombre d'enregistrements."""
        with self._lock:
            return len(self.records)

    def __iter__(self) -> Iterator[UsageRecord]:
        """Itère sur une copie des enregistrements."""
        with self._lock:
            return iter(list(self.records))

    @property
    def called(self) -> bool:
        """Vrai si au moins un appel Albert a été enregistré (réussi ou en échec)."""
        with self._lock:
            return bool(self.records) or self.errors > 0

    def totals(self) -> Dict[str, Any]:
        """Totaux : appels, tokens, coût, impacts et modèles servis.

        Returns:
            ``{"calls", "errors", "prompt_tokens", "completion_tokens",
            "total_tokens", "cost", "kWh", "kgCO2eq", "models"}``.
        """
        with self._lock:
            records = list(self.records)
            errors = self.errors
        totals: Dict[str, Any] = {
            "calls": len(records),
            "errors": errors,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
            "kWh": 0.0,
            "kgCO2eq": 0.0,
        }
        models: List[str] = []
        for entry in records:
            for name in _TOKEN_FIELDS:
                totals[name] += entry.get(name) or 0
            totals["cost"] += entry.get("cost") or 0.0
            impacts = entry.get("impacts") or {}
            totals["kWh"] += impacts.get("kWh") or 0.0
            totals["kgCO2eq"] += impacts.get("kgCO2eq") or 0.0
            served = entry.get("model") or entry.get("response_model")
            if served and served not in models:
                models.append(served)
        totals["models"] = models
        return totals

    def summary_line(self) -> str:
        """Ligne de synthèse en français (à n'afficher que si Albert a été appelé)."""
        t = self.totals()
        models = ", ".join(t["models"]) if t["models"] else "-"
        line = (
            f"Albert : {t['calls']} appel(s), {t['prompt_tokens']} tokens d'entrée, "
            f"{t['completion_tokens']} tokens de sortie, coût {t['cost']:.6g}, "
            f"{t['kWh']:.3g} kWh, {t['kgCO2eq']:.3g} kgCO2eq ; modèles : {models}"
        )
        if t["errors"]:
            line += f" ; {t['errors']} échec(s)"
        return line

    def write_jsonl(self, path: Any) -> Optional[str]:
        """Ajoute au fichier JSONL les enregistrements non encore écrits.

        Rien n'est écrit (et aucun fichier n'est créé) si aucun enregistrement
        n'est en attente.

        Args:
            path: chemin du fichier (``str`` ou ``Path``), ou dossier
                (``albert_usage.jsonl`` y est créé).

        Returns:
            Le chemin écrit, ou ``None`` si rien n'était à écrire.
        """
        with self._lock:
            pending = self.records[self._written:]
            if not pending:
                return None
            target = os.path.join(path, USAGE_FILENAME) if os.path.isdir(path) else os.fspath(path)
            folder = os.path.dirname(os.path.abspath(target))
            os.makedirs(folder, exist_ok=True)
            with open(target, "a", encoding="utf-8") as fh:
                for entry in pending:
                    fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
                fh.flush()
            self._written = len(self.records)
            return target


__all__ = ["USAGE_FILENAME", "UsageLedger", "UsageRecord", "extract_usage"]
