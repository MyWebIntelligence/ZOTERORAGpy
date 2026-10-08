"""Catalogue des modèles Albert (DINUM) et chaînes de repli par rôle — stdlib seulement.

Le catalogue est une **donnée** : ids épinglés, alias, type, contexte,
raisonnement, dimension d'embedding, dates de retrait (``deprecated_on``) et
de réexamen (``review_on``). Les valeurs viennent de la sonde P2
(``tests/fixtures/albert/P2_models.json``, décision D2) et de la référence
``albertai.md`` §4.

* ``MODELS`` : modèles utilisables par défaut ou en repli, indexés par id.
* ``EXCLUDED`` : modèles jamais retenus par défaut ni en repli (nom → motif).
* ``ROLES`` : rôle → point d'accès et chaîne ordonnée (primaire puis replis).
* ``fallback_chain(role, *, today)`` : chaîne effective à une date donnée
  (modèles retirés écartés, remplaçants activés à leur date). Les embeddings
  n'ont **jamais** de repli.
* ``resolve_model(requested, listing)`` : alias ou id → id réel d'après la
  réponse de ``/v1/models`` (``response.model`` renvoie l'alias demandé, D5
  et D12 : il ne sert jamais à connaître l'id) ; ``listing_name_map(listing)``
  donne la même résolution pour tous les noms de la liste d'un coup.
* ``check_endpoint_type``, ``is_reasoning``, ``embedding_dim``,
  ``canonical_id`` : lectures statiques du catalogue, sans réseau.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .errors import AlbertPermanentError

# ---------------------------------------------------------------------------
# Types de modèles et points d'accès
# ---------------------------------------------------------------------------
TYPE_TEXT = "text-generation"
TYPE_VISION = "image-text-to-text"
TYPE_OCR = "image-to-text"
TYPE_EMBEDDINGS = "text-embeddings-inference"
TYPE_RERANK = "text-classification"
TYPE_ASR = "automatic-speech-recognition"

ENDPOINT_TYPES: Mapping[str, Tuple[str, ...]] = MappingProxyType(
    {
        "chat": (TYPE_TEXT, TYPE_VISION),
        "embeddings": (TYPE_EMBEDDINGS,),
        "ocr": (TYPE_OCR, TYPE_VISION),
        "rerank": (TYPE_RERANK,),
        "audio": (TYPE_ASR,),
    }
)
"""Types de modèle acceptés par chaque point d'accès (albertai.md §4.1)."""

_ENDPOINT_ALIASES = {
    "chat": "chat",
    "chat/completions": "chat",
    "/v1/chat/completions": "chat",
    "/chat/completions": "chat",
    "embeddings": "embeddings",
    "embed": "embeddings",
    "/v1/embeddings": "embeddings",
    "/embeddings": "embeddings",
    "ocr": "ocr",
    "/v1/ocr": "ocr",
    "/ocr": "ocr",
    "rerank": "rerank",
    "/v1/rerank": "rerank",
    "/rerank": "rerank",
    "audio": "audio",
    "audio/transcriptions": "audio",
    "/v1/audio/transcriptions": "audio",
    "/audio/transcriptions": "audio",
}

_ENDPOINT_PATHS = {
    "chat": "/v1/chat/completions",
    "embeddings": "/v1/embeddings",
    "ocr": "/v1/ocr",
    "rerank": "/v1/rerank",
    "audio": "/v1/audio/transcriptions",
}


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """Fiche d'un modèle du catalogue.

    Attributes:
        id: identifiant épinglé (clé de cache, reproductibilité).
        aliases: alias acceptés par l'API (``openweight-*``, noms Hugging Face).
        type: type Albert (``text-generation``, ``image-text-to-text``…).
        context: contexte maximal en tokens (``None`` si inconnu).
        reasoning: modèle à raisonnement (marge de ``max_tokens`` nécessaire).
        dim: dimension des vecteurs (modèles d'embeddings seulement).
        deprecated_on: date de retrait ; à partir de ce jour le modèle quitte
            toutes les chaînes.
        review_on: date à laquelle le statut doit être revérifié (fin de phase
            de test annoncée), sans retrait automatique.
        temperature: température recommandée (``None`` = défaut du modèle).
        restricted: accès sur demande seulement.
        batch_max: lot maximal par requête (embeddings).
        note: remarque en français.
    """

    id: str
    aliases: Tuple[str, ...]
    type: str
    context: Optional[int]
    reasoning: bool = False
    dim: Optional[int] = None
    deprecated_on: Optional[date] = None
    review_on: Optional[date] = None
    temperature: Optional[float] = None
    restricted: bool = False
    batch_max: Optional[int] = None
    note: str = ""

    def names(self) -> Tuple[str, ...]:
        """Renvoie l'id suivi des alias."""
        return (self.id,) + tuple(self.aliases)


@dataclass(frozen=True)
class ChainEntry:
    """Maillon d'une chaîne de rôle.

    Attributes:
        model: id épinglé du modèle.
        since: date d'entrée dans la chaîne (``None`` = toujours) ; sert aux
            remplaçants annoncés (``gemma-4-31b-it`` au retrait de mistral-small).
    """

    model: str
    since: Optional[date] = None


@dataclass(frozen=True)
class RoleSpec:
    """Rôle applicatif : point d'accès, chaîne ordonnée et budget du limiteur.

    Attributes:
        name: nom du rôle.
        endpoint: ``chat``, ``ocr``, ``embeddings``, ``rerank`` ou ``audio``.
        chain: maillons dans l'ordre (primaire en tête).
        bucket: budget du limiteur proactif (``recode``, ``notes``, ``ocr``,
            ``embed``, ``rerank``, ``audio``).
        optional: rôle dont l'absence au catalogue du compte n'est pas une
            erreur (``ocr_doc`` : accès restreint, D3).
        description: sens du rôle, en français.
    """

    name: str
    endpoint: str
    chain: Tuple[ChainEntry, ...]
    bucket: str
    optional: bool = False
    description: str = ""

    @property
    def primary(self) -> str:
        """Id du modèle primaire (premier maillon)."""
        return self.chain[0].model

    @property
    def fallbacks(self) -> Tuple[str, ...]:
        """Ids des maillons de repli, toutes dates confondues."""
        return tuple(entry.model for entry in self.chain[1:])


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------
MISTRAL_SMALL_RETIREMENT = date(2026, 12, 1)
"""Retrait de ``mistral-small-3-2-24b-instruct-2506`` (remplacé par gemma-4-31b-it)."""

TEST_PHASE_END = date(2026, 10, 1)
"""Fin de phase de test annoncée (qwen3-coder, deepseek, qwen3-vl, statut LightOnOCR)."""

_MODELS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        id="gpt-oss-120b",
        aliases=("openweight-large", "openai/gpt-oss-120b"),
        type=TYPE_TEXT,
        context=131072,
        reasoning=True,
        note="Raisonnement : prévoir une marge de max_tokens (piège content vide, P9).",
    ),
    ModelSpec(
        id="ministral-3-8b-instruct-2512",
        aliases=("mistralai/Ministral-3-8B-Instruct-2512", "openweight-small"),
        type=TYPE_VISION,
        context=262144,
        temperature=0.1,
        note="Modèle de volume : recodage, filtre de citations, structure de livre.",
    ),
    ModelSpec(
        id="mistral-small-3-2-24b-instruct-2506",
        aliases=("mistralai/Mistral-Small-3.2-24B-Instruct-2506", "openweight-medium"),
        type=TYPE_VISION,
        context=128000,
        deprecated_on=MISTRAL_SMALL_RETIREMENT,
        temperature=0.15,
        note="Retiré le 2026-12-01 ; l'alias openweight-medium passe à gemma-4-31b-it.",
    ),
    ModelSpec(
        id="gemma-4-31b-it",
        aliases=("google/gemma-4-31B-it",),
        type=TYPE_VISION,
        context=262144,
        review_on=MISTRAL_SMALL_RETIREMENT,
        note="Expérimental jusqu'au 2026-12-01 ; remplaçant de mistral-small dans les replis.",
    ),
    ModelSpec(
        id="lightonocr-2-1b",
        aliases=("lightonai/LightOnOCR-2-1B", "openweight-ocr"),
        type=TYPE_VISION,
        context=16384,
        review_on=TEST_PHASE_END,
        note="OCR par chat (image seule) ; statut à revérifier après le 2026-10-01.",
    ),
    ModelSpec(
        id="mistral-ocr-2512",
        aliases=(),
        type=TYPE_OCR,
        context=16384,
        restricted=True,
        note="/v1/ocr, accès restreint (absent du compte sondé, D3).",
    ),
    ModelSpec(
        id="bge-m3",
        aliases=("BAAI/bge-m3", "openweight-embeddings"),
        type=TYPE_EMBEDDINGS,
        context=8192,
        dim=1024,
        batch_max=64,
        note="1024 dimensions, lots de 64 textes ; jamais de repli.",
    ),
    ModelSpec(
        id="bge-reranker-v2-m3",
        aliases=("BAAI/bge-reranker-v2-m3", "openweight-rerank"),
        type=TYPE_RERANK,
        context=8192,
        batch_max=64,
        note="Rerank (convention Cohere v2), 64 textes au plus (P22 : 65 → 413) ; scores non calibrés.",
    ),
    ModelSpec(
        id="whisper-large-v3",
        aliases=("openai/whisper-large-v3", "openweight-audio"),
        type=TYPE_ASR,
        context=None,
        note="Transcription (mp3 ou wav, 20 Mo au plus par fichier) ; verbose_json donne les segments (P26).",
    ),
)

MODELS: Mapping[str, ModelSpec] = MappingProxyType({spec.id: spec for spec in _MODELS})
"""Modèles utilisables par défaut ou en repli, indexés par id épinglé."""

_EXCLUDED_SPECS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        id="qwen3-coder-30b-a3b-instruct",
        aliases=("openweight-code", "Qwen/Qwen3-Coder-30B-A3B-Instruct"),
        type=TYPE_TEXT,
        context=262144,
        deprecated_on=TEST_PHASE_END,
    ),
    ModelSpec(
        id="deepseek-v4-flash-0731",
        aliases=("deepseek-ai/DeepSeek-V4-Flash-0731", "deepseek-v4-flash"),
        type=TYPE_TEXT,
        context=131072,
        reasoning=True,
        review_on=TEST_PHASE_END,
    ),
    ModelSpec(
        id="qwen3-vl-embedding-8b",
        aliases=("Qwen/Qwen3-VL-Embedding-8B",),
        type=TYPE_EMBEDDINGS,
        context=32768,
        dim=4096,
        review_on=TEST_PHASE_END,
    ),
    ModelSpec(
        id="mistral-medium-2508",
        aliases=(),
        type=TYPE_VISION,
        context=131072,
        restricted=True,
    ),
)

EXCLUDED: Mapping[str, str] = MappingProxyType(
    {
        "qwen3-coder-30b-a3b-instruct": "retiré le 2026-10-01",
        "deepseek-v4-flash-0731": "fin de phase de test le 2026-10-01 ; censure possible sur des sujets sensibles",
        "deepseek-v4-flash": "fin de phase de test le 2026-10-01 ; censure possible sur des sujets sensibles",
        "qwen3-vl-embedding-8b": "espace de 4096 dimensions ; fin de phase de test le 2026-10-01",
        "mistral-medium-2508": "accès restreint (modèle partenaire)",
    }
)
"""Modèles exclus des défauts et des replis (nom → motif)."""

_SMALL_CHAIN: Tuple[ChainEntry, ...] = (
    ChainEntry("ministral-3-8b-instruct-2512"),
    ChainEntry("mistral-small-3-2-24b-instruct-2506"),
    ChainEntry("gemma-4-31b-it", since=MISTRAL_SMALL_RETIREMENT),
)

ROLES: Mapping[str, RoleSpec] = MappingProxyType(
    {
        "recode": RoleSpec(
            "recode", "chat", _SMALL_CHAIN, "recode",
            description="Recodage des chunks (rad_chunk).",
        ),
        "citation": RoleSpec(
            "citation", "chat", _SMALL_CHAIN, "recode",
            description="Pré-filtre et filtre de citations.",
        ),
        "notes": RoleSpec(
            "notes", "chat",
            (ChainEntry("gpt-oss-120b"), ChainEntry("ministral-3-8b-instruct-2512")),
            "notes",
            description="Notes Zotero (extended, pedagogique, evaluation) et synthèse de livre.",
        ),
        "book_structure": RoleSpec(
            "book_structure", "chat", _SMALL_CHAIN, "recode",
            description="Phase 1 des fiches de livre (structure en JSON).",
        ),
        "long_context": RoleSpec(
            "long_context", "chat", _SMALL_CHAIN, "recode",
            description="Invites trop longues pour gpt-oss (plus de 0,9 × 131 072 tokens).",
        ),
        "ocr_doc": RoleSpec(
            "ocr_doc", "ocr",
            (ChainEntry("mistral-ocr-2512"), ChainEntry("lightonocr-2-1b")),
            "ocr",
            optional=True,
            description="OCR de document par /v1/ocr (accès restreint), puis LightOnOCR.",
        ),
        "ocr_chat": RoleSpec(
            "ocr_chat", "chat", (ChainEntry("lightonocr-2-1b"),), "ocr",
            description="OCR par chat sur pages rastérisées (image seule).",
        ),
        "embed": RoleSpec(
            "embed", "embeddings", (ChainEntry("bge-m3"),), "embed",
            description="Embeddings bge-m3 (1024 d) ; aucun repli, jamais.",
        ),
        "answer": RoleSpec(
            "answer", "chat",
            (ChainEntry("gpt-oss-120b"), ChainEntry("ministral-3-8b-instruct-2512")),
            "recode",
            description="Réponses sourcées (RAG) ; le budget suit le modèle envoyé (gpt-oss → notes).",
        ),
        "vision": RoleSpec(
            "vision", "chat", _SMALL_CHAIN, "recode",
            description="Description d'images par un modèle vision (pièces jointes images).",
        ),
        "rerank": RoleSpec(
            "rerank", "rerank", (ChainEntry("bge-reranker-v2-m3"),), "rerank",
            description="Reclassement de passages (bge-reranker-v2-m3) ; aucun repli.",
        ),
        "audio": RoleSpec(
            "audio", "audio", (ChainEntry("whisper-large-v3"),), "audio",
            description="Transcription d'enregistrements (whisper-large-v3) ; aucun repli.",
        ),
    }
)
"""Rôles applicatifs et leurs chaînes (tableau « Modèles par rôle » du sprint)."""

NO_FALLBACK_ROLES = frozenset({"embed", "ocr_chat", "rerank", "audio"})
"""Rôles sans repli Albert (embeddings : jamais d'espace mélangé ; rerank et
audio : un seul modèle par type)."""


# ---------------------------------------------------------------------------
# Index des noms
# ---------------------------------------------------------------------------
def _build_name_index(specs: Iterable[ModelSpec]) -> Dict[str, ModelSpec]:
    """Construit l'index nom (id ou alias, et sa forme minuscule) → fiche."""
    index: Dict[str, ModelSpec] = {}
    for spec in specs:
        for name in spec.names():
            index.setdefault(name, spec)
    for spec in specs:
        for name in spec.names():
            index.setdefault(name.lower(), spec)
    return index


_ALL_SPECS: Tuple[ModelSpec, ...] = _MODELS + _EXCLUDED_SPECS
_NAME_INDEX: Dict[str, ModelSpec] = _build_name_index(_ALL_SPECS)


class UnknownRoleError(ValueError, KeyError):
    """Rôle absent de ``ROLES`` (``ValueError`` et ``KeyError`` à la fois)."""

    def __str__(self) -> str:
        """Message sans les guillemets ajoutés par ``KeyError``."""
        return str(self.args[0]) if self.args else ""


class ModelNotFoundError(AlbertPermanentError, ValueError, LookupError):
    """Modèle absent de ``/v1/models`` : erreur explicite, jamais de repli silencieux."""


def _clean_name(model: Any) -> str:
    """Nom de modèle nettoyé (``''`` pour ``None``)."""
    return str(model).strip() if model is not None else ""


def model_spec(model: Any) -> Optional[ModelSpec]:
    """Fiche du catalogue pour un id ou un alias (casse ignorée), exclus compris.

    Args:
        model: id ou alias.

    Returns:
        La fiche, ou ``None`` si le nom est inconnu du catalogue.
    """
    name = _clean_name(model)
    if not name:
        return None
    return _NAME_INDEX.get(name) or _NAME_INDEX.get(name.lower())


def canonical_id(model: Any) -> str:
    """Id épinglé d'un id ou d'un alias connu du catalogue.

    Args:
        model: id ou alias (``openweight-small``, ``BAAI/bge-m3``…).

    Returns:
        L'id épinglé ; un nom inconnu est renvoyé tel quel (sans blancs).
    """
    spec = model_spec(model)
    return spec.id if spec is not None else _clean_name(model)


def is_excluded(model: Any) -> bool:
    """Vrai si le modèle (id ou alias) est exclu des défauts et des replis."""
    name = _clean_name(model)
    if not name:
        return False
    return name in EXCLUDED or canonical_id(name) in EXCLUDED


def is_reasoning(model: Any) -> bool:
    """Vrai pour un modèle à raisonnement (gpt-oss) : marge de ``max_tokens`` requise.

    Args:
        model: id ou alias ; un nom inconnu contenant ``gpt-oss`` est
            considéré comme un modèle à raisonnement.

    Returns:
        ``True`` si des tokens de raisonnement consomment ``max_tokens``.
    """
    spec = model_spec(model)
    if spec is not None:
        return spec.reasoning
    return "gpt-oss" in _clean_name(model).lower()


def embedding_dim(model: Any) -> Optional[int]:
    """Dimension des vecteurs d'un modèle d'embeddings (``None`` si inconnue)."""
    spec = model_spec(model)
    return spec.dim if spec is not None else None


def context_length(model: Any) -> Optional[int]:
    """Contexte maximal (tokens) d'un modèle du catalogue (``None`` si inconnu)."""
    spec = model_spec(model)
    return spec.context if spec is not None else None


def recommended_temperature(model: Any) -> Optional[float]:
    """Température recommandée d'un modèle (``None`` = défaut du modèle)."""
    spec = model_spec(model)
    return spec.temperature if spec is not None else None


def role_spec(role: Any) -> RoleSpec:
    """Fiche d'un rôle.

    Args:
        role: nom du rôle (``recode``, ``citation``, ``notes``, ``book_structure``,
            ``long_context``, ``ocr_doc``, ``ocr_chat``, ``embed``, ``answer``,
            ``vision``, ``rerank``, ``audio``).

    Returns:
        La fiche du rôle.

    Raises:
        UnknownRoleError: rôle inconnu (``ValueError`` et ``KeyError``).
    """
    name = _clean_name(role).lower()
    spec = ROLES.get(name)
    if spec is None:
        raise UnknownRoleError(
            f"Rôle Albert inconnu : {role!r} (attendu : {', '.join(ROLES)})."
        )
    return spec


def as_date(today: Any) -> date:
    """Convertit ``today`` (``date``, ``datetime`` ou chaîne ISO) en ``date``.

    Raises:
        TypeError: valeur absente ou d'un type non pris en charge.
        ValueError: chaîne qui n'est pas une date ISO.
    """
    if isinstance(today, datetime):
        return today.date()
    if isinstance(today, date):
        return today
    if isinstance(today, str):
        return date.fromisoformat(today.strip()[:10])
    raise TypeError("today doit être une date (datetime.date), une datetime ou une chaîne ISO.")


def fallback_chain(role: Any, *, today: Any) -> List[str]:
    """Chaîne effective d'un rôle à la date ``today`` (primaire en tête).

    Un modèle quitte la chaîne le jour de son ``deprecated_on`` ; un maillon
    ``since`` n'y entre qu'à sa date ; un modèle exclu n'y figure jamais. Les
    embeddings (``embed``) n'ont jamais de repli.

    Args:
        role: nom du rôle.
        today: date de référence (injectée pour des tests déterministes).

    Returns:
        Les ids épinglés, dans l'ordre d'essai (liste éventuellement vide).

    Raises:
        UnknownRoleError: rôle inconnu.
    """
    spec = role_spec(role)
    day = as_date(today)
    out: List[str] = []
    for entry in spec.chain:
        if entry.since is not None and day < entry.since:
            continue
        model = model_spec(entry.model)
        if model is None or is_excluded(model.id):
            continue
        if model.deprecated_on is not None and day >= model.deprecated_on:
            continue
        if model.id not in out:
            out.append(model.id)
    if spec.name in NO_FALLBACK_ROLES:
        return out[:1]
    return out


def primary_model(role: Any, *, today: Any) -> Optional[str]:
    """Premier modèle de la chaîne effective du rôle (``None`` si la chaîne est vide)."""
    chain = fallback_chain(role, today=today)
    return chain[0] if chain else None


def normalise_endpoint(endpoint: Any) -> str:
    """Nom canonique d'un point d'accès (``chat``, ``embeddings``, ``ocr``, ``rerank``, ``audio``).

    Raises:
        ValueError: point d'accès inconnu.
    """
    key = _clean_name(endpoint).lower()
    if key in _ENDPOINT_ALIASES:
        return _ENDPOINT_ALIASES[key]
    raise ValueError(f"Point d'accès Albert inconnu : {endpoint!r}.")


# ---------------------------------------------------------------------------
# Listes /v1/models
# ---------------------------------------------------------------------------
def _entry_field(entry: Any, name: str) -> Any:
    """Champ d'une entrée de ``/v1/models`` (dict ou objet à attributs)."""
    if isinstance(entry, Mapping):
        return entry.get(name)
    return getattr(entry, name, None)


def listing_entries(listing: Any) -> List[Any]:
    """Entrées d'une réponse ``/v1/models`` (``{"data": [...]}``, liste ou ``None``)."""
    if listing is None:
        return []
    if isinstance(listing, Mapping):
        data = listing.get("data")
        return list(data) if isinstance(data, (list, tuple)) else []
    data = getattr(listing, "data", None)
    if data is not None and not isinstance(listing, (list, tuple)):
        return list(data)
    return list(listing)


def _find_in_listing(requested: str, listing: Any) -> Optional[Any]:
    """Entrée de ``listing`` dont l'id ou un alias vaut ``requested`` (exact, puis sans casse)."""
    entries = listing_entries(listing)
    for exact in (True, False):
        wanted = requested if exact else requested.lower()
        for entry in entries:
            ident = _entry_field(entry, "id")
            aliases = _entry_field(entry, "aliases") or ()
            names = [ident] + list(aliases)
            for name in names:
                if not isinstance(name, str):
                    continue
                if (name if exact else name.lower()) == wanted:
                    return entry
    return None


def resolve_model(requested: Any, listing: Any) -> str:
    """Résout un id ou un alias en id réel d'après la réponse de ``/v1/models``.

    La correspondance exacte (id puis alias) est prioritaire, puis la même
    recherche sans tenir compte de la casse. Aucun repli par type : un modèle
    absent est une erreur explicite.

    Args:
        requested: id ou alias demandé.
        listing: réponse de ``/v1/models`` (``{"data": [...]}``) ou sa liste.

    Returns:
        L'id réel du modèle.

    Raises:
        ValueError: nom vide.
        ModelNotFoundError: modèle absent (``AlbertPermanentError`` de motif
            ``not_found``, aussi ``ValueError`` et ``LookupError``).
    """
    name = _clean_name(requested)
    if not name:
        raise ValueError("Nom de modèle Albert vide.")
    entry = _find_in_listing(name, listing)
    if entry is None:
        raise ModelNotFoundError(
            f"Modèle Albert introuvable : {name!r} n'apparaît pas dans /v1/models "
            "(id épinglé retiré ou accès manquant) ; vérifier la liste des modèles du compte.",
            reason="not_found",
            endpoint="/v1/models",
        )
    ident = _entry_field(entry, "id")
    return str(ident)


def listing_name_map(listing: Any) -> Dict[str, str]:
    """Table nom → id réel de tous les modèles d'une réponse ``/v1/models``.

    Chaque id et chaque alias pointent vers l'id de leur entrée ; les formes
    en minuscules sont ajoutées ensuite sans écraser un nom exact (même
    priorité que ``resolve_model`` : exact, puis sans tenir compte de la casse).

    Args:
        listing: réponse de ``/v1/models`` (``{"data": [...]}``) ou sa liste.

    Returns:
        Le dictionnaire (vide si la liste l'est).
    """
    pairs: List[Tuple[str, str]] = []
    for entry in listing_entries(listing):
        ident = _entry_field(entry, "id")
        if not isinstance(ident, str) or not ident.strip():
            continue
        aliases = _entry_field(entry, "aliases") or ()
        for name in [ident] + list(aliases):
            if isinstance(name, str) and name.strip():
                pairs.append((name.strip(), ident.strip()))
    mapping: Dict[str, str] = {}
    for name, ident in pairs:
        mapping.setdefault(name, ident)
    for name, ident in pairs:
        mapping.setdefault(name.lower(), ident)
    return mapping


def listing_type(model: Any, listing: Any) -> Optional[str]:
    """Type déclaré par ``/v1/models`` pour un id ou un alias (``None`` si absent)."""
    name = _clean_name(model)
    if not name:
        return None
    entry = _find_in_listing(name, listing)
    if entry is None:
        return None
    value = _entry_field(entry, "type")
    return str(value) if value is not None else None


def check_endpoint_type(model: Any, endpoint: Any, *, listing: Any = None) -> Optional[str]:
    """Vérifie, avant l'appel, que le type du modèle convient au point d'accès.

    Le type vient de ``listing`` (réponse de ``/v1/models``) s'il est fourni et
    connaît le modèle, sinon du catalogue statique. Un 422 « Wrong model type »
    est permanent : mieux vaut le détecter sans requête.

    Args:
        model: id ou alias.
        endpoint: ``chat``, ``embeddings``, ``ocr``, ``rerank`` ou le chemin
            ``/v1/...`` correspondant.
        listing: réponse de ``/v1/models`` (optionnelle).

    Returns:
        Le type du modèle s'il est compatible, ``None`` s'il est inconnu (la
        décision revient alors au serveur).

    Raises:
        ValueError: point d'accès inconnu.
        AlbertPermanentError: type incompatible (motif ``validation``).
    """
    kind = normalise_endpoint(endpoint)
    model_type = listing_type(model, listing) if listing is not None else None
    if model_type is None:
        spec = model_spec(model)
        model_type = spec.type if spec is not None else None
    if model_type is None:
        return None
    allowed = ENDPOINT_TYPES[kind]
    if model_type not in allowed:
        raise AlbertPermanentError(
            f"Mauvais type de modèle pour {kind} : {_clean_name(model)!r} est de type "
            f"{model_type} (attendu : {', '.join(allowed)}).",
            reason="validation",
            endpoint=_ENDPOINT_PATHS.get(kind),
        )
    return model_type


def lifecycle_warnings(models: Sequence[str], *, today: Any, margin_days: int = 7) -> List[str]:
    """Avertissements de cycle de vie pour des modèles (retrait ou réexamen proches).

    Args:
        models: ids ou alias.
        today: date de référence.
        margin_days: horizon (jours) des avertissements.

    Returns:
        Messages français, un par échéance comprise dans l'horizon.
    """
    day = as_date(today)
    out: List[str] = []
    seen = set()
    for model in models:
        spec = model_spec(model)
        if spec is None or spec.id in seen:
            continue
        seen.add(spec.id)
        if spec.deprecated_on is not None:
            left = (spec.deprecated_on - day).days
            if 0 <= left <= margin_days:
                out.append(
                    f"Le modèle Albert {spec.id} est retiré le {spec.deprecated_on.isoformat()} "
                    f"(dans {left} jour(s)) : il quittera les chaînes de repli."
                )
        if spec.review_on is not None:
            left = (spec.review_on - day).days
            if -margin_days <= left <= margin_days:
                out.append(
                    f"Statut du modèle Albert {spec.id} à revérifier autour du "
                    f"{spec.review_on.isoformat()} (fin de phase de test annoncée)."
                )
    return out


__all__ = [
    "ENDPOINT_TYPES",
    "EXCLUDED",
    "MODELS",
    "NO_FALLBACK_ROLES",
    "ROLES",
    "ChainEntry",
    "ModelNotFoundError",
    "ModelSpec",
    "RoleSpec",
    "UnknownRoleError",
    "as_date",
    "canonical_id",
    "check_endpoint_type",
    "context_length",
    "embedding_dim",
    "fallback_chain",
    "is_excluded",
    "is_reasoning",
    "lifecycle_warnings",
    "listing_entries",
    "listing_name_map",
    "listing_type",
    "model_spec",
    "normalise_endpoint",
    "primary_model",
    "recommended_temperature",
    "resolve_model",
    "role_spec",
]
