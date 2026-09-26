"""Gardes d'espace vectoriel (lot 6, tâches 4 et 5) : connecteurs, rebuild, clustering, Tier 3.

Un fichier d'embeddings ne se mélange jamais avec une cible d'un autre espace
(invariant 26) :

* **Pinecone** : la dimension est lue sur l'appel ``describe_index`` déjà
  existant, et comparée seulement si c'est un entier ; incompatibilité → erreur
  **avant** la dédup et l'upsert ;
* **Qdrant** : collection existante → ``config.params.vectors.size`` ; nouvelle
  collection → dimension uniforme du fichier, vecteurs nuls ignorés ;
* **Weaviate** : ``fetch_objects(limit=1, include_vector=True)`` **seulement**
  pour un espace hors défaut (aucun appel supplémentaire sur le chemin par défaut) ;
* fichier mélangé : refusé localement, sans aucune lecture ni écriture de la cible ;
* ``rebuild_pinecone_index`` : contrôle préalable, code 2 **avant** ``recreate_index`` ;
* ``rad_clustering`` : message clair sur des dimensions mélangées ; vecteurs nuls
  du fichier historique inchangés (ignorés seulement hors défaut) ;
* dédup : Tier 3 désactivé pour bge-m3 sans ``DEDUP_SIM_THRESHOLD_BGE_M3``
  (un WARNING unique), seuil explicite appliqué sinon (invariant 41).

Les SDK des bases sont remplacés par des doubles en mémoire : aucun réseau.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

_THIS = os.path.dirname(os.path.abspath(__file__))
RAGPY_ROOT = os.path.dirname(_THIS)
for _p in (RAGPY_ROOT, os.path.join(RAGPY_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts import rad_clustering  # noqa: E402
from scripts import rad_dedup  # noqa: E402
from scripts import rad_vectordb as rv  # noqa: E402
from tests.albert_fakes import fake_embedding  # noqa: E402

import rebuild_pinecone_index as rebuild  # noqa: E402  (script CLI, importé par son nom)

ALBERT_DIM = 1024
OPENAI_DIM = 3072
FAKE_PINECONE_KEY = "fake-pinecone-0001"
FAKE_QDRANT_URL = "http://127.0.0.1:6333"
FAKE_WEAVIATE_URL = "https://fake-weaviate.invalid"
FAKE_WEAVIATE_KEY = "fake-weaviate-0001"
SPACE_FIELDS = ("embedding_provider", "embedding_model", "embedding_dim")
CLEARED_PREFIXES = ("DEDUP_", "RECODE_", "ALBERT_")
CLEARED_NAMES = ("PINECONE_API_KEY", "EMBEDDING_PROVIDER", "PINECONE_BATCH_SIZE", "QDRANT_BATCH_SIZE",
                 "WEAVIATE_BATCH_SIZE")


# ---------------------------------------------------------------------------
# Fichiers d'embeddings
# ---------------------------------------------------------------------------
def _text(i, prefix="Passage"):
    """Texte de chunk distinct, assez long pour être éligible à la dédup."""
    return (f"{prefix} {i} : le langage ordinaire structure l'expérience sociale ; "
            f"chaque énoncé renvoie à un contexte d'usage précis et situé.")


def _chunk(i, *, space, doc_id=None, prefix="Passage", dedup=True, embedding="auto"):
    """Chunk d'un fichier ``*_with_embeddings*.json`` dans l'espace ``space`` (``albert`` ou ``openai``).

    L'espace Albert porte les trois champs d'espace ; l'espace OpenAI historique
    n'en porte aucun. ``embedding`` remplace le vecteur calculé (``None``, zéros…).
    """
    text = _text(i, prefix)
    chunk = {
        "id": f"{doc_id or '111111111111'}_{i}",
        "doc_id": doc_id or "111111111111",
        "chunk_index": i,
        "total_chunks": 10,
        "text": text,
        "title": "Le langage ordinaire",
    }
    dim = ALBERT_DIM if space == "albert" else OPENAI_DIM
    chunk["embedding"] = fake_embedding(text, dim) if embedding == "auto" else embedding
    if space == "albert":
        chunk.update({"embedding_provider": "albert", "embedding_model": "bge-m3", "embedding_dim": ALBERT_DIM})
    if dedup:
        chash, eligible, _cid = rad_dedup.compute_dedup_fields(text, i)
        chunk["content_hash"] = chash
        chunk["dedup_eligible"] = eligible
    return chunk


def _albert_chunks(n=3, **kwargs):
    """``n`` chunks bge-m3 (1024 dimensions, champs d'espace)."""
    return [_chunk(i, space="albert", **kwargs) for i in range(1, n + 1)]


def _openai_chunks(n=3, **kwargs):
    """``n`` chunks de l'espace OpenAI historique (3072 dimensions, sans champ d'espace)."""
    return [_chunk(i, space="openai", **kwargs) for i in range(1, n + 1)]


def _mixed_chunks():
    """Fichier mélangé : deux chunks OpenAI (3072) puis deux chunks bge-m3 (1024)."""
    return _openai_chunks(2) + [_chunk(i, space="albert") for i in (3, 4)]


def _write(tmp_path, chunks, name="output_chunks_with_embeddings_sparse.json"):
    """Écrit ``chunks`` en JSON ; renvoie le chemin (str)."""
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False)
    return str(path)


def _message(result):
    """Message d'un résultat de connecteur (dict unifié)."""
    return str((result or {}).get("message", ""))


# ---------------------------------------------------------------------------
# Doubles des SDK
# ---------------------------------------------------------------------------
class FakePineconeIndex:
    """Index Pinecone en mémoire : journalise ``upsert``, ``fetch`` et ``query``."""

    def __init__(self):
        """Crée un index vide."""
        self.upserts = []
        self.fetches = []
        self.queries = []

    def upsert(self, **kwargs):
        """Enregistre un upsert."""
        self.upserts.append(kwargs)

    def fetch(self, **kwargs):
        """Enregistre une lecture par ids (dédup Tier 2) ; aucun vecteur existant."""
        self.fetches.append(kwargs)
        return SimpleNamespace(vectors={})

    def query(self, **kwargs):
        """Enregistre une requête de voisinage (Tier 3) ; aucun voisin."""
        self.queries.append(kwargs)
        return SimpleNamespace(matches=[])


class FakePinecone:
    """Client Pinecone en mémoire : un index nommé, ``describe_index`` configurable."""

    def __init__(self, description, index_name="articles"):
        """``description`` : objet renvoyé (ou exception levée) par ``describe_index``."""
        self.description = description
        self.index_name = index_name
        self.index = FakePineconeIndex()
        self.describe_calls = 0
        self.deleted = []
        self.created = []

    def list_indexes(self):
        """Liste contenant le seul index configuré."""
        return SimpleNamespace(indexes=[SimpleNamespace(name=self.index_name)])

    def Index(self, name):  # noqa: N802 (nom du SDK)
        """Index cible."""
        return self.index

    def describe_index(self, name):
        """Description de l'index (compte les appels) ; lève si c'est une exception."""
        self.describe_calls += 1
        if isinstance(self.description, BaseException):
            raise self.description
        return self.description

    def delete_index(self, name):
        """Enregistre une suppression d'index."""
        self.deleted.append(name)

    def create_index(self, **kwargs):
        """Enregistre une création d'index."""
        self.created.append(kwargs)


def _install_pinecone(monkeypatch, description):
    """Remplace ``rv.Pinecone`` par un double ; renvoie ce double."""
    fake = FakePinecone(description)
    monkeypatch.setattr(rv, "Pinecone", lambda *a, **k: fake)
    return fake


class FakeQdrant:
    """Client Qdrant en mémoire : collection existante (taille configurable) ou absente."""

    def __init__(self, *, existing_vectors=None, neighbours=None):
        """``existing_vectors`` : ``params.vectors`` de la collection (``None`` = absente) ;
        ``neighbours`` : points renvoyés par ``query_points`` (Tier 3)."""
        self.existing_vectors = existing_vectors
        self.neighbours = list(neighbours or [])
        self.get_collection_calls = 0
        self.created = []
        self.upserts = []
        self.retrieves = []
        self.queries = []

    def get_collections(self):
        """Liste des collections (contrôle de connexion)."""
        return SimpleNamespace(collections=[])

    def get_collection(self, collection_name=None, **kwargs):
        """Description de la collection, ou exception si elle est absente."""
        self.get_collection_calls += 1
        if self.existing_vectors is None:
            raise RuntimeError(f"Collection {collection_name} not found")
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=self.existing_vectors)))

    def create_collection(self, **kwargs):
        """Enregistre une création de collection."""
        self.created.append(kwargs)

    def retrieve(self, **kwargs):
        """Enregistre une lecture par ids (dédup Tier 2) ; aucun point existant."""
        self.retrieves.append(kwargs)
        return []

    def query_points(self, **kwargs):
        """Enregistre une requête de voisinage (Tier 3) ; renvoie les voisins configurés."""
        self.queries.append(kwargs)
        return SimpleNamespace(points=list(self.neighbours))

    def upsert(self, **kwargs):
        """Enregistre un upsert terminé."""
        self.upserts.append(kwargs)
        return SimpleNamespace(status=rv.models.UpdateStatus.COMPLETED)

    def close(self):
        """Fermeture (sans effet)."""
        return None


def _install_qdrant(monkeypatch, fake):
    """Remplace ``QdrantClient`` (vu par ``rad_vectordb``) par une fabrique renvoyant ``fake``."""
    monkeypatch.setattr(rv.qdrant_client, "QdrantClient", lambda *a, **k: fake)
    return fake


def _vectors_params(size):
    """``params.vectors`` d'une collection Qdrant à vecteur unique de taille ``size``."""
    return SimpleNamespace(size=size, distance="Cosine")


class FakeWeaviate:
    """Module ``weaviate`` et collection multi-tenant simulés (``MagicMock``)."""

    def __init__(self, target_vectors=()):
        """``target_vectors`` : vecteurs des objets déjà présents dans le tenant."""
        self.fetch_calls = []
        self.target_objects = [
            SimpleNamespace(uuid=uuid.uuid4(), properties={}, metadata=None, vector={"default": list(vec)})
            for vec in target_vectors
        ]
        self.module = MagicMock(name="weaviate")
        self.client = MagicMock(name="client")
        self.module.connect_to_weaviate_cloud.return_value = self.client
        self.client.is_ready.return_value = True
        self.collection = MagicMock(name="collection")
        self.client.collections.get.return_value = self.collection
        self.collection.tenants.get.return_value = {"alakel": object()}
        self.tenant = MagicMock(name="tenant_collection")
        self.collection.with_tenant.return_value = self.tenant
        self.tenant.query.fetch_objects.side_effect = self._fetch_objects
        self.tenant.data.insert_many.return_value = SimpleNamespace(has_errors=False, errors={})

    def _fetch_objects(self, *args, **kwargs):
        """Enregistre ``fetch_objects`` et renvoie les objets du tenant (limite respectée)."""
        self.fetch_calls.append(kwargs)
        limit = kwargs.get("limit")
        objects = self.target_objects[:limit] if isinstance(limit, int) else list(self.target_objects)
        return SimpleNamespace(objects=objects)

    @property
    def inserted(self):
        """Nombre d'appels ``insert_many``."""
        return self.tenant.data.insert_many.call_count

    @property
    def untenanted_fetches(self):
        """Appels ``fetch_objects`` faits sans tenant (jamais attendus)."""
        return self.collection.query.fetch_objects.call_count


def _install_weaviate(monkeypatch, fake):
    """Remplace le module ``weaviate`` vu par ``rad_vectordb`` ; renvoie le double."""
    monkeypatch.setattr(rv, "weaviate", fake.module)
    return fake


def _insert_weaviate(path):
    """Insertion Weaviate de ``path`` dans ``Article``/``alakel``."""
    return rv.insert_to_weaviate_hybrid(path, FAKE_WEAVIATE_URL, FAKE_WEAVIATE_KEY,
                                        class_name="Article", tenant_name="alakel")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Dédup, cache et Albert désactivés ; aucune clé ni réglage de lot hérité."""
    for name in list(os.environ):
        if name.startswith(CLEARED_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    for name in CLEARED_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(rv.time, "sleep", lambda *_a, **_k: None)
    yield


def _set_bge_threshold(monkeypatch, value):
    """Pose ``DEDUP_SIM_THRESHOLD_BGE_M3`` (environnement et éventuelle constante de module)."""
    if value is None:
        monkeypatch.delenv("DEDUP_SIM_THRESHOLD_BGE_M3", raising=False)
    else:
        monkeypatch.setenv("DEDUP_SIM_THRESHOLD_BGE_M3", value)
    if hasattr(rv, "DEDUP_SIM_THRESHOLD_BGE_M3"):
        current = getattr(rv, "DEDUP_SIM_THRESHOLD_BGE_M3")
        if isinstance(current, str):
            converted = value or ""
        else:
            converted = float(value) if value not in (None, "") else None
        monkeypatch.setattr(rv, "DEDUP_SIM_THRESHOLD_BGE_M3", converted)


# ---------------------------------------------------------------------------
# Pinecone
# ---------------------------------------------------------------------------
PINECONE_MISMATCH = [
    ("albert_file_openai_index", "albert", OPENAI_DIM),
    ("openai_file_albert_index", "openai", ALBERT_DIM),
]


@pytest.mark.parametrize("case,space,index_dim", PINECONE_MISMATCH, ids=[c[0] for c in PINECONE_MISMATCH])
def test_pinecone_dim_mismatch_before_upsert(case, space, index_dim, monkeypatch, tmp_path):
    monkeypatch.setenv("DEDUP_ENABLED", "1")
    fake = _install_pinecone(monkeypatch, SimpleNamespace(metric="dotproduct", dimension=index_dim))
    chunks = _albert_chunks() if space == "albert" else _openai_chunks()
    result = rv.insert_to_pinecone(_write(tmp_path, chunks), index_name="articles",
                                   pinecone_api_key=FAKE_PINECONE_KEY)
    assert result["status"] == "error"
    assert result["inserted_count"] == 0
    assert str(ALBERT_DIM) in _message(result) and str(OPENAI_DIM) in _message(result)
    assert fake.index.upserts == []
    assert fake.index.fetches == []  # refus AVANT la dédup
    assert fake.describe_calls == 1  # dimension lue sur l'appel existant, rien de plus
    # Contrôle positif : même fichier vers un index de la bonne dimension.
    good_dim = ALBERT_DIM if space == "albert" else OPENAI_DIM
    ok = _install_pinecone(monkeypatch, SimpleNamespace(metric="dotproduct", dimension=good_dim))
    result = rv.insert_to_pinecone(_write(tmp_path, chunks, "ok.json"), index_name="articles",
                                   pinecone_api_key=FAKE_PINECONE_KEY)
    assert result["status"] == "success"
    assert result["inserted_count"] == len(chunks)
    assert ok.index.upserts
    assert ok.describe_calls == 1


PINECONE_NON_INT = [
    ("magicmock", "magicmock"),
    ("dimension_none", SimpleNamespace(metric="dotproduct", dimension=None)),
    ("dimension_str", SimpleNamespace(metric="dotproduct", dimension="3072")),
    ("dimension_float", SimpleNamespace(metric="dotproduct", dimension=3072.0)),
    ("dimension_bool", SimpleNamespace(metric="dotproduct", dimension=True)),
    ("no_dimension_attr", SimpleNamespace(metric="cosine")),
    ("describe_fails", RuntimeError("describe_index indisponible")),
]


@pytest.mark.parametrize("case,description", PINECONE_NON_INT, ids=[c[0] for c in PINECONE_NON_INT])
def test_pinecone_non_int_dimension_proceeds(case, description, monkeypatch, tmp_path):
    if description == "magicmock":
        description = MagicMock(name="IndexDescription")  # forme des mocks existants
    fake = _install_pinecone(monkeypatch, description)
    chunks = _albert_chunks()
    result = rv.insert_to_pinecone(_write(tmp_path, chunks), index_name="articles",
                                   pinecone_api_key=FAKE_PINECONE_KEY)
    assert result["status"] == "success"
    assert result["inserted_count"] == len(chunks)
    assert sum(len(u["vectors"]) for u in fake.index.upserts) == len(chunks)
    assert fake.describe_calls == 1


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------
def test_qdrant_existing_size_mismatch(monkeypatch, tmp_path):
    monkeypatch.setenv("DEDUP_ENABLED", "1")
    fake = _install_qdrant(monkeypatch, FakeQdrant(existing_vectors=_vectors_params(OPENAI_DIM)))
    result = rv.insert_to_qdrant(_write(tmp_path, _albert_chunks()), "corpus", qdrant_url=FAKE_QDRANT_URL)
    assert result["status"] == "error"
    assert result["inserted_count"] == 0
    assert str(ALBERT_DIM) in _message(result) and str(OPENAI_DIM) in _message(result)
    assert fake.upserts == [] and fake.created == [] and fake.retrieves == []
    assert fake.get_collection_calls == 1
    # Collection existante compatible : insertion normale.
    ok = _install_qdrant(monkeypatch, FakeQdrant(existing_vectors=_vectors_params(ALBERT_DIM)))
    result = rv.insert_to_qdrant(_write(tmp_path, _albert_chunks(), "ok.json"), "corpus",
                                 qdrant_url=FAKE_QDRANT_URL)
    assert result["status"] == "success" and result["inserted_count"] == 3
    assert ok.upserts and ok.created == []
    # Vecteurs nommés (pas de ``size`` unique) : aucune garde, insertion normale.
    named = _install_qdrant(monkeypatch, FakeQdrant(existing_vectors={"dense": _vectors_params(OPENAI_DIM)}))
    result = rv.insert_to_qdrant(_write(tmp_path, _albert_chunks(), "named.json"), "corpus",
                                 qdrant_url=FAKE_QDRANT_URL)
    assert result["status"] == "success"
    assert named.upserts


@pytest.mark.parametrize("first", ["zero_vector", "none", "empty_list"])
def test_qdrant_new_collection_uniform_dim(first, monkeypatch, tmp_path):
    fake = _install_qdrant(monkeypatch, FakeQdrant(existing_vectors=None))
    lead = {"zero_vector": [0.0] * OPENAI_DIM, "none": None, "empty_list": []}[first]
    chunks = [_chunk(1, space="openai", embedding=lead)] + [_chunk(i, space="albert") for i in (2, 3)]
    rv.insert_to_qdrant(_write(tmp_path, chunks), "nouvelle", qdrant_url=FAKE_QDRANT_URL)
    assert len(fake.created) == 1
    params = fake.created[0]["vectors_config"]
    assert params.size == ALBERT_DIM  # dimension uniforme du fichier, vecteurs nuls ignorés
    # Fichier mélangé : aucune collection créée.
    mixed = _install_qdrant(monkeypatch, FakeQdrant(existing_vectors=None))
    result = rv.insert_to_qdrant(_write(tmp_path, _mixed_chunks(), "mixed.json"), "nouvelle",
                                 qdrant_url=FAKE_QDRANT_URL)
    assert result["status"] == "error"
    assert mixed.created == [] and mixed.upserts == []


# ---------------------------------------------------------------------------
# Weaviate
# ---------------------------------------------------------------------------
def test_weaviate_fetch_only_non_default(monkeypatch, tmp_path):
    # Espace par défaut : aucune lecture supplémentaire de la cible.
    default = _install_weaviate(monkeypatch, FakeWeaviate(target_vectors=[[0.1] * ALBERT_DIM]))
    result = _insert_weaviate(_write(tmp_path, _openai_chunks(), "default.json"))
    assert result["status"] == "success" and result["inserted_count"] == 3
    assert default.fetch_calls == []
    assert default.untenanted_fetches == 0
    # Espace bge-m3, tenant vide : une seule lecture (limit=1, include_vector=True), puis insertion.
    empty = _install_weaviate(monkeypatch, FakeWeaviate(target_vectors=()))
    result = _insert_weaviate(_write(tmp_path, _albert_chunks(), "albert_empty.json"))
    assert result["status"] == "success" and result["inserted_count"] == 3
    assert len(empty.fetch_calls) == 1
    assert empty.fetch_calls[0].get("limit") == 1
    assert empty.fetch_calls[0].get("include_vector") is True
    assert empty.untenanted_fetches == 0
    # Espace bge-m3, tenant en 1024 dimensions : compatible.
    same = _install_weaviate(monkeypatch, FakeWeaviate(target_vectors=[fake_embedding("existant", ALBERT_DIM)]))
    result = _insert_weaviate(_write(tmp_path, _albert_chunks(), "albert_same.json"))
    assert result["status"] == "success"
    assert same.inserted >= 1
    # Espace bge-m3, tenant en 3072 dimensions : refus, rien n'est inséré.
    other = _install_weaviate(monkeypatch, FakeWeaviate(target_vectors=[fake_embedding("existant", OPENAI_DIM)]))
    result = _insert_weaviate(_write(tmp_path, _albert_chunks(), "albert_other.json"))
    assert result["status"] == "error"
    assert result["inserted_count"] == 0
    assert str(ALBERT_DIM) in _message(result) and str(OPENAI_DIM) in _message(result)
    assert other.inserted == 0
    assert len(other.fetch_calls) == 1


# ---------------------------------------------------------------------------
# Fichier mélangé : refus local, sans lecture ni écriture de la cible
# ---------------------------------------------------------------------------
def _mixed_pinecone(monkeypatch, path):
    """Insertion Pinecone du fichier mélangé ; renvoie (résultat, traces de la cible)."""
    fake = _install_pinecone(monkeypatch, SimpleNamespace(metric="dotproduct", dimension=OPENAI_DIM))
    result = rv.insert_to_pinecone(path, index_name="articles", pinecone_api_key=FAKE_PINECONE_KEY)
    touched = fake.index.upserts + fake.index.fetches + fake.index.queries
    assert fake.describe_calls <= 1
    return result, touched


def _mixed_qdrant(monkeypatch, path):
    """Insertion Qdrant du fichier mélangé ; renvoie (résultat, traces de la cible)."""
    fake = _install_qdrant(monkeypatch, FakeQdrant(existing_vectors=_vectors_params(OPENAI_DIM)))
    result = rv.insert_to_qdrant(path, "corpus", qdrant_url=FAKE_QDRANT_URL)
    return result, fake.upserts + fake.created + fake.retrieves + fake.queries


def _mixed_weaviate(monkeypatch, path):
    """Insertion Weaviate du fichier mélangé ; renvoie (résultat, traces de la cible)."""
    fake = _install_weaviate(monkeypatch, FakeWeaviate(target_vectors=[fake_embedding("x", OPENAI_DIM)]))
    result = _insert_weaviate(path)
    touched = list(fake.fetch_calls) + [None] * (fake.inserted + fake.untenanted_fetches)
    return result, touched


@pytest.mark.parametrize("connector", [_mixed_pinecone, _mixed_qdrant, _mixed_weaviate],
                         ids=["pinecone", "qdrant", "weaviate"])
def test_mixed_file_refused_offline(connector, monkeypatch, tmp_path):
    monkeypatch.setenv("DEDUP_ENABLED", "1")
    result, touched = connector(monkeypatch, _write(tmp_path, _mixed_chunks()))
    assert result["status"] == "error"
    assert result["inserted_count"] == 0
    assert "mélang" in _message(result).lower()
    assert touched == []


def test_mixed_file_check_is_local():
    """La garde de fichier mélangé ne dépend d'aucune cible (``check_uniform_space`` pur)."""
    from scripts.rad_providers import check_uniform_space

    with pytest.raises(ValueError) as excinfo:
        check_uniform_space(_mixed_chunks())
    assert "mélang" in str(excinfo.value).lower()
    legacy_with_zero = _openai_chunks(2) + [_chunk(3, space="openai", embedding=[0.0] * ALBERT_DIM)]
    assert check_uniform_space(legacy_with_zero).dim == OPENAI_DIM


# ---------------------------------------------------------------------------
# rebuild_pinecone_index : contrôle préalable avant recreate_index
# ---------------------------------------------------------------------------
@pytest.fixture
def rebuild_env(monkeypatch):
    """``rebuild_pinecone_index`` isolé : Pinecone, ``recreate_index`` et l'upload sont des doubles."""
    monkeypatch.setenv("PINECONE_API_KEY", FAKE_PINECONE_KEY)
    fake_pc = FakePinecone(SimpleNamespace(metric="dotproduct", dimension=OPENAI_DIM,
                                           status={"ready": True}))
    monkeypatch.setattr(rebuild, "Pinecone", lambda *a, **k: fake_pc)
    recreated = []
    uploads = []

    def fake_recreate(pc, name, dimension, metric, cloud, region):
        """Enregistre la recréation demandée (aucune suppression réelle)."""
        recreated.append({"name": name, "dimension": dimension, "metric": metric})

    def fake_insert(**kwargs):
        """Enregistre l'upload demandé et le déclare réussi."""
        uploads.append(kwargs)
        return {"status": "success", "message": "ok", "inserted_count": 1}

    monkeypatch.setattr(rebuild, "recreate_index", fake_recreate)
    monkeypatch.setattr(rebuild, "insert_to_pinecone", fake_insert)
    return SimpleNamespace(pc=fake_pc, recreated=recreated, uploads=uploads)


def _rebuild_main(monkeypatch, args):
    """Lance ``rebuild_pinecone_index.main()`` avec ``args`` ; renvoie son code."""
    monkeypatch.setattr(sys, "argv", ["rebuild_pinecone_index.py"] + [str(a) for a in args])
    try:
        return rebuild.main()
    except SystemExit as exc:
        return 0 if exc.code is None else exc.code


def _session_file(tmp_path, name, chunks):
    """Écrit ``chunks`` dans ``<tmp>/uploads/<name>/output_chunks_with_embeddings_sparse.json``."""
    session_dir = tmp_path / "uploads" / name
    session_dir.mkdir(parents=True)
    return _write(session_dir, chunks)


@pytest.mark.parametrize("case", ["albert_file", "mixed_sessions"])
def test_rebuild_preflight_exit_2(case, rebuild_env, monkeypatch, tmp_path, capsys):
    albert_path = _session_file(tmp_path, "session_albert", _albert_chunks())
    sessions = [albert_path]
    if case == "mixed_sessions":
        sessions.insert(0, _session_file(tmp_path, "session_openai", _openai_chunks()))
    code = _rebuild_main(monkeypatch, ["--index", "articles", "--recreate", "--dimension", OPENAI_DIM,
                                       "--sessions", *sessions])
    out = capsys.readouterr().out
    assert code == 2
    assert rebuild_env.recreated == []  # contrôle AVANT toute recréation destructive
    assert rebuild_env.pc.deleted == [] and rebuild_env.pc.created == []
    assert rebuild_env.uploads == []
    assert str(ALBERT_DIM) in out
    # Contrôle positif : dimension demandée = dimension du fichier.
    code = _rebuild_main(monkeypatch, ["--index", "articles", "--recreate", "--dimension", ALBERT_DIM,
                                       "--sessions", albert_path])
    assert code == 0
    assert [r["dimension"] for r in rebuild_env.recreated] == [ALBERT_DIM]
    assert len(rebuild_env.uploads) == 1


def test_rebuild_existing_index_dimension_checked(rebuild_env, monkeypatch, tmp_path, capsys):
    """Sans ``--recreate`` : dimension de l'index existant (``describe_index``) contrôlée avant l'upload."""
    albert_path = _session_file(tmp_path, "session_albert", _albert_chunks())
    code = _rebuild_main(monkeypatch, ["--index", "articles", "--sessions", albert_path])
    assert code == 2
    assert rebuild_env.uploads == [] and rebuild_env.recreated == []
    assert str(ALBERT_DIM) in capsys.readouterr().out
    openai_path = _session_file(tmp_path, "session_openai", _openai_chunks())
    assert _rebuild_main(monkeypatch, ["--index", "articles", "--sessions", openai_path]) == 0
    assert len(rebuild_env.uploads) == 1


# ---------------------------------------------------------------------------
# rad_clustering
# ---------------------------------------------------------------------------
def _cluster_chunk(doc_id, idx, vector, *, space):
    """Chunk minimal pour le clustering (document, index, vecteur, champs d'espace si Albert)."""
    chunk = {"doc_id": doc_id, "chunk_index": idx, "embedding": vector, "title": f"Titre {doc_id}",
             "authors": "Dupont, Jeanne", "itemKey": f"KEY{doc_id}"}
    if space == "albert":
        chunk.update({"embedding_provider": "albert", "embedding_model": "bge-m3", "embedding_dim": len(vector)})
    return chunk


@pytest.fixture
def no_umap(monkeypatch):
    """UMAP et HDBSCAN remplacés : enregistrent la matrice reçue, jamais de calcul lourd."""
    seen = []

    def fake_umap(embeddings, n_components=2, **kwargs):
        """Enregistre la matrice des documents et renvoie une projection nulle."""
        seen.append(np.array(embeddings))
        return np.zeros((embeddings.shape[0], max(1, n_components)), dtype=np.float32)

    def fake_hdbscan(embeddings, min_cluster_size=2, **kwargs):
        """Tous les documents en bruit (-1)."""
        return np.full(embeddings.shape[0], -1)

    monkeypatch.setattr(rad_clustering, "reduce_dimensions_umap", fake_umap)
    monkeypatch.setattr(rad_clustering, "cluster_documents_hdbscan", fake_hdbscan)
    return seen


@pytest.mark.parametrize("layout", ["across_documents", "within_document"])
def test_clustering_mixed_dims_clear_error(layout, no_umap, tmp_path):
    openai_vec = lambda seed: fake_embedding(f"openai-{seed}", OPENAI_DIM)  # noqa: E731
    albert_vec = lambda seed: fake_embedding(f"albert-{seed}", ALBERT_DIM)  # noqa: E731
    chunks = [
        _cluster_chunk("doc1", 0, openai_vec(1), space="openai"),
        _cluster_chunk("doc2", 0, openai_vec(2), space="openai"),
    ]
    if layout == "across_documents":
        chunks.append(_cluster_chunk("doc3", 0, albert_vec(3), space="albert"))
    else:
        chunks.append(_cluster_chunk("doc3", 0, openai_vec(3), space="openai"))
        chunks.append(_cluster_chunk("doc1", 1, albert_vec(1), space="albert"))
    path = _write(tmp_path, chunks)
    with pytest.raises(ValueError) as excinfo:
        rad_clustering.run_clustering_pipeline(path, str(tmp_path), "Test")
    message = str(excinfo.value)
    # Message explicite (et non l'erreur brute de numpy) nommant les deux dimensions.
    assert "concatenation axis" not in message
    assert "input array dimensions" not in message
    assert str(ALBERT_DIM) in message and str(OPENAI_DIM) in message
    assert no_umap == []


def test_clustering_legacy_zero_unchanged(no_umap, tmp_path):
    v1 = fake_embedding("legacy-1", OPENAI_DIM)
    v2 = fake_embedding("legacy-2", OPENAI_DIM)
    v3 = fake_embedding("legacy-3", OPENAI_DIM)
    zeros = [0.0] * OPENAI_DIM
    chunks = [
        _cluster_chunk("doc1", 0, v1, space="openai"),
        _cluster_chunk("doc1", 1, zeros, space="openai"),  # repli historique : pris en compte
        _cluster_chunk("doc2", 0, v2, space="openai"),
        _cluster_chunk("doc3", 0, v3, space="openai"),
        _cluster_chunk("doc3", 1, None, space="openai"),  # None : ignoré, comme avant
    ]
    doc_embeddings = rad_clustering.aggregate_chunk_embeddings(chunks, method="mean")
    expected_doc1 = np.mean(np.vstack([np.array(v1, dtype=np.float32), np.array(zeros, dtype=np.float32)]), axis=0)
    assert np.array_equal(doc_embeddings["doc1"], expected_doc1)
    assert np.array_equal(doc_embeddings["doc2"], np.array(v2, dtype=np.float32))
    assert np.array_equal(doc_embeddings["doc3"], np.array(v3, dtype=np.float32))
    # Pipeline complet : les zéros historiques ne déclenchent pas la garde des dimensions.
    results = rad_clustering.run_clustering_pipeline(_write(tmp_path, chunks), str(tmp_path), "Test")
    assert results["n_documents"] == 3
    assert len(no_umap) == 1
    assert no_umap[0].shape == (3, OPENAI_DIM)
    assert np.allclose(no_umap[0][0], expected_doc1)  # doc1 = première ligne (ordre d'apparition)


def test_clustering_non_default_zero_ignored():
    """Hors défaut (bge-m3), un vecteur nul n'entre pas dans la moyenne du document."""
    a1 = fake_embedding("albert-doc1", ALBERT_DIM)
    chunks = [
        _cluster_chunk("doc1", 0, a1, space="albert"),
        _cluster_chunk("doc1", 1, [0.0] * ALBERT_DIM, space="albert"),
        _cluster_chunk("doc2", 0, fake_embedding("albert-doc2", ALBERT_DIM), space="albert"),
    ]
    doc_embeddings = rad_clustering.aggregate_chunk_embeddings(chunks, method="mean")
    assert np.allclose(doc_embeddings["doc1"], np.array(a1, dtype=np.float32))


# ---------------------------------------------------------------------------
# Dédup Tier 3 : bge-m3 sans seuil calibré (invariant 41)
# ---------------------------------------------------------------------------
def _neighbour(chunk, score):
    """Point Qdrant voisin : mêmes titre, index et texte que ``chunk``, similarité ``score``."""
    payload = {"title": chunk["title"], "chunk_index": chunk["chunk_index"], "text": chunk["text"],
               "content_hash": "0" * 64, "id": "voisin-existant"}
    return SimpleNamespace(id=str(uuid.uuid4()), score=score, payload=payload)


def _tier3_env(monkeypatch, *, general="0.97"):
    """Dédup ON avec Tier 3 (sémantique) actif ; seuil général ``DEDUP_SIM_THRESHOLD``."""
    monkeypatch.setenv("DEDUP_ENABLED", "1")
    monkeypatch.setenv("DEDUP_SEMANTIC", "1")
    monkeypatch.setenv("DEDUP_SIM_THRESHOLD", general)


def _bge_warnings(caplog):
    """Enregistrements WARNING qui annoncent le Tier 3 sauté pour bge-m3."""
    return [r for r in caplog.records
            if r.levelno >= logging.WARNING
            and ("bge" in r.getMessage().lower() or "DEDUP_SIM_THRESHOLD_BGE_M3" in r.getMessage())]


def _journal_thresholds(tmp_path):
    """Seuils ``threshold_used`` des refus ``semantic_match`` du journal de dédup."""
    path = rad_dedup.journal_path_for(str(tmp_path))
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return [r["threshold_used"] for r in rows if r.get("reason") == "semantic_match"]


def test_tier3_disabled_for_bge_m3_by_default(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    _tier3_env(monkeypatch)
    _set_bge_threshold(monkeypatch, None)
    chunks = _albert_chunks(2)
    fake = _install_qdrant(monkeypatch, FakeQdrant(existing_vectors=_vectors_params(ALBERT_DIM),
                                                   neighbours=[_neighbour(chunks[0], 0.999)]))
    result = rv.insert_to_qdrant(_write(tmp_path, chunks), "corpus", qdrant_url=FAKE_QDRANT_URL)
    assert result["status"] == "success"
    assert result["inserted_count"] == 2
    assert result.get("skipped_count", 0) == 0
    assert fake.queries == []  # aucune requête de voisinage
    assert fake.retrieves  # Tiers 1-2 inchangés
    assert len(_bge_warnings(caplog)) == 1
    # Seuil vide explicite : même effet.
    caplog.clear()
    _set_bge_threshold(monkeypatch, "")
    fake = _install_qdrant(monkeypatch, FakeQdrant(existing_vectors=_vectors_params(ALBERT_DIM),
                                                   neighbours=[_neighbour(chunks[0], 0.999)]))
    rv.insert_to_qdrant(_write(tmp_path, chunks, "again.json"), "corpus", qdrant_url=FAKE_QDRANT_URL)
    assert fake.queries == []
    assert len(_bge_warnings(caplog)) == 1
    # Fichier de l'espace par défaut : Tier 3 inchangé (seuil général 0.97).
    caplog.clear()
    legacy = _openai_chunks(2)
    fake = _install_qdrant(monkeypatch, FakeQdrant(existing_vectors=_vectors_params(OPENAI_DIM),
                                                   neighbours=[_neighbour(legacy[0], 0.99)]))
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    result = rv.insert_to_qdrant(_write(legacy_dir, legacy), "corpus", qdrant_url=FAKE_QDRANT_URL)
    assert fake.queries
    assert result.get("skipped_count", 0) == 1
    assert _journal_thresholds(legacy_dir) == [0.97]
    assert _bge_warnings(caplog) == []


@pytest.mark.parametrize("bge_threshold,score,general,expected_skipped", [
    ("0.9", 0.93, "0.97", 1),   # seuil bge-m3 plus bas que le général : appliqué
    ("0.95", 0.93, "0.5", 0),   # seuil général très bas ignoré : c'est le seuil bge-m3 qui compte
], ids=["applied_below_general", "general_not_used"])
def test_tier3_bge_m3_explicit_threshold(bge_threshold, score, general, expected_skipped, monkeypatch, tmp_path,
                                         caplog):
    caplog.set_level(logging.WARNING)
    _tier3_env(monkeypatch, general=general)
    _set_bge_threshold(monkeypatch, bge_threshold)
    chunks = _albert_chunks(2)
    fake = _install_qdrant(monkeypatch, FakeQdrant(existing_vectors=_vectors_params(ALBERT_DIM),
                                                   neighbours=[_neighbour(chunks[0], score)]))
    result = rv.insert_to_qdrant(_write(tmp_path, chunks), "corpus", qdrant_url=FAKE_QDRANT_URL)
    assert fake.queries, "Tier 3 attendu avec un seuil bge-m3 explicite"
    assert result.get("skipped_count", 0) == expected_skipped
    if expected_skipped:
        assert _journal_thresholds(tmp_path) == [float(bge_threshold)]
    assert _bge_warnings(caplog) == []
