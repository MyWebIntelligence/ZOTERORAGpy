"""Collections Albert, 4e cible vectorielle (lot 5) : tests hors ligne sur ``FakeAlbert``.

Contrats vérifiés (``SPRINT_albert.md``, lot 5, invariants 27 à 29) :

* métadonnées en liste blanche (10 champs au plus, ``content_id``,
  ``content_hash``, ``chunk_index`` et ``DEDUP_META_FIELDS`` imposés, jamais
  ``path``) et scalaires conformes à l'API (D16) ;
* idempotence toujours active par ``content_id`` (même sans champs de dédup),
  noms de documents stables ;
* interrupteur maître : ``ALBERT_ENABLED`` coupé, la bibliothèque refuse
  avant tout (ni client construit, ni appel, ni manifeste) ;
* collection privée forcée, correspondance exacte du nom côté client,
  homonymes refusés en listant les ids (D14) ; une visibilité absente ou vide
  compte comme non privée (échec fermé) ;
* document en multipart sans fichier ni métadonnées (D15), chunks par
  tranches de 64, aucun vecteur envoyé ;
* créations non idempotentes : une issue incertaine est rapprochée par une
  nouvelle liste, jamais renvoyée à l'aveugle (D14, D16, D18) ;
* rollback par document, manifeste écrit tranche par tranche avec les ids de
  chunks renvoyés (D16), suivi d'une ligne d'événement ``rollback`` ;
* dédup Tier 2 en mémoire, corroborée aussi par les champs dérivés de la
  liste blanche (``item_key``, ``year``…) nommés dans ``DEDUP_META_FIELDS`` ;
* envois sous le limiteur du rôle ``embed`` (D19) ;
* bloc ``=== Result ===`` inchangé, lignes Albert propres à ``--db albert``,
  acquittement de rétention exigé par la CLI.

Aucun appel réseau : le client Albert parle à ``FakeAlbert`` par un
``httpx.MockTransport`` (directement ou par un transport scripté qui injecte
des incidents).
"""

from __future__ import annotations

import fnmatch
import json
import math
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from scripts import rad_dedup
from scripts import rad_vectordb as rv
from scripts.rad_albert import client as albert_client_mod
from scripts.rad_albert import collections as col
from scripts.rad_albert import limiter as albert_limiter
from scripts.rad_albert import preflight as albert_preflight
from scripts.rad_albert.config import DEFAULT_METADATA_FIELDS, AlbertConfig
from scripts.rad_albert.client import AlbertInputError
from scripts.rad_albert.usage import USAGE_FILENAME
from tests.albert_fakes import FAKE_ALBERT_KEY, FakeAlbert, load_fixture
import tests.test_albert_off_golden as golden_harness

RAGPY_ROOT = Path(__file__).resolve().parents[1]
VECTORDB_SCRIPT = str(RAGPY_ROOT / "scripts" / "rad_vectordb.py")
COLLECTION = "ragpy-test-corpus"
LONG_TITLE = ("Une histoire sociale des infrastructures numériques de la recherche publique " * 5).strip()
CHUNK_POSTS = "/v1/documents/*/chunks"
DECISIONS = load_fixture("decisions.json")


# ---------------------------------------------------------------------------
# Outils
# ---------------------------------------------------------------------------
def _no_sleep(_seconds=0.0):
    """Sommeil neutralisé (aucune attente réelle)."""
    return None


class SleepRecorder:
    """Sommeil factice : enregistre les durées demandées sans attendre."""

    def __init__(self):
        """Crée un enregistreur vide."""
        self.calls = []

    def __call__(self, seconds):
        """Enregistre une demande de sommeil."""
        self.calls.append(float(seconds))


class RecordingLimiter:
    """Limiteur espion : journalise chaque acquisition puis délègue au limiteur réel."""

    def __init__(self, inner, role, journal):
        """Enveloppe ``inner`` pour le rôle ``role`` ; les acquisitions vont dans ``journal``."""
        self._inner = inner
        self._role = role
        self._journal = journal

    def acquire(self, tokens=0, requests=1, **kwargs):
        """Enregistre (rôle, jetons, requêtes) puis acquiert sur le limiteur réel."""
        self._journal.append({"role": self._role, "tokens": tokens, "requests": requests})
        if requests != 1:
            return self._inner.acquire(tokens, requests=requests, **kwargs)
        return self._inner.acquire(tokens, **kwargs)

    def __getattr__(self, name):
        """Délègue les autres attributs (``pause``…) au limiteur réel."""
        return getattr(self._inner, name)


class ScriptedTransport:
    """``httpx.MockTransport`` placé devant ``FakeAlbert`` : incidents sur la n-ième requête.

    Actions possibles pour une règle :

    * ``"lost_response"`` : le faux serveur traite la requête, puis la réponse
      est perdue (``httpx.ReadTimeout``) : issue incertaine côté client ;
    * ``"timeout_unprocessed"`` : délai de lecture sans que le serveur ait
      traité la requête (issue tout aussi incertaine pour le client) ;
    * ``"drop_ids"`` : réponse d'envoi de chunks privée de son champ ``ids`` ;
    * toute autre valeur : entrée d'injection de ``FakeAlbert.inject``
      (statut, nom d'erreur, dictionnaire…), consommée par cette requête.

    ``seen`` ne contient que les requêtes émises par le client (pas celles
    qu'un crochet envoie lui-même au faux serveur).
    """

    def __init__(self, fake):
        """Branche le transport sur ``fake``."""
        self.fake = fake
        self.seen = []
        self._rules = []
        self.transport = httpx.MockTransport(self._handle)

    def on(self, method, pattern, nth, action, hook=None):
        """Programme ``action`` sur la ``nth``-ième requête ``method`` dont le chemin correspond à ``pattern``."""
        self._rules.append({"method": method.upper(), "pattern": pattern, "nth": nth,
                            "action": action, "hook": hook, "count": 0})
        return self

    def count(self, method, pattern):
        """Nombre de requêtes du client correspondant à ``method`` et ``pattern``."""
        return sum(1 for m, p in self.seen if m == method.upper() and fnmatch.fnmatchcase(p, pattern))

    def _handle(self, request):
        """Journalise la requête, applique la règle échue éventuelle, sinon délègue au faux serveur."""
        method = request.method.upper()
        path = FakeAlbert.normalise_path(request.url.path)
        self.seen.append((method, path))
        for rule in self._rules:
            if rule["method"] != method or not fnmatch.fnmatchcase(path, rule["pattern"]):
                continue
            rule["count"] += 1
            if rule["count"] == rule["nth"]:
                if rule["hook"] is not None:
                    rule["hook"](request)
                return self._apply(rule["action"], request, method, path)
        return self.fake.handler(request)

    def _apply(self, action, request, method, path):
        """Exécute une action programmée."""
        if action == "lost_response":
            self.fake.handler(request)
            raise httpx.ReadTimeout("réponse perdue (simulée)", request=request)
        if action == "timeout_unprocessed":
            raise httpx.ReadTimeout("délai de lecture (simulé)", request=request)
        if action == "drop_ids":
            response = self.fake.handler(request)
            body = response.json()
            body.pop("ids", None)
            return httpx.Response(response.status_code, json=body)
        self.fake.inject(method, path, action)
        return self.fake.handler(request)


def make_chunks(n, *, item_key="ITEMA001", filename="article-a.pdf", title="Titre du document A",
                dedup=True, doc_id="424242424242", start=0, texts=None, total=None):
    """Chunks tels que les écrit ``rad_chunk`` (vecteurs dense et sparse, ``path`` local compris).

    Args:
        n: nombre de chunks.
        item_key: ``itemKey`` Zotero du document.
        filename: nom du fichier source.
        title: titre du document.
        dedup: ``True`` : champs de dédup (``content_hash``, ``dedup_eligible``,
            ``id`` adressé par contenu) ; ``False`` : ``id`` = ``{doc_id}_{index}``
            comme avec ``DEDUP_ENABLED=0``.
        doc_id: ``doc_id`` (aléatoire dans ``rad_chunk``).
        start: premier ``chunk_index``.
        texts: textes imposés (sinon générés à partir du titre et de l'index).
        total: ``total_chunks`` (défaut ``start + n``).
    """
    chunks = []
    for offset in range(n):
        index = start + offset
        if texts is not None:
            text = texts[offset]
        else:
            text = (f"{title} — passage {index} : la politique des données de la recherche "
                    f"publique engage des choix d'infrastructure durables (marqueur {item_key}-{index}).")
        chunk = {
            "id": f"{doc_id}_{index}",
            "doc_id": doc_id,
            "chunk_index": index,
            "total_chunks": total if total is not None else start + n,
            "text": text,
            "title": title,
            "authors": "Dupont, Jeanne; Martin, Paul",
            "date": "2021-05-03",
            "itemKey": item_key,
            "filename": filename,
            "path": f"/Users/exemple/Zotero/storage/{item_key}/{filename}",
            "embedding": [0.125] * 8,
            "sparse_embedding": {"indices": [3, 7], "values": [0.5, 0.25]},
        }
        if dedup:
            chash, eligible, content_id = rad_dedup.compute_dedup_fields(text, index)
            chunk.update(content_hash=chash, dedup_eligible=eligible, id=content_id)
        chunks.append(chunk)
    return chunks


def write_chunks(directory, chunks, name="output_chunks_with_embeddings_sparse.json"):
    """Écrit ``chunks`` en JSON dans ``directory`` et renvoie le chemin."""
    path = Path(directory) / name
    path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    return str(path)


def push(path, transport, *, collection_name=COLLECTION, collection_id=None, create=True, ack=True,
         sleep=None):
    """Appelle ``insert_to_albert`` avec la clé factice et le transport fourni."""
    return rv.insert_to_albert(
        str(path),
        collection_id=collection_id,
        collection_name=collection_name,
        albert_api_key=FAKE_ALBERT_KEY,
        create_collection=create,
        ack_retention=ack,
        transport=transport,
        sleep=sleep if sleep is not None else _no_sleep,
    )


def seed_collection(fake, name, visibility="private"):
    """Crée une collection dans le faux serveur (API publique) et renvoie son id."""
    with fake.client() as http:
        response = http.post("/collections", json={"name": name, "visibility": visibility})
    assert response.status_code == 201
    return response.json()["id"]


def seed_document(fake, collection_id, name):
    """Crée un document vide (multipart sans fichier) dans le faux serveur et renvoie son id."""
    with fake.client() as http:
        response = http.post("/documents", files={"name": (None, name), "collection_id": (None, str(collection_id))})
    assert response.status_code == 201
    return response.json()["id"]


def manifest_rows(directory):
    """Lignes du manifeste ``albert_manifest.jsonl`` de ``directory``."""
    return col.read_manifest(col.manifest_path_for(str(directory)))


def chunk_posts(fake):
    """Appels ``POST /v1/documents/{id}/chunks`` reçus par le faux serveur."""
    return fake.calls_to("POST", CHUNK_POSTS)


def writes(fake):
    """Appels d'écriture (POST, PATCH, DELETE) reçus par le faux serveur."""
    return [c for c in fake.calls if c.method in ("POST", "PATCH", "DELETE")]


def run_cli(monkeypatch, capsys, args):
    """Exécute ``scripts/rad_vectordb.py`` comme ``__main__`` ; renvoie (code de sortie, lignes)."""
    monkeypatch.setattr(sys, "argv", ["rad_vectordb.py"] + list(args))
    capsys.readouterr()
    code = None
    try:
        runpy.run_path(VECTORDB_SCRIPT, run_name="__main__")
    except SystemExit as exc:
        code = exc.code
    return code, capsys.readouterr().out.splitlines()


def result_block(lines):
    """Lignes à partir de ``=== Result ===``."""
    return lines[lines.index("=== Result ==="):]


@pytest.fixture(autouse=True)
def _albert_on(monkeypatch):
    """Albert activé ; identifiants de la baseline, dédup et cache retirés ; aucune attente réelle."""
    for name in golden_harness.BASELINE_ENV_NAMES + golden_harness.EXTRA_CLEARED_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.startswith(("DEDUP_", "RECODE_")):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    monkeypatch.setattr(time, "sleep", _no_sleep)
    albert_limiter.reset_limiters()
    albert_preflight.clear_preflight_cache()
    yield
    albert_limiter.reset_limiters()
    albert_preflight.clear_preflight_cache()


@pytest.fixture
def cli_fake(monkeypatch):
    """CLI branchée sur ``FakeAlbert`` : ``AlbertClient`` remplacé par une fabrique à transport factice."""
    fake = FakeAlbert()
    real_client = albert_client_mod.AlbertClient
    built = []

    def factory(cfg, api_key, **kwargs):
        """Construit le vrai client, mais sur le transport du faux serveur et sans attente."""
        kwargs["transport"] = fake.transport
        kwargs["sleep"] = _no_sleep
        client = real_client(cfg, api_key, **kwargs)
        built.append(client)
        return client

    monkeypatch.setattr(albert_client_mod, "AlbertClient", factory)
    monkeypatch.setenv("ALBERT_API_KEY", FAKE_ALBERT_KEY)
    return SimpleNamespace(fake=fake, built=built)


# ---------------------------------------------------------------------------
# Helpers purs : scalaires, noms, content_id, liste blanche
# ---------------------------------------------------------------------------
def test_sanitize_rules():
    # Scalaires : vides écartés, float entier -> int, grands nombres -> chaîne, listes jointes, 255.
    for blank in ("", "   ", None, float("nan"), float("inf"), float("-inf"), []):
        assert col.albert_scalar(blank) is None
    year = col.albert_scalar(2021.0)
    assert year == 2021 and type(year) is int
    assert col.albert_scalar(12.5) == 12.5
    assert col.albert_scalar(10 ** 16) == 10 ** 16
    for huge in (10 ** 16 + 1, 1e17, -1e17, 10 ** 30):
        assert isinstance(col.albert_scalar(huge), str)
    assert col.albert_scalar(True) is True
    joined = col.albert_scalar(["Dupont", "", None, "Martin"])
    assert isinstance(joined, str) and "Dupont" in joined and "Martin" in joined and "None" not in joined
    assert len(col.albert_scalar("x" * 300)) == 255
    assert len(col.albert_scalar(["y" * 200, "z" * 200])) == 255

    chunk = make_chunks(1, title=LONG_TITLE)[0]
    chunk["filename"] = "C:\\Zotero\\storage\\ITEMA001\\article-a.pdf"
    chunk["url"] = "https://doi.org/10.1234/exemple"
    chunk["extra_nan"] = float("nan")
    meta = col.sanitize_metadata(chunk, DEFAULT_METADATA_FIELDS)
    assert "path" not in meta
    assert set(meta) <= set(DEFAULT_METADATA_FIELDS) and len(meta) <= 10
    assert meta["doi"] == chunk["url"]
    assert meta["filename"] == "article-a.pdf"
    assert meta["item_key"] == "ITEMA001"
    assert meta["year"] == 2021 and type(meta["year"]) is int
    assert len(meta["title"]) == 255 and LONG_TITLE.startswith(meta["title"])
    assert meta["content_id"] == col.content_id_for(chunk)
    assert meta["chunk_index"] == 0
    assert FakeAlbert.metadata_errors(meta, ["body", "chunks", 0, "metadata"]) == []
    # Un DOI présent l'emporte sur l'URL ; « path » n'est jamais envoyé, même demandé.
    chunk["doi"] = "10.1234/exemple"
    assert col.sanitize_metadata(chunk, DEFAULT_METADATA_FIELDS)["doi"] == "10.1234/exemple"
    forced = col.sanitize_metadata(chunk, ("content_id", "content_hash", "chunk_index", "title", "path"))
    assert "path" not in forced
    # Aucune valeur de métadonnée ne reprend le chemin local.
    assert all("/Users/exemple" not in str(value) for value in meta.values())
    # Liste blanche par défaut : 10 champs, les champs imposés présents, jamais path.
    fields = col.effective_metadata_fields({})
    assert tuple(fields) == tuple(DEFAULT_METADATA_FIELDS) and len(fields) == 10
    assert {"content_id", "content_hash", "chunk_index", "title"} <= set(fields) and "path" not in fields


def test_document_name_stable_unique():
    base = make_chunks(3, item_key="ABCD1234", filename="/Users/exemple/Zotero/storage/ABCD1234/Mon article.pdf")
    names = {col.document_name_for(chunk) for chunk in base}
    assert names == {"ragpy:ABCD1234:Mon article.pdf"}
    # Stable d'un run à l'autre : le doc_id aléatoire n'y entre jamais.
    rerun = make_chunks(3, item_key="ABCD1234", filename="Mon article.pdf", doc_id="999999999999", dedup=False)
    assert {col.document_name_for(chunk) for chunk in rerun} == names
    assert all("999999999999" not in col.document_name_for(chunk) for chunk in rerun)
    # Unique par document source.
    other_item = make_chunks(1, item_key="WXYZ5678", filename="Mon article.pdf")[0]
    other_file = make_chunks(1, item_key="ABCD1234", filename="Annexe.pdf")[0]
    assert col.document_name_for(other_item) not in names
    assert col.document_name_for(other_file) not in names
    # Sans nom de fichier : repli sur le titre.
    no_file = dict(base[0])
    no_file.pop("filename")
    assert col.document_name_for(no_file) == "ragpy:ABCD1234:" + base[0]["title"]
    # Au-delà de 255 caractères : tronqué, suffixé d'un hash, toujours unique.
    long_a = dict(base[0], filename="a" * 400 + "-1.pdf")
    long_b = dict(base[0], filename="a" * 400 + "-2.pdf")
    name_a, name_b = col.document_name_for(long_a), col.document_name_for(long_b)
    assert len(name_a) <= 255 and len(name_b) <= 255
    assert name_a.startswith("ragpy:ABCD1234:") and name_a != name_b
    assert col.document_name_for(long_a) == name_a


def test_content_id_reuses_rad_dedup(monkeypatch):
    text = "Un passage suffisamment long pour être éligible à la déduplication du pipeline RAGpy."
    with_fields = make_chunks(1, texts=[text], start=4)[0]
    without_fields = make_chunks(1, texts=[text], start=4, dedup=False, doc_id="123456789012")[0]
    random_id_with_hash = dict(with_fields, id="777777777777_4")
    expected = rad_dedup.content_id(rad_dedup.content_hash(text), 4)
    assert expected == rad_dedup.compute_dedup_fields(text, 4)[2]
    # 1) id déjà adressé par contenu ; 2) content_hash présent ; 3) hash du texte (DEDUP_ENABLED=0).
    assert col.content_id_for(with_fields) == with_fields["id"] == expected
    assert col.content_id_for(random_id_with_hash) == expected
    assert col.content_id_for(without_fields) == expected
    assert without_fields["id"] != expected
    # Délégation réelle à rad_dedup (aucune réimplémentation du hachage).
    calls = []
    real_content_id = col.rad_dedup.content_id

    def spy(chash, chunk_index):
        """Compte les appels à ``rad_dedup.content_id``."""
        calls.append(chunk_index)
        return real_content_id(chash, chunk_index)

    monkeypatch.setattr(col.rad_dedup, "content_id", spy)
    assert col.content_id_for(without_fields) == expected
    assert col.content_id_for(random_id_with_hash) == expected
    assert calls == [4, 4]


@pytest.mark.parametrize("override, dedup_meta", [
    ("content_hash,chunk_index,title", None),
    ("content_id,chunk_index,title", None),
    ("content_id,content_hash,title", None),
    ("content_id,content_hash,chunk_index", None),
    ("content_id,content_hash,chunk_index,title", "title,authors"),
    ("content_id,content_hash,chunk_index,title,path", None),
    ("content_id,content_hash,chunk_index,title,authors,year,doi,item_key,filename,total_chunks,url", None),
])
def test_metadata_override_must_keep_content_id(override, dedup_meta, tmp_path, monkeypatch):
    fake = FakeAlbert()
    monkeypatch.setenv("ALBERT_METADATA_FIELDS", override)
    if dedup_meta:
        monkeypatch.setenv("DEDUP_META_FIELDS", dedup_meta)
    with pytest.raises(ValueError):
        col.effective_metadata_fields()
    path = write_chunks(tmp_path, make_chunks(2))
    res = push(path, fake.transport)
    assert res["status"] == "error"
    assert "ALBERT_METADATA_FIELDS" in res["message"]
    assert fake.calls == []  # refus avant tout appel réseau


def test_metadata_valid_override_restricts_sent_keys(tmp_path, monkeypatch):
    fake = FakeAlbert()
    monkeypatch.setenv("ALBERT_METADATA_FIELDS", "content_id,content_hash,chunk_index,title")
    path = write_chunks(tmp_path, make_chunks(3))
    res = push(path, fake.transport)
    assert res["status"] == "success"
    for call in chunk_posts(fake):
        for item in call.json["chunks"]:
            assert set(item["metadata"]) == {"content_id", "content_hash", "chunk_index", "title"}


def test_resolve_albert_input_order(tmp_path):
    assert col.resolve_albert_input(None) is None
    assert col.resolve_albert_input(tmp_path / "absent") is None
    assert col.resolve_albert_input(tmp_path) is None
    names = ["output_chunks.json", "output_chunks_with_embeddings.json", "output_chunks_with_embeddings_sparse.json"]
    for name in names:
        (tmp_path / name).write_text("[]", encoding="utf-8")
        # Le fichier le plus avancé présent l'emporte : sparse > dense > chunks.
        assert os.path.basename(col.resolve_albert_input(tmp_path)) == name
    (tmp_path / "output_chunks_with_embeddings_sparse.json").unlink()
    assert os.path.basename(col.resolve_albert_input(str(tmp_path))) == "output_chunks_with_embeddings.json"
    (tmp_path / "output_chunks_with_embeddings.json").unlink()
    assert os.path.basename(col.resolve_albert_input(tmp_path)) == "output_chunks.json"


# ---------------------------------------------------------------------------
# Collections : get-or-create, nom exact, homonymes, privée forcée
# ---------------------------------------------------------------------------
def _substring_listing(fake):
    """Réponse ``GET /v1/collections`` d'un serveur qui filtre ``name`` par sous-chaîne (défaut D14)."""

    def reply(request):
        """Page de collections dont le nom contient ``name``."""
        params = dict(request.url.params)
        offset = int(params.get("offset", 0))
        limit = int(params.get("limit", 10))
        needle = params.get("name", "")
        rows = [dict(c) for _cid, c in sorted(fake.collections.items()) if needle in c["name"]]
        return (200, {"object": "list", "data": rows[offset:offset + limit]})

    return reply


def test_collection_get_or_create_exact_name_paginated(tmp_path):
    # Cible exacte en 2e page, derrière 120 homonymes partiels.
    fake = FakeAlbert()
    for i in range(120):
        seed_collection(fake, f"{COLLECTION}-{i:03d}")
    target = seed_collection(fake, COLLECTION)
    fake.reset_calls()
    fake.inject("GET", "/v1/collections", *[_substring_listing(fake) for _ in range(6)])
    path = write_chunks(tmp_path, make_chunks(2))
    res = push(path, fake.transport)
    assert res["status"] == "success"
    listings = fake.calls_to("GET", "/v1/collections")
    assert [c.params.get("offset") for c in listings] == ["0", "100"]
    assert {c.params.get("limit") for c in listings} == {"100"}
    assert fake.calls_to("POST", "/v1/collections") == []
    assert {d["collection_id"] for d in fake.documents.values()} == {target}

    # Aucun nom exact (seulement des homonymes partiels) : création d'une collection privée, une seule fois.
    fake = FakeAlbert()
    for name in (f"{COLLECTION}-a", f"archive {COLLECTION}", f"{COLLECTION}s"):
        seed_collection(fake, name)
    fake.reset_calls()
    fake.inject("GET", "/v1/collections", *[_substring_listing(fake) for _ in range(6)])
    run_dir = tmp_path / "create"
    run_dir.mkdir()
    path = write_chunks(run_dir, make_chunks(2))
    assert push(path, fake.transport)["status"] == "success"
    (created,) = fake.calls_to("POST", "/v1/collections")
    assert created.json["name"] == COLLECTION and created.json["visibility"] == "private"
    fake.reset_calls()
    rerun = push(path, fake.transport)
    assert rerun["status"] == "success" and rerun["inserted_count"] == 0
    assert fake.calls_to("POST", "/v1/collections") == []
    assert sum(1 for c in fake.collections.values() if c["name"] == COLLECTION) == 1

    # Sans autorisation de création : erreur, aucune écriture.
    fake = FakeAlbert()
    seed_collection(fake, f"{COLLECTION}-a")
    fake.reset_calls()
    res = push(path, fake.transport, create=False)
    assert res["status"] == "error"
    assert writes(fake) == []


def test_duplicate_collection_names_error(tmp_path):
    fake = FakeAlbert()
    first = seed_collection(fake, COLLECTION)
    second = seed_collection(fake, COLLECTION)
    fake.reset_calls()
    path = write_chunks(tmp_path, make_chunks(2))
    res = push(path, fake.transport)
    assert res["status"] == "error"
    assert str(first) in res["message"] and str(second) in res["message"]
    assert writes(fake) == []
    with pytest.raises(col.AlbertTargetError) as excinfo:
        col.select_collection(list(fake.collections.values()), COLLECTION)
    assert sorted(excinfo.value.ids) == sorted([first, second])

    # Documents homonymes dans la collection cible : erreur listant les ids, aucun envoi.
    fake = FakeAlbert()
    target = seed_collection(fake, COLLECTION)
    doc_name = col.document_name_for(make_chunks(1)[0])
    doc_a = seed_document(fake, target, doc_name)
    doc_b = seed_document(fake, target, doc_name)
    fake.reset_calls()
    res = push(path, fake.transport)
    assert res["status"] == "error"
    assert str(doc_a) in res["message"] and str(doc_b) in res["message"]
    assert writes(fake) == []


def test_public_refused_private_forced(tmp_path):
    path = write_chunks(tmp_path, make_chunks(2))
    # Création : toujours privée.
    fake = FakeAlbert()
    assert push(path, fake.transport)["status"] == "success"
    (created,) = fake.calls_to("POST", "/v1/collections")
    assert created.json["visibility"] == "private"
    assert all(c["visibility"] == "private" for c in fake.collections.values())

    # Collection publique de même nom : refus, pas de doublon privé créé en silence.
    fake = FakeAlbert()
    public_id = seed_collection(fake, COLLECTION, visibility="public")
    fake.reset_calls()
    res = push(path, fake.transport)
    assert res["status"] == "error" and str(public_id) in res["message"]
    assert writes(fake) == []

    # Collection publique désignée par son id : refus, aucune écriture.
    fake.reset_calls()
    res = push(path, fake.transport, collection_id=public_id, collection_name=None)
    assert res["status"] == "error"
    assert writes(fake) == []

    # Le client refuse toute visibilité autre que privée, avant tout envoi.
    fake = FakeAlbert()
    client = albert_client_mod.AlbertClient(AlbertConfig(enabled=True), FAKE_ALBERT_KEY,
                                            transport=fake.transport, sleep=_no_sleep)
    with pytest.raises(AlbertInputError):
        client.create_collection(COLLECTION, visibility="public")
    assert fake.calls == []
    client.close()


_ABSENT = object()
"""Marqueur : clé ``visibility`` absente de la réponse de l'API."""


def _with_visibility(collection, visibility):
    """Copie de ``collection`` dont la visibilité est retirée (``_ABSENT``) ou remplacée."""
    view = dict(collection)
    if visibility is _ABSENT:
        view.pop("visibility", None)
    else:
        view["visibility"] = visibility
    return view


@pytest.mark.parametrize("visibility", [_ABSENT, None, "", "  "], ids=["absent", "null", "empty", "blank"])
def test_missing_visibility_refused(visibility, tmp_path):
    # Visibilité non confirmée = non privée (échec fermé), au niveau des helpers purs…
    listed = _with_visibility({"id": 7, "name": COLLECTION, "visibility": "private"}, visibility)
    assert col.visibility_of(listed) is None
    assert col.is_private(listed) is False
    with pytest.raises(col.AlbertTargetError) as excinfo:
        col.select_collection([listed], COLLECTION)
    assert excinfo.value.ids == [7]

    path = write_chunks(tmp_path, make_chunks(2))
    # … par id : GET /v1/collections/{id} sans visibilité confirmée -> refus, aucune écriture.
    fake = FakeAlbert()
    target = seed_collection(fake, COLLECTION)
    fake.inject("GET", f"/v1/collections/{target}", (200, _with_visibility(fake.collections[target], visibility)))
    fake.reset_calls()
    res = push(path, fake.transport, collection_id=target, collection_name=None)
    assert res["status"] == "error"
    assert "visibilité inconnue" in res["message"] and str(target) in res["message"]
    assert len(fake.calls_to("GET", f"/v1/collections/{target}")) == 1
    assert fake.calls_to("GET", "/v1/documents") == []  # refus avant le préchargement
    assert writes(fake) == []
    assert res["manifest_path"] is None
    assert not os.path.exists(col.manifest_path_for(str(tmp_path)))

    # … par nom : seule collection de ce nom, visibilité non confirmée -> refus, et aucune
    # collection privée homonyme créée en silence (création pourtant autorisée).
    fake = FakeAlbert()
    target = seed_collection(fake, COLLECTION)
    fake.collections[target] = _with_visibility(fake.collections[target], visibility)
    fake.reset_calls()
    res = push(path, fake.transport, create=True)
    assert res["status"] == "error"
    assert "visibilité inconnue" in res["message"] and str(target) in res["message"]
    assert fake.calls_to("GET", "/v1/collections")
    assert fake.calls_to("POST", "/v1/collections") == []
    assert writes(fake) == []
    assert [c["id"] for c in fake.collections.values() if c["name"] == COLLECTION] == [target]


def test_collection_by_id_private_and_name_checked(tmp_path):
    fake = FakeAlbert()
    target = seed_collection(fake, COLLECTION)
    fake.reset_calls()
    path = write_chunks(tmp_path, make_chunks(2))
    res = push(path, fake.transport, collection_id=target, collection_name=None, create=False)
    assert res["status"] == "success" and res["inserted_count"] == 2
    assert fake.calls_to("GET", "/v1/collections") == []
    assert fake.calls_to("POST", "/v1/collections") == []
    # Id et nom incohérents : refus.
    fake.reset_calls()
    res = push(path, fake.transport, collection_id=target, collection_name="un-autre-nom")
    assert res["status"] == "error"
    assert writes(fake) == []


# ---------------------------------------------------------------------------
# Documents, tranches, payloads
# ---------------------------------------------------------------------------
def test_document_multipart_no_metadata(tmp_path):
    fake = FakeAlbert()
    chunks = make_chunks(3)
    path = write_chunks(tmp_path, chunks)
    assert push(path, fake.transport)["status"] == "success"
    (call,) = fake.calls_to("POST", "/v1/documents")
    assert call.headers.get("content-type", "").startswith("multipart/form-data")
    cid = next(iter(fake.collections))
    assert call.form == {"name": col.document_name_for(chunks[0]), "collection_id": str(cid)}
    assert call.files == {}
    assert "metadata" not in call.form
    assert call.json is None


def test_chunks_sliced_64(tmp_path):
    fake = FakeAlbert()
    path = write_chunks(tmp_path, make_chunks(130))
    res = push(path, fake.transport)
    assert res["status"] == "success" and res["inserted_count"] == 130
    posts = chunk_posts(fake)
    assert len(posts) == 3 == math.ceil(130 / 64)
    assert [len(c.json["chunks"]) for c in posts] == [64, 64, 2]
    assert len(fake.calls_to("POST", "/v1/documents")) == 1
    (did,) = fake.documents
    assert len(fake.chunks[did]) == 130


def test_no_vectors_in_payloads(tmp_path):
    fake = FakeAlbert()
    chunks = make_chunks(70)
    path = write_chunks(tmp_path, chunks)
    assert push(path, fake.transport)["status"] == "success"
    assert fake.calls_to(None, "/v1/embeddings") == []
    assert fake.calls_to(None, "/v1/search") == []
    for call in fake.calls:
        body = call.content or b""
        assert b"embedding" not in body
        assert b"/Users/exemple" not in body
        assert b"path" not in body
    for call in chunk_posts(fake):
        for item in call.json["chunks"]:
            assert set(item) == {"content", "metadata"}
            assert set(item["metadata"]) <= set(DEFAULT_METADATA_FIELDS)
            assert FakeAlbert.metadata_errors(item["metadata"], ["m"]) == []
    sent = [item["content"] for call in chunk_posts(fake) for item in call.json["chunks"]]
    assert sent == [c["text"] for c in chunks]


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------
def test_rerun_idempotent_zero_post(tmp_path):
    fake = FakeAlbert()
    chunks = make_chunks(130)
    path = write_chunks(tmp_path, chunks)
    first = push(path, fake.transport)
    assert first["status"] == "success" and first["inserted_count"] == 130
    rows_after_first = manifest_rows(tmp_path)
    fake.reset_calls()
    second = push(path, fake.transport)
    assert second["status"] == "success"
    assert second["inserted_count"] == 0 and second["existing_count"] == 130
    assert writes(fake) == []
    assert manifest_rows(tmp_path) == rows_after_first
    # Ajout de chunks : seuls les nouveaux partent, dans le document existant.
    more = chunks + make_chunks(5, start=130, total=135)
    write_chunks(tmp_path, more)
    fake.reset_calls()
    third = push(path, fake.transport)
    assert third["status"] == "success"
    assert third["inserted_count"] == 5 and third["existing_count"] == 130
    assert fake.calls_to("POST", "/v1/documents") == []
    assert [len(c.json["chunks"]) for c in chunk_posts(fake)] == [5]
    assert len(fake.documents) == 1


def test_rerun_idempotent_without_dedup_fields(tmp_path):
    fake = FakeAlbert()
    first_chunks = make_chunks(10, dedup=False, doc_id="111111111111")
    path = write_chunks(tmp_path, first_chunks)
    first = push(path, fake.transport)
    assert first["status"] == "success" and first["inserted_count"] == 10
    sent_ids = [item["metadata"]["content_id"] for c in chunk_posts(fake) for item in c.json["chunks"]]
    assert sent_ids == [rad_dedup.content_id(rad_dedup.content_hash(c["text"]), c["chunk_index"])
                        for c in first_chunks]
    # rad_chunk relancé avec DEDUP_ENABLED=0 : nouveau doc_id aléatoire, mêmes textes.
    second_chunks = make_chunks(10, dedup=False, doc_id="222222222222")
    assert {c["id"] for c in second_chunks}.isdisjoint({c["id"] for c in first_chunks})
    write_chunks(tmp_path, second_chunks)
    fake.reset_calls()
    second = push(path, fake.transport)
    assert second["status"] == "success"
    assert second["inserted_count"] == 0 and second["existing_count"] == 10
    assert writes(fake) == []


def test_manifest_merge_covers_indexing_window(tmp_path, monkeypatch):
    # D18 : des chunks tout juste envoyés peuvent manquer à la liste ; le manifeste local comble la fenêtre.
    fake = FakeAlbert()
    path = write_chunks(tmp_path, make_chunks(4))
    assert push(path, fake.transport)["inserted_count"] == 4
    listing_path = "/v1/documents/*/chunks"
    fake.inject("GET", listing_path, (200, {"object": "list", "data": []}))
    fake.reset_calls()
    res = push(path, fake.transport)
    assert res["status"] == "success"
    assert res["inserted_count"] == 0 and res["existing_count"] == 4
    assert chunk_posts(fake) == []


# ---------------------------------------------------------------------------
# Échecs, rollback, manifeste
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("incident", [422, "lost_response"])
def test_slice_failure_deletes_new_document(incident, tmp_path):
    fake = FakeAlbert()
    scripted = ScriptedTransport(fake).on("POST", CHUNK_POSTS, 2, incident)
    doc_a = make_chunks(130, item_key="ITEMA001", filename="a.pdf")
    doc_b = make_chunks(3, item_key="ITEMB002", filename="b.pdf", title="Titre du document B")
    path = write_chunks(tmp_path, doc_a + doc_b)
    res = push(path, scripted.transport)
    assert res["status"] == "partial_error"
    assert res["inserted_count"] == 3
    # Aucun renvoi de la tranche en échec (création non idempotente).
    assert scripted.count("POST", CHUNK_POSTS) == 3
    name_a = col.document_name_for(doc_a[0])
    name_b = col.document_name_for(doc_b[0])
    (deleted,) = fake.calls_to("DELETE", "/v1/documents/*")
    assert all(d["name"] != name_a for d in fake.documents.values())
    assert [d["name"] for d in fake.documents.values()] == [name_b]
    deleted_id = deleted.path.rsplit("/", 1)[-1]
    assert deleted_id not in {str(i) for i in fake.documents}
    # Manifeste : la tranche du document supprimé reste (append-only), suivie d'une ligne
    # d'événement « rollback » ; la lecture par défaut ne renvoie que les lignes de tranche.
    manifest = col.manifest_path_for(str(tmp_path))
    all_rows = col.read_manifest(manifest, include_events=True)
    events = [r for r in all_rows if "event" in r]
    (rollback,) = events
    slices_a = [r for r in all_rows if "event" not in r and r["document_name"] == name_a]
    assert [r["slice"] for r in slices_a] == [1]
    assert rollback["event"] == "rollback"
    assert rollback["document_name"] == name_a and str(rollback["document_id"]) == deleted_id
    assert rollback["chunks_removed"] == sum(r["count"] for r in slices_a) == 64
    assert rollback["collection_id"] in fake.collections
    assert "content_ids" not in rollback
    assert all_rows.index(rollback) > all_rows.index(slices_a[-1])
    assert [r["document_name"] for r in manifest_rows(tmp_path)] == [name_a, name_b]
    assert manifest_rows(tmp_path) == [r for r in all_rows if "event" not in r]
    # Relance : le document supprimé est renvoyé en entier, rien d'autre.
    fake.reset_calls()
    rerun = push(path, fake.transport)
    assert rerun["status"] == "success"
    assert rerun["inserted_count"] == 130 and rerun["existing_count"] == 3


def test_manifest_written_per_slice(tmp_path):
    fake = FakeAlbert()
    seen_at_failure = []

    def check_manifest(_request):
        """Au moment de la 2e tranche : la 1re ligne est déjà écrite et flushée."""
        seen_at_failure.append(manifest_rows(tmp_path))

    scripted = ScriptedTransport(fake).on("POST", CHUNK_POSTS, 2, 422, hook=check_manifest)
    path = write_chunks(tmp_path, make_chunks(130))
    res = push(path, scripted.transport)
    assert res["status"] == "partial_error"
    assert res["manifest_path"] == col.manifest_path_for(str(tmp_path))
    (rows_then,) = seen_at_failure
    assert len(rows_then) == 1
    row = rows_then[0]
    assert row["slice"] == 1 and row["count"] == 64 and len(row["content_ids"]) == 64
    assert row["document_created"] is True
    assert row["chunk_ids"] == list(range(64))  # ids renvoyés par l'envoi (D16)
    # Append-only : la ligne reste après le rollback.
    assert manifest_rows(tmp_path) == rows_then

    # Succès complet : une ligne par tranche, ids de chunks du serveur (D16).
    fake = FakeAlbert()
    run_dir = tmp_path / "ok"
    run_dir.mkdir()
    chunks = make_chunks(130)
    res = push(write_chunks(run_dir, chunks), fake.transport)
    rows = manifest_rows(run_dir)
    assert res["manifest_path"] == col.manifest_path_for(str(run_dir))
    assert [r["slice"] for r in rows] == [1, 2, 3]
    assert [r["count"] for r in rows] == [64, 64, 2]
    (did,) = fake.documents
    assert [i for r in rows for i in r["chunk_ids"]] == [c["id"] for c in fake.chunks[did]]
    assert [cid for r in rows for cid in r["content_ids"]] == [col.content_id_for(c) for c in chunks]
    assert {r["collection_id"] for r in rows} == set(fake.collections)
    assert {r["document_id"] for r in rows} == {did}


def test_manifest_without_chunk_ids_keeps_document_slice_count(tmp_path):
    fake = FakeAlbert()
    scripted = ScriptedTransport(fake).on("POST", CHUNK_POSTS, 1, "drop_ids")
    path = write_chunks(tmp_path, make_chunks(3))
    assert push(path, scripted.transport)["status"] == "success"
    (row,) = manifest_rows(tmp_path)
    assert row["chunk_ids"] is None
    assert row["slice"] == 1 and row["count"] == 3
    assert row["document_id"] in fake.documents


@pytest.mark.parametrize("where", ["documents", "chunks", "auth"])
@pytest.mark.parametrize("dedup_enabled", ["0", "1"])
def test_preload_failure_is_loud(where, dedup_enabled, tmp_path, monkeypatch):
    monkeypatch.setenv("DEDUP_ENABLED", dedup_enabled)
    fake = FakeAlbert()
    path = write_chunks(tmp_path, make_chunks(3))
    if where == "chunks":
        assert push(path, fake.transport)["status"] == "success"
        write_chunks(tmp_path, make_chunks(5))
        fake.inject("GET", "/v1/documents/*/chunks", *([500] * 8))
    elif where == "documents":
        fake.inject("GET", "/v1/documents", *([500] * 8))
    else:
        fake.inject("GET", "/v1/documents", "invalid_key")
    fake.reset_calls()
    res = push(path, fake.transport)
    assert res["status"] == "error"
    assert res["inserted_count"] == 0
    if where == "auth":
        assert res.get("credential_required") == "albert_api_key"
    else:
        assert "préchargement" in res["message"]
    # Rien n'est écrit (ni document, ni chunk) : l'échec n'est pas avalé par la dédup.
    assert fake.calls_to("POST", "/v1/documents") == []
    assert chunk_posts(fake) == []


@pytest.mark.parametrize("entry", ["invalid_key", "account_expired", "budget_exhausted",
                                   {"status": 429, "retry_after": 600}])
def test_account_error_aborts_whole_job(entry, tmp_path, capsys):
    fake = FakeAlbert()
    scripted = ScriptedTransport(fake).on("POST", CHUNK_POSTS, 2, entry)
    doc_a = make_chunks(130, item_key="ITEMA001", filename="a.pdf")
    doc_b = make_chunks(3, item_key="ITEMB002", filename="b.pdf", title="Titre du document B")
    path = write_chunks(tmp_path, doc_a + doc_b)
    res = push(path, scripted.transport)
    assert res["status"] == "error"
    assert res.get("credential_required") == "albert_api_key"
    # Arrêt du job entier : rien pour le document B, document A (créé) supprimé.
    assert scripted.count("POST", CHUNK_POSTS) == 2
    assert scripted.count("POST", "/v1/documents") == 1
    assert len(fake.calls_to("DELETE", "/v1/documents/*")) == 1
    assert fake.documents == {}
    assert res["inserted_count"] == 0
    out = capsys.readouterr().out
    assert FAKE_ALBERT_KEY not in out and FAKE_ALBERT_KEY not in json.dumps(res)


def test_errors_returned_as_dict_never_raised(tmp_path):
    fake = FakeAlbert()
    missing = push(tmp_path / "absent.json", fake.transport)
    assert missing["status"] == "error"
    not_a_list = tmp_path / "objet.json"
    not_a_list.write_text(json.dumps({"chunks": []}), encoding="utf-8")
    assert push(not_a_list, fake.transport)["status"] == "error"
    path = write_chunks(tmp_path, make_chunks(2))
    no_key = rv.insert_to_albert(path, collection_name=COLLECTION, albert_api_key="  ",
                                 create_collection=True, ack_retention=True, transport=fake.transport,
                                 sleep=_no_sleep)
    assert no_key["status"] == "error" and no_key.get("credential_required") == "albert_api_key"
    no_target = push(path, fake.transport, collection_name=None)
    assert no_target["status"] == "error"
    assert fake.calls == []
    for res in (missing, no_key, no_target):
        assert {"status", "message", "inserted_count", "existing_count", "manifest_path"} <= set(res)


@pytest.mark.parametrize("enabled", ["0", "false", None])
def test_library_refuses_when_albert_disabled(enabled, tmp_path, monkeypatch, capsys):
    # Interrupteur maître coupé (explicitement ou par défaut) : la bibliothèque refuse avant tout,
    # sans construire de client ni émettre d'appel, et avant même la question de l'acquittement.
    if enabled is None:
        monkeypatch.delenv("ALBERT_ENABLED", raising=False)
    else:
        monkeypatch.setenv("ALBERT_ENABLED", enabled)
    fake = FakeAlbert()
    real_client = albert_client_mod.AlbertClient
    built = []

    def factory(*args, **kwargs):
        """Enregistre toute construction de client puis délègue au vrai client."""
        built.append(True)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(albert_client_mod, "AlbertClient", factory)
    path = write_chunks(tmp_path, make_chunks(3))
    capsys.readouterr()
    for ack in (True, False):
        res = push(path, fake.transport, ack=ack)
        assert res["status"] == "error"
        assert "ALBERT_ENABLED=0" in res["message"]
        assert "--albert-ack-retention" not in res["message"]
        assert res["inserted_count"] == 0 and res["existing_count"] == 0
        assert res["manifest_path"] is None
        assert "credential_required" not in res
    assert fake.calls == []
    assert built == []
    assert not os.path.exists(col.manifest_path_for(str(tmp_path)))
    assert not os.path.exists(os.path.join(str(tmp_path), USAGE_FILENAME))
    assert FAKE_ALBERT_KEY not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Écritures à l'issue incertaine : rapprochement, jamais de renvoi aveugle
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", ["processed", "unprocessed", "concurrent_duplicate"])
def test_uncertain_collection_creation_reconciled_by_relist(scenario, tmp_path):
    fake = FakeAlbert()
    scripted = ScriptedTransport(fake)
    if scenario == "processed":
        scripted.on("POST", "/v1/collections", 1, "lost_response")
    elif scenario == "unprocessed":
        scripted.on("POST", "/v1/collections", 1, "timeout_unprocessed")
    else:

        def concurrent_creation(_request):
            """Un autre processus crée une collection homonyme pendant la requête."""
            seed_collection(fake, COLLECTION)

        scripted.on("POST", "/v1/collections", 1, "lost_response", hook=concurrent_creation)
    path = write_chunks(tmp_path, make_chunks(3))
    res = push(path, scripted.transport)
    # Jamais de second POST : la liste est relue.
    assert scripted.count("POST", "/v1/collections") == 1
    post_at = scripted.seen.index(("POST", "/v1/collections"))
    assert ("GET", "/v1/collections") in scripted.seen[post_at + 1:]
    same_name = [c for c in fake.collections.values() if c["name"] == COLLECTION]
    if scenario == "processed":
        assert res["status"] == "success" and res["inserted_count"] == 3
        (created,) = same_name
        assert {d["collection_id"] for d in fake.documents.values()} == {created["id"]}
    elif scenario == "unprocessed":
        assert res["status"] == "error"
        assert same_name == []
        assert scripted.count("POST", "/v1/documents") == 0
    else:
        assert res["status"] == "error"
        assert len(same_name) == 2
        assert all(str(c["id"]) in res["message"] for c in same_name)
        assert scripted.count("POST", "/v1/documents") == 0


@pytest.mark.parametrize("scenario", ["processed", "unprocessed"])
def test_uncertain_document_creation_reconciled_by_relist(scenario, tmp_path):
    fake = FakeAlbert()
    action = "lost_response" if scenario == "processed" else "timeout_unprocessed"
    scripted = ScriptedTransport(fake).on("POST", "/v1/documents", 1, action)
    chunks = make_chunks(3)
    path = write_chunks(tmp_path, chunks)
    res = push(path, scripted.transport)
    assert scripted.count("POST", "/v1/documents") == 1
    name = col.document_name_for(chunks[0])
    if scenario == "processed":
        assert res["status"] == "success" and res["inserted_count"] == 3
        (doc,) = fake.documents.values()
        assert doc["name"] == name and len(fake.chunks[doc["id"]]) == 3
    else:
        assert res["status"] == "partial_error" and res["inserted_count"] == 0
        assert fake.documents == {}
        assert scripted.count("POST", CHUNK_POSTS) == 0


@pytest.mark.parametrize("scenario", ["processed", "unprocessed"])
def test_uncertain_chunk_push_on_existing_document_reconciled(scenario, tmp_path):
    fake = FakeAlbert()
    base = make_chunks(3)
    path = write_chunks(tmp_path, base)
    assert push(path, fake.transport)["inserted_count"] == 3
    (did,) = fake.documents
    write_chunks(tmp_path, base + make_chunks(2, start=3, total=5))
    action = "lost_response" if scenario == "processed" else "timeout_unprocessed"
    scripted = ScriptedTransport(fake).on("POST", CHUNK_POSTS, 1, action)
    sleeps = SleepRecorder()
    res = push(path, scripted.transport, sleep=sleeps)
    assert scripted.count("POST", CHUNK_POSTS) == 1  # jamais renvoyé
    # Relecture après l'attente d'indexation (D18 ≈ 2,4 s), puis décision.
    assert any(s >= DECISIONS["D18"]["value"]["index_latency_s"] for s in sleeps.calls)
    post_at = scripted.seen.index(("POST", f"/v1/documents/{did}/chunks"))
    assert ("GET", f"/v1/documents/{did}/chunks") in scripted.seen[post_at + 1:]
    # Un document préexistant n'est jamais supprimé.
    assert fake.calls_to("DELETE", "/v1/documents/*") == []
    assert did in fake.documents
    if scenario == "processed":
        assert res["status"] == "success" and res["inserted_count"] == 2
        assert len(fake.chunks[did]) == 5
        last = manifest_rows(tmp_path)[-1]
        assert last.get("reconciled") is True and last["count"] == 2
    else:
        assert res["status"] == "partial_error" and res["inserted_count"] == 0
        assert len(fake.chunks[did]) == 3


# ---------------------------------------------------------------------------
# Quotas (D19) et concurrence
# ---------------------------------------------------------------------------
def test_push_uses_embed_role_limiter(tmp_path, monkeypatch):
    fake = FakeAlbert()
    probe = albert_client_mod.AlbertClient(AlbertConfig(enabled=True), FAKE_ALBERT_KEY,
                                           transport=fake.transport, sleep=_no_sleep)
    assert albert_limiter.bucket_for_role("push") == "embed"
    assert probe._limiter_for("push") is probe._limiter_for("embed")
    probe.close()
    acquisitions = []
    real_limiter_for = albert_client_mod.AlbertClient._limiter_for

    def spy(self, role):
        """Enveloppe le limiteur du rôle dans un espion (``None`` conservé)."""
        limiter = real_limiter_for(self, role)
        return None if limiter is None else RecordingLimiter(limiter, role, acquisitions)

    monkeypatch.setattr(albert_client_mod.AlbertClient, "_limiter_for", spy)
    path = write_chunks(tmp_path, make_chunks(130))
    assert push(path, fake.transport)["status"] == "success"
    posts = chunk_posts(fake)
    assert [a["role"] for a in acquisitions] == ["push"] * len(posts)
    assert {albert_limiter.bucket_for_role(a["role"]) for a in acquisitions} == {"embed"}
    per_request = albert_client_mod.CHUNKS_PER_EMBED_REQUEST
    assert [a["requests"] for a in acquisitions] == [math.ceil(len(c.json["chunks"]) / per_request) for c in posts]
    assert acquisitions[0]["requests"] == DECISIONS["D19"]["value"]["embed_requests_per_push_64"]


def test_push_concurrency_keeps_slices_ordered_per_document(tmp_path, monkeypatch):
    monkeypatch.setenv("ALBERT_PUSH_CONCURRENCY", "3")
    fake = FakeAlbert()
    chunks = []
    for i in range(4):
        chunks += make_chunks(70, item_key=f"ITEM{i:04d}", filename=f"doc-{i}.pdf", title=f"Titre {i}")
    path = write_chunks(tmp_path, chunks)
    res = push(path, fake.transport)
    assert res["status"] == "success" and res["inserted_count"] == 280
    assert len(chunk_posts(fake)) == 8
    rows = manifest_rows(tmp_path)
    by_doc = {}
    for row in rows:
        by_doc.setdefault(row["document_id"], []).append(row["slice"])
    assert sorted(by_doc) == sorted(fake.documents)
    assert all(slices == [1, 2] for slices in by_doc.values())
    for did in fake.documents:
        assert [c["metadata"]["chunk_index"] for c in fake.chunks[did]] == list(range(70))


# ---------------------------------------------------------------------------
# Dédup Tier 2 en mémoire
# ---------------------------------------------------------------------------
def test_tier2_rehydrates_truncated_title(tmp_path, monkeypatch):
    monkeypatch.setenv("DEDUP_ENABLED", "1")
    fake = FakeAlbert()
    texts = [f"Passage partagé numéro {i} entre deux dépôts du même article, assez long pour la dédup."
             for i in range(3)]
    first = make_chunks(3, item_key="ITEMA001", filename="a.pdf", title=LONG_TITLE, texts=texts)
    run1 = tmp_path / "run1"
    run1.mkdir()
    assert push(write_chunks(run1, first), fake.transport)["inserted_count"] == 3
    (did,) = fake.documents
    stored_titles = {c["metadata"]["title"] for c in fake.chunks[did]}
    assert {len(t) for t in stored_titles} == {255}

    # En mémoire, sans appel HTTP : le titre stocké tronqué est réhydraté.
    client = albert_client_mod.AlbertClient(AlbertConfig(enabled=True), FAKE_ALBERT_KEY,
                                            transport=fake.transport, sleep=_no_sleep)
    adapter = rv._AlbertDedupAdapter(client, next(iter(fake.collections)), meta_fields=("title",),
                                     collections_module=col)
    adapter.preload()
    duplicate = make_chunks(3, item_key="ITEMB002", filename="b.pdf", title=LONG_TITLE, texts=texts)
    calls_before = len(fake.calls)
    found = adapter.existing(duplicate)
    assert len(fake.calls) == calls_before
    assert set(found) == {c["content_hash"] for c in duplicate}
    assert all(meta["title"] == LONG_TITLE for metas in found.values() for meta in metas)
    client.close()

    # De bout en bout : même texte et même titre long dans un autre document -> refusé en Tier 2.
    run2 = tmp_path / "run2"
    run2.mkdir()
    fake.reset_calls()
    res = push(write_chunks(run2, duplicate), fake.transport)
    assert res["status"] == "success"
    assert res["skipped_count"] == 3 and res["inserted_count"] == 0
    assert writes(fake) == []
    journal = [json.loads(line) for line in Path(res["journal_path"]).read_text(encoding="utf-8").splitlines()]
    assert {entry["reason"] for entry in journal} == {"existing_hash_match"}
    assert {entry["target_db"] for entry in journal} == {"albert"}

    # Titre différent : pas de corroboration, les chunks partent.
    run3 = tmp_path / "run3"
    run3.mkdir()
    other = make_chunks(3, item_key="ITEMC003", filename="c.pdf", title="Un tout autre titre", texts=texts)
    fake.reset_calls()
    res = push(write_chunks(run3, other), fake.transport)
    assert res["inserted_count"] == 3 and res.get("skipped_count", 0) == 0


@pytest.mark.parametrize("field, source_key, stored, other_source", [
    ("item_key", "itemKey", "ITEMA001", "ITEMZ999"),
    ("year", "date", 2021, "1999-01-01"),
])
def test_tier2_derived_meta_field_corroborates(field, source_key, stored, other_source, tmp_path, monkeypatch):
    # DEDUP_META_FIELDS peut nommer un champ dérivé de la liste blanche (item_key <- itemKey,
    # year <- date) : absent tel quel des chunks, il doit tout de même corroborer le Tier 2.
    monkeypatch.setenv("DEDUP_ENABLED", "1")
    monkeypatch.setenv("DEDUP_META_FIELDS", f"title,{field}")
    fake = FakeAlbert()
    texts = [f"Passage déposé deux fois numéro {i}, sous deux noms de fichier, assez long pour la dédup."
             for i in range(3)]
    first = make_chunks(3, item_key="ITEMA001", filename="a.pdf", texts=texts)
    assert all(field not in chunk and source_key in chunk for chunk in first)
    run1 = tmp_path / "run1"
    run1.mkdir()
    assert push(write_chunks(run1, first), fake.transport)["inserted_count"] == 3
    (did,) = fake.documents
    assert {c["metadata"][field] for c in fake.chunks[did]} == {stored}

    # Même source déposée sous un autre fichier (autre document Albert) : refus Tier 2, 3 chunks.
    run2 = tmp_path / "run2"
    run2.mkdir()
    same = make_chunks(3, item_key="ITEMA001", filename="a-copie.pdf", texts=texts)
    assert col.document_name_for(same[0]) != col.document_name_for(first[0])
    path2 = write_chunks(run2, same)
    fake.reset_calls()
    res = push(path2, fake.transport)
    assert res["status"] == "success"
    assert res.get("skipped_count", 0) == 3 and res["inserted_count"] == 0 and res["existing_count"] == 0
    assert writes(fake) == []
    journal = [json.loads(line) for line in Path(res["journal_path"]).read_text(encoding="utf-8").splitlines()]
    assert len(journal) == 3
    assert {entry["reason"] for entry in journal} == {"existing_hash_match"}
    for entry in journal:
        assert all({"title", field} <= set(m["metadata_match"]) for m in entry["matched_existing"])
    # Seules les copies en mémoire reçoivent le champ dérivé : le fichier d'entrée est intact.
    on_disk = json.loads(Path(path2).read_text(encoding="utf-8"))
    assert all(field not in chunk for chunk in on_disk)

    # Valeur dérivée différente (même texte, même titre) : pas de corroboration, les chunks partent.
    run3 = tmp_path / "run3"
    run3.mkdir()
    other = make_chunks(3, item_key="ITEMA001", filename="a-autre.pdf", texts=texts)
    for chunk in other:
        chunk[source_key] = other_source
    fake.reset_calls()
    res = push(write_chunks(run3, other), fake.transport)
    assert res["status"] == "success"
    assert res["inserted_count"] == 3 and res.get("skipped_count", 0) == 0


# ---------------------------------------------------------------------------
# CLI : bloc Result, lignes Albert, acquittement, argparse
# ---------------------------------------------------------------------------
def test_stdout_extra_lines_albert_only(tmp_path, monkeypatch, capsys, cli_fake):
    path = write_chunks(tmp_path, make_chunks(3))
    args = ["--input", path, "--db", "albert", "--albert-collection-name", COLLECTION,
            "--albert-create-collection", "--albert-ack-retention"]
    code, lines = run_cli(monkeypatch, capsys, args)
    assert code == 0
    block = result_block(lines)
    manifest = col.manifest_path_for(str(tmp_path))
    assert block[1] == "Status: success"
    assert block[2].startswith("Message: ")
    assert block[3:] == ["Inserted: 3", "Skipped (dedup): 0", "Skipped (existing): 0",
                         f"Albert manifest: {manifest}"]
    code, lines = run_cli(monkeypatch, capsys, args)
    assert code == 0
    assert result_block(lines)[3:] == ["Inserted: 0", "Skipped (dedup): 0", "Skipped (existing): 3"]
    assert all(FAKE_ALBERT_KEY not in line for line in lines)

    # Les 3 autres bases (Albert ON dans l'env) n'émettent jamais les lignes Albert.
    monkeypatch.setenv("PINECONE_API_KEY", "fake-pinecone-0001")
    monkeypatch.setenv("WEAVIATE_URL", "fake-weaviate-url-0001")
    monkeypatch.setenv("WEAVIATE_API_KEY", "fake-weaviate-0001")
    monkeypatch.setenv("QDRANT_URL", "fake-qdrant-url-0001")
    monkeypatch.setitem(sys.modules, "pinecone", golden_harness._fake_pinecone_module("dotproduct", False))
    for name, module in golden_harness._fake_weaviate_modules(["golden-tenant"]).items():
        monkeypatch.setitem(sys.modules, name, module)
    for name, module in golden_harness._fake_qdrant_modules(True, ["completed"]).items():
        monkeypatch.setitem(sys.modules, name, module)
    other_input = write_chunks(tmp_path, golden_harness._vectordb_chunks(False), name="other_sparse.json")
    for db_args in (["--db", "pinecone", "--index", "golden-index"],
                    ["--db", "weaviate", "--class-name", "GoldenArticle", "--tenant", "golden-tenant"],
                    ["--db", "qdrant", "--collection", "golden-collection"]):
        code, lines = run_cli(monkeypatch, capsys, ["--input", other_input] + db_args)
        block = result_block(lines)
        assert code == 0, db_args
        assert [line.split(":", 1)[0] for line in block[1:]] == ["Status", "Message", "Inserted", "Skipped (dedup)"]
        assert not any(line.startswith(("Skipped (existing)", "Albert manifest")) for line in lines)


def test_cli_requires_ack_retention(tmp_path, monkeypatch, capsys, cli_fake):
    path = write_chunks(tmp_path, make_chunks(3))
    code, lines = run_cli(monkeypatch, capsys, ["--input", path, "--db", "albert",
                                                "--albert-collection-name", COLLECTION,
                                                "--albert-create-collection"])
    assert code == 1
    assert any("--albert-ack-retention" in line for line in lines)
    assert cli_fake.built == [] and cli_fake.fake.calls == []
    assert not os.path.exists(col.manifest_path_for(str(tmp_path)))
    # Même règle côté bibliothèque : refus avant tout appel réseau.
    res = push(path, cli_fake.fake.transport, ack=False)
    assert res["status"] == "error"
    assert cli_fake.fake.calls == []


def test_cli_albert_disabled_exit_2_no_network(tmp_path, monkeypatch, capsys, cli_fake):
    monkeypatch.setenv("ALBERT_ENABLED", "0")
    path = write_chunks(tmp_path, make_chunks(3))
    code, lines = run_cli(monkeypatch, capsys, ["--input", path, "--db", "albert",
                                                "--albert-collection-name", COLLECTION,
                                                "--albert-ack-retention"])
    assert code == 2
    assert cli_fake.built == [] and cli_fake.fake.calls == []


def test_cli_account_error_exit_2(tmp_path, monkeypatch, capsys, cli_fake):
    cli_fake.fake.inject("GET", "/v1/collections", "invalid_key")
    path = write_chunks(tmp_path, make_chunks(3))
    code, lines = run_cli(monkeypatch, capsys, ["--input", path, "--db", "albert",
                                                "--albert-collection-name", COLLECTION,
                                                "--albert-create-collection", "--albert-ack-retention"])
    assert code == 2
    block = result_block(lines)
    assert block[1] == "Status: error"
    assert writes(cli_fake.fake) == []
    assert all(FAKE_ALBERT_KEY not in line for line in lines)


def test_argparse_class_prefix_still_resolves(tmp_path, monkeypatch, capsys):
    code, lines = run_cli(monkeypatch, capsys, ["--help"])
    assert code == 0
    albert_options = [line for line in lines if line.startswith("  --albert-")]
    assert len(albert_options) == 4
    option_names = {line.split()[0] for line in albert_options}
    assert option_names == {"--albert-collection-id", "--albert-collection-name",
                            "--albert-create-collection", "--albert-ack-retention"}
    for prefix in ("--class", "--tenant", "--index", "--namespace", "--collection"):
        assert not any(option.startswith(prefix) for option in option_names)
    # Abréviation utilisée par les routes et Celery : « --class » -> « --class-name ».
    monkeypatch.setenv("WEAVIATE_URL", "fake-weaviate-url-0001")
    monkeypatch.setenv("WEAVIATE_API_KEY", "fake-weaviate-0001")
    for name, module in golden_harness._fake_weaviate_modules(["golden-tenant"]).items():
        monkeypatch.setitem(sys.modules, name, module)
    other_input = write_chunks(tmp_path, golden_harness._vectordb_chunks(False), name="other_sparse.json")
    code, lines = run_cli(monkeypatch, capsys, ["--input", other_input, "--db", "weaviate",
                                                "--class", "GoldenArticle", "--tenant", "golden-tenant"])
    assert code == 0
    assert "Class: GoldenArticle" in lines


def test_pinecone_result_block_equals_golden(tmp_path, monkeypatch, capsys):
    # G5 : bloc Result Pinecone identique au golden, Albert OFF puis Albert ON.
    with open(os.path.join(golden_harness.GOLDEN_DIR, "g5_vectordb_pinecone_result_block.json"), "rb") as fh:
        expected = fh.read()
    for albert_env in ({}, {"ALBERT_ENABLED": "1", "ALBERT_API_KEY": FAKE_ALBERT_KEY}):
        monkeypatch.delenv("ALBERT_ENABLED", raising=False)
        monkeypatch.delenv("ALBERT_API_KEY", raising=False)
        for name, value in albert_env.items():
            monkeypatch.setenv(name, value)
        golden_harness._set_fake_env(monkeypatch, "PINECONE_API_KEY", "WEAVIATE_URL", "WEAVIATE_API_KEY",
                                     "QDRANT_URL", "QDRANT_API_KEY")
        monkeypatch.setenv("PINECONE_BATCH_SIZE", "100")
        results = []
        for case, metric, fail_upsert, missing, namespace in golden_harness.G5_PINECONE_CASES:
            monkeypatch.setitem(sys.modules, "pinecone", golden_harness._fake_pinecone_module(metric, fail_upsert))
            db_args = ["--db", "pinecone", "--index", "golden-index"]
            if namespace:
                db_args += ["--namespace", namespace]
            results.append(golden_harness._run_vectordb_cli(monkeypatch, capsys, tmp_path, case, missing, db_args))
        assert golden_harness._dump(results).encode("utf-8") == expected


def test_import_is_lazy_off_path():
    # Albert OFF : importer rad_vectordb ne charge ni le paquet rad_albert ni son client httpx.
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import scripts.rad_vectordb; "
            "print(sorted(m for m in sys.modules if 'rad_albert' in m.split('.')))")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ALBERT_", "DEDUP_"))}
    proc = subprocess.run([sys.executable, "-c", code, str(RAGPY_ROOT)], cwd=str(RAGPY_ROOT), env=env,
                          capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines()[-1] == "[]"
