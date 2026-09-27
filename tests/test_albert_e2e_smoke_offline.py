"""Tests hors ligne des aides de ``scripts/albert_e2e_smoke.py`` (aucun serveur, aucun appel réseau).

Le test de bout en bout lui-même appelle l'API Albert de production et ne
tourne qu'à la main (``ALBERT_LIVE=1``). Ces tests couvrent ses parties pures :

* le corpus (3 PDF valides, dont un PDF image seule sans couche texte) et
  l'export Zotero JSON ;
* l'archive, acceptée par la vraie route ``upload_zip`` (harnais des goldens,
  base temporaire) puis par le chargeur Zotero de ``rad_dataframe`` (OCR
  factice) ;
* les refus (code 2) sans ``ALBERT_LIVE=1``, avec ``ALBERT_LIVE`` dans le
  ``.env``, sans clé, ou quand ``python3`` ne se résout pas dans ``.venv/bin`` ;
* le format de la dernière ligne ;
* le balayage des collections, limité aux noms de ce script
  (``ragpy-probe-e2e-AAAAMMJJ-HHMMSS``, jamais ceux de la suite live), sur
  l'API simulée ``FakeAlbert`` ;
* les aides de contrôle (SSE, marqueurs de pages, embeddings, ``recode_model``,
  notes, masquage de la clé, recherche de la clé dans les journaux) et de
  nettoyage (sessions sous ``uploads/``, copies de manifeste) ;
* les contrôles on1, on2 et on6 sur des sorties fabriquées (API factice) :
  OCR partiel du numérisé, recodage retombé sur le texte brut, ledger des
  notes exigé (enregistrement servi par l'id épinglé de gpt-oss avec un coût
  numérique, journal écrit juste après l'événement ``complete``) ;
* le déroulé du run : nettoyage et résumé après Ctrl-C, exception, SIGTERM ou
  SIGHUP (signaux ignorés pendant le nettoyage, gestionnaires restaurés),
  étapes de nettoyage isolées, serveurs arrêtés avant les collections, arrêt
  du groupe de processus du serveur, refus d'une ``DATABASE_URL`` déjà liée.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import signal
import sys
import time
import zipfile
from datetime import date

import fitz
import pytest

from scripts import albert_e2e_smoke as smoke
from scripts.rad_albert.client import AlbertClient
from scripts.rad_albert.config import AlbertConfig
from tests.albert_fakes import FAKE_ALBERT_KEY, FAKE_BASE_URL, FakeAlbert
# Harnais des goldens réutilisé (l'hygiène d'env est autouse dans ce module).
from tests.test_albert_off_golden_routes import _golden_env_hygiene, golden_app  # noqa: F401

STAMP = "20260927-101500"
TODAY = date(2026, 9, 27)


# ---------------------------------------------------------------------------
# Aides
# ---------------------------------------------------------------------------
def _corpus(tmp_path, size=3):
    """Corpus de test construit sous ``tmp_path``."""
    return smoke.build_corpus(tmp_path / "corpus", size, STAMP)


def _env_file(tmp_path, text):
    """Fichier ``.env`` factice."""
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


def _executable(folder, name="python3"):
    """Exécutable factice ``name`` dans ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _no_sleep(_seconds):
    """Sommeil factice : aucune attente réelle."""


def _silent(_message):
    """Affichage factice : rien n'est écrit."""


def _client(fake):
    """Client Albert branché sur l'API simulée, sans limiteur."""
    return AlbertClient(dataclasses.replace(AlbertConfig(enabled=True)), FAKE_ALBERT_KEY,
                        transport=fake.transport, use_limiter=False, sleep=_no_sleep)


@pytest.fixture
def no_run(monkeypatch):
    """Fait échouer le test si ``main`` passe les gardes (aucun run ne doit démarrer)."""

    def forbidden(*_args, **_kwargs):
        """Toute tentative de run ou de dossier temporaire est une erreur."""
        raise AssertionError("le run a démarré malgré un refus")

    monkeypatch.setattr(smoke, "SmokeRun", forbidden)
    monkeypatch.setattr(smoke.tempfile, "mkdtemp", forbidden)


# ---------------------------------------------------------------------------
# Corpus et archive
# ---------------------------------------------------------------------------
def test_corpus_builder_three_valid_pdfs_one_image_only(tmp_path):
    """Trois PDF valides : deux avec couche texte, un image seule ; export Zotero cohérent."""
    corpus = _corpus(tmp_path)
    assert [d.kind for d in corpus.docs] == ["text", "text", "scan"]
    assert len(corpus.scanned) == 1
    for doc in corpus.docs:
        path = corpus.path_of(doc)
        assert path.read_bytes().startswith(b"%PDF-")
        with fitz.open(str(path)) as pdf:
            assert pdf.page_count == doc.pages == smoke.PAGES_PER_DOC
            texts = [page.get_text().strip() for page in pdf]
            images = [len(page.get_images()) for page in pdf]
        if doc.kind == "scan":
            assert texts == [""] * doc.pages  # aucune couche texte
            assert all(count >= 1 for count in images)
        else:
            assert all(STAMP in text for text in texts)  # contenu propre au run
            assert images == [0] * doc.pages
    items = json.loads((corpus.root / corpus.json_name).read_text(encoding="utf-8"))
    assert [item["key"] for item in items] == [doc.key for doc in corpus.docs]
    for item, doc in zip(items, corpus.docs):
        assert item["attachments"][0]["path"] == doc.relpath
        assert (corpus.root / item["attachments"][0]["path"]).is_file()
    # Rien qui contienne « albert » : les pages OFF affichent projet et session.
    assert "albert" not in json.dumps(items, ensure_ascii=False).lower()
    assert "albert" not in (smoke.CORPUS_PREFIX + corpus.json_name).lower()
    with pytest.raises(ValueError):
        smoke.build_corpus(tmp_path / "petit", 1, STAMP)


def test_corpus_size_option_puts_the_scan_last(tmp_path):
    """``--corpus-size N`` : N-1 PDF texte puis le numérisé ; 2 documents au moins."""
    corpus = _corpus(tmp_path, size=5)
    assert [d.kind for d in corpus.docs] == ["text"] * 4 + ["scan"]
    assert len({d.key for d in corpus.docs}) == 5
    assert smoke.parse_args(["--corpus-size", "5", "--cleanup"]).corpus_size == 5
    assert smoke.parse_args([]).corpus_size == smoke.DEFAULT_CORPUS_SIZE
    with pytest.raises(SystemExit):
        smoke.parse_args(["--corpus-size", "1"])


def test_zip_layout_accepted_by_upload_route_and_zotero_loader(golden_app, tmp_path, monkeypatch):
    """L'archive passe par la vraie route ``upload_zip`` puis par le chargeur Zotero de ``rad_dataframe``."""
    from scripts import rad_dataframe as rad

    corpus = _corpus(tmp_path)
    stem = smoke.CORPUS_PREFIX + STAMP
    archive = smoke.zip_corpus(corpus, tmp_path / f"{stem}.zip")
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    assert names[0] == corpus.json_name
    assert sorted(names[1:]) == sorted(doc.relpath for doc in corpus.docs)

    env = golden_app
    resp = env.client.post(
        f"/api/pipeline/projects/{env.project.id}/upload_zip",
        files={"file": (archive.name, archive.read_bytes(), "application/zip")},
        headers=env.headers["member_nokeys"],  # propriétaire du projet
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Deux entrées à la racine de l'archive : la session est le dossier lui-même.
    assert "/" not in body["path"]
    processing = env.uploads / body["path"]
    assert corpus.json_name in body["tree"]

    # Nommage de la route reconnu par le nettoyage (dossier et archive).
    entries = smoke.upload_entries_for(env.uploads, stem)
    assert sorted(p.name for p in entries) == sorted([body["path"], body["path"] + ".zip"])

    # Sélection du JSON Zotero, comme process_dataframe_sse.
    json_files = [f for f in os.listdir(processing)
                  if f.lower().endswith(".json") and not f.startswith(smoke.EXCLUDED_JSON_PREFIXES)]
    assert json_files == [corpus.json_name]

    def fake_ocr(pdf_path, max_pages=None, *, return_details=False, max_retries=0):
        """OCR factice : un marqueur par page, fournisseur legacy."""
        with fitz.open(pdf_path) as pdf:
            pages = pdf.page_count
        text = "\n".join(f"<!-- Page {n} -->\ntexte {os.path.basename(pdf_path)}" for n in range(1, pages + 1))
        result = rad.OCRResult(text=text, provider="legacy", pages_done=pages, pages_total=pages)
        return result if return_details else text

    monkeypatch.setattr(rad, "extract_text_with_ocr_retry", fake_ocr)
    out_csv = processing / "output.csv"
    rad.load_zotero_to_dataframe_incremental(str(processing / json_files[0]), str(processing), str(out_csv))
    rows = smoke.read_output_csv(out_csv)
    assert sorted(row["filename"] for row in rows) == sorted(doc.filename for doc in corpus.docs)
    for row in rows:
        assert os.path.isfile(row["path"])
        assert smoke.markers_continuous(smoke.page_markers(row["texteocr"]), smoke.PAGES_PER_DOC)
    errors = json.loads((processing / "output_errors.json").read_text(encoding="utf-8"))
    assert errors["total_errors"] == 0

    assert sorted(smoke.remove_upload_entries(env.uploads, entries)) == sorted(p.name for p in entries)
    assert smoke.upload_entries_for(env.uploads, stem) == []


def test_minimal_output_csv_roundtrip(tmp_path):
    """``output.csv`` minimal : relu au format de ``rad_dataframe``, jamais écrasé."""
    path = smoke.write_minimal_output_csv(tmp_path / "session" / "output.csv")
    rows = smoke.read_output_csv(path)
    assert len(rows) == 1
    assert rows[0]["texteocr_provider"] == "legacy"
    assert smoke.page_markers(rows[0]["texteocr"]) == [1]
    path.write_text("existant", encoding="utf-8")
    smoke.write_minimal_output_csv(path)
    assert path.read_text(encoding="utf-8") == "existant"  # jamais écrasé


# ---------------------------------------------------------------------------
# Refus (code 2) et environnement du serveur
# ---------------------------------------------------------------------------
def test_refusal_without_albert_live(tmp_path, no_run, capsys):
    """Sans ``ALBERT_LIVE=1`` dans le shell : code 2, aucun run, clé jamais affichée."""
    env_file = _env_file(tmp_path, f"ALBERT_API_KEY={FAKE_ALBERT_KEY}\n")
    for environ in ({}, {"ALBERT_LIVE": "0"}, {"ALBERT_LIVE": "true"}):
        assert smoke.main([], environ=environ, env_file=env_file) == 2
    err = capsys.readouterr().err
    assert "ALBERT_LIVE=1" in err
    assert FAKE_ALBERT_KEY not in err


def test_refusal_when_albert_live_in_env_file(tmp_path, no_run, capsys):
    """``ALBERT_LIVE`` dans le ``.env`` : code 2, le fichier est nommé."""
    env_file = _env_file(tmp_path, f"ALBERT_LIVE=1\nALBERT_API_KEY={FAKE_ALBERT_KEY}\n")
    assert smoke.main(["--cleanup"], environ={"ALBERT_LIVE": "1"}, env_file=env_file) == 2
    err = capsys.readouterr().err
    assert str(env_file) in err
    assert FAKE_ALBERT_KEY not in err


def test_refusal_without_valid_key_in_env_file(tmp_path, no_run, capsys):
    """Clé absente du ``.env`` (même présente dans le shell) ou mal formée : code 2."""
    env_file = _env_file(tmp_path, "AUTRE=1\n")
    assert smoke.main([], environ={"ALBERT_LIVE": "1", "ALBERT_API_KEY": FAKE_ALBERT_KEY}, env_file=env_file) == 2
    assert "ALBERT_API_KEY absente" in capsys.readouterr().err
    _env_file(tmp_path, "ALBERT_API_KEY=court\n")
    assert smoke.main([], environ={"ALBERT_LIVE": "1"}, env_file=env_file) == 2
    assert "mal formée" in capsys.readouterr().err


def test_refusal_when_python3_is_outside_the_venv(tmp_path, monkeypatch, no_run, capsys):
    """``python3`` du PATH du serveur hors de ``.venv/bin`` : code 2."""
    env_file = _env_file(tmp_path, f"ALBERT_API_KEY={FAKE_ALBERT_KEY}\n")
    venv_bin = tmp_path / "depot" / ".venv" / "bin"
    venv_bin.mkdir(parents=True)  # sans python3
    other = tmp_path / "systeme"
    _executable(other)
    monkeypatch.setattr(smoke, "VENV_BIN", venv_bin)
    assert smoke.main([], environ={"ALBERT_LIVE": "1", "PATH": str(other)}, env_file=env_file) == 2
    err = capsys.readouterr().err
    assert "python3" in err and str(venv_bin) in err
    assert FAKE_ALBERT_KEY not in err


def test_venv_python_check(tmp_path):
    """``.venv/bin`` en tête du PATH du serveur et résolution de ``python3``."""
    venv_bin = tmp_path / "venv" / "bin"
    other = tmp_path / "systeme"
    _executable(venv_bin)
    _executable(other)
    server_path = smoke.build_server_path(str(other), venv_bin)
    assert server_path.split(os.pathsep) == [str(venv_bin), str(other)]
    assert smoke.build_server_path("", venv_bin) == str(venv_bin)
    assert smoke.venv_python_problem(server_path, venv_bin) is None
    assert "n'est pas celui" in smoke.venv_python_problem(str(other), venv_bin)
    assert "introuvable" in smoke.venv_python_problem(str(tmp_path / "vide"), venv_bin)


def test_server_env_on_off(tmp_path, monkeypatch):
    """Environnement du serveur : interrupteurs ON/OFF, base, secret JWT, défauts forcés, jamais ``ALBERT_LIVE``."""
    monkeypatch.setenv("ALBERT_LIVE", "1")
    monkeypatch.setenv("DEDUP_ENABLED", "1")
    run = smoke.SmokeRun(smoke.parse_args([]), FAKE_ALBERT_KEY, smoke.build_server_path("/usr/bin"),
                         env_file=tmp_path / ".env", log=_silent)
    run.db_url = f"sqlite:///{tmp_path / 'e2e.db'}"
    run.jwt_secret = "fake-jwt-secret-e2e-0001"
    on, off = run.server_env(True), run.server_env(False)
    assert "ALBERT_LIVE" not in on and "ALBERT_LIVE" not in off
    assert on["PATH"].split(os.pathsep)[0] == str(smoke.VENV_BIN)
    assert (on["ALBERT_ENABLED"], off["ALBERT_ENABLED"]) == ("1", "0")
    assert on["OCR_ENABLE_ALBERT"] == "1"
    assert on["DATABASE_URL"] == run.db_url and on["JWT_SECRET_KEY"] == run.jwt_secret
    assert all(on[name] == value for name, value in smoke.PINNED_ENV.items())
    assert on["ALBERT_OCR_SKIP_RECODE"] == "0" and on["ALBERT_USAGE_LOG"] == "1"
    assert on["DEDUP_ENABLED"] == "0"
    assert run.collection_name.startswith(smoke.COLLECTION_PREFIX)
    assert smoke.is_smoke_collection_name(run.collection_name)
    assert "albert" not in run.zip_stem.lower()


# ---------------------------------------------------------------------------
# Dernière ligne
# ---------------------------------------------------------------------------
def test_summary_line_format():
    """Format exact de la dernière ligne (OK, ÉCHEC, anomalies) et nombre de contrôles par groupe."""
    assert smoke.summary_line(7, 3, 2) == "E2E ALBERT OK (ON 7/7, OFF 3/3, isolation 2/2)"
    assert smoke.summary_line(5, 3, 1) == "E2E ALBERT ÉCHEC (ON 5/7, OFF 3/3, isolation 1/2)"
    assert smoke.summary_line(7, 3, 2, ["collections restantes : 1"]) == (
        "E2E ALBERT ÉCHEC (ON 7/7, OFF 3/3, isolation 2/2 ; collections restantes : 1)")
    assert smoke.summary_line(7, 3, 2, ["", "  "]) == "E2E ALBERT OK (ON 7/7, OFF 3/3, isolation 2/2)"
    assert [(group, len(checks)) for group, checks in smoke.CHECK_GROUPS] == [
        ("on", 7), ("isolation", 2), ("off", 3)]


def test_unexecuted_checks_count_as_failures(tmp_path):
    """Seuls les contrôles réussis comptent ; message et détail d'un échec masqués."""
    lines = []
    run = smoke.SmokeRun(smoke.parse_args([]), FAKE_ALBERT_KEY, "/usr/bin", env_file=tmp_path / ".env",
                         log=lines.append)

    def passed():
        """Contrôle réussi."""
        return {"ok": 1}

    run.run_check("on", "on1", passed)

    def refused():
        """Contrôle en échec dont le message cite la clé."""
        raise smoke.CheckFailed(f"refus {FAKE_ALBERT_KEY}", {"cle": FAKE_ALBERT_KEY})

    run.run_check("on", "on2", refused)
    assert run.counts() == {"on": 1, "isolation": 0, "off": 0}
    assert all(FAKE_ALBERT_KEY not in line for line in lines)
    assert run.results["on"]["on2"]["message"] == "refus [masqué]"
    assert run.results["on"]["on2"]["detail"] == {"cle": "[masqué]"}


# ---------------------------------------------------------------------------
# Collections (FakeAlbert)
# ---------------------------------------------------------------------------
def test_cleanup_sweep_only_deletes_this_script_collections():
    """Le balayage ne supprime que les collections privées de ce script, jamais celles de la suite live."""
    fake = FakeAlbert()
    client = _client(fake)
    e2e = [smoke.COLLECTION_PREFIX + STAMP, smoke.COLLECTION_PREFIX + "20260926-080000"]
    live = smoke.COLLECTION_PREFIX + "20260927T101500-4242"  # nom de tests/live/test_albert_live_e2e.py
    others = ["ragpy-probe-20260927-101500", "mes-articles", "copie-" + smoke.COLLECTION_PREFIX + STAMP,
              "RAGPY-PROBE-E2E-MAJUSCULES", live, smoke.COLLECTION_PREFIX + STAMP + "-bis"]
    for name in e2e + others:
        client.create_collection(name)
    public = fake.client(FAKE_BASE_URL).post(
        "/collections", json={"name": smoke.COLLECTION_PREFIX + "publique", "visibility": "public"})
    assert public.status_code == 201
    deleted = smoke.sweep_collections(client)
    assert sorted(name for _cid, name in deleted) == sorted(e2e)
    remaining = sorted(c["name"] for c in client.list_collections())
    assert remaining == sorted(others + [smoke.COLLECTION_PREFIX + "publique"])
    assert len(fake.calls_to("DELETE")) == 2
    assert smoke.sweep_collections(client) == []  # rien d'autre n'est jamais balayé
    assert len(fake.calls_to("DELETE")) == 2
    assert [smoke.is_smoke_collection_name(n) for n in e2e] == [True, True]
    assert not any(smoke.is_smoke_collection_name(n) for n in others + [None, 3])


def test_delete_collections_named_exact_match_only():
    """Suppression par nom exact, doublons compris, jamais un nom voisin."""
    fake = FakeAlbert()
    client = _client(fake)
    name = smoke.COLLECTION_PREFIX + STAMP
    ids = [client.create_collection(name), client.create_collection(name)]
    keep = client.create_collection(name + "-bis")
    assert sorted(smoke.delete_collections_named(client, name)) == sorted(ids)
    assert smoke.count_collections_named(client, name) == 0
    assert [c["id"] for c in client.list_collections()] == [keep]


# ---------------------------------------------------------------------------
# Nettoyage : sessions, manifestes, recherche de la clé
# ---------------------------------------------------------------------------
def test_upload_entries_match_only_the_run_sessions(tmp_path):
    """Nettoyage de ``uploads/`` limité aux sessions et archives du run."""
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    stem = smoke.CORPUS_PREFIX + STAMP
    mine = ["0a1b2c3d_" + stem, "0a1b2c3d_" + stem + ".zip", "ffffffff_" + stem]
    theirs = ["0a1b2c3d_" + stem + "-autre", "ABCDEF12_" + stem, "12345678_mes-articles", stem,
              "0a1b2c3d_" + stem + ".zip.bak"]
    for name in mine + theirs:
        target = uploads / name
        if ".zip" in name:
            target.write_bytes(b"PK")
        else:
            target.mkdir()
            (target / "output.csv").write_text("x", encoding="utf-8")
    found = smoke.upload_entries_for(uploads, stem)
    assert sorted(p.name for p in found) == sorted(mine)
    assert sorted(smoke.remove_upload_entries(uploads, found)) == sorted(mine)
    assert sorted(p.name for p in uploads.iterdir()) == sorted(theirs)
    outside = tmp_path / ("0a1b2c3d_" + stem)
    outside.mkdir()
    assert smoke.remove_upload_entries(uploads, [outside]) == []
    assert outside.exists()
    assert smoke.upload_entries_for(tmp_path / "absent", stem) == []


def test_manifest_copies_only_new_files_of_the_session(tmp_path):
    """Copies de manifeste du run seulement ; libellé identique à celui de la route."""
    from app.routes import processing

    session = "0a1b2c3d_" + smoke.CORPUS_PREFIX + STAMP
    for sample in (session, "abc/def ghi", "../x", "", ".cache"):
        assert smoke.manifest_label(sample) == processing._safe_label(sample)
    folder = tmp_path / "7"
    folder.mkdir()
    label = smoke.manifest_label(session)
    old = folder / f"{label}-20260101T000000000000Z.jsonl"
    new = folder / f"{label}-20260927T101500000000Z.jsonl"
    for path in (old, new, folder / "autre-session-20260927T101500000000Z.jsonl", folder / f"{label}-notes.txt"):
        path.write_text("{}", encoding="utf-8")
    assert smoke.manifest_copies(folder, session, before={old.name}) == [new]
    assert smoke.manifest_copies(tmp_path / "absent", session) == []


def test_files_containing_secret_after_offset(tmp_path):
    """Recherche de la clé après un offset, journal tourné, bloc à cheval, fichier absent."""
    log = tmp_path / "app.log"
    log.write_text(f"ancienne ligne {FAKE_ALBERT_KEY}\n", encoding="utf-8")
    offset = log.stat().st_size
    with open(log, "a", encoding="utf-8") as handle:
        handle.write("nouvelle ligne sans secret\n")
    assert smoke.files_containing([(log, offset)], FAKE_ALBERT_KEY) == []
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(f"fuite {FAKE_ALBERT_KEY}\n")
    assert smoke.files_containing([(log, offset)], FAKE_ALBERT_KEY) == [str(log)]
    rotated = tmp_path / "tourne.log"
    rotated.write_text(f"{FAKE_ALBERT_KEY}\n", encoding="utf-8")
    assert smoke.files_containing([(rotated, 10 ** 9)], FAKE_ALBERT_KEY) == [str(rotated)]
    boundary = tmp_path / "limite.log"
    boundary.write_bytes(b"x" * ((1 << 20) - 5) + FAKE_ALBERT_KEY.encode() + b"\n")
    assert smoke.files_containing([(boundary, 0)], FAKE_ALBERT_KEY) == [str(boundary)]
    assert smoke.files_containing([(tmp_path / "absent.log", 0)], FAKE_ALBERT_KEY) == []
    assert smoke.files_containing([(log, 0)], "") == []


def test_redact_masks_the_key_everywhere():
    """Masquage récursif de la clé (valeurs, clés, listes, tuples)."""
    payload = {"message": f"clé {FAKE_ALBERT_KEY} refusée",
               FAKE_ALBERT_KEY: [FAKE_ALBERT_KEY, 3, None, ("x", FAKE_ALBERT_KEY)]}
    out = smoke.redact(payload, [FAKE_ALBERT_KEY, "", "ab"])
    assert FAKE_ALBERT_KEY not in json.dumps(out, ensure_ascii=False)
    assert out["message"] == "clé [masqué] refusée"
    assert smoke.redact("abc", ["ab"]) == "abc"  # secret trop court : ignoré


# ---------------------------------------------------------------------------
# Aides de contrôle
# ---------------------------------------------------------------------------
def test_sse_parsing_and_terminal_event():
    """Lecture SSE : lignes ignorées, JSON invalide toléré, dernier événement complete ou error."""
    lines = [
        'data: {"type": "init", "total": 3}',
        "",
        ": commentaire",
        'data: {"type": "error", "message": "ligne de journal error"}',
        'data: {"type": "heartbeat", "message": "Processing..."}',
        'data: {"type": "error", "message": "texte "cité" invalide"}',
        b'data: {"type": "complete", "message": "Process completed successfully", "count": 3}',
    ]
    events = list(smoke.iter_sse_events(lines))
    assert [e["type"] for e in events] == ["init", "error", "heartbeat", "error", "complete"]
    assert events[3]["brut"] is True
    assert smoke.terminal_event(events)["count"] == 3
    assert smoke.terminal_event(events[:-1])["message"].startswith('{"type": "error"')
    assert smoke.terminal_event(events[:1]) is None


def test_require_complete_outcomes():
    """Issue d'un flux : seul ``complete`` passe ; ``credential_required`` conservé."""
    done = smoke.SseOutcome(200, [{"type": "complete"}], {"type": "complete", "count": 2})
    assert smoke.require_complete(done)["count"] == 2
    failed = smoke.SseOutcome(
        200, [{"type": "error", "message": "ligne"}],
        {"type": "error", "message": "Clé requise", "credential_required": "albert_api_key"})
    with pytest.raises(smoke.CheckFailed) as info:
        smoke.require_complete(failed)
    assert info.value.detail["credential_required"] == "albert_api_key"
    for outcome in (smoke.SseOutcome(401, [], None, "Not authenticated"),
                    smoke.SseOutcome(200, [], None),
                    smoke.SseOutcome(200, [], None, "délai", timed_out=True)):
        with pytest.raises(smoke.CheckFailed):
            smoke.require_complete(outcome)


def test_page_markers_continuity():
    """Marqueurs ``<!-- Page N -->`` : extraction et continuité."""
    text = ("<!-- Page 1 -->\nA\n<!--Page 2-->\nB <!-- OCR ÉCHOUÉ (albert_lightonocr) : délai -->\n"
            "<!-- Part 1/1 (pages 1-2) -->")
    assert smoke.page_markers(text) == [1, 2]
    assert smoke.markers_continuous([1, 2], 2)
    assert smoke.markers_continuous([1, 2, 3])
    assert not smoke.markers_continuous([1, 2], 3)
    assert not smoke.markers_continuous([1, 3])
    assert not smoke.markers_continuous([2, 1])
    assert not smoke.markers_continuous([])


def test_embedding_problems():
    """Défauts des embeddings : dimension, longueur, vecteur nul, valeur non finie, champ absent."""
    good = {"id": "ok", "embedding_dim": 1024, "embedding": [0.0] * 1023 + [1.0]}
    assert smoke.embedding_problems([good]) == []
    bad = [
        {"id": "dim", "embedding_dim": 3072, "embedding": [0.1] * 1024},
        {"id": "len", "embedding_dim": 1024, "embedding": [0.1] * 3},
        {"id": "nul", "embedding_dim": 1024, "embedding": [0.0] * 1024},
        {"id": "nan", "embedding_dim": 1024, "embedding": [float("nan")] + [0.1] * 1023},
        {"id": "bool", "embedding_dim": True, "embedding": [0.1] * 1024},
        {"id": "absent"},
    ]
    problems = smoke.embedding_problems(bad)
    for label in ("dim", "len", "nul", "nan", "bool", "absent"):
        assert any(p.startswith(label + " ") for p in problems), label
    assert smoke.embedding_problems([]) == ["aucun chunk"]


def test_recode_model_problem():
    """``recode_model`` : préfixe, id épinglé (pas d'alias), chaîne du rôle recode datée."""
    assert smoke.recode_model_problem("albert/ministral-3-8b-instruct-2512", today=TODAY) is None
    # repli du rôle recode avant son retrait du 2026-12-01
    assert smoke.recode_model_problem("albert/mistral-small-3-2-24b-instruct-2506", today=TODAY) is None
    assert smoke.recode_model_problem("albert/mistral-small-3-2-24b-instruct-2506", today=date(2026, 12, 1))
    assert "alias" in smoke.recode_model_problem("albert/openweight-small", today=TODAY)
    assert "préfixe" in smoke.recode_model_problem("ministral-3-8b-instruct-2512", today=TODAY)
    assert "chaîne" in smoke.recode_model_problem("albert/gpt-oss-120b", today=TODAY)
    assert "inconnu" in smoke.recode_model_problem("albert/modele-imaginaire", today=TODAY)
    assert smoke.recode_model_problem("albert/", today=TODAY)
    assert smoke.recode_model_problem(None, today=TODAY)


def test_note_problem():
    """Note courte : ni vide, ni gabarit de repli, ni copie du résumé Zotero."""
    abstract = "Document de contrôle 1 du corpus de bout en bout (20260927-101500)."
    generated = ("Cet article analyse la transformation des pratiques de lecture savante "
                 "par la numérisation des bibliothèques universitaires.")
    assert smoke.note_problem(generated, abstract) is None
    assert "vide" in smoke.note_problem("", abstract)
    assert "vide" in smoke.note_problem("court", abstract)
    assert "vide" in smoke.note_problem(None)
    assert "identique" in smoke.note_problem(abstract, abstract)
    assert "gabarit" in smoke.note_problem("Fiche de lecture. Problématique : à compléter. " * 2, "")


def test_finish_writes_redacted_summary_and_prints_the_line_last(tmp_path, monkeypatch):
    """Résumé JSON masqué, contrôles non exécutés listés, dernière ligne affichée en dernier."""
    monkeypatch.setattr(smoke, "SUMMARY_DIR", tmp_path / "e2e_albert")
    lines = []
    run = smoke.SmokeRun(smoke.parse_args([]), FAKE_ALBERT_KEY, "/usr/bin", env_file=tmp_path / ".env",
                         log=lines.append)

    def passed():
        """Contrôle réussi."""
        return {"statut": 200}

    run.run_check("off", "off3", passed)
    run.extras.append(f"clé {FAKE_ALBERT_KEY} trouvée")
    assert run.finish() == 1
    assert lines[-1] == "E2E ALBERT ÉCHEC (ON 0/7, OFF 1/3, isolation 0/2 ; clé [masqué] trouvée)"
    path = tmp_path / "e2e_albert" / f"summary-{run.stamp}.json"
    text = path.read_text(encoding="utf-8")
    assert FAKE_ALBERT_KEY not in text
    summary = json.loads(text)
    assert summary["resultat"] == lines[-1]
    assert summary["compteurs"] == {"on": "0/7", "isolation": "0/2", "off": "1/3"}
    assert [c["id"] for c in summary["controles"]["on"]] == [cid for cid, _route in smoke.ON_CHECKS]
    assert summary["controles"]["on"][0]["message"] == "non exécuté"


# ---------------------------------------------------------------------------
# Contrôles on1, on2 et on6 sur des sorties fabriquées (API factice)
# ---------------------------------------------------------------------------
CSV_FIELDS = ["itemKey", "type", "title", "abstract", "date", "url", "doi", "authors", "filename", "path",
              "attachment_title", "texteocr", "texteocr_provider", "texteocr_partial",
              "texteocr_pages_done", "texteocr_pages_total"]
GENERATED_NOTE = ("Cet article analyse la transformation des pratiques de lecture savante "
                  "par la numérisation des bibliothèques universitaires.")
FAILED_PAGE = "<!-- OCR ÉCHOUÉ (albert_lightonocr) : délai dépassé -->"


class _FakeApi:
    """API factice : chaque route SSE renvoie un flux terminé par l'événement ``terminal``."""

    def __init__(self, terminal=None):
        """Mémorise l'événement final renvoyé par chaque appel."""
        self.terminal = dict(terminal or {"type": "complete"})
        self.calls = []
        self.closed = False

    def sse(self, path, token, form, **_kwargs):
        """Consigne l'appel et renvoie un flux terminé par ``complete``."""
        self.calls.append((path, dict(form)))
        return smoke.SseOutcome(200, [dict(self.terminal)], dict(self.terminal))

    def close(self):
        """Fermeture factice."""
        self.closed = True


def _check_run(tmp_path, monkeypatch, terminal=None):
    """Run prêt pour un contrôle ON : corpus, session de A sous un ``uploads/`` temporaire, API factice."""
    uploads = tmp_path / "uploads"
    session = "0a1b2c3d_" + smoke.CORPUS_PREFIX + STAMP
    (uploads / session).mkdir(parents=True)
    monkeypatch.setattr(smoke, "UPLOAD_DIR", uploads)
    run = smoke.SmokeRun(smoke.parse_args([]), FAKE_ALBERT_KEY, "/usr/bin", env_file=tmp_path / ".env",
                         log=_silent)
    run.corpus = _corpus(tmp_path)
    run.sessions = {"a": session}
    run.tokens = {"a": "jeton-factice-a"}
    run.api = _FakeApi(terminal)
    return run, uploads / session


def _pages_text(pages, failed=()):
    """Texte OCR à marqueurs ``<!-- Page N -->`` ; une page en échec garde son marqueur."""
    blocks = []
    for number in range(1, pages + 1):
        body = FAILED_PAGE if number in failed else f"Texte lu de la page {number}."
        blocks.append(f"<!-- Page {number} -->\n{body}")
    return "\n\n".join(blocks)


def _ocr_row(doc, provider="albert_lightonocr", failed=()):
    """Ligne ``output.csv`` fabriquée pour un document du corpus."""
    return {"itemKey": doc.key, "type": "journalArticle", "title": doc.title, "filename": doc.filename,
            "path": doc.relpath, "texteocr": _pages_text(doc.pages, failed), "texteocr_provider": provider,
            "texteocr_partial": bool(failed), "texteocr_pages_done": doc.pages - len(failed),
            "texteocr_pages_total": doc.pages}


def _write_output_csv(path, rows):
    """Écrit ``output.csv`` au format de ``rad_dataframe`` (UTF-8 avec BOM, échappement ``\\``)."""
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, escapechar="\\", quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in CSV_FIELDS})


def _write_errors(path, entries):
    """Écrit ``output_errors.json`` au format de ``rad_dataframe``."""
    path.write_text(json.dumps({"total_errors": len(entries), "errors": entries}, ensure_ascii=False),
                    encoding="utf-8")


def _write_jsonl(path, records):
    """Écrit un ledger ``albert_usage.jsonl``."""
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_scan_row_problems_catches_a_half_failed_lightonocr_page(tmp_path):
    """Numérisé à moitié lu : marqueurs continus et fournisseur albert, mais drapeau partiel et OCR_PARTIAL."""
    scan = _corpus(tmp_path).scanned[0]
    clean = _ocr_row(scan)
    assert smoke.scan_row_problems(scan, clean, []) == []
    half = _ocr_row(scan, failed=(2,))
    half["texteocr_partial"] = "True"  # valeur relue dans output.csv
    assert smoke.markers_continuous(smoke.page_markers(half["texteocr"]), scan.pages)  # le défaut d'origine
    problems = smoke.scan_row_problems(scan, half, [{"itemKey": scan.key, "error_type": "OCR_PARTIAL"}])
    assert len(problems) == 2
    assert "OCR partiel" in problems[0] and "OCR_PARTIAL" in problems[1]
    # Erreur bloquante seule (drapeau faux) : repli de fournisseur signalé dans le fichier d'erreurs.
    assert "OCR_PROVIDER_FALLBACK" in smoke.scan_row_problems(
        scan, clean, [{"itemKey": scan.key, "error_type": "OCR_PROVIDER_FALLBACK"}])[0]
    # Entrées d'un autre document ou non bloquantes : ignorées.
    assert smoke.scan_row_problems(scan, clean, [{"itemKey": "AUTRE", "error_type": "OCR_PARTIAL"},
                                                 {"itemKey": scan.key, "error_type": "PDF_NOT_FOUND"}]) == []
    assert "fournisseur" in smoke.scan_row_problems(scan, _ocr_row(scan, provider="legacy"), [])[0]
    for value in (False, "False", "false", "0", "", None):
        assert smoke.partial_flag_set(value) is False
    for value in (True, "True", "1", "oui"):
        assert smoke.partial_flag_set(value) is True


def test_check_on1_fails_on_a_partial_scan_and_passes_when_complete(tmp_path, monkeypatch):
    """on1 échoue sur un OCR partiel du numérisé (CSV fabriqué), réussit quand il est complet."""
    run, folder = _check_run(tmp_path, monkeypatch)
    scan = run.corpus.scanned[0]
    rows = [_ocr_row(doc, failed=(2,) if doc.kind == "scan" else ()) for doc in run.corpus.docs]
    _write_output_csv(folder / "output.csv", rows)
    _write_errors(folder / "output_errors.json", [{"itemKey": scan.key, "error_type": "OCR_PARTIAL"}])
    assert run.run_check("on", "on1", run.check_on1) is False
    result = run.results["on"]["on1"]
    assert "OCR partiel" in result["message"] and "OCR_PARTIAL" in result["message"]
    assert result["detail"]["erreurs_bloquantes"] == {scan.filename: ["OCR_PARTIAL"]}
    assert run.api.calls == [("/process_dataframe_sse", {"path": run.sessions["a"]})]

    _write_output_csv(folder / "output.csv", [_ocr_row(doc) for doc in run.corpus.docs])
    _write_errors(folder / "output_errors.json", [])
    assert run.run_check("on", "on1", run.check_on1) is True
    assert run.results["on"]["on1"]["detail"]["erreurs_bloquantes"] == {}

    (folder / "output_errors.json").write_text("{illisible", encoding="utf-8")
    assert run.run_check("on", "on1", run.check_on1) is False
    assert "illisible" in run.results["on"]["on1"]["message"]


def test_recode_status_problems():
    """Recodage : au moins un chunk ``recoded`` ; replis, statut absent et LightOnOCR sauté sont des défauts."""
    lighton = {"texteocr_provider": "albert_lightonocr"}
    recoded = {**lighton, "recode_status": "recoded"}
    assert smoke.recode_status_problems([recoded, recoded]) == []
    # Chunks d'un fournisseur exclu du recodage (Mistral OCR servi par Albert) : admis.
    assert smoke.recode_status_problems([recoded, {"texteocr_provider": "albert_mistral_ocr",
                                                   "recode_status": "skipped"}]) == []
    all_raw = smoke.recode_status_problems([{**lighton, "recode_status": "fallback_raw"}] * 2)
    assert all_raw[0].startswith("aucun chunk recodé") and "fallback_raw=2" in all_raw[0]
    mixed = smoke.recode_status_problems([recoded, {**lighton, "recode_status": "fallback_truncated"}, lighton])
    assert mixed == ["2 chunk(s) non recodé(s) par Albert (recode_status : None=1, fallback_truncated=1, "
                     "recoded=1)"]
    skipped = smoke.recode_status_problems([recoded, {**lighton, "recode_status": "skipped"}])
    assert len(skipped) == 1 and "ALBERT_OCR_SKIP_RECODE=0" in skipped[0]
    assert smoke.recode_status_problems([]) == ["aucun chunk"]


def test_check_on2_requires_recoded_chunks_and_a_recode_ledger_record(tmp_path, monkeypatch):
    """on2 échoue si le recodage retombe sur le texte brut (sortie 0) ou sans enregistrement ``role=recode``."""
    run, folder = _check_run(tmp_path, monkeypatch)
    label = smoke.RECODE_MODEL  # étiquette présente même sans réponse servie

    def chunks(status):
        """Deux chunks LightOnOCR au statut ``status``."""
        return [{"id": f"c{i}", "recode_status": status, "recode_model": label,
                 "texteocr_provider": "albert_lightonocr"} for i in (1, 2)]

    ocr_record = {"role": "ocr_chat", "model": "lightonocr-2-1b", "cost": 0.0}
    recode_record = {"role": "recode", "model": "ministral-3-8b-instruct-2512", "cost": 0.0001}
    (folder / "output_chunks.json").write_text(json.dumps(chunks("fallback_raw")), encoding="utf-8")
    _write_jsonl(folder / "albert_usage.jsonl", [ocr_record])
    assert run.run_check("on", "on2", run.check_on2) is False
    message = run.results["on"]["on2"]["message"]
    assert "aucun chunk recodé" in message and "fallback_raw=2" in message and "role=recode" in message
    assert run.results["on"]["on2"]["detail"]["recode_status"] == {"fallback_raw": 2}

    (folder / "output_chunks.json").write_text(json.dumps(chunks("recoded")), encoding="utf-8")
    assert run.run_check("on", "on2", run.check_on2) is False  # ledger sans recodage
    assert "role=recode" in run.results["on"]["on2"]["message"]

    _write_jsonl(folder / "albert_usage.jsonl", [ocr_record, recode_record])
    assert run.run_check("on", "on2", run.check_on2) is True
    detail = run.results["on"]["on2"]["detail"]
    assert detail["ledger_recodage"]["modeles_servis"] == ["ministral-3-8b-instruct-2512"]
    assert detail["recode_status"] == {"recoded": 2}


PINNED_NOTES_MODEL = "gpt-oss-120b"
NOTES_RECORD = {"role": "notes", "model": PINNED_NOTES_MODEL, "cost": 0.002}


def test_notes_ledger_verdict():
    """Ledger des notes exigé : un enregistrement servi par l'id épinglé de gpt-oss avec un coût numérique."""
    assert smoke.NOTES_LEDGER_REQUIRED is True
    assert smoke.notes_pinned_model() == PINNED_NOTES_MODEL
    issue, info = smoke.notes_ledger_verdict([{"role": "recode", "model": "m", "cost": 0.1}])
    assert "sans enregistrement des notes" in issue
    assert info["statut"] == "exigé" and info["enregistrements"] == 0 and info["avec_modele_epingle"] == 0
    # Mode non exigé (NOTES_LEDGER_REQUIRED = False) : absence informative seulement.
    issue, info = smoke.notes_ledger_verdict([], required=False)
    assert issue is None and info["statut"] == smoke.NOTES_LEDGER_PENDING
    issue, info = smoke.notes_ledger_verdict(iter([NOTES_RECORD, {"role": "long_context", "model": "m2",
                                                                  "cost": 0.001}]))
    assert issue is None and info["statut"] == "vérifié"
    assert info["modele_epingle"] == PINNED_NOTES_MODEL and info["avec_modele_epingle"] == 1
    assert info["modeles_servis"] == [PINNED_NOTES_MODEL, "m2"] and info["cout"] == 0.003
    for bad in ({**NOTES_RECORD, "cost": None},
                {**NOTES_RECORD, "model": ""},
                {**NOTES_RECORD, "cost": True},
                {**NOTES_RECORD, "cost": float("nan")},
                {**NOTES_RECORD, "cost": "0.002"},
                {**NOTES_RECORD, "model": "ministral-3-8b-instruct-2512"},  # repli : pas gpt-oss
                {**NOTES_RECORD, "model": "openai/gpt-oss-120b"},  # alias, pas l'id épinglé
                {**NOTES_RECORD, "role": "recode"}):  # enregistrement d'une autre étape
        for required in (True, False):
            issue, info = smoke.notes_ledger_verdict([bad], required=required)
            if bad["role"] == "recode" and not required:
                assert issue is None, bad
            else:
                assert issue, (bad, required)
                assert info["avec_modele_epingle"] == 0, bad
    issue, _info = smoke.notes_ledger_verdict([{**NOTES_RECORD, "model": "ministral-3-8b-instruct-2512"}])
    assert PINNED_NOTES_MODEL in issue and "ministral-3-8b-instruct-2512" in issue
    # Un enregistrement conforme suffit, même à côté d'un enregistrement incomplet.
    assert smoke.notes_ledger_verdict([{**NOTES_RECORD, "cost": None}, NOTES_RECORD])[0] is None


def test_wait_for_notes_ledger_reads_a_journal_written_after_complete(tmp_path):
    """Journal écrit par le ``finally`` de la route juste après ``complete`` : relu jusqu'à l'enregistrement des notes."""
    path = tmp_path / "albert_usage.jsonl"
    _write_jsonl(path, [{"role": "recode", "model": "m", "cost": 0.1}])
    waits = []

    def late_write(seconds):
        """Attente factice : le journal des notes apparaît à la deuxième attente."""
        waits.append(seconds)
        if len(waits) == 2:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(NOTES_RECORD) + "\n")

    records = smoke.wait_for_notes_ledger(path, attempts=5, wait_s=0.25, sleep=late_write)
    assert waits == [0.25, 0.25]
    assert smoke.notes_ledger_verdict(records)[0] is None
    # Enregistrement des notes déjà présent : aucune attente.
    waits.clear()
    assert smoke.wait_for_notes_ledger(path, attempts=5, wait_s=0.25, sleep=late_write) == records
    assert waits == []
    # Journal absent : relectures bornées, puis liste vide.
    waits.clear()
    assert smoke.wait_for_notes_ledger(tmp_path / "absent.jsonl", attempts=3, wait_s=0.0, sleep=waits.append) == []
    assert waits == [0.0, 0.0, 0.0]


def test_check_on6_ledger_is_informational_until_required(tmp_path, monkeypatch):
    """on6 avec NOTES_LEDGER_REQUIRED = False : absence informative ; exigé (valeur du script) → échec.

    Le script fixe désormais ``NOTES_LEDGER_REQUIRED = True`` ; le mode
    informatif reste disponible en le rabattant à ``False``, et les
    enregistrements de notes présents y sont toujours vérifiés.
    """
    assert smoke.NOTES_LEDGER_REQUIRED is True
    monkeypatch.setattr(smoke, "NOTES_LEDGER_WAIT_S", 0.0)
    terminal = {"type": "complete", "summary": {"created": 3, "errors": 0}}
    run, folder = _check_run(tmp_path, monkeypatch, terminal)
    notes = [{"item_key": doc.key, "summary": GENERATED_NOTE} for doc in run.corpus.docs]
    (folder / "generated_notes.json").write_text(json.dumps(notes, ensure_ascii=False), encoding="utf-8")
    _write_jsonl(folder / "albert_usage.jsonl", [{"role": "recode", "model": "m", "cost": 0.1}])

    monkeypatch.setattr(smoke, "NOTES_LEDGER_REQUIRED", False)
    assert run.run_check("on", "on6", run.check_on6) is True
    assert run.results["on"]["on6"]["detail"]["ledger_notes"]["statut"] == smoke.NOTES_LEDGER_PENDING
    assert run.api.calls[-1] == ("/generate_zotero_notes_sse", {"session": run.sessions["a"],
                                                                "note_mode": "short", "model": smoke.NOTES_MODEL})
    _write_jsonl(folder / "albert_usage.jsonl", [{**NOTES_RECORD, "cost": None}])
    assert run.run_check("on", "on6", run.check_on6) is False  # enregistrement présent mais incomplet
    _write_jsonl(folder / "albert_usage.jsonl", [NOTES_RECORD])
    assert run.run_check("on", "on6", run.check_on6) is True
    assert run.results["on"]["on6"]["detail"]["ledger_notes"]["statut"] == "vérifié"

    monkeypatch.setattr(smoke, "NOTES_LEDGER_REQUIRED", True)
    _write_jsonl(folder / "albert_usage.jsonl", [{"role": "recode", "model": "m", "cost": 0.1}])
    assert run.run_check("on", "on6", run.check_on6) is False
    assert "sans enregistrement des notes" in run.results["on"]["on6"]["message"]


def test_check_on6_requires_the_notes_ledger(tmp_path, monkeypatch):
    """on6 : notes générées mais journal des notes absent, incomplet ou d'un autre modèle → échec ; conforme → succès."""
    monkeypatch.setattr(smoke, "NOTES_LEDGER_WAIT_S", 0.0)
    terminal = {"type": "complete", "summary": {"created": 3, "errors": 0}}
    run, folder = _check_run(tmp_path, monkeypatch, terminal)
    notes = [{"item_key": doc.key, "summary": GENERATED_NOTE} for doc in run.corpus.docs]
    (folder / "generated_notes.json").write_text(json.dumps(notes, ensure_ascii=False), encoding="utf-8")

    assert run.run_check("on", "on6", run.check_on6) is False  # aucun journal dans la session
    assert "sans enregistrement des notes" in run.results["on"]["on6"]["message"]
    assert run.results["on"]["on6"]["detail"]["journal_present"] is False
    assert run.api.calls[-1] == ("/generate_zotero_notes_sse", {"session": run.sessions["a"],
                                                                "note_mode": "short", "model": smoke.NOTES_MODEL})

    _write_jsonl(folder / "albert_usage.jsonl", [{"role": "recode", "model": "m", "cost": 0.1}])
    assert run.run_check("on", "on6", run.check_on6) is False  # journal des étapes précédentes seulement
    result = run.results["on"]["on6"]
    assert "sans enregistrement des notes" in result["message"]
    assert result["detail"]["journal_present"] is True and result["detail"]["ledger_notes"]["statut"] == "exigé"

    for bad in ({**NOTES_RECORD, "cost": None}, {**NOTES_RECORD, "model": "ministral-3-8b-instruct-2512"}):
        _write_jsonl(folder / "albert_usage.jsonl", [{"role": "recode", "model": "m", "cost": 0.1}, bad])
        assert run.run_check("on", "on6", run.check_on6) is False, bad
        assert PINNED_NOTES_MODEL in run.results["on"]["on6"]["message"], bad

    _write_jsonl(folder / "albert_usage.jsonl", [{"role": "recode", "model": "m", "cost": 0.1}, NOTES_RECORD])
    assert run.run_check("on", "on6", run.check_on6) is True
    ledger_detail = run.results["on"]["on6"]["detail"]["ledger_notes"]
    assert ledger_detail["statut"] == "vérifié" and ledger_detail["avec_modele_epingle"] == 1
    assert ledger_detail["modele_epingle"] == PINNED_NOTES_MODEL

    # Le journal conforme ne rattrape pas une note issue du gabarit.
    fallback = [{"item_key": doc.key, "summary": "Résumé à compléter : la note n'a pas pu être générée."}
                for doc in run.corpus.docs]
    (folder / "generated_notes.json").write_text(json.dumps(fallback, ensure_ascii=False), encoding="utf-8")
    assert run.run_check("on", "on6", run.check_on6) is False
    assert "gabarit" in run.results["on"]["on6"]["message"]


# ---------------------------------------------------------------------------
# Déroulé du run : nettoyage, signaux, arrêt du serveur, base temporaire
# ---------------------------------------------------------------------------
class _FakeServer:
    """Serveur factice : consigne son arrêt, et peut échouer."""

    def __init__(self, label, order, log_path, fail=False):
        """Mémorise le libellé, la liste d'ordre et l'échec éventuel."""
        self.label = label
        self.order = order
        self.log_path = log_path
        self.fail = fail

    def stop(self):
        """Consigne l'arrêt ; lève si ``fail``."""
        self.order.append(f"stop:{self.label}")
        if self.fail:
            raise RuntimeError("arrêt impossible")


def _recorder(order, name, exc=None):
    """Étape de nettoyage factice : consigne ``name`` puis lève ``exc`` s'il est donné."""

    def step():
        """Consigne l'étape."""
        order.append(name)
        if exc is not None:
            raise exc

    return step


def test_cleanup_failing_step_never_skips_the_others(tmp_path, monkeypatch):
    """Une étape de nettoyage qui lève devient une anomalie ; les suivantes s'exécutent, serveurs d'abord."""
    monkeypatch.setattr(smoke, "SUMMARY_DIR", tmp_path / "e2e_albert")
    lines = []
    run = smoke.SmokeRun(smoke.parse_args([]), FAKE_ALBERT_KEY, "/usr/bin", env_file=tmp_path / ".env",
                         log=lines.append)
    order = []
    run.servers = [_FakeServer("on", order, tmp_path / "on.log"),
                   _FakeServer("off", order, tmp_path / "off.log", fail=True)]
    api = _FakeApi()
    run.api = api
    steps = (("_cleanup_collections", "collections", None),
             ("_scan_for_key", "scan", PermissionError(f"accès refusé {FAKE_ALBERT_KEY}")),
             ("_cleanup_uploads", "uploads", None),
             ("_cleanup_manifests", "manifests", None),
             ("_cleanup_tmp", "tmp", None))
    for attr, name, exc in steps:
        monkeypatch.setattr(run, attr, _recorder(order, name, exc))
    run.cleanup()
    assert order == ["stop:on", "stop:off", "collections", "scan", "uploads", "manifests", "tmp"]
    assert api.closed is True and run.api is None
    assert run.extras == ["arrêt du serveur off en échec (RuntimeError)",
                          "recherche de la clé en échec (PermissionError)"]
    assert len(run.cleanup_report["erreurs"]) == 2
    assert FAKE_ALBERT_KEY not in json.dumps(run.cleanup_report, ensure_ascii=False)
    assert run.finish() == 1
    assert lines[-1].endswith("recherche de la clé en échec (PermissionError))")


def test_cleanup_stops_servers_first_then_removes_the_run_traces(tmp_path, monkeypatch):
    """Nettoyage réel (FakeAlbert, dossiers temporaires) : serveurs arrêtés avant l'ouverture du client Albert."""
    uploads, manifests, logs = tmp_path / "uploads", tmp_path / "manifests", tmp_path / "logs"
    uploads.mkdir()
    logs.mkdir()
    for name, value in (("UPLOAD_DIR", uploads), ("MANIFEST_ROOT", manifests), ("LOG_DIR", logs)):
        monkeypatch.setattr(smoke, name, value)
    run = smoke.SmokeRun(smoke.parse_args([]), FAKE_ALBERT_KEY, "/usr/bin", env_file=tmp_path / ".env",
                         log=_silent)
    fake = FakeAlbert()
    seed = _client(fake)
    run_id = seed.create_collection(run.collection_name)
    live_name = smoke.COLLECTION_PREFIX + "20260927T101500-4242"
    seed.create_collection(live_name)
    order = []

    def cleanup_client():
        """Client de nettoyage branché sur FakeAlbert ; consigne son ouverture."""
        order.append("client")
        return _client(fake)

    monkeypatch.setattr(run, "albert_client", cleanup_client)
    run.servers = [_FakeServer("on", order, tmp_path / "on.log"), _FakeServer("off", order, tmp_path / "off.log")]
    session = "0a1b2c3d_" + run.zip_stem
    (uploads / session).mkdir()
    (uploads / session / "output.csv").write_text("x", encoding="utf-8")
    (uploads / (session + ".zip")).write_bytes(b"PK")
    other = uploads / "12345678_mes-articles"
    other.mkdir()
    run.sessions = {"a": session}
    run.user_ids = {"a": smoke.E2E_USER_IDS["a"]}
    folder = manifests / str(run.user_ids["a"])
    folder.mkdir(parents=True)
    copy = folder / f"{smoke.manifest_label(session)}-20260927T101500000000Z.jsonl"
    copy.write_text("{}", encoding="utf-8")
    run.tmpdir = tmp_path / "run-tmp"
    run.tmpdir.mkdir()
    (run.tmpdir / "e2e.db").write_bytes(b"")
    (logs / "app.log").write_text("aucun secret\n", encoding="utf-8")

    run.cleanup()
    assert order == ["stop:on", "stop:off", "client"]
    assert run.extras == []
    assert run.cleanup_report["collections"] == {"supprimees": [run_id], "restantes": 0}
    assert [c["name"] for c in seed.list_collections()] == [live_name]
    assert sorted(p.name for p in uploads.iterdir()) == [other.name]
    assert not copy.exists() and not manifests.exists()
    assert not run.tmpdir.exists()
    assert run.cleanup_report["fuite_de_cle"] == []


@pytest.mark.parametrize("interruption", ["ctrl-c", "exception", "SIGTERM", "SIGHUP"])
def test_main_cleans_up_and_summarises_after_an_interruption(interruption, tmp_path, monkeypatch, capsys):
    """Ctrl-C, exception, SIGTERM ou SIGHUP pendant le run : nettoyage puis résumé, code 1, signaux restaurés."""
    if not hasattr(signal, interruption) and interruption.startswith("SIG"):
        pytest.skip(f"{interruption} absent de cette plateforme")
    env_file = _env_file(tmp_path, f"ALBERT_API_KEY={FAKE_ALBERT_KEY}\n")
    venv_bin = tmp_path / "depot" / ".venv" / "bin"
    _executable(venv_bin)
    monkeypatch.setattr(smoke, "VENV_BIN", venv_bin)
    for name in ("UPLOAD_DIR", "MANIFEST_ROOT", "LOG_DIR", "SUMMARY_DIR"):
        monkeypatch.setattr(smoke, name, tmp_path / name.lower())
    fake = FakeAlbert()
    watched = list(smoke.SHIELDED_SIGNALS)
    before = {sig: signal.getsignal(sig) for sig in watched}
    calls, handlers_during_run, handlers_during_cleanup = [], [], []

    def execute(self):
        """Déroulé factice, interrompu dès le début."""
        calls.append("execute")
        if interruption == "ctrl-c":
            raise KeyboardInterrupt
        if interruption == "exception":
            raise RuntimeError("panne simulée")
        handler = signal.getsignal(getattr(signal, interruption))
        handlers_during_run.append(handler)
        if not callable(handler):  # jamais l'action par défaut : le processus mourrait sans nettoyage
            raise RuntimeError(f"{interruption} non intercepté")
        handler(getattr(signal, interruption), None)

    original_cleanup, original_finish = smoke.SmokeRun.cleanup, smoke.SmokeRun.finish

    def cleanup(self):
        """Nettoyage espionné : consigne les gestionnaires de signaux actifs."""
        calls.append("cleanup")
        handlers_during_cleanup.append({sig: signal.getsignal(sig) for sig in watched})
        original_cleanup(self)

    def finish(self):
        """Résumé espionné."""
        calls.append("finish")
        return original_finish(self)

    def albert_client(self):
        """Client de nettoyage branché sur FakeAlbert (aucun appel réseau)."""
        return _client(fake)

    for attr, value in (("execute", execute), ("cleanup", cleanup), ("finish", finish),
                        ("albert_client", albert_client)):
        monkeypatch.setattr(smoke.SmokeRun, attr, value)
    code = smoke.main([], environ={"ALBERT_LIVE": "1", "PATH": str(tmp_path / "systeme")}, env_file=env_file)

    assert code == 1
    assert calls == ["execute", "cleanup", "finish"]
    assert handlers_during_cleanup == [{sig: signal.SIG_IGN for sig in watched}]
    if interruption.startswith("SIG"):
        assert handlers_during_run and callable(handlers_during_run[0])
    assert {sig: signal.getsignal(sig) for sig in watched} == before
    out = capsys.readouterr().out
    assert FAKE_ALBERT_KEY not in out
    assert out.strip().splitlines()[-1] == (
        "E2E ALBERT ÉCHEC (ON 0/7, OFF 0/3, isolation 0/2 ; préparation : 1 erreur(s))")
    summary = json.loads(next((tmp_path / "summary_dir").glob("summary-*.json")).read_text(encoding="utf-8"))
    expected = "déroulé : RuntimeError" if interruption == "exception" else "run interrompu"
    assert any(expected in entry for entry in summary["preparation"])
    assert summary["nettoyage"]["collections"] == {"supprimees": [], "restantes": 0}


class _GroupServer(smoke.ServerProcess):
    """Serveur factice : un parent Python qui lance un enfant, tous deux dans le groupe du serveur."""

    def __init__(self, log_path, ignore_term):
        """Prépare le parent (``ignore_term`` : parent et enfant ignorent SIGTERM)."""
        super().__init__(dict(os.environ), log_path, "test")
        self.ignore_term = ignore_term

    def command(self, port):
        """Parent qui lance un enfant, affiche son pid, puis attend."""
        ignore = "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if self.ignore_term else ""
        script = ("import signal, subprocess, sys, time\n" + ignore
                  + "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
                  + "print(child.pid, flush=True)\n"
                  + "time.sleep(120)\n")
        return [sys.executable, "-c", script]


def _alive(pid):
    """Vrai si le processus ``pid`` existe encore."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_until(predicate, timeout=20.0):
    """Attend que ``predicate()`` soit vrai (``False`` après ``timeout`` secondes)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return bool(predicate())


def _child_pid(log_path):
    """Pid de l'enfant écrit par le parent dans le journal (``None`` tant qu'il manque)."""
    try:
        text = log_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return int(text.splitlines()[0]) if text and text.splitlines()[0].isdigit() else None


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="groupes de processus POSIX")
@pytest.mark.parametrize("ignore_term", [False, True])
def test_server_stop_kills_the_whole_process_group(ignore_term, tmp_path, monkeypatch):
    """``ServerProcess.stop`` arrête le parent et son enfant (SIGTERM au groupe, puis SIGKILL)."""
    monkeypatch.setattr(smoke, "SERVER_STOP_TIMEOUT_S", 1.0)
    server = _GroupServer(tmp_path / "serveur.log", ignore_term)
    server.start()

    def child_started():
        """Vrai quand le parent a écrit le pid de son enfant."""
        return _child_pid(server.log_path) is not None

    try:
        assert _wait_until(child_started), server.log_tail()
        child = _child_pid(server.log_path)
        assert _alive(child)

        def child_gone():
            """Vrai quand l'enfant n'existe plus."""
            return not _alive(child)

        server.stop()
        assert server.proc.returncode is not None
        assert _wait_until(child_gone), "enfant toujours vivant après stop()"
        assert server._log is None
        server.stop()  # une seule fois : sans effet
    finally:
        if server.proc is not None and server.proc.poll() is None:
            os.killpg(server.proc.pid, signal.SIGKILL)
            server.proc.wait(timeout=10)


def test_prepare_database_refuses_a_database_url_already_bound(tmp_path, monkeypatch):
    """``DATABASE_URL`` déjà liée par un import antérieur : refus avant toute création de schéma ou de jeton."""
    import app.database.init_db as init_db
    from app.config import settings

    monkeypatch.setenv("DATABASE_URL", "sqlite:///placeholder")  # valeur d'origine restaurée après le test

    def forbidden(*_args, **_kwargs):
        """Toute création de schéma est une erreur."""
        raise AssertionError("schéma créé malgré le refus")

    monkeypatch.setattr(init_db, "init_database", forbidden)
    run = smoke.SmokeRun(smoke.parse_args([]), FAKE_ALBERT_KEY, "/usr/bin", env_file=tmp_path / ".env",
                         log=_silent)
    run.db_url = f"sqlite:///{tmp_path / 'e2e.db'}"
    assert settings.DATABASE_URL != run.db_url
    with pytest.raises(RuntimeError, match="DATABASE_URL non pris en compte"):
        run.prepare_database()
    assert not (tmp_path / "e2e.db").exists()
    assert run.user_ids == {} and run.tokens == {} and run.jwt_secret is None
