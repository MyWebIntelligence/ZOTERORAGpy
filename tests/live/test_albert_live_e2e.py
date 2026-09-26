"""Test live de bout en bout de la chaîne CLI souveraine Albert (lot 8).

Marqueur ``albert_live`` : sautés sauf si ``ALBERT_LIVE=1`` est exporté dans le
shell (``ALBERT_LIVE`` ne doit jamais figurer dans un fichier ``.env``). La clé
vient de l'environnement du shell, sinon de ``ALBERT_API_KEY`` dans le ``.env``
racine ; elle n'est jamais affichée (sorties masquées avant toute assertion) ni
comparée à une valeur.

Les quatre scripts du pipeline sont lancés en sous-processus avec
``sys.executable`` (jamais un interpréteur résolu par le ``PATH``), dans un dossier temporaire, avec
l'environnement du shell complété de ``ALBERT_ENABLED=1``,
``OCR_ENABLE_ALBERT=1`` et ``EMBEDDING_PROVIDER=albert`` :

1. ``rad_dataframe.py`` sur un corpus de type Zotero (JSON + dossier
   ``files/``) de deux PDF générés avec fitz : un PDF texte et un PDF image
   seule (page rastérisée portant ``SOUVERAIN_E2E_7Q``, sans couche texte).
   Chaque page porte au moins 800 caractères : la garde de densité
   (``OCR_MIN_CHARS_PER_PAGE``, 500 par défaut, appliquée aussi à LightOnOCR)
   reste active et ne doit signaler aucun des deux PDF comme partiel. Le
   scanné est lu par le maillon Albert (``texteocr_provider`` commence par
   ``albert_``), avec le marqueur ``<!-- Page 1 -->`` ;
2. ``rad_chunk.py --phase initial --model albert/ministral-3-8b-instruct-2512`` :
   chaque chunk porte ``recode_model`` en ``albert/…`` et un ``recode_status``
   ``recoded`` (``skipped`` seulement pour un texte ``albert_mistral_ocr``),
   jamais ``fallback_*`` : au moins un chunk a donc été recodé par Albert ;
3. ``rad_chunk.py --phase dense --embedding-provider albert`` : chaque
   vecteur fait 1024 dimensions (bge-m3) et aucun n'est nul ;
4. ``rad_vectordb.py --db albert`` vers une collection privée
   ``ragpy-probe-e2e-<ts>`` créée pour l'occasion (``Inserted: N``, N > 0),
   puis relance à l'identique (``Inserted: 0`` et ``Skipped (existing): N``) ;
5. contrôle de fuite sur les fichiers écrits : tout l'espace temporaire
   (``output.csv``, ``output_errors.json``, chunks, ``chunking.log``,
   ``albert_usage.jsonl``, ``albert_manifest.jsonl``…) et la partie de chaque
   ``logs/*.log`` écrite depuis la création de l'espace. Seule la présence de
   la clé est testée (booléen), jamais sa valeur.

Chaîne strictement souveraine : les clés Mistral, OpenAI et OpenRouter sont
vidées dans l'environnement des sous-processus et les replis OpenAI et OCR
local sont coupés, si bien qu'un échec d'Albert ne peut pas être masqué par un
autre fournisseur. La dédup (``DEDUP_ENABLED=0``) et les caches de recodage
sont coupés : la relance ne repose que sur l'idempotence par ``content_id``.
Les appels restent peu nombreux (une page par PDF, un lot d'embeddings), le
compte étant limité à 1000 requêtes par jour.

La collection est supprimée par un finaliseur (``AlbertClient``), même en cas
d'échec ; ``scripts/albert_probe.py --cleanup --dry-run`` doit ensuite afficher
``ragpy-probe restantes: 0``.

Commande : ``ALBERT_LIVE=1 .venv/bin/python -m pytest tests/live/test_albert_live_e2e.py -m albert_live -v -p no:cacheprovider``
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import fitz  # type: ignore[import-not-found]
import pandas as pd
import pytest

from scripts.rad_albert import collections as col
from scripts.rad_albert.client import AlbertClient
from scripts.rad_albert.config import AlbertConfig

pytestmark = pytest.mark.albert_live

RAGPY_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = RAGPY_ROOT / "scripts"
DATAFRAME_SCRIPT = SCRIPTS_DIR / "rad_dataframe.py"
CHUNK_SCRIPT = SCRIPTS_DIR / "rad_chunk.py"
VECTORDB_SCRIPT = SCRIPTS_DIR / "rad_vectordb.py"

RECODE_MODEL = "albert/ministral-3-8b-instruct-2512"
ALBERT_PREFIX = "albert/"
ALBERT_OCR_PREFIX = "albert_"
EMBED_DIM = 1024
COLLECTION_PREFIX = "ragpy-probe-e2e-"
CLI_TIMEOUT_S = 900

SCAN_TOKEN = "SOUVERAIN_E2E_7Q"
TEXT_TOKEN = "TEXTE_E2E_7Q"
TEXT_KEY = "E2ETEXTE"
SCAN_KEY = "E2ESCAN1"
TEXT_PDF = "e2e_texte.pdf"
SCAN_PDF = "e2e_scan.pdf"
# Garde de densité de rad_dataframe.py (valeur par défaut, posée explicitement
# pour qu'un .env ou un shell ne la modifie pas) et plancher du corpus : chaque
# page porte nettement plus que le seuil, la garde reste active sans se déclencher.
OCR_DENSITY_MIN = 500
MIN_PAGE_CHARS = 800
TEXT_SENTENCES = (
    "La souveraineté numérique des infrastructures de recherche suppose des traitements "
    "hébergés en France, de l'OCR jusqu'aux collections de passages indexés.",
    "Ce document de contrôle vérifie que la couche texte d'un PDF natif traverse la chaîne "
    "Albert sans perte ni troncature.",
    "Chaque page porte assez de caractères pour franchir la garde de densité qui signale "
    "les extractions incomplètes.",
    "Les passages sont ensuite découpés, recodés par un modèle de langage hébergé, puis "
    "convertis en vecteurs de 1024 dimensions.",
    "Le recodage corrige la ponctuation et les césures sans résumer ni reformuler le propos "
    "de l'auteur.",
    "Les métadonnées bibliographiques accompagnent chaque passage jusqu'à la collection "
    "privée créée pour le test.",
    "Une relance à l'identique ne doit rien renvoyer, car chaque passage est identifié par "
    "l'empreinte de son contenu.",
    "La collection est supprimée à la fin du test, même lorsque l'une des étapes échoue.",
    "Aucune clé d'accès ne doit apparaître dans les sorties, les journaux ou les fichiers "
    "produits par le pipeline.",
)
SCAN_SENTENCES = (
    "Cette page numérisée ne contient aucune couche texte : seule une image rastérisée la "
    "compose.",
    "Le maillon OCR souverain doit en restituer le contenu, avec le marqueur de page attendu "
    "par la suite du pipeline.",
    "La numérisation des corpus reste une étape coûteuse pour les équipes de recherche en "
    "sciences humaines et sociales.",
    "Les ouvrages anciens, les rapports administratifs et les archives de presse arrivent "
    "souvent sous forme d'images sans texte exploitable.",
    "La reconnaissance optique de caractères transforme ces images en un texte que l'on peut "
    "découper, recoder et indexer.",
    "Une transcription trop courte serait signalée comme partielle par la garde de densité, "
    "jamais comme un succès silencieux.",
    "Le jeton de contrôle placé en tête de page permet de vérifier que la transcription porte "
    "bien sur ce document.",
    "Les passages extraits rejoignent ensuite la même collection privée que ceux du PDF texte.",
)
PAGE_MARKER_RE = re.compile(r"<!--\s*Page\s+(\d+)\s*-->")
# Seul fournisseur d'OCR de cette chaîne dont le texte n'est pas recodé
# (ALBERT_OCR_SKIP_RECODE=0, clé Mistral vide, OCR local coupé). Le compte n'a
# pas accès à /v1/ocr : tout le texte vient de LightOnOCR et doit être recodé.
RECODE_SKIPPED_OCR_PROVIDERS = ("albert_mistral_ocr",)
LOGS_DIR = RAGPY_ROOT / "logs"
# Fichiers que chaque étape réussie doit avoir écrits (contrôle de fuite non vide).
STEP_OUTPUTS = {
    "ocr": ("output.csv", "albert_usage.jsonl", "pdf_processing.log"),
    "initial": ("output_chunks.json", "chunking.log"),
    "dense": ("output_chunks_with_embeddings.json",),
    "push": (col.MANIFEST_FILENAME,),
}

# Environnement ajouté à celui du shell pour chaque sous-processus.
CLI_ENV_OVERRIDES = {
    # Albert de bout en bout (OCR, recodage, embeddings, collections).
    "ALBERT_ENABLED": "1",
    "OCR_ENABLE_ALBERT": "1",
    "EMBEDDING_PROVIDER": "albert",
    # Aucun autre fournisseur : un échec Albert reste visible, jamais masqué.
    "MISTRAL_API_KEY": "",
    "OPENAI_API_KEY": "",
    "OPENROUTER_API_KEY": "",
    "OCR_ENABLE_OPENAI_FALLBACK": "0",
    "OCR_ENABLE_LOCAL_FALLBACK": "0",
    # Garde de densité active à sa valeur par défaut (LightOnOCR compris).
    "OCR_MIN_CHARS_PER_PAGE": str(OCR_DENSITY_MIN),
    # Ledger d'usage écrit à chaque étape : il fait partie du contrôle de fuite.
    "ALBERT_USAGE_LOG": "1",
    # D21 : texte LightOnOCR recodé (défaut) ; aucun cache disque partagé.
    "ALBERT_OCR_SKIP_RECODE": "0",
    "RECODE_CACHE_ENABLED": "0",
    "RECODE_EMBED_CACHE_ENABLED": "0",
    # Idempotence par content_id seule : la relance compte des chunks « existing ».
    "DEDUP_ENABLED": "0",
    # Appels séquentiels (quota partagé par le compte), limiteur local.
    "PDF_EXTRACTION_WORKERS": "1",
    "ALBERT_PUSH_CONCURRENCY": "1",
    "ALBERT_LIMITER_BACKEND": "local",
    "PYTHONIOENCODING": "utf-8",
}
# Variables du shell jamais transmises aux sous-processus.
CLI_ENV_DROPPED = ("ALBERT_LIVE", "RAGPY_DOTENV_DENY")


# ---------------------------------------------------------------------------
# Outils
# ---------------------------------------------------------------------------
def _live_key() -> str:
    """Clé Albert du shell, sinon du ``.env`` racine (valeur jamais affichée)."""
    key = (os.environ.get("ALBERT_API_KEY") or "").strip()
    if not key:
        try:
            from dotenv import dotenv_values
        except ImportError:
            dotenv_values = None
        if dotenv_values is not None:
            key = (dotenv_values(RAGPY_ROOT / ".env").get("ALBERT_API_KEY") or "").strip()
    return key


def _masked(text: Optional[str], key: str) -> str:
    """Texte sans la clé (remplacée par ``***``)."""
    text = text or ""
    return text.replace(key, "***") if key else text


def _cli_env(key: str) -> Dict[str, str]:
    """Environnement d'un sous-processus : celui du shell, Albert activé, clé explicite.

    ``rad_vectordb.py`` ne relit pas le ``.env`` : la clé est donc toujours
    posée dans l'environnement transmis.
    """
    env = dict(os.environ)
    for name in CLI_ENV_DROPPED:
        env.pop(name, None)
    env.update(CLI_ENV_OVERRIDES)
    env["ALBERT_API_KEY"] = key
    return env


@dataclass
class CliRun:
    """Résultat masqué d'un script lancé en sous-processus.

    Attributes:
        code: code de sortie.
        out: sortie standard, clé masquée.
        err: sortie d'erreur, clé masquée.
        key_leaked: vrai si la clé figurait dans une sortie brute (booléen
            seulement : la valeur n'est jamais conservée).
    """

    code: int
    out: str
    err: str
    key_leaked: bool

    def tail(self, size: int = 3000) -> str:
        """Fin des sorties standard et d'erreur, pour les messages d'assertion."""
        return f"--- stdout ---\n{self.out[-size:]}\n--- stderr ---\n{self.err[-size:]}"


def _run_cli(args: List[str], key: str) -> CliRun:
    """Lance ``sys.executable <args>`` depuis la racine du dépôt ; sorties masquées."""
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=str(RAGPY_ROOT),
        env=_cli_env(key),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=CLI_TIMEOUT_S,
        stdin=subprocess.DEVNULL,
    )
    raw_out, raw_err = proc.stdout or "", proc.stderr or ""
    leaked = bool(key) and (key in raw_out or key in raw_err)
    return CliRun(proc.returncode, _masked(raw_out, key), _masked(raw_err, key), leaked)


def _alnum_upper(text: str) -> str:
    """Lettres et chiffres seulement, en majuscules (insensible à l'échappement markdown)."""
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def _markers(text: str) -> List[int]:
    """Numéros des marqueurs ``<!-- Page N -->`` d'un texte."""
    return [int(n) for n in PAGE_MARKER_RE.findall(text or "")]


def _result_block(stdout: str) -> List[str]:
    """Lignes du bloc ``=== Result ===`` de ``rad_vectordb.py``."""
    lines = stdout.splitlines()
    assert "=== Result ===" in lines, stdout[-3000:]
    return lines[lines.index("=== Result ==="):]


def _value(block: List[str], label: str) -> Optional[str]:
    """Valeur d'une ligne ``label: valeur`` du bloc Result (``None`` si absente)."""
    prefix = label + ": "
    for line in block:
        if line.startswith(prefix):
            return line[len(prefix):]
    return None


def _load_json_list(path: Path) -> List[dict]:
    """Liste JSON écrite par ``rad_chunk.py`` (erreur explicite si le fichier manque)."""
    assert path.is_file(), f"fichier attendu absent : {path.name}"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list), f"{path.name} ne contient pas une liste"
    return data


def _log_offsets() -> Dict[Path, int]:
    """Taille actuelle de chaque ``logs/*.log`` (début du contrôle de fuite).

    Les sauvegardes de rotation (``app.log.1``…) sont exclues : leur contenu
    antérieur au test ne peut pas être distingué de ce qui a été écrit depuis.
    """
    offsets: Dict[Path, int] = {}
    if LOGS_DIR.is_dir():
        for path in LOGS_DIR.glob("*.log"):
            try:
                if path.is_file():
                    offsets[path] = path.stat().st_size
            except OSError:
                continue
    return offsets


def _file_contains(path: Path, offset: int, needle: bytes) -> bool:
    """Vrai si ``needle`` figure dans ``path`` à partir de ``offset`` (lecture par blocs).

    Un fichier devenu plus court que ``offset`` (rotation) est relu depuis le
    début. Seul un booléen sort : le contenu lu n'est ni conservé ni affiché.
    """
    if not needle:
        return False
    overlap = len(needle) - 1
    with open(path, "rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        handle.seek(offset if 0 <= offset <= size else 0)
        tail = b""
        while True:
            block = handle.read(1 << 20)
            if not block:
                return False
            data = tail + block
            if needle in data:
                return True
            tail = data[-overlap:] if overlap else b""


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
def _write_page(page, token: str, sentences: Tuple[str, ...]) -> str:
    """Écrit ``token`` en gros titre puis ``sentences`` dans un cadre ; renvoie le texte posé.

    Raises:
        ValueError: le texte déborde du cadre (page incomplète).
    """
    page.insert_text((56, 110), token, fontsize=30)
    body = " ".join(sentences)
    left = page.insert_textbox(fitz.Rect(56, 150, 539, 786), body, fontsize=12, fontname="helv")
    if left < 0:
        raise ValueError("texte trop long pour une page A4")
    return f"{token}\n{body}"


def _text_pdf(path: Path) -> None:
    """PDF d'une page avec couche texte : ``TEXT_TOKEN`` puis ``TEXT_SENTENCES``."""
    doc = fitz.open()
    written = _write_page(doc.new_page(width=595, height=842), TEXT_TOKEN, TEXT_SENTENCES)
    assert len(written) >= MIN_PAGE_CHARS, len(written)
    doc.save(str(path))
    doc.close()


def _scan_pdf(path: Path) -> None:
    """PDF d'une page image seule : ``SCAN_TOKEN`` et ``SCAN_SENTENCES`` rastérisés, sans couche texte."""
    source = fitz.open()
    written = _write_page(source.new_page(width=595, height=842), SCAN_TOKEN, SCAN_SENTENCES)
    assert len(written) >= MIN_PAGE_CHARS, len(written)
    png = source[0].get_pixmap(dpi=200, colorspace=fitz.csGRAY).tobytes("png")
    source.close()
    doc = fitz.open()
    target = doc.new_page(width=595, height=842)
    target.insert_image(target.rect, stream=png)
    doc.save(str(path))
    doc.close()


def _zotero_item(key: str, title: str, relative_pdf: str) -> dict:
    """Élément minimal d'export Zotero tel que le lit ``rad_dataframe.py``."""
    return {
        "key": key,
        "itemType": "journalArticle",
        "title": title,
        "abstractNote": "",
        "date": "2026-09-27",
        "url": "",
        "DOI": "",
        "creators": [{"lastName": "RAGpy", "firstName": "Sonde"}],
        "attachments": [{"path": relative_pdf, "title": os.path.basename(relative_pdf)}],
    }


@dataclass
class E2EWorkspace:
    """Dossiers et fichiers du test de bout en bout, et étapes réussies.

    Attributes:
        corpus_dir: corpus Zotero (JSON et ``files/``).
        json_path: export Zotero JSON.
        out_dir: sorties des scripts.
        done: étapes réussies (``ocr``, ``initial``, ``dense``, ``push``).
        log_offsets: taille des fichiers de ``logs/`` à la création de l'espace.
        extra_files: fichiers écrits hors de l'espace temporaire (manifeste
            annoncé par ``rad_vectordb.py``), inclus dans le contrôle de fuite.
    """

    corpus_dir: Path
    json_path: Path
    out_dir: Path
    done: Set[str] = field(default_factory=set)
    log_offsets: Dict[Path, int] = field(default_factory=dict)
    extra_files: List[Path] = field(default_factory=list)

    @property
    def root(self) -> Path:
        """Dossier temporaire racine (corpus et sorties)."""
        return self.out_dir.parent

    @property
    def csv_path(self) -> Path:
        """CSV produit par ``rad_dataframe.py``."""
        return self.out_dir / "output.csv"

    @property
    def chunks_path(self) -> Path:
        """Chunks recodés produits par la phase ``initial``."""
        return self.out_dir / "output_chunks.json"

    @property
    def dense_path(self) -> Path:
        """Chunks avec embeddings produits par la phase ``dense``."""
        return self.out_dir / "output_chunks_with_embeddings.json"

    def require(self, step: str) -> None:
        """Saute le test si l'étape ``step`` n'a pas réussi (aucun appel inutile)."""
        if step not in self.done:
            pytest.skip(f"étape préalable « {step} » absente ou en échec : test non lancé")

    def written_files(self) -> List[Tuple[str, Path, int]]:
        """Fichiers du contrôle de fuite : ``(libellé, chemin, décalage de lecture)``.

        Tout fichier de l'espace temporaire (lu en entier), les fichiers de
        ``extra_files`` situés ailleurs, et chaque ``logs/*.log`` lu à
        partir de sa taille à la création de l'espace (0 s'il est apparu depuis).
        """
        entries: List[Tuple[str, Path, int]] = []
        seen: Set[Path] = set()
        for path in sorted(p for p in self.root.rglob("*") if p.is_file()):
            entries.append((path.relative_to(self.root).as_posix(), path, 0))
            seen.add(path.resolve())
        for path in self.extra_files:
            if path.is_file() and path.resolve() not in seen:
                entries.append((path.name, path, 0))
                seen.add(path.resolve())
        if LOGS_DIR.is_dir():
            for path in sorted(p for p in LOGS_DIR.glob("*.log") if p.is_file()):
                entries.append((f"logs/{path.name}", path, self.log_offsets.get(path, 0)))
        return entries


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def live_key() -> str:
    """Clé Albert réelle ; saute le module si elle est absente."""
    key = _live_key()
    if not key:
        pytest.skip("ALBERT_API_KEY absente du shell et du .env : tests live impossibles")
    return key


@pytest.fixture(scope="module")
def e2e(tmp_path_factory) -> E2EWorkspace:
    """Corpus Zotero minimal (JSON + ``files/``) de deux PDF, dans un dossier temporaire.

    Les tailles des fichiers de ``logs/`` sont relevées avant tout lancement de
    script : le contrôle de fuite ne lit que ce qui a été écrit depuis.
    """
    log_offsets = _log_offsets()
    root = tmp_path_factory.mktemp("albert_e2e")
    corpus_dir = root / "corpus"
    items = []
    for key, title, name, build in (
        (TEXT_KEY, "Sonde E2E Albert : PDF texte", TEXT_PDF, _text_pdf),
        (SCAN_KEY, "Sonde E2E Albert : PDF scanné", SCAN_PDF, _scan_pdf),
    ):
        relative = f"files/{key}/{name}"
        target = corpus_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        build(target)
        items.append(_zotero_item(key, title, relative))
    with fitz.open(str(corpus_dir / f"files/{SCAN_KEY}/{SCAN_PDF}")) as scan:
        assert scan.page_count == 1
        assert scan[0].get_text().strip() == "", "le PDF scanné ne doit avoir aucune couche texte"
    with fitz.open(str(corpus_dir / f"files/{TEXT_KEY}/{TEXT_PDF}")) as text_doc:
        assert text_doc.page_count == 1
        assert len(text_doc[0].get_text()) >= MIN_PAGE_CHARS, "PDF texte sous le plancher de densité"
    json_path = corpus_dir / "corpus_e2e.json"
    json_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    out_dir = root / "out"
    out_dir.mkdir()
    return E2EWorkspace(corpus_dir=corpus_dir, json_path=json_path, out_dir=out_dir,
                        log_offsets=log_offsets)


@pytest.fixture(scope="module")
def live_client(live_key):
    """Client Albert réel du finaliseur (configuration du shell, Albert activé)."""
    env = dict(os.environ)
    env["ALBERT_ENABLED"] = "1"
    client = AlbertClient(AlbertConfig.from_env(env), live_key)
    yield client
    client.close()


@pytest.fixture(scope="module")
def probe_collection_name(live_client, request) -> str:
    """Nom ``ragpy-probe-e2e-<ts>`` ; le finaliseur supprime toute collection de ce nom exact."""
    name = f"{COLLECTION_PREFIX}{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"

    def cleanup() -> None:
        """Supprime les collections de la sonde (nom exact), même après un échec."""
        for collection in live_client.list_collections(name=name):
            live_client.delete_collection(collection["id"])
        assert live_client.list_collections(name=name) == [], f"collection {name} non supprimée"

    request.addfinalizer(cleanup)
    return name


# ---------------------------------------------------------------------------
# Étapes
# ---------------------------------------------------------------------------
def _csv_rows(path: Path) -> Dict[str, dict]:
    """Lignes du CSV de ``rad_dataframe.py`` indexées par ``itemKey``."""
    assert path.is_file(), "output.csv absent"
    frame = pd.read_csv(path, encoding="utf-8-sig", escapechar="\\", keep_default_na=False)
    return {str(row["itemKey"]): row for row in frame.to_dict(orient="records")}


def _ocr_error_types(csv_path: Path) -> List[Tuple[str, str]]:
    """(itemKey, error_type) du fichier d'erreurs voisin du CSV (vide s'il n'existe pas)."""
    errors_path = csv_path.with_name(csv_path.stem + "_errors.json")
    if not errors_path.is_file():
        return []
    payload = json.loads(errors_path.read_text(encoding="utf-8"))
    return [(str(e.get("itemKey")), str(e.get("error_type"))) for e in payload.get("errors") or []]


def test_e2e_1_ocr_dataframe_albert(e2e, live_key):
    """``rad_dataframe.py`` : le PDF scanné est lu par le maillon Albert, marqueur de page compris."""
    run = _run_cli(
        [str(DATAFRAME_SCRIPT), "--json", str(e2e.json_path), "--dir", str(e2e.corpus_dir),
         "--output", str(e2e.csv_path)],
        live_key,
    )
    assert run.code == 0, run.tail()
    rows = _csv_rows(e2e.csv_path)
    assert set(rows) == {TEXT_KEY, SCAN_KEY}, run.tail()

    scan = rows[SCAN_KEY]
    provider = str(scan["texteocr_provider"])
    assert provider.startswith(ALBERT_OCR_PREFIX), f"provider du scanné : {provider!r}\n{run.tail()}"
    scan_text = str(scan["texteocr"])
    assert _markers(scan_text) == [1], scan_text[:1000]
    assert _alnum_upper(SCAN_TOKEN) in _alnum_upper(scan_text), scan_text[:1000]

    text_row = rows[TEXT_KEY]
    assert str(text_row["texteocr_provider"]).strip(), "provider du PDF texte vide"
    assert str(text_row["texteocr"]).strip(), "texte du PDF texte vide"
    # Garde de densité active (seuil par défaut) : aucun des deux PDF n'est partiel.
    for key, row in ((SCAN_KEY, scan), (TEXT_KEY, text_row)):
        partial = str(row["texteocr_partial"]).strip().lower()
        assert partial in ("false", "0"), (
            f"{key} partiel ({row['texteocr_provider']}, {len(str(row['texteocr']))} caractères)"
        )

    blocking = {"OCR_FAILED", "OCR_PARTIAL", "OCR_PROVIDER_FALLBACK", "EXTRACTION_FAILED"}
    errors = [(k, t) for k, t in _ocr_error_types(e2e.csv_path) if t in blocking]
    assert errors == [], errors
    assert run.key_leaked is False
    e2e.done.add("ocr")


def test_e2e_2_recode_initial_albert(e2e, live_key):
    """``rad_chunk.py --phase initial`` avec un modèle ``albert/…`` : chunks réellement recodés par Albert.

    Un repli (``fallback_raw``, ``fallback_truncated``) garde un libellé
    ``albert/…`` et un code de sortie nul : seul ``recode_status == "recoded"``
    prouve le recodage. ``skipped`` n'est admis que pour un texte
    ``albert_mistral_ocr`` (jamais recodé) ; au moins un chunk doit être recodé.
    """
    e2e.require("ocr")
    run = _run_cli(
        [str(CHUNK_SCRIPT), "--input", str(e2e.csv_path), "--output", str(e2e.out_dir),
         "--phase", "initial", "--model", RECODE_MODEL],
        live_key,
    )
    assert run.code == 0, run.tail()
    chunks = _load_json_list(e2e.chunks_path)
    assert chunks, run.tail()
    statuses: Dict[str, int] = {}
    for chunk in chunks:
        status = str(chunk.get("recode_status") or "")
        statuses[status] = statuses.get(status, 0) + 1
    for chunk in chunks:
        model = str(chunk.get("recode_model") or "")
        assert model.startswith(ALBERT_PREFIX) and len(model) > len(ALBERT_PREFIX), model
        provider = str(chunk.get("texteocr_provider") or "")
        expected = "skipped" if provider in RECODE_SKIPPED_OCR_PROVIDERS else "recoded"
        assert chunk.get("recode_status") == expected, (
            f"{chunk.get('id')} : statut {chunk.get('recode_status')!r} (attendu {expected!r}, "
            f"provider {provider!r}, statuts {statuses})\n{run.tail()}"
        )
        assert str(chunk.get("text") or "").strip(), chunk.get("id")
    assert statuses.get("recoded", 0) >= 1, f"aucun chunk recodé par Albert : {statuses}"
    assert {chunk.get("filename") for chunk in chunks} == {TEXT_PDF, SCAN_PDF}
    scan_providers = {str(c.get("texteocr_provider")) for c in chunks if c.get("filename") == SCAN_PDF}
    assert scan_providers and all(p.startswith(ALBERT_OCR_PREFIX) for p in scan_providers), scan_providers
    assert run.key_leaked is False
    e2e.done.add("initial")


def test_e2e_3_dense_albert_1024(e2e, live_key):
    """``rad_chunk.py --phase dense --embedding-provider albert`` : vecteurs 1024, aucun nul."""
    e2e.require("initial")
    chunks = _load_json_list(e2e.chunks_path)
    run = _run_cli(
        [str(CHUNK_SCRIPT), "--input", str(e2e.chunks_path), "--output", str(e2e.out_dir),
         "--phase", "dense", "--embedding-provider", "albert"],
        live_key,
    )
    assert run.code == 0, run.tail()
    dense = _load_json_list(e2e.dense_path)
    assert [c.get("id") for c in dense] == [c.get("id") for c in chunks]
    for chunk in dense:
        vector = chunk.get("embedding")
        assert isinstance(vector, list) and len(vector) == EMBED_DIM, chunk.get("id")
        assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in vector), chunk.get("id")
        assert any(v != 0.0 for v in vector), f"vecteur nul : {chunk.get('id')}"
        assert chunk.get("embedding_provider") == "albert"
        assert chunk.get("embedding_dim") == EMBED_DIM
    assert run.key_leaked is False
    e2e.done.add("dense")


def test_e2e_4_push_collection_then_rerun_idempotent(e2e, live_key, live_client, probe_collection_name):
    """``rad_vectordb.py --db albert`` : envoi (``Inserted: N``) puis relance sans nouvel envoi."""
    e2e.require("dense")
    expected = sum(1 for c in _load_json_list(e2e.dense_path) if str(c.get("text") or "").strip())
    assert expected > 0
    args = [
        str(VECTORDB_SCRIPT), "--input", str(e2e.dense_path), "--db", "albert",
        "--albert-collection-name", probe_collection_name,
        "--albert-create-collection", "--albert-ack-retention",
    ]

    # 1) Premier envoi : collection privée créée, N chunks, manifeste écrit.
    first = _run_cli(args, live_key)
    assert first.code == 0, first.tail()
    block = _result_block(first.out)
    assert _value(block, "Status") == "success", block
    inserted = int(_value(block, "Inserted") or "0")
    assert inserted > 0, block
    assert inserted == expected, block
    assert _value(block, "Skipped (existing)") == "0", block
    manifest = _value(block, "Albert manifest")
    assert manifest and Path(manifest).is_file(), block
    e2e.extra_files.append(Path(manifest))

    # 2) Relance à l'identique : rien n'est renvoyé, aucune collection en double.
    second = _run_cli(args, live_key)
    assert second.code == 0, second.tail()
    block = _result_block(second.out)
    assert _value(block, "Status") == "success", block
    assert _value(block, "Inserted") == "0", block
    assert _value(block, "Skipped (existing)") == str(inserted), block
    listed = live_client.list_collections(name=probe_collection_name)
    assert len(listed) == 1 and col.is_private(listed[0])
    assert (first.key_leaked, second.key_leaked) == (False, False)
    e2e.done.add("push")


def test_e2e_5_no_key_in_written_files(e2e, live_key):
    """Aucune trace de la clé dans les fichiers écrits par les scripts, ni dans ``logs/``.

    Contrôle tout l'espace temporaire et la partie des fichiers de ``logs/``
    écrite depuis la création de l'espace, même après l'échec d'une étape. Les
    fichiers attendus des étapes réussies doivent figurer parmi les fichiers
    lus (contrôle jamais vide). Seuls des libellés de fichiers sortent en cas
    d'échec, jamais la valeur de la clé.
    """
    entries = e2e.written_files()
    names = {label.rsplit("/", 1)[-1] for label, _, _ in entries}
    expected: Set[str] = set()
    for step, produced in STEP_OUTPUTS.items():
        if step in e2e.done:
            expected.update(produced)
    missing = sorted(expected - names)
    assert not missing, f"fichiers attendus absents du contrôle : {missing}"

    needle = live_key.encode("utf-8")
    leaked: List[str] = []
    unreadable: List[str] = []
    for label, path, offset in entries:
        try:
            if _file_contains(path, offset, needle):
                leaked.append(label)
        except OSError:
            unreadable.append(label)
    assert not unreadable, f"fichiers illisibles, fuite non vérifiable : {unreadable}"
    assert not leaked, f"clé présente dans : {leaked}"
