"""Tests des gabarits Albert (lot 7, tâche 10) : blocs Jinja en ligne, OFF identique à G9.

Couverture (``.claude/tasks/SPRINT_albert.md``, lot 7) :

* Albert OFF (interrupteur absent, ``0`` ou ``false``) : les pages pipeline,
  profil et détail de projet rendues pour un admin et un non-admin ont
  exactement l'empreinte du golden G9 et ne contiennent pas le mot
  « albert » ;
* Albert ON : aides ``albert/<modèle>`` et ``<datalist>`` alimentée par
  ``/api/albert/models``, sélecteur ``embedding_provider``, option
  ``albert`` de la base vectorielle et bloc ``#albertParams``, suppression
  de collection confirmée, section base déverrouillée dès le chunking,
  aide ``albert/<modèle>`` du détail de projet ;
* confirmation RGPD présente : rétention jusqu'à suppression, collection
  privée, DINUM sous-traitant (art. 28), droits d'auteur et données
  personnelles, case ``albert_gdpr_ack`` ;
* aucune clé Albert rendue dans le HTML (base, env serveur, fichier ``.env``) ;
* aucun gabarit ni script statique n'appelle l'hôte de l'API Albert (CORS) :
  les appels passent par les routes ``/api/albert/*`` de RAGpy.

Harnais : ``golden_app`` et l'hygiène d'env de
``tests/test_albert_off_golden_routes.py`` (identifiants factices, clé
Albert présente en base et dans l'env, Albert OFF par défaut).

Run: pytest tests/test_albert_ui.py -q
"""

from __future__ import annotations

import hashlib
import html
import os
import re
from pathlib import Path

import pytest

from tests import test_albert_off_golden_routes as golden
# Fixtures réutilisées (l'hygiène d'env des goldens est autouse dans ce module).
from tests.test_albert_off_golden_routes import _golden_env_hygiene, golden_app  # noqa: F401

# ---------------------------------------------------------------------------
# Constantes littérales
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = REPO_ROOT / "app" / "templates"
STATIC_DIR = REPO_ROOT / "app" / "static"
G9_GOLDEN = "g9_pages_html.json"
G9_PERSONAS = ("admin", "member_nokeys")

# Hôte de l'API Albert (et variante réécrite) : jamais dans un gabarit.
ALBERT_API_HOSTS = ("albert.api.etalab.gouv.fr", "api.albert.")
# Seul lien externe Albert admis : le playground (création de clé), en href.
ALLOWED_ALBERT_LINK_HOSTS = ("albert.playground.etalab.gouv.fr",)
ETALAB_URL_RE = re.compile(r"https?://([A-Za-z0-9.-]*etalab\.gouv\.fr)[^\s\"'<>`)]*")
FETCH_ABSOLUTE_RE = re.compile(r"fetch\(\s*[`'\"](https?:)?//", re.IGNORECASE)

# Marqueurs des blocs ON de la page pipeline (index.html). L'attribut ``list=``
# des champs de modèle est posé par le script (``setAttribute``), pas dans le
# HTML rendu : aucun marqueur ne le cherche.
INDEX_ON_MARKERS = {
    "option albert de la base vectorielle": re.compile(r"<option[^>]*value=[\"']albert[\"']"),
    "option albert du fournisseur d'embeddings": re.compile(
        r"<select[^>]*name=[\"']embedding_provider[\"'][^>]*>(?:(?!</select>)[\s\S])*"
        r"<option[^>]*value=[\"']albert[\"']"
    ),
    "bloc #albertParams": re.compile(r"id=[\"']albertParams[\"']"),
    "datalist des modèles": re.compile(r"<datalist[^>]*id=[\"']albertModelOptions[\"']"),
    "catalogue /api/albert/models": re.compile(r"/api/albert/models"),
    "sélecteur embedding_provider": re.compile(r"name=[\"']embedding_provider[\"']"),
    "embedding_provider transmis au FormData": re.compile(r"append\(\s*['\"]embedding_provider['\"]"),
    "listing des collections": re.compile(r"/api/albert/collections"),
    "suppression confirmée": re.compile(r"\?confirm=true"),
    "méthode DELETE": re.compile(r"['\"]DELETE['\"]"),
    "confirmation de suppression en ligne": re.compile(r"id=[\"']albertDeleteConfirm[\"']"),
    "case d'acquittement RGPD": re.compile(r"name=[\"']albert_gdpr_ack[\"']"),
    "case d'acquittement RGPD (id)": re.compile(r"<input[^>]*id=[\"']albertGdprAck[\"']"),
    "aide albert/<modèle>": re.compile(r"albert/(&lt;|<|[a-z0-9])", re.IGNORECASE),
}
PROJECT_ON_MARKERS = {
    "aide albert/<modèle> du détail de projet": re.compile(r"albert/(&lt;|<|[a-z0-9])", re.IGNORECASE),
}
# Mentions RGPD exigées (texte visible, entités décodées, casse et apostrophes normalisées).
GDPR_MENTIONS = {
    "rétention jusqu'à suppression": re.compile(r"jusqu'(à|a) (la |leur )?suppression"),
    "collection privée": re.compile(r"priv(é|e)e"),
    "DINUM": re.compile(r"dinum"),
    "article 28 RGPD": re.compile(r"art(icle|\.)? ?28( du)? rgpd"),
    "droits d'auteur": re.compile(r"droits? d'auteur"),
    "données personnelles": re.compile(r"donn(é|e)es personnelles"),
}

FAKE_ENV_FILE_ALBERT_KEY = "fake-albert-env-file-0003"
FAKE_SERVER_ALBERT_KEY = "fake-albert-server-env-0003"


# ---------------------------------------------------------------------------
# Aides
# ---------------------------------------------------------------------------
def _pages(env):
    """Les 4 pages du golden G9 : (gabarit, URL)."""
    pid = env.project.id
    return (
        ("index.html", "/pipeline"),
        ("index.html:project", f"/pipeline?project={pid}&session=gsess-full"),
        ("user/profile.html", "/profile"),
        ("user/project_detail.html", f"/project/{pid}"),
    )


def _render(env, url, persona):
    """HTML normalisé (chemins, dates) d'une page rendue pour ``persona``."""
    resp = env.client.get(url, headers=env.headers[persona], follow_redirects=False)
    assert resp.status_code == 200, (url, persona, resp.status_code)
    assert golden._take_server_errors(env) == []
    return env.norm.text(resp.text)


def _visible_text(page):
    """Texte d'une page : entités décodées, minuscules, apostrophes typographiques normalisées."""
    text = html.unescape(page).lower()
    return text.replace("’", "'").replace(" ", " ")


def _all_fake_albert_keys(env):
    """Toutes les clés Albert factices connues du harnais (base, env serveur, fichier .env)."""
    keys = {golden.FAKE_ALBERT_KEY, FAKE_ENV_FILE_ALBERT_KEY, FAKE_SERVER_ALBERT_KEY}
    for creds in golden.FAKE_DB_CREDENTIALS.values():
        if creds.get("albert_api_key"):
            keys.add(creds["albert_api_key"])
    return sorted(keys)


def _template_files():
    """Gabarits HTML et scripts statiques du dépôt."""
    files = sorted(TEMPLATES_DIR.rglob("*.html"))
    if STATIC_DIR.is_dir():
        files += sorted(STATIC_DIR.rglob("*.js"))
    return files


# ---------------------------------------------------------------------------
# Albert OFF : HTML identique au golden G9
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("switch", ["0", None, "false"])
def test_off_html_equals_golden(golden_app, monkeypatch, switch):
    if switch is None:
        monkeypatch.delenv("ALBERT_ENABLED", raising=False)
    else:
        monkeypatch.setenv("ALBERT_ENABLED", switch)
    env = golden_app
    expected = {case["case"]: case for case in golden._read_golden(G9_GOLDEN)["cases"]}
    cases, htmls = [], {}
    for template, url in _pages(env):
        for persona in G9_PERSONAS:
            page = _render(env, url, persona)
            case = f"{template}:{persona}"
            htmls[case] = page
            cases.append({"case": case, "sha256": hashlib.sha256(page.encode("utf-8")).hexdigest()})
    mismatched = [c["case"] for c in cases if expected.get(c["case"], {}).get("sha256") != c["sha256"]]
    hint = ""
    if mismatched:
        hint = golden._g9_line_diagnostics(cases, htmls, golden._read_golden(golden.G9_LINES_GOLDEN)["pages"])
    assert mismatched == [], f"pages OFF différentes du golden G9 : {mismatched}{hint}"
    for case, page in htmls.items():
        assert "albert" not in page.lower(), case
    assert env.recorder.take() == []


# ---------------------------------------------------------------------------
# Albert ON : blocs présents
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("persona", G9_PERSONAS)
def test_on_blocks_present(golden_app, monkeypatch, persona):
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    env = golden_app
    pages = dict(_pages(env))
    index = _render(env, pages["index.html"], persona)
    missing = [label for label, pattern in INDEX_ON_MARKERS.items() if not pattern.search(index)]
    assert missing == [], f"blocs Albert absents de index.html : {missing}"
    # La section base est déverrouillée dès le chunking, pour la seule cible albert.
    assert re.search(r"unlockSection\(\s*['\"]db-section['\"]\s*\)", index), "déverrouillage anticipé absent"

    project = _render(env, pages["user/project_detail.html"], persona)
    missing = [label for label, pattern in PROJECT_ON_MARKERS.items() if not pattern.search(project)]
    assert missing == [], f"aide Albert absente du détail de projet : {missing}"

    # ON ne retire rien : les options historiques restent présentes.
    for value in ("pinecone", "weaviate", "qdrant"):
        assert re.search(r"<option[^>]*value=[\"']" + value + r"[\"']", index), value

    # La page de pipeline ON ne diffère de la page OFF que par des ajouts.
    monkeypatch.setenv("ALBERT_ENABLED", "0")
    off = _render(env, pages["index.html"], persona)
    on_lines = index.split("\n")
    off_lines = off.split("\n")
    assert len(on_lines) > len(off_lines)
    pos = 0
    for line in off_lines:
        found = False
        while pos < len(on_lines):
            candidate = on_lines[pos]
            pos += 1
            if candidate == line or candidate.startswith(line):
                found = True
                break
        assert found, f"ligne OFF introuvable dans la page ON : {line[:160]}"


def test_on_gdpr_confirm_present(golden_app, monkeypatch):
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    env = golden_app
    index = _render(env, "/pipeline", "member_nokeys")
    text = _visible_text(index)
    missing = [label for label, pattern in GDPR_MENTIONS.items() if not pattern.search(text)]
    assert missing == [], f"mentions RGPD absentes : {missing}"
    # La case d'acquittement est une case à cocher envoyée comme albert_gdpr_ack.
    checkbox = re.search(r"<input[^>]*name=[\"']albert_gdpr_ack[\"'][^>]*>", index)
    assert checkbox and re.search(r"type=[\"']checkbox[\"']", checkbox.group(0))
    assert "albert_gdpr_ack" in re.sub(r"<input[^>]*>", "", index), "acquittement non transmis par le script d'envoi"

    monkeypatch.setenv("ALBERT_ENABLED", "0")
    off_text = _visible_text(_render(env, "/pipeline", "member_nokeys"))
    assert "albert_gdpr_ack" not in off_text and "dinum" not in off_text


def test_key_never_rendered(golden_app, monkeypatch):
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    monkeypatch.setenv("ALBERT_API_KEY", FAKE_SERVER_ALBERT_KEY)
    env = golden_app
    (env.settings_home / ".env").write_text(
        f"# fake settings file\nALBERT_API_KEY={FAKE_ENV_FILE_ALBERT_KEY}\nOTHER_SETTING=fake-other-0003\n",
        encoding="utf-8",
    )
    keys = _all_fake_albert_keys(env)
    visits = [(persona, url) for persona in G9_PERSONAS for _template, url in _pages(env)]
    # member_keys (clé Albert en base) n'est pas membre du projet : pages hors projet.
    visits += [("member_keys", "/pipeline"), ("member_keys", "/profile")]
    for persona, url in visits:
        page = _render(env, url, persona)
        leaked = [True for key in keys if key in page]
        assert not leaked, (persona, url)
        # Pas même la fin de clé du masque « •••• » (réservé au JSON des identifiants).
        assert not re.search(r"••••[A-Za-z0-9_-]{4}", page), (persona, url)


def test_no_albert_host_in_templates():
    files = _template_files()
    assert files, "aucun gabarit trouvé"
    for path in files:
        text = path.read_text(encoding="utf-8")
        rel = os.path.relpath(str(path), str(REPO_ROOT))
        for host in ALBERT_API_HOSTS:
            assert host not in text, (rel, host)
        assert not FETCH_ABSOLUTE_RE.search(text), (rel, "fetch vers une URL absolue")
        for match in ETALAB_URL_RE.finditer(text):
            host = match.group(1).lower()
            assert host in ALLOWED_ALBERT_LINK_HOSTS, (rel, host)
            before = text[max(0, match.start() - 8):match.start()]
            assert "href=" in before, (rel, "lien etalab hors d'un href")
        # Les appels Albert du navigateur passent tous par les routes /api/albert/* de RAGpy.
        for call in re.finditer(r"fetch\(\s*[`'\"]([^`'\"]*)", text):
            url = call.group(1)
            if "albert" in url.lower():
                assert url.startswith("/api/albert/"), (rel, url)
