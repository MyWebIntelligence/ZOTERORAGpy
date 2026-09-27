"""Journal d'usage Albert côté web (lot 9, tâche 0) : contrat des utilitaires.

Contrat gelé du journal d'usage web : les points d'entrée des notes (courtes,
étendues, résumés), des fiches de livre (trois phases) et du filtre de
citations (pré-filtre, filtre complet, traitement parallèle) acceptent un
dernier argument nommé ``albert_usage_ledger=None`` (un ``UsageLedger``).

* Chemin Albert : le journal est remis à chaque ``AlbertClient`` construit pour
  la requête (``ledger=``) ; chaque appel y est inscrit avec l'id **épinglé**
  envoyé (jamais l'alias saisi), le ``response.model`` renvoyé, les tokens, le
  coût et les impacts lus dans la réponse ; les échecs sont comptés.
* ``None`` : construction historique des clients (aucun ``ledger=``), corps des
  requêtes identiques, aucun fichier ``albert_usage.jsonl`` (l'écriture est
  réservée aux routes, jamais faite par les utilitaires).
* Chemins OpenAI/OpenRouter : le journal n'est jamais touché (journal piège),
  aucun client Albert n'est construit, et le golden G11 est rejoué à
  l'identique avec le journal passé en argument, Albert désactivé puis activé.

``LedgerFakeAlbert`` (sous-classe locale de ``tests.albert_fakes.FakeAlbert`` :
coût non nul, écho de l'alias dans ``response.model``, réponses conservées) est
branché sur tout ``AlbertClient`` construit pendant le test. Les coroutines sont
exécutées avec ``asyncio.run`` ; aucun test n'attend réellement.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import sys
from types import SimpleNamespace

import httpx
import pytest

_THIS = os.path.dirname(os.path.abspath(__file__))
RAGPY_ROOT = os.path.dirname(_THIS)
for _p in (RAGPY_ROOT, os.path.join(RAGPY_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tests import test_albert_off_golden as golden  # noqa: E402
from tests.test_albert_notes import _assert_matches_golden_json, _g11_entries  # noqa: E402

from app.utils import book_note_generator as bng  # noqa: E402
from app.utils import citation_fetcher  # noqa: E402
from app.utils import parallel_citation_processor as pcp  # noqa: E402
from scripts.rad_albert import catalog as albert_catalog  # noqa: E402
from scripts.rad_albert import limiter as albert_limiter  # noqa: E402
from scripts.rad_albert import preflight as albert_preflight  # noqa: E402
from scripts.rad_albert.errors import AlbertAuthError  # noqa: E402
from scripts.rad_albert.usage import USAGE_FILENAME, UsageLedger  # noqa: E402
from tests.albert_fakes import FAKE_ALBERT_KEY, RATELIMIT_HEADERS, FakeAlbert  # noqa: E402

lng = golden.lng
cfilter = golden.cfilter

CHAT_PATH = "/v1/chat/completions"
SMALL_ALIAS = "openweight-small"
SMALL_MODEL = "ministral-3-8b-instruct-2512"
LARGE_ALIAS = "openweight-large"
REASONING_MODEL = "gpt-oss-120b"
ALIASES = [(SMALL_ALIAS, SMALL_MODEL), (LARGE_ALIAS, REASONING_MODEL)]
FAKE_OPENAI_KEY = golden.FAKE_ENV["OPENAI_API_KEY"]
FAKE_OPENROUTER_KEY = golden.FAKE_ENV["OPENROUTER_API_KEY"]
COST_PER_TOKEN = 2.5e-06
ABSENT = object()
HIGH_RATES = {
    "ALBERT_RECODE_RPM": "1000000",
    "ALBERT_NOTES_RPM": "1000000",
    "ALBERT_OCR_RPM": "1000000",
    "ALBERT_EMBED_RPM": "1000000",
    "ALBERT_CHAT_TPM": "1000000000",
}
TEMPLATE_MARK = "Fiche générée automatiquement (template)"
METADATA = {
    "title": "Le langage ordinaire",
    "authors": "Dupont, Jeanne",
    "date": "2021",
    "abstract": "Résumé de l'article sur le langage ordinaire.",
    "language": "fr",
}
TEXT = "Texte intégral de l'article. " * 40
NOTE_HTML = "<h2>Fiche</h2><p>Analyse de l'article sur le langage ordinaire.</p>"
BOOK_METADATA = {
    "title": "Le langage ordinaire, un livre",
    "authors": "Dupont, Jeanne",
    "date": "2021",
    "language": "fr",
    "itemType": "book",
}
PREFILTER_MARK = 'Réponds UNIQUEMENT par "RELEVANT" ou "NA"'
PROJECT = {
    "project_name": "Langage ordinaire",
    "project_description": "Philosophie du langage ordinaire et sociologie des usages.",
    "collection_name": "Corpus",
    "collection_description": "Articles retenus.",
}
FILTER_JSON = {
    "relevance_score": 82,
    "relevance_reason": "Article central pour le projet.",
    "zotero_item": {
        "itemType": "journalArticle",
        "title": "Le langage ordinaire",
        "creators": [{"creatorType": "author", "firstName": "Jeanne", "lastName": "Dupont"}],
        "date": "2021",
    },
}
FILTER_REPLY = "```json\n" + json.dumps(FILTER_JSON, ensure_ascii=False) + "\n```"


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------
class LedgerFakeAlbert(FakeAlbert):
    """``FakeAlbert`` au coût non nul, qui conserve ses réponses de chat et peut renvoyer un alias.

    Attributes:
        chat_bodies: corps JSON des réponses de chat 200 servies par la route
            (hors injections), dans l'ordre d'envoi.
        response_model_echo: valeur imposée au champ ``model`` des réponses
            (l'API recopie le nom qu'elle reçoit, alias compris, D5), ou ``None``.
    """

    def __init__(self, *args, **kwargs):
        """Construit le faux ; aucun écho d'alias, aucune réponse conservée."""
        super().__init__(*args, **kwargs)
        self.chat_bodies = []
        self.response_model_echo = None

    def _usage(self, prompt_tokens, completion_tokens, *, carbon=True):
        """Usage du faux, avec un coût proportionnel aux tokens (jamais nul)."""
        usage = super()._usage(prompt_tokens, completion_tokens, carbon=carbon)
        usage["cost"] = round((prompt_tokens + completion_tokens) * COST_PER_TOKEN, 9)
        return usage

    def _route_post_v1_chat_completions(self, call):
        """Réponse de chat du faux, conservée, avec l'écho d'alias éventuel."""
        response = super()._route_post_v1_chat_completions(call)
        if response.status_code != 200:
            return response
        body = json.loads(response.content)
        if self.response_model_echo is not None:
            body["model"] = self.response_model_echo
        with self._lock:
            self.chat_bodies.append(body)
        return httpx.Response(200, json=body, headers=RATELIMIT_HEADERS)


class TripwireLedger:
    """Journal piège : tout accès (attribut, taille, vérité) est consigné puis refusé.

    Le code appelant peut avaler l'exception (repli « template », pré-filtre
    « pertinent ») : le test vérifie donc ``touched``, et non l'exception.
    """

    def __init__(self):
        """Crée le piège, sans accès consigné."""
        object.__setattr__(self, "touched", [])

    def __getattribute__(self, name):
        """Consigne et refuse tout attribut autre que ``touched``."""
        if name == "touched":
            return object.__getattribute__(self, name)
        object.__getattribute__(self, "touched").append(name)
        raise AssertionError(f"journal d'usage touché hors chemin Albert : {name}")

    def __len__(self):
        """Consigne et refuse ``len()``."""
        self.touched.append("__len__")
        raise AssertionError("journal d'usage touché hors chemin Albert : __len__")

    def __bool__(self):
        """Consigne et refuse le test de vérité."""
        self.touched.append("__bool__")
        raise AssertionError("journal d'usage touché hors chemin Albert : __bool__")


def _client_classes():
    """Classes ``AlbertClient`` (import ``scripts.rad_albert`` et, s'il est chargé, import CLI ``rad_albert``)."""
    importlib.import_module("scripts.rad_albert.client")
    classes = []
    for name in ("scripts.rad_albert.client", "rad_albert.client"):
        module = sys.modules.get(name)
        cls = getattr(module, "AlbertClient", None) if module is not None else None
        if cls is not None and cls not in classes:
            classes.append(cls)
    return classes


def _no_sleep(seconds):
    """Sommeil synchrone factice des réessais du client (aucune attente)."""
    return None


def route_albert_to_fake(monkeypatch, fake, builds):
    """Branche tout ``AlbertClient`` construit sur ``fake`` ; ``builds`` reçoit le ``ledger`` passé (ou ``ABSENT``)."""
    for cls in _client_classes():
        original = cls.__init__

        def patched(self, *args, __original=original, **kwargs):
            """Note le journal reçu, puis construit sur le transport du faux, sans sommeil réel."""
            builds.append(kwargs["ledger"] if "ledger" in kwargs else ABSENT)
            if kwargs.get("transport") is None:
                kwargs["transport"] = fake.transport
            kwargs.setdefault("sleep", _no_sleep)
            __original(self, *args, **kwargs)

        monkeypatch.setattr(cls, "__init__", patched)


def _module_semaphores():
    """Sémaphores asyncio tenus au niveau du module des notes (global et Albert)."""
    return {name: value for name, value in vars(lng).items() if isinstance(value, asyncio.Semaphore)}


async def _no_fetch(*args, **kwargs):
    """Double de ``fetch_citation_content`` : aucun contenu web, aucune requête."""
    return "", "none"


async def _collect(generator):
    """Consomme un générateur asynchrone d'événements et renvoie la liste."""
    return [event async for event in generator]


def _book_text(chapters=3):
    """Texte de livre OCR avec marqueurs de page et ``chapters`` titres de chapitre markdown."""
    parts = []
    page = 1
    for num in range(1, chapters + 1):
        parts.append(f"<!-- Page {page} -->")
        parts.append(f"# Chapitre {num}. Titre du chapitre {num}")
        body = " ".join(
            f"Paragraphe {i} du chapitre {num} : l'analyse du langage ordinaire progresse."
            for i in range(1, 25)
        )
        parts.append(body)
        page += 1
        parts.append(f"<!-- Page {page} -->")
        parts.append(body)
        page += 1
    return "\n\n".join(parts)


def _book_phase(user):
    """Phase d'une fiche de livre reconnue dans l'invite (``phase1``, ``phase2`` ou ``phase3``)."""
    if "===SECTION_A===" in user:
        return "phase3"
    if "SUMMARY:" in user:
        return "phase2"
    return "phase1"


def _reply_for(user):
    """Réponse commune (Albert et OpenAI) : pré-filtre, filtre, phases de livre, sinon une note."""
    if PREFILTER_MARK in user:
        return "RELEVANT"
    if "relevance_score" in user:
        return FILTER_REPLY
    phase = _book_phase(user)
    if phase == "phase3":
        return ("===SECTION_A===\n<h3>1. Identification</h3><p>Livre.</p>\n"
                "===SECTION_B===\n<h3>4. Synthèse</h3><p>Synthèse.</p>\n===END===")
    if phase == "phase2":
        return "<h3>Chapitre</h3><p>Analyse du chapitre.</p>\nSUMMARY: résumé du chapitre."
    if "TOC_RAW" in user or "book_type" in user:
        return '{"book_type": "monograph", "chapters": []}'
    return NOTE_HTML


def _fake_reply(body):
    """Réponse de ``FakeAlbert`` selon l'invite utilisateur."""
    return _reply_for(body["messages"][-1]["content"])


def _openai_double(calls):
    """Remplaçant de ``_get_llm_clients`` : clients OpenAI/OpenRouter enregistreurs, réponses de ``_reply_for``."""
    def fake_get_llm_clients(*args, **extra):
        """Renvoie deux clients enregistreurs et le modèle par défaut historique."""
        def content(label, kwargs):
            """Contenu selon l'invite (même règle que le faux Albert)."""
            return _reply_for(kwargs["messages"][-1]["content"])

        return (golden._FakeLLMClient("openai", calls, content),
                golden._FakeLLMClient("openrouter", calls, content),
                "gpt-4o-mini")
    return fake_get_llm_clients


def _albert_env(monkeypatch, enabled):
    """Albert ON ou OFF, débits élevés ; la clé n'est jamais posée dans l'environnement."""
    monkeypatch.setenv("ALBERT_ENABLED", "1" if enabled else "0")
    for name, value in HIGH_RATES.items():
        monkeypatch.setenv(name, value)


def _citation(i=1):
    """Citation Publish or Perish minimale (sans URL : aucune récupération web)."""
    return {
        "title": f"Le langage ordinaire, étude {i}",
        "authors": ["Dupont, Jeanne"],
        "abstract": "Étude des usages du langage ordinaire.",
        "source": "Revue de philosophie",
        "year": 2021,
        "cites": 3,
    }


def _prefilter(model, **kwargs):
    """Coroutine de pré-filtrage d'une citation pour le projet de test."""
    return cfilter.pre_filter_citation(
        _citation(), PROJECT["project_name"], PROJECT["project_description"],
        PROJECT["collection_name"], PROJECT["collection_description"], model=model, **kwargs,
    )


def _full_filter(model, **kwargs):
    """Coroutine de filtrage complet d'une citation pour le projet de test."""
    return cfilter.filter_citation_with_llm(
        _citation(), "", "none", PROJECT["project_name"], PROJECT["project_description"],
        PROJECT["collection_name"], PROJECT["collection_description"], model=model, **kwargs,
    )


def _chat_calls(fake):
    """Appels de chat reçus par le faux."""
    return fake.calls_to("POST", CHAT_PATH)


def _usage_files(root):
    """Fichiers ``albert_usage.jsonl`` présents sous ``root``."""
    return sorted(str(p) for p in root.rglob(USAGE_FILENAME))


def _assert_records_match_responses(ledger, fake, *, model, roles):
    """Chaque réponse de chat servie est inscrite une fois, avec ses tokens, son coût et l'id épinglé.

    Args:
        ledger: journal rempli par l'appel testé.
        fake: faux Albert (réponses conservées dans ``chat_bodies``).
        model: id épinglé attendu dans ``record["model"]`` (jamais l'alias).
        roles: rôle attendu par identifiant de réponse, ou un seul rôle pour tous.
    """
    bodies = {body["id"]: body for body in fake.chat_bodies}
    records = list(ledger)
    assert sorted(r["request_id"] for r in records) == sorted(bodies)
    for record in records:
        body = bodies[record["request_id"]]
        usage = body["usage"]
        expected_role = roles if isinstance(roles, str) else roles[record["request_id"]]
        assert record["endpoint"] == CHAT_PATH
        assert record["role"] == expected_role
        assert record["model"] == model
        assert record["response_model"] == body["model"]
        assert record["status"] == 200
        assert record["finish_reason"] == "stop"
        assert record["fallback_from"] is None
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            assert record[name] == usage[name] and record[name] > 0, name
        assert record["cost"] == pytest.approx(usage["cost"]) and record["cost"] > 0
        assert record["impacts"]["kWh"] == pytest.approx(usage["impacts"]["kWh"])
        assert record["impacts"]["kgCO2eq"] == pytest.approx(usage["impacts"]["kgCO2eq"])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """Identifiants retirés, sémaphores, limiteurs et preflight remis à zéro, web neutralisé, cwd isolé."""
    for name in golden.BASELINE_ENV_NAMES + golden.EXTRA_CLEARED_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.startswith(golden.CLEARED_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    for name in list(_module_semaphores()):
        monkeypatch.setattr(lng, name, None)
    monkeypatch.setattr(citation_fetcher, "fetch_citation_content", _no_fetch)
    monkeypatch.chdir(tmp_path)
    albert_limiter.reset_limiters()
    albert_preflight.clear_preflight_cache()
    yield
    albert_limiter.reset_limiters()
    albert_preflight.clear_preflight_cache()


@pytest.fixture
def write_spy(monkeypatch):
    """Appels à ``UsageLedger.write_jsonl`` (les utilitaires ne doivent jamais écrire le journal)."""
    calls = []
    for name in ("scripts.rad_albert.usage", "rad_albert.usage"):
        module = sys.modules.get(name)
        if module is None:
            continue
        original = module.UsageLedger.write_jsonl

        def spy(self, path, __original=original):
            """Note l'écriture demandée, puis délègue."""
            calls.append(str(path))
            return __original(self, path)

        monkeypatch.setattr(module.UsageLedger, "write_jsonl", spy)
    return calls


@pytest.fixture
def albert_on(monkeypatch, write_spy):
    """Albert ON sur ``LedgerFakeAlbert`` ; clients OpenAI/OpenRouter enregistreurs (jamais appelés ici)."""
    fake = LedgerFakeAlbert()
    fake.chat_reply = _fake_reply
    builds = []
    route_albert_to_fake(monkeypatch, fake, builds)
    _albert_env(monkeypatch, enabled=True)
    openai_calls = []
    monkeypatch.setattr(lng, "_get_llm_clients", _openai_double(openai_calls))
    monkeypatch.setattr(cfilter, "_get_llm_clients", _openai_double(openai_calls))
    return SimpleNamespace(fake=fake, builds=builds, openai=openai_calls, writes=write_spy)


@pytest.fixture
def fast_async_sleep(monkeypatch):
    """``asyncio.sleep`` ramené à zéro (attentes non nulles enregistrées)."""
    original = asyncio.sleep
    records = []

    async def spy(delay, *args, **kwargs):
        """Enregistre les attentes non nulles puis cède la main sans attendre."""
        if delay and delay > 0:
            records.append(float(delay))
        return await original(0, *args, **kwargs)

    monkeypatch.setattr(asyncio, "sleep", spy)
    return records


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------
LEDGER_ENTRY_POINTS = (
    lng._generate_with_llm,
    lng.build_note_html,
    lng.build_abstract_text,
    lng.build_note_html_async,
    lng.build_abstract_text_async,
    bng._phase1_detect_structure,
    bng._phase2_analyse_chapter,
    bng._phase3_synthesise,
    bng.build_book_note_async,
    cfilter._call_llm_api,
    cfilter.pre_filter_citation,
    cfilter.filter_citation_with_llm,
    pcp.process_citations_parallel,
)


def test_ledger_is_trailing_keyword_only_default_none():
    """Le journal est le dernier paramètre, nommé seulement, ``None`` par défaut, après la clé Albert."""
    for func in LEDGER_ENTRY_POINTS:
        params = list(inspect.signature(func).parameters.values())
        assert params[-1].name == "albert_usage_ledger", func.__name__
        assert params[-1].kind is inspect.Parameter.KEYWORD_ONLY, func.__name__
        assert params[-1].default is None, func.__name__
        # La clé Albert reste juste avant : les appels positionnels historiques ne bougent pas.
        assert params[-2].name == "albert_api_key", func.__name__
        assert params[-2].default is None, func.__name__


def test_positional_calls_without_ledger_still_work(albert_on):
    """Les appels positionnels historiques restent valides ; le journal ne se passe jamais par position."""
    model = "albert/" + SMALL_ALIAS
    sentinel, html = lng.build_note_html(METADATA, TEXT, model, True, "short", None, None, FAKE_ALBERT_KEY)
    assert NOTE_HTML in html and TEMPLATE_MARK not in html
    assert lng.build_abstract_text(METADATA, TEXT, model, None, None, FAKE_ALBERT_KEY)
    assert lng._generate_with_llm("Invite.", model, 0.2, "short", None, None, FAKE_ALBERT_KEY) == NOTE_HTML
    assert cfilter._call_llm_api("Invite.", model, 0.2, None, None, FAKE_ALBERT_KEY) == NOTE_HTML
    config = dict(PROJECT, model=model)
    events = asyncio.run(_collect(pcp.process_citations_parallel(
        [_citation(1)], config, None, None, 10, 10, FAKE_ALBERT_KEY)))
    assert [kind for kind, _ in events] == ["init", "progress", "complete"]
    # Construction historique : aucun journal transmis au client.
    assert albert_on.builds and all(build is ABSENT for build in albert_on.builds)
    # Le journal ne se passe jamais par position (argument nommé seulement).
    with pytest.raises(TypeError):
        lng.build_note_html(METADATA, TEXT, model, True, "short", None, None, FAKE_ALBERT_KEY, UsageLedger())
    with pytest.raises(TypeError):
        cfilter._call_llm_api("Invite.", model, 0.2, None, None, FAKE_ALBERT_KEY, UsageLedger())


# ---------------------------------------------------------------------------
# Chemin Albert : notes (courtes et étendues)
# ---------------------------------------------------------------------------
def _note_async(mode):
    """Note asynchrone du mode donné (``build_note_html_async``)."""
    def run(model, **kwargs):
        """Exécute la coroutine et renvoie le HTML."""
        return asyncio.run(lng.build_note_html_async(METADATA, text_content=TEXT, model=model, mode=mode,
                                                     **kwargs))[1]
    return run


def _note_sync(mode):
    """Note synchrone du mode donné (``build_note_html``)."""
    def run(model, **kwargs):
        """Appelle la fonction et renvoie le HTML."""
        return lng.build_note_html(METADATA, text_content=TEXT, model=model, mode=mode, **kwargs)[1]
    return run


def _abstract_async(model, **kwargs):
    """Résumé asynchrone (``build_abstract_text_async``, mode court)."""
    return asyncio.run(lng.build_abstract_text_async(METADATA, text_content=TEXT, model=model, **kwargs))


def _abstract_sync(model, **kwargs):
    """Résumé synchrone (``build_abstract_text``, mode court)."""
    return lng.build_abstract_text(METADATA, text_content=TEXT, model=model, **kwargs)


def _generate(mode):
    """Appel direct de ``_generate_with_llm`` dans le mode donné."""
    def run(model, **kwargs):
        """Appelle la fonction et renvoie le contenu."""
        return lng._generate_with_llm("Invite de note.", model=model, mode=mode, **kwargs)
    return run


NOTE_ENTRY_POINTS = {
    "note_async_short": _note_async("short"),
    "note_async_extended": _note_async("extended"),
    "note_sync_short": _note_sync("short"),
    "note_sync_extended": _note_sync("extended"),
    "abstract_async": _abstract_async,
    "abstract_sync": _abstract_sync,
    "generate_short": _generate("short"),
    "generate_extended": _generate("extended"),
}


@pytest.mark.parametrize("alias,pinned", ALIASES)
@pytest.mark.parametrize("entry", sorted(NOTE_ENTRY_POINTS))
def test_notes_record_pinned_model_tokens_cost(entry, alias, pinned, albert_on, tmp_path):
    """Notes et résumés : un enregistrement par appel, id épinglé (pas l'alias), tokens et coût de la réponse."""
    assert albert_catalog.canonical_id(alias) == pinned
    fake = albert_on.fake
    fake.response_model_echo = alias
    ledger = UsageLedger()
    result = NOTE_ENTRY_POINTS[entry]("albert/" + alias, openai_api_key=FAKE_OPENAI_KEY,
                                      openrouter_api_key=FAKE_OPENROUTER_KEY, albert_api_key=FAKE_ALBERT_KEY,
                                      albert_usage_ledger=ledger)
    assert TEMPLATE_MARK not in result and "langage ordinaire" in result
    chats = _chat_calls(fake)
    assert len(chats) == len(ledger) == 1
    assert chats[0].json["model"] == pinned
    _assert_records_match_responses(ledger, fake, model=pinned, roles="notes")
    record = list(ledger)[0]
    assert record["model"] != alias and record["response_model"] == alias
    assert ledger.errors == 0 and ledger.called
    assert albert_on.builds and all(build is ledger for build in albert_on.builds)
    assert albert_on.openai == []
    assert albert_on.writes == [] and _usage_files(tmp_path) == []


@pytest.mark.parametrize("entry", ["note_async_extended", "generate_extended"])
def test_default_albert_model_records_into_ledger(entry, albert_on, monkeypatch):
    """Un modèle Albert par défaut (sans modèle explicite) alimente aussi le journal."""
    fake = albert_on.fake

    def default_model(openrouter_model=None):
        """Modèle web par défaut du test (jamais lu dans un fichier .env)."""
        return openrouter_model or "albert/" + SMALL_ALIAS

    def clients_with_albert_default(*args, **extra):
        """Aucun client OpenAI/OpenRouter ; modèle par défaut Albert."""
        return None, None, default_model()

    monkeypatch.setattr(lng, "resolve_default_llm_model", default_model)
    monkeypatch.setattr(lng, "_get_llm_clients", clients_with_albert_default)
    ledger = UsageLedger()
    result = NOTE_ENTRY_POINTS[entry](None, openai_api_key=None, openrouter_api_key=None,
                                      albert_api_key=FAKE_ALBERT_KEY, albert_usage_ledger=ledger)
    assert TEMPLATE_MARK not in result
    assert len(_chat_calls(fake)) == len(ledger) == 1
    _assert_records_match_responses(ledger, fake, model=SMALL_MODEL, roles="notes")
    assert all(build is ledger for build in albert_on.builds)


def test_retries_and_account_errors_are_counted(albert_on, fast_async_sleep):
    """Réessais : la réponse servie est inscrite, l'échec d'un envoi compté ; erreur de compte comptée et relevée."""
    fake = albert_on.fake
    ledger = UsageLedger()
    # Erreur transitoire puis succès : un enregistrement, un échec compté (réessai de run_llm_slot).
    fake.inject("POST", CHAT_PATH, "server_error")
    html = _note_async("short")("albert/" + SMALL_ALIAS, openai_api_key=None, openrouter_api_key=None,
                                albert_api_key=FAKE_ALBERT_KEY, albert_usage_ledger=ledger)
    assert TEMPLATE_MARK not in html
    assert len(_chat_calls(fake)) == 2
    assert len(ledger) == 1 and ledger.errors == 1
    _assert_records_match_responses(ledger, fake, model=SMALL_MODEL, roles="notes")
    # Chemin synchrone (réessais internes du client) : la réponse servie est inscrite.
    fake.inject("POST", CHAT_PATH, "server_error")
    lng._generate_with_llm("Invite.", model="albert/" + SMALL_ALIAS, mode="short",
                           albert_api_key=FAKE_ALBERT_KEY, albert_usage_ledger=ledger)
    assert len(_chat_calls(fake)) == 4
    assert len(ledger) == 2
    _assert_records_match_responses(ledger, fake, model=SMALL_MODEL, roles="notes")
    # Erreur de compte : relevée, comptée comme échec, aucun enregistrement détaillé ajouté.
    errors_before = ledger.errors
    fake.inject("POST", CHAT_PATH, "invalid_key")
    with pytest.raises(AlbertAuthError):
        _note_async("short")("albert/" + SMALL_ALIAS, openai_api_key=None, openrouter_api_key=None,
                             albert_api_key=FAKE_ALBERT_KEY, albert_usage_ledger=ledger)
    assert len(ledger) == 2 and ledger.errors == errors_before + 1
    assert albert_on.writes == []


# ---------------------------------------------------------------------------
# Chemin Albert : fiches de livre (trois phases)
# ---------------------------------------------------------------------------
def test_book_note_three_phases_record_into_ledger(albert_on, monkeypatch, tmp_path):
    """Fiche de livre : les trois phases reçoivent le journal et chaque appel y est inscrit, dans l'ordre."""
    fake = albert_on.fake
    fake.response_model_echo = SMALL_ALIAS
    seen = []
    for name in ("_phase1_detect_structure", "_phase2_analyse_chapter", "_phase3_synthesise"):
        real = getattr(bng, name)

        async def spy(*args, __real=real, __name=name, **kwargs):
            """Note le journal reçu par la phase, puis délègue."""
            seen.append((__name, kwargs.get("albert_usage_ledger", ABSENT)))
            return await __real(*args, **kwargs)

        monkeypatch.setattr(bng, name, spy)
    ledger = UsageLedger()
    sentinel, html = asyncio.run(bng.build_book_note_async(
        BOOK_METADATA, _book_text(), model="albert/" + SMALL_ALIAS,
        openai_api_key=FAKE_OPENAI_KEY, openrouter_api_key=FAKE_OPENROUTER_KEY,
        albert_api_key=FAKE_ALBERT_KEY, albert_usage_ledger=ledger,
    ))
    assert sentinel in html
    names = [name for name, _ in seen]
    assert names[0] == "_phase1_detect_structure" and names[-1] == "_phase3_synthesise"
    assert "_phase2_analyse_chapter" in names
    assert all(received is ledger for _, received in seen)
    chats = _chat_calls(fake)
    phases = [_book_phase(call.json["messages"][-1]["content"]) for call in chats]
    assert phases[0] == "phase1" and phases[-1] == "phase3" and "phase2" in phases
    assert len(chats) == len(ledger) == len(fake.chat_bodies) == len(seen)
    # Envoi séquentiel : l'ordre des enregistrements suit celui des requêtes.
    records = list(ledger)
    assert [r["request_id"] for r in records] == [b["id"] for b in fake.chat_bodies]
    roles = {body["id"]: ("book_structure" if phase == "phase1" else "notes")
             for body, phase in zip(fake.chat_bodies, phases)}
    _assert_records_match_responses(ledger, fake, model=SMALL_MODEL, roles=roles)
    assert all(r["response_model"] == SMALL_ALIAS for r in records)
    assert all(build is ledger for build in albert_on.builds)
    assert albert_on.openai == []
    assert albert_on.writes == [] and _usage_files(tmp_path) == []


# ---------------------------------------------------------------------------
# Chemin Albert : filtre de citations
# ---------------------------------------------------------------------------
def test_citation_prefilter_filter_and_call_record(albert_on, tmp_path):
    """Pré-filtre, filtre complet et appel direct du filtre : un enregistrement ``citation`` par appel."""
    fake = albert_on.fake
    fake.response_model_echo = SMALL_ALIAS
    model = "albert/" + SMALL_ALIAS
    keys = dict(openai_api_key=FAKE_OPENAI_KEY, openrouter_api_key=FAKE_OPENROUTER_KEY,
                albert_api_key=FAKE_ALBERT_KEY)
    ledger = UsageLedger()
    assert asyncio.run(_prefilter(model, albert_usage_ledger=ledger, **keys)) is True
    assert len(ledger) == 1
    result = asyncio.run(_full_filter(model, albert_usage_ledger=ledger, **keys))
    assert isinstance(result, dict) and result["relevance_score"] == 82
    assert len(ledger) == 2
    assert cfilter._call_llm_api("Invite.", model=model, albert_usage_ledger=ledger, **keys) == NOTE_HTML
    assert len(ledger) == len(_chat_calls(fake)) == 3
    assert all(call.json["model"] == SMALL_MODEL for call in _chat_calls(fake))
    _assert_records_match_responses(ledger, fake, model=SMALL_MODEL, roles="citation")
    assert all(r["response_model"] == SMALL_ALIAS for r in ledger)
    assert ledger.errors == 0
    assert all(build is ledger for build in albert_on.builds) and len(albert_on.builds) == 3
    assert albert_on.openai == []
    assert albert_on.writes == [] and _usage_files(tmp_path) == []


@pytest.mark.parametrize("alias,pinned", ALIASES)
def test_process_citations_parallel_records_every_call(alias, pinned, albert_on, tmp_path):
    """Traitement parallèle : pré-filtre et filtre de chaque citation inscrits dans le même journal."""
    fake = albert_on.fake
    fake.response_model_echo = alias
    ledger = UsageLedger()
    config = dict(PROJECT, model="albert/" + alias)
    citations = [_citation(i) for i in range(1, 4)]
    events = asyncio.run(_collect(pcp.process_citations_parallel(
        citations, config, FAKE_OPENAI_KEY, FAKE_OPENROUTER_KEY,
        albert_api_key=FAKE_ALBERT_KEY, albert_usage_ledger=ledger,
    )))
    progress = [data for kind, data in events if kind == "progress"]
    assert len(progress) == 3 and all(data["status"] == "relevant" for data in progress)
    # Pré-filtre puis filtre complet pour chacune des trois citations.
    chats = _chat_calls(fake)
    assert len(chats) == len(ledger) == 6
    prompts = [call.json["messages"][-1]["content"] for call in chats]
    assert sum(PREFILTER_MARK in p for p in prompts) == 3
    _assert_records_match_responses(ledger, fake, model=pinned, roles="citation")
    assert all(r["model"] != alias and r["response_model"] == alias for r in ledger)
    assert len(albert_on.builds) == 6 and all(build is ledger for build in albert_on.builds)
    assert albert_on.openai == []
    assert albert_on.writes == [] and _usage_files(tmp_path) == []


# ---------------------------------------------------------------------------
# Journal absent : aucun effet
# ---------------------------------------------------------------------------
def _albert_scenario(ledger_kwargs):
    """Rejoue notes, résumé, fiche de livre et filtre de citations sur Albert ; renvoie les sorties comparables."""
    model = "albert/" + SMALL_ALIAS
    keys = dict(openai_api_key=None, openrouter_api_key=None, albert_api_key=FAKE_ALBERT_KEY)
    out = {
        "note_async": asyncio.run(lng.build_note_html_async(
            METADATA, text_content=TEXT, model=model, mode="short", **keys, **ledger_kwargs))[1].split("\n", 1)[1],
        "note_sync": lng.build_note_html(
            METADATA, text_content=TEXT, model=model, mode="extended", **keys, **ledger_kwargs)[1].split("\n", 1)[1],
        "abstract": asyncio.run(lng.build_abstract_text_async(
            METADATA, text_content=TEXT, model=model, **keys, **ledger_kwargs)),
        "book": asyncio.run(bng.build_book_note_async(
            BOOK_METADATA, _book_text(), model=model, **keys, **ledger_kwargs))[1].count("<h3>"),
        "prefilter": asyncio.run(_prefilter(model, **keys, **ledger_kwargs)),
        "filter": asyncio.run(_full_filter(model, **keys, **ledger_kwargs)),
        "call": cfilter._call_llm_api("Invite.", model=model, **keys, **ledger_kwargs),
        "parallel": [data["status"] for kind, data in asyncio.run(_collect(pcp.process_citations_parallel(
            [_citation(1), _citation(2)], dict(PROJECT, model=model), None, None,
            albert_api_key=FAKE_ALBERT_KEY, **ledger_kwargs))) if kind == "progress"],
    }
    return out


def test_none_ledger_has_no_side_effect(albert_on, tmp_path):
    """Sans journal : construction historique des clients, requêtes et résultats identiques, aucun fichier."""
    fake = albert_on.fake
    without = _albert_scenario({})
    bodies_without = [call.json for call in _chat_calls(fake)]
    builds_without = list(albert_on.builds)
    fake.reset_calls()
    del albert_on.builds[:]
    explicit_none = _albert_scenario({"albert_usage_ledger": None})
    bodies_none = [call.json for call in _chat_calls(fake)]
    builds_none = list(albert_on.builds)
    fake.reset_calls()
    del albert_on.builds[:]
    ledger = UsageLedger()
    with_ledger = _albert_scenario({"albert_usage_ledger": ledger})
    bodies_ledger = [call.json for call in _chat_calls(fake)]
    # Sans journal : construction historique des clients (aucun ledger= transmis).
    assert builds_without and all(build is ABSENT for build in builds_without)
    assert builds_none and all(build is ABSENT for build in builds_none)
    # Le journal ne change ni les requêtes envoyées ni les résultats.
    assert explicit_none == without == with_ledger
    assert _sorted_bodies(bodies_none) == _sorted_bodies(bodies_without) == _sorted_bodies(bodies_ledger)
    assert len(ledger) == len(bodies_ledger)
    # Les utilitaires n'écrivent jamais le journal (écriture réservée aux routes).
    assert albert_on.writes == [] and _usage_files(tmp_path) == []
    assert albert_on.openai == []


def _sorted_bodies(bodies):
    """Corps de requête triés (ordre indépendant du parallélisme des citations)."""
    return sorted(json.dumps(body, sort_keys=True, ensure_ascii=False) for body in bodies)


# ---------------------------------------------------------------------------
# Chemins OpenAI / OpenRouter : journal jamais touché, G11 inchangé
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("model", ["gpt-4o-mini", "google/gemini-2.5-flash"])
def test_openai_path_never_touches_ledger(model, enabled, monkeypatch, write_spy, tmp_path):
    """Chemins OpenAI/OpenRouter : journal piège jamais touché, aucun client Albert, aucun fichier."""
    fake = LedgerFakeAlbert()
    builds = []
    route_albert_to_fake(monkeypatch, fake, builds)
    _albert_env(monkeypatch, enabled=enabled)
    calls = []
    monkeypatch.setattr(lng, "_get_llm_clients", _openai_double(calls))
    monkeypatch.setattr(cfilter, "_get_llm_clients", _openai_double(calls))
    trap = TripwireLedger()
    keys = dict(openai_api_key=FAKE_OPENAI_KEY, openrouter_api_key=FAKE_OPENROUTER_KEY,
                albert_api_key=FAKE_ALBERT_KEY, albert_usage_ledger=trap)
    for mode in ("short", "extended"):
        html = asyncio.run(lng.build_note_html_async(METADATA, text_content=TEXT, model=model, mode=mode, **keys))[1]
        assert TEMPLATE_MARK not in html
        html = lng.build_note_html(METADATA, text_content=TEXT, model=model, mode=mode, **keys)[1]
        assert TEMPLATE_MARK not in html
        assert lng._generate_with_llm("Invite.", model=model, mode=mode, **keys) == NOTE_HTML
    assert asyncio.run(lng.build_abstract_text_async(METADATA, text_content=TEXT, model=model, **keys))
    assert lng.build_abstract_text(METADATA, text_content=TEXT, model=model, **keys)
    _sentinel, book = asyncio.run(bng.build_book_note_async(BOOK_METADATA, _book_text(), model=model, **keys))
    assert "Analyse de ce chapitre indisponible" not in book
    assert cfilter._call_llm_api("Invite.", model=model, **keys) == NOTE_HTML
    assert asyncio.run(_prefilter(model, **keys)) is True
    assert isinstance(asyncio.run(_full_filter(model, **keys)), dict)
    events = asyncio.run(_collect(pcp.process_citations_parallel(
        [_citation(1), _citation(2)], dict(PROJECT, model=model), FAKE_OPENAI_KEY, FAKE_OPENROUTER_KEY,
        albert_api_key=FAKE_ALBERT_KEY, albert_usage_ledger=trap)))
    assert [data["status"] for kind, data in events if kind == "progress"] == ["relevant", "relevant"]
    assert calls, "le fournisseur historique doit avoir été appelé"
    assert trap.touched == []
    assert builds == [] and fake.calls == []
    assert write_spy == [] and _usage_files(tmp_path) == []


@pytest.mark.parametrize("enabled", [False, True])
def test_g11_golden_unchanged_with_ledger_argument(enabled, monkeypatch):
    """Golden G11 rejoué à l'identique avec le journal passé en argument (Albert OFF puis ON)."""
    fake = LedgerFakeAlbert()
    builds = []
    route_albert_to_fake(monkeypatch, fake, builds)
    _albert_env(monkeypatch, enabled=enabled)
    monkeypatch.setenv("ALBERT_API_KEY", FAKE_ALBERT_KEY)
    trap = TripwireLedger()
    entries = _g11_entries(monkeypatch, {"albert_api_key": FAKE_ALBERT_KEY, "albert_usage_ledger": trap})
    _assert_matches_golden_json("g11_llm_call_kwargs.json", entries)
    assert trap.touched == []
    assert builds == [] and fake.calls == []
