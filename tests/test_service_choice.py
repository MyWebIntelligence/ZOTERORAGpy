"""Règle serveur + modèle par service (sprint « configuration unifiée », lot L1).

Règle d'Amar : un serveur et un modèle par défaut ; un service qui ne donne
qu'un modèle l'envoie au serveur par défaut ; un serveur déclaré pour le
service prime. Le serveur est l'adresse de l'API ; le nom du modèle part tel
quel ; une adresse non déclarée au bloc 1 est refusée.
"""

from __future__ import annotations

import pytest

from scripts.rad_settings.models import (
    LOCAL_ENGINE_DEF,
    ServiceConfigError,
    match_server,
    normalize_url,
    resolve_service,
    split_legacy,
)

OPENROUTER = "https://openrouter.ai/api/v1"
MISTRAL = "https://api.mistral.ai/v1"
ALBERT = "https://albert.api.etalab.gouv.fr/v1"
OPENAI = "https://api.openai.com/v1"

BASE = {
    "OPENROUTER_API_BASE_URL": OPENROUTER,
    "MISTRAL_API_BASE_URL": MISTRAL,
    "ALBERT_BASE_URL": ALBERT,
    "LLM_DEFAULT_SERVER": OPENROUTER,
    "LLM_DEFAULT_MODEL": "google/gemini-3.8-flash",
}


def test_model_only_goes_to_the_default_server():
    """Un service qui ne donne qu'un modèle l'envoie au serveur par défaut."""
    values = dict(BASE, LLM_RECODE_MODEL="openai/gpt-4o-mini")
    choice = resolve_service("recode", values)
    assert (choice.server_url, choice.model) == (OPENROUTER, "openai/gpt-4o-mini")
    assert choice.server.key == "openrouter" and choice.key_var == "OPENROUTER_API_KEY"
    assert choice.model_from_default is False


def test_service_server_overrides_the_default_server():
    """Un serveur déclaré pour le service prime sur le serveur par défaut."""
    values = dict(BASE, LLM_NOTES_SERVER=MISTRAL, LLM_NOTES_MODEL="mistral-small-2603")
    choice = resolve_service("notes", values)
    assert (choice.server_url, choice.model, choice.server.key) == (MISTRAL, "mistral-small-2603", "mistral")


def test_empty_service_uses_both_defaults():
    """Service vide : serveur et modèle par défaut."""
    choice = resolve_service("citations", BASE)
    assert (choice.server_url, choice.model) == (OPENROUTER, "google/gemini-3.8-flash")


def test_specific_server_with_default_model_is_flagged():
    """Serveur propre au service mais modèle par défaut : signalé (noms différents)."""
    values = dict(BASE, LLM_BOOK_SERVER=MISTRAL)
    choice = resolve_service("book", values)
    assert choice.model == "google/gemini-3.8-flash" and choice.model_from_default is True


def test_priority_order_override_user_service_default():
    """Champ de l'étape > choix personnel > variable du service > défaut."""
    values = dict(BASE, LLM_NOTES_SERVER=MISTRAL, LLM_NOTES_MODEL="mistral-large-latest")
    user = {"LLM_NOTES_MODEL": "ministral-8b-latest"}
    assert resolve_service("notes", values, user_values=user).model == "ministral-8b-latest"
    chosen = resolve_service("notes", values, user_values=user, override_server=ALBERT, override_model="gpt-oss-120b")
    assert (chosen.server.key, chosen.model) == ("albert", "gpt-oss-120b")


def test_model_name_is_sent_as_is():
    """Le nom du modèle n'est jamais transformé (pas de préfixe à retirer)."""
    values = dict(BASE, LLM_RECODE_SERVER=ALBERT, LLM_RECODE_MODEL="albert/ministral-3-8b-instruct-2512")
    assert resolve_service("recode", values).model == "albert/ministral-3-8b-instruct-2512"


def test_unknown_server_address_is_refused():
    """Une adresse non déclarée au bloc 1 est refusée (la clé ne part pas ailleurs)."""
    values = dict(BASE, LLM_NOTES_SERVER="https://evil.example.org/v1")
    with pytest.raises(ServiceConfigError) as info:
        resolve_service("notes", values)
    assert info.value.variables == ("LLM_NOTES_SERVER",)
    assert "serveur inconnu" in str(info.value)


def test_url_normalization_matches_declared_address():
    """Casse du schéma et de l'hôte, ``/`` final : même serveur."""
    url, sdef = match_server("HTTPS://OpenRouter.AI/api/v1/", BASE)
    assert (url, sdef.key) == (OPENROUTER, "openrouter")


@pytest.mark.parametrize("declared", [MISTRAL, "https://api.mistral.ai"])
def test_mistral_address_with_or_without_v1_is_matched(declared):
    """Mistral : ``https://api.mistral.ai`` et ``…/v1`` désignent le même serveur, des deux côtés
    (comme ``MISTRAL_API_BASE_URL``) ; l'adresse rendue est celle du bloc 1."""
    values = dict(BASE, MISTRAL_API_BASE_URL=declared)
    url, sdef = match_server("https://api.mistral.ai", values)
    assert (url, sdef.key) == (normalize_url(declared), "mistral")
    choice = resolve_service("ocr_fallback", dict(values, OCR_SERVER_FALLBACK="https://api.mistral.ai/",
                                                  OCR_MODEL_FALLBACK="mistral-ocr-latest"))
    assert (choice.server_url, choice.server.key) == (normalize_url(declared), "mistral")


def test_v1_tolerance_is_mistral_only():
    """Les autres serveurs gardent la comparaison exacte de l'adresse."""
    with pytest.raises(ValueError, match="serveur inconnu"):
        match_server("https://openrouter.ai/api", BASE)


@pytest.mark.parametrize("bad", ["", "   ", "ftp://x.org", "https://", "https://a b.org", "https://a.org/v1?x=1",
                                 "https://a.org\\v1", "https://a.org/\x00"])
def test_invalid_addresses(bad):
    """Adresses vides, mal formées ou avec caractères de contrôle refusées."""
    with pytest.raises(ValueError):
        normalize_url(bad)


def test_missing_server_and_model_name_the_variables():
    """Sans serveur ni modèle, l'erreur nomme les variables à déclarer."""
    with pytest.raises(ServiceConfigError) as info:
        resolve_service("recode", {"OPENROUTER_API_BASE_URL": OPENROUTER})
    assert info.value.variables == ("LLM_RECODE_SERVER", "LLM_DEFAULT_SERVER")
    with pytest.raises(ServiceConfigError) as info:
        resolve_service("recode", {"OPENROUTER_API_BASE_URL": OPENROUTER, "LLM_DEFAULT_SERVER": OPENROUTER})
    assert info.value.variables == ("LLM_RECODE_MODEL", "LLM_DEFAULT_MODEL")


def test_non_chat_services_never_take_the_default_model():
    """OCR, embeddings, rerank, audio : modèle obligatoire, jamais le modèle de langage."""
    values = dict(BASE, OCR_SERVER=MISTRAL)
    with pytest.raises(ServiceConfigError) as info:
        resolve_service("ocr", values)
    assert info.value.variables == ("OCR_MODEL",)


def test_ocr_refused_on_a_server_without_ocr():
    """OpenRouter hérité comme serveur d'OCR : refus, il faut déclarer OCR_SERVER."""
    values = dict(BASE, OCR_MODEL="mistral-ocr-latest")
    with pytest.raises(ServiceConfigError) as info:
        resolve_service("ocr", values)
    assert info.value.variables == ("OCR_SERVER",)


def test_ocr_on_mistral_and_albert():
    """OCR sur Mistral et sur Albert : adresses déclarées, modèles tels quels."""
    mistral = resolve_service("ocr", dict(BASE, OCR_SERVER=MISTRAL, OCR_MODEL="mistral-ocr-latest"))
    assert (mistral.server.protocol, mistral.model) == ("mistral", "mistral-ocr-latest")
    albert = resolve_service("ocr", dict(BASE, OCR_SERVER=ALBERT, OCR_MODEL="lightonocr-2-1b"))
    assert (albert.server.protocol, albert.key_var) == ("albert", "ALBERT_API_KEY")


def test_local_keyword_for_internal_ocr_engines():
    """``local`` désigne Docling et PyMuPDF, sans clé ; tout autre nom est refusé."""
    choice = resolve_service("ocr", dict(BASE, OCR_SERVER="local", OCR_MODEL="docling"))
    assert choice.server is LOCAL_ENGINE_DEF and choice.server_url == "local" and choice.key_var == ""
    with pytest.raises(ServiceConfigError):
        resolve_service("ocr", dict(BASE, OCR_SERVER="local", OCR_MODEL="tesseract"))
    with pytest.raises(ServiceConfigError):
        resolve_service("recode", dict(BASE, LLM_RECODE_SERVER="local"))


def test_albert_custom_address_is_matched():
    """Albert à une autre adresse (OpenGateLLM auto-hébergé) : reconnu par ``ALBERT_BASE_URL``."""
    custom = "https://albert.example.gouv.fr/v1"
    values = dict(BASE, ALBERT_BASE_URL=custom, LLM_NOTES_SERVER=custom, LLM_NOTES_MODEL="gpt-oss-120b")
    assert resolve_service("notes", values).server.key == "albert"


def test_same_address_declared_twice_is_refused():
    """Deux serveurs ne peuvent pas déclarer la même adresse (clé ambiguë)."""
    values = dict(BASE, OPENAI_API_BASE_URL=OPENROUTER)
    with pytest.raises(ServiceConfigError):
        resolve_service("notes", values)


def test_unknown_service():
    """Service inconnu : erreur explicite."""
    with pytest.raises(ServiceConfigError):
        resolve_service("translation", BASE)


@pytest.mark.parametrize("legacy, expected", [
    ("", ("", "")),
    ("gpt-4o-mini", (OPENAI, "gpt-4o-mini")),
    ("google/gemini-2.5-flash", (OPENROUTER, "google/gemini-2.5-flash")),
    ("openai/gpt-4o-mini", (OPENROUTER, "openai/gpt-4o-mini")),
    ("mistralai/mistral-small-2603", (OPENROUTER, "mistralai/mistral-small-2603")),
    ("albert/ministral-3-8b-instruct-2512", (ALBERT, "ministral-3-8b-instruct-2512")),
    ("Albert/openai/gpt-oss-120b", (ALBERT, "openai/gpt-oss-120b")),
])
def test_split_legacy_keeps_todays_routing(legacy, expected):
    """D2 : séparation des anciennes valeurs en conservant l'acheminement actuel."""
    assert split_legacy(legacy, openai_url=OPENAI, openrouter_url=OPENROUTER, albert_url=ALBERT) == expected


def test_split_legacy_refuses_empty_albert_model():
    """``albert/`` seul : erreur, jamais un modèle vide."""
    with pytest.raises(ValueError):
        split_legacy("albert/ ", openai_url=OPENAI, openrouter_url=OPENROUTER, albert_url=ALBERT)
