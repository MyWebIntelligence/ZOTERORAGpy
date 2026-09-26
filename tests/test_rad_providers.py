"""Tests de ``scripts/rad_providers.py`` : résolveur LLM, parité historique, espaces d'embeddings.

La parité avec l'heuristique historique est vérifiée contre le **code réel** de
``rad_chunk.py`` : routage de ``gpt_recode_batch`` (clients factices) et libellé
de fournisseur passé à ``rad_recode_cache.recode_key`` par ``recode_batch_cached``.
Aucun réseau, aucun credential réel.
"""

import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

RAGPY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (RAGPY_ROOT, os.path.join(RAGPY_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts import rad_providers as rp  # noqa: E402
from scripts.rad_albert.errors import AlbertDisabledError  # noqa: E402

# rad_chunk demande la clé OpenAI par input() si elle manque à l'import : une clé
# factice est posée pour cet import seulement (motif des goldens), puis retirée.
_FAKE_OPENAI_KEY = "fake-openai-key-0001"
_IMPORT_KEY_SET = "OPENAI_API_KEY" not in os.environ
if _IMPORT_KEY_SET:
    os.environ["OPENAI_API_KEY"] = _FAKE_OPENAI_KEY
try:
    import rad_chunk as rc  # noqa: E402
finally:
    if _IMPORT_KEY_SET and os.environ.get("OPENAI_API_KEY") == _FAKE_OPENAI_KEY:
        del os.environ["OPENAI_API_KEY"]


# Chaînes SANS préfixe albert/ : elles doivent être routées exactement comme aujourd'hui.
LEGACY_MATRIX = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4o-mini-2024-07-18",
    "o3-mini",
    "text-embedding-3-large",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash:free",
    "openai/gpt-oss-120b",
    "openai/gpt-4o-mini",
    "cohere/command-r-plus",
    "meta-llama/llama-3.1-70b-instruct",
    "mistralai/mistral-small-3.2-24b-instruct",
    "deepseek/deepseek-chat",
    "qwen/qwen-2.5-72b-instruct",
    "x-ai/grok-2",
    "openrouter/auto",
    "a/b/c",
    "/leading-slash",
    "trailing-slash/",
    "albert",
    "albert-gpt",
    "albertine/model",
    "albert-x/model",
    "alber/t",
    "my-albert/model",
    " albert/gpt-oss-120b",
    "albert\\gpt-oss-120b",
    "albert:gpt-oss-120b",
    "openweight-small",
    "ministral-3-8b-instruct-2512",
    "gpt-oss-120b",
    "",
]


class _RecordingClient:
    """Faux client OpenAI : journalise (client, modèle) de chaque ``chat.completions.create``."""

    def __init__(self, name, log):
        """Crée le client ``name`` qui écrit dans ``log``."""
        self.name = name
        self._log = log
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        """Enregistre l'appel et renvoie une complétion ``stop`` non vide."""
        self._log.append((self.name, kwargs.get("model")))
        message = SimpleNamespace(content="texte recodé")
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


@pytest.fixture
def recode_clients(monkeypatch):
    """Clients OpenAI / OpenRouter factices dans ``rad_chunk`` (un seul worker)."""
    log = []
    monkeypatch.setattr(rc, "client", _RecordingClient("openai", log))
    monkeypatch.setattr(rc, "openrouter_client", _RecordingClient("openrouter", log))
    monkeypatch.setattr(rc, "DEFAULT_MAX_WORKERS", 1)
    return log


def test_matrix_is_large_and_has_no_albert_prefix():
    assert len(LEGACY_MATRIX) >= 30
    assert len(set(LEGACY_MATRIX)) == len(LEGACY_MATRIX)
    assert not any(m[:7].lower() == "albert/" for m in LEGACY_MATRIX)


def test_resolver_matches_legacy_matrix(recode_clients):
    for model in LEGACY_MATRIX:
        del recode_clients[:]
        texts, statuses = rc.gpt_recode_batch(["texte brut"], "consigne", model=model)
        assert statuses == ["recoded"], model
        assert len(recode_clients) == 1, model
        routed, sent_model = recode_clients[0]
        assert rp.legacy_provider(model) == routed, model
        for enabled in (False, True):
            res = rp.resolve_llm_provider(model, albert_enabled=enabled)
            assert res.provider == routed, model
            assert res.wire_model == model == sent_model
            assert res.credential_key == rp.CREDENTIAL_KEYS[routed]


def test_harden_routing_with_pinned_openrouter_matches_legacy(recode_clients):
    """Branche durcie (``rad_chunk`` : ``("/" in model) and openrouter_client``) avec client OpenRouter."""
    for model in LEGACY_MATRIX:
        del recode_clients[:]
        cfg = rc.rad_recode_cache.RecodeConfig(
            harden_enabled=True, model=model, prefer_openai=False, openrouter_provider="fake-provider"
        )
        rc.gpt_recode_batch(["texte brut"], "consigne", model="ignored-by-harden", recode_cfg=cfg)
        assert [c[0] for c in recode_clients] == [rp.legacy_provider(model)], model


def test_legacy_provider_none_and_empty():
    assert rp.legacy_provider(None) == "openai"
    assert rp.legacy_provider("") == "openai"
    res = rp.resolve_llm_provider(None, albert_enabled=True)
    assert res == rp.ProviderResolution("openai", "", "openai_api_key")


def test_albert_prefix_split_first_slash():
    res = rp.resolve_llm_provider("albert/openai/gpt-oss-120b", albert_enabled=True)
    assert res == rp.ProviderResolution("albert", "openai/gpt-oss-120b", "albert_api_key")
    res = rp.resolve_llm_provider("albert/gpt-oss-120b", albert_enabled=True)
    assert res.wire_model == "gpt-oss-120b"
    res = rp.resolve_llm_provider("albert/mistralai/Ministral-3-8B-Instruct-2512", albert_enabled=True)
    assert res.wire_model == "mistralai/Ministral-3-8B-Instruct-2512"


def test_albert_prefix_case_insensitive():
    for model in ("Albert/openweight-large", "ALBERT/openweight-large", "aLbErT/openweight-large"):
        res = rp.resolve_llm_provider(model, albert_enabled=True)
        assert res.provider == "albert"
        assert res.wire_model == "openweight-large"
        assert res.credential_key == "albert_api_key"


def test_albert_prefix_disabled_raises():
    for model in ("albert/gpt-oss-120b", "ALBERT/openai/gpt-oss-120b", "albert/"):
        with pytest.raises(AlbertDisabledError) as exc:
            rp.resolve_llm_provider(model, albert_enabled=False)
        assert isinstance(exc.value, ValueError)
    # Sans préfixe, Albert OFF ne change rien.
    assert rp.resolve_llm_provider("google/gemini-2.5-flash", albert_enabled=False).provider == "openrouter"


def test_empty_wire_model_raises():
    for model in ("albert/", "ALBERT/", "albert/   "):
        with pytest.raises(ValueError) as exc:
            rp.resolve_llm_provider(model, albert_enabled=True)
        assert not isinstance(exc.value, AlbertDisabledError)


def test_legacy_cache_label_matches_rad_chunk_312(recode_clients, monkeypatch, tmp_path):
    cache = rc.rad_recode_cache
    seen = []
    real_recode_key = cache.recode_key

    def recode_key_spy(content_hash, model, provider, prompt_version, decode_params_json):
        """Capture le libellé de fournisseur passé par ``recode_batch_cached``."""
        seen.append((model, provider))
        return real_recode_key(content_hash, model, provider, prompt_version, decode_params_json)

    monkeypatch.setattr(cache, "recode_key", recode_key_spy)
    monkeypatch.setattr(cache, "get_recode", lambda cfg, key: None)
    monkeypatch.setattr(cache, "put_recode", lambda cfg, key, text: text)
    monkeypatch.setenv("RECODE_CACHE_ENABLED", "1")
    monkeypatch.setenv("RECODE_CACHE_PATH", str(tmp_path / "cache.sqlite"))
    monkeypatch.delenv("RECODE_HARDEN_ENABLED", raising=False)
    checked = 0
    for prefer in (True, False):
        monkeypatch.setenv("RECODE_PREFER_OPENAI", "1" if prefer else "0")
        for model in LEGACY_MATRIX:
            del seen[:]
            rc.recode_batch_cached(["Texte brut du chunk pour la clé de cache."], "consigne", model)
            assert seen == [(model, rp.legacy_cache_provider_label(model, prefer_openai=prefer))], (model, prefer)
            checked += 1
    assert checked == 2 * len(LEGACY_MATRIX)
    assert rp.legacy_cache_provider_label("google/gemini-2.5-flash", prefer_openai=False) == "openrouter"
    assert rp.legacy_cache_provider_label("google/gemini-2.5-flash", prefer_openai=True) == "openai"
    assert rp.legacy_cache_provider_label(None, prefer_openai=False) == "openai"


def test_embed_params_json_openai_exact():
    expected = json.dumps({"model": "text-embedding-3-large"}, sort_keys=True)
    for env in ({}, {"EMBEDDING_PROVIDER": ""}, {"EMBEDDING_PROVIDER": "openai"}, {"EMBEDDING_PROVIDER": " OpenAI "}):
        cfg = rp.EmbeddingConfig.from_env(env)
        assert cfg.provider == "openai"
        assert cfg.space is rp.OPENAI_DEFAULT
        assert cfg.embed_params_json() == expected
    assert rp.OPENAI_DEFAULT.cache_model == "text-embedding-3-large"


def test_embed_params_json_parity_with_rad_chunk_embed_cache(monkeypatch):
    cache = rc.rad_recode_cache
    seen = []

    def embed_key_spy(text, embed_model, embed_params_json):
        """Capture le modèle et les paramètres de la clé d'embedding historique."""
        seen.append((embed_model, embed_params_json))
        return "fake-key"

    monkeypatch.setattr(cache, "embed_key", embed_key_spy)
    monkeypatch.setattr(cache, "get_embed", lambda cfg, key: [0.5])
    rc._embed_with_cache(["texte"], cache.RecodeConfig())
    cfg = rp.EmbeddingConfig.from_env({})
    assert seen == [(cfg.space.cache_model, cfg.embed_params_json())]


def test_embedding_config_reads_os_environ_by_default(monkeypatch):
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    assert rp.EmbeddingConfig.from_env().space is rp.OPENAI_DEFAULT
    monkeypatch.setenv("EMBEDDING_PROVIDER", "albert")
    with pytest.raises(AlbertDisabledError):
        rp.EmbeddingConfig.from_env()


def test_embedding_config_albert_when_disabled_raises():
    for env in ({"EMBEDDING_PROVIDER": "albert"}, {"EMBEDDING_PROVIDER": "Albert", "ALBERT_ENABLED": "0"},
                {"EMBEDDING_PROVIDER": "albert", "ALBERT_ENABLED": ""}):
        with pytest.raises(AlbertDisabledError):
            rp.EmbeddingConfig.from_env(env)
    with pytest.raises(ValueError) as exc:
        rp.EmbeddingConfig.from_env({"EMBEDDING_PROVIDER": "cohere"})
    assert not isinstance(exc.value, AlbertDisabledError)


def test_embedding_config_albert_enabled_space():
    cfg = rp.EmbeddingConfig.from_env({"EMBEDDING_PROVIDER": "albert", "ALBERT_ENABLED": "1"})
    assert cfg.provider == "albert"
    assert cfg.space is rp.ALBERT_BGE_M3
    assert (cfg.space.model, cfg.space.dim, cfg.space.batch_max, cfg.space.norm) == ("bge-m3", 1024, 64, "l2")
    assert cfg.space.is_default is False
    assert cfg.space.cache_model == "albert:bge-m3"
    assert json.loads(cfg.embed_params_json()) == {"dim": 1024, "model": "bge-m3", "norm": "l2", "provider": "albert"}
    assert cfg.embed_params_json() == json.dumps(json.loads(cfg.embed_params_json()), sort_keys=True)
    alias = rp.EmbeddingConfig.from_env(
        {"EMBEDDING_PROVIDER": "albert", "ALBERT_ENABLED": "1", "ALBERT_EMBED_MODEL": "openweight-embeddings"}
    )
    assert alias.space is rp.ALBERT_BGE_M3
    no_norm = rp.EmbeddingConfig.from_env(
        {"EMBEDDING_PROVIDER": "albert", "ALBERT_ENABLED": "1", "ALBERT_EMBED_L2_NORMALIZE": "0"}
    )
    assert no_norm.space.norm == "none"
    assert no_norm.embed_params_json() != cfg.embed_params_json()
    small_batch = rp.EmbeddingConfig.from_env(
        {"EMBEDDING_PROVIDER": "albert", "ALBERT_ENABLED": "1", "ALBERT_EMBED_BATCH": "16"}
    )
    assert small_batch.space is rp.ALBERT_BGE_M3
    with pytest.raises(ValueError):
        rp.EmbeddingConfig.from_env(
            {"EMBEDDING_PROVIDER": "albert", "ALBERT_ENABLED": "1", "ALBERT_EMBED_MODEL": "qwen3-vl-embedding-8b"}
        )


def test_space_constants():
    assert rp.OPENAI_DEFAULT == rp.EmbeddingSpace("openai", "text-embedding-3-large", 3072, 2048, True, "none")
    assert rp.ALBERT_BGE_M3 == rp.EmbeddingSpace("albert", "bge-m3", 1024, 64, False, "l2")


def _vec(dim, value=0.1):
    """Vecteur constant non nul de dimension ``dim``."""
    return [value] * dim


def test_check_uniform_space_mixed_zero_none():
    assert rp.check_uniform_space([]) is None
    assert rp.check_uniform_space(None) is None
    assert rp.check_uniform_space([{"embedding": None}, {"embedding": [0.0] * 3072}, {"embedding": []}, {}]) is None

    legacy = [{"embedding": _vec(3072)}, {"embedding": None}, {"embedding": [0.0] * 3072}, "pas un chunk"]
    assert rp.check_uniform_space(legacy) is rp.OPENAI_DEFAULT

    albert = [
        {"embedding": _vec(1024), "embedding_provider": "albert", "embedding_model": "bge-m3", "embedding_dim": 1024},
        {"embedding": None, "embedding_provider": "albert", "embedding_model": "bge-m3", "embedding_dim": 1024},
    ]
    assert rp.check_uniform_space(albert) is rp.ALBERT_BGE_M3

    with pytest.raises(ValueError, match="mélangés"):
        rp.check_uniform_space(legacy + albert)
    with pytest.raises(ValueError, match="mélangés"):
        rp.check_uniform_space([{"embedding": _vec(3072)}, {"embedding": _vec(1024)}])
    same_dim_other_model = {"embedding": _vec(1024), "embedding_provider": "albert", "embedding_model": "autre-modele"}
    with pytest.raises(ValueError, match="mélangés"):
        rp.check_uniform_space(albert + [same_dim_other_model])
    with pytest.raises(ValueError, match="embedding_dim"):
        rp.check_uniform_space([{"id": "c1", "embedding": _vec(1024), "embedding_provider": "albert",
                                 "embedding_dim": 3072}])

    odd = rp.check_uniform_space([{"embedding": _vec(768)}])
    assert (odd.provider, odd.model, odd.dim, odd.is_default) == ("openai", "text-embedding-3-large", 768, False)


def test_target_mismatch_ignores_non_int():
    for target in (None, "3072", 3072.0, 1024.0, True, False, mock.MagicMock(), object()):
        assert rp.target_mismatch_message(target, rp.OPENAI_DEFAULT) is None
    assert rp.target_mismatch_message(3072, rp.OPENAI_DEFAULT) is None
    assert rp.target_mismatch_message(1024, rp.ALBERT_BGE_M3) is None
    assert rp.target_mismatch_message(1024, None) is None
    message = rp.target_mismatch_message(1024, rp.OPENAI_DEFAULT)
    assert isinstance(message, str) and "1024" in message and "3072" in message
    message = rp.target_mismatch_message(3072, rp.ALBERT_BGE_M3)
    assert "albert/bge-m3" in message


def test_import_is_stdlib_light():
    code = (
        "import sys; import scripts.rad_providers; "
        "print('httpx' in sys.modules, 'openai' in sys.modules, 'requests' in sys.modules)"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=RAGPY_ROOT, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["False", "False", "False"]


def test_cli_import_pattern():
    code = "import sys; sys.path.insert(0, 'scripts'); import rad_albert, rad_providers; print('cli-import-ok')"
    proc = subprocess.run([sys.executable, "-c", code], cwd=RAGPY_ROOT, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "cli-import-ok"
