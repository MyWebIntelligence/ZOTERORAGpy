"""Lecture en flux des fichiers de chunks (``app/utils/json_stream.py``).

Un fichier d'embeddings de plusieurs Go ne doit jamais être chargé en entier
par le processus web (le conteneur, limité à 8 Go, était tué à l'ouverture
de la session 2a335a68_LaBible : 1,8 Go, 21 147 chunks).
"""

from __future__ import annotations

import json
import time

import pytest

from app.utils import json_stream


@pytest.mark.parametrize("data", [
    [],
    [1, 2, 3],
    [{"a": "x]y,\"z{", "embedding": [0.1] * 40}] * 6,
    [12345678901234567890, -1.5e10, True, None, "s", 0.000123e-7] * 200,
    [{"t": "é" * 3000, "n": {"k": [1, {"z": None}]}}] * 4,
])
@pytest.mark.parametrize("indent", [None, 2])
@pytest.mark.parametrize("block", [1, 7, 4096, 1 << 20])
def test_iter_matches_json_load(tmp_path, data, indent, block):
    """Mêmes éléments que ``json.load``, quelle que soit la taille des lectures (nombres jamais coupés)."""
    path = tmp_path / "c.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=indent), encoding="utf-8")
    assert list(json_stream.iter_json_array(str(path), block_size=block)) == data


@pytest.mark.parametrize("text", ["{}", "[1,2", "[1 2]", "", "[1,]", "[{\"a\": 1}"])
def test_invalid_or_non_list_is_refused(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        json_stream.count_json_array(str(path))


def test_summary_finds_the_first_space_fields(tmp_path):
    path = tmp_path / "dense.json"
    chunks = [{"text": "a"}, {"text": "b", "embedding_provider": "albert", "embedding_dim": 1024},
              {"text": "c", "embedding_provider": "albert", "embedding_dim": 1024, "embedding_model": "bge-m3"}]
    path.write_text(json.dumps(chunks), encoding="utf-8")
    summary = json_stream.summarize_json_array(str(path), ("embedding_provider", "embedding_model", "embedding_dim"))
    assert summary == {"count": 3, "fields": {"embedding_provider": "albert", "embedding_dim": 1024}}


def test_large_file_is_summarised_in_the_background_then_cached(tmp_path):
    """Au-delà du seuil : ``None`` tout de suite, puis le résultat en cache ; nouvelle version = nouveau calcul."""
    path = tmp_path / "big.json"
    path.write_text(json.dumps([{"embedding": [0.5] * 100}] * 50), encoding="utf-8")
    first = json_stream.cached_summary(str(path), sync_max_bytes=10)
    assert first is None
    for _ in range(100):
        summary = json_stream.cached_summary(str(path), sync_max_bytes=10)
        if summary is not None:
            break
        time.sleep(0.02)
    assert summary == {"count": 50, "fields": {}}
    path.write_text(json.dumps([1, 2]), encoding="utf-8")
    assert json_stream.cached_summary(str(path)) == {"count": 2, "fields": {}}


@pytest.mark.parametrize("data", [
    [],
    [{"text": "é « guillemets » \n ligne", "embedding": [0.1, -2.5e-07, 3], "sparse_embedding": {"indices": [], "values": []}}],
    [{"a": 1, "b": [1, [2, {"c": None}]], "d": "x"}] * 3,
    [1, "deux", None, True, 4.5],
])
def test_streamed_writer_matches_json_dump_bytes(tmp_path, data):
    """``write_json_array_atomic`` : octets identiques à ``json.dump(indent=2, ensure_ascii=False)``."""
    from scripts.rad_json_stream import write_json_array_atomic

    reference = tmp_path / "ref.json"
    with open(reference, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    target = tmp_path / "out.json"
    assert write_json_array_atomic(iter(data), str(target)) == len(data)
    assert target.read_bytes() == reference.read_bytes()
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_streamed_writer_keeps_the_old_file_on_failure(tmp_path):
    """Interruption pendant l'écriture : l'ancien fichier reste intact, aucun temporaire."""
    from scripts.rad_json_stream import write_json_array_atomic

    target = tmp_path / "out.json"
    target.write_text("[1]", encoding="utf-8")

    def broken():
        yield 1
        raise RuntimeError("interrompu")

    with pytest.raises(RuntimeError):
        write_json_array_atomic(broken(), str(target))
    assert target.read_text(encoding="utf-8") == "[1]"
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]


@pytest.mark.parametrize("text", [
    '[{"id": "a", "embedding": [0.5, 1e-07]}, {"id": "b"}]',
    '  \n [1, 2]',
    '[]',
    '{"not": "a list"}',
    '"chaîne"',
    '42',
])
def test_load_json_equals_json_load(tmp_path, text):
    """``load_json`` : même objet que ``json.load`` (liste lue en flux, le reste par ``json.load``)."""
    from scripts.rad_json_stream import load_json

    path = tmp_path / "f.json"
    path.write_text(text, encoding="utf-8")
    assert load_json(str(path)) == json.loads(text)


@pytest.mark.parametrize("text", ["[1, 2", "[1 2]", "{bad", ""])
def test_load_json_invalid_raises_json_decode_error(tmp_path, text):
    """JSON invalide : ``json.JSONDecodeError`` (messages des connecteurs inchangés)."""
    from scripts.rad_json_stream import load_json

    path = tmp_path / "f.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_json(str(path))
