"""Tests hors ligne de ``scripts/rad_audio.py`` : enregistrements audio → ``output.csv`` du pipeline.

Le client Albert est branché sur ``tests/albert_fakes.FakeAlbert`` (``httpx.MockTransport``,
clé factice), le ``.env`` n'est jamais chargé (``_load_env`` neutralisé) et ffmpeg/ffprobe sont
simulés (``ffmpeg_available``, ``_run_ffmpeg``, ``_probe_duration``) : aucun réseau, aucun
binaire externe. Les WAV sont produits par le module ``wave``.
"""

from __future__ import annotations

import io
import json
import os
import re
import wave

import pandas as pd
import pytest

from scripts import rad_audio
from scripts.rad_albert.client import AlbertClient
from tests.albert_fakes import FAKE_ALBERT_KEY, FakeAlbert

TRANSCRIPTIONS = "/v1/audio/transcriptions"
ABORT_RE = re.compile(r"^Albert abort: kind=(\w+) reason=(\w+)(?: credential_required=(\w+))?\s*$", re.MULTILINE)
REQUIRED_COLUMNS = {
    "itemKey", "title", "authors", "date", "filename", "path", "texteocr", "texteocr_provider",
    "texteocr_partial", "texteocr_pages_done", "texteocr_pages_total", "language", "duration_s", "segments",
}


# ---------------------------------------------------------------------------
# Outils
# ---------------------------------------------------------------------------
def _wav_bytes(seconds=1.0, rate=8000):
    """WAV mono 16 bits de silence."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


def _write_wav(path, seconds=1.0, salt=b""):
    """Écrit un WAV (``salt`` en fin de fichier pour varier l'empreinte sans changer la durée)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_wav_bytes(seconds) + salt)
    return path


def _read_csv(path):
    """Relit le CSV comme ``rad_chunk.py`` (``pd.read_csv`` sans option)."""
    return pd.read_csv(path)


def _read_json(path):
    """Relit un JSON écrit par le script."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _calls(fake):
    """Appels de transcription reçus par le faux serveur."""
    return fake.calls_to("POST", TRANSCRIPTIONS)


@pytest.fixture
def fake(monkeypatch):
    """Albert simulé, interrupteurs ON, clé factice, ``.env`` jamais lu, pas de ffmpeg."""
    monkeypatch.setattr(rad_audio, "_load_env", lambda: None)
    monkeypatch.setenv("ALBERT_ENABLED", "1")
    monkeypatch.setenv("ALBERT_AUDIO_ENABLED", "1")
    monkeypatch.setenv("ALBERT_API_KEY", FAKE_ALBERT_KEY)
    albert = FakeAlbert()

    def build(cfg, api_key, *, ledger):
        """Client branché sur le faux, sans limiteur ni attente réelle."""
        return AlbertClient(cfg, api_key, transport=albert.transport, ledger=ledger,
                            sleep=lambda _s: None, use_limiter=False)

    def no_ffmpeg(args, *, timeout):
        """ffmpeg ne doit pas être lancé dans ce scénario."""
        raise AssertionError("ffmpeg ne doit pas être appelé")

    monkeypatch.setattr(rad_audio, "_build_client", build)
    monkeypatch.setattr(rad_audio, "ffmpeg_available", lambda: False)
    monkeypatch.setattr(rad_audio, "_run_ffmpeg", no_ffmpeg)
    monkeypatch.setattr(rad_audio, "_probe_duration", lambda path: None)
    return albert


@pytest.fixture
def splitter(monkeypatch, fake):
    """ffmpeg simulé : découpe un enregistrement de 72 s en segments de 30, 30 et 12 s."""
    monkeypatch.setenv("ALBERT_AUDIO_SEGMENT_SECONDS", "30")
    state = {"runs": [], "tmpdirs": []}
    durations = {"seg_0000.mp3": 30.0, "seg_0001.mp3": 30.0, "seg_0002.mp3": 12.0}

    def run_ffmpeg(args, *, timeout):
        """Écrit trois faux segments mp3 selon le motif de sortie (dernier argument)."""
        state["runs"].append(list(args))
        pattern = args[-1]
        state["tmpdirs"].append(os.path.dirname(pattern))
        for index in range(3):
            with open(pattern % index, "wb") as handle:
                handle.write(b"ID3 faux segment-%d" % index)

    def probe(path):
        """Durées simulées : 72 s pour la source, puis celles des segments."""
        name = os.path.basename(path)
        if name in durations:
            return durations[name]
        return 72.0 if name.endswith(".wav") else None

    monkeypatch.setattr(rad_audio, "ffmpeg_available", lambda: True)
    monkeypatch.setattr(rad_audio, "_run_ffmpeg", run_ffmpeg)
    monkeypatch.setattr(rad_audio, "_probe_duration", probe)
    fake.audio_reply = lambda filename, data: "Contenu du segment " + data.decode().rsplit("-", 1)[-1] + "."
    return state


def _run(tmp_path, *extra):
    """Lance la CLI sur ``tmp_path/in`` vers ``tmp_path/out/output.csv``."""
    out = tmp_path / "out" / "output.csv"
    code = rad_audio.main(["--input", str(tmp_path / "in"), "--output", str(out), *extra])
    return code, out


# ---------------------------------------------------------------------------
# Transcription d'un WAV
# ---------------------------------------------------------------------------
def test_wav_transcription_produces_pipeline_csv(fake, tmp_path, capsys):
    source = _write_wav(tmp_path / "in" / "entretien.wav", seconds=2.0)
    (tmp_path / "in" / "notes.txt").write_text("pas un enregistrement")
    fake.audio_reply = "Bonjour et bienvenue au séminaire de recherche."

    code, out = _run(tmp_path)

    assert code == 0
    df = _read_csv(out)
    assert REQUIRED_COLUMNS <= set(df.columns)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["texteocr_provider"] == "albert_whisper"
    assert bool(row["texteocr_partial"]) is False
    assert (row["texteocr_pages_done"], row["texteocr_pages_total"], row["segments"]) == (1, 1, 1)
    assert row["title"] == "entretien"
    assert row["filename"] == "entretien.wav" and row["path"] == "entretien.wav"
    assert re.fullmatch(r"audio-[0-9a-f]{16}", row["itemKey"])
    assert row["language"] == "fr"
    assert row["duration_s"] == pytest.approx(2.0)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["date"])
    assert row["texteocr"] == (
        "<!-- Segment 1 (00:00:00–00:00:02) -->\nBonjour et bienvenue au séminaire de recherche."
    )
    # Ce que rad_chunk lit : un texte non vide et un fournisseur connu.
    assert row["texteocr"].strip()

    call = _calls(fake)[0]
    assert call.form["model"] == "whisper-large-v3"
    assert call.form["language"] == "fr" and call.form["response_format"] == "verbose_json"
    # Nom neutre : le nom local du fichier ne part jamais vers Albert.
    assert call.files["file"][0] == row["itemKey"] + ".wav"
    assert call.files["file"][1] == source.read_bytes()

    stdout = capsys.readouterr().out
    assert "Fichier 1/1 : entretien.wav" in stdout
    assert "Segment 1/1 : transcrit" in stdout
    assert FAKE_ALBERT_KEY not in stdout
    assert _read_json(rad_audio.errors_file_path(str(out)))["total_errors"] == 0
    usage = (tmp_path / "out" / "albert_usage.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(usage) == 1 and json.loads(usage[0])["role"] == "audio"


def test_language_and_prompt_options(fake, tmp_path):
    _write_wav(tmp_path / "in" / "a.wav")
    code, _out = _run(tmp_path, "--language", "auto", "--prompt", "Latour, ANT, STS")
    assert code == 0
    form = _calls(fake)[0].form
    assert "language" not in form
    assert form["prompt"] == "Latour, ANT, STS"


# ---------------------------------------------------------------------------
# Découpage, marqueurs et horodatages
# ---------------------------------------------------------------------------
def test_split_markers_and_offset_timestamps(fake, splitter, tmp_path):
    _write_wav(tmp_path / "in" / "seminaire.wav")

    code, out = _run(tmp_path)

    assert code == 0
    row = _read_csv(out).iloc[0]
    text = row["texteocr"]
    assert "<!-- Segment 1 (00:00:00–00:00:30) -->\nContenu du segment 0." in text
    assert "<!-- Segment 2 (00:00:30–00:01:00) -->\nContenu du segment 1." in text
    assert "<!-- Segment 3 (00:01:00–00:01:12) -->\nContenu du segment 2." in text
    assert text.index("Segment 1 (") < text.index("Segment 2 (") < text.index("Segment 3 (")
    assert (row["segments"], row["texteocr_pages_done"], row["texteocr_pages_total"]) == (3, 3, 3)
    assert row["duration_s"] == pytest.approx(72.0)

    # Commande ffmpeg : sans shell, mono 16 kHz, mp3 48 kb/s, segments de 30 s remis à zéro.
    args = splitter["runs"][0]
    for option, value in (("-ac", "1"), ("-ar", "16000"), ("-b:a", "48k"), ("-f", "segment"),
                          ("-segment_time", "30"), ("-reset_timestamps", "1")):
        assert args[args.index(option) + 1] == value
    assert os.path.isabs(args[args.index("-i") + 1])
    assert not os.path.exists(splitter["tmpdirs"][0])  # dossier temporaire supprimé

    # Horodatages absolus : décalés de l'offset du segment, croissants, dans la fenêtre.
    report = _read_json(rad_audio.segments_file_path(str(out)))
    segments = report["files"][0]["segments"]
    assert [(s["start"], s["end"]) for s in segments] == [(0.0, 30.0), (30.0, 60.0), (60.0, 72.0)]
    assert [u["start"] for u in segments[1]["utterances"]] == [30.0, 35.0]
    starts = [u["start"] for s in segments for u in s["utterances"]]
    assert starts == sorted(starts)
    for s in segments:
        assert all(s["start"] <= u["start"] <= u["end"] <= s["end"] for u in s["utterances"])

    # Noms neutres des segments envoyés.
    assert [c.files["file"][0] for c in _calls(fake)] == [
        f"{row['itemKey']}-seg{i:04d}.mp3" for i in (1, 2, 3)
    ]


def test_concurrent_segments_keep_their_order(fake, splitter, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ALBERT_AUDIO_CONCURRENCY", "3")
    _write_wav(tmp_path / "in" / "seminaire.wav")
    code, out = _run(tmp_path)
    assert code == 0 and len(_calls(fake)) == 3
    text = _read_csv(out).iloc[0]["texteocr"]
    assert text.index("segment 0.") < text.index("segment 1.") < text.index("segment 2.")
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("  Segment ")]
    assert [line.split(" : ")[0].strip() for line in lines] == ["Segment 1/3", "Segment 2/3", "Segment 3/3"]


def test_abort_marker_kinds():
    errors = rad_audio._rad_albert
    assert rad_audio.albert_abort_marker(errors.AlbertPermanentError(reason="not_found")) == (
        "Albert abort: kind=model reason=not_found"
    )
    assert rad_audio.albert_abort_marker(errors.AlbertTransientError()) == "Albert abort: kind=service reason=transient"
    assert rad_audio.albert_abort_marker(errors.AlbertQuotaExhausted()) == (
        "Albert abort: kind=account reason=quota_exhausted credential_required=albert_api_key"
    )


def test_checkpoints_are_confined_and_reused_without_resending(fake, splitter, tmp_path, capsys):
    _write_wav(tmp_path / "in" / "seminaire.wav")
    code, out = _run(tmp_path)
    assert code == 0 and len(_calls(fake)) == 3 and len(splitter["runs"]) == 1
    first = _read_csv(out).iloc[0]

    root = tmp_path / "out" / rad_audio.CHECKPOINT_DIRNAME
    checkpoints = sorted(root.rglob("seg_*.json"))
    assert [p.name for p in checkpoints] == ["seg_0001.json", "seg_0002.json", "seg_0003.json"]
    out_dir = os.path.realpath(tmp_path / "out")
    for path in checkpoints:
        assert os.path.realpath(path).startswith(out_dir + os.sep)
        assert path.parent.parent.name == first["itemKey"].split("-", 1)[1]
    assert (checkpoints[0].parent / "plan.json").is_file()
    assert _read_json(checkpoints[1])["text"] == "Contenu du segment 1."
    capsys.readouterr()

    # Relance : tout est repris, ni ffmpeg ni transcription.
    code, out = _run(tmp_path)
    assert code == 0
    assert len(_calls(fake)) == 3 and len(splitter["runs"]) == 1
    second = _read_csv(out).iloc[0]
    assert second["texteocr"] == first["texteocr"] and second["itemKey"] == first["itemKey"]
    assert capsys.readouterr().out.count("repris (point de reprise)") == 3

    # Un point de reprise retiré : seul ce segment repart.
    checkpoints[1].unlink()
    code, out = _run(tmp_path)
    assert code == 0
    assert len(_calls(fake)) == 4 and len(splitter["runs"]) == 2
    assert _read_csv(out).iloc[0]["texteocr"] == first["texteocr"]


def test_corrupt_checkpoint_is_set_aside_and_redone(fake, tmp_path):
    _write_wav(tmp_path / "in" / "a.wav")
    assert _run(tmp_path)[0] == 0
    checkpoint = next((tmp_path / "out" / rad_audio.CHECKPOINT_DIRNAME).rglob("seg_0001.json"))
    checkpoint.write_text("{ illisible", encoding="utf-8")
    assert _run(tmp_path)[0] == 0
    assert len(_calls(fake)) == 2
    assert list(checkpoint.parent.glob("seg_0001.json.corrupt-*"))
    assert _read_json(checkpoint)["status"] == "ok"


def test_checkpoint_dir_outside_output_is_refused(fake, tmp_path):
    _write_wav(tmp_path / "in" / "a.wav")
    outside = tmp_path / "ailleurs"
    outside.mkdir()
    (tmp_path / "out").mkdir()
    os.symlink(outside, tmp_path / "out" / rad_audio.CHECKPOINT_DIRNAME)

    code, out = _run(tmp_path)

    assert code == 1
    assert list(outside.iterdir()) == []
    assert _calls(fake) == []
    errors = _read_json(rad_audio.errors_file_path(str(out)))["errors"]
    assert errors[0]["error_type"] == "CHECKPOINT_REFUSED"


# ---------------------------------------------------------------------------
# Échecs de segments
# ---------------------------------------------------------------------------
def test_failed_segment_marks_row_partial_and_is_retried_later(fake, splitter, tmp_path):
    _write_wav(tmp_path / "in" / "seminaire.wav")
    ok_payload = {"id": "t1", "text": "Premier segment.", "model": "whisper-large-v3", "language": "fr",
                  "duration": 30.0, "segments": [{"start": 0.0, "end": 4.0, "text": "Premier segment."}]}
    fake.inject("POST", TRANSCRIPTIONS, (200, ok_payload), "bad_request")

    code, out = _run(tmp_path)

    assert code == 0
    row = _read_csv(out).iloc[0]
    assert bool(row["texteocr_partial"]) is True
    assert (row["texteocr_pages_done"], row["texteocr_pages_total"]) == (2, 3)
    marker = "<!-- Segment 2 (00:00:30–00:01:00) -->\n<!-- TRANSCRIPTION ÉCHOUÉE (albert_whisper) : "
    assert marker in row["texteocr"]
    failure = row["texteocr"].split(marker, 1)[1].split("\n", 1)[0]
    assert failure.endswith(" -->") and "--" not in failure[:-4]
    errors = _read_json(rad_audio.errors_file_path(str(out)))["errors"]
    assert [e["error_type"] for e in errors] == ["TRANSCRIPTION_PARTIAL"]
    assert errors[0]["failed_segments"][0]["segment"] == 2
    assert (errors[0]["pages_done"], errors[0]["pages_total"]) == (2, 3)
    ckpt = next((tmp_path / "out" / rad_audio.CHECKPOINT_DIRNAME).rglob("plan.json")).parent
    assert not (ckpt / "seg_0002.json").exists()

    # Relance : seul le segment en échec repart ; la ligne n'est plus partielle.
    code, out = _run(tmp_path)
    assert code == 0 and len(_calls(fake)) == 4
    row = _read_csv(out).iloc[0]
    assert bool(row["texteocr_partial"]) is False and "TRANSCRIPTION ÉCHOUÉE" not in row["texteocr"]


def test_no_text_at_all_exits_1_without_csv(fake, tmp_path, capsys):
    _write_wav(tmp_path / "in" / "a.wav")
    fake.inject("POST", TRANSCRIPTIONS, "bad_request")

    code, out = _run(tmp_path)

    assert code == 1
    assert not out.exists()
    errors = _read_json(rad_audio.errors_file_path(str(out)))["errors"]
    assert errors[0]["error_type"] == "TRANSCRIPTION_FAILED"
    assert "Aucun texte transcrit" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Erreurs de compte : arrêt de tout le traitement
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("entry,retry_after,reason", [
    ("invalid_key", None, "invalid_key"),
    ("account_expired", None, "account_expired"),
    (429, 3600, "quota_exhausted"),
])
def test_account_error_stops_run_with_abort_line(fake, tmp_path, capsys, entry, retry_after, reason):
    _write_wav(tmp_path / "in" / "a.wav", salt=b"a")
    _write_wav(tmp_path / "in" / "b.wav", salt=b"b")
    fake.inject("POST", TRANSCRIPTIONS, entry, retry_after=retry_after)

    code, out = _run(tmp_path)

    assert code == 2
    assert len(_calls(fake)) == 1  # le second fichier n'est jamais envoyé
    stdout = capsys.readouterr().out
    aborts = ABORT_RE.findall(stdout)
    assert aborts == [("account", reason, "albert_api_key")]
    assert stdout.rstrip().splitlines()[-1].startswith("Albert abort:")
    assert FAKE_ALBERT_KEY not in stdout
    errors = _read_json(rad_audio.errors_file_path(str(out)))["errors"]
    assert errors[-1]["error_type"] == "ALBERT_ACCOUNT_ERROR"
    assert errors[-1]["unprocessed_files"] == ["a.wav", "b.wav"]
    assert not out.exists()


# ---------------------------------------------------------------------------
# Refus avant tout traitement
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("env,expected,abort", [
    ({"ALBERT_ENABLED": None}, "ALBERT_ENABLED=1", ("config", "disabled", "")),
    ({"ALBERT_AUDIO_ENABLED": "0"}, "ALBERT_AUDIO_ENABLED=1", ("config", "disabled", "")),
    ({"ALBERT_API_KEY": None}, "ALBERT_API_KEY", ("account", "invalid_key", "albert_api_key")),
])
def test_refused_when_switches_are_off_or_key_missing(fake, tmp_path, capsys, monkeypatch, env, expected, abort):
    _write_wav(tmp_path / "in" / "a.wav")
    for name, value in env.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    code, out = _run(tmp_path)

    assert code == 2
    stdout = capsys.readouterr().out
    assert "Transcription audio refusée" in stdout and expected in stdout
    assert ABORT_RE.findall(stdout) == [abort]
    assert fake.calls == []
    assert not (tmp_path / "out").exists()


def test_output_must_be_a_csv(fake, tmp_path):
    _write_wav(tmp_path / "in" / "a.wav")
    with pytest.raises(SystemExit) as info:
        rad_audio.main(["--input", str(tmp_path / "in"), "--output", str(tmp_path / "out" / "output.json")])
    assert info.value.code == 2


# ---------------------------------------------------------------------------
# Formats à convertir sans ffmpeg
# ---------------------------------------------------------------------------
def test_m4a_without_ffmpeg_fails_explicitly_and_others_continue(fake, tmp_path, capsys):
    _write_wav(tmp_path / "in" / "a.wav")
    (tmp_path / "in" / "b.m4a").write_bytes(b"\x00\x00\x00\x18ftypM4A fausse piste")

    code, out = _run(tmp_path)

    assert code == 0
    df = _read_csv(out)
    assert list(df["filename"]) == ["a.wav"]
    assert len(_calls(fake)) == 1
    errors = _read_json(rad_audio.errors_file_path(str(out)))["errors"]
    assert [(e["filename"], e["error_type"]) for e in errors] == [("b.m4a", "FFMPEG_MISSING")]
    assert errors[0]["error_message"] == "ffmpeg introuvable : convertir en mp3 ou wav de moins de 20 Mo"
    assert "ffmpeg introuvable : convertir en mp3 ou wav de moins de 20 Mo" in capsys.readouterr().out


def test_wav_over_size_limit_without_ffmpeg_fails(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(rad_audio, "MAX_AUDIO_BYTES", 1000)
    _write_wav(tmp_path / "in" / "long.wav")
    code, out = _run(tmp_path)
    assert code == 1 and _calls(fake) == []
    assert _read_json(rad_audio.errors_file_path(str(out)))["errors"][0]["error_type"] == "FFMPEG_MISSING"


def test_long_wav_without_ffmpeg_is_sent_whole(fake, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ALBERT_AUDIO_SEGMENT_SECONDS", "30")
    _write_wav(tmp_path / "in" / "long.wav", seconds=31.0)
    code, out = _run(tmp_path)
    assert code == 0 and len(_calls(fake)) == 1
    assert _read_csv(out).iloc[0]["segments"] == 1
    assert "envoyé en un seul segment" in capsys.readouterr().out


def test_ffmpeg_failure_is_reported_without_local_paths(fake, tmp_path, monkeypatch):
    _write_wav(tmp_path / "in" / "a.wav")
    (tmp_path / "in" / "video.mp4").write_bytes(b"pas une vraie video")

    def broken(args, *, timeout):
        """ffmpeg en échec, avec les chemins locaux dans son message."""
        raise rad_audio.AudioConversionError(f"échec : {args[args.index('-i') + 1]} -> {args[-1]}")

    monkeypatch.setattr(rad_audio, "ffmpeg_available", lambda: True)
    monkeypatch.setattr(rad_audio, "_run_ffmpeg", broken)
    code, out = _run(tmp_path)
    assert code == 0
    errors = _read_json(rad_audio.errors_file_path(str(out)))["errors"]
    assert errors[0]["error_type"] == "AUDIO_CONVERSION_FAILED"
    assert str(tmp_path) not in errors[0]["error_message"] and "video.mp4" in errors[0]["error_message"]


# ---------------------------------------------------------------------------
# Aucun chemin local dans les sorties
# ---------------------------------------------------------------------------
def test_no_absolute_path_in_outputs(fake, splitter, tmp_path, monkeypatch):
    _write_wav(tmp_path / "in" / "seminaire.wav")
    (tmp_path / "in" / "c.ogg").write_bytes(b"OggS fausse piste")
    split_wav = rad_audio._run_ffmpeg

    def only_wav(args, *, timeout):
        """Découpe le WAV, refuse l'ogg (message d'erreur contenant des chemins locaux)."""
        if args[args.index("-i") + 1].endswith(".ogg"):
            raise rad_audio.AudioConversionError(
                "Invalid data found when processing input " + args[args.index("-i") + 1] + " " + args[-1]
            )
        split_wav(args, timeout=timeout)

    monkeypatch.setattr(rad_audio, "_run_ffmpeg", only_wav)
    code, out = _run(tmp_path)

    assert code == 0
    errors = _read_json(rad_audio.errors_file_path(str(out)))["errors"]
    assert [(e["filename"], e["error_type"]) for e in errors] == [("c.ogg", "AUDIO_CONVERSION_FAILED")]
    roots = {str(tmp_path), os.path.realpath(tmp_path)}
    for path in (out, rad_audio.errors_file_path(str(out)), rad_audio.segments_file_path(str(out))):
        content = open(path, encoding="utf-8-sig").read()
        assert not any(root in content for root in roots), path
    df = _read_csv(out)
    assert all(os.path.basename(v) == v for v in df["path"]) and all(os.path.basename(v) == v for v in df["filename"])


# ---------------------------------------------------------------------------
# Entrées et utilitaires
# ---------------------------------------------------------------------------
def test_list_audio_files_is_sorted_non_recursive_and_bounded(tmp_path):
    folder = tmp_path / "in"
    for name in ("b.MP3", "a.wav", ".cache.wav", "._a.wav", "notes.txt", "c.webm"):
        _write_wav(folder / name)
    (folder / "sous-dossier").mkdir()
    _write_wav(folder / "sous-dossier" / "d.wav")

    names = [os.path.basename(p) for p in rad_audio.list_audio_files(str(folder))]
    assert names == ["a.wav", "b.MP3", "c.webm"]
    assert [os.path.basename(p) for p in rad_audio.list_audio_files(str(folder), 2)] == ["a.wav", "b.MP3"]
    with pytest.raises(ValueError):
        rad_audio.list_audio_files(str(folder / "notes.txt"))
    with pytest.raises(FileNotFoundError):
        rad_audio.list_audio_files(str(folder / "absent.wav"))


def test_max_files_option(fake, tmp_path):
    for name in ("a.wav", "b.wav", "c.wav"):
        _write_wav(tmp_path / "in" / name, salt=name.encode())
    code, out = _run(tmp_path, "--max-files", "2")
    assert code == 0 and list(_read_csv(out)["filename"]) == ["a.wav", "b.wav"]


def test_duplicate_content_is_transcribed_once(fake, tmp_path):
    _write_wav(tmp_path / "in" / "a.wav")
    _write_wav(tmp_path / "in" / "copie.wav")
    code, out = _run(tmp_path)
    assert code == 0 and len(_calls(fake)) == 1
    errors = _read_json(rad_audio.errors_file_path(str(out)))["errors"]
    assert [(e["filename"], e["error_type"]) for e in errors] == [("copie.wav", "DUPLICATE_AUDIO")]


def test_helpers_format_and_sanitise():
    assert rad_audio._hms(0) == "00:00:00" and rad_audio._hms(3725.6) == "01:02:06"
    assert rad_audio._hms(None) == "00:00:00"
    safe = rad_audio._marker_safe("erreur --> <!-- injection\n sur deux lignes")
    assert "--" not in safe and "<" not in safe and ">" not in safe and "\n" not in safe
    assert rad_audio.normalise_language(None, "fr") == "fr"
    assert rad_audio.normalise_language("AUTO", "fr") == ""
    with pytest.raises(ValueError):
        rad_audio.normalise_language("fr_FR!", "fr")
    utterances = rad_audio.normalise_utterances(
        [{"start": 5.0, "end": 9.0, "text": " b "}, {"start": 3.0, "end": 2.0, "text": "c"},
         {"start": 28.0, "end": 45.0, "text": "d"}, {"start": "x", "end": 1, "text": "ignoré"}],
        30.0,
    )
    assert [(u["start"], u["end"], u["text"]) for u in utterances] == [
        (5.0, 9.0, "b"), (5.0, 5.0, "c"), (28.0, 30.0, "d"),
    ]
    assert rad_audio.params_key("whisper-large-v3", "fr", "", 600) != rad_audio.params_key(
        "whisper-large-v3", "en", "", 600
    )


def test_speakers_are_scoped_to_their_segment():
    result = rad_audio._SegmentResult(
        index=2, start=30.0, duration=30.0, ok=True, text="Oui. Non. Peut-être.",
        utterances=[{"start": 0.0, "end": 1.0, "text": "Oui.", "speaker": "SPEAKER_00"},
                    {"start": 1.0, "end": 2.0, "text": "Non.", "speaker": "SPEAKER_00"},
                    {"start": 2.0, "end": 3.0, "text": "Peut-être.", "speaker": "SPEAKER_01"}],
    )
    assert rad_audio._segment_body(result) == (
        "[SPEAKER_00 · segment 2] Oui. Non.\n[SPEAKER_01 · segment 2] Peut-être."
    )
