"""Régressions de la revue du sprint Albert (lot F1) : réessai, limiteur, OCR, runner Celery.

Tout est hors ligne : ``FakeAlbert`` derrière un ``httpx.MockTransport``,
``FakeRedis`` en mémoire, clés factices seulement.

* **429 sans Retry-After** : un dépassement de débit par minute n'est plus
  transformé en « quota journalier atteint » avant qu'une fenêtre complète
  d'une minute se soit écoulée ;
* **limiteur en asynchrone** : l'attente du limiteur n'occupe plus les fils
  de l'exécuteur par défaut de la boucle (bouton d'arrêt, notes OpenAI…) ;
* **pause du limiteur** : la pause posée après un 429 (Redis) ne bloque plus
  la boucle d'événements ;
* **trace de repli OCR** : les entrées ``OCR_PROVIDER_FALLBACK`` survivent à
  une relance de l'extraction et à une interruption ;
* **journal d'usage OCR** : ``albert_usage.jsonl`` est écrit document par
  document, pas seulement à la fin normale du processus ;
* **runner Celery** : le chemin du journal de dédup n'est plus tronqué au
  premier espace.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.tasks import runner
from scripts import rad_dataframe as rad
from scripts.rad_albert import limiter as albert_limiter
from scripts.rad_albert import retry as albert_retry
from scripts.rad_albert.config import AlbertConfig
from scripts.rad_albert.errors import AlbertQuotaExhausted, classify_http_error
from tests.albert_fakes import FakeAlbert, FakeRedis
from tests.test_albert_ocr import AlbertChain, _build_pdf, _sizes, _zotero_item
from tests.test_celery_tasks import world  # noqa: F401 - fixture réutilisée

CHAT_ENDPOINT = "/v1/chat/completions"
RPM_BODY = '{"detail":"Rate limit exceeded"}'
CSV_FIELDS = [
    "itemKey", "type", "title", "abstract", "date", "url", "doi",
    "authors", "filename", "path", "attachment_title", "texteocr", "texteocr_provider",
    "texteocr_partial", "texteocr_pages_done", "texteocr_pages_total",
]


@pytest.fixture(autouse=True)
def _fresh_limiters():
    """Oublie les limiteurs partagés du processus avant et après chaque test."""
    albert_limiter.reset_limiters()
    yield
    albert_limiter.reset_limiters()


class VirtualClock:
    """Horloge virtuelle : le sommeil avance le temps au lieu d'attendre."""

    def __init__(self):
        """Démarre à t = 0 sans aucun sommeil."""
        self.now = 0.0
        self.sleeps = []

    def sleep(self, seconds):
        """Avance l'horloge de ``seconds`` secondes."""
        self.sleeps.append(float(seconds))
        self.now += float(seconds)

    async def asleep(self, seconds):
        """Version asynchrone de ``sleep`` (aucune attente réelle)."""
        self.sleep(seconds)


def _default_policy(**env):
    """Politique de réessai des défauts du sprint (Albert ON), avec surcharges d'environnement."""
    settings = {"ALBERT_ENABLED": "1"}
    settings.update(env)
    return albert_retry.RetryPolicy.from_config(AlbertConfig.from_env(settings))


def _rpm_429():
    """429 de débit par minute, sans en-tête Retry-After (forme observée en P13)."""
    return classify_http_error(429, RPM_BODY, {}, endpoint=CHAT_ENDPOINT)


# ---------------------------------------------------------------------------
# 1. Un 429 par minute n'est pas un quota journalier
# ---------------------------------------------------------------------------
def test_rpm_429_without_retry_after_is_retried_until_the_minute_window_resets():
    """Un 429 de débit sans Retry-After qui cède à t = 40 s aboutit (défauts du sprint)."""
    clock = VirtualClock()
    attempts = []

    def send():
        """429 tant que la fenêtre de débit du serveur n'est pas remise à zéro (t = 40 s)."""
        attempts.append(clock.now)
        if clock.now < 40.0:
            raise _rpm_429()
        return "ok"

    result = albert_retry.call_with_retry(send, policy=_default_policy(), sleep=clock.sleep, rng=random.Random(0))

    assert result == "ok"
    assert attempts[-1] >= 40.0


def test_rpm_429_async_path_is_retried_until_the_minute_window_resets():
    """Même garantie sur le chemin asynchrone (``acall_with_retry``)."""
    clock = VirtualClock()

    async def send():
        """429 jusqu'à t = 40 s, puis succès."""
        if clock.now < 40.0:
            raise _rpm_429()
        return "ok"

    result = asyncio.run(albert_retry.acall_with_retry(
        send, policy=_default_policy(), sleep=clock.asleep, rng=random.Random(0),
    ))
    assert result == "ok"


def test_persistent_rpm_429_becomes_quota_only_after_a_full_window_and_one_extra_attempt():
    """Un 429 permanent sans Retry-After devient quota après une fenêtre complète, un essai de plus."""
    clock = VirtualClock()
    attempts = []

    def send():
        """429 permanent, sans Retry-After."""
        attempts.append(clock.now)
        raise _rpm_429()

    policy = _default_policy()
    with pytest.raises(AlbertQuotaExhausted):
        albert_retry.call_with_retry(send, policy=policy, sleep=clock.sleep, rng=random.Random(0))
    # Une fenêtre d'une minute complète s'est écoulée depuis le premier 429 ...
    assert attempts[-1] - attempts[0] >= 60.0
    # ... au prix d'un seul essai supplémentaire (boucle bornée).
    assert len(attempts) == policy.max_retries + 2


def test_persistent_429_with_retry_after_keeps_the_existing_escalation():
    """Avec un Retry-After annoncé, l'escalade en quota reste celle d'avant (réessais épuisés)."""
    clock = VirtualClock()
    attempts = []

    def send():
        """429 permanent avec un Retry-After court (délai annoncé par le serveur)."""
        attempts.append(clock.now)
        raise classify_http_error(429, RPM_BODY, {"Retry-After": "1"}, endpoint=CHAT_ENDPOINT)

    policy = _default_policy(ALBERT_MAX_RETRIES="2")
    with pytest.raises(AlbertQuotaExhausted):
        albert_retry.call_with_retry(send, policy=policy, sleep=clock.sleep, rng=random.Random(0))
    assert len(attempts) == 3
    assert clock.sleeps == [1.0, 1.0]


def test_rpm_429_without_retries_allowed_is_not_waited():
    """``ALBERT_MAX_RETRIES=0`` : aucun réessai ni attente, même pour compléter la fenêtre."""
    clock = VirtualClock()

    def send():
        """429 permanent, sans Retry-After."""
        raise _rpm_429()

    with pytest.raises(AlbertQuotaExhausted):
        albert_retry.call_with_retry(send, policy=_default_policy(ALBERT_MAX_RETRIES="0"),
                                     sleep=clock.sleep, rng=random.Random(0))
    assert clock.sleeps == []


# ---------------------------------------------------------------------------
# 2. L'attente du limiteur n'occupe pas l'exécuteur par défaut
# ---------------------------------------------------------------------------
def _send_ok():
    """Envoi synchrone réussi."""
    return "ok"


def _noop():
    """Travail sans rapport confié à l'exécuteur par défaut."""
    return None


def _paused_limiter(kind, seconds):
    """Limiteur local ou Redis (faux) mis en pause pour ``seconds`` secondes réelles."""
    if kind == "local":
        limiter = albert_limiter.TokenBucket(45, 115000, name="recode")
    else:
        limiter = albert_limiter.RedisWindowLimiter(45, 115000, name="recode", client=FakeRedis(clock=time.time))
    limiter.pause(seconds)
    return limiter


@pytest.mark.parametrize("kind", ["local", "redis"])
def test_limiter_wait_does_not_hold_default_executor_threads(kind):
    """Pendant une pause du limiteur, l'exécuteur par défaut reste libre (seau local et Redis)."""
    async def scenario():
        """Dix envois limités pendant une pause ; mesure un ``to_thread`` sans rapport."""
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(6))
        limiter = _paused_limiter(kind, 1.0)
        tasks = [
            asyncio.ensure_future(albert_retry.acall_with_retry(
                _send_ok, policy=albert_retry.RetryPolicy(), limiter=limiter, tokens=100,
            ))
            for _ in range(10)
        ]
        await asyncio.sleep(0.05)
        started = time.monotonic()
        await asyncio.to_thread(_noop)
        elapsed = time.monotonic() - started
        results = await asyncio.gather(*tasks)
        return elapsed, results

    elapsed, results = asyncio.run(scenario())
    assert results == ["ok"] * 10
    assert elapsed < 0.2, f"exécuteur par défaut saturé par les attentes du limiteur ({elapsed:.2f} s)"


def test_local_limiter_wait_still_delays_the_send():
    """Le seau local réservé sans fil d'exécuteur retarde toujours l'envoi jusqu'à la fin de la pause."""
    async def scenario():
        """Un envoi pendant une pause de 0,3 s ne part qu'après la pause."""
        limiter = _paused_limiter("local", 0.3)
        started = time.monotonic()
        await albert_retry.acall_with_retry(_send_ok, policy=albert_retry.RetryPolicy(), limiter=limiter)
        return time.monotonic() - started

    assert asyncio.run(scenario()) >= 0.25


# ---------------------------------------------------------------------------
# 3. La pause posée après un 429 ne bloque pas la boucle
# ---------------------------------------------------------------------------
class SlowPauseRedis(FakeRedis):
    """Faux Redis dont la lecture de l'échéance de pause est lente (lien Redis dégradé)."""

    delay = 0.4

    def get(self, name):
        """Lecture ; celle de la clé de pause prend ``delay`` secondes."""
        if str(name).endswith(":pause"):
            time.sleep(self.delay)
        return super().get(name)


def test_redis_pause_after_429_does_not_block_the_event_loop():
    """La pause Redis posée après un 429 ne bloque pas la boucle d'événements."""
    limiter = albert_limiter.RedisWindowLimiter(45, None, name="recode", client=SlowPauseRedis(clock=time.time))
    calls = []

    async def send():
        """Un 429 (Retry-After 0,1 s), puis succès."""
        calls.append(1)
        if len(calls) == 1:
            raise classify_http_error(429, "", {"Retry-After": "0.1"}, endpoint=CHAT_ENDPOINT)
        return "ok"

    async def scenario():
        """Mesure le plus grand écart entre deux tics de la boucle pendant l'appel."""
        gaps = []
        stop = asyncio.Event()

        async def ticker():
            """Tic toutes les 5 ms ; note l'écart réel entre deux tics."""
            last = time.monotonic()
            while not stop.is_set():
                await asyncio.sleep(0.005)
                now = time.monotonic()
                gaps.append(now - last)
                last = now

        task = asyncio.ensure_future(ticker())
        await asyncio.sleep(0)
        result = await albert_retry.acall_with_retry(send, policy=albert_retry.RetryPolicy(), limiter=limiter)
        stop.set()
        await task
        return result, max(gaps)

    result, worst_gap = asyncio.run(scenario())
    assert result == "ok"
    assert len(calls) == 2
    assert worst_gap < 0.2, f"boucle d'événements bloquée {worst_gap:.2f} s par la pause Redis"


# ---------------------------------------------------------------------------
# 4. Trace durable des replis OCR (OCR_PROVIDER_FALLBACK)
# ---------------------------------------------------------------------------
def _write_zotero_export(folder, keys):
    """Export Zotero minimal (un PDF par élément) ; renvoie le chemin du JSON."""
    items = [_zotero_item(key, f"files/{key.lower()}.pdf", title=f"Titre {key}") for key in keys]
    path = folder / "export.json"
    path.write_text(json.dumps({"items": items}, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _fake_fallback_item(item, pdf_base_dir, item_index=0):
    """Élément servi par Mistral alors qu'Albert était sélectionné (repli tracé)."""
    key = item["key"]
    record = {name: "" for name in CSV_FIELDS}
    record.update({
        "itemKey": key, "title": item["title"], "filename": f"{key.lower()}.pdf",
        "texteocr": f"Texte Mistral de {key}", "texteocr_provider": "mistral",
        "texteocr_partial": False, "texteocr_pages_done": 1, "texteocr_pages_total": 1,
    })
    error = {
        "itemKey": key, "title": item["title"], "error_type": "OCR_PROVIDER_FALLBACK",
        "error_message": "OCR Albert (albert_lightonocr) en échec ou indisponible : texte servi par mistral.",
        "provider": "mistral", "fallback_from": "albert_lightonocr",
        "pdf_path": os.path.join(pdf_base_dir, f"files/{key.lower()}.pdf"),
        "timestamp": "2026-09-27 10:00:00",
    }
    return rad.ItemProcessingResult(item_key=key, records=[record], errors=[error], success=True)


def _fallback_entries(output_csv):
    """Entrées ``OCR_PROVIDER_FALLBACK`` du fichier d'erreurs, par clé d'élément."""
    with open(rad.get_errors_file_path(output_csv), encoding="utf-8") as fh:
        data = json.load(fh)
    return sorted(entry["itemKey"] for entry in data["errors"] if entry.get("error_type") == "OCR_PROVIDER_FALLBACK")


@pytest.mark.parametrize("workers", [1, 2])
def test_fallback_trace_survives_a_second_extraction_click(tmp_path, monkeypatch, workers):
    """Un second lancement de l'extraction garde les entrées de repli des lignes déjà traitées."""
    monkeypatch.setattr(rad, "PDF_EXTRACTION_WORKERS", workers)
    monkeypatch.setattr(rad, "_process_single_zotero_item", _fake_fallback_item)
    json_path = _write_zotero_export(tmp_path, ["A1", "B2"])
    output_csv = str(tmp_path / "output.csv")

    rad.load_zotero_to_dataframe_incremental(json_path, str(tmp_path), output_csv)
    assert _fallback_entries(output_csv) == ["A1", "B2"]

    # Second clic sur l'étape d'extraction : tout est déjà fait.
    rad.load_zotero_to_dataframe_incremental(json_path, str(tmp_path), output_csv)
    assert _fallback_entries(output_csv) == ["A1", "B2"]


def test_fallback_trace_survives_an_interrupted_run(tmp_path, monkeypatch):
    """Après un arrêt brutal puis une reprise, le repli du premier élément reste tracé."""
    monkeypatch.setattr(rad, "PDF_EXTRACTION_WORKERS", 1)
    seen = []

    def interrupted(item, pdf_base_dir, item_index=0):
        """Premier élément traité, arrêt brutal pendant le second (bouton Stop)."""
        seen.append(item["key"])
        if len(seen) == 2:
            raise KeyboardInterrupt
        return _fake_fallback_item(item, pdf_base_dir, item_index)

    monkeypatch.setattr(rad, "_process_single_zotero_item", interrupted)
    json_path = _write_zotero_export(tmp_path, ["A1", "B2"])
    output_csv = str(tmp_path / "output.csv")
    with pytest.raises(KeyboardInterrupt):
        rad.load_zotero_to_dataframe_incremental(json_path, str(tmp_path), output_csv)

    # Reprise : seul B2 est traité ; le repli de A1 (ligne déjà dans le CSV) reste tracé.
    monkeypatch.setattr(rad, "_process_single_zotero_item", _fake_fallback_item)
    rad.load_zotero_to_dataframe_incremental(json_path, str(tmp_path), output_csv)
    assert _fallback_entries(output_csv) == ["A1", "B2"]


def test_fresh_run_discards_the_previous_fallback_trace(tmp_path, monkeypatch):
    """Un nouveau départ (progression effacée) n'hérite pas des replis de l'ancien run."""
    monkeypatch.setattr(rad, "PDF_EXTRACTION_WORKERS", 1)
    monkeypatch.setattr(rad, "_process_single_zotero_item", _fake_fallback_item)
    json_path = _write_zotero_export(tmp_path, ["A1"])
    output_csv = str(tmp_path / "output.csv")
    rad.load_zotero_to_dataframe_incremental(json_path, str(tmp_path), output_csv)

    # Nouveau départ (progression effacée) : Albert sert cette fois le texte, aucun repli.
    os.remove(rad.get_progress_file_path(output_csv))

    def served_by_albert(item, pdf_base_dir, item_index=0):
        """Même élément, servi par Albert (aucune entrée de repli)."""
        result = _fake_fallback_item(item, pdf_base_dir, item_index)
        record = dict(result.records[0], texteocr_provider="albert_lightonocr")
        return rad.ItemProcessingResult(item_key=result.item_key, records=[record], errors=[], success=True)

    monkeypatch.setattr(rad, "_process_single_zotero_item", served_by_albert)
    rad.load_zotero_to_dataframe_incremental(json_path, str(tmp_path), output_csv)
    assert _fallback_entries(output_csv) == []


def test_no_fallback_leaves_no_trace_file(tmp_path, monkeypatch):
    """Sans repli (Albert désactivé), aucun fichier nouveau n'est créé."""
    monkeypatch.setattr(rad, "PDF_EXTRACTION_WORKERS", 1)

    def plain(item, pdf_base_dir, item_index=0):
        """Élément servi sans repli (Albert désactivé)."""
        result = _fake_fallback_item(item, pdf_base_dir, item_index)
        return rad.ItemProcessingResult(item_key=result.item_key, records=result.records, errors=[], success=True)

    monkeypatch.setattr(rad, "_process_single_zotero_item", plain)
    json_path = _write_zotero_export(tmp_path, ["A1"])
    before = set(os.listdir(tmp_path))
    rad.load_zotero_to_dataframe_incremental(json_path, str(tmp_path), str(tmp_path / "output.csv"))
    created = set(os.listdir(tmp_path)) - before
    assert created == {"output.csv", "output.progress.json", "output_errors.json"}


# ---------------------------------------------------------------------------
# 5. Journal d'usage OCR écrit document par document
# ---------------------------------------------------------------------------
def _usage_records(folder):
    """Enregistrements de ``albert_usage.jsonl`` d'un dossier ([] si absent)."""
    path = folder / "albert_usage.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _lightonocr_session(tmp_path, monkeypatch, keys):
    """Session d'extraction avec le maillon LightOnOCR actif et le journal d'usage activé."""
    albert = AlbertChain(monkeypatch, FakeAlbert(), mode="chat", env={"ALBERT_USAGE_LOG": "1"})
    sizes = _sizes(2)
    albert.router(sizes)
    for key in keys:
        _build_pdf(tmp_path / "files", key.lower(), sizes=sizes)
    monkeypatch.setattr(rad, "PDF_EXTRACTION_WORKERS", 1)
    return albert, _write_zotero_export(tmp_path, keys)


def test_ocr_usage_ledger_is_flushed_before_an_interruption(tmp_path, monkeypatch):
    """Le ledger OCR des documents persistés est sur disque même si le processus est arrêté."""
    albert, json_path = _lightonocr_session(tmp_path, monkeypatch, ["A1", "B2"])
    real_process = rad._process_single_zotero_item
    seen = []

    def interrupted(item, pdf_base_dir, item_index=0):
        """Premier PDF traité par LightOnOCR, arrêt brutal pendant le second."""
        seen.append(item["key"])
        if len(seen) == 2:
            raise KeyboardInterrupt
        return real_process(item, pdf_base_dir, item_index)

    monkeypatch.setattr(rad, "_process_single_zotero_item", interrupted)
    output_csv = str(tmp_path / "output.csv")
    with pytest.raises(KeyboardInterrupt):
        rad.load_zotero_to_dataframe_incremental(json_path, str(tmp_path), output_csv)
    # Le processus s'arrête : le ledger en mémoire est perdu, _write_albert_ocr_usage jamais appelé.
    monkeypatch.setattr(rad, "_ALBERT_OCR_LEDGER", None)

    with open(output_csv, encoding="utf-8-sig") as fh:
        assert "albert_lightonocr" in fh.read()
    records = _usage_records(tmp_path)
    assert records, "albert_usage.jsonl absent alors qu'une ligne servie par Albert est persistée"
    assert all(record["model"] == "lightonocr-2-1b" for record in records)
    assert len(records) == len(albert.chat_calls())


def test_ocr_usage_ledger_written_once_per_record_after_a_full_run(tmp_path, monkeypatch):
    """Écriture par document puis finale : chaque enregistrement une seule fois, chemin renvoyé."""
    albert, json_path = _lightonocr_session(tmp_path, monkeypatch, ["A1", "B2"])
    output_csv = str(tmp_path / "output.csv")
    rad.load_zotero_to_dataframe_incremental(json_path, str(tmp_path), output_csv)
    written = rad._write_albert_ocr_usage(output_csv)

    assert written == str(tmp_path / "albert_usage.jsonl")
    records = _usage_records(tmp_path)
    assert len(records) == len(albert.chat_calls()) == 4
    assert len({(r["ts"], r["request_id"]) for r in records}) == len(records)


# ---------------------------------------------------------------------------
# 6. Runner Celery : chemin du journal de dédup avec espaces
# ---------------------------------------------------------------------------
SPACED_JOURNAL = "/tmp/dir with space/dedup_journal.jsonl"
SPACED_STDOUT = (
    "=== Result ===\n"
    "Status: success\n"
    "Inserted: 3\n"
    "Skipped (dedup): 1\n"
    f"Dedup journal: {SPACED_JOURNAL}\n"
)


def test_parse_vectordb_stdout_keeps_a_spaced_journal_path():
    """Le chemin du journal de dédup avec espaces est lu en entier."""
    assert runner.parse_vectordb_stdout(SPACED_STDOUT)["journal_path"] == SPACED_JOURNAL
    # Ligne qui ne nomme pas le journal de rad_vectordb : premier mot, comme la route.
    assert runner.parse_vectordb_stdout("Dedup journal: /a/b c.jsonl\n")["journal_path"] == "/a/b"


def test_celery_vectordb_task_returns_the_full_journal_path(world, monkeypatch):  # noqa: F811
    """La tâche Celery vectordb renvoie le chemin complet du journal (dossier avec espaces)."""
    def _run_script(cmd, env, *, on_progress=None, timeout=None, **kwargs):
        """rad_vectordb.py réussi dont le journal est sous un dossier avec espaces."""
        return runner.ScriptResult(0, SPACED_STDOUT, "")

    monkeypatch.setattr(runner, "run_script", _run_script)
    extra = {"db_choice": "pinecone", "pinecone_index_name": "idx"}
    assert world.submit("vectordb", "member_keys", "csess-a", extra).status_code == 200
    result = world.run_last_delay()
    assert result["journal_path"] == SPACED_JOURNAL
    assert result["skipped_count"] == 1
