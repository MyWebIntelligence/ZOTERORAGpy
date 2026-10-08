"""Tests hors ligne des outils d'évaluation humaine Albert (``scripts/eval/albert``).

Aucun appel réseau : la fiche OCR est construite depuis un petit PDF créé par
PyMuPDF, la fiche audio depuis un petit WAV avec un découpage simulé (le vrai
découpage ffmpeg n'est testé que si ffmpeg est installé). Tous les fichiers
sont écrits dans ``tmp_path``. Le test du script de
la page HTML s'exécute sous Node avec un DOM minimal ; il est sauté si Node
n'est pas installé. (La fiche de pertinence de la recherche a été retirée le
2026-10-03 avec la fonction de réponses sourcées.)
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pytest

from scripts.eval.albert import audio_sheets as A
from scripts.eval.albert import common as C
from scripts.eval.albert import metrics as M
from scripts.eval.albert import ocr_sheets as O

# ---------------------------------------------------------------------------
# Données synthétiques
# ---------------------------------------------------------------------------
def _read_sheet(path: Path) -> List[Dict[str, str]]:
    """Lignes d'une fiche CSV produite (BOM, « ; »)."""
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), delimiter=";"))


def _without_images(page: str) -> str:
    """Page HTML sans ses images base64 (un mot cherché ne doit pas y être trouvé par hasard)."""
    return re.sub(r"data:image/[a-z]+;base64,[A-Za-z0-9+/=]+", "data:image", page)


def _key(path: Path) -> List[Dict]:
    """Enregistrements d'une clé JSON Lines."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_filled(path: Path, rows: List[Dict[str, str]], columns: Sequence[str], delimiter: str = ";",
                  bom: bool = True) -> Path:
    """Écrit une fiche remplie (séparateur et BOM au choix)."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(columns), delimiter=delimiter, lineterminator="\r\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: row.get(c, "") for c in columns})
    path.write_bytes((("﻿" if bom else "") + buf.getvalue()).encode("utf-8"))
    return path


def _make_pdf(path: Path, pages: int = 3) -> Path:
    """Petit PDF de ``pages`` pages (texte simple)."""
    import fitz

    doc = fitz.open()
    for n in range(1, pages + 1):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {n} du livre de test : texte imprimé.")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()
    return path


def _ocr_session(tmp_path: Path, with_compare: bool = True, extra: str = "") -> Dict[str, Path]:
    """Session OCR synthétique : un PDF de 3 pages, la page 2 en échec LightOnOCR.

    Args:
        tmp_path: dossier temporaire du test.
        with_compare: ajouter une colonne recodée pour la comparaison A/B.
        extra: texte ajouté à la transcription brute de la page 3.

    Returns:
        Chemins ``session``, ``csv`` et ``pdf``.
    """
    session = tmp_path / "session"
    pdf = _make_pdf(session / "files" / "KEY1" / "livre.pdf")
    raw = ("Préambule ignoré\n<!-- Page 1 -->\nPage 1 du livre de test : texte imprimé.\n"
           f"<!-- Page 2 -->\n<!-- OCR ÉCHOUÉ (albert_lightonocr) : délai dépassé {pdf} -->\n"
           "<!-- Page 3 -->\nPage 3 du livre de test : texte imprimé." + extra)
    recoded = "\n".join(f"<!-- Page {n} -->\nPage {n} du livre (version recodée)." for n in (1, 2, 3))
    csv_path = session / "output.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["title", "filename", "path", "texteocr", "texteocr_provider", "texte_recode"])
        writer.writerow(["Livre", "livre.pdf", str(pdf), raw, "albert_lightonocr", recoded if with_compare else ""])
    return {"session": session, "csv": csv_path, "pdf": pdf}


def _ocr_argv(tmp_path: Path, out: Path, with_compare: bool) -> List[str]:
    """Arguments de ``ocr_sheets.main`` pour la session synthétique de ``tmp_path``."""
    argv = ["--pdf-dir", str(tmp_path / "session"), "--csv", str(tmp_path / "session" / "output.csv"),
            "--out", str(out), "--pages-per-doc", "3"]
    return argv + (["--compare-column", "texte_recode"] if with_compare else [])


def _build_ocr(tmp_path: Path, with_compare: bool = True, name: str = "camp", extra: str = "") -> Path:
    """Génère la fiche OCR ; renvoie le dossier de campagne."""
    _ocr_session(tmp_path, with_compare, extra)
    out = tmp_path / name
    assert O.main(_ocr_argv(tmp_path, out, with_compare)) == 0
    return out


def _fill_ocr(out: Path, prefer: str = "brut", delimiter: str = ";", bom: bool = True) -> Path:
    """Remplit la fiche OCR : 3 aux pages réussies, 0 aux échecs, préférence pour ``prefer``.

    Args:
        out: dossier de campagne.
        prefer: version préférée dans les comparaisons A/B (``brut`` ou ``comparée``).
        delimiter: séparateur de la fiche écrite.
        bom: écrire un BOM UTF-8.

    Returns:
        Le chemin de ``fiche_ocr_rempli.csv``.
    """
    key = M.load_key(out / "ocr_items.jsonl")
    rows = _read_sheet(out / "fiche_ocr.csv")
    for row in rows:
        item = key["items"][row["item_id"]]
        if item["kind"] == "ab":
            row["preference"] = "A" if item["a"] == prefer else "B"
            continue
        row["note"] = "0" if item.get("failed") else "3"
    return _write_filled(out / "fiche_ocr_rempli.csv", rows, O.COLUMNS, delimiter, bom)


# ---------------------------------------------------------------------------
# Fiche OCR
# ---------------------------------------------------------------------------
def test_split_pages_markers_and_failures():
    """Découpage sur les marqueurs ; échecs de page et de part repérés."""
    text = ("avant\n<!-- Part 1/2 (pages 1-2) -->\n<!-- Page 1 -->\nUn\n<!-- Page 2 -->\n"
            "<!-- OCR ÉCHOUÉ (albert_lightonocr) : délai -->\n"
            "<!-- Part 2/2 (pages 3-4) — OCR ÉCHOUÉ: 429 -->")
    pages = O.split_pages(text)
    assert sorted(pages) == [1, 2, 3, 4]
    assert pages[1].text == "Un" and not pages[1].failed
    assert pages[2].failed and "OCR ÉCHOUÉ" in pages[2].text
    assert pages[3].failed and pages[4].failed and "Part 2/2" in pages[3].text
    assert O.split_pages("sans marqueur") == {}
    assert O.neutral_text("<!-- OCR ÉCHOUÉ (albert_lightonocr) : x -->", Path("/nulle/part")) == \
        "<!-- OCR ÉCHOUÉ : x -->"


def test_ocr_sheet_embeds_images_without_local_paths(tmp_path):
    """Image intégrée, page en échec présente, aucun chemin absolu, fournisseur masqué."""
    out = _build_ocr(tmp_path, with_compare=False)
    raw_page = (out / "fiche_ocr.html").read_text(encoding="utf-8")
    page = _without_images(raw_page)
    sheet = (out / "fiche_ocr.csv").read_text(encoding="utf-8-sig")
    key_text = (out / "ocr_items.jsonl").read_text(encoding="utf-8")
    assert raw_page.count("data:image/png;base64,") == 3
    for text in (page, sheet, key_text):
        assert str(tmp_path) not in text and os.path.realpath(str(tmp_path)) not in text
        assert "KEY1" not in text
    for text in (page, sheet):
        assert "albert_lightonocr" not in text
        for banned in ("http://", "https://", "<link"):
            assert banned not in text
    rows = _read_sheet(out / "fiche_ocr.csv")
    assert list(rows[0].keys()) == list(O.COLUMNS)
    first = [r for r in rows if r["item_id"].startswith("p-")]
    assert sorted(int(r["page"]) for r in first) == [1, 2, 3]
    failed = [r for r in first if r["page"] == "2"]
    assert "OCR ÉCHOUÉ" in failed[0]["transcription"] and "livre.pdf" in failed[0]["transcription"]
    items = [r for r in _key(out / "ocr_items.jsonl") if r["record"] == "item"]
    assert any(r["failed"] for r in items if r["kind"] == "fidelity")
    assert {r["document"] for r in items} == {"livre.pdf"}
    assert [r for r in rows if r["item_id"].startswith("r-")]
    for text in ("0</b> illisible/faux", "1</b> nombreuses erreurs", "2</b> erreurs mineures", "3</b> fidèle"):
        assert text in page


def test_ocr_sheet_blind_ab_comparison(tmp_path):
    """Comparaison A/B à l'aveugle : ni nom de colonne, ni version désignée."""
    out = _build_ocr(tmp_path, with_compare=True)
    page = _without_images((out / "fiche_ocr.html").read_text(encoding="utf-8"))
    sheet = (out / "fiche_ocr.csv").read_text(encoding="utf-8-sig")
    for text in (page, sheet):
        assert "texte_recode" not in text and "brut" not in text and "comparée" not in text
    assert "Version A" in page and "Version B" in page and "Comparaison A/B" in page
    rows = _read_sheet(out / "fiche_ocr.csv")
    assert len([r for r in rows if r["type"] == "comparaison"]) == 3
    assert len([r for r in rows if r["item_id"].startswith("p-")]) == 6
    pairs = [r for r in _key(out / "ocr_items.jsonl") if r.get("kind") == "ab"]
    assert all({p["a"], p["b"]} == {"brut", "comparée"} for p in pairs)
    # Reproductible à graine fixe
    again = _build_ocr(tmp_path / "bis", with_compare=True)
    assert (again / "fiche_ocr.csv").read_bytes() == (out / "fiche_ocr.csv").read_bytes()


def test_ocr_html_is_self_contained_and_escaped(tmp_path):
    """Page autonome : aucune ressource externe, aucun dialogue natif, textes échappés."""
    out = _build_ocr(tmp_path, with_compare=False, extra=" <script>alert(1)</script> <img src=x onerror=y>")
    page = _without_images((out / "fiche_ocr.html").read_text(encoding="utf-8"))
    assert "<script>alert" not in page and "<img src=x" not in page
    assert "\\u003cscript\\u003e" in page
    for banned in ("http://", "https://", "<link", "@import", "<script src", "url("):
        assert banned not in page
    code = re.findall(r"<script>(.*?)</script>", page, re.S)
    assert len(code) == 1
    for banned in ("confirm(", "alert(", "prompt(", "innerHTML", "eval(", "fetch(", "XMLHttpRequest"):
        assert banned not in code[0]
    assert "textContent" in code[0] and "localStorage" in code[0] and "try {" in code[0]
    for text in ("Exporter le CSV", "Réinitialiser", "Oui, tout effacer"):
        assert text in page
    data = json.loads(re.search(r'id="sheet-data">(.*?)</script>', page, re.S).group(1))
    assert data["seed"] == C.DEFAULT_SEED and data["sheet_id"].startswith("fiche_ocr-")
    assert data["columns"] == list(O.COLUMNS)


def test_ocr_regeneration_never_overwrites_judgments(tmp_path):
    """Une fiche remplie n'est réécrite qu'avec --force, et une copie est gardée."""
    out = _build_ocr(tmp_path, with_compare=False)
    rows = _read_sheet(out / "fiche_ocr.csv")
    rows[0]["note"] = "2"
    _write_filled(out / "fiche_ocr.csv", rows, O.COLUMNS)
    argv = _ocr_argv(tmp_path, out, with_compare=False)
    assert O.main(argv) == 2
    assert _read_sheet(out / "fiche_ocr.csv")[0]["note"] == "2"
    assert O.main(argv + ["--force"]) == 0
    assert _read_sheet(out / "fiche_ocr.csv")[0]["note"] == ""
    assert list((out / "sauvegardes").glob("fiche_ocr_*.csv"))


# ---------------------------------------------------------------------------
# Mesures
# ---------------------------------------------------------------------------
def test_weighted_kappa_values():
    """Kappa = 1 pour des jugements identiques ; 11/12 sur l'exemple connu ; indéfini sans variabilité."""
    assert C.weighted_kappa([(0, 0), (1, 1), (2, 2), (3, 3), (2, 2)]) == pytest.approx(1.0)
    assert C.weighted_kappa([(0, 0), (1, 1), (2, 3), (3, 3)]) == pytest.approx(11 / 12)
    # Linéaire et non pondéré sur le même exemple (calculés à la main)
    assert C.weighted_kappa([(0, 0), (1, 1), (2, 3), (3, 3)], weights="none") == pytest.approx(2 / 3)
    assert C.weighted_kappa([(2, 2), (2, 2)]) is None
    assert C.weighted_kappa([(1, 2)]) is None
    assert C.weighted_kappa([(0, 3), (3, 0), (1, 2), (2, 1)]) < 0


def test_csv_reading_semicolon_bom_comma_and_cp1252(tmp_path):
    """Lecture tolérante : « ; » avec BOM, « , » sans BOM, Windows-1252 signalé."""
    a = tmp_path / "a.csv"
    a.write_bytes("﻿item_id;note;commentaire\r\nq1-i1;2;très bien\r\n".encode("utf-8"))
    b = tmp_path / "b.csv"
    b.write_bytes("Item_ID , Note ,commentaire\nq1-i1,2,très bien\n".encode("utf-8"))
    c = tmp_path / "c.csv"
    c.write_bytes("item_id;preference\nab-001;égal\n".encode("cp1252"))
    rows_a, warn_a = C.read_csv_rows(a)
    rows_b, warn_b = C.read_csv_rows(b)
    assert rows_a[0][1] == rows_b[0][1] == {"item_id": "q1-i1", "note": "2", "commentaire": "très bien"}
    assert warn_a == warn_b == []
    rows_c, warn_c = C.read_csv_rows(c)
    assert C.parse_preference(rows_c[0][1]["preference"]) == "égal" and warn_c
    assert C.parse_grade("2,0") == 2 and C.parse_grade(" ") is None
    with pytest.raises(ValueError):
        C.parse_grade("4")
    assert C.parse_preference("egal") == "égal" and C.parse_preference("b") == "B"


def test_metrics_comma_separated_sheet_gives_same_decision(tmp_path):
    """Une fiche OCR réenregistrée avec « , » et sans BOM donne la même décision."""
    out = _build_ocr(tmp_path, with_compare=True)
    key = M.load_key(out / "ocr_items.jsonl")
    _fill_ocr(out)
    values, errors, _warnings = M.read_judgments(out / "fiche_ocr_rempli.csv", key)
    assert errors == []
    reference = M.decide_d21(M.ocr_metrics(key, values))
    _fill_ocr(out, delimiter=",", bom=False)
    values, errors, _warnings = M.read_judgments(out / "fiche_ocr_rempli.csv", key)
    assert errors == []
    decision = M.decide_d21(M.ocr_metrics(key, values))
    assert decision == reference and decision["skip_recode"] is True


def test_metrics_incomplete_or_stale_sheet_blocks_decision(tmp_path):
    """Un élément non jugé, ou une ligne qui ne correspond plus à la clé, bloque la décision D21."""
    out = _build_ocr(tmp_path, with_compare=True)
    _fill_ocr(out)
    rows = _read_sheet(out / "fiche_ocr_rempli.csv")
    stale = next(r for r in rows if r["item_id"].startswith("p-"))
    stale["page"] = str(int(stale["page"]) + 10)
    _write_filled(out / "fiche_ocr_rempli.csv", rows, O.COLUMNS)
    key = M.load_key(out / "ocr_items.jsonl")
    values, errors, _warnings = M.read_judgments(out / "fiche_ocr_rempli.csv", key)
    assert len(errors) == 1 and "différent(s) de la clé" in errors[0] and stale["item_id"] in errors[0]
    decision = M.decide_d21(M.ocr_metrics(key, values))
    assert decision["verdict"] == "jugements incomplets : pas de décision D21"
    assert decision["skip_recode"] is False and decision["decided"] is False


def test_ocr_metrics_and_d21_gate_end_to_end(tmp_path):
    """Pages réussies fidèles, version brute préférée → ne plus recoder ; recodée préférée → garder."""
    out = _build_ocr(tmp_path, with_compare=True)
    key = M.load_key(out / "ocr_items.jsonl")
    rows = _read_sheet(out / "fiche_ocr.csv")

    def fill(prefer: str) -> Dict:
        """Remplit la fiche : 3 aux pages réussies, 0 aux échecs ; préférence pour ``prefer``."""
        for row in rows:
            item = key["items"][row["item_id"]]
            if item["kind"] == "ab":
                row["preference"] = "A" if item["a"] == prefer else "B"
            else:
                row["note"] = "0" if item.get("failed") else "3"
        _write_filled(out / "fiche_ocr_rempli.csv", rows, O.COLUMNS)
        assert M.main(["--dir", str(out)]) == 0
        return json.loads((out / "metriques.json").read_text(encoding="utf-8"))["ocr"]

    result = fill("brut")
    assert result["metrics"]["population"]["mean"] == pytest.approx(3.0)
    assert result["metrics"]["failed"]["n"] == 1
    assert result["metrics"]["ab"]["compared_share"] == pytest.approx(0.0)
    assert result["decision"]["skip_recode"] is True
    result = fill("comparée")
    assert result["metrics"]["ab"]["compared_share"] == pytest.approx(1.0)
    assert result["decision"]["skip_recode"] is False
    assert "garder le recodage" in result["decision"]["verdict"]
    report = (out / "rapport_decision.md").read_text(encoding="utf-8")
    assert "D21" in report and "ALBERT_OCR_SKIP_RECODE" in report


def _ocr_summary(grades: Sequence[int], compared: int = 0, raw: int = 0, missing: Sequence[str] = ()) -> Dict:
    """Résumé minimal pour ``decide_d21``."""
    n = len(grades)
    return {"provider": M.D21_PROVIDER, "missing": list(missing),
            "population": {"n": n, "judged": n, "mean": (sum(grades) / n) if n else None,
                           "low_share": (sum(1 for g in grades if g <= 1) / n) if n else None},
            "ab": {"n": compared + raw, "decided": compared + raw,
                   "compared_share": (compared / (compared + raw)) if compared + raw else None}}


@pytest.mark.parametrize("grades, compared, raw, missing, skip", [
    ([3] * 19 + [2], 0, 0, (), True),
    ([3] * 19 + [1], 0, 0, (), True),            # 5 % de pages ≤ 1 : seuil tenu
    ([3] * 18 + [1, 1], 0, 0, (), False),        # 10 % de pages ≤ 1
    ([2] * 10 + [3] * 9, 0, 0, (), False),       # moyenne 2,47
    ([3] * 20, 6, 4, (), False),                 # recodée préférée dans 60 % des paires
    ([3] * 20, 5, 5, (), True),                  # 50 %
    ([3] * 20, 0, 0, ("p-001",), False),         # fiche incomplète
])
def test_d21_gate_outcomes(grades, compared, raw, missing, skip):
    """Critères D21 : moyenne ≥ 2,5, ≤ 5 % de pages ≤ 1, recodée préférée < 60 %."""
    decision = M.decide_d21(_ocr_summary(grades, compared, raw, missing))
    assert decision["skip_recode"] is skip


def test_metrics_without_material_exits_2(tmp_path):
    """Dossier sans fiche ni clé : code 2, aucun rapport."""
    assert M.main(["--dir", str(tmp_path)]) == 2
    assert not (tmp_path / "rapport_decision.md").exists()


# ---------------------------------------------------------------------------
# Script de la page HTML (Node, DOM minimal)
# ---------------------------------------------------------------------------
_HARNESS = r"""
const fs = require('fs');
const html = fs.readFileSync(process.argv[2], 'utf8');
const data = html.match(/<script type="application\/json" id="sheet-data">([\s\S]*?)<\/script>/)[1];
const code = html.match(/<script>([\s\S]*?)<\/script>/)[1];
class El {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase(); this.children = []; this._text = ''; this.attrs = {};
    this.listeners = {}; this.style = {}; this.className = ''; this.value = ''; this.disabled = false;
    this.paused = true; this.currentTime = 5;
    const self = this;
    this.classList = {
      add(c) { const s = new Set(self.className.split(' ').filter(Boolean)); s.add(c); self.className = [...s].join(' '); },
      remove(c) { self.className = self.className.split(' ').filter(x => x && x !== c).join(' '); },
      toggle(c, f) { const has = self.className.split(' ').includes(c); const w = f === undefined ? !has : f; if (w) this.add(c); else this.remove(c); },
      contains(c) { return self.className.split(' ').includes(c); },
    };
  }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() { return this._text + this.children.map(c => c.textContent).join(''); }
  appendChild(c) { this.children.push(c); return c; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  click() { (this.listeners.click || []).forEach(f => f({ target: this })); }
  focus() { doc.activeElement = this; }
  blur() { doc.activeElement = null; }
  remove() {}
  scrollIntoView() {}
  play() { this.paused = false; }
  pause() { this.paused = true; }
  find(p) { if (p(this)) return this; for (const c of this.children) { const r = c.find(p); if (r) return r; } return null; }
}
const byId = {};
['list', 'item', 'status', 'progress-text', 'progress-fill', 'export', 'next-undone', 'import', 'reset',
 'reset-no', 'reset-yes', 'confirm'].forEach(id => { byId[id] = new El('div'); });
byId['sheet-data'] = { textContent: data };
const docListeners = {};
const doc = { activeElement: null, body: new El('body'), createElement: t => new El(t),
  getElementById: id => byId[id] || byId.item.find(e => e.id === id),
  addEventListener: (t, f) => { (docListeners[t] = docListeners[t] || []).push(f); } };
const mem = {};
const storage = { getItem: k => (k in mem ? mem[k] : null), setItem: (k, v) => { mem[k] = String(v); },
  removeItem: k => { delete mem[k]; } };
let exported = null;
global.document = doc;
global.window = { localStorage: storage, scrollTo() {} };
global.Blob = class { constructor(parts) { exported = parts.join(''); } };
global.URL = { createObjectURL: () => 'blob:x', revokeObjectURL() {} };
global.FileReader = class {};
global.setTimeout = () => 0;
function key(k, target) { (docListeners.keydown || []).forEach(f => f({ key: k, target: target || doc.body, preventDefault() {} })); }
new Function(code)();
const storeKey = () => Object.keys(mem).find(k => !k.endsWith(':position'));
const out = {};
key('2'); key('3'); key('ArrowLeft'); key('k'); key('j');
out.store = JSON.parse(mem[storeKey()]);
out.storeKey = storeKey();
key('c');
const ta = doc.activeElement;
out.commentFocused = !!ta && ta.tagName === 'TEXTAREA';
ta.value = 'remarque ; avec "guillemets"';
(ta.listeners.input || []).forEach(f => f({}));
key('0', ta);
out.typingIgnored = JSON.parse(mem[storeKey()])[Object.keys(out.store)[1]].note === '3';
key('Escape', ta);
out.blurred = doc.activeElement === null;
const abButton = byId.list.find(e => e.tagName === 'BUTTON' && e.textContent.startsWith('ab-'));
if (abButton) { abButton.click(); key('b'); out.preference = JSON.parse(mem[storeKey()])[abButton.textContent.split(' ')[0]].preference; }
const clip = byId.item.find(e => e.id === 'clip');
if (clip) {
  out.hasAudioSource = clip.src.startsWith('data:audio/');
  key('p'); out.playing = !clip.paused;
  key('p'); out.pausedAgain = clip.paused;
  key('r'); out.replayed = !clip.paused && clip.currentTime === 0;
}
out.progress = byId['progress-text'].textContent;
byId.export.click();
out.exported = exported;
byId.reset.click();
out.confirmOpen = byId.confirm.classList.contains('open');
byId['reset-yes'].click();
out.left = Object.keys(mem).length;
out.progressAfterReset = byId['progress-text'].textContent;
console.log(JSON.stringify(out));
"""


def _run_page_script(tmp_path: Path, page: Path) -> Dict:
    """Exécute le script de la page sous Node avec un DOM minimal."""
    harness = tmp_path / "harness.js"
    harness.write_text(_HARNESS, encoding="utf-8")
    done = subprocess.run(["node", str(harness), str(page)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node n'est pas installé")
def test_page_script_keyboard_storage_export_and_reset(tmp_path):
    """Clavier, stockage local par fiche et graine, export CSV avec BOM, réinitialisation confirmée."""
    out = _build_ocr(tmp_path, with_compare=False)
    result = _run_page_script(tmp_path, out / "fiche_ocr.html")
    assert result["storeKey"].startswith("ragpy-albert-eval:fiche_ocr-") and result["storeKey"].endswith(
        f":{C.DEFAULT_SEED}")
    first, second = [r["item_id"] for r in _read_sheet(out / "fiche_ocr.csv")][:2]
    notes = {k: v["note"] for k, v in result["store"].items()}
    assert notes == {first: "2", second: "3"}
    assert result["commentFocused"] and result["blurred"] and result["typingIgnored"]
    assert result["progress"].startswith("2 / ")
    lines = result["exported"].split("\r\n")
    assert lines[0] == "﻿" + ";".join(O.COLUMNS)
    assert lines[1].startswith(f"{first};") and lines[1].endswith(";2;;")
    assert lines[2].endswith(';3;;"remarque ; avec ""guillemets"""')
    exported = tmp_path / "fiche_ocr_rempli.csv"
    exported.write_text(result["exported"], encoding="utf-8")
    values, errors, _w = M.read_judgments(exported, M.load_key(out / "ocr_items.jsonl"))
    assert errors == [] and values[first]["note"] == 2 and values[second]["note"] == 3
    assert result["confirmOpen"] and result["left"] == 0 and result["progressAfterReset"].startswith("0 / ")


@pytest.mark.skipif(shutil.which("node") is None, reason="Node n'est pas installé")
def test_page_script_ab_preference(tmp_path):
    """Comparaison A/B : la touche b enregistre la préférence B."""
    out = _build_ocr(tmp_path, with_compare=True)
    result = _run_page_script(tmp_path, out / "fiche_ocr.html")
    assert result["preference"] == "B"


# ---------------------------------------------------------------------------
# Fiche audio
# ---------------------------------------------------------------------------
def _audio_session(tmp_path: Path, n_utterances: int = 25) -> Dict[str, Path]:
    """Session audio synthétique : un WAV de 40 s, un segment réussi, un segment en échec."""
    import wave

    audio_dir = tmp_path / "audio" / "entretiens"
    audio_dir.mkdir(parents=True)
    wav = audio_dir / "entretien.wav"
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(1)
        handle.setframerate(8000)
        handle.writeframes(bytes([128]) * 8000 * 40)
    utterances = [{"start": float(i), "end": i + 0.9, "text": f"Phrase numéro {i} de l'entretien."}
                  for i in range(n_utterances)]
    utterances.append({"start": 26.0, "end": 60.0, "text": "Énoncé trop long pour un extrait."})
    segments = {"version": 1, "provider": "albert_whisper", "files": [{
        "itemKey": "audio-0123456789abcdef", "filename": "entretien.wav", "language": "fr", "duration_s": 40.0,
        "segments": [
            {"index": 1, "start": 0.0, "end": 30.0, "status": "ok", "reused": False, "utterances": utterances},
            {"index": 2, "start": 30.0, "end": 40.0, "status": "failed", "reused": False, "utterances": [],
             "error": f"délai dépassé {wav}"},
        ]}]}
    csv_path = tmp_path / "audio" / "output.csv"
    (tmp_path / "audio" / "output_audio_segments.json").write_text(json.dumps(segments, ensure_ascii=False),
                                                                  encoding="utf-8")
    texte = ("<!-- Segment 1 (00:00:00–00:00:30) -->\n" + " ".join(u["text"] for u in utterances)
             + "\n\n<!-- Segment 2 (00:00:30–00:00:40) -->\n"
             "<!-- TRANSCRIPTION ÉCHOUÉE (albert_whisper) : délai dépassé -->")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["itemKey", "title", "filename", "path", "texteocr", "texteocr_provider"])
        writer.writerow(["audio-0123456789abcdef", "entretien", "entretien.wav", "entretien.wav", texte,
                         "albert_whisper"])
    return {"audio_dir": tmp_path / "audio", "csv": csv_path, "wav": wav}


@pytest.fixture
def fake_clips(monkeypatch):
    """Découpage simulé : octets factices, appels relevés (aucun ffmpeg lancé)."""
    calls: List = []

    def fake(source: Path, start: float, duration: float):
        """Extrait factice."""
        calls.append((Path(source).name, round(start, 3), round(duration, 3)))
        return b"ID3" + f"{start:.3f}".encode("ascii"), "audio/mpeg"

    monkeypatch.setattr(A, "cut_clip", fake)
    return calls


def _build_audio(tmp_path: Path, name: str = "camp") -> Path:
    """Génère la fiche audio ; renvoie le dossier de campagne."""
    files = _audio_session(tmp_path)
    out = tmp_path / name
    assert A.main(["--audio-dir", str(files["audio_dir"]), "--csv", str(files["csv"]), "--out", str(out)]) == 0
    return out


def test_audio_sheet_embeds_clips_without_local_paths(tmp_path, fake_clips):
    """~20 extraits tirés, audio intégré, second jugement, aucun chemin local ni fournisseur."""
    out = _build_audio(tmp_path)
    raw_page = (out / "fiche_audio.html").read_text(encoding="utf-8")
    sheet = (out / "fiche_audio.csv").read_text(encoding="utf-8-sig")
    key_text = (out / "audio_items.jsonl").read_text(encoding="utf-8")
    rows = _read_sheet(out / "fiche_audio.csv")
    assert list(rows[0].keys()) == list(A.COLUMNS)
    first = [r for r in rows if r["item_id"].startswith("a-")]
    second = [r for r in rows if r["item_id"].startswith("r-")]
    assert len(first) == A.DEFAULT_CLIPS_PER_FILE and len(second) == math.ceil(0.2 * len(first))
    assert raw_page.count("data:audio/mpeg;base64,") == len(first)  # un extrait par énoncé, réutilisé en second
    assert len(fake_clips) == len(first)
    assert all(duration <= A.DEFAULT_MAX_CLIP_SECONDS for _name, _start, duration in fake_clips)
    assert "trop long" not in sheet
    page = _without_images(raw_page)
    for text in (page, sheet, key_text):
        assert str(tmp_path) not in text and os.path.realpath(str(tmp_path)) not in text
        assert "entretiens/" not in text
    for text in (page, sheet):
        assert "albert_whisper" not in text
        for banned in ("http://", "https://", "<link"):
            assert banned not in text
    for text in ("0</b> inaudible/faux", "1</b> nombreuses erreurs", "2</b> erreurs mineures", "3</b> fidèle",
                 "p pour écouter"):
        assert text in page
    docs = [r for r in _key(out / "audio_items.jsonl") if r["record"] == "document"]
    assert docs[0]["segments_failed"] == 1 and docs[0]["utterances_too_long"] == 1
    assert A.hms(3725.3) == "01:02:05.3"


def test_audio_sheet_reproducible(tmp_path, fake_clips):
    """Même graine : même fiche."""
    a = _build_audio(tmp_path / "a")
    b = _build_audio(tmp_path / "b")
    assert (a / "fiche_audio.csv").read_bytes() == (b / "fiche_audio.csv").read_bytes()
    assert (a / "fiche_audio.html").read_bytes() == (b / "fiche_audio.html").read_bytes()


def test_audio_sheet_refused_without_ffmpeg(tmp_path, monkeypatch, capsys):
    """Sans ffmpeg : raison explicite, code 2, aucune fiche écrite."""
    monkeypatch.setattr(A, "ffmpeg_path", lambda: None)
    files = _audio_session(tmp_path)
    out = tmp_path / "camp"
    assert A.main(["--audio-dir", str(files["audio_dir"]), "--csv", str(files["csv"]), "--out", str(out)]) == 2
    assert "ffmpeg introuvable" in capsys.readouterr().err
    assert not (out / "fiche_audio.html").exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg n'est pas installé")
def test_cut_clip_with_real_ffmpeg(tmp_path):
    """Découpage réel d'un extrait de 2 s (mp3 non vide)."""
    files = _audio_session(tmp_path)
    data, mime = A.cut_clip(files["wav"], 1.0, 2.0)
    assert mime == "audio/mpeg" and len(data) > 100


def test_audio_metrics_and_gate_end_to_end(tmp_path, fake_clips):
    """Une note ≤ 1 sur 20 (5 %) : utilisable ; deux (10 %) : relecture nécessaire."""
    out = _build_audio(tmp_path)
    rows = _read_sheet(out / "fiche_audio.csv")

    def fill(low: int) -> Dict:
        """3 partout sauf ``low`` extraits du premier jugement notés 1 ; seconds jugements identiques."""
        lows = {r["item_id"] for r in rows if r["item_id"].startswith("a-")}
        lows = set(sorted(lows)[:low])
        key = M.load_key(out / "audio_items.jsonl")
        for row in rows:
            original = key["items"][row["item_id"]].get("of", row["item_id"])
            row["note"] = "1" if original in lows else "3"
        _write_filled(out / "fiche_audio_rempli.csv", rows, A.COLUMNS)
        assert M.main(["--dir", str(out)]) == 0
        return json.loads((out / "metriques.json").read_text(encoding="utf-8"))["audio"]

    result = fill(1)
    assert result["metrics"]["population"]["low_share"] == pytest.approx(0.05)
    assert result["metrics"]["segments_failed"] == 1
    assert result["decision"]["usable"] is True
    result = fill(2)
    assert result["decision"]["usable"] is False
    report = (out / "rapport_decision.md").read_text(encoding="utf-8")
    assert "## 2. Transcription audio" in report and "relecture humaine nécessaire" in report


@pytest.mark.parametrize("grades, missing, verdict, usable", [
    ([3] * 20, (), "utilisable sans relecture", True),
    ([2] * 10 + [3] * 9, (), "relecture humaine", False),     # moyenne 2,47
    ([3] * 18 + [1, 0], (), "relecture humaine", False),      # 10 % ≤ 1
    ([3] * 20, ("a-001",), "jugements incomplets", False),
])
def test_audio_gate_outcomes(grades, missing, verdict, usable):
    """Critères audio : moyenne ≥ 2,5 et au plus 5 % d'extraits notés ≤ 1."""
    n = len(grades)
    summary = {"missing": list(missing), "population": {"n": n, "judged": n, "mean": sum(grades) / n,
                                                         "low_share": sum(1 for g in grades if g <= 1) / n}}
    decision = M.decide_audio(summary)
    assert verdict in decision["verdict"] and decision["usable"] is usable


@pytest.mark.skipif(shutil.which("node") is None, reason="Node n'est pas installé")
def test_page_script_audio_keys(tmp_path, fake_clips):
    """Fiche audio : la touche p lance puis met en pause l'extrait, r le relance depuis le début."""
    out = _build_audio(tmp_path)
    result = _run_page_script(tmp_path, out / "fiche_audio.html")
    assert result["hasAudioSource"] and result["playing"] and result["pausedAgain"] and result["replayed"]
