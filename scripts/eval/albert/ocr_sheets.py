"""Fiche de jugement de fidélité OCR (décision D21 : recoder ou non les textes LightOnOCR).

À partir d'une session RAGpy (PDF + ``output.csv`` de ``rad_dataframe.py``),
quelques pages par document sont tirées à graine fixe. Chaque page est
présentée avec son image (rendue par PyMuPDF, environ 110 DPI, intégrée en
base64 dans la page HTML) à côté de sa transcription (texte entre deux
marqueurs ``<!-- Page N -->`` de la colonne ``texteocr``). Les pages en échec
(marqueur ``<!-- OCR ÉCHOUÉ … -->``, ou part découpée en échec) sont aussi
présentées : elles se jugent comme les autres et sont comptées à part.

Avec ``--compare-column``, une seconde transcription de la même page (par
exemple le texte recodé, avec les mêmes marqueurs de page) est jugée à
l'aveugle :

- chacune des deux transcriptions est notée seule, mêlée aux autres pages ;
- une comparaison « Version A / Version B » (ordre tiré au hasard, graine
  fixe) demande une préférence : A, B ou égal.

Le juge ne voit ni le fournisseur d'OCR, ni le nom de la colonne comparée,
ni quelle version est la transcription brute. Environ 20 % des notes de
fidélité sont demandées une seconde fois (identifiants ``r-<n>``, autre
ordre) pour mesurer l'accord.

Fichiers écrits dans le dossier de campagne (``--out``, défaut
``data/albert_eval/<AAAA-MM-JJ>``) : ``fiche_ocr.html`` (à juger dans un
navigateur, l'image est nécessaire), ``fiche_ocr.csv`` (même fiche, sans les
images) et ``ocr_items.jsonl`` (clé privée, à ne pas ouvrir avant d'avoir
jugé). Aucun chemin local absolu n'y figure : seulement des noms de fichier.

Exemple (une ligne) ::

    .venv/bin/python scripts/eval/albert/ocr_sheets.py --pdf-dir uploads/<session> --csv uploads/<session>/output.csv --provider albert_lightonocr
"""

from __future__ import annotations

import argparse
import base64
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.eval.albert import common as C  # noqa: E402

COLUMNS: Tuple[str, ...] = ("item_id", "type", "document", "page", "transcription", "version_a", "version_b",
                            "note", "preference", "commentaire")
TEXT_COLUMNS: Tuple[str, ...] = ("transcription", "version_a", "version_b", "commentaire")
SHEET_STEM = "fiche_ocr"
ITEMS_FILE = "ocr_items.jsonl"
TYPE_FIDELITY = "fidélité"
TYPE_AB = "comparaison"
RAW = "brut"
COMPARED = "comparée"

DEFAULT_PAGES_PER_DOC = 5
DEFAULT_DPI = 110
MAX_SIDE_PX = 1600
FALLBACK_DPI = 72
WARN_HTML_MB = 40
DEFAULT_MAX_HTML_MB = 120
SECONDS_FIDELITY = 30
"""Durée estimée pour noter une transcription face à l'image de la page."""
SECONDS_AB = 45
"""Durée estimée d'une comparaison A/B."""

PAGE_RE = re.compile(r"<!--\s*Page\s+(\d{1,6})\s*-->")
PART_RE = re.compile(r"<!--\s*Part\s+\d+\s*/\s*\d+\s*\(\s*pages\s+(\d+)\s*-\s*(\d+)\s*\)(.*?)-->", re.S)
FAILED_RE = re.compile(r"OCR\s+[ÉE]CHOU[ÉE]", re.I)
_FAILED_PROVIDER_RE = re.compile(r"(OCR\s+[ÉE]CHOU[ÉE])\s*\([^)\n]*\)", re.I)


def neutral_text(text: str, pdf_dir: Path) -> str:
    """Texte montré au juge : sans chemin local ni nom de fournisseur dans les marqueurs d'échec.

    Args:
        text: transcription d'une page.
        pdf_dir: dossier des PDF (retiré du texte).

    Returns:
        Le texte nettoyé (``OCR ÉCHOUÉ (albert_lightonocr) : …`` devient ``OCR ÉCHOUÉ : …``).
    """
    return _FAILED_PROVIDER_RE.sub(r"\1", C.scrub_paths(text, [pdf_dir]))


@dataclass
class PageText:
    """Transcription d'une page.

    Attributes:
        number: numéro de page (à partir de 1).
        text: texte entre son marqueur et le suivant (marqueur d'échec compris).
        failed: page en échec d'OCR.
    """

    number: int
    text: str
    failed: bool = False


@dataclass
class Document:
    """Document retenu pour la fiche.

    Attributes:
        name: nom de fichier affiché (unique dans la fiche).
        pdf: chemin du PDF (jamais écrit dans les fiches).
        provider: fournisseur d'OCR (``texteocr_provider``).
        pages: transcriptions par numéro de page.
        compare: transcriptions comparées par numéro de page.
        page_count: nombre de pages du PDF.
        sampled: pages tirées.
    """

    name: str
    pdf: Path
    provider: str
    pages: Dict[int, PageText]
    compare: Dict[int, PageText] = field(default_factory=dict)
    page_count: int = 0
    sampled: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Texte par page
# ---------------------------------------------------------------------------
def split_pages(text: Any) -> Dict[int, PageText]:
    """Découpe une transcription sur ses marqueurs ``<!-- Page N -->``.

    Les marqueurs de part (``<!-- Part i/N (pages A-B) -->``) sont retirés ;
    une part en échec (``… OCR ÉCHOUÉ …``) produit une page en échec, de
    texte égal au marqueur, pour chacune de ses pages sans marqueur. Le texte
    placé avant le premier marqueur est ignoré.

    Args:
        text: contenu de ``texteocr``.

    Returns:
        Les pages, par numéro (vide s'il n'y a aucun marqueur de page).
    """
    raw = "" if text is None else str(text)
    failed_parts: List[Tuple[int, int, str]] = []
    for match in PART_RE.finditer(raw):
        if FAILED_RE.search(match.group(3) or ""):
            failed_parts.append((int(match.group(1)), int(match.group(2)), match.group(0).strip()))
    cleaned = PART_RE.sub("", raw)
    marks = list(PAGE_RE.finditer(cleaned))
    pages: Dict[int, PageText] = {}
    for i, match in enumerate(marks):
        number = int(match.group(1))
        end = marks[i + 1].start() if i + 1 < len(marks) else len(cleaned)
        body = cleaned[match.end():end].strip()
        failed = bool(FAILED_RE.search(body))
        if number in pages:
            previous = pages[number]
            pages[number] = PageText(number, (previous.text + "\n\n" + body).strip(), previous.failed or failed)
        else:
            pages[number] = PageText(number, body, failed)
    for first, last, marker in failed_parts:
        for number in range(first, last + 1):
            if number not in pages:
                pages[number] = PageText(number, marker, True)
    return pages


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
class PdfIndex:
    """Recherche des PDF d'une session par nom de fichier (sans lire les chemins absolus du CSV)."""

    def __init__(self, root: Path) -> None:
        """Prépare l'index (construit à la première recherche).

        Args:
            root: dossier des PDF (``--pdf-dir``).
        """
        self.root = Path(root)
        self._by_name: Optional[Dict[str, List[Path]]] = None

    def _index(self) -> Dict[str, List[Path]]:
        """Fichiers ``.pdf`` du dossier, par nom en minuscules."""
        if self._by_name is None:
            self._by_name = {}
            for path in sorted(self.root.rglob("*")):
                if path.is_file() and path.suffix.lower() == ".pdf":
                    self._by_name.setdefault(path.name.lower(), []).append(path)
        return self._by_name

    def find(self, filename: str, path_value: str = "") -> Tuple[Optional[Path], Optional[str]]:
        """Trouve le PDF d'une ligne du CSV.

        Ordre : ``<dossier>/<nom>``, ``<dossier>/<CLÉ>/<nom>`` (dernier
        dossier du chemin d'origine), puis recherche par nom dans tout le
        dossier ; plusieurs homonymes : celui dont le dossier parent est la
        clé du chemin d'origine, sinon le premier (avertissement).

        Args:
            filename: nom de fichier (colonne ``filename``).
            path_value: chemin d'origine (colonne ``path``), seulement pour ses
                deux derniers composants.

        Returns:
            ``(chemin ou None, avertissement ou None)``.
        """
        parts = [p for p in re.split(r"[\\/]", path_value or "") if p]
        name = filename or (parts[-1] if parts else "")
        if not name:
            return None, None
        parent = parts[-2] if len(parts) >= 2 else ""
        for candidate in (self.root / name, self.root / parent / name if parent else None,
                          self.root / "files" / parent / name if parent else None):
            if candidate is not None and candidate.is_file():
                return candidate, None
        matches = self._index().get(name.lower(), [])
        if len(matches) == 1:
            return matches[0], None
        if len(matches) > 1:
            same_parent = [m for m in matches if parent and m.parent.name == parent]
            chosen = same_parent[0] if same_parent else matches[0]
            return chosen, f"{name} : {len(matches)} fichiers homonymes, {chosen.parent.name}/{chosen.name} retenu"
        return None, None


def render_page(doc: Any, number: int, *, dpi: int, image_format: str, gray: bool = False) -> str:
    """Rend une page PDF en image ``data:`` (PNG ou JPEG, base64).

    Args:
        doc: document PyMuPDF ouvert.
        number: numéro de page (à partir de 1).
        dpi: résolution (plus grand côté plafonné à ``MAX_SIDE_PX``).
        image_format: ``png`` ou ``jpeg``.
        gray: rendu en niveaux de gris (fiche trop lourde).

    Returns:
        L'URI ``data:image/...;base64,...``.
    """
    import fitz

    page = doc.load_page(number - 1)
    rect = page.rect
    zoom = max(float(dpi), 1.0) / 72.0
    longest = max(float(rect.width), float(rect.height))
    if longest > 0:
        zoom = min(zoom, MAX_SIDE_PX / longest)
    kwargs: Dict[str, Any] = {"matrix": fitz.Matrix(zoom, zoom)}
    if gray:
        kwargs["colorspace"] = fitz.csGRAY
    pix = page.get_pixmap(**kwargs)
    if image_format == "jpeg":
        data, mime = pix.tobytes("jpeg", jpg_quality=80), "image/jpeg"
    else:
        data, mime = pix.tobytes("png"), "image/png"
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


# ---------------------------------------------------------------------------
# Lecture de la session
# ---------------------------------------------------------------------------
def load_documents(csv_path: Path, pdf_dir: Path, *, compare_column: Optional[str], providers: Sequence[str],
                   pages_per_doc: int, seed: int) -> Tuple[List[Document], List[str]]:
    """Lit le CSV de session, retrouve les PDF et tire les pages à juger.

    Args:
        csv_path: ``output.csv`` de la session.
        pdf_dir: dossier des PDF.
        compare_column: colonne de la seconde transcription (facultatif).
        providers: fournisseurs gardés (vide : tous).
        pages_per_doc: pages tirées par document.
        seed: graine.

    Returns:
        ``(documents, avertissements)`` ; les avertissements ne contiennent
        que des noms de fichier.

    Raises:
        C.EvalError: colonne obligatoire absente.
    """
    import fitz

    rows, warnings = C.read_csv_rows(csv_path)
    if not rows:
        raise C.EvalError(f"{csv_path.name} : aucune ligne")
    header = rows[0][1]
    if "texteocr" not in header:
        raise C.EvalError(f"{csv_path.name} : colonne texteocr absente")
    if "filename" not in header and "path" not in header:
        raise C.EvalError(f"{csv_path.name} : colonne filename ou path absente")
    compare = (compare_column or "").strip().lower() or None
    if compare and compare not in header:
        raise C.EvalError(f"{csv_path.name} : colonne de comparaison {compare_column!r} absente")
    wanted = {p.strip().lower() for p in providers if p.strip()}
    index = PdfIndex(pdf_dir)
    docs: List[Document] = []
    names: Dict[str, int] = {}
    for line, row in rows:
        provider = row.get("texteocr_provider", "")
        if wanted and provider.lower() not in wanted:
            continue
        parts = [p for p in re.split(r"[\\/]", row.get("path", "")) if p]
        filename = row.get("filename", "").strip() or (parts[-1] if parts else "")
        filename = re.split(r"[\\/]", filename)[-1]
        if not filename:
            warnings.append(f"{csv_path.name} ligne {line} : ni filename ni path, ligne ignorée")
            continue
        if not filename.lower().endswith(".pdf"):
            warnings.append(f"{filename} : pas un PDF, ignoré")
            continue
        pages = split_pages(row.get("texteocr", ""))
        if not pages:
            warnings.append(f"{filename} : aucun marqueur <!-- Page N --> dans texteocr, ignoré")
            continue
        pdf, note = index.find(filename, row.get("path", ""))
        if note:
            warnings.append(note)
        if pdf is None:
            warnings.append(f"{filename} : PDF introuvable dans le dossier des PDF, ignoré")
            continue
        try:
            with fitz.open(str(pdf)) as handle:
                page_count = int(handle.page_count)
        except Exception as exc:  # noqa: BLE001 — PDF illisible : signalé, document ignoré
            warnings.append(f"{filename} : PDF illisible ({type(exc).__name__}), ignoré")
            continue
        names[filename] = names.get(filename, 0) + 1
        name = filename if names[filename] == 1 else f"{filename} ({names[filename]})"
        compare_pages = split_pages(row.get(compare, "")) if compare else {}
        if compare and not compare_pages:
            warnings.append(f"{name} : colonne comparée sans marqueur de page, pas de comparaison A/B")
        valid = sorted(n for n in pages if 1 <= n <= page_count)
        if len(valid) < len(pages):
            warnings.append(f"{name} : {len(pages) - len(valid)} marqueur(s) de page au-delà des {page_count} "
                            "pages du PDF, ignoré(s)")
        sampled = sorted(random.Random(f"{seed}:ocr-pages:{name}").sample(valid, min(pages_per_doc, len(valid))))
        docs.append(Document(name=name, pdf=pdf, provider=provider, pages=pages, compare=compare_pages,
                             page_count=page_count, sampled=sampled))
    return docs, [C.scrub_paths(w, [pdf_dir]) for w in warnings]


# ---------------------------------------------------------------------------
# Fiche
# ---------------------------------------------------------------------------
def _doc_block(document: str, page: int) -> Dict[str, str]:
    """Bloc d'en-tête (document et page) d'un élément."""
    return {"label": "Document", "text": f"{document} — page {page}", "cls": "source"}


def build_sheet(docs: Sequence[Document], *, seed: int, second_ratio: float, pdf_dir: Path,
                compare_column: Optional[str]) -> Dict[str, Any]:
    """Construit la fiche à l'aveugle : lignes CSV, éléments HTML, clé privée.

    Args:
        docs: documents et pages tirées.
        seed: graine.
        second_ratio: part des notes de fidélité redemandées.
        pdf_dir: dossier des PDF (retiré de tout texte écrit).
        compare_column: colonne comparée (consignée dans la clé seulement).

    Returns:
        ``{rows, items, key_records, needed_images, n_fidelity, n_ab, n_second}``.
    """
    fidelity: List[Dict[str, Any]] = []
    pairs: List[Dict[str, Any]] = []
    key_records: List[Dict[str, Any]] = []
    for doc in docs:
        key_records.append({"record": "document", "document": doc.name, "provider": doc.provider,
                            "pages_total": doc.page_count, "pages_marked": len(doc.pages),
                            "pages_failed": sum(1 for p in doc.pages.values() if p.failed),
                            "sampled": list(doc.sampled)})
        for number in doc.sampled:
            page = doc.pages[number]
            raw_text = neutral_text(page.text, pdf_dir)
            fidelity.append({"document": doc.name, "page": number, "version": RAW, "text": raw_text,
                             "failed": page.failed, "provider": doc.provider})
            other = doc.compare.get(number)
            if other is None:
                continue
            other_text = neutral_text(other.text, pdf_dir)
            fidelity.append({"document": doc.name, "page": number, "version": COMPARED, "text": other_text,
                             "failed": other.failed, "provider": doc.provider})
            raw_first = random.Random(f"{seed}:ocr-ab:{doc.name}:{number}").random() < 0.5
            pairs.append({"document": doc.name, "page": number, "provider": doc.provider,
                          "a": RAW if raw_first else COMPARED, "b": COMPARED if raw_first else RAW,
                          "text_a": raw_text if raw_first else other_text,
                          "text_b": other_text if raw_first else raw_text})
    def order_key(entry: Mapping[str, Any]) -> Tuple[str, int, str]:
        """Ordre stable avant mélange : document, page, version."""
        return entry["document"], entry["page"], entry.get("version", "")

    fidelity = C.seeded_shuffle(sorted(fidelity, key=order_key), seed, "ocr:fidelite")
    pairs = C.seeded_shuffle(sorted(pairs, key=order_key), seed, "ocr:ab")
    rows: List[Dict[str, str]] = []
    items: List[Dict[str, Any]] = []
    image_of: Dict[Tuple[str, int], str] = {}
    for doc_name, page in sorted({(e["document"], e["page"]) for e in fidelity}):
        image_of[(doc_name, page)] = ""
    for n, (doc_name, page) in enumerate(sorted(image_of), 1):
        image_of[(doc_name, page)] = f"img-{n:0{C.id_width(len(image_of))}d}"

    def add(row: Dict[str, str], *, section: str, input_kind: str, blocks: List[Dict[str, str]],
            doc_name: str, page: int) -> None:
        """Ajoute une ligne CSV et l'élément HTML correspondant."""
        rows.append(row)
        items.append({"id": row["item_id"], "section": section, "input": input_kind,
                      "row": {c: C.csv_cell(row.get(c, ""), text=c in TEXT_COLUMNS) for c in COLUMNS},
                      "blocks": blocks, "image": image_of[(doc_name, page)],
                      "image_alt": f"Page {page} de {doc_name}"})

    width = C.id_width(len(fidelity))
    by_id: Dict[str, Dict[str, Any]] = {}
    for n, entry in enumerate(fidelity, 1):
        item_id = f"p-{n:0{width}d}"
        by_id[item_id] = entry
        add({"item_id": item_id, "type": TYPE_FIDELITY, "document": entry["document"], "page": str(entry["page"]),
             "transcription": entry["text"]}, section="Fidélité", input_kind="note",
            blocks=[_doc_block(entry["document"], entry["page"]),
                    {"label": "Transcription", "text": entry["text"], "cls": "transcription"}],
            doc_name=entry["document"], page=entry["page"])
        key_records.append({"record": "item", "item_id": item_id, "kind": "fidelity", "document": entry["document"],
                            "page": entry["page"], "provider": entry["provider"], "version": entry["version"],
                            "failed": entry["failed"]})
    width_ab = C.id_width(len(pairs))
    for n, pair in enumerate(pairs, 1):
        item_id = f"ab-{n:0{width_ab}d}"
        add({"item_id": item_id, "type": TYPE_AB, "document": pair["document"], "page": str(pair["page"]),
             "version_a": pair["text_a"], "version_b": pair["text_b"]}, section="Comparaison A/B",
            input_kind="preference",
            blocks=[_doc_block(pair["document"], pair["page"]),
                    {"label": "Version A", "text": pair["text_a"], "cls": "version"},
                    {"label": "Version B", "text": pair["text_b"], "cls": "version"}],
            doc_name=pair["document"], page=pair["page"])
        key_records.append({"record": "item", "item_id": item_id, "kind": "ab", "document": pair["document"],
                            "page": pair["page"], "provider": pair["provider"], "a": pair["a"], "b": pair["b"]})
    second = C.second_sample(list(by_id), second_ratio, seed, "ocr")
    width_r = C.id_width(len(second))
    for n, original in enumerate(second, 1):
        entry = by_id[original]
        item_id = f"r-{n:0{width_r}d}"
        add({"item_id": item_id, "type": TYPE_FIDELITY, "document": entry["document"], "page": str(entry["page"]),
             "transcription": entry["text"]}, section="Second jugement", input_kind="note",
            blocks=[_doc_block(entry["document"], entry["page"]),
                    {"label": "Transcription", "text": entry["text"], "cls": "transcription"}],
            doc_name=entry["document"], page=entry["page"])
        key_records.append({"record": "item", "item_id": item_id, "kind": "second", "of": original,
                            "document": entry["document"], "page": entry["page"], "provider": entry["provider"],
                            "version": entry["version"], "failed": entry["failed"]})
    sheet_id = C.sheet_fingerprint(SHEET_STEM, [seed, [(r["item_id"], r["document"], r["page"],
                                                        r.get("transcription", ""), r.get("version_a", ""))
                                                       for r in rows]])
    providers = sorted({d.provider for d in docs})
    meta = {"record": "meta", "sheet": "ocr", "sheet_id": sheet_id, "seed": seed, "second_ratio": second_ratio,
            "compare_column": compare_column or None, "providers": providers, "columns": list(COLUMNS),
            "seconds_fidelity": SECONDS_FIDELITY, "seconds_ab": SECONDS_AB}
    return {"rows": rows, "items": items, "key_records": [meta, *key_records], "image_of": image_of,
            "sheet_id": sheet_id, "n_fidelity": len(fidelity), "n_ab": len(pairs), "n_second": len(second)}


def render_images(docs: Sequence[Document], image_of: Mapping[Tuple[str, int], str], *, dpi: int,
                  image_format: str, max_bytes: int) -> Tuple[Dict[str, str], List[str]]:
    """Rend les images des pages de la fiche, dans un budget de taille.

    Au-delà du budget, une page est rendue en niveaux de gris à 72 DPI ;
    si cela ne suffit pas, son image est omise (la page HTML le signale).

    Args:
        docs: documents.
        image_of: identifiant d'image par (document, page).
        dpi: résolution.
        image_format: ``png`` ou ``jpeg``.
        max_bytes: taille totale maximale des images encodées.

    Returns:
        ``(images {id: URI data:}, avertissements)``.
    """
    import fitz

    pdf_of = {d.name: d.pdf for d in docs}
    images: Dict[str, str] = {}
    warnings: List[str] = []
    used = 0
    reduced = omitted = 0
    by_doc: Dict[str, List[int]] = {}
    for doc_name, page in sorted(image_of):
        by_doc.setdefault(doc_name, []).append(page)
    for doc_name, pages in by_doc.items():
        with fitz.open(str(pdf_of[doc_name])) as handle:
            for page in pages:
                uri = render_page(handle, page, dpi=dpi, image_format=image_format)
                if used + len(uri) > max_bytes:
                    uri = render_page(handle, page, dpi=FALLBACK_DPI, image_format=image_format, gray=True)
                    reduced += 1
                    if used + len(uri) > max_bytes:
                        omitted += 1
                        continue
                images[image_of[(doc_name, page)]] = uri
                used += len(uri)
    if reduced:
        warnings.append(f"taille maximale atteinte : {reduced} page(s) rendue(s) en gris à {FALLBACK_DPI} DPI")
    if omitted:
        warnings.append(f"taille maximale atteinte : {omitted} image(s) omise(s) ; réduire --pages-per-doc, "
                        "passer --image-format jpeg ou relever --max-html-mb")
    return images, warnings


INTRO = (
    "Fiche de jugement à l'aveugle de la fidélité des transcriptions OCR : comparez chaque transcription à "
    "l'image de la page. Les pages sont présentées dans un ordre aléatoire, sans indication de l'outil qui "
    "les a produites.",
    "Les éléments « Comparaison A/B » présentent deux transcriptions de la même page dans un ordre tiré au "
    "hasard : choisissez la plus fidèle et la plus lisible. Les éléments « Second jugement » (r-…) "
    "reprennent environ 20 % des notes dans un autre ordre : jugez-les sans revenir au premier jugement.",
)
RULES = (
    "Comparez le texte, l'ordre de lecture, les chiffres, les noms propres et les notes ; la mise en forme "
    "(titres, gras, tableaux en Markdown) ne compte pas.",
    "Une page en échec affiche un marqueur « OCR ÉCHOUÉ » : notez 0 si la page porte du texte, 3 si elle est "
    "réellement vide.",
    "Un ajout absent de la page (texte inventé, résumé, reformulation) est une erreur, même s'il est plausible.",
    "En cas d'hésitation entre deux notes, prenez la plus basse et notez la raison en commentaire.",
)


def write_outputs(out_dir: Path, sheet: Mapping[str, Any], images: Mapping[str, str], *, seed: int,
                  force: bool) -> Dict[str, Path]:
    """Écrit la fiche (CSV, HTML) et la clé.

    Args:
        out_dir: dossier de campagne.
        sheet: résultat de ``build_sheet``.
        images: images rendues.
        seed: graine.
        force: écraser une fiche déjà remplie (copie gardée).

    Returns:
        Les chemins écrits, par nom.

    Raises:
        C.EvalError: fiche déjà remplie, ou export rempli d'une autre fiche, sans ``force``.
    """
    out_dir = Path(out_dir)
    csv_path, html_path, items_path = out_dir / f"{SHEET_STEM}.csv", out_dir / f"{SHEET_STEM}.html", \
        out_dir / ITEMS_FILE
    filled_export = out_dir / f"{SHEET_STEM}_rempli.csv"
    previous_id = None
    if items_path.exists():
        try:
            previous_id = next((r.get("sheet_id") for r in C.read_jsonl(items_path) if r.get("record") == "meta"),
                               None)
        except C.EvalError:
            previous_id = None
    stale = filled_export.exists() and previous_id != sheet["sheet_id"]
    if stale and not force:
        raise C.EvalError(f"{filled_export.name} est l'export d'une autre fiche : rien n'a été écrit. "
                          "Relancer avec --force pour le ranger dans sauvegardes/.")
    backups = C.guard_overwrite([csv_path], force)
    if stale:
        backups.append(C.backup_file(filled_export))
        filled_export.unlink()
    written = {"csv": C.write_csv(csv_path, sheet["rows"], COLUMNS, TEXT_COLUMNS)}
    preferences = C.PREFERENCES if sheet["n_ab"] else None
    written["html"] = C.write_text(html_path, C.render_sheet_html(
        title=f"Fidélité OCR — campagne {out_dir.name}", intro=INTRO, rules=RULES, sheet_id=sheet["sheet_id"],
        seed=seed, file_stem=SHEET_STEM, columns=COLUMNS, items=sheet["items"], scale=C.OCR_SCALE,
        scale_title="Échelle de fidélité de la transcription à l'image de la page :", preferences=preferences,
        images=images))
    written["items"] = C.write_jsonl(items_path, sheet["key_records"])
    for path in backups:
        print(f"copie de sauvegarde : {C.display_path(path)}")
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Point d'entrée de la ligne de commande.

    Args:
        argv: arguments (défaut : ``sys.argv[1:]``).

    Returns:
        0 si la fiche est écrite, 2 en cas d'entrée invalide.
    """
    parser = argparse.ArgumentParser(description="Fiche de fidélité OCR à l'aveugle (D21).")
    parser.add_argument("--pdf-dir", required=True, help="dossier des PDF de la session")
    parser.add_argument("--csv", required=True, help="output.csv de la session (texteocr, texteocr_provider…)")
    parser.add_argument("--out", help="dossier de campagne (défaut : data/albert_eval/<date du jour>)")
    parser.add_argument("--pages-per-doc", type=int, default=DEFAULT_PAGES_PER_DOC)
    parser.add_argument("--seed", type=int, default=C.DEFAULT_SEED)
    parser.add_argument("--second-ratio", type=float, default=C.DEFAULT_SECOND_RATIO)
    parser.add_argument("--compare-column", help="seconde transcription (même marqueurs de page) pour l'A/B")
    parser.add_argument("--provider", default="",
                        help="fournisseurs gardés, séparés par des virgules (ex. albert_lightonocr ; défaut : tous)")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--image-format", choices=("png", "jpeg"), default="png")
    parser.add_argument("--max-html-mb", type=float, default=DEFAULT_MAX_HTML_MB,
                        help="taille maximale des images de la page HTML (Mo)")
    parser.add_argument("--force", action="store_true", help="écraser une fiche déjà remplie (copie gardée)")
    args = parser.parse_args(argv)
    try:
        if args.pages_per_doc < 1:
            raise C.EvalError("--pages-per-doc doit être au moins 1")
        if not 0 <= args.second_ratio < 1:
            raise C.EvalError("--second-ratio doit être dans [0, 1[")
        if not 36 <= args.dpi <= 300:
            raise C.EvalError("--dpi doit être compris entre 36 et 300")
        pdf_dir, csv_path = Path(args.pdf_dir), Path(args.csv)
        if not pdf_dir.is_dir():
            raise C.EvalError(f"dossier des PDF introuvable : {pdf_dir.name}")
        out_dir = Path(args.out) if args.out else C.default_out_dir()
        docs, warnings = load_documents(csv_path, pdf_dir, compare_column=args.compare_column,
                                        providers=args.provider.split(","), pages_per_doc=args.pages_per_doc,
                                        seed=args.seed)
        if not docs:
            for message in warnings:
                print(f"attention : {message}", file=sys.stderr)
            raise C.EvalError("aucun document utilisable (voir les avertissements)")
        sheet = build_sheet(docs, seed=args.seed, second_ratio=args.second_ratio, pdf_dir=pdf_dir,
                            compare_column=args.compare_column)
        images, more = render_images(docs, sheet["image_of"], dpi=args.dpi, image_format=args.image_format,
                                     max_bytes=int(args.max_html_mb * 1024 * 1024))
        warnings += more
        written = write_outputs(out_dir, sheet, images, seed=args.seed, force=args.force)
    except C.EvalError as exc:
        print(f"erreur : {exc}", file=sys.stderr)
        return 2
    for message in warnings:
        print(f"attention : {message}")
    size_mb = written["html"].stat().st_size / (1024 * 1024)
    if size_mb > WARN_HTML_MB:
        print(f"attention : fiche HTML de {size_mb:.0f} Mo (au-delà de {WARN_HTML_MB} Mo, un navigateur peut "
              "ralentir) : réduire --pages-per-doc ou passer --image-format jpeg")
    seconds = (sheet["n_fidelity"] + sheet["n_second"]) * SECONDS_FIDELITY + sheet["n_ab"] * SECONDS_AB
    where = C.display_path(out_dir)
    print(f"fiche OCR : {len(docs)} documents, {sheet['n_fidelity']} transcriptions à noter, "
          f"{sheet['n_ab']} comparaisons A/B, {sheet['n_second']} en second jugement (r-…) ; "
          f"page HTML de {size_mb:.1f} Mo")
    print(f"durée estimée : ({sheet['n_fidelity']} + {sheet['n_second']}) × {SECONDS_FIDELITY} s + "
          f"{sheet['n_ab']} × {SECONDS_AB} s ≈ {C.duration_label(seconds)}")
    print(f"à juger : {where}/{SHEET_STEM}.html (l'image de la page est nécessaire)")
    print(f"ne pas ouvrir avant d'avoir jugé : {where}/{ITEMS_FILE}")
    print(f"ensuite : .venv/bin/python scripts/eval/albert/metrics.py --dir {where}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
