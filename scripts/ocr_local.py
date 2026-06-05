"""
ocr_local.py — OCR LOCAL isolé (Lot 4, SPRINT_ocr_local_docling).

Script CLI **autonome** invoqué en subprocess par `scripts/rad_dataframe.py`
(`_extract_text_with_local`). Il transcrit un PDF en Markdown + marqueurs
`<!-- Page N -->` (même contrat que Mistral) à l'aide d'un moteur OCR **local**
(Docling par défaut), **sans clé API, sans cap de pages, hors-ligne**.

Pourquoi un script séparé : les dépendances lourdes (Docling → torch/easyocr,
plusieurs Go) ne doivent **jamais** être importées dans le process FastAPI. Ici
elles ne sont chargées que dans ce subprocess éphémère (import paresseux).

Usage :
    python scripts/ocr_local.py --input book.pdf --output book.md
    python scripts/ocr_local.py --input book.pdf            # markdown sur stdout
    python scripts/ocr_local.py --check                     # 0 si moteur dispo, 1 sinon

Codes de sortie : 0 = OK ; 1 = moteur indisponible (--check) ; 2 = sortie vide ;
3 = erreur de conversion. Les messages d'erreur vont sur stderr.
"""

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - ocr_local - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("ocr_local")

# Paquet Python à sonder pour `available()`/`--check`, par moteur.
ENGINE_IMPORT = {
    "docling": "docling",
    "mineru": "magic_pdf",
    "marker": "marker",
}


def render_markdown_with_page_markers(doc, max_pages=None) -> str:
    """
    Rend un `DoclingDocument` en Markdown avec marqueurs `<!-- Page N -->`.

    Contrat aval (chunking, book_note_generator) : markdown + une borne
    `<!-- Page N -->` par page, numérotation **continue**. Docling
    `export_to_markdown()` exporte le document entier sans bornes ; on tente
    donc un export **par page** (`export_to_markdown(page_no=...)`), avec repli
    sur un export global mono-marqueur si la version installée ne supporte pas
    l'argument.

    Cette fonction est **duck-typée** (testable sans Docling) : `doc` doit
    exposer `.pages` (mapping {num_page: ...}) et `.export_to_markdown([page_no])`.

    Args:
        doc: DoclingDocument (ou objet compatible).
        max_pages: cap optionnel sur le nombre de pages rendues (None/0 = tout).

    Returns:
        Markdown paginé (str). Chaîne vide si le document ne produit rien.
    """
    page_numbers = []
    pages_attr = getattr(doc, "pages", None)
    if pages_attr:
        try:
            page_numbers = sorted(int(p) for p in pages_attr.keys())
        except (AttributeError, TypeError, ValueError):
            page_numbers = []
    if max_pages and max_pages > 0:
        page_numbers = page_numbers[:max_pages]

    if page_numbers:
        try:
            blocks = []
            for pno in page_numbers:
                md = (doc.export_to_markdown(page_no=pno) or "").strip()
                blocks.append(f"<!-- Page {pno} -->\n{md}".rstrip())
            return "\n\n".join(blocks)
        except TypeError:
            # Version Docling sans paramètre `page_no` → repli export global.
            logger.warning(
                "export_to_markdown(page_no=...) non supporté — repli sur un "
                "export global mono-page (pagination approximative)."
            )

    whole = (doc.export_to_markdown() or "").strip()
    return f"<!-- Page 1 -->\n{whole}" if whole else ""


def _build_docling_converter(device: str):
    """Construit un DocumentConverter Docling (OCR activé, device best-effort).

    L'API d'options pipeline varie selon les versions ; on tente la
    configuration accelerator/OCR et on retombe sur le converter par défaut si
    l'API diffère.
    """
    from docling.document_converter import DocumentConverter  # type: ignore

    try:
        from docling.datamodel.base_models import InputFormat  # type: ignore
        from docling.datamodel.pipeline_options import (  # type: ignore
            PdfPipelineOptions,
            AcceleratorOptions,
            AcceleratorDevice,
        )
        from docling.document_converter import PdfFormatOption  # type: ignore

        try:
            n_threads = int(os.getenv("LOCAL_OCR_THREADS", "0")) or min(os.cpu_count() or 4, 8)
        except (TypeError, ValueError):
            n_threads = min(os.cpu_count() or 4, 8)
        accel = AcceleratorOptions(
            device=AcceleratorDevice.CUDA if device == "cuda" else AcceleratorDevice.CPU,
            num_threads=n_threads,
        )
        popts = PdfPipelineOptions()
        popts.accelerator_options = accel
        popts.do_ocr = True  # indispensable pour les PDF scannés
        # Perf : pas de détection de tableaux par défaut (TableFormer est lourd
        # sur CPU et inutile pour des livres de prose). Réactivable via
        # LOCAL_OCR_TABLES=1 si le corpus contient beaucoup de tableaux.
        popts.do_table_structure = os.getenv("LOCAL_OCR_TABLES", "0") in ("1", "true", "True")
        # Résolution de rendu (DPI) : images_scale=1.0 ≈ 72 DPI. Baisser =
        # plus rapide mais OCR moins précis ; monter = plus lent, plus précis.
        # 0 (défaut) = laisser le défaut Docling.
        try:
            scale = float(os.getenv("LOCAL_OCR_IMAGE_SCALE", "0") or 0)
        except (TypeError, ValueError):
            scale = 0.0
        if scale > 0:
            popts.images_scale = scale
        logger.info(
            "Docling perf: threads=%d, table_structure=%s, images_scale=%s",
            n_threads, popts.do_table_structure, getattr(popts, "images_scale", "default"),
        )

        # Choix du moteur OCR (LOCAL_OCR_ENGINE_OCR) :
        #  - "tesseract" (défaut) : binaire CPU, langues FR/EN via apt, 100 % offline.
        #  - "rapidocr"          : PP-OCR (plus rapide CPU). NB: le rapidocr unifié
        #    télécharge ses modèles (modelscope) au 1er run ; use_cls=False évite
        #    le modèle de classification (orientation) inutile sur scans droits.
        ocr_engine = os.getenv("LOCAL_OCR_ENGINE_OCR", "tesseract").strip().lower()
        try:
            if ocr_engine == "rapidocr":
                from docling.datamodel.pipeline_options import RapidOcrOptions  # type: ignore

                use_cls = os.getenv("LOCAL_OCR_USE_CLS", "0") in ("1", "true", "True")
                popts.ocr_options = RapidOcrOptions(force_full_page_ocr=False, use_cls=use_cls)
                logger.info("Moteur OCR: RapidOCR (use_cls=%s).", use_cls)
            else:
                from docling.datamodel.pipeline_options import TesseractCliOcrOptions  # type: ignore

                popts.ocr_options = TesseractCliOcrOptions(lang=["fra", "eng"])
                logger.info("Moteur OCR: Tesseract CLI (lang=fra+eng).")
        except Exception as exc:  # noqa: BLE001 — repli sur le moteur OCR par défaut
            logger.warning("Options OCR '%s' indisponibles (%s) — moteur Docling par défaut.", ocr_engine, exc)

        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=popts)}
        )
    except Exception as exc:  # noqa: BLE001 — API d'options best-effort
        logger.warning("Options pipeline Docling indisponibles (%s) — converter par défaut.", exc)
        return DocumentConverter()


def convert(input_path: str, engine: str = "docling", device: str = "cpu", max_pages=None) -> str:
    """Convertit un PDF en markdown paginé via le moteur local choisi."""
    if engine != "docling":
        raise NotImplementedError(
            f"Moteur OCR local '{engine}' non encore implémenté (seul 'docling' l'est)."
        )
    converter = _build_docling_converter(device)
    logger.info("Docling: conversion de %s (device=%s, max_pages=%s)…", input_path, device, max_pages)
    # IMPORTANT : limiter la conversion (donc l'OCR réel) à N pages via
    # `page_range`, sinon Docling OCRise tout le document avant la troncature au
    # rendu (un livre de 357 p en CPU = très long).
    if max_pages and max_pages > 0:
        try:
            result = converter.convert(input_path, page_range=(1, max_pages))
        except TypeError:
            logger.warning("page_range non supporté par cette version Docling — conversion complète puis troncature.")
            result = converter.convert(input_path)
    else:
        result = converter.convert(input_path)
    return render_markdown_with_page_markers(result.document, max_pages=max_pages)


def check_engine(engine: str) -> bool:
    """True si le paquet du moteur est importable (sans charger torch ici)."""
    import importlib.util

    pkg = ENGINE_IMPORT.get(engine, engine)
    try:
        return importlib.util.find_spec(pkg) is not None
    except (ImportError, ValueError):
        return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="OCR local (Docling) → markdown paginé.")
    parser.add_argument("--input", help="Chemin du PDF à OCRiser.")
    parser.add_argument("--output", help="Fichier markdown de sortie (défaut: stdout).")
    parser.add_argument("--engine", default="docling", help="Moteur OCR local (défaut: docling).")
    parser.add_argument("--device", default="cpu", help="cpu | cuda (défaut: cpu).")
    parser.add_argument("--max-pages", type=int, default=0, help="Cap de pages (0 = aucun).")
    parser.add_argument("--check", action="store_true", help="Teste la disponibilité du moteur puis sort.")
    args = parser.parse_args(argv)

    if args.check:
        ok = check_engine(args.engine)
        print(f"{args.engine}: {'available' if ok else 'unavailable'}")
        return 0 if ok else 1

    if not args.input:
        logger.error("--input requis (hors --check).")
        return 3

    try:
        markdown = convert(
            args.input,
            engine=args.engine,
            device=args.device,
            max_pages=args.max_pages or None,
        )
    except Exception as exc:  # noqa: BLE001 — toute erreur moteur → rc=3 + stderr
        logger.error("Échec OCR local (%s) pour %s: %s", args.engine, args.input, exc)
        return 3

    if not markdown.strip():
        logger.error("OCR local (%s) : sortie vide pour %s", args.engine, args.input)
        return 2

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(markdown)
        logger.info("Markdown écrit dans %s (%d caractères).", args.output, len(markdown))
    else:
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
