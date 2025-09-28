import os
import json
import re
import unicodedata
import base64
import pandas as pd

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def ascii_flat(s):
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii').lower()

def alphanum_only(s):
    return ''.join(c for c in ascii_flat(s) if c.isalnum())

def levenshtein(a, b):
    # Simple Levenshtein distance (not optimal, but fine for short names)
    if len(a) < len(b):
        return levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    previous_row = range(len(b) + 1)
    for i, ca in enumerate(a):
        current_row = [i + 1]
        for j, cb in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (ca != cb)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]
import fitz  # PyMuPDF
from tqdm import tqdm
import logging
from typing import Optional, List, NamedTuple
import argparse
import requests
from dotenv import load_dotenv

# --- Path and Logging Setup ---
SCRIPT_FILE_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_FILE_PATH)  # Should be /.../__RAG/ragpy/scripts
RAGPY_DIR_SCRIPT = os.path.dirname(SCRIPT_DIR)    # Should be /.../__RAG/ragpy
LOG_DIR_SCRIPT = os.path.join(RAGPY_DIR_SCRIPT, "logs")

os.makedirs(LOG_DIR_SCRIPT, exist_ok=True)
pdf_processing_log_file = os.path.join(LOG_DIR_SCRIPT, 'pdf_processing.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(pdf_processing_log_file),
        logging.StreamHandler() # Keep console output for the script as well
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Script SCRIPT_DIR: {SCRIPT_DIR}")
logger.info(f"Script RAGPY_DIR_SCRIPT: {RAGPY_DIR_SCRIPT}")
logger.info(f"Script LOG_DIR_SCRIPT: {LOG_DIR_SCRIPT}")
logger.info(f"Script log file: {pdf_processing_log_file}")
# --- End Path and Logging Setup ---

load_dotenv()

def _truthy_env(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"Valeur invalide pour {name}='{raw}', fallback sur {default}.")
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"Valeur invalide pour {name}='{raw}', fallback sur {default}.")
        return default


MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_BASE_URL = os.getenv("MISTRAL_API_BASE_URL", "https://api.mistral.ai")
MISTRAL_OCR_MODEL = os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest")
MISTRAL_OCR_TIMEOUT = _env_int("MISTRAL_OCR_TIMEOUT", 300)
MISTRAL_DELETE_UPLOADED_FILE = _truthy_env(os.getenv("MISTRAL_DELETE_UPLOADED_FILE"), True)

OPENAI_OCR_MODEL = os.getenv("OPENAI_OCR_MODEL", "gpt-4o-mini")
OPENAI_OCR_PROMPT = os.getenv(
    "OPENAI_OCR_PROMPT",
    "Transcris cette page PDF en Markdown lisible sans résumer ni modifier le contenu."
)
OPENAI_OCR_MAX_PAGES = _env_int("OPENAI_OCR_MAX_PAGES", 10)
OPENAI_OCR_MAX_TOKENS = _env_int("OPENAI_OCR_MAX_TOKENS", 2048)
OPENAI_OCR_RENDER_SCALE = _env_float("OPENAI_OCR_RENDER_SCALE", 2.0)


class OCRExtractionError(Exception):
    """Raised when OCR extraction fails for all providers."""


class OCRResult(NamedTuple):
    text: str
    provider: str


def _extract_text_with_legacy_pdf(pdf_path: str, max_pages: Optional[int] = None) -> str:
    full_text: List[str] = []
    try:
        with fitz.open(pdf_path) as doc:
            num_pages = min(max_pages, len(doc)) if max_pages else len(doc)
            for page_num in tqdm(
                range(num_pages),
                desc=f"Extracting {os.path.basename(pdf_path)} (legacy)",
            ):
                try:
                    page = doc.load_page(page_num)
                    text = page.get_text("text").strip()
                    if len(text.split()) < 50:
                        text = page.get_text("ocr").strip()
                    full_text.append(text)
                except Exception as page_error:
                    logger.warning(f"Page {page_num} error in {pdf_path}: {page_error}")
                    continue
    except Exception as e:
        logger.error(f"Failed to process {pdf_path}: {e}")
        return ""
    return "\n\n".join(filter(None, full_text))


def _extract_text_with_mistral(pdf_path: str, max_pages: Optional[int] = None) -> str:
    if not MISTRAL_API_KEY:
        raise OCRExtractionError("MISTRAL_API_KEY manquante.")

    base_url = MISTRAL_API_BASE_URL.rstrip("/")
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}

    with requests.Session() as session:
        with open(pdf_path, "rb") as pdf_file:
            files = {
                "file": (os.path.basename(pdf_path), pdf_file, "application/pdf")
            }
            data = {"purpose": "ocr"}
            upload_resp = session.post(
                f"{base_url}/v1/files",
                headers=headers,
                files=files,
                data=data,
                timeout=MISTRAL_OCR_TIMEOUT,
            )

        upload_resp.raise_for_status()
        upload_payload = upload_resp.json()
        file_id = (
            upload_payload.get("id")
            or upload_payload.get("file_id")
            or upload_payload.get("data", {}).get("id")
        )
        if not file_id:
            raise OCRExtractionError(
                "La réponse Mistral ne contient pas d'identifiant de fichier pour l'OCR."
            )

        payload = {
            "model": MISTRAL_OCR_MODEL,
            "document": {
                "type": "file",
                "file_id": file_id,
            },
            "include_image_base64": False,
        }
        if max_pages:
            payload["page_ranges"] = [
                {
                    "start": 1,
                    "end": max_pages,
                }
            ]

        response = session.post(
            f"{base_url}/v1/ocr",
            headers={**headers, "Content-Type": "application/json"},
            json=payload,
            timeout=MISTRAL_OCR_TIMEOUT,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError:
            logger.warning(
                "Appel Mistral OCR échoué (%s) pour %s: %s",
                response.status_code,
                pdf_path,
                response.text,
            )
            raise

        response_payload = response.json()

        text_fragments: List[str] = []
        if isinstance(response_payload, dict):
            full_text = response_payload.get("text")
            if isinstance(full_text, str) and full_text.strip():
                text_fragments.append(full_text.strip())

            pages = response_payload.get("pages")
            if isinstance(pages, list):
                for page in pages:
                    if isinstance(page, dict):
                        page_text = page.get("text")
                        if isinstance(page_text, str) and page_text.strip():
                            text_fragments.append(page_text.strip())

        markdown_text = "\n\n".join(dict.fromkeys(text_fragments))
        markdown_text = markdown_text.strip()
        if not markdown_text:
            raise OCRExtractionError("La réponse Mistral est vide.")

        if MISTRAL_DELETE_UPLOADED_FILE:
            try:
                session.delete(
                    f"{base_url}/v1/files/{file_id}",
                    headers=headers,
                    timeout=15,
                )
            except requests.RequestException as cleanup_error:
                logger.debug(
                    "Échec du nettoyage du fichier OCR Mistral %s: %s",
                    file_id,
                    cleanup_error,
                )

        return markdown_text


def _extract_text_with_openai(
    pdf_path: str,
    api_key: str,
    max_pages: Optional[int] = None,
) -> str:
    from openai import OpenAI

    base_limit = max_pages if max_pages is not None else float("inf")
    max_allowed = OPENAI_OCR_MAX_PAGES if OPENAI_OCR_MAX_PAGES > 0 else float("inf")

    outputs: List[str] = []
    client = OpenAI(api_key=api_key)

    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        limit = int(min(base_limit, max_allowed, total_pages))

        for page_index in tqdm(
            range(limit),
            desc=f"OpenAI OCR {os.path.basename(pdf_path)}",
        ):
            page = doc.load_page(page_index)
            matrix = fitz.Matrix(OPENAI_OCR_RENDER_SCALE, OPENAI_OCR_RENDER_SCALE)
            pix = page.get_pixmap(matrix=matrix)
            image_bytes = pix.tobytes("png")
            image_b64 = base64.b64encode(image_bytes).decode("ascii")

            user_content = [
                {
                    "type": "text",
                    "text": f"{OPENAI_OCR_PROMPT}\nPage {page_index + 1}.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ]

            response = client.chat.completions.create(
                model=OPENAI_OCR_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a meticulous OCR engine that outputs Markdown without omitting any content.",
                    },
                    {"role": "user", "content": user_content},
                ],
                max_tokens=OPENAI_OCR_MAX_TOKENS,
            )

            choice = response.choices[0] if response.choices else None
            page_text = ""
            if choice and getattr(choice, "message", None):
                page_text = (choice.message.content or "").strip()

            if page_text:
                outputs.append(f"<!-- Page {page_index + 1} -->\n{page_text}")

    if not outputs:
        raise OCRExtractionError("La réponse OpenAI est vide.")

    return "\n\n".join(outputs)


def _finalize_ocr_result(text: str, provider: str, return_details: bool):
    if return_details:
        return OCRResult(text=text, provider=provider)
    return text


def extract_text_with_ocr(
    pdf_path: str,
    max_pages: Optional[int] = None,
    *,
    return_details: bool = False,
):
    """Attempt Markdown-first OCR with Mistral, fall back to OpenAI then legacy extraction."""

    last_error: Optional[Exception] = None

    if MISTRAL_API_KEY:
        try:
            logger.debug("Tentative d'OCR Mistral pour %s", pdf_path)
            mistral_text = _extract_text_with_mistral(pdf_path, max_pages=max_pages)
            return _finalize_ocr_result(mistral_text, "mistral", return_details)
        except Exception as mistral_error:
            last_error = mistral_error
            logger.warning(
                "Échec Mistral OCR pour %s: %s",
                pdf_path,
                mistral_error,
            )
    else:
        logger.debug("MISTRAL_API_KEY absente, OCR Mistral ignoré pour %s", pdf_path)

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            logger.debug("Fallback OpenAI OCR pour %s", pdf_path)
            openai_text = _extract_text_with_openai(pdf_path, openai_key, max_pages=max_pages)
            return _finalize_ocr_result(openai_text, "openai", return_details)
        except Exception as openai_error:
            last_error = openai_error
            logger.warning(
                "Échec OpenAI OCR pour %s: %s",
                pdf_path,
                openai_error,
            )
    else:
        logger.debug("OPENAI_API_KEY absente, OCR OpenAI ignoré pour %s", pdf_path)

    if not MISTRAL_API_KEY and not openai_key:
        logger.error(
            "Aucune clé API OCR configurée (MISTRAL_API_KEY ou OPENAI_API_KEY). "
            "Basculer sur l'extraction locale peut dégrader la qualité du texte."
        )

    if last_error:
        logger.info("Retour au processus OCR historique pour %s", pdf_path)

    legacy_text = _extract_text_with_legacy_pdf(pdf_path, max_pages=max_pages)
    if legacy_text.strip():
        return _finalize_ocr_result(legacy_text, "legacy", return_details)

    error_message = (
        "Impossible d'extraire le texte du document. Configurez MISTRAL_API_KEY ou OPENAI_API_KEY "
        "pour activer l'OCR en Markdown."
    )
    raise OCRExtractionError(error_message)

def load_zotero_to_dataframe(json_path: str, pdf_base_dir: str) -> pd.DataFrame:
    """
    Charge les métadonnées Zotero depuis un JSON vers un DataFrame
    avec extraction OCR du texte complet pour chaque PDF.
    Les chemins PDF relatifs dans le JSON sont résolus par rapport à pdf_base_dir.
    """
    records = []
    try:
        logger.info(f"Chargement du fichier JSON Zotero depuis : {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in tqdm(data.get("items", []), desc="Processing Zotero items"):
            try:
                # Extraction des métadonnées de base
                metadata = {
                    "type": item.get("itemType", ""),
                    "title": item.get("title", ""),
                    "date": item.get("date", ""),
                    "url": item.get("url", ""),
                    "doi": item.get("DOI", ""),
                    "authors": ", ".join([
                        f"{c.get('lastName', '').strip()} {c.get('firstName', '').strip()}"
                        for c in item.get("creators", [])
                        if c.get('lastName') or c.get('firstName')
                    ])
                }
                
                # Traitement des attachments PDF
                for attachment in item.get("attachments", []):
                    path_from_json = attachment.get("path", "").strip()
                    if path_from_json and path_from_json.lower().endswith(".pdf"):
                        
                        # Résoudre le chemin du PDF
                        if os.path.isabs(path_from_json):
                            actual_pdf_path = path_from_json
                        else:
                            actual_pdf_path = os.path.join(pdf_base_dir, path_from_json)
                        
                        if not os.path.exists(actual_pdf_path):
                            # Recherche fuzzy avancée : NFC, NFD, sans accents, insensible à la casse
                            base_dir, rel_path = os.path.split(actual_pdf_path)
                            candidates = []
                            for root, dirs, files in os.walk(os.path.dirname(actual_pdf_path)):
                                for f in files:
                                    candidates.append(f)
                            logger.info(f"Fichiers candidats dans {os.path.dirname(actual_pdf_path)} : {candidates}")
                            target_names = [
                                unicodedata.normalize('NFC', os.path.basename(path_from_json)).lower(),
                                unicodedata.normalize('NFD', os.path.basename(path_from_json)).lower(),
                                strip_accents(unicodedata.normalize('NFC', os.path.basename(path_from_json))).lower(),
                                strip_accents(unicodedata.normalize('NFD', os.path.basename(path_from_json))).lower(),
                                ascii_flat(os.path.basename(path_from_json)),
                                alphanum_only(os.path.basename(path_from_json))
                            ]
                            found = False
                            for f in candidates:
                                f_forms = [
                                    unicodedata.normalize('NFC', f).lower(),
                                    unicodedata.normalize('NFD', f).lower(),
                                    strip_accents(unicodedata.normalize('NFC', f)).lower(),
                                    strip_accents(unicodedata.normalize('NFD', f)).lower(),
                                    ascii_flat(f),
                                    alphanum_only(f)
                                ]
                                # Log détaillé pour debug
                                logger.info(f"Comparaison pour {path_from_json} :")
                                logger.info(f"  target_names = {target_names}")
                                logger.info(f"  f = {f}")
                                logger.info(f"  f_forms = {f_forms}")
                                for t in target_names:
                                    for ff in f_forms:
                                        if t == ff:
                                            logger.info(f"  MATCH: t='{t}' == ff='{ff}'")
                                # Fuzzy match (Levenshtein) sur la forme alphanumérique only
                                fuzzy_match = False
                                t_alpha = alphanum_only(os.path.basename(path_from_json))
                                f_alpha = alphanum_only(f)
                                lev = levenshtein(t_alpha, f_alpha)
                                if lev <= 2 and min(len(t_alpha), len(f_alpha)) > 0:
                                    logger.info(f"  FUZZY MATCH (levenshtein={lev}): t_alpha='{t_alpha}' vs f_alpha='{f_alpha}'")
                                    fuzzy_match = True
                                if any(t == ff for t in target_names for ff in f_forms) or fuzzy_match:
                                    actual_pdf_path = os.path.join(os.path.dirname(actual_pdf_path), f)
                                    logger.info(f"Correspondance fuzzy avancée trouvée pour {path_from_json} : {actual_pdf_path}")
                                    found = True
                                    break
                            if not found:
                                logger.warning(f"PDF non trouvé au chemin résolu : {actual_pdf_path} (chemin original: {path_from_json}, base: {pdf_base_dir})")
                                continue # Passer au prochain attachment si le PDF n'est pas trouvé

                        logger.info(f"Traitement du PDF : {actual_pdf_path}")
                        try:
                            ocr_payload = extract_text_with_ocr(
                                actual_pdf_path,
                                return_details=True,
                            )
                        except OCRExtractionError as ocr_error:
                            logger.error(
                                "Échec OCR pour %s (%s): %s",
                                actual_pdf_path,
                                path_from_json,
                                ocr_error,
                            )
                            continue

                        records.append({
                            **metadata,
                            "filename": os.path.basename(path_from_json), # Conserve le nom de fichier original du JSON
                            "path": actual_pdf_path, # Stocke le chemin résolu et existant
                            "attachment_title": attachment.get("title", ""),
                            "texteocr": ocr_payload.text,
                            "texteocr_provider": ocr_payload.provider,
                        })
            except Exception as item_error:
                logger.error(f"Error processing item: {item_error}")
                continue
                
    except Exception as e:
        logger.error(f"Failed to load Zotero JSON: {e}")
    
    return pd.DataFrame(records)

def extract_pdf_metadata_to_dataframe(pdf_directory: str) -> pd.DataFrame:
    """
    Extrait métadonnées + texte OCR des PDF d'un répertoire.
    
    Args:
        pdf_directory: Chemin du répertoire contenant les PDF
        
    Returns:
        DataFrame pandas avec métadonnées et texte extrait
    """
    if not os.path.exists(pdf_directory):
        logger.error(f"Directory not found: {pdf_directory}")
        return pd.DataFrame()
    
    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_directory}")
        return pd.DataFrame()
        
    records = []
    for filename in tqdm(pdf_files, desc="Processing PDF files"):
        full_path = os.path.join(pdf_directory, filename)
        try:
            with fitz.open(full_path) as doc:
                try:
                    ocr_payload = extract_text_with_ocr(
                        full_path,
                        return_details=True,
                    )
                except OCRExtractionError as ocr_error:
                    logger.error(
                        "Échec OCR pour %s: %s",
                        full_path,
                        ocr_error,
                    )
                    continue

                records.append({
                    "type": "article",
                    "authors": doc.metadata.get('author', ''),
                    "title": doc.metadata.get('title', ''),
                    "date": format_pdf_date(doc.metadata.get('creationDate', '')),
                    "url": "",
                    "doi": extract_doi_from_pdf(doc),
                    "filename": filename,
                    "path": full_path,
                    "attachment_title": os.path.splitext(filename)[0],
                    "texteocr": ocr_payload.text,
                    "texteocr_provider": ocr_payload.provider,
                })
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            continue
            
    return pd.DataFrame(records)

def format_pdf_date(date_string: str) -> str:
    """Formate une date PDF en texte lisible"""
    if not date_string:
        return ""
    try:
        if isinstance(date_string, str) and date_string.startswith('D:'):
            date_str = date_string[2:14]  # Extraire YYYYMMDDHHmm
            if len(date_str) >= 8:
                return f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
        return str(date_string)
    except:
        return str(date_string)

def extract_doi_from_pdf(doc: fitz.Document) -> str:
    """Tente d'extraire un DOI du document PDF"""
    doi_pattern = r'(10\.\d{4,}(?:\.\d+)*\/\S+[^;,.\s])'
    if doc.metadata.get('doi'):
        return doc.metadata.get('doi')
    for page_num in range(min(3, doc.page_count)):
        text = doc[page_num].get_text()
        if match := re.search(doi_pattern, text):
            return match.group(0)
    return ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Zotero JSON and associated PDFs to create a CSV with extracted text.")
    parser.add_argument("--json", required=True, help="Path to the Zotero JSON file.")
    parser.add_argument("--dir", required=True, help="Base directory for resolving relative PDF paths from the JSON.")
    parser.add_argument("--output", required=True, help="Path to save the output CSV file.")
    
    args = parser.parse_args()

    logger.info(f"Starting Zotero data processing for JSON: {args.json} with PDF base directory: {args.dir}")

    # Charger les données Zotero et extraire le texte des PDF
    df_zotero = load_zotero_to_dataframe(args.json, args.dir)

    if not df_zotero.empty:
        # S'assurer que le répertoire de sortie existe
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Created output directory: {output_dir}")
            except Exception as e:
                logger.error(f"Failed to create output directory {output_dir}: {e}")
                print(f"Error creating output directory {output_dir}: {e}")
                # Quitter si le répertoire ne peut pas être créé, car la sauvegarde échouera
                exit()

        # Sauvegarder le DataFrame en CSV
        try:
            df_zotero.to_csv(args.output, index=False, encoding='utf-8-sig', escapechar='\\')
            logger.info(f"DataFrame successfully saved to {args.output}")
            print(f"Output CSV saved to: {args.output}")
        except Exception as e:
            logger.error(f"Failed to save DataFrame to CSV: {e}")
            print(f"Error saving CSV: {e}")
    else:
        logger.warning("No data processed from Zotero JSON. Output CSV will not be created.")
        print("No data processed. Output CSV not created.")
