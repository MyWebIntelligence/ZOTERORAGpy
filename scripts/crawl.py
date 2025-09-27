import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import shutil

# PDFkit (optionnel)
try:
    import pdfkit
    WKHTMLTOPDF_PATH = shutil.which("wkhtmltopdf")
    PDFKIT_AVAILABLE = WKHTMLTOPDF_PATH is not None
    if PDFKIT_AVAILABLE:
        config_pdfkit = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)
except ImportError:
    PDFKIT_AVAILABLE = False

# Playwright (fallback)
from playwright.sync_api import sync_playwright

# === CONFIGURATION ===
START_URL = "https://docs.n8n.io/integrations/"
DOMAIN = urlparse(START_URL).netloc
VISITED = set()

PDF_DIR = "pages_pdf"
MD_DIR = "pages_md"
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(MD_DIR, exist_ok=True)

# === UTILS ===
def is_internal_link(url):
    parsed = urlparse(url)
    return parsed.netloc in ["", DOMAIN] and parsed.scheme in ["http", "https"]

def sanitize_filename(url):
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_") or "index"
    return path.split("#")[0]  # ignore fragment

# === PDF CONVERSION ===
def save_pdf(url, filename_base):
    filepath = os.path.join(PDF_DIR, filename_base + ".pdf")

    if PDFKIT_AVAILABLE:
        try:
            pdfkit.from_url(url, filepath, configuration=config_pdfkit)
            print(f"✅ PDF enregistré (pdfkit) : {filepath}")
            return
        except Exception as e:
            print(f"⚠️ Erreur pdfkit, tentative Playwright : {e}")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url, wait_until="networkidle")
            page.pdf(path=filepath, format="A4", print_background=True)
            browser.close()
        print(f"✅ PDF enregistré (Playwright) : {filepath}")
    except Exception as e:
        print(f"❌ PDF échoué pour {url} : {e}")

# === HTML → MARKDOWN BASIQUE ===
def html_to_markdown_with_bs(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text_blocks = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "p", "li", "a", "strong", "em", "code"]):
        text = el.get_text(strip=True)
        if not text:
            continue
        if el.name.startswith("h"):
            level = int(el.name[1])
            text_blocks.append(f"{'#' * level} {text}")
        elif el.name == "a":
            href = el.get("href", "#")
            text_blocks.append(f"[{text}]({href})")
        elif el.name == "strong":
            text_blocks.append(f"**{text}**")
        elif el.name == "em":
            text_blocks.append(f"*{text}*")
        elif el.name == "code":
            text_blocks.append(f"`{text}`")
        else:
            text_blocks.append(text)
    return "\n\n".join(text_blocks)

def save_markdown(html, filename_base):
    filepath = os.path.join(MD_DIR, filename_base + ".md")
    try:
        markdown = html_to_markdown_with_bs(html)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"✅ Markdown enregistré : {filepath}")
    except Exception as e:
        print(f"❌ Markdown échoué : {e}")

# === CRAWLER PRINCIPAL ===
def crawl(url):
    if url in VISITED:
        return
    VISITED.add(url)

    try:
        response = requests.get(url)
        if response.status_code != 200
            print(f"⚠️ Page inaccessible : {url}")
            return

        soup = BeautifulSoup(response.text, "html.parser")
        filename_base = sanitize_filename(url)

        save_pdf(url, filename_base)
        save_markdown(response.text, filename_base)

        for link in soup.find_all("a", href=True):
            full_url = urljoin(url, link["href"])
            if is_internal_link(full_url) and full_url not in VISITED:
                crawl(full_url)

    except Exception as e:
        print(f"❌ Erreur générale pour {url} : {e}")

# === LANCEMENT ===
crawl(START_URL)
