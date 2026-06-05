# RAGpy - Dockerfile
# Pipeline de traitement de documents pour RAG

FROM python:3.11-slim

# Métadonnées
LABEL maintainer="RAGpy"
LABEL description="Pipeline RAG pour documents académiques (PDF, Zotero, CSV)"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Répertoire de travail
WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers de dépendances
COPY scripts/requirements.txt /app/requirements.txt

# Installation des dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    playwright install --with-deps chromium

# OCR LOCAL (Lot 4) — dépendances OPTIONNELLES et lourdes (Docling → torch).
# Désactivé par défaut. Activer : INSTALL_LOCAL_OCR=true dans .env puis
#   docker compose up -d --build
#
# ISOLATION : Docling est installé dans un VENV DÉDIÉ (/opt/ocr-venv), PAS dans
# l'env principal — car Docling exige numpy 2.x / httpx 0.28 qui casseraient
# spacy/thinc (numpy<2) et mistralai/weaviate (httpx<0.28) du pipeline. Le
# subprocess scripts/ocr_local.py tourne avec /opt/ocr-venv/bin/python.
# torch est installé en CPU-only (évite ~5 Go de libs CUDA en device=cpu).
ARG INSTALL_LOCAL_OCR=false
COPY scripts/requirements-ocr-local.txt /app/requirements-ocr-local.txt
# Couche 1 (lourde, mise en cache) : venv dédié + Docling + torch CPU-only.
RUN if [ "$INSTALL_LOCAL_OCR" = "true" ]; then \
        python -m venv /opt/ocr-venv && \
        /opt/ocr-venv/bin/pip install --no-cache-dir --upgrade pip && \
        /opt/ocr-venv/bin/pip install --no-cache-dir \
            torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
        /opt/ocr-venv/bin/pip install --no-cache-dir -r requirements-ocr-local.txt; \
    else \
        echo "OCR local non installé (INSTALL_LOCAL_OCR=false)"; \
    fi
# NB : le moteur OCR par défaut est Tesseract (ci-dessous). Le moteur RapidOCR
# optionnel (LOCAL_OCR_ENGINE_OCR=rapidocr) nécessiterait `onnxruntime` + des
# modèles téléchargés — non installé par défaut car, sur CPU ARM, RapidOCR s'est
# révélé PLUS LENT que Tesseract (benchmark 2026-06-04 : 8,7 vs 5,1 s/page) et
# dépend de modelscope.cn. À installer manuellement seulement sur serveur x86/GPU.
#
# Couche 2 (légère) : libs image + moteur OCR Tesseract FR/EN. Données de langue
# installées via apt → AUCUN téléchargement de modèle au runtime (hors-ligne).
RUN if [ "$INSTALL_LOCAL_OCR" = "true" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            libgl1 libglib2.0-0 \
            tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# Téléchargement du modèle spaCy français
RUN python -m spacy download fr_core_news_md

# Copie du code source
COPY . /app/

# Création des répertoires nécessaires
RUN mkdir -p /app/uploads /app/logs /app/data

# Exposition du port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Variables Uvicorn avec valeurs par défaut
# NOTE: Using 1 worker for SSE streaming reliability
# Multiple workers cause duplicate logs and SSE connection issues
ENV UVICORN_WORKERS=1
ENV UVICORN_TIMEOUT_KEEP_ALIVE=120
ENV UVICORN_LIMIT_CONCURRENCY=100

# Commande de démarrage (shell form pour expansion variables)
# --proxy-headers: Trust X-Forwarded-Proto, X-Forwarded-For from reverse proxy
# --forwarded-allow-ips: Allow all IPs to set forwarded headers (use specific IP in production)
CMD uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers ${UVICORN_WORKERS} \
    --timeout-keep-alive ${UVICORN_TIMEOUT_KEEP_ALIVE} \
    --limit-concurrency ${UVICORN_LIMIT_CONCURRENCY} \
    --backlog 2048 \
    --proxy-headers \
    --forwarded-allow-ips "*"
