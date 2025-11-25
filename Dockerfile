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
    pip install --no-cache-dir -r requirements.txt

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
ENV UVICORN_WORKERS=4
ENV UVICORN_TIMEOUT_KEEP_ALIVE=120
ENV UVICORN_LIMIT_CONCURRENCY=100

# Commande de démarrage (shell form pour expansion variables)
CMD uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers ${UVICORN_WORKERS} \
    --timeout-keep-alive ${UVICORN_TIMEOUT_KEEP_ALIVE} \
    --limit-concurrency ${UVICORN_LIMIT_CONCURRENCY} \
    --backlog 2048
