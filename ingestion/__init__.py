"""
Package ingestion - Modules d'ingestion de données

Contient les pipelines d'ingestion pour différentes sources de données
(CSV, PDF/OCR, futures sources).
"""

from .csv_ingestion import (
    ingest_csv,
    ingest_csv_to_dataframe,
    CSVIngestionConfig,
    CSVIngestionError,
)

__all__ = [
    "ingest_csv",
    "ingest_csv_to_dataframe",
    "CSVIngestionConfig",
    "CSVIngestionError",
]
