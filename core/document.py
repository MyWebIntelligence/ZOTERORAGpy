"""
Module core.document - Classe Document unifiée

Définit la structure de données commune à toutes les sources d'ingestion
(PDF/OCR, CSV, futures sources). Garantit l'uniformité du pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """
    Représentation unifiée d'un document dans le pipeline RAGpy.

    Toutes les sources d'ingestion (PDF/OCR, CSV, etc.) doivent produire
    des instances de cette classe pour garantir la compatibilité avec
    le reste du pipeline (chunking, embeddings, vectorisation).

    Attributs:
        texteocr (str): Contenu textuel du document. Variable pivot unique
                        pour tout le pipeline. Ne peut pas être vide.
        meta (Dict[str, Any]): Métadonnées arbitraires associées au document.
                               Peut contenir n'importe quels champs selon la source.
                               Exemples: title, authors, date, filename, etc.
        source_type (str): Type de source d'ingestion ("pdf", "csv", etc.).
                          Ajouté automatiquement dans meta si absent.

    Exemples:
        >>> # Document issu d'un PDF via OCR
        >>> doc_pdf = Document(
        ...     texteocr="Texte extrait du PDF...",
        ...     meta={
        ...         "title": "Article scientifique",
        ...         "authors": "Smith, J.",
        ...         "date": "2023-05-15",
        ...         "filename": "article.pdf",
        ...         "source_type": "pdf",
        ...         "texteocr_provider": "mistral"
        ...     }
        ... )

        >>> # Document issu d'un CSV
        >>> doc_csv = Document(
        ...     texteocr="Contenu de la colonne 'text'",
        ...     meta={
        ...         "title": "Entrée CSV #42",
        ...         "category": "Support",
        ...         "priority": "High",
        ...         "custom_field": "Valeur quelconque",
        ...         "source_type": "csv",
        ...         "row_index": 42
        ...     }
        ... )
    """

    texteocr: str
    meta: Dict[str, Any] = field(default_factory=dict)
    source_type: Optional[str] = None

    def __post_init__(self):
        """
        Validation et enrichissement automatique après initialisation.

        - Vérifie que texteocr n'est pas vide
        - Vérifie que meta est bien un dictionnaire
        - Ajoute source_type dans meta s'il est fourni
        - Ajoute un timestamp d'ingestion
        """
        self.validate()
        self._enrich_metadata()

    def validate(self):
        """
        Valide la structure du document.

        Raises:
            ValueError: Si texteocr est vide ou meta n'est pas un dict
        """
        if not self.texteocr or not isinstance(self.texteocr, str):
            raise ValueError(
                "Le champ 'texteocr' est obligatoire et doit être une chaîne non vide. "
                f"Reçu: {type(self.texteocr)} = {repr(self.texteocr[:100] if isinstance(self.texteocr, str) else self.texteocr)}"
            )

        if not isinstance(self.meta, dict):
            raise ValueError(
                f"Le champ 'meta' doit être un dictionnaire. Reçu: {type(self.meta)}"
            )

        # Avertir si texteocr est très court (possiblement une erreur)
        if len(self.texteocr.strip()) < 10:
            logger.warning(
                f"Document avec texteocr très court ({len(self.texteocr)} caractères). "
                f"Métadonnées: {self.meta.get('filename', self.meta.get('title', 'N/A'))}"
            )

    def _enrich_metadata(self):
        """
        Enrichit automatiquement les métadonnées avec des champs système.

        Ajoute:
        - source_type: si fourni à l'initialisation
        - ingested_at: timestamp ISO de création du Document
        """
        from datetime import datetime

        # Ajouter source_type dans meta s'il est fourni
        if self.source_type and "source_type" not in self.meta:
            self.meta["source_type"] = self.source_type

        # Ajouter timestamp d'ingestion s'il n'existe pas
        if "ingested_at" not in self.meta:
            self.meta["ingested_at"] = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit le Document en dictionnaire pour sérialisation.

        Returns:
            Dict avec clés 'texteocr' et toutes les métadonnées aplaties
        """
        return {
            "texteocr": self.texteocr,
            **self.meta
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], text_field: str = "texteocr") -> "Document":
        """
        Crée un Document depuis un dictionnaire.

        Utile pour reconstruire des Documents depuis JSON ou CSV row.

        Args:
            data: Dictionnaire contenant au moins le champ texte
            text_field: Nom de la clé contenant le texte (défaut: "texteocr")

        Returns:
            Instance de Document

        Raises:
            KeyError: Si text_field absent du dictionnaire

        Exemples:
            >>> data = {"texteocr": "Mon texte", "title": "Doc", "date": "2023"}
            >>> doc = Document.from_dict(data)

            >>> # Avec un nom de champ personnalisé
            >>> csv_row = {"text": "Contenu", "category": "Info"}
            >>> doc = Document.from_dict(csv_row, text_field="text")
        """
        if text_field not in data:
            raise KeyError(
                f"Le champ '{text_field}' est absent du dictionnaire. "
                f"Clés disponibles: {list(data.keys())}"
            )

        texteocr = data[text_field]
        meta = {k: v for k, v in data.items() if k != text_field}

        return cls(texteocr=texteocr, meta=meta)

    def __repr__(self) -> str:
        """Représentation lisible du Document pour debug."""
        text_preview = self.texteocr[:50] + "..." if len(self.texteocr) > 50 else self.texteocr
        meta_keys = list(self.meta.keys())
        return f"Document(texteocr='{text_preview}', meta_keys={meta_keys})"

    def get_metadata_summary(self) -> str:
        """
        Génère un résumé textuel des métadonnées pour logging.

        Returns:
            Chaîne formatée avec les métadonnées principales
        """
        title = self.meta.get("title", "Sans titre")
        source = self.meta.get("source_type", "inconnu")
        text_len = len(self.texteocr)
        return f"Document '{title}' (source={source}, texte={text_len} chars, meta={len(self.meta)} champs)"
