"""
Module ingestion.csv_ingestion - Ingestion de fichiers CSV

Permet d'injecter des fichiers CSV dans le pipeline RAGpy en contournant
l'étape OCR. Mappe une colonne CSV vers la variable pivot 'texteocr' et
conserve toutes les autres colonnes comme métadonnées.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import pandas as pd

# Imports conditionnels pour la détection d'encodage
try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False
    logging.warning(
        "Le package 'chardet' n'est pas installé. La détection automatique "
        "d'encodage ne sera pas disponible. Installez-le via: pip install chardet"
    )

from core.document import Document

logger = logging.getLogger(__name__)


class CSVIngestionError(Exception):
    """Exception levée lors d'erreurs d'ingestion CSV."""
    pass


class CSVIngestionConfig:
    """
    Configuration pour l'ingestion CSV.

    Attributs:
        text_column (str): Nom de la colonne contenant le texte (défaut: "text")
        encoding (str): Encodage du fichier ("auto", "utf-8", "latin-1", etc.)
        delimiter (str): Séparateur CSV (défaut: ",")
        meta_columns (List[str]): Colonnes à inclure dans meta. Si vide, toutes sauf text_column
        skip_empty (bool): Ignorer les lignes avec texte vide (défaut: True)
        add_row_index (bool): Ajouter 'row_index' dans meta (défaut: True)
    """

    def __init__(
        self,
        text_column: str = "text",
        encoding: str = "auto",
        delimiter: str = ",",
        meta_columns: Optional[List[str]] = None,
        skip_empty: bool = True,
        add_row_index: bool = True,
    ):
        self.text_column = text_column
        self.encoding = encoding
        self.delimiter = delimiter
        self.meta_columns = meta_columns or []
        self.skip_empty = skip_empty
        self.add_row_index = add_row_index

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CSVIngestionConfig":
        """Crée une config depuis un dictionnaire (ex: chargé depuis YAML)."""
        return cls(
            text_column=config_dict.get("text_column", "text"),
            encoding=config_dict.get("encoding", "auto"),
            delimiter=config_dict.get("delimiter", ","),
            meta_columns=config_dict.get("meta_columns", []),
            skip_empty=config_dict.get("skip_empty", True),
            add_row_index=config_dict.get("add_row_index", True),
        )


def detect_encoding(file_path: str, sample_size: int = 10000) -> str:
    """
    Détecte l'encodage d'un fichier CSV.

    Args:
        file_path: Chemin du fichier CSV
        sample_size: Nombre d'octets à lire pour la détection (défaut: 10000)

    Returns:
        Nom de l'encodage détecté (ex: "utf-8", "latin-1")

    Raises:
        CSVIngestionError: Si chardet n'est pas disponible
    """
    if not CHARDET_AVAILABLE:
        logger.warning(
            "chardet non disponible, impossible de détecter l'encodage. "
            "Fallback sur UTF-8."
        )
        return "utf-8"

    try:
        with open(file_path, "rb") as f:
            raw_data = f.read(sample_size)
            result = chardet.detect(raw_data)
            encoding = result.get("encoding", "utf-8")
            confidence = result.get("confidence", 0.0)

            logger.info(
                f"Encodage détecté pour {file_path}: {encoding} "
                f"(confiance: {confidence:.2%})"
            )

            # Fallback sur UTF-8 si confiance trop faible
            if confidence < 0.7:
                logger.warning(
                    f"Confiance faible pour l'encodage détecté ({confidence:.2%}). "
                    f"Utilisation de UTF-8 par sécurité."
                )
                return "utf-8"

            return encoding or "utf-8"

    except Exception as e:
        logger.error(f"Erreur lors de la détection d'encodage: {e}")
        return "utf-8"


def sanitize_column_name(col: str) -> str:
    """
    Nettoie un nom de colonne CSV pour le rendre compatible JSON/Pinecone.

    - Remplace les espaces et caractères spéciaux par des underscores
    - Convertit en snake_case
    - Supprime les underscores multiples consécutifs

    Args:
        col: Nom de colonne brut

    Returns:
        Nom de colonne nettoyé

    Exemples:
        >>> sanitize_column_name("Nom du Client")
        'nom_du_client'
        >>> sanitize_column_name("Date (création)")
        'date_creation'
    """
    import re

    # Supprimer les parenthèses et leur contenu
    col = re.sub(r"\([^)]*\)", "", col)

    # Remplacer espaces et caractères spéciaux par underscore
    col = re.sub(r"[^a-zA-Z0-9_]", "_", col)

    # Supprimer underscores multiples
    col = re.sub(r"_+", "_", col)

    # Supprimer underscores en début/fin
    col = col.strip("_")

    # Convertir en minuscules
    col = col.lower()

    return col or "unnamed"


def sanitize_metadata_value(value: Any) -> Any:
    """
    Nettoie une valeur de métadonnée pour compatibilité JSON/Pinecone.

    - Convertit pd.NA, np.nan → None ou chaîne vide
    - Convertit dates → ISO strings
    - Convertit booléens/nombres → types natifs Python
    - Fallback vers str() pour types inconnus

    Args:
        value: Valeur brute depuis le CSV

    Returns:
        Valeur nettoyée (str, int, float, bool, ou None)
    """
    # Gérer les NaN pandas/numpy
    if pd.isna(value):
        return None

    # Gérer les timestamps/dates pandas
    if isinstance(value, (pd.Timestamp, pd.DatetimeTZDtype)):
        return value.isoformat()

    # Gérer les dates Python
    from datetime import datetime, date
    if isinstance(value, (datetime, date)):
        return value.isoformat()

    # Types primitifs OK
    if isinstance(value, (str, int, float, bool)):
        return value

    # Listes/tuples → convertir en listes de strings
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]

    # Fallback: convertir en string
    logger.debug(f"Type non géré pour métadonnée: {type(value)}, conversion en str")
    return str(value)


def csv_row_to_document(
    row: pd.Series,
    text_column: str,
    meta_columns: Optional[List[str]] = None,
    row_index: Optional[int] = None,
) -> Document:
    """
    Convertit une ligne CSV (pandas Series) en Document.

    Args:
        row: Ligne du DataFrame pandas
        text_column: Nom de la colonne contenant le texte
        meta_columns: Colonnes à inclure dans meta (si None, toutes sauf text_column)
        row_index: Index de la ligne (ajouté dans meta si fourni)

    Returns:
        Instance de Document

    Raises:
        KeyError: Si text_column absent de la ligne
        ValueError: Si le texte est vide
    """
    if text_column not in row.index:
        raise KeyError(
            f"Colonne texte '{text_column}' absente de la ligne CSV. "
            f"Colonnes disponibles: {list(row.index)}"
        )

    texteocr = str(row[text_column]).strip()

    if not texteocr:
        raise ValueError(
            f"Texte vide pour la ligne (index={row_index}). "
            f"Contenu: {dict(row)}"
        )

    # Construire les métadonnées
    if meta_columns:
        # Utiliser uniquement les colonnes spécifiées
        meta_dict = {col: row[col] for col in meta_columns if col in row.index}
    else:
        # Utiliser toutes les colonnes sauf text_column
        meta_dict = row.drop(text_column).to_dict()

    # Nettoyer les noms de colonnes
    meta_dict = {sanitize_column_name(k): sanitize_metadata_value(v) for k, v in meta_dict.items()}

    # Ajouter row_index si fourni
    if row_index is not None:
        meta_dict["row_index"] = row_index

    # Ajouter automatiquement source_type et texteocr_provider
    meta_dict["source_type"] = "csv"
    meta_dict["texteocr_provider"] = "csv"

    return Document(texteocr=texteocr, meta=meta_dict, source_type="csv")


def ingest_csv(
    csv_path: Union[str, Path],
    config: Optional[CSVIngestionConfig] = None,
) -> List[Document]:
    """
    Ingère un fichier CSV et retourne une liste de Documents.

    Point d'entrée principal pour l'ingestion CSV dans le pipeline RAGpy.

    Args:
        csv_path: Chemin du fichier CSV ou DataFrame pandas
        config: Configuration d'ingestion (si None, utilise les valeurs par défaut)

    Returns:
        Liste de Documents prêts pour le chunking/embeddings

    Raises:
        CSVIngestionError: Si le fichier n'existe pas, est vide, ou mal formé
        ValueError: Si la colonne texte est absente

    Exemples:
        >>> # Ingestion basique avec colonne "text"
        >>> docs = ingest_csv("data/documents.csv")

        >>> # Ingestion avec configuration personnalisée
        >>> config = CSVIngestionConfig(
        ...     text_column="description",
        ...     encoding="utf-8",
        ...     meta_columns=["title", "category", "priority"]
        ... )
        >>> docs = ingest_csv("data/tickets.csv", config=config)
    """
    config = config or CSVIngestionConfig()

    # Validation du fichier
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise CSVIngestionError(f"Fichier CSV introuvable: {csv_path}")

    if csv_path.stat().st_size == 0:
        raise CSVIngestionError(f"Fichier CSV vide: {csv_path}")

    logger.info(f"Début de l'ingestion CSV: {csv_path}")
    logger.info(f"Configuration initiale: text_column='{config.text_column}', encoding='{config.encoding}'")

    # Détection d'encodage si nécessaire
    encoding = config.encoding
    if encoding == "auto":
        encoding = detect_encoding(str(csv_path))
        logger.info(f"Encodage détecté: {encoding}")

    # Chargement du CSV
    try:
        df = pd.read_csv(
            csv_path,
            encoding=encoding,
            delimiter=config.delimiter,
            on_bad_lines="warn",  # Pandas 1.3+
        )
        logger.info(f"CSV chargé: {len(df)} lignes, {len(df.columns)} colonnes")
        logger.info(f"Colonnes détectées: {list(df.columns)}")

    except UnicodeDecodeError as e:
        # Tentative de fallback sur utf-8
        logger.warning(f"Erreur d'encodage avec '{encoding}': {e}")
        logger.info("Nouvelle tentative avec UTF-8...")
        try:
            df = pd.read_csv(csv_path, encoding="utf-8", delimiter=config.delimiter)
            logger.info("Succès avec UTF-8")
        except Exception as e2:
            raise CSVIngestionError(
                f"Impossible de lire le CSV avec les encodages testés: {e2}"
            )

    except Exception as e:
        raise CSVIngestionError(f"Erreur lors de la lecture du CSV: {e}")

    # Auto-détection de la colonne texte : priorité à 'texteocr' si présente
    # Cela permet d'accepter des CSV avec contenu déjà préparé
    TEXT_COLUMN_CANDIDATES = ["texteocr", "text", "content", "body", "description"]

    if config.text_column not in df.columns:
        # Chercher une colonne candidate (case-insensitive)
        df_columns_lower = {col.lower(): col for col in df.columns}
        detected_column = None

        for candidate in TEXT_COLUMN_CANDIDATES:
            if candidate in df_columns_lower:
                detected_column = df_columns_lower[candidate]
                break

        if detected_column:
            logger.info(
                f"Colonne texte '{config.text_column}' non trouvée, "
                f"auto-détection: utilisation de '{detected_column}'"
            )
            config.text_column = detected_column
        else:
            raise ValueError(
                f"Colonne texte '{config.text_column}' absente du CSV et aucune "
                f"colonne candidate trouvée ({TEXT_COLUMN_CANDIDATES}). "
                f"Colonnes disponibles: {list(df.columns)}"
            )
    else:
        logger.info(f"Colonne texte trouvée: '{config.text_column}'")

    # Nettoyage des noms de colonnes
    original_columns = list(df.columns)
    df.columns = [sanitize_column_name(col) for col in df.columns]
    logger.info(f"Noms de colonnes nettoyés: {dict(zip(original_columns, df.columns))}")

    # Mise à jour du text_column si nécessaire
    config.text_column = sanitize_column_name(config.text_column)

    # Conversion en Documents
    documents = []
    skipped_count = 0

    for idx, row in df.iterrows():
        try:
            doc = csv_row_to_document(
                row,
                text_column=config.text_column,
                meta_columns=config.meta_columns,
                row_index=idx if config.add_row_index else None,
            )
            documents.append(doc)

        except ValueError as e:
            # Texte vide
            if config.skip_empty:
                skipped_count += 1
                logger.debug(f"Ligne {idx} ignorée (texte vide): {e}")
            else:
                logger.error(f"Ligne {idx} échouée: {e}")
                raise

        except Exception as e:
            logger.error(f"Erreur lors du traitement de la ligne {idx}: {e}")
            raise CSVIngestionError(f"Échec ligne {idx}: {e}")

    logger.info(
        f"Ingestion CSV terminée: {len(documents)} documents créés, "
        f"{skipped_count} lignes ignorées (texte vide)"
    )

    if not documents:
        logger.warning(
            f"Aucun document créé depuis {csv_path}. Vérifiez que la colonne "
            f"'{config.text_column}' contient des données."
        )

    # Afficher un échantillon de métadonnées pour debug
    if documents:
        sample_doc = documents[0]
        logger.info(f"Échantillon de document: {sample_doc.get_metadata_summary()}")
        logger.debug(f"Métadonnées complètes: {sample_doc.meta}")

    return documents


def ingest_csv_to_dataframe(
    csv_path: Union[str, Path],
    config: Optional[CSVIngestionConfig] = None,
) -> pd.DataFrame:
    """
    Ingère un CSV et retourne un DataFrame compatible avec rad_chunk.py.

    Utile pour intégration avec le pipeline existant qui attend un CSV
    avec colonne 'texteocr'.

    Args:
        csv_path: Chemin du fichier CSV
        config: Configuration d'ingestion

    Returns:
        DataFrame avec colonnes 'texteocr' + métadonnées aplaties

    Exemples:
        >>> df = ingest_csv_to_dataframe("data/docs.csv")
        >>> # Peut ensuite être sauvegardé en CSV pour rad_chunk.py
        >>> df.to_csv("output/processed.csv", index=False, encoding="utf-8-sig")
    """
    documents = ingest_csv(csv_path, config=config)

    # Convertir les Documents en dictionnaires
    records = [doc.to_dict() for doc in documents]

    df = pd.DataFrame(records)

    logger.info(f"DataFrame créé: {len(df)} lignes, colonnes: {list(df.columns)}")

    return df
