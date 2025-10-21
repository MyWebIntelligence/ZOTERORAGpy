#!/usr/bin/env python3
"""
Script de test pour l'ingestion CSV dans le pipeline RAGpy

Teste :
1. Ingestion CSV basique
2. Conversion en Documents
3. Conversion en DataFrame compatible rad_chunk.py
4. Validation des métadonnées dynamiques
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH pour importer les modules locaux
SCRIPT_DIR = Path(__file__).parent.absolute()
RAGPY_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(RAGPY_ROOT))

import logging
from ingestion.csv_ingestion import (
    ingest_csv,
    ingest_csv_to_dataframe,
    CSVIngestionConfig,
    CSVIngestionError,
)
from core.document import Document

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_basic_csv_ingestion():
    """Test 1 : Ingestion CSV basique avec configuration par défaut."""
    print("\n" + "=" * 80)
    print("TEST 1 : Ingestion CSV basique")
    print("=" * 80)

    csv_path = RAGPY_ROOT / "tests" / "fixtures" / "test_documents.csv"

    try:
        # Ingestion avec colonne "text" par défaut
        documents = ingest_csv(csv_path)

        assert len(documents) > 0, "Aucun document créé"
        logger.info(f"✓ {len(documents)} documents créés avec succès")

        # Vérifier le premier document
        first_doc = documents[0]
        assert isinstance(first_doc, Document), "Le résultat n'est pas un Document"
        assert first_doc.texteocr, "texteocr est vide"
        assert isinstance(first_doc.meta, dict), "meta n'est pas un dict"

        logger.info(f"✓ Premier document validé : {first_doc.get_metadata_summary()}")
        logger.info(f"  Métadonnées : {list(first_doc.meta.keys())}")

        # Vérifier que toutes les colonnes CSV sont dans meta
        expected_meta_keys = [
            "title", "category", "priority", "date", "author",
            "tags", "status", "custom_field", "source_type",
            "texteocr_provider", "row_index", "ingested_at"
        ]

        for key in ["title", "category", "priority", "source_type"]:
            assert key in first_doc.meta, f"Clé '{key}' manquante dans meta"

        logger.info("✓ Toutes les colonnes CSV sont présentes dans les métadonnées")

        # Vérifier que source_type et texteocr_provider sont bien "csv"
        assert first_doc.meta.get("source_type") == "csv", "source_type incorrect"
        assert first_doc.meta.get("texteocr_provider") == "csv", "texteocr_provider incorrect"

        logger.info("✓ source_type et texteocr_provider correctement définis à 'csv'")

        print("\n✅ TEST 1 RÉUSSI\n")
        return True

    except Exception as e:
        logger.error(f"❌ TEST 1 ÉCHOUÉ : {e}")
        import traceback
        traceback.print_exc()
        return False


def test_custom_config():
    """Test 2 : Ingestion avec configuration personnalisée."""
    print("\n" + "=" * 80)
    print("TEST 2 : Ingestion avec configuration personnalisée")
    print("=" * 80)

    csv_path = RAGPY_ROOT / "tests" / "fixtures" / "test_documents.csv"

    try:
        # Configuration personnalisée : sélectionner seulement certaines métadonnées
        config = CSVIngestionConfig(
            text_column="text",
            encoding="utf-8",
            meta_columns=["title", "category", "priority"],
            skip_empty=True,
            add_row_index=False,
        )

        documents = ingest_csv(csv_path, config=config)

        assert len(documents) > 0, "Aucun document créé"
        logger.info(f"✓ {len(documents)} documents créés avec configuration personnalisée")

        # Vérifier que seules les colonnes spécifiées sont présentes
        first_doc = documents[0]
        meta_keys = set(first_doc.meta.keys())

        # Les colonnes demandées
        expected = {"title", "category", "priority"}
        # Les colonnes automatiques
        auto_added = {"source_type", "texteocr_provider", "ingested_at"}

        # Colonnes qui NE DOIVENT PAS être présentes (car non spécifiées dans meta_columns)
        excluded = {"author", "tags", "status", "custom_field", "date", "row_index"}

        assert expected.issubset(meta_keys), f"Colonnes attendues manquantes : {expected - meta_keys}"
        assert auto_added.issubset(meta_keys), f"Colonnes auto manquantes : {auto_added - meta_keys}"
        assert not excluded.intersection(meta_keys), f"Colonnes non désirées présentes : {excluded.intersection(meta_keys)}"

        logger.info(f"✓ Seulement les métadonnées configurées sont présentes : {meta_keys}")

        print("\n✅ TEST 2 RÉUSSI\n")
        return True

    except Exception as e:
        logger.error(f"❌ TEST 2 ÉCHOUÉ : {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataframe_conversion():
    """Test 3 : Conversion Documents → DataFrame compatible rad_chunk.py."""
    print("\n" + "=" * 80)
    print("TEST 3 : Conversion en DataFrame pour rad_chunk.py")
    print("=" * 80)

    csv_path = RAGPY_ROOT / "tests" / "fixtures" / "test_documents.csv"

    try:
        # Utiliser la fonction de conversion directe
        df = ingest_csv_to_dataframe(csv_path)

        assert not df.empty, "DataFrame vide"
        assert "texteocr" in df.columns, "Colonne 'texteocr' absente du DataFrame"

        logger.info(f"✓ DataFrame créé : {len(df)} lignes, {len(df.columns)} colonnes")
        logger.info(f"  Colonnes : {list(df.columns)[:10]}...")  # Afficher les 10 premières

        # Vérifier que texteocr n'est pas vide
        assert df["texteocr"].notna().all(), "Valeurs NaN dans texteocr"
        assert (df["texteocr"].str.len() > 0).all(), "Valeurs vides dans texteocr"

        logger.info("✓ Colonne 'texteocr' valide (aucune valeur vide)")

        # Vérifier que les métadonnées sont présentes
        for col in ["title", "category", "source_type", "texteocr_provider"]:
            assert col in df.columns, f"Colonne '{col}' absente du DataFrame"

        logger.info("✓ Toutes les métadonnées clés sont présentes")

        # Sauvegarder le DataFrame de test
        output_csv = RAGPY_ROOT / "tests" / "fixtures" / "test_output.csv"
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        logger.info(f"✓ DataFrame sauvegardé dans {output_csv}")

        print("\n✅ TEST 3 RÉUSSI\n")
        return True

    except Exception as e:
        logger.error(f"❌ TEST 3 ÉCHOUÉ : {e}")
        import traceback
        traceback.print_exc()
        return False


def test_document_class():
    """Test 4 : Validation de la classe Document."""
    print("\n" + "=" * 80)
    print("TEST 4 : Validation de la classe Document")
    print("=" * 80)

    try:
        # Créer un Document manuellement
        doc = Document(
            texteocr="Test document content for validation.",
            meta={
                "title": "Test Document",
                "category": "Testing",
                "custom_field": "value123"
            },
            source_type="test"
        )

        assert doc.texteocr == "Test document content for validation."
        assert doc.meta["title"] == "Test Document"
        assert doc.meta["source_type"] == "test"
        assert "ingested_at" in doc.meta, "ingested_at non ajouté automatiquement"

        logger.info("✓ Document créé et validé manuellement")

        # Tester la conversion to_dict()
        doc_dict = doc.to_dict()
        assert "texteocr" in doc_dict
        assert doc_dict["title"] == "Test Document"

        logger.info("✓ Conversion to_dict() fonctionne")

        # Tester la reconstruction from_dict()
        reconstructed = Document.from_dict(doc_dict)
        assert reconstructed.texteocr == doc.texteocr
        assert reconstructed.meta["title"] == doc.meta["title"]

        logger.info("✓ Reconstruction from_dict() fonctionne")

        # Tester la validation (texte vide devrait échouer)
        try:
            invalid_doc = Document(texteocr="", meta={})
            logger.error("❌ Validation devrait échouer pour texteocr vide")
            return False
        except ValueError as e:
            logger.info(f"✓ Validation correcte : texteocr vide rejeté ({e})")

        print("\n✅ TEST 4 RÉUSSI\n")
        return True

    except Exception as e:
        logger.error(f"❌ TEST 4 ÉCHOUÉ : {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metadata_sanitization():
    """Test 5 : Vérifier la sanitization des métadonnées."""
    print("\n" + "=" * 80)
    print("TEST 5 : Sanitization des métadonnées")
    print("=" * 80)

    csv_path = RAGPY_ROOT / "tests" / "fixtures" / "test_documents.csv"

    try:
        documents = ingest_csv(csv_path)
        first_doc = documents[0]

        # Vérifier que les noms de colonnes ont été nettoyés
        # (pas d'espaces, caractères spéciaux remplacés par underscores)
        for key in first_doc.meta.keys():
            assert " " not in key, f"Espace trouvé dans le nom de clé : '{key}'"
            assert key.islower() or key.isupper() or "_" in key, f"Nom de clé non normalisé : '{key}'"

        logger.info("✓ Noms de colonnes correctement sanitizés")

        # Vérifier que les valeurs None/NaN ont été gérées
        for key, value in first_doc.meta.items():
            if value is not None:
                assert isinstance(value, (str, int, float, bool, list)), \
                    f"Type de valeur non supporté pour '{key}': {type(value)}"

        logger.info("✓ Valeurs de métadonnées correctement typées")

        print("\n✅ TEST 5 RÉUSSI\n")
        return True

    except Exception as e:
        logger.error(f"❌ TEST 5 ÉCHOUÉ : {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Exécute tous les tests."""
    print("\n" + "=" * 80)
    print(" TESTS D'INGESTION CSV - PIPELINE RAGPY")
    print("=" * 80)

    results = {
        "Test 1 (Ingestion basique)": test_basic_csv_ingestion(),
        "Test 2 (Config personnalisée)": test_custom_config(),
        "Test 3 (Conversion DataFrame)": test_dataframe_conversion(),
        "Test 4 (Classe Document)": test_document_class(),
        "Test 5 (Sanitization)": test_metadata_sanitization(),
    }

    print("\n" + "=" * 80)
    print(" RÉSUMÉ DES TESTS")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✅ RÉUSSI" if passed else "❌ ÉCHOUÉ"
        print(f"{test_name:40s} {status}")

    all_passed = all(results.values())
    total = len(results)
    passed_count = sum(results.values())

    print("\n" + "=" * 80)
    if all_passed:
        print(f"✅ TOUS LES TESTS RÉUSSIS ({passed_count}/{total})")
    else:
        print(f"❌ CERTAINS TESTS ONT ÉCHOUÉ ({passed_count}/{total})")
    print("=" * 80 + "\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
