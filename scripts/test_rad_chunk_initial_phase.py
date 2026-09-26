"""Phase ``initial`` de ``rad_chunk.py`` (découpage + recodage) sur un CSV de référence.

Le CSV est ``tests/fixtures/test_documents.csv``, converti au format du pipeline
par le module d'ingestion CSV (mêmes colonnes que la sortie de ``rad_dataframe.py``).
Le client OpenAI du module est remplacé par un faux client (aucun appel réseau)
et la sortie JSON est écrite dans un répertoire temporaire.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAGPY_DIR = os.path.dirname(SCRIPT_DIR)
FIXTURE_CSV = os.path.join(RAGPY_DIR, "tests", "fixtures", "test_documents.csv")

# rad_chunk est importé comme module de premier niveau (scripts/ sur sys.path),
# sinon via le paquet scripts ; les patchs visent l'objet module importé.
try:
    import rad_chunk
except ImportError:
    if RAGPY_DIR not in sys.path:
        sys.path.insert(0, RAGPY_DIR)
    from scripts import rad_chunk

if RAGPY_DIR not in sys.path:
    sys.path.insert(0, RAGPY_DIR)
from ingestion import ingest_csv_to_dataframe

# Configuration hermétique : cache de recodage, durcissement et dédup coupés,
# quelle que soit la configuration de l'hôte.
HERMETIC_ENV = {
    "RECODE_CACHE_ENABLED": "0",
    "RECODE_HARDEN_ENABLED": "0",
    "RECODE_EMBED_CACHE_ENABLED": "0",
    "DEDUP_ENABLED": "0",
}

RECODE_MARKER = "Texte à recoder :\n"
RECODE_END = "\n\nTexte recodé :"


def _load_documents():
    """DataFrame du pipeline construit depuis le CSV de référence (10 documents)."""
    return ingest_csv_to_dataframe(FIXTURE_CSV)


def _fake_completion(*args, **kwargs):
    """Réponse chat simulée : « Recodage simulé de: » + début du chunk reçu."""
    content = "Texte recodé simulé."
    messages = kwargs.get("messages") or []
    if messages:
        user_content = messages[-1]["content"]
        if RECODE_MARKER in user_content:
            start = user_content.find(RECODE_MARKER) + len(RECODE_MARKER)
            end = user_content.find(RECODE_END, start)
            content = f"Recodage simulé de: {user_content[start:end][:50]}..."
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = "stop"
    completion = MagicMock()
    completion.choices = [choice]
    return completion


class TestRadChunkInitialPhase(unittest.TestCase):
    """``process_all_documents`` : découpage, recodage et métadonnées des chunks."""

    def setUp(self):
        """Répertoire de sortie temporaire, environnement hermétique, faux client OpenAI."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.output_json_path = os.path.join(self._tmp.name, "output_chunks.json")

        env_patcher = patch.dict(os.environ, HERMETIC_ENV)
        env_patcher.start()
        self.addCleanup(env_patcher.stop)

        self.mock_client = MagicMock(name="openai_client")
        self.mock_client.chat.completions.create.side_effect = _fake_completion
        client_patcher = patch.object(rad_chunk, "client", self.mock_client)
        client_patcher.start()
        self.addCleanup(client_patcher.stop)

    def _run(self, df):
        """Lance la phase initiale sur ``df`` et relit le JSON produit."""
        self.assertIsNotNone(rad_chunk.TEXT_SPLITTER, "TEXT_SPLITTER n'a pas été initialisé dans rad_chunk.")
        rad_chunk.process_all_documents(df, json_file=self.output_json_path)
        self.assertTrue(os.path.exists(self.output_json_path), "Le fichier JSON de sortie n'a pas été créé.")
        with open(self.output_json_path, "r", encoding="utf-8") as f:
            output_data = json.load(f)
        self.assertIsInstance(output_data, list, "La sortie JSON n'est pas une liste.")
        return output_data

    def test_process_all_documents_with_real_csv(self):
        df = _load_documents()
        self.assertEqual(len(df), 10)
        # Texte d'OCR brut (fournisseur hors RECODE_SKIP_PROVIDERS) : recodage attendu
        df["texteocr_provider"] = "legacy"

        output_data = self._run(df)

        self.assertTrue(len(output_data) > 0, "La liste des chunks en sortie est vide.")
        # Textes courts : un chunk par document
        self.assertEqual(len(output_data), len(df))
        self.assertEqual({c["title"] for c in output_data}, set(df["title"]))
        self.assertEqual(len({c["doc_id"] for c in output_data}), len(df))

        for chunk in output_data:
            for key in ("id", "type", "title", "authors", "date", "filename",
                        "doc_id", "chunk_index", "total_chunks", "text", "texteocr_provider"):
                self.assertIn(key, chunk)
            self.assertNotIn("texteocr", chunk)
            self.assertEqual(chunk["id"], f"{chunk['doc_id']}_{chunk['chunk_index']}")
            self.assertEqual((chunk["chunk_index"], chunk["total_chunks"]), (1, 1))
            self.assertEqual(chunk["texteocr_provider"], "legacy")
            self.assertTrue(chunk["text"].startswith("Recodage simulé de:"),
                            f"Le texte du chunk ne correspond pas au mock: {chunk['text'][:100]}...")
            # Dédup OFF : aucun champ de dédup ni de statut de recodage
            self.assertNotIn("content_hash", chunk)
            self.assertNotIn("recode_status", chunk)

        # Un appel de recodage par chunk, avec le modèle par défaut
        create = self.mock_client.chat.completions.create
        self.assertEqual(create.call_count, len(output_data))
        self.assertEqual({c.kwargs["model"] for c in create.call_args_list}, {"gpt-4o-mini"})

    def test_csv_provider_skips_recoding(self):
        df = _load_documents()
        self.assertEqual(set(df["texteocr_provider"]), {"csv"})

        output_data = self._run(df)

        # Texte déjà propre (CSV) : chunks utilisés tels quels, aucun appel LLM
        self.mock_client.chat.completions.create.assert_not_called()
        self.assertEqual(len(output_data), len(df))
        texts_by_title = dict(zip(df["title"], df["texteocr"]))
        for chunk in output_data:
            self.assertEqual(chunk["text"], texts_by_title[chunk["title"]].strip())
            self.assertEqual(chunk["texteocr_provider"], "csv")


if __name__ == '__main__':
    unittest.main(verbosity=2)
