import unittest
import os
import pandas as pd
import json
from unittest.mock import patch, MagicMock, ANY

# Assurez-vous que rad_chunk est importable.
# Si ce script est dans le même dossier que rad_chunk.py et exécuté depuis ce dossier :
try:
    from rad_chunk import process_all_documents, TEXT_SPLITTER, DEFAULT_MAX_WORKERS, DEFAULT_BATCH_SIZE_GPT
except ImportError:
    print("Erreur d'importation de rad_chunk. Assurez-vous que le PYTHONPATH est correct ou exécutez depuis le dossier 'scripts'.")
    # Tentative d'ajustement du sys.path pour exécution depuis la racine du projet
    import sys
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(SCRIPT_DIR) # Devrait être ragpy/
    if PARENT_DIR not in sys.path:
         sys.path.insert(0, PARENT_DIR) # Ajoute ragpy/ au path pour permettre from scripts.rad_chunk
    
    # Réessayer l'importation
    from scripts.rad_chunk import process_all_documents, TEXT_SPLITTER, DEFAULT_MAX_WORKERS, DEFAULT_BATCH_SIZE_GPT


class TestRadChunkInitialPhase(unittest.TestCase):

    def setUp(self):
        # Chemin vers le fichier CSV de test fourni par l'utilisateur
        self.test_csv_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), # ragpy/scripts/
            "..", # ragpy/
            "uploads", "d91919cf_Saussure_Art", "Saussure_Art", "output.csv"
        ))
        
        # Fichier JSON de sortie temporaire pour le test
        self.output_json_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "test_output_chunks.json"
        ))

        # Vérifier si le CSV de test existe
        if not os.path.exists(self.test_csv_path):
            self.fail(f"Le fichier CSV de test n'a pas été trouvé : {self.test_csv_path}")
            
        # Nettoyer le fichier de sortie potentiel d'un test précédent
        if os.path.exists(self.output_json_path):
            os.remove(self.output_json_path)

    def tearDown(self):
        # Nettoyer le fichier JSON de sortie après le test
        if os.path.exists(self.output_json_path):
            try:
                os.remove(self.output_json_path)
                print(f"\nFichier de sortie de test nettoyé : {self.output_json_path}")
            except OSError as e:
                print(f"\nErreur lors du nettoyage du fichier de sortie de test {self.output_json_path}: {e}")


    @patch('rad_chunk.client') # Mocker le client OpenAI dans le module rad_chunk
    def test_process_all_documents_with_real_csv(self, mock_openai_client):
        print(f"\nLancement de test_process_all_documents_with_real_csv...")
        print(f"  Utilisation du CSV : {self.test_csv_path}")
        print(f"  Fichier JSON de sortie attendu : {self.output_json_path}")

        # Configurer le mock pour simuler les réponses de l'API OpenAI gpt_recode_batch
        # Chaque appel à create doit retourner un objet avec une structure spécifique.
        def mock_create_completion(*args, **kwargs):
            mock_completion = MagicMock()
            # Le contenu recodé sera basé sur le message d'entrée pour le rendre un peu dynamique
            # ou simplement un texte fixe.
            input_text_prompt = "Texte recodé simulé."
            if 'messages' in kwargs and kwargs['messages']:
                user_content = kwargs['messages'][-1]['content'] # Dernier message est celui de l'utilisateur
                # Extrait une partie du texte original pour le "recode"
                original_text_marker = "Texte à recoder :\n"
                if original_text_marker in user_content:
                    start_idx = user_content.find(original_text_marker) + len(original_text_marker)
                    end_idx = user_content.find("\n\nTexte recodé :", start_idx)
                    original_sample = user_content[start_idx:end_idx][:50] # Prend les 50 premiers caractères
                    input_text_prompt = f"Recodage simulé de: {original_sample}..."
            
            mock_completion.choices = [MagicMock()]
            mock_completion.choices[0].message = MagicMock()
            mock_completion.choices[0].message.content = input_text_prompt
            return mock_completion

        mock_openai_client.chat.completions.create.side_effect = mock_create_completion
        
        # Charger le DataFrame depuis le CSV de test
        try:
            df = pd.read_csv(self.test_csv_path)
            print(f"  Nombre de documents (lignes) dans le CSV : {len(df)}")
            if df.empty:
                 self.fail(f"Le fichier CSV de test {self.test_csv_path} est vide.")
        except Exception as e:
            self.fail(f"Erreur lors de la lecture du CSV de test {self.test_csv_path}: {e}")

        # S'assurer que TEXT_SPLITTER est initialisé (il devrait l'être au chargement du module rad_chunk)
        self.assertIsNotNone(TEXT_SPLITTER, "TEXT_SPLITTER n'a pas été initialisé dans rad_chunk.")

        # Exécuter la fonction à tester
        print(f"  Appel de process_all_documents...")
        process_all_documents(df, json_file=self.output_json_path)

        # Vérifications
        print(f"  Vérification de l'existence du fichier de sortie : {self.output_json_path}")
        self.assertTrue(os.path.exists(self.output_json_path), "Le fichier JSON de sortie n'a pas été créé.")
        
        print(f"  Lecture et validation du contenu du fichier JSON de sortie...")
        with open(self.output_json_path, 'r', encoding='utf-8') as f:
            output_data = json.load(f)
        
        self.assertIsInstance(output_data, list, "La sortie JSON n'est pas une liste.")
        self.assertTrue(len(output_data) > 0, "La liste des chunks en sortie est vide.")
        
        print(f"  Nombre total de chunks générés : {len(output_data)}")

        # Vérifier la structure du premier chunk (exemple)
        first_chunk = output_data[0]
        self.assertIn("id", first_chunk)
        self.assertIn("type", first_chunk)
        self.assertIn("title", first_chunk)
        self.assertIn("authors", first_chunk)
        self.assertIn("date", first_chunk)
        self.assertIn("filename", first_chunk)
        self.assertIn("doc_id", first_chunk)
        self.assertIn("chunk_index", first_chunk)
        self.assertIn("total_chunks", first_chunk)
        self.assertIn("text", first_chunk)
        self.assertTrue(first_chunk["text"].startswith("Recodage simulé de:"), 
                        f"Le texte du chunk ne correspond pas au mock: {first_chunk['text'][:100]}...")

        # Vérifier que l'API OpenAI a été appelée
        # Le nombre d'appels dépendra du nombre de chunks et de DEFAULT_BATCH_SIZE_GPT
        # Pour chaque document, il y a des lots de chunks.
        # Chaque lot appelle gpt_recode_batch, qui fait des appels API.
        self.assertTrue(mock_openai_client.chat.completions.create.called, "L'API OpenAI (create) n'a pas été appelée.")
        print(f"  Nombre total d'appels simulés à OpenAI API: {mock_openai_client.chat.completions.create.call_count}")
        
        print("  Test test_process_all_documents_with_real_csv terminé avec succès.")

if __name__ == '__main__':
    print("Démarrage des tests unitaires pour la phase initiale de rad_chunk.py...")
    # Nécessite que rad_chunk.py soit dans le même dossier ou que PYTHONPATH soit configuré.
    # Exécuter avec `python test_rad_chunk_initial_phase.py` depuis le dossier `scripts`.
    unittest.main(verbosity=2)
