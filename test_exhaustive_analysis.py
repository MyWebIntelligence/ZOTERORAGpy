#!/usr/bin/env python3
"""
Test script for exhaustive academic analysis system
"""

import sys
import os
import pandas as pd
from dotenv import load_dotenv

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Load environment variables
load_dotenv()

# Import the note generator
from app.utils import llm_note_generator

def test_exhaustive_analysis():
    """Test the exhaustive analysis system with real data"""

    print("=" * 80)
    print("TEST DU SYSTÈME D'ANALYSE ACADÉMIQUE EXHAUSTIVE")
    print("=" * 80)
    print()

    # Load test data
    session_path = "/Users/amarlakel/Google Drive/____ProjetRecherche/__RAG/ragpy/uploads/4495fa4e_TodoBak"
    csv_path = os.path.join(session_path, "output.csv")

    print(f"📁 Chargement des données de test...")
    print(f"   Session: {session_path}")
    print()

    # Read CSV
    df = pd.read_csv(csv_path)

    # Get first article
    first_article = df.iloc[0]

    print("📄 Article sélectionné pour le test:")
    print(f"   Titre: {first_article['title'][:80]}...")
    print(f"   Auteur: {first_article['authors']}")
    print(f"   Date: {first_article['date']}")
    print(f"   Type: {first_article['type']}")
    print()

    # Prepare metadata
    metadata = {
        "title": first_article['title'],
        "authors": first_article['authors'],
        "date": first_article['date'],
        "doi": first_article.get('doi', ''),
        "url": first_article.get('url', ''),
        "abstract": "",  # No abstract in this dataset
        "language": "fr",
        "problematique": "Comment la théorie dialogique de Bakhtine peut-elle éclairer l'analyse des discours et des interactions sociales ?"
    }

    # Get text content (limited for this test to avoid huge API calls)
    text_content = first_article['texteocr']
    text_length = len(text_content) if pd.notna(text_content) else 0

    print(f"📝 Texte OCR disponible: {text_length:,} caractères")
    print()

    # For testing, use a smaller excerpt to avoid huge costs
    # In production, the full text would be used
    if text_length > 15000:
        text_excerpt = text_content[:15000]
        print(f"⚠️  Pour ce test, utilisation d'un extrait de 15,000 caractères")
        print(f"   (En production, tout le texte serait utilisé)")
        print()
    else:
        text_excerpt = text_content

    # Check LLM availability
    print("🔧 Vérification de la configuration LLM...")
    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    default_model = os.getenv("OPENROUTER_DEFAULT_MODEL", "gpt-4o-mini")

    if openai_key:
        print(f"   ✅ OpenAI API Key configurée")
    if openrouter_key:
        print(f"   ✅ OpenRouter API Key configurée")
    print(f"   📌 Modèle par défaut: {default_model}")
    print()

    if not (openai_key or openrouter_key):
        print("❌ ERREUR: Aucune API key configurée!")
        print("   Vérifiez votre fichier .env")
        return

    # Test the generation
    print("🚀 Génération de la note exhaustive...")
    print(f"   Modèle: {default_model}")
    print(f"   Problématique personnalisée: Oui")
    print(f"   Texte complet: Oui (extrait pour test)")
    print()
    print("   ⏳ Cela peut prendre 30-90 secondes...")
    print()

    try:
        sentinel, note_html = llm_note_generator.build_note_html(
            metadata=metadata,
            text_content=text_excerpt,
            model=None,  # Use default model
            use_llm=True
        )

        print("=" * 80)
        print("✅ GÉNÉRATION RÉUSSIE!")
        print("=" * 80)
        print()

        # Analyze the result
        note_length = len(note_html)
        note_words = len(note_html.split())

        print(f"📊 STATISTIQUES:")
        print(f"   Longueur totale: {note_length:,} caractères")
        print(f"   Nombre de mots estimé: {note_words:,} mots")
        print(f"   Sentinel: {sentinel}")
        print()

        # Check for expected HTML elements
        has_h2 = "<h2>" in note_html
        has_h3 = "<h3>" in note_html
        has_table = "<table>" in note_html
        has_problematique = "Problématique" in note_html or "problématique" in note_html
        has_methodologie = "Méthodologie" in note_html or "méthodologie" in note_html
        has_evaluation = "évaluation" in note_html or "Évaluation" in note_html

        print(f"📋 ÉLÉMENTS STRUCTURELS:")
        print(f"   {'✅' if has_h2 else '❌'} Titres de section (<h2>)")
        print(f"   {'✅' if has_h3 else '❌'} Sous-sections (<h3>)")
        print(f"   {'✅' if has_table else '❌'} Tableaux (<table>)")
        print(f"   {'✅' if has_problematique else '❌'} Section Problématique")
        print(f"   {'✅' if has_methodologie else '❌'} Section Méthodologie")
        print(f"   {'✅' if has_evaluation else '❌'} Section Évaluation")
        print()

        # Quality check
        print("🎯 ÉVALUATION DE LA QUALITÉ:")
        if note_words < 500:
            print(f"   ⚠️  Note très courte ({note_words} mots) - Attendu: 3000-8000 mots")
            print(f"       Le modèle n'a peut-être pas suivi les instructions exhaustives")
        elif note_words < 2000:
            print(f"   ⚠️  Note courte ({note_words} mots) - Attendu: 3000-8000 mots")
            print(f"       Peut être normal pour un extrait de 15K caractères")
        elif note_words < 5000:
            print(f"   ✅ Longueur moyenne ({note_words} mots) - Conforme pour extrait")
        else:
            print(f"   ✅ Analyse exhaustive ({note_words} mots) - Excellent!")
        print()

        # Show preview
        print("📄 APERÇU DU DÉBUT DE LA NOTE:")
        print("-" * 80)
        preview = note_html[:1500]
        print(preview)
        if len(note_html) > 1500:
            print("\n[... suite tronquée ...]")
        print("-" * 80)
        print()

        # Save to file for inspection
        output_path = "/Users/amarlakel/Google Drive/____ProjetRecherche/__RAG/ragpy/test_note_exhaustive.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(note_html)

        print(f"💾 Note complète sauvegardée dans:")
        print(f"   {output_path}")
        print()
        print("   Vous pouvez l'ouvrir dans un navigateur pour voir le rendu complet.")
        print()

        print("=" * 80)
        print("🎉 TEST TERMINÉ AVEC SUCCÈS!")
        print("=" * 80)

    except Exception as e:
        print("=" * 80)
        print("❌ ERREUR LORS DE LA GÉNÉRATION")
        print("=" * 80)
        print()
        print(f"Erreur: {e}")
        print()
        import traceback
        traceback.print_exc()
        print()
        print("💡 SUGGESTIONS:")
        print("   - Vérifiez que les API keys sont valides")
        print("   - Vérifiez que vous avez des crédits sur votre compte OpenAI/OpenRouter")
        print("   - Consultez les logs de l'application")

if __name__ == "__main__":
    test_exhaustive_analysis()
