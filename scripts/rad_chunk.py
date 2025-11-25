import os
import json
import random
import time
import threading
import pandas as pd
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
import spacy
from collections import Counter
from dotenv import load_dotenv, find_dotenv, set_key
import subprocess # Added for spacy download subprocess
import logging

# Attempt to import RecursiveCharacterTextSplitter from langchain_text_splitters
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    print("Warning: langchain_text_splitters not found. TEXT_SPLITTER will not be initialized.")
    print("Please install it via 'pip install langchain-text-splitters'")
    RecursiveCharacterTextSplitter = None

# ----------------------------------------------------------------------
# Helper function to manage .env file
# ----------------------------------------------------------------------
def update_env_file(key, value):
    """Updates or adds a key-value pair to the .env file."""
    dotenv_path = find_dotenv()  # Try to find existing .env
    if not dotenv_path:
        # If .env is not found by find_dotenv (e.g. in parent dirs),
        # default to creating/using .env in the current working directory.
        dotenv_path = os.path.join(os.getcwd(), ".env")
        print(f".env file not found by find_dotenv(), will create/use: {dotenv_path}")

    # set_key will create the file if it doesn't exist.
    success = set_key(dotenv_path, key, value, quote_mode="always")
    
    if success:
        print(f"Successfully saved/updated {key} in {dotenv_path}.")
        # Reload dotenv so subsequent os.getenv calls in the same script run pick up the new/changed value
        load_dotenv(dotenv_path=dotenv_path, override=True)
    else:
        # This case should ideally not happen if file permissions are okay.
        print(f"Warning: python-dotenv's set_key function indicated an issue saving/updating {key} in {dotenv_path}.")
        print("Please check file permissions and ensure the path is correct.")

# ----------------------------------------------------------------------
# Global Configuration and Initializations
# ----------------------------------------------------------------------
SAVE_LOCK = threading.Lock()

# Load environment variables from .env file
load_dotenv()

# OpenAI API Client Initialization
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("OPENAI_API_KEY not found in environment variables or .env file.")
    user_api_key = input("Please enter your OpenAI API Key: ").strip()
    if user_api_key:
        OPENAI_API_KEY = user_api_key
        save_env = input("Do you want to save this API key to a .env file for future use? (yes/no): ").strip().lower()
        if save_env == 'yes':
            update_env_file("OPENAI_API_KEY", user_api_key)
    else:
        raise ValueError("OPENAI_API_KEY is required to proceed.")

client = OpenAI(api_key=OPENAI_API_KEY)

# OpenRouter Client Initialization (optional, for cost-effective alternatives)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
openrouter_client = None
if OPENROUTER_API_KEY:
    openrouter_client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )
    print("OpenRouter client initialized successfully.")
else:
    print("OpenRouter API key not found. Will use OpenAI for all LLM calls.")

# Text Splitter Initialization
if RecursiveCharacterTextSplitter:
    TEXT_SPLITTER = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="text-embedding-3-large",  # This model is for token counting for the splitter
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "#", "##", "\n", " ", ""] # Ajout des nouveaux séparateurs avec priorité
    )
else:
    TEXT_SPLITTER = None # Fallback or error if not available

# spaCy Model Initialization
try:
    nlp = spacy.load("fr_core_news_md")
except OSError:
    print("Le modèle spaCy 'fr_core_news_md' n'est pas trouvé. Tentative de téléchargement...")
    try:
        subprocess.run(["python", "-m", "spacy", "download", "fr_core_news_md"], check=True)
        nlp = spacy.load("fr_core_news_md")
        print("Modèle spaCy 'fr_core_news_md' téléchargé et chargé avec succès.")
    except Exception as e:
        print(f"Erreur lors du téléchargement du modèle spaCy : {e}")
        print("Veuillez installer le modèle manuellement : python -m spacy download fr_core_news_md")
        nlp = None # Fallback or error

# ----------------------------------------------------------------------
# Environment variable helper with validation
# ----------------------------------------------------------------------
def get_env_int(key: str, default: int, min_val: int = 1) -> int:
    """Get integer from environment with validation and fallback."""
    try:
        value = int(os.getenv(key, default))
        return max(min_val, value)
    except (ValueError, TypeError):
        logging.warning(f"Invalid {key}, using default {default}")
        return default

# Default constants from master code (configurable via .env)
DEFAULT_JSON_FILE_CHUNKS = "df_chunks.json"
_cpu_count = os.cpu_count() or 2
DEFAULT_MAX_WORKERS = get_env_int('DEFAULT_MAX_WORKERS', max(1, _cpu_count - 1))
DEFAULT_BATCH_SIZE_GPT = get_env_int('DEFAULT_BATCH_SIZE_GPT', 5)
DEFAULT_EMBEDDING_BATCH_SIZE = get_env_int('DEFAULT_EMBEDDING_BATCH_SIZE', 32)
DEFAULT_DOC_WORKERS = get_env_int('DEFAULT_DOC_WORKERS', 3)
DEFAULT_INPUT_JSON_WITH_EMBEDDINGS = "df_chunks_with_embeddings.json"
DEFAULT_OUTPUT_JSON_SPARSE = "df_chunks_with_embeddings_sparse.json"

# ----------------------------------------------------------------------
# PART 1: Découpage en CHUNKs assisté par gpt_recode
# ----------------------------------------------------------------------

def gpt_recode_batch(chunks, instructions, model="gpt-4o-mini", temperature=0.3, max_tokens=8000):
    """
    Recoder un lot de textes selon des instructions précises en parallèle,
    puis retenter séquentiellement en cas d'erreur.

    Args:
        model: Nom du modèle (ex: "gpt-4o-mini" pour OpenAI, "openai/gemini-2.5-flash" pour OpenRouter)
               Si le modèle contient "/" → utilise OpenRouter, sinon OpenAI
    """
    # Auto-detect which client to use based on model format
    use_openrouter = "/" in model  # OpenRouter models have format "provider/model"
    active_client = openrouter_client if (use_openrouter and openrouter_client) else client

    if use_openrouter and not openrouter_client:
        print(f"Warning: OpenRouter model '{model}' requested but OpenRouter client not initialized.")
        print("Falling back to OpenAI gpt-4o-mini")
        model = "gpt-4o-mini"
        active_client = client

    print(f"Using {'OpenRouter' if (use_openrouter and openrouter_client) else 'OpenAI'} with model: {model}")

    messages_list = []
    for chunk in chunks:
        prompt = (
            f"Instructions : {instructions}\n\n"
            f"Texte à recoder :\n{chunk}\n\n"
            "Texte recodé :"
        )
        messages_list.append([
            {"role": "system", "content": "Assistant spécialisé en recodage de textes académiques."},
            {"role": "user",   "content": prompt}
        ])

    recoded = [None] * len(chunks)
    # Use DEFAULT_MAX_WORKERS for the ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                lambda msgs, mdl=model: active_client.chat.completions.create(
                    model=mdl,
                    messages=msgs,
                    temperature=temperature,
                    max_tokens=max_tokens
                ),
                msgs
            ): idx
            for idx, msgs in enumerate(messages_list)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                resp = future.result()
                recoded[idx] = resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"Erreur chunk #{idx+1} (1ʳᵉ passe) : {e}")

    # Deuxième passe séquentielle pour les None
    for i, text in enumerate(recoded):
        if text is None: # Check if recoding failed in the first pass
            print(f"Tentative de 2ᵉ passe pour le chunk #{i+1}")
            try:
                time.sleep(2) # Wait before retrying
                resp = active_client.chat.completions.create(
                    model=model,
                    messages=messages_list[i],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                recoded[i] = resp.choices[0].message.content.strip()
                print(f"Chunk #{i+1} recodé avec succès en 2ᵉ passe.")
            except Exception as e:
                print(f"Échec chunk #{i+1} après 2ᵉ passe : {e}")
                recoded[i] = chunks[i] # Fallback to original chunk if 2nd pass fails

    return recoded

def save_raw_chunks_to_json_incrementally(chunks_to_add, json_file):
    """
    Sauvegarde les nouveaux chunks dans `json_file` de manière incrémentale et thread-safe.
    """
    with SAVE_LOCK:
        existing_chunks = []
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    existing_chunks = json.load(f)
            except json.JSONDecodeError:
                print(f"Fichier JSON '{json_file}' corrompu ou vide. On repart d'une liste vide.")
                existing_chunks = []
        
        merged_chunks = existing_chunks + chunks_to_add
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(merged_chunks, f, ensure_ascii=False, indent=2)

def process_document_chunks(row_data, json_file=DEFAULT_JSON_FILE_CHUNKS, model="gpt-4o-mini"):
    """
    Traite un document (représenté par row_data, ex: une ligne de DataFrame).
    1. Extraction et nettoyage du texte (implicite par TEXT_SPLITTER)
    2. Découpage en chunks avec TEXT_SPLITTER
    3. Recodage par batch avec gpt_recode_batch
    4. Sauvegarde des chunks avec save_raw_chunks_to_json_incrementally

    Args:
        model: Modèle LLM pour le recodage (ex: "gpt-4o-mini" ou "openai/gemini-2.5-flash")
    """
    if TEXT_SPLITTER is None:
        print("Erreur: TEXT_SPLITTER n'est pas initialisé. Impossible de traiter le document.")
        return []

    text = row_data.get("texteocr", "").strip()
    if not text:
        print(f"Document ignoré (texte vide) : {row_data.get('filename', 'Nom de fichier inconnu')}")
        return []

    provider_raw = row_data.get("texteocr_provider", "")
    if isinstance(provider_raw, float) and pd.isna(provider_raw):
        provider_raw = ""
    provider = str(provider_raw).strip().lower()
    # Skip recodage GPT si OCR Mistral (déjà Markdown) ou source CSV (déjà propre)
    recode_required = provider not in ("mistral", "csv")

    doc_id = str(random.randint(10**11, 10**12 - 1))
    text_chunks = TEXT_SPLITTER.split_text(text)
    
    filename = row_data.get('filename', f'doc_{doc_id}')
    print(f"Traitement de '{filename}': {len(text_chunks)} chunks bruts générés.")

    all_processed_chunks = []
    for start_index in range(0, len(text_chunks), DEFAULT_BATCH_SIZE_GPT):
        batch_to_recode = text_chunks[start_index : start_index + DEFAULT_BATCH_SIZE_GPT]
        total_batches = ((len(text_chunks) - 1) // DEFAULT_BATCH_SIZE_GPT) + 1

        if recode_required:
            print(f"  Lot {start_index // DEFAULT_BATCH_SIZE_GPT + 1} / {total_batches} en recodage...")
            cleaned_batch = gpt_recode_batch(
                batch_to_recode,
                instructions="ce chunk est issu d'un ocr brut qui laisse beaucoup de blocs de texte inutiles comme des titres de pages, des numeros, etc. Nettoie ce chunk pour en faire un texte propre qui commence par une phrase complète et se termine par un point. Supprime le bruit d'OCR et les imperfections en conservant le sens original. Ne echange ni ajoute aucun mot du texte d'origine. C'est une correction et un nettoyage de texte (suppression des erreurs) pas une réécriture",
                model=model,
                temperature=0.3,
                max_tokens=8000
            )
        else:
            if start_index == 0:
                print("  OCR Mistral détecté → recodage GPT sauté (chunks utilisés tels quels).")
            cleaned_batch = batch_to_recode

        for i, cleaned_text in enumerate(cleaned_batch):
            original_chunk_index = start_index + i + 1
            
            # Sanitize metadata values, especially for NaN or other non-JSON-friendly types
            # Convert pandas.NA or numpy.nan to empty strings or None
            # Pinecone metadata values should be string, number, boolean, or list of strings.
            
            def sanitize_metadata_value(value, default=""):
                if pd.isna(value):
                    return default
                # Ensure it's a basic type suitable for JSON and Pinecone metadata
                if isinstance(value, (str, int, float, bool)):
                    return value
                return str(value) # Fallback to string representation

            # Construction dynamique des métadonnées
            # Injecte TOUTES les colonnes du row_data (compatibilité CSV)
            chunk_metadata = {
                "id":           f"{doc_id}_{original_chunk_index}",
                "doc_id":       doc_id,
                "chunk_index":  original_chunk_index,
                "total_chunks": len(text_chunks),
                "text":         cleaned_text,
            }

            # Injecter toutes les métadonnées source (sauf texteocr qui est dans "text")
            for key, value in row_data.items():
                # Exclure les champs déjà gérés ou trop volumineux
                if key not in ("texteocr", "text", "id", "doc_id", "chunk_index", "total_chunks"):
                    chunk_metadata[key] = sanitize_metadata_value(value, "")

            # S'assurer que ocr_provider est présent (backward compatibility)
            if "ocr_provider" not in chunk_metadata and "texteocr_provider" not in chunk_metadata:
                chunk_metadata["ocr_provider"] = sanitize_metadata_value(provider, "")
            all_processed_chunks.append(chunk_metadata)

    if all_processed_chunks:
        save_raw_chunks_to_json_incrementally(all_processed_chunks, json_file)
        print(f"→ {len(all_processed_chunks)} chunks traités et sauvegardés pour le document '{filename}' (doc_id={doc_id}) dans '{json_file}'.")
        # Ajout d'un log pour chaque chunk individuel ajouté (peut être verbeux)
        # for idx, chunk_data in enumerate(all_processed_chunks):
        #     print(f"    Chunk {idx+1}/{len(all_processed_chunks)} (ID: {chunk_data.get('id')}) ajouté au fichier JSON.")
    
    return all_processed_chunks

def process_all_documents(df, json_file=DEFAULT_JSON_FILE_CHUNKS, model="gpt-4o-mini"):
    """
    Lance le traitement de tous les documents d'un DataFrame en parallèle.
    Utilise un nombre limité de workers pour process_document_chunks pour éviter la surcharge API.

    Args:
        model: Modèle LLM pour le recodage (ex: "gpt-4o-mini" ou "openai/gemini-2.5-flash")
    """
    # Max 3 documents processed in parallel for their chunking/recoding stages
    num_doc_workers = min(DEFAULT_DOC_WORKERS, DEFAULT_MAX_WORKERS)

    total_docs = len(df)
    # Emit init event for SSE progress tracking
    print(f"PROGRESS|init|{total_docs}|Found {total_docs} documents to chunk", flush=True)

    with ThreadPoolExecutor(max_workers=num_doc_workers) as executor:
        # df.iterrows() returns (index, Series)
        futures = {
            executor.submit(process_document_chunks, row_series, json_file, model): idx
            for idx, row_series in df.iterrows()
        }

        completed_count = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Traitement des Documents (Chunking)"):
            doc_idx = futures[future]
            completed_count += 1
            try:
                result_chunks = future.result()
                chunk_count = len(result_chunks) if result_chunks else 0
                # Emit row-level progress
                print(f"PROGRESS|row|{completed_count}/{total_docs}|Document #{doc_idx}: {chunk_count} chunks", flush=True)
                print(f"Document #{doc_idx} traité, {chunk_count} chunks produits.")
            except Exception as e:
                print(f"PROGRESS|row|{completed_count}/{total_docs}|Document #{doc_idx}: error", flush=True)
                print(f"Erreur lors du traitement du document #{doc_idx}: {e}")

# ----------------------------------------------------------------------
# PART 2: Chunk Embedding (Dense)
# ----------------------------------------------------------------------

def get_embeddings_batch(texts, model="text-embedding-3-large"):
    """
    Calcule les embeddings pour un lot de textes en une seule requête API.
    """
    try:
        response = client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"Erreur lors du calcul des embeddings pour un lot: {e}")
        # Tentative de retry simple après une pause
        time.sleep(2) 
        try:
            print("Nouvelle tentative de calcul des embeddings pour le lot...")
            response = client.embeddings.create(input=texts, model=model)
            return [item.embedding for item in response.data]
        except Exception as e_retry:
            print(f"Échec du calcul des embeddings pour le lot après nouvelle tentative: {e_retry}")
            return [None] * len(texts) # Retourne None pour les embeddings échoués

def process_chunks_for_embedding(chunks_batch):
    """
    Traite un lot de chunks pour y ajouter les embeddings denses.
    Modifie les dictionnaires de chunks en place.
    """
    texts_to_embed = [chunk.get("text", "") for chunk in chunks_batch]
    embeddings = get_embeddings_batch(texts_to_embed)
    
    for i, embedding in enumerate(embeddings):
        if embedding is not None:
            chunks_batch[i]["embedding"] = embedding
        else:
            # Marquer l'échec ou laisser vide, selon la stratégie souhaitée
            chunks_batch[i]["embedding"] = None 
            print(f"Avertissement: Embedding non généré pour le chunk ID {chunks_batch[i].get('id', 'Inconnu')}")
    return chunks_batch # Retourne le lot modifié

def save_processed_chunks_to_json_overwrite(all_chunks, json_file):
    """
    Sauvegarde la liste complète des chunks (avec embeddings) dans un fichier JSON, en écrasant le contenu existant.
    """
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"Tous les chunks ({len(all_chunks)}) ont été sauvegardés dans {json_file}")

def generate_and_save_embeddings(input_json_file, output_json_file=None):
    """
    Charge les chunks depuis `input_json_file`, génère les embeddings denses,
    et les sauvegarde dans `output_json_file`.
    """
    if output_json_file is None:
        base_name = os.path.splitext(input_json_file)[0]
        output_json_file = f"{base_name}_with_embeddings.json"
    
    if not os.path.exists(input_json_file):
        print(f"Le fichier d'entrée '{input_json_file}' n'existe pas.")
        return None
    
    with open(input_json_file, 'r', encoding='utf-8') as f:
        all_chunks_from_file = json.load(f)
    
    total_chunks = len(all_chunks_from_file)
    print(f"Chargement de {total_chunks} chunks depuis '{input_json_file}' pour génération d'embeddings.")

    # Emit init event for SSE progress tracking (chunks only - documents were processed in step 3.1)
    print(f"PROGRESS|init|{total_chunks}|Generating embeddings for {total_chunks} chunks", flush=True)

    # Regrouper par doc_id pour le traitement par lot (optimisation API)
    chunks_by_doc = {}
    for chunk in all_chunks_from_file:
        doc_id = chunk.get("doc_id", "unknown_doc")
        if doc_id not in chunks_by_doc:
            chunks_by_doc[doc_id] = []
        chunks_by_doc[doc_id].append(chunk)

    all_chunks_with_embeddings = []

    # Process chunks grouped by document (for API efficiency)
    processed_chunk_count = 0
    for doc_id, doc_chunks_list in tqdm(chunks_by_doc.items(), desc="Génération Embeddings"):
        print(f"  Traitement du document {doc_id} ({len(doc_chunks_list)} chunks) pour embeddings...")

        embedded_doc_chunks = []
        for i in range(0, len(doc_chunks_list), DEFAULT_EMBEDDING_BATCH_SIZE):
            batch = doc_chunks_list[i : i + DEFAULT_EMBEDDING_BATCH_SIZE]
            processed_batch = process_chunks_for_embedding(batch)  # Modifies batch in-place
            embedded_doc_chunks.extend(processed_batch)
            # Emit chunk-level progress (primary progress bar)
            current_in_doc = min(i + DEFAULT_EMBEDDING_BATCH_SIZE, len(doc_chunks_list))
            print(f"PROGRESS|chunk|{processed_chunk_count + current_in_doc}/{total_chunks}|Chunk {processed_chunk_count + current_in_doc}/{total_chunks}", flush=True)

        all_chunks_with_embeddings.extend(embedded_doc_chunks)
        processed_chunk_count += len(embedded_doc_chunks)

        # Sauvegarde intermédiaire (optionnelle)
        if processed_chunk_count > 0 and processed_chunk_count % 1000 < DEFAULT_EMBEDDING_BATCH_SIZE:  # Approx every 1000
            temp_output_file = f"{os.path.splitext(output_json_file)[0]}_temp_embeddings.json"
            save_processed_chunks_to_json_overwrite(all_chunks_with_embeddings, temp_output_file)
            print(f"Sauvegarde temporaire de {len(all_chunks_with_embeddings)} chunks avec embeddings dans '{temp_output_file}'.")

    save_processed_chunks_to_json_overwrite(all_chunks_with_embeddings, output_json_file)
    print(f"Tous les embeddings denses ont été générés. Total {len(all_chunks_with_embeddings)} chunks sauvegardés dans '{output_json_file}'.")
    return output_json_file

# ----------------------------------------------------------------------
# PART 3: Chunk sparse embedding
# ----------------------------------------------------------------------

def extract_sparse_features(text):
    """
    Extrait les lemmes des mots pertinents et crée une représentation sparse.
    Utilise le `nlp` global (modèle spaCy).
    """
    if nlp is None:
        print("Erreur: Modèle spaCy (nlp) non initialisé. Impossible d'extraire les features sparse.")
        return {"indices": [], "values": []}
        
    # Limiter la taille des textes très longs pour la performance de spaCy
    if len(text) > nlp.max_length: # Check against model's max_length
         print(f"Warning: Text too long for spaCy ({len(text)} chars), truncating to {nlp.max_length}")
         text = text[:nlp.max_length]
    elif len(text) > 50000: # Fallback if max_length is very large or not restrictive enough
         print(f"Warning: Text quite long ({len(text)} chars), truncating to 50000 for sparse features")
         text = text[:50000]

    doc = nlp(text)
    relevant_pos = {"NOUN", "PROPN", "ADJ", "VERB"}
    lemmas = [
        token.lemma_.lower() for token in doc 
        if token.pos_ in relevant_pos 
        and not token.is_stop 
        and not token.is_punct 
        and len(token.lemma_) > 1 # Exclure les lemmes d'un seul caractère
    ]
    
    counts = Counter(lemmas)
    sparse_dict = {}
    # Utiliser un simple hachage pour créer un indice unique, limité à 100k dimensions
    # La normalisation (TF) est appliquée ici. IDF nécessiterait une connaissance globale du corpus.
    total_lemmas_in_doc = sum(counts.values())
    if total_lemmas_in_doc > 0:
        for lemma, count in counts.items():
            index = hash(lemma) % 100000  # Dimensionnalité de l'espace sparse
            sparse_dict[str(index)] = count / total_lemmas_in_doc # TF (Term Frequency)

    return {
        "indices": list(sparse_dict.keys()), # Convertir les indices en string comme dans le master code
        "values": list(sparse_dict.values())
    }

def generate_sparse_embeddings(input_json_file=DEFAULT_INPUT_JSON_WITH_EMBEDDINGS, 
                               output_json_file=DEFAULT_OUTPUT_JSON_SPARSE):
    """
    Charge les chunks (qui incluent déjà les embeddings denses) depuis `input_json_file`,
    génère les embeddings sparses pour chaque chunk, et sauvegarde le tout dans `output_json_file`.
    """
    if not os.path.exists(input_json_file):
        print(f"Le fichier d'entrée '{input_json_file}' pour les embeddings sparses n'existe pas.")
        return None
    
    with open(input_json_file, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)
    
    total_chunks = len(all_chunks)
    print(f"Chargement de {total_chunks} chunks depuis '{input_json_file}' pour génération d'embeddings sparses.")
    # Emit init event for SSE progress tracking
    print(f"PROGRESS|init|{total_chunks}|Loading {total_chunks} chunks for sparse embedding", flush=True)

    for i, chunk in enumerate(tqdm(all_chunks, desc="Génération Embeddings Sparses")):
        chunk_text = chunk.get("text", "")
        if not chunk_text:
            print(f"Chunk ID {chunk.get('id', i)} a un texte vide, embedding sparse sera vide.")
            sparse_embedding = {"indices": [], "values": []}
        else:
            try:
                sparse_embedding = extract_sparse_features(chunk_text)
            except Exception as e:
                print(f"Erreur lors de la génération de l'embedding sparse pour le chunk ID {chunk.get('id', i)}: {e}")
                sparse_embedding = {"indices": [], "values": []}  # Fallback

        all_chunks[i]["sparse_embedding"] = sparse_embedding  # Ajoute/met à jour la clé "sparse_embedding"

        # Emit chunk-level progress every 50 chunks to avoid overwhelming SSE
        if (i + 1) % 50 == 0 or (i + 1) == total_chunks:
            print(f"PROGRESS|chunk|{i + 1}/{total_chunks}|SpaCy processing chunk {i + 1}", flush=True)

    # Sauvegarde finale des chunks (maintenant avec embeddings denses et sparses)
    # Utilise la même fonction de sauvegarde que pour les embeddings denses (overwrite)
    save_processed_chunks_to_json_overwrite(all_chunks, output_json_file)
    
    print(f"Traitement des embeddings sparses terminé. Fichier sauvegardé: {output_json_file}")
    return output_json_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process text data through chunking and embedding phases.")
    parser.add_argument("--input", required=True, help="Path to the input file (CSV for 'initial' phase, JSON for 'dense' and 'sparse' phases).")
    parser.add_argument("--output", required=True, help="Directory to save the output JSON files.")
    parser.add_argument("--phase", choices=['initial', 'dense', 'sparse', 'all'], default='all',
                        help="Specify processing phase: 'initial' (chunking), 'dense' (dense embeddings), 'sparse' (sparse embeddings), or 'all'.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="LLM model for text recoding. Use 'gpt-4o-mini' (OpenAI) or 'openai/gemini-2.5-flash' (OpenRouter). Default: gpt-4o-mini")

    args = parser.parse_args()

    # Setup logging for chunking phase
    chunking_log_path = os.path.join(args.output, "chunking.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(chunking_log_path, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("chunking")

    # Ensure output directory exists
    if not os.path.exists(args.output):
        try:
            os.makedirs(args.output, exist_ok=True)
            print(f"Output directory '{args.output}' created.")
            logger.info(f"Output directory '{args.output}' created.")
        except Exception as e:
            print(f"Error creating output directory '{args.output}': {e}")
            logger.error(f"Error creating output directory '{args.output}': {e}")
            exit(1)

    # Determine filenames based on the phase and input.
    base_name_for_outputs = "output" # Consistent with main.py's expectation for intermediate files
    if args.phase == "dense" or args.phase == "sparse":
        input_basename = os.path.splitext(os.path.basename(args.input))[0]
        if input_basename.endswith("_chunks_with_embeddings"):
            base_name_for_outputs = input_basename.replace("_chunks_with_embeddings", "")
        elif input_basename.endswith("_chunks"):
            base_name_for_outputs = input_basename.replace("_chunks", "")

    initial_chunks_json = os.path.join(args.output, f"{base_name_for_outputs}_chunks.json")
    chunks_with_dense_json = os.path.join(args.output, f"{base_name_for_outputs}_chunks_with_embeddings.json")
    chunks_with_sparse_json = os.path.join(args.output, f"{base_name_for_outputs}_chunks_with_embeddings_sparse.json")

    print(f"rad_chunk.py - Phase: {args.phase}")
    print(f"  Input File: {args.input}")
    print(f"  Output Directory: {args.output}")
    print(f"  Initial Chunks JSON: {initial_chunks_json}")
    print(f"  Dense Embeddings JSON: {chunks_with_dense_json}")
    print(f"  Sparse Embeddings JSON: {chunks_with_sparse_json}")
    logger.info(f"rad_chunk.py - Phase: {args.phase}")
    logger.info(f"  Input File: {args.input}")
    logger.info(f"  Output Directory: {args.output}")
    logger.info(f"  Initial Chunks JSON: {initial_chunks_json}")
    logger.info(f"  Dense Embeddings JSON: {chunks_with_dense_json}")
    logger.info(f"  Sparse Embeddings JSON: {chunks_with_sparse_json}")
    
    if TEXT_SPLITTER is None:
        print("Erreur critique: TEXT_SPLITTER n'est pas initialisé (langchain_text_splitters manquant?). Arrêt.")
        logger.error("Erreur critique: TEXT_SPLITTER n'est pas initialisé (langchain_text_splitters manquant?). Arrêt.")
        exit(1)
    if nlp is None:
        print("Erreur critique: Modèle spaCy (nlp) n'est pas initialisé. Arrêt.")
        logger.error("Erreur critique: Modèle spaCy (nlp) n'est pas initialisé. Arrêt.")
        exit(1)
    if client is None:
        print("Erreur critique: Client OpenAI non initialisé (OPENAI_API_KEY manquante?). Arrêt.")
        logger.error("Erreur critique: Client OpenAI non initialisé (OPENAI_API_KEY manquante?). Arrêt.")
        exit(1)

    # Phase-specific execution
    if args.phase == 'initial' or args.phase == 'all':
        print("\n--- Phase 3.1 : Découpage initial (initial chunk) ---")
        logger.info("=== Phase 3.1 : Découpage initial (initial chunk) ===")
        if not args.input.lower().endswith(".csv"):
            print(f"Erreur: La phase 'initial' attend un fichier CSV en entrée, reçu: {args.input}")
            logger.error(f"Erreur: La phase 'initial' attend un fichier CSV en entrée, reçu: {args.input}")
            exit(1)
        try:
            df = pd.read_csv(args.input)
            print(f"Chargement de {len(df)} lignes depuis '{args.input}'.")
            logger.info(f"Chargement de {len(df)} lignes depuis '{args.input}'.")
        except FileNotFoundError:
            print(f"Erreur: Le fichier d'entrée CSV '{args.input}' n'a pas été trouvé.")
            logger.error(f"Erreur: Le fichier d'entrée CSV '{args.input}' n'a pas été trouvé.")
            exit(1)
        except Exception as e:
            print(f"Erreur lors du chargement du fichier CSV '{args.input}': {e}")
            logger.error(f"Erreur lors du chargement du fichier CSV '{args.input}': {e}")
            exit(1)

        if os.path.exists(initial_chunks_json):
            print(f"Nettoyage du fichier de chunks existant: {initial_chunks_json}")
            logger.info(f"Nettoyage du fichier de chunks existant: {initial_chunks_json}")
            try:
                os.remove(initial_chunks_json)
            except OSError as e:
                print(f"Avertissement: Impossible de supprimer {initial_chunks_json}: {e}. Le contenu pourrait être ajouté.")
                logger.warning(f"Avertissement: Impossible de supprimer {initial_chunks_json}: {e}. Le contenu pourrait être ajouté.")

        process_all_documents(df, json_file=initial_chunks_json, model=args.model)
        if not os.path.exists(initial_chunks_json) or os.path.getsize(initial_chunks_json) == 0:
            print(f"Erreur: Aucun chunk n'a été généré dans '{initial_chunks_json}'.")
            logger.error(f"Erreur: Aucun chunk n'a été généré dans '{initial_chunks_json}'.")
            exit(1)
        print(f"Phase 'initial' terminée. Output: {initial_chunks_json}")
        logger.info(f"Phase 'initial' terminée. Output: {initial_chunks_json}")

    if args.phase == 'dense' or args.phase == 'all':
        print("\n--- Phase: Dense Embedding Generation ---")
        logger.info("--- Phase: Dense Embedding Generation ---")
        input_for_dense = initial_chunks_json if args.phase == 'all' else args.input
        if not input_for_dense.lower().endswith("_chunks.json") and args.phase != 'all':
             if not input_for_dense.lower().endswith(".json"):
                print(f"Erreur: La phase 'dense' attend un fichier JSON de chunks en entrée (ex: ..._chunks.json), reçu: {input_for_dense}")
                logger.error(f"Erreur: La phase 'dense' attend un fichier JSON de chunks en entrée (ex: ..._chunks.json), reçu: {input_for_dense}")
                exit(1)

        dense_output_file = generate_and_save_embeddings(
            input_json_file=input_for_dense,
            output_json_file=chunks_with_dense_json
        )
        if dense_output_file is None or not os.path.exists(dense_output_file) or os.path.getsize(dense_output_file) == 0:
            print(f"Erreur: Le fichier d'embeddings denses '{chunks_with_dense_json}' n'a pas été généré ou est vide.")
            logger.error(f"Erreur: Le fichier d'embeddings denses '{chunks_with_dense_json}' n'a pas été généré ou est vide.")
            exit(1)
        print(f"Phase 'dense' terminée. Output: {chunks_with_dense_json}")
        logger.info(f"Phase 'dense' terminée. Output: {chunks_with_dense_json}")

    if args.phase == 'sparse' or args.phase == 'all':
        print("\n--- Phase: Sparse Embedding Generation ---")
        logger.info("--- Phase: Sparse Embedding Generation ---")
        input_for_sparse = chunks_with_dense_json if args.phase == 'all' else args.input
        if not input_for_sparse.lower().endswith("_chunks_with_embeddings.json") and args.phase != 'all':
            if not input_for_sparse.lower().endswith(".json"):
                print(f"Erreur: La phase 'sparse' attend un fichier JSON avec embeddings denses (ex: ..._chunks_with_embeddings.json), reçu: {input_for_sparse}")
                logger.error(f"Erreur: La phase 'sparse' attend un fichier JSON avec embeddings denses (ex: ..._chunks_with_embeddings.json), reçu: {input_for_sparse}")
                exit(1)

        sparse_output_file = generate_sparse_embeddings(
            input_json_file=input_for_sparse,
            output_json_file=chunks_with_sparse_json
        )
        if sparse_output_file is None or not os.path.exists(sparse_output_file) or os.path.getsize(sparse_output_file) == 0:
            print(f"Erreur: Le fichier d'embeddings sparses '{chunks_with_sparse_json}' n'a pas été généré ou est vide.")
            logger.error(f"Erreur: Le fichier d'embeddings sparses '{chunks_with_sparse_json}' n'a pas été généré ou est vide.")
            exit(1)
        print(f"Phase 'sparse' terminée. Output: {chunks_with_sparse_json}")
        logger.info(f"Phase 'sparse' terminée. Output: {chunks_with_sparse_json}")
        
    print(f"\nTraitement ({args.phase}) terminé.")
    logger.info(f"Traitement ({args.phase}) terminé.")
