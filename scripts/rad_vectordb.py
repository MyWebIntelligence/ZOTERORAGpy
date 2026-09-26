# II - Implémentation dans une base de données vectorielle

from __future__ import annotations


##  BASE VECTORIELLE - Pinecone

import os
import warnings
# Suppress ResourceWarning, often related to unclosed SSL sockets by HTTP client libraries at script exit.
# Using simplefilter as the more specific filterwarnings might be more effective if message matching was an issue.
warnings.simplefilter("ignore", ResourceWarning)
import json
import time
from tqdm import tqdm
import traceback # Ajout pour traceback.print_exc()

try:
    from pinecone import Pinecone  # Reverted import
    _pinecone_import_error = None
except ImportError as exc:  # pragma: no cover - only triggered when dependency missing
    Pinecone = None
    _pinecone_import_error = exc

# Module de déduplication partagé (Lot 2/3/4). Stdlib seul ; même normalisation/hash
# que le write-path rad_chunk. Import robuste contexte package (Celery/tests) vs CLI.
try:
    from scripts import rad_dedup
except ImportError:
    import rad_dedup
# ----------------------------------------------------------------------
# Environment variable helper with validation
# ----------------------------------------------------------------------
def get_env_int(key: str, default: int, min_val: int = 1) -> int:
    """Get integer from environment with validation and fallback."""
    try:
        value = int(os.getenv(key, default))
        return max(min_val, value)
    except (ValueError, TypeError):
        print(f"Warning: Invalid {key}, using default {default}")
        return default

# Configuration des tailles de lots (configurable via .env)
PINECONE_BATCH_SIZE = get_env_int('PINECONE_BATCH_SIZE', 100)

def upsert_batch_to_pinecone(index, vectors_batch, namespace=None):
    """Upserts a batch of vectors to a Pinecone index.

    Includes a simple retry mechanism for transient errors.

    Args:
        index (pinecone.Index): The initialized Pinecone index object.
        vectors_batch (list[dict]): A list of vector dictionaries formatted for
                                    Pinecone's upsert method. Each dictionary
                                    should contain 'id', 'values', and optionally
                                    'metadata' and 'sparse_values'.
        namespace (str | None): Optional namespace to target within the index.
    Returns:
        bool: True if the upsert was successful (or succeeded on retry),
              False otherwise.
    """
    try:
        upsert_kwargs = {"vectors": vectors_batch}
        if namespace:
            upsert_kwargs["namespace"] = namespace
        index.upsert(**upsert_kwargs)
        return True
    except Exception as e:
        print(f"Erreur lors de l'upsert par lot dans Pinecone: {e}")
        print("Nouvelle tentative dans 2 secondes...")
        time.sleep(2)
        try:
            upsert_kwargs = {"vectors": vectors_batch}
            if namespace:
                upsert_kwargs["namespace"] = namespace
            index.upsert(**upsert_kwargs)
            print("Nouvelle tentative d'upsert réussie.")
            return True
        except Exception as e_retry:
            print(f"Échec après nouvelle tentative d'upsert: {e_retry}")
            return False

def prepare_vectors_for_pinecone(chunks, include_sparse=True):
    """
    Prépare les vecteurs au format attendu par Pinecone, incluant les données de vecteurs sparse si disponibles.
    Chaque 'chunk' d'entrée est supposé être un dictionnaire.
    - 'embedding': contient le vecteur dense (liste de flottants).
    - 'sparse_embedding': (optionnel) contient les données du vecteur sparse
      sous la forme d'un dictionnaire {"indices": [...], "values": [...]}.
    - 'id': l'identifiant unique du vecteur.
    - Autres clés: utilisées comme métadonnées.

    Args:
        chunks (list[dict]): Chunks à préparer.
        include_sparse (bool): Si False, les ``sparse_values`` sont omis. Pinecone
            n'accepte les vecteurs sparse que sur les index ``metric=dotproduct`` ;
            sur un index ``cosine``/``euclidean`` l'upsert d'un vecteur portant des
            ``sparse_values`` est rejeté (HTTP 400). Mettre ce flag à False permet un
            upsert dense uniquement sur ces index. Défaut True (rétro-compatibilité).
    """
    vectors = []
    for chunk in chunks:
        # Vérifier que l'embedding dense existe et n'est pas None
        dense_embedding = chunk.get("embedding")

        if dense_embedding is not None:
            # Construction dynamique des métadonnées
            # Injecte TOUTES les clés du chunk (compatibilité CSV et autres sources)
            metadata = {}
            for key, value in chunk.items():
                # Exclure les champs techniques (vecteurs et identifiants)
                if key not in ("id", "embedding", "sparse_embedding", "values"):
                    metadata[key] = value

            # S'assurer que "text" est présent (backward compatibility)
            if "text" not in metadata and "chunk_text" in chunk:
                metadata["text"] = chunk.get("chunk_text", "")

            vector_data = {
                "id": chunk["id"],
                "values": dense_embedding,  # Vecteur dense
                "metadata": metadata
            }
            
            # Vérifier et ajouter les données du vecteur sparse si elles existent
            # (uniquement si l'index cible le supporte — cf. include_sparse)
            sparse_embedding_data = chunk.get("sparse_embedding")
            if include_sparse and \
               sparse_embedding_data and \
               isinstance(sparse_embedding_data, dict) and \
               "indices" in sparse_embedding_data and \
               "values" in sparse_embedding_data:
                
                # Assurer que les indices sont des entiers et les valeurs des flottants
                try:
                    sparse_indices = [int(i) for i in sparse_embedding_data["indices"]]
                    sparse_values_float = [float(v) for v in sparse_embedding_data["values"]]

                    # Pinecone rejette un sparse_values vide (HTTP 400 "Sparse vector
                    # must contain at least one value") et tout le lot échoue. Certains
                    # chunks (texte = stopwords/ponctuation, très courts) n'ont aucune
                    # feature sparse → on les laisse en dense uniquement (valide sur un
                    # index dotproduct). On exige aussi indices/valeurs de même longueur.
                    if sparse_indices and sparse_values_float and \
                       len(sparse_indices) == len(sparse_values_float):
                        vector_data["sparse_values"] = {
                            "indices": sparse_indices,
                            "values": sparse_values_float
                        }
                except (ValueError, TypeError) as e:
                    print(f"Avertissement: Erreur de formatage des données sparse pour le chunk ID {chunk.get('id', 'ID inconnu')}: {e}. Vecteur sparse ignoré.")

            vectors.append(vector_data)
        else:
            print(f"Avertissement: Embedding dense manquant pour le chunk ID {chunk.get('id', 'N/A')}. Chunk ignoré.")
            
    return vectors

def insert_to_pinecone(embeddings_json_file, index_name="articles", pinecone_api_key=None, namespace=None):
    """Inserts embeddings from a JSON file into a Pinecone index.

    This function handles initializing the Pinecone client, checking for the
    existence of the target index, reading embedding data from the specified
    JSON file, preparing vectors (including dense and sparse if available),
    and upserting them to Pinecone in batches.

    Args:
        embeddings_json_file (str): Path to the JSON file containing embedding data.
                                    Each item in the JSON list should be a dictionary
                                    representing a chunk with at least 'id' and 'embedding'
                                    (for dense vectors), and optionally 'sparse_embedding'.
        index_name (str, optional): The name of the Pinecone index to upsert to.
                                    Defaults to "articles".
        pinecone_api_key (str, optional): The API key for Pinecone. If None, the function
                                          will raise an error internally as it's required.
        namespace (str, optional): Pinecone namespace to target within the index. Defaults
                                   to None which uses the index default namespace.

    Returns:
        dict: A dictionary containing the status of the operation, a descriptive
              message, and the count of successfully inserted vectors.
              Example:
              {
                  "status": "success" | "error" | "partial_error" | "success_partial_data",
              "message": "Descriptive message of the outcome.",
              "inserted_count": int (number of vectors successfully upserted)
          }
    """
    if Pinecone is None:
        raise ImportError(
            "Le paquet 'pinecone' est requis pour l'insertion Pinecone. Installez-le via 'pip install pinecone'."
        ) from _pinecone_import_error

    if not os.path.exists(embeddings_json_file):
        msg = f"Le fichier {embeddings_json_file} n'existe pas."
        print(msg)
        return {"status": "error", "message": msg, "inserted_count": 0}
    
    if not pinecone_api_key:
        msg = "PINECONE_API_KEY is required to initialize Pinecone client."
        print(msg)
        return {"status": "error", "message": msg, "inserted_count": 0}

    try:
        pc = Pinecone(api_key=pinecone_api_key) # Reverted to Pinecone
    except Exception as e:
        msg = f"Erreur lors de l'initialisation du client Pinecone: {e}"
        print(msg)
        traceback.print_exc()
        return {"status": "error", "message": msg, "inserted_count": 0}

    index_list_response = None
    try:
        index_list_response = pc.list_indexes()
        print("Successfully connected to Pinecone and listed indexes.")
    except Exception as e:
        msg = f"Failed to connect to Pinecone or list indexes: {e}"
        print(msg)
        traceback.print_exc()
        return {"status": "error", "message": msg, "inserted_count": 0}

    if index_list_response is None:
        msg = "Pinecone pc.list_indexes() returned None. Cannot proceed."
        print(msg)
        return {"status": "error", "message": msg, "inserted_count": 0}

    existing_index_names = []
    try:
        # For pinecone-client v3.x and later:
        if hasattr(index_list_response, 'indexes'):
            indexes_list = index_list_response.indexes
            if indexes_list is None:
                 msg = "pc.list_indexes().indexes was None. Cannot retrieve index names."
                 print(msg)
                 return {"status": "error", "message": msg, "inserted_count": 0}
            
            if not isinstance(indexes_list, list):
                msg = f"pc.list_indexes().indexes was not a list (type: {type(indexes_list)}). Cannot retrieve index names."
                print(msg)
                return {"status": "error", "message": msg, "inserted_count": 0}

            for idx_description_obj in indexes_list:
                if idx_description_obj is not None and hasattr(idx_description_obj, 'name') and isinstance(idx_description_obj.name, str):
                    existing_index_names.append(idx_description_obj.name)
                else:
                    print(f"Warning: Found an invalid index description object or one without a valid 'name' attribute in pc.list_indexes().indexes. Object: {idx_description_obj}")
        
        # Fallback for older versions (e.g., pinecone-client v2.x)
        elif hasattr(index_list_response, 'names'):
            if callable(index_list_response.names):
                names_result = index_list_response.names()
                if isinstance(names_result, list) and all(isinstance(name, str) for name in names_result):
                    existing_index_names = names_result
                else:
                    msg = "pc.list_indexes().names() did not return a list of strings."
                    print(msg)
                    return {"status": "error", "message": msg, "inserted_count": 0}
            elif isinstance(index_list_response.names, list) and all(isinstance(name, str) for name in index_list_response.names):
                existing_index_names = index_list_response.names
            else:
                msg = "pc.list_indexes().names was neither a callable returning a list of strings, nor a list of strings."
                print(msg)
                return {"status": "error", "message": msg, "inserted_count": 0}
        else:
            msg = (f"Could not extract index names from Pinecone's list_indexes response. "
                   f"The response object (type: {type(index_list_response)}) did not have an 'indexes' attribute (for v3+) "
                   f"or a 'names' attribute/method (for v2.x). Please check Pinecone client version and response structure.")
            print(msg)
            return {"status": "error", "message": msg, "inserted_count": 0}
            
        print(f"Available indexes: {existing_index_names}")
        if not existing_index_names: # This is a warning, not an error that stops processing yet.
            print("Warning: No indexes found in Pinecone account.")
            
    except Exception as e:
        msg = f"Error processing index list from Pinecone: {e}"
        print(msg)
        traceback.print_exc()
        return {"status": "error", "message": msg, "inserted_count": 0}

    if index_name not in existing_index_names:
        msg = f"Index '{index_name}' does not exist. Please create it in Pinecone first. Available indexes: {existing_index_names}"
        print(msg)
        return {"status": "error", "message": msg, "inserted_count": 0}
    
    index = None
    try:
        index = pc.Index(index_name)
        ns_msg = f" (namespace: '{namespace}')" if namespace else ""
        print(f"Connecté à l'index Pinecone: {index_name}{ns_msg}")
    except Exception as e:
        msg = f"Failed to connect to Pinecone index '{index_name}': {e}"
        print(msg)
        traceback.print_exc()
        return {"status": "error", "message": msg, "inserted_count": 0}

    # Pinecone n'accepte les vecteurs sparse que sur les index metric=dotproduct.
    # Sur un index cosine/euclidean, un upsert portant des sparse_values est rejeté
    # (HTTP 400) → tous les lots échouent silencieusement. On détecte le metric une
    # fois et on bascule en upsert dense uniquement si l'index n'est pas dotproduct.
    index_metric = None
    try:
        index_metric = pc.describe_index(index_name).metric
    except Exception as e:
        print(f"Avertissement: describe_index('{index_name}') a échoué ({e}); "
              f"vecteurs sparse désactivés par sécurité.")
    include_sparse = (index_metric == "dotproduct")
    if not include_sparse:
        print(f"Index '{index_name}' metric={index_metric!r} (non-dotproduct): "
              f"vecteurs sparse omis, upsert dense uniquement.")
    
    all_chunks = []
    try:
        with open(embeddings_json_file, 'r', encoding='utf-8') as f:
            all_chunks = json.load(f)
        print(f"Chargement des embeddings depuis {embeddings_json_file} réussi. {len(all_chunks)} chunks chargés.")
    except json.JSONDecodeError as e:
        msg = f"Erreur de décodage JSON dans le fichier {embeddings_json_file}: {e}"
        print(msg)
        traceback.print_exc()
        return {"status": "error", "message": msg, "inserted_count": 0}
    except Exception as e:
        msg = f"Erreur lors du chargement du fichier {embeddings_json_file}: {e}"
        print(msg)
        traceback.print_exc()
        return {"status": "error", "message": msg, "inserted_count": 0}
        
    # Déduplication (Lot 2/3) sur la liste PLATE, AVANT le regroupement par doc :
    # batche l'existence À TRAVERS les documents (chunks_by_doc n'existe que pour
    # l'ordre d'upsert). Réutilise index_metric déjà calculé. No-op si DEDUP_ENABLED=0.
    _dedup_adapter = _PineconeDedupAdapter(index, namespace=namespace, metric=index_metric)
    all_chunks, dedup_skipped, dedup_journal = _run_dedup(
        all_chunks, _dedup_adapter, embeddings_json_file,
        target_desc=f"{index_name}/{namespace or 'default'}",
    )

    chunks_by_doc = {}
    for chunk_data in all_chunks: # Renamed 'chunk' to 'chunk_data' to avoid conflict if 'chunk' is a key in the dict
        doc_id = chunk_data.get("doc_id", "unknown_document")
        if doc_id not in chunks_by_doc:
            chunks_by_doc[doc_id] = []
        chunks_by_doc[doc_id].append(chunk_data)
    
    total_inserted_count = 0
    total_processed_chunks = 0
    any_batch_failed = False

    for doc_id, doc_chunks in tqdm(chunks_by_doc.items(), desc="Insertion des documents dans Pinecone"):
        print(f"\nTraitement du document {doc_id} ({len(doc_chunks)} chunks)")
        
        for i in range(0, len(doc_chunks), PINECONE_BATCH_SIZE):
            batch_chunks = doc_chunks[i:i+PINECONE_BATCH_SIZE]
            vectors_to_upsert = prepare_vectors_for_pinecone(batch_chunks, include_sparse=include_sparse)
            total_processed_chunks += len(batch_chunks) 
            
            if vectors_to_upsert:
                success_upsert = upsert_batch_to_pinecone(index, vectors_to_upsert, namespace=namespace)
                if success_upsert:
                    total_inserted_count += len(vectors_to_upsert)
                    print(f"Lot {i//PINECONE_BATCH_SIZE + 1}: {len(vectors_to_upsert)} vecteurs insérés avec succès pour le document {doc_id}.")
                else:
                    any_batch_failed = True
                    print(f"Lot {i//PINECONE_BATCH_SIZE + 1}: Échec de l'insertion du lot pour le document {doc_id}.")
            elif batch_chunks: 
                 print(f"Lot {i//PINECONE_BATCH_SIZE + 1}: Aucun vecteur valide à insérer pour le document {doc_id}.")

    final_message_parts = ["Insertion terminée."]
    if namespace:
        final_message_parts.append(f"Namespace ciblé: {namespace}.")
    final_message_parts.extend([
        f"Total de chunks traités (tentative de préparation): {total_processed_chunks}.",
        f"Total de chunks effectivement préparés et insérés avec succès dans Pinecone: {total_inserted_count}",
        f"(sur {len(all_chunks)} chunks initialement chargés si tous étaient valides)."
    ])
    final_message = " ".join(final_message_parts)
    print(f"\n{final_message}")

    # Note (sémantique de comptage) : la dédup a filtré all_chunks AVANT le
    # regroupement, donc total_processed_chunks ne compte QUE les chunks conservés →
    # un ré-ingest entièrement dédupliqué puis inséré reste 'success' (pas
    # 'success_partial_data'). dedup_skipped est surfacé séparément.
    if any_batch_failed:
        return _vectordb_result("partial_error", f"{final_message} Au moins un lot n'a pas pu être inséré.", inserted_count=total_inserted_count, skipped_count=dedup_skipped, journal_path=dedup_journal)
    elif total_inserted_count == 0 and len(all_chunks) > 0: # Processed chunks but none inserted
         return _vectordb_result("error", f"{final_message} Aucun chunk n'a été inséré.", inserted_count=total_inserted_count, skipped_count=dedup_skipped, journal_path=dedup_journal)
    elif total_inserted_count < total_processed_chunks and not any_batch_failed: # Some chunks were invalid but all valid upserted
        return _vectordb_result("success_partial_data", f"{final_message} Certains chunks étaient invalides et n'ont pas été préparés pour l'insertion.", inserted_count=total_inserted_count, skipped_count=dedup_skipped, journal_path=dedup_journal)

    return _vectordb_result("success", final_message, inserted_count=total_inserted_count, skipped_count=dedup_skipped, journal_path=dedup_journal)

    
    # IMPORTANT: Décommentez et configurez les lignes suivantes pour exécuter l'insertion
    # Assurez-vous que le fichier JSON existe et est correctement formaté.
    # Exemple de création d'un fichier JSON de test :
    # dummy_data = [
    #   {
    #     "id": "doc1_chunk1", "embedding": [0.1]*128, 
    #     "sparse_embedding": {"indices": [10, 25], "values": [0.5, 0.8]},
    #     "doc_id": "doc1", "text": "Texte du chunk 1."
    #   },
    #   {
    #     "id": "doc1_chunk2", "embedding": [0.2]*128, 
    #     "doc_id": "doc1", "text": "Texte du chunk 2."
    #   }
    # ]
    # with open(embeddings_json_file_with_sparse, 'w', encoding='utf-8') as f:
    #    json.dump(dummy_data, f, indent=2)
    # print(f"Fichier de test '{embeddings_json_file_with_sparse}' créé. Veuillez le vérifier/modifier si nécessaire.")

# Commented out the execution part as it requires a live Pinecone instance and API key
# api_key_to_use = "pcsk_2r6Wb3_H1BjFjiKG6qQH4ro1BmzrAd8Gnz4a9wzo6J6SPzZyzcVkPdfYjUvZ91tLo2pfaA"
# embeddings_json_file_with_sparse = "path_to_your_embeddings_file.json" # Placeholder

# insert_to_pinecone(
#   embeddings_json_file=embeddings_json_file_with_sparse,
#   index_name="articles", # REMPLACEZ par le nom de votre index Pinecone
#   pinecone_api_key=api_key_to_use
# )


## BASE VECTORIELLE Weaviate

from uuid import uuid5, NAMESPACE_DNS
from dateutil import parser
import re

try:
    import weaviate
    from weaviate.classes.init import Auth
    from weaviate.classes.query import Filter as WeaviateFilter, MetadataQuery as WeaviateMetadataQuery
    _weaviate_import_error = None
except ImportError as exc:  # pragma: no cover - optional dependency
    weaviate = None
    Auth = None
    WeaviateFilter = None
    WeaviateMetadataQuery = None
    _weaviate_import_error = exc

try:
    import qdrant_client
    from qdrant_client import models
    _qdrant_import_error = None
except ImportError as exc:  # pragma: no cover - optional dependency
    qdrant_client = None
    models = None
    _qdrant_import_error = exc

# Configuration des tailles de lots
WEAVIATE_BATCH_SIZE = get_env_int('WEAVIATE_BATCH_SIZE', 100)
QDRANT_BATCH_SIZE = get_env_int('QDRANT_BATCH_SIZE', 100)  # Taille de lot pour Qdrant


# ======================================================================
# Déduplication (Lot 2/3/4) — adapters concrets + wiring
# ======================================================================
#
# Choix d'implémentation : l'existence Tier 2 se fait par **ID adressé par contenu**
# pour les 3 bases (Pinecone `fetch`, Weaviate `fetch_objects(by_id)`, Qdrant
# `retrieve`) — l'ID/UUID encode déjà (content_hash, chunk_index). C'est uniforme,
# métrique-indépendant, schéma-agnostique (robuste à un schéma Weaviate Article figé,
# pas besoin que content_hash soit une propriété filtrable) et sans risque de
# starvation de `limit` qu'aurait un filtre `content_hash` contains_any/MatchAny sur
# un hash « chaud ». Pinecone N'utilise JAMAIS query+$in (tronqué à top_k sur
# serverless → present-set incomplet → fuite de doublons). Le refus exige toujours
# une corroboration métadonnée (titre) côté rad_dedup.dedup_filter.

class _PineconeDedupAdapter:
    """Existence via ``index.fetch(ids=content_ids, namespace=ns)`` (lookup-clé,
    pas de query+$in). ``nearest`` (Tier 3) via ``index.query`` filtré titre."""

    db_name = "pinecone"

    def __init__(self, index, namespace=None, metric=None):
        """Mémorise l'index Pinecone, le namespace (optionnel) et la métrique de
        l'index (sens du seuil Tier 3)."""
        self._index = index
        self._namespace = namespace
        self.metric = metric

    def existing(self, chunks_batch):
        """Tier 2 : ``index.fetch`` des IDs adressés par contenu du lot. Renvoie
        ``{content_hash: [meta, ...]}`` pour les vecteurs déjà présents (``id`` et
        ``content_hash`` ajoutés à la métadonnée s'ils manquent)."""
        id_to_hash = {
            rad_dedup.content_id(c["content_hash"], c["chunk_index"]): c["content_hash"]
            for c in chunks_batch
        }
        kwargs = {"ids": list(id_to_hash.keys())}
        if self._namespace:
            kwargs["namespace"] = self._namespace
        resp = self._index.fetch(**kwargs)
        vectors = getattr(resp, "vectors", None)
        if vectors is None and isinstance(resp, dict):
            vectors = resp.get("vectors", {})
        out = {}
        for vid, vec in (vectors or {}).items():
            meta = getattr(vec, "metadata", None)
            if meta is None and isinstance(vec, dict):
                meta = vec.get("metadata", {})
            meta = dict(meta or {})
            meta.setdefault("id", vid)
            chash = id_to_hash.get(vid)
            if chash:
                meta.setdefault("content_hash", chash)
                out.setdefault(chash, []).append(meta)
        return out

    def nearest(self, embedding, meta_filter):
        """Tier 3 : ``index.query`` top-3 avec métadonnées, filtré par égalité sur les
        champs non vides de ``meta_filter``. Renvoie ``[{"score", "meta", "text"}]``."""
        flt = {k: {"$eq": v} for k, v in (meta_filter or {}).items() if v not in (None, "")}
        kwargs = {"vector": list(embedding), "top_k": 3, "include_metadata": True}
        if self._namespace:
            kwargs["namespace"] = self._namespace
        if flt:
            kwargs["filter"] = flt
        resp = self._index.query(**kwargs)
        matches = getattr(resp, "matches", None)
        if matches is None and isinstance(resp, dict):
            matches = resp.get("matches", [])
        out = []
        for m in (matches or []):
            meta = getattr(m, "metadata", None)
            if meta is None and isinstance(m, dict):
                meta = m.get("metadata", {})
            meta = dict(meta or {})
            score = getattr(m, "score", None)
            if score is None and isinstance(m, dict):
                score = m.get("score")
            out.append({"score": score, "meta": meta, "text": meta.get("text", "")})
        return out


class _WeaviateDedupAdapter:
    """Existence via ``fetch_objects(filters=Filter.by_id().contains_any(uuids))`` —
    l'UUID = ``generate_uuid(content_id)`` est adressé par contenu. ``text`` EXCLU des
    ``return_properties`` (egress). ``near_vector`` renvoie une DISTANCE → metric."""

    db_name = "weaviate"
    metric = "distance"

    def __init__(self, collection_with_tenant):
        """Mémorise la collection Weaviate déjà scopée sur le tenant."""
        self._col = collection_with_tenant

    def existing(self, chunks_batch):
        """Tier 2 : ``fetch_objects`` filtré sur les UUID (``generate_uuid(id)``) du
        lot, sans la propriété ``text``. Renvoie ``{content_hash: [props, ...]}`` pour
        les objets déjà présents."""
        uuid_to_hash = {generate_uuid(c["id"]): c["content_hash"] for c in chunks_batch}
        resp = self._col.query.fetch_objects(
            filters=WeaviateFilter.by_id().contains_any(list(uuid_to_hash.keys())),
            return_properties=["content_hash", "title", "authors", "chunk_index"],
            limit=len(uuid_to_hash) + 1,
        )
        out = {}
        for obj in getattr(resp, "objects", []) or []:
            props = dict(obj.properties or {})
            chash = uuid_to_hash.get(str(obj.uuid))
            if chash:
                props.setdefault("content_hash", chash)
                props.setdefault("id", str(obj.uuid))
                out.setdefault(chash, []).append(props)
        return out

    def nearest(self, embedding, meta_filter):
        """Tier 3 : ``near_vector`` top-3, filtré sur ``title`` s'il est fourni.
        Renvoie ``[{"score", "meta", "text"}]`` où ``score`` est une DISTANCE."""
        title = (meta_filter or {}).get("title")
        filters = WeaviateFilter.by_property("title").equal(title) if title not in (None, "") else None
        resp = self._col.query.near_vector(
            near_vector=list(embedding),
            limit=3,
            filters=filters,
            return_properties=["content_hash", "title", "authors", "chunk_index", "text"],
            return_metadata=WeaviateMetadataQuery(distance=True),
        )
        out = []
        for obj in getattr(resp, "objects", []) or []:
            props = dict(obj.properties or {})
            md = getattr(obj, "metadata", None)
            dist = getattr(md, "distance", None) if md is not None else None
            out.append({"score": dist, "meta": props, "text": props.get("text", "")})
        return out


class _QdrantDedupAdapter:
    """Existence via ``client.retrieve(ids=uuids)`` — UUID adressé par contenu, pas
    de payload index requis. Qdrant COSINE : score ∈ [0,1], plus haut = plus proche."""

    db_name = "qdrant"
    metric = "cosine"

    def __init__(self, client, collection_name):
        """Mémorise le client Qdrant et le nom de la collection cible."""
        self._client = client
        self._collection = collection_name

    def existing(self, chunks_batch):
        """Tier 2 : ``client.retrieve`` des UUID (``generate_uuid(id)``) du lot, payload
        restreint et sans vecteurs. Renvoie ``{content_hash: [payload, ...]}`` pour les
        points déjà présents."""
        id_to_hash = {generate_uuid(c["id"]): c["content_hash"] for c in chunks_batch}
        points = self._client.retrieve(
            collection_name=self._collection,
            ids=list(id_to_hash.keys()),
            with_payload=["content_hash", "title", "authors", "chunk_index"],
            with_vectors=False,
        )
        out = {}
        for p in points or []:
            payload = dict(getattr(p, "payload", None) or {})
            chash = id_to_hash.get(str(p.id))
            if chash:
                payload.setdefault("content_hash", chash)
                payload.setdefault("id", str(p.id))
                out.setdefault(chash, []).append(payload)
        return out

    def nearest(self, embedding, meta_filter):
        """Tier 3 : ``query_points`` top-3, filtré sur ``title`` s'il est fourni.
        Renvoie ``[{"score", "meta", "text"}]`` (score cosinus)."""
        title = (meta_filter or {}).get("title")
        flt = None
        if title not in (None, ""):
            flt = models.Filter(must=[models.FieldCondition(key="title", match=models.MatchValue(value=title))])
        res = self._client.query_points(
            collection_name=self._collection,
            query=list(embedding),
            limit=3,
            query_filter=flt,
            with_payload=True,
            with_vectors=False,
        )
        points = getattr(res, "points", res) or []
        out = []
        for p in points:
            payload = dict(getattr(p, "payload", None) or {})
            out.append({"score": getattr(p, "score", None), "meta": payload, "text": payload.get("text", "")})
        return out


def _run_dedup(all_chunks, adapter, embeddings_json_file, target_desc):
    """Wrapper commun aux 3 connecteurs : lit ``DedupConfig.from_env()``, place le
    journal à côté du JSON d'embeddings, lance ``rad_dedup.dedup_filter``.

    Returns:
        tuple: ``(kept_chunks, skipped_count, journal_path|None)``. No-op vrai
        (renvoie ``(all_chunks, 0, None)`` sans toucher l'adapter) si ``DEDUP_ENABLED``
        est faux.
    """
    cfg = rad_dedup.DedupConfig.from_env()
    if not cfg.enabled:
        return all_chunks, 0, None
    journal_dir = cfg.journal_dir or os.path.dirname(os.path.abspath(embeddings_json_file))

    # Lot 7.d : ne PAS upserter les chunks au recodage échoué (RAW stocké) — sinon la
    # dédup exact-hash refuserait au run suivant la version propre et la base garderait
    # le RAW à jamais. Exclus AVANT le filtre dédup (le content_hash, sur le brut, est
    # inchangé : la clé de décision n'est pas affectée). Champ absent si Lot 7 OFF.
    _fallback = {"fallback_raw", "fallback_truncated"}
    dedup_input = [c for c in all_chunks if c.get("recode_status") not in _fallback]
    fallback_count = len(all_chunks) - len(dedup_input)
    if fallback_count:
        print(f"Déduplication ({target_desc}) : {fallback_count} chunk(s) au recodage échoué "
              f"(fallback) exclus de l'upsert (Lot 7.d).")

    kept, rejections = rad_dedup.dedup_filter(
        dedup_input, adapter, cfg, journal_dir=journal_dir, target_desc=target_desc
    )
    skipped = len(rejections)
    journal_path = rad_dedup.journal_path_for(journal_dir) if rejections else None
    if skipped:
        print(f"Déduplication ({target_desc}) : {skipped} chunk(s) refusé(s) sur {len(all_chunks)} ; "
              f"{len(kept)} conservé(s). Journal : {journal_path}")
    return kept, skipped, journal_path


def generate_uuid(identifier):
    """Generates a stable UUID version 5 from a given string identifier.

    This is used to create consistent IDs for vector database entries
    based on their original chunk IDs.

    Args:
        identifier (str): The string to base the UUID on.

    Returns:
        str: The generated UUID as a string.
    """
    return str(uuid5(NAMESPACE_DNS, identifier))

def normalize_date_to_rfc3339(date_str):
    """Converts a heterogeneous date string to RFC3339 format (YYYY-MM-DDTHH:MM:SSZ).

    Handles various common date formats including year-only, year-month,
    and full dates. If parsing fails or the input is invalid/empty,
    it defaults to "1970-01-01T00:00:00Z".

    Args:
        date_str (str | None): The date string to normalize. Can be None or empty.

    Returns:
        str: The date string in RFC3339 format, or a default if conversion fails.
             The 'Z' indicates UTC timezone.
    """
    if not date_str or not isinstance(date_str, str) or date_str.strip() == "":
        return "1970-01-01T00:00:00Z"
        
    try:
        # Cas 1: Année seule (YYYY)
        if re.fullmatch(r"^\d{4}$", date_str.strip()):
            return f"{date_str.strip()}-01-01T00:00:00Z"
        
        # Cas 2: Année et mois (YYYY-MM ou YYYY/MM)
        if re.fullmatch(r"^\d{4}[-/]\d{1,2}$", date_str.strip()):
            dt = parser.parse(date_str.strip() + "-01") # Ajoute un jour pour parser
            return dt.strftime("%Y-%m-%dT00:00:00Z")

        # Cas 3: Date complète (YYYY-MM-DD, YYYY/MM/DD, DD-MM-YYYY, etc.)
        # dateutil.parser est assez flexible pour gérer de nombreux formats
        dt = parser.parse(date_str.strip())
        return dt.isoformat(timespec='seconds') + "Z" # Assure le format RFC3339 avec Z
        
    except (ValueError, TypeError, parser.ParserError) as e:
        # print(f"Avertissement: Impossible de parser la date '{date_str}'. Utilisation de la date par défaut. Erreur: {e}")
        return "1970-01-01T00:00:00Z"


def _vectordb_result(status, message, inserted_count=0, skipped_count=0, journal_path=None):
    """Retour unifié des connecteurs (Lot 0.a). Mirrors la forme Pinecone
    ``{status, message, inserted_count}`` et porte optionnellement les compteurs
    de dédup (Lot 3) ``skipped_count`` / ``journal_path``.
    """
    res = {"status": status, "message": message, "inserted_count": inserted_count}
    if skipped_count:
        res["skipped_count"] = skipped_count
    if journal_path:
        res["journal_path"] = journal_path
    return res


def insert_to_weaviate_hybrid(embeddings_json_file, url, api_key, class_name="Article", tenant_name="alakel"):
    """Inserts embeddings from a JSON file into a Weaviate collection with multi-tenancy.

    Handles connection to Weaviate Cloud, tenant creation if not exists,
    reading embeddings from JSON, preparing data objects, and batch inserting
    them into the specified Weaviate class (collection) under the given tenant.

    Args:
        embeddings_json_file (str): Path to the JSON file containing embedding data.
                                    Each item should be a chunk dictionary with 'id',
                                    'embedding', and other metadata.
        url (str): The Weaviate cluster URL.
        api_key (str): The Weaviate API key.
        class_name (str, optional): The name of the Weaviate class (collection)
                                    to insert data into. Defaults to "Article".
        tenant_name (str, optional): The name of the tenant to use. Defaults to "alakel".

    Returns:
        dict: ``{"status", "message", "inserted_count"}`` (+ optionnellement
            ``"skipped_count"``/``"journal_path"`` si la dédup est active). ``status``
            ∈ {"success", "success_partial_data", "error"}. Forme unifiée (Lot 0.a)
            avec ``insert_to_pinecone`` pour que le dispatch CLI et l'appelant Celery
            soient homogènes.
    """
    if weaviate is None or Auth is None:
        raise ImportError(
            "Le paquet 'weaviate-client' est requis pour l'insertion Weaviate. Installez-le via 'pip install weaviate-client'."
        ) from _weaviate_import_error

    if not os.path.exists(embeddings_json_file):
        print(f"Le fichier {embeddings_json_file} n'existe pas.")
        return _vectordb_result("error", f"Le fichier {embeddings_json_file} n'existe pas.")

    client = None  # Initialiser client à None
    
    if not url:
        raise ValueError("Weaviate Cluster URL (url) is required.")
    if not api_key:
        raise ValueError("Weaviate API Key (api_key) is required.")

    try:
        # Connexion à Weaviate Cloud
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=Auth.api_key(api_key),
        )
        
        if not client.is_ready():
            print("Le serveur Weaviate n'est pas prêt.")
            if client: client.close()
            return _vectordb_result("error", "Le serveur Weaviate n'est pas prêt.")
            
        print("Connexion réussie à Weaviate Cloud")
        
        # Obtenir la référence à la collection
        collection = client.collections.get(class_name)
        
        # Vérifier les tenants existants
        try:
            tenants_data = collection.tenants.get() # Returns a weaviate.collections.classes.tenants.Tenants object (Dict[str, Tenant])
            existing_tenant_names = []
            if tenants_data is not None:
                # The keys of the Tenants object are the tenant names (strings)
                existing_tenant_names = list(tenants_data.keys())
            
            print(f"Tenants existants: {existing_tenant_names}")
            
            if tenant_name not in existing_tenant_names:
                print(f"Le tenant '{tenant_name}' n'existe pas. Création en cours...")
                collection.tenants.create(tenant_name)
                print(f"Tenant '{tenant_name}' créé avec succès.")
            else:
                print(f"Le tenant '{tenant_name}' existe déjà.")
                
        except Exception as e:
            print(f"Erreur lors de la vérification/création des tenants: {e}")
            # Tenter de créer le tenant directement comme fallback
            try:
                print(f"Tentative de création directe du tenant '{tenant_name}'...")
                collection.tenants.create(tenant_name)
                print(f"Tenant '{tenant_name}' créé avec succès (après fallback).")
            except Exception as e_create:
                print(f"Impossible de créer le tenant '{tenant_name}' même en fallback: {e_create}")
                if client: client.close()
                return _vectordb_result("error", f"Impossible de créer le tenant '{tenant_name}': {e_create}")
        
        # Charger les chunks avec embeddings
        print(f"Chargement des embeddings depuis {embeddings_json_file}")
        with open(embeddings_json_file, 'r', encoding='utf-8') as f:
            all_chunks = json.load(f)
        
        print(f"Chargement de {len(all_chunks)} chunks avec embeddings")
        
        # Traiter les chunks par lots
        total_inserted = 0
        
        # Utiliser la collection spécifique au tenant pour le batching
        collection_with_tenant = collection.with_tenant(tenant_name)

        # Déduplication (Lot 2/3) — requête la cible EXACTE (tenant résolu), avant la
        # boucle d'insertion. No-op si DEDUP_ENABLED=0.
        _dedup_adapter = _WeaviateDedupAdapter(collection_with_tenant)
        all_chunks, dedup_skipped, dedup_journal = _run_dedup(
            all_chunks, _dedup_adapter, embeddings_json_file,
            target_desc=f"{class_name}/{tenant_name}",
        )

        for i in range(0, len(all_chunks), WEAVIATE_BATCH_SIZE):
            batch_data_objects = [] # Liste pour stocker les objets à insérer dans ce lot
            
            current_batch_chunks = all_chunks[i:i+WEAVIATE_BATCH_SIZE]
            
            for chunk in current_batch_chunks:
                if chunk.get("embedding") is not None:
                    uuid_str = generate_uuid(chunk["id"]) # Ensure this is a string for DataObject

                    # Construction dynamique des properties
                    # Injecte TOUTES les clés du chunk (compatibilité CSV et autres sources)
                    properties = {}
                    for key, value in chunk.items():
                        # Exclure les champs techniques (vecteurs et identifiants)
                        if key not in ("id", "embedding", "sparse_embedding"):
                            # Normaliser les dates pour Weaviate (RFC3339)
                            if key in ("date", "created_at", "updated_at", "published_at") and value:
                                properties[key] = normalize_date_to_rfc3339(str(value))
                            else:
                                properties[key] = value

                    # S'assurer que "text" est présent (backward compatibility)
                    if "text" not in properties and "chunk_text" in chunk:
                        properties["text"] = chunk.get("chunk_text", "")
                    
                    batch_data_objects.append(
                        weaviate.classes.data.DataObject(
                            properties=properties,
                            uuid=uuid_str, # uuid parameter expects a string or UUID object
                            vector=chunk["embedding"]
                        )
                    )
            
            if batch_data_objects:
                try:
                    results = collection_with_tenant.data.insert_many(batch_data_objects) # Should return BatchResults
                    
                    num_successful_in_batch = 0
                    if results.has_errors: # Check this first
                        num_failed_in_batch = len(results.errors)
                        num_successful_in_batch = len(batch_data_objects) - num_failed_in_batch
                        
                        print(f"  {num_failed_in_batch} objets sur {len(batch_data_objects)} ont échoué dans ce lot.")
                        for original_idx, error_obj in results.errors.items():
                            # original_idx is the index in the input batch_data_objects list
                            print(f"    Erreur pour l'objet à l'index original {original_idx} (UUID: {batch_data_objects[original_idx].uuid}): {error_obj.message}")
                    else:
                        # No errors in the batch
                        num_successful_in_batch = len(batch_data_objects)
                    
                    total_inserted += num_successful_in_batch
                    print(f"Lot {i//WEAVIATE_BATCH_SIZE + 1}/{(len(all_chunks) + WEAVIATE_BATCH_SIZE - 1)//WEAVIATE_BATCH_SIZE}: {num_successful_in_batch}/{len(batch_data_objects)} objets insérés avec succès.")

                except Exception as e_batch:
                    print(f"Erreur majeure lors de l'insertion du lot {i//WEAVIATE_BATCH_SIZE + 1}: {e_batch}")
                    traceback.print_exc() 
            else:
                print(f"Lot {i//WEAVIATE_BATCH_SIZE + 1}: Aucun objet valide à insérer.")

        print(f"Insertion terminée. {total_inserted}/{len(all_chunks)} chunks insérés avec succès dans Weaviate (tenant: {tenant_name}).")
        if client: client.close()
        msg = f"Insertion Weaviate terminée (tenant: {tenant_name}). {total_inserted}/{len(all_chunks)} chunks insérés."
        if total_inserted == 0 and len(all_chunks) > 0:
            return _vectordb_result("error", msg + " Aucun chunk inséré.", inserted_count=0, skipped_count=dedup_skipped, journal_path=dedup_journal)
        if total_inserted < len(all_chunks):
            return _vectordb_result(
                "success_partial_data",
                msg + " Certains chunks sans embedding (ou en échec de lot) n'ont pas été insérés.",
                inserted_count=total_inserted, skipped_count=dedup_skipped, journal_path=dedup_journal,
            )
        return _vectordb_result("success", msg, inserted_count=total_inserted, skipped_count=dedup_skipped, journal_path=dedup_journal)

    except Exception as e:
        print(f"Erreur globale lors du traitement Weaviate: {e}")
        traceback.print_exc() # Imprime le traceback complet
        if client:
            try:
                client.close()
            except:
                pass
        return _vectordb_result("error", f"Erreur globale Weaviate: {e}")


## BASE VECTORIELLE Qdrant

def prepare_points_for_qdrant(chunks):
    """Prepares points (vectors and metadata) for Qdrant.

    Converts a list of chunk dictionaries into a list of Qdrant PointStruct objects.
    Each chunk is expected to have an 'id' and an 'embedding' (dense vector).
    Other keys in the chunk dictionary are stored in the 'payload' of the PointStruct.
    A stable UUID is generated from the chunk's 'id' to serve as the Qdrant point ID.

    Args:
        chunks (list[dict]): A list of chunk dictionaries. Each dictionary should
                             contain at least 'id' and 'embedding'.

    Returns:
        list[qdrant_client.models.PointStruct]: A list of PointStruct objects
                                                ready for upsertion to Qdrant.
                                                Chunks missing 'embedding' are skipped.
    """
    if qdrant_client is None or models is None:
        raise ImportError(
            "Le paquet 'qdrant-client' est requis pour l'insertion Qdrant. Installez-le via 'pip install qdrant-client'."
        ) from _qdrant_import_error

    points = []
    for chunk in chunks:
        dense_embedding = chunk.get("embedding")

        if dense_embedding is not None:
            # Utiliser l'ID du chunk comme ID du point Qdrant.
            # Qdrant accepte les UUIDs (chaînes ou objets UUID) ou les entiers comme ID.
            # Générer un UUID v5 stable à partir de l'ID original du chunk pour assurer la compatibilité.
            point_id = generate_uuid(chunk["id"])

            # Construction dynamique du payload
            # Injecte TOUTES les clés du chunk (compatibilité CSV et autres sources)
            payload = {"original_id": chunk["id"]}  # Garder l'ID original dans le payload

            for key, value in chunk.items():
                # Exclure les champs techniques (vecteurs et identifiants)
                if key not in ("id", "embedding", "sparse_embedding"):
                    payload[key] = value

            # S'assurer que "text" est présent (backward compatibility)
            if "text" not in payload and "chunk_text" in chunk:
                payload["text"] = chunk.get("chunk_text", "")
            
            # Créer l'objet PointStruct
            point = models.PointStruct(
                id=point_id,
                vector=dense_embedding,
                payload=payload
            )
            points.append(point)
        else:
            print(f"Avertissement: Embedding dense manquant pour le chunk ID {chunk.get('id', 'N/A')}. Chunk ignoré pour Qdrant.")
            
    return points

def upsert_batch_to_qdrant(client: qdrant_client.QdrantClient, collection_name: str, points_batch: list):
    """Upserts a batch of points to a Qdrant collection.

    Includes a simple retry mechanism for transient errors.

    Args:
        client (qdrant_client.QdrantClient): The initialized Qdrant client.
        collection_name (str): The name of the Qdrant collection.
        points_batch (list[qdrant_client.models.PointStruct]): A list of PointStruct
                                                              objects to upsert.

    Returns:
        tuple[bool, int]: A tuple containing:
                          - bool: True if the upsert was successful (or succeeded on retry),
                                  False otherwise.
                                  - int: The number of points successfully processed in the batch
                                 (0 if the operation failed).
    """
    if qdrant_client is None or models is None:
        raise ImportError(
            "Le paquet 'qdrant-client' est requis pour l'insertion Qdrant. Installez-le via 'pip install qdrant-client'."
        ) from _qdrant_import_error

    try:
        # Utiliser wait=True pour s'assurer que l'opération est terminée avant de continuer
        operation_info = client.upsert(collection_name=collection_name, points=points_batch, wait=True)
        # print(f"Qdrant upsert result: {operation_info}") # Décommenter pour le débogage
        if operation_info.status == models.UpdateStatus.COMPLETED:
             return True, len(points_batch) # Succès, retourne le nombre de points dans le lot
        else:
             print(f"Avertissement: Statut d'upsert Qdrant inattendu: {operation_info.status}")
             return False, 0 # Échec partiel ou inconnu
    except Exception as e:
        print(f"Erreur lors de l'upsert par lot dans Qdrant: {e}")
        print("Nouvelle tentative dans 2 secondes...")
        time.sleep(2)
        try:
            operation_info_retry = client.upsert(collection_name=collection_name, points=points_batch, wait=True)
            if operation_info_retry.status == models.UpdateStatus.COMPLETED:
                print("Nouvelle tentative d'upsert Qdrant réussie.")
                return True, len(points_batch)
            else:
                print(f"Échec après nouvelle tentative d'upsert Qdrant. Statut: {operation_info_retry.status}")
                return False, 0
        except Exception as e_retry:
            print(f"Échec après nouvelle tentative d'upsert Qdrant: {e_retry}")
            return False, 0

def insert_to_qdrant(embeddings_json_file, collection_name, qdrant_url=None, qdrant_api_key=None):
    """Inserts embeddings from a JSON file into a Qdrant collection.

    Handles Qdrant client initialization, collection creation if it doesn't exist
    (determining vector size from the first valid chunk), reading embeddings from
    the JSON file, preparing Qdrant points, and batch upserting them.

    Args:
        embeddings_json_file (str): Path to the JSON file containing embedding data.
                                    Each item should be a chunk dictionary with 'id',
                                    'embedding', and other metadata.
        collection_name (str): The name of the Qdrant collection.
        qdrant_url (str, optional): The URL of the Qdrant instance. Required.
        qdrant_api_key (str, optional): The API key for Qdrant (if secured).
                                        Defaults to None.

    Returns:
        dict: ``{"status", "message", "inserted_count"}`` (+ optionnellement
            ``"skipped_count"``/``"journal_path"`` si la dédup est active). ``status``
            ∈ {"success", "success_partial_data", "error"}. Forme unifiée (Lot 0.a)
            avec ``insert_to_pinecone``.
    """
    if qdrant_client is None or models is None:
        raise ImportError(
            "Le paquet 'qdrant-client' est requis pour l'insertion Qdrant. Installez-le via 'pip install qdrant-client'."
        ) from _qdrant_import_error

    if not os.path.exists(embeddings_json_file):
        print(f"Le fichier {embeddings_json_file} n'existe pas.")
        return _vectordb_result("error", f"Le fichier {embeddings_json_file} n'existe pas.")

    if not qdrant_url:
        raise ValueError("Qdrant URL (qdrant_url) is required.")
    # qdrant_api_key can be None for local unsecured instances.

    client = None
    try:
        print(f"Connexion à Qdrant à l'URL: {qdrant_url}")
        client = qdrant_client.QdrantClient(
            url=qdrant_url, 
            api_key=qdrant_api_key # This can be None
        )
        # Vérifier la connexion en listant les collections (ou une autre opération légère)
        client.get_collections() 
        print("Connexion à Qdrant réussie.")

    except Exception as e:
        print(f"Erreur lors de la connexion à Qdrant: {e}")
        traceback.print_exc()
        if client: client.close()
        return _vectordb_result("error", f"Erreur lors de la connexion à Qdrant: {e}")

    # Vérifier si la collection existe, la créer si nécessaire
    try:
        collection_info = client.get_collection(collection_name=collection_name)
        print(f"La collection '{collection_name}' existe déjà.")
        # Idéalement, vérifier si la dimension du vecteur correspond, mais nécessite de connaître la dimension attendue.
        # vector_size_expected = 1536 # Exemple: à adapter selon le modèle d'embedding utilisé
        # if collection_info.vectors_config.params.size != vector_size_expected:
        #     print(f"Erreur: La dimension des vecteurs de la collection ({collection_info.vectors_config.params.size}) ne correspond pas à la dimension attendue ({vector_size_expected}).")
        #     client.close()
        #     return 0

    except Exception as e:
        # Supposer que l'erreur signifie que la collection n'existe pas (à affiner si nécessaire)
        print(f"La collection '{collection_name}' n'existe pas ou erreur lors de la récupération: {e}. Tentative de création...")
        try:
            # Déterminer la taille du vecteur à partir du premier chunk valide
            vector_size = None
            temp_chunks = []
            with open(embeddings_json_file, 'r', encoding='utf-8') as f_temp:
                 temp_chunks = json.load(f_temp)
            for chunk in temp_chunks:
                if chunk.get("embedding") is not None:
                    vector_size = len(chunk["embedding"])
                    break
            
            if vector_size is None:
                 print("Erreur: Impossible de déterminer la taille du vecteur à partir du fichier JSON.")
                 if client: client.close()
                 return _vectordb_result("error", "Impossible de déterminer la taille du vecteur depuis le JSON.")

            print(f"Création de la collection '{collection_name}' avec des vecteurs de taille {vector_size} et distance Cosine.")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
                # Ajouter ici la configuration pour les vecteurs sparse si nécessaire
                # sparse_vectors_config={...} 
            )
            print(f"Collection '{collection_name}' créée avec succès.")
        except Exception as e_create:
            print(f"Erreur lors de la création de la collection '{collection_name}': {e_create}")
            traceback.print_exc()
            if client: client.close()
            return _vectordb_result("error", f"Erreur lors de la création de la collection '{collection_name}': {e_create}")

    # Charger les chunks avec embeddings
    print(f"Chargement des embeddings depuis {embeddings_json_file}")
    try:
        with open(embeddings_json_file, 'r', encoding='utf-8') as f:
            all_chunks = json.load(f)
    except Exception as e:
        print(f"Erreur lors du chargement du fichier {embeddings_json_file}: {e}")
        traceback.print_exc()
        if client: client.close()
        return _vectordb_result("error", f"Erreur lors du chargement du fichier {embeddings_json_file}: {e}")

    print(f"Chargement de {len(all_chunks)} chunks avec embeddings")

    # Déduplication (Lot 2/3) avant la boucle d'insertion. No-op si DEDUP_ENABLED=0.
    _dedup_adapter = _QdrantDedupAdapter(client, collection_name)
    all_chunks, dedup_skipped, dedup_journal = _run_dedup(
        all_chunks, _dedup_adapter, embeddings_json_file,
        target_desc=collection_name,
    )

    total_inserted_count = 0
    total_processed_chunks = 0

    # Traiter les chunks par lots
    for i in tqdm(range(0, len(all_chunks), QDRANT_BATCH_SIZE), desc=f"Insertion dans Qdrant collection '{collection_name}'"):
        batch_chunks = all_chunks[i:i+QDRANT_BATCH_SIZE]
        points_to_upsert = prepare_points_for_qdrant(batch_chunks)
        total_processed_chunks += len(batch_chunks) 
        
        if points_to_upsert:
            success, count_in_batch = upsert_batch_to_qdrant(client, collection_name, points_to_upsert)
            if success:
                total_inserted_count += count_in_batch
                # print(f"Lot {i//QDRANT_BATCH_SIZE + 1}: {count_in_batch} points insérés/mis à jour avec succès.")
            else:
                print(f"Lot {i//QDRANT_BATCH_SIZE + 1}: Échec partiel ou total de l'insertion du lot.")
        elif batch_chunks: 
             print(f"Lot {i//QDRANT_BATCH_SIZE + 1}: Aucun point valide à insérer.")

    print(f"\nInsertion Qdrant terminée.")
    print(f"Total de chunks traités (tentative de préparation): {total_processed_chunks}")
    print(f"Total de points effectivement insérés/mis à jour dans Qdrant: {total_inserted_count} (sur {len(all_chunks)} chunks initialement chargés si tous étaient valides).")

    if client: client.close()
    msg = f"Insertion Qdrant terminée (collection: {collection_name}). {total_inserted_count}/{len(all_chunks)} points insérés."
    if total_inserted_count == 0 and len(all_chunks) > 0:
        return _vectordb_result("error", msg + " Aucun point inséré.", inserted_count=0, skipped_count=dedup_skipped, journal_path=dedup_journal)
    if total_inserted_count < len(all_chunks):
        return _vectordb_result(
            "success_partial_data",
            msg + " Certains chunks sans embedding (ou en échec de lot) n'ont pas été insérés.",
            inserted_count=total_inserted_count, skipped_count=dedup_skipped, journal_path=dedup_journal,
        )
    return _vectordb_result("success", msg, inserted_count=total_inserted_count, skipped_count=dedup_skipped, journal_path=dedup_journal)


# ----------------------------------------------------------------------
# CLI Entry Point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload embeddings to vector databases (Pinecone, Weaviate, Qdrant)"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the JSON file containing embeddings"
    )
    parser.add_argument(
        "--db",
        choices=["pinecone", "weaviate", "qdrant"],
        required=True,
        help="Target vector database"
    )
    parser.add_argument(
        "--index",
        help="Pinecone index name (required for pinecone)"
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="Pinecone namespace (optional)"
    )
    parser.add_argument(
        "--class-name",
        default="Article",
        help="Weaviate class name (default: Article)"
    )
    parser.add_argument(
        "--tenant",
        default="default",
        help="Weaviate tenant name (default: default)"
    )
    parser.add_argument(
        "--collection",
        help="Qdrant collection name (required for qdrant)"
    )

    args = parser.parse_args()

    # Get credentials from environment
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    weaviate_url = os.getenv("WEAVIATE_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")

    print(f"=== rad_vectordb.py ===")
    print(f"Input file: {args.input}")
    print(f"Target DB: {args.db}")

    result = None

    if args.db == "pinecone":
        if not args.index:
            print("ERROR: --index is required for Pinecone")
            exit(1)
        if not pinecone_api_key:
            print("ERROR: PINECONE_API_KEY environment variable not set")
            exit(1)

        print(f"Index: {args.index}")
        print(f"Namespace: {args.namespace or '(default)'}")
        print(f"API Key: {pinecone_api_key[:10]}...")

        result = insert_to_pinecone(
            embeddings_json_file=args.input,
            index_name=args.index,
            pinecone_api_key=pinecone_api_key,
            namespace=args.namespace
        )

    elif args.db == "weaviate":
        if not weaviate_url:
            print("ERROR: WEAVIATE_URL environment variable not set")
            exit(1)
        if not weaviate_api_key:
            print("ERROR: WEAVIATE_API_KEY environment variable not set")
            exit(1)

        print(f"URL: {weaviate_url}")
        print(f"Class: {args.class_name}")
        print(f"Tenant: {args.tenant}")

        result = insert_to_weaviate_hybrid(
            embeddings_json_file=args.input,
            url=weaviate_url,
            api_key=weaviate_api_key,
            class_name=args.class_name,
            tenant_name=args.tenant
        )

    elif args.db == "qdrant":
        if not args.collection:
            print("ERROR: --collection is required for Qdrant")
            exit(1)
        if not qdrant_url:
            print("ERROR: QDRANT_URL environment variable not set")
            exit(1)

        print(f"URL: {qdrant_url}")
        print(f"Collection: {args.collection}")

        result = insert_to_qdrant(
            embeddings_json_file=args.input,
            collection_name=args.collection,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key
        )

    # Print result
    print(f"\n=== Result ===")
    if isinstance(result, dict):
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Message: {result.get('message', '')}")
        # Ligne canonique 'Inserted: N' — ancre du parser app/routes/processing.py
        # (Lot 0.c). Émise pour les 3 bases depuis l'unification des retours (Lot 0.a).
        print(f"Inserted: {result.get('inserted_count', 0)}")
        # Marqueurs dédup (Lot 3) — toujours émis (0 par défaut) pour un format
        # stdout stable et parsable ; 'Dedup journal' seulement si un journal existe.
        print(f"Skipped (dedup): {result.get('skipped_count', 0)}")
        if result.get('journal_path'):
            print(f"Dedup journal: {result['journal_path']}")
        # success_partial_data = tous les chunks *valides* insérés, certains chunks
        # d'entrée n'avaient pas d'embedding → succès non-fatal. Seuls partial_error
        # (échec d'upsert réel) et error doivent faire échouer (exit 1).
        exit(0 if result.get('status') in ('success', 'success_partial_data') else 1)
    else:
        # Repli défensif : depuis le Lot 0.a, les 3 connecteurs renvoient un dict.
        print(f"Inserted: {result if isinstance(result, int) else 0}")
        exit(0 if result and result > 0 else 1)
