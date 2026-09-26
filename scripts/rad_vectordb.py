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


def _albert_root():
    """Racine d'import du paquet ``rad_albert`` : celle de ``rad_dedup``.

    ``scripts.rad_albert`` en contexte paquet (application, Celery, tests),
    ``rad_albert`` en CLI : les classes d'erreur levées par le client sont
    ainsi celles que ce module intercepte.
    """
    return "scripts.rad_albert" if rad_dedup.__name__.startswith("scripts.") else "rad_albert"


def _albert_config_from_env():
    """``AlbertConfig.from_env()`` en n'important que ``rad_albert.config`` (stdlib).

    Sert au contrôle de l'interrupteur maître ``ALBERT_ENABLED`` avant tout
    le reste : ni le client ni httpx ne sont chargés pour un refus.

    Raises:
        ValueError: configuration Albert invalide.
    """
    import importlib

    return importlib.import_module(_albert_root() + ".config").AlbertConfig.from_env()


def _albert_modules():
    """Importe paresseusement le paquet ``rad_albert`` (client, collections, config, errors).

    Même racine que ``rad_dedup`` (``_albert_root``). Rien n'est importé tant
    qu'Albert n'est pas sollicité (httpx compris).
    """
    import importlib
    from types import SimpleNamespace

    root = _albert_root()
    return SimpleNamespace(
        client=importlib.import_module(root + ".client"),
        collections=importlib.import_module(root + ".collections"),
        config=importlib.import_module(root + ".config"),
        errors=importlib.import_module(root + ".errors"),
    )


class _AlbertDedupAdapter:
    """Index en mémoire d'une collection Albert : dédup Tier 2 et idempotence.

    ``preload()`` liste les documents de la collection puis leurs chunks
    (``list_chunks`` paginé) et construit trois index : documents par nom,
    ``content_id`` connus par document (append-skip, toujours actif) et
    métadonnées par ``content_hash`` (Tier 2). Il est appelé par
    ``insert_to_albert`` **avant** ``_run_dedup``, hors de
    ``rad_dedup.dedup_filter`` (qui avale les exceptions de ``existing``) : un
    échec du préchargement fait échouer l'envoi bruyamment, sans aucune
    écriture. ``existing()`` répond ensuite sans appel HTTP. Pas de
    ``nearest`` : le Tier 3 est sauté.

    Cohérence (D18) : l'indexation côté serveur prend environ 2,4 s ; des
    chunks envoyés juste avant peuvent manquer à la liste. Les lignes du
    manifeste local (``merge_manifest``) comblent cette fenêtre, pour les
    documents toujours présents et pour les lignes de moins de
    ``_ALBERT_MANIFEST_TRUST_S`` secondes seulement : au-delà, la liste du
    serveur fait foi. Limite connue : un chunk supprimé seul côté serveur
    (``DELETE /v1/documents/{id}/chunks/{chunk_id}``) moins de
    ``_ALBERT_MANIFEST_TRUST_S`` secondes après son envoi n'est renvoyé qu'à
    une relance postérieure à cette fenêtre.

    Tier 2 : ``rad_dedup`` compare ``chunk.get(champ)`` ; pour les champs
    dérivés de la liste blanche (``item_key``, ``year``, ``doi``,
    ``filename``), ``insert_to_albert`` recopie d'abord la valeur dérivée sur
    ses copies des chunks (``_albert_align_dedup_fields``).
    """

    db_name = "albert"
    metric = None

    def __init__(self, client, collection_id, meta_fields=("title",), collections_module=None):
        """Mémorise le client, la collection cible et les ``DEDUP_META_FIELDS``.

        Args:
            client: ``AlbertClient`` déjà construit.
            collection_id: collection cible (entier).
            meta_fields: ``DEDUP_META_FIELDS`` effectifs (réhydratés par
                ``existing``).
            collections_module: module ``rad_albert.collections`` (défaut :
                import paresseux).
        """
        self._client = client
        self._collection_id = int(collection_id)
        self._meta_fields = tuple(meta_fields or ())
        self._col = collections_module if collections_module is not None else _albert_modules().collections
        self.documents = {}        # nom -> [ids de documents]
        self.document_names = {}   # id -> nom
        self._content_ids = {}     # nom -> set(content_id)
        self._by_hash = {}         # content_hash -> [métadonnées]
        self.preloaded = False
        self.chunk_count = 0

    def preload(self, document_names=None):
        """Charge les documents de la collection et les chunks de ceux retenus.

        Args:
            document_names: noms des documents dont les chunks sont listés
                (``None`` : tous, nécessaire au Tier 2 inter-documents quand la
                dédup est active).

        Returns:
            int: nombre de chunks indexés.

        Raises:
            AlbertError: toute erreur de l'API (jamais avalée ici).
        """
        wanted = None if document_names is None else set(document_names)
        docs = self._client.list_documents(self._collection_id)
        targets = []
        for doc in docs or []:
            if not isinstance(doc, dict) or doc.get("id") is None:
                continue
            did = int(doc["id"])
            name = doc.get("name")
            self.documents.setdefault(name, []).append(did)
            self.document_names[did] = name
            if wanted is None or name in wanted:
                targets.append((did, name))
        for did, name in targets:
            for chunk in self._client.list_chunks(did) or []:
                if isinstance(chunk, dict):
                    self._index_chunk(did, name, chunk)
        self.preloaded = True
        return self.chunk_count

    def _index_chunk(self, document_id, document_name, chunk):
        """Ajoute un chunk listé aux index ``content_id`` et ``content_hash``."""
        meta = dict(chunk.get("metadata") or {})
        cid = meta.get("content_id")
        if cid:
            self._content_ids.setdefault(document_name, set()).add(str(cid))
        chash = meta.get("content_hash")
        if chash:
            entry = dict(meta)
            entry["id"] = f"{document_id}:{chunk.get('id')}"
            entry.setdefault("content_hash", chash)
            entry["document_id"] = document_id
            entry["document_name"] = document_name
            self._by_hash.setdefault(str(chash), []).append(entry)
        self.chunk_count += 1

    def merge_manifest(self, rows, *, now=None, max_age_s=None):
        """Ajoute les ``content_id`` des lignes de manifeste récentes de cette collection.

        Seules comptent les lignes dont le document existe toujours (un
        document supprimé, par rollback ou par l'utilisateur, est ignoré) et
        dont l'âge ne dépasse pas la fenêtre d'indexation (au-delà, ou sans
        horodatage lisible, la liste du serveur fait foi).

        Args:
            rows: lignes lues par ``rad_albert.collections.read_manifest``.
            now: instant de référence (``datetime`` ; défaut : maintenant, UTC).
            max_age_s: âge maximal d'une ligne prise en compte (défaut :
                ``_ALBERT_MANIFEST_TRUST_S``).

        Returns:
            int: nombre de ``content_id`` ajoutés.
        """
        window = _ALBERT_MANIFEST_TRUST_S if max_age_s is None else float(max_age_s)
        added = 0
        for row in rows or []:
            if not isinstance(row, dict) or "event" in row:
                continue
            try:
                if int(row.get("collection_id")) != self._collection_id:
                    continue
                did = int(row.get("document_id"))
            except (TypeError, ValueError):
                continue
            name = self.document_names.get(did)
            if name is None:
                continue
            age = self._col.manifest_row_age_s(row, now=now)
            if age is None or age > window:
                continue
            known = self._content_ids.setdefault(name, set())
            for cid in row.get("content_ids") or []:
                if cid and str(cid) not in known:
                    known.add(str(cid))
                    added += 1
        return added

    def duplicate_names(self, names):
        """Noms (parmi ``names``) portés par plusieurs documents : ``{nom: [ids]}``."""
        return {name: list(self.documents[name]) for name in dict.fromkeys(names)
                if len(self.documents.get(name, [])) > 1}

    def document_id(self, name):
        """Id du document existant de ce nom, ou ``None``."""
        ids = self.documents.get(name) or []
        return ids[0] if len(ids) == 1 else None

    def known_content_ids(self, name):
        """``content_id`` déjà présents dans le document ``name`` (ensemble, éventuellement vide)."""
        return self._content_ids.get(name, set())

    def existing(self, chunks_batch):
        """Tier 2 en mémoire : ``{content_hash: [meta, ...]}`` pour les chunks déjà présents.

        Les ``DEDUP_META_FIELDS`` stockés ont été assainis à l'envoi (titre
        tronqué à 255 caractères…) : quand la valeur stockée égale la forme
        assainie de celle du chunk, la valeur d'origine du chunk est
        réhydratée pour que la corroboration de ``rad_dedup`` compare des
        valeurs comparables. Aucun appel HTTP.
        """
        out = {}
        for chunk in chunks_batch:
            chash = chunk.get("content_hash")
            metas = self._by_hash.get(str(chash)) if chash else None
            if not metas:
                continue
            rows = []
            for meta in metas:
                row = dict(meta)
                for field in self._meta_fields:
                    stored = row.get(field)
                    if stored is None:
                        continue
                    raw = self._col.field_value(chunk, field)
                    if raw is not None and self._col.albert_scalar(raw) == stored:
                        row[field] = raw
                rows.append(row)
            out[chash] = rows
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


## BASE VECTORIELLE Albert (collections DINUM)

_ALBERT_RETENTION_NOTICE = (
    "Acquittement de rétention requis : les textes et métadonnées envoyés à une "
    "collection Albert sont stockés par la DINUM (sous-traitant, art. 28 RGPD) "
    "jusqu'à leur suppression. Relancer avec l'acquittement explicite "
    "(--albert-ack-retention en ligne de commande)."
)
"""Message de refus d'un envoi Albert sans acquittement de rétention."""

_ALBERT_DISABLED_MESSAGE = (
    "Albert est désactivé sur ce serveur (ALBERT_ENABLED=0) : cible albert indisponible"
)
"""Message de refus d'un envoi Albert quand l'interrupteur maître est coupé."""

_ALBERT_RECONCILE_WAITS_S = (3.0, 3.0, 4.0)
"""Attentes (s) avant chacune des relectures qui rapprochent une tranche à
l'issue incertaine : la première couvre l'indexation mesurée (D18 : environ
2,4 s pour un petit envoi), le total (10 s) l'attente retenue par les tests
live (D18 ``live_test_index_wait_s``), une tranche de 64 chunks pouvant être
plus lente à indexer."""

_ALBERT_MANIFEST_TRUST_S = 600.0
"""Âge maximal (s) d'une ligne de manifeste prise en compte par le
préchargement : au-delà, la liste des chunks du serveur fait foi."""


def _albert_result(status, message, inserted_count=0, skipped_count=0, journal_path=None,
                   existing_count=0, manifest_path=None, credential_required=None):
    """Retour de ``insert_to_albert`` : forme unifiée de ``_vectordb_result``
    enrichie de ``existing_count`` (chunks déjà présents, non renvoyés) et de
    ``manifest_path`` (manifeste écrit pendant ce run, sinon ``None``) ;
    ``credential_required`` est ajouté pour une erreur de compte."""
    res = _vectordb_result(status, message, inserted_count=inserted_count,
                           skipped_count=skipped_count, journal_path=journal_path)
    res["existing_count"] = int(existing_count)
    res["manifest_path"] = manifest_path
    if credential_required:
        res["credential_required"] = credential_required
    return res


def _albert_safe_text(value, api_key=None):
    """Texte d'erreur sans la clé (masquage de ``rad_albert.errors.redact`` si disponible)."""
    text = str(value)
    try:
        return _albert_modules().errors.redact(text, secrets=(api_key,) if api_key else (), limit=500)
    except Exception:
        if api_key and len(str(api_key)) >= 4:
            text = text.replace(str(api_key), "***")
        return text[:500]


class _AlbertPushRun:
    """État partagé d'un envoi Albert : compteurs, échecs, manifeste, arrêt du job.

    Thread-safe (``ALBERT_PUSH_CONCURRENCY`` > 1). Le manifeste est ouvert à la
    première tranche réussie, puis chaque ligne est écrite en append et
    flushée immédiatement : un envoi interrompu laisse un manifeste exact.
    """

    def __init__(self, manifest_path, collection_id, collection_name, collections_module, sleep=None):
        """Prépare l'état d'un envoi vers ``collection_id`` (manifeste à ``manifest_path``).

        ``sleep`` sert à l'attente d'indexation avant un rapprochement
        (défaut ``time.sleep``).
        """
        import threading

        self.sleep = sleep if sleep is not None else time.sleep
        self._lock = threading.Lock()
        self._col = collections_module
        self.manifest_path = manifest_path
        self.collection_id = int(collection_id)
        self.collection_name = collection_name
        self._fh = None
        self.manifest_written = False
        self.inserted = 0
        self.slices = 0
        self.failures = []
        self.notes = []
        self.account_error = None
        self.aborted = False

    def _append_row(self, row):
        """Écrit une ligne au manifeste (append + flush) ; verrou tenu par l'appelant.

        Un échec d'écriture arrête l'envoi : sans manifeste exact, la relance
        et l'audit ne sont plus fiables.
        """
        try:
            if self._fh is None:
                os.makedirs(os.path.dirname(os.path.abspath(self.manifest_path)), exist_ok=True)
                self._fh = open(self.manifest_path, "a", encoding="utf-8")
            self._fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            self._fh.flush()
            self.manifest_written = True
        except OSError as exc:
            self.failures.append(f"manifeste non écrit ({exc}) : envoi arrêté")
            self.aborted = True

    def record_slice(self, document_id, document_name, created, slice_index, content_ids,
                     chunk_ids=None, reconciled=False):
        """Compte une tranche réussie et l'écrit au manifeste (append + flush)."""
        row = self._col.manifest_row(
            collection_id=self.collection_id, collection_name=self.collection_name,
            document_id=document_id, document_name=document_name, document_created=created,
            slice_index=slice_index, content_ids=content_ids, chunk_ids=chunk_ids,
            reconciled=reconciled,
        )
        with self._lock:
            self.inserted += len(content_ids)
            self.slices += 1
            self._append_row(row)

    def rolled_back(self, document_id, document_name, pushed):
        """Retire du compteur les chunks d'un document supprimé par rollback.

        Quand des tranches de ce document figurent déjà au manifeste
        (``pushed`` > 0), une ligne d'événement ``rollback`` y est ajoutée
        (même chemin append + flush) : le relevé d'audit ne présente plus ces
        chunks comme stockés.
        """
        pushed = int(pushed)
        row = None
        if pushed > 0:
            row = self._col.manifest_rollback_row(
                collection_id=self.collection_id, collection_name=self.collection_name,
                document_id=document_id, document_name=document_name, chunks_removed=pushed,
            )
        with self._lock:
            self.inserted -= pushed
            if row is not None:
                self._append_row(row)

    def fail(self, message):
        """Enregistre l'échec d'un document (statut ``partial_error``)."""
        with self._lock:
            self.failures.append(message)

    def note(self, message):
        """Enregistre une remarque ajoutée au message final."""
        with self._lock:
            self.notes.append(message)

    def abort(self, error):
        """Erreur de compte ou de quota : arrêt du job entier (décision 22)."""
        with self._lock:
            if self.account_error is None:
                self.account_error = error
            self.aborted = True

    def close(self):
        """Ferme le manifeste."""
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


def _albert_resolve_collection(client, mods, collection_id, collection_name, create_collection):
    """Collection cible : par id (vérifiée privée) ou par nom exact (get-or-create privée).

    Une création à l'issue incertaine (``AlbertUncertainWriteError``) n'est
    jamais renvoyée : elle est rapprochée par une nouvelle liste exacte du nom
    (une correspondance : utilisée ; plusieurs : erreur listant les ids ;
    aucune : erreur, relancer).

    Returns:
        tuple: ``(collection_id, collection_name, created)``.

    Raises:
        AlbertTargetError: cible absente, non privée, de visibilité inconnue,
            ambiguë ou incohérente.
        AlbertError: erreur de l'API.
    """
    col = mods.collections
    if collection_id is not None:
        info = client.get_collection(collection_id)
        if not isinstance(info, dict):
            info = {}
        if col.visibility_of(info) is None:
            raise col.AlbertTargetError(
                f"La collection Albert {collection_id} : visibilité inconnue, envoi refusé "
                "(RAGpy n'envoie que vers des collections dont la visibilité « private » est "
                "confirmée).", ids=[collection_id])
        if not col.is_private(info):
            raise col.AlbertTargetError(
                f"La collection Albert {collection_id} n'est pas privée : envoi refusé "
                "(RAGpy n'envoie que vers des collections privées).", ids=[collection_id])
        actual_name = info.get("name")
        if collection_name and actual_name is not None and actual_name != collection_name:
            raise col.AlbertTargetError(
                f"La collection Albert {collection_id} s'appelle « {actual_name} » et non "
                f"« {collection_name} » : envoi refusé.", ids=[collection_id])
        return int(info.get("id", collection_id)), actual_name or collection_name, False

    found = col.select_collection(client.list_collections(name=collection_name), collection_name)
    if found is not None:
        return int(found["id"]), collection_name, False
    if not create_collection:
        raise col.AlbertTargetError(
            f"Collection Albert privée « {collection_name} » introuvable ; autoriser sa "
            "création (--albert-create-collection) ou donner son id.")
    try:
        new_id = client.create_collection(collection_name, visibility="private")
        print(f"Collection Albert privée « {collection_name} » créée (id {new_id}).")
        return int(new_id), collection_name, True
    except mods.errors.AlbertUncertainWriteError:
        found = col.select_collection(client.list_collections(name=collection_name), collection_name)
        if found is None:
            raise col.AlbertTargetError(
                f"Création de la collection « {collection_name} » à l'issue incertaine, "
                "introuvable après rapprochement : relancer (l'envoi est idempotent).")
        print(f"Collection Albert « {collection_name} » : création incertaine rapprochée (id {found['id']}).")
        return int(found["id"]), collection_name, True


def _albert_rollback(client, mods, state, document_id, document_name, pushed):
    """Supprime un document créé pendant ce run après l'échec d'une de ses tranches.

    Une suppression réussie est inscrite au manifeste (ligne ``rollback``)
    dès que des tranches de ce document y figuraient.
    """
    try:
        client.delete_document(document_id)
        state.rolled_back(document_id, document_name, pushed)
        print(f"  Rollback : document {document_id} (« {document_name} ») supprimé "
              f"({pushed} chunk(s) retiré(s)).")
    except mods.errors.AlbertError as exc:
        state.note(f"document {document_id} (« {document_name} ») non supprimé après échec "
                   f"({exc}) : à supprimer manuellement")


def _albert_reconcile_slice(client, errors, state, document_id, content_ids):
    """Rapproche par relecture une tranche à l'issue incertaine d'un document préexistant.

    Avant chaque relecture des chunks du document, attend l'un des délais de
    ``_ALBERT_RECONCILE_WAITS_S`` (hors sémaphore). Un chunk visible prouve la
    tranche entière (envoi atomique, D16). Rien n'est jamais renvoyé.

    Returns:
        tuple: ``(verdict, detail)`` ; ``verdict`` vaut ``"present"``,
        ``"absent"`` (aucun chunk visible après toutes les relectures),
        ``"unreadable"`` (relecture en échec : ``detail`` porte l'erreur) ou
        ``"aborted"`` (erreur de compte : job arrêté via ``state.abort``).
    """
    wanted = {str(cid) for cid in content_ids}
    for wait in _ALBERT_RECONCILE_WAITS_S:
        state.sleep(wait)  # hors sémaphore : laisser l'indexation (D18)
        try:
            listed = client.list_chunks(document_id)
        except errors.AlbertAuthError as exc:
            state.abort(exc)
            return "aborted", None
        except errors.AlbertError as exc:
            return "unreadable", exc
        present = {str((c.get("metadata") or {}).get("content_id")) for c in listed or [] if isinstance(c, dict)}
        if present & wanted:
            return "present", None
    return "absent", None


def _albert_push_document(client, mods, adapter, state, document_name, entries, fields, semaphore):
    """Crée (si besoin) un document puis y envoie ses chunks par tranches de 64.

    Un échec de tranche d'un document créé pendant ce run déclenche
    ``delete_document`` (rollback) ; sur un document préexistant, une issue
    incertaine est rapprochée par relecture des chunks (``content_id``),
    jamais renvoyée. Une erreur de compte ou de quota arrête le job entier.
    """
    errors = mods.errors
    col = mods.collections
    if state.aborted:
        return
    did = adapter.document_id(document_name)
    created = False
    if did is None:
        try:
            did = client.create_document(state.collection_id, document_name)
            created = True
        except errors.AlbertUncertainWriteError:
            found = client.list_documents(state.collection_id, name=document_name)
            if len(found) == 1:
                did, created = int(found[0]["id"]), True
            elif found:
                ids = ", ".join(str(d.get("id")) for d in found)
                state.fail(f"« {document_name} » : plusieurs documents de ce nom après une création "
                           f"incertaine (ids : {ids}) ; à vérifier")
                return
            else:
                state.fail(f"« {document_name} » : création du document à l'issue incertaine, "
                           "introuvable après rapprochement ; relancer (idempotent)")
                return
        except errors.AlbertAuthError as exc:
            state.abort(exc)
            return
        except errors.AlbertError as exc:
            state.fail(f"« {document_name} » : création du document impossible ({exc})")
            return

    pushed = 0
    for slice_index, part in enumerate(col.iter_slices(entries), start=1):
        if state.aborted:
            if created:
                _albert_rollback(client, mods, state, did, document_name, pushed)
            return
        content_ids = [cid for cid, _chunk in part]
        payload = [
            {"content": col.chunk_text(chunk), "metadata": col.sanitize_metadata(chunk, fields, content_id=cid)}
            for cid, chunk in part
        ]
        try:
            chunk_ids = client.add_chunks(did, payload, semaphore=semaphore)
        except errors.AlbertAuthError as exc:
            state.abort(exc)
            if created:
                _albert_rollback(client, mods, state, did, document_name, pushed)
            return
        except errors.AlbertUncertainWriteError as exc:
            if created:
                state.fail(f"« {document_name} » : tranche {slice_index} à l'issue incertaine ({exc})")
                _albert_rollback(client, mods, state, did, document_name, pushed)
                return
            verdict, detail = _albert_reconcile_slice(client, errors, state, did, content_ids)
            if verdict == "aborted":
                return
            if verdict == "present":
                # Envoi atomique (D16) : un chunk visible prouve la tranche entière.
                state.record_slice(did, document_name, created, slice_index, content_ids, reconciled=True)
                pushed += len(content_ids)
                continue
            if verdict == "unreadable":
                state.fail(f"« {document_name} » : tranche {slice_index} à l'issue incertaine, "
                           f"relecture impossible ({detail}) : issue inconnue, relancer "
                           "(l'envoi est idempotent)")
                return
            waited = sum(_ALBERT_RECONCILE_WAITS_S)
            state.fail(f"« {document_name} » : tranche {slice_index} à l'issue incertaine, absente "
                       f"après {len(_ALBERT_RECONCILE_WAITS_S)} relecture(s) sur {waited:g} s ; "
                       "relancer (l'envoi est idempotent)")
            return
        except errors.AlbertError as exc:
            state.fail(f"« {document_name} » : échec de la tranche {slice_index} ({exc})")
            if created:
                _albert_rollback(client, mods, state, did, document_name, pushed)
            return
        state.record_slice(did, document_name, created, slice_index, content_ids, chunk_ids=chunk_ids)
        pushed += len(content_ids)
    origin = "créé" if created else "existant"
    print(f"  Document {did} ({origin}) « {document_name} » : {pushed} chunk(s) envoyé(s).")


def insert_to_albert(embeddings_json_file, collection_id=None, collection_name=None, albert_api_key=None, *,
                     create_collection=False, ack_retention=False, transport=None, sleep=None):
    """Envoie les chunks d'un fichier JSON vers une collection Albert privée (DINUM).

    Les embeddings sont calculés côté serveur : ``embedding`` et
    ``sparse_embedding`` sont retirés et seuls le texte et une liste blanche
    de métadonnées scalaires sont envoyés. Étapes :

    1. refus avant tout appel réseau : Albert désactivé (``ALBERT_ENABLED=0``,
       contrôlé en premier) ou mal configuré, sans ``ack_retention``
       (rétention des textes jusqu'à suppression), sans clé, sans cible, ou
       avec une ``ALBERT_METADATA_FIELDS`` invalide ;
    2. collection par id (vérifiée privée) ou par nom exact (get-or-create
       d'une collection privée ; plusieurs homonymes = erreur listant les ids) ;
    3. ``_AlbertDedupAdapter.preload()`` (échec bruyant), puis ``_run_dedup``
       (Tier 2 en mémoire, pas de Tier 3), puis append-skip par
       ``content_id_for`` (toujours actif : une relance n'envoie rien de déjà
       présent) ;
    4. un document par ``document_name_for``, chunks envoyés par tranches de 64
       (``add_chunks``) sous ``ALBERT_PUSH_CONCURRENCY`` et le limiteur du rôle
       ``embed`` (D19) ; une ligne de manifeste par tranche réussie (append +
       flush, ids de chunks D16) ; échec d'une tranche d'un document créé
       pendant ce run : ``delete_document`` (ligne ``rollback`` au
       manifeste) puis ``partial_error`` ;
    5. une erreur de compte ou de quota arrête le job entier ;
    6. si Albert a été appelé, ``albert_usage.jsonl`` est écrit à côté du
       fichier d'entrée (sauf ``ALBERT_USAGE_LOG=0``) et la ligne de synthèse
       du ledger est affichée (décision 8).

    Les créations à l'issue incertaine ne sont jamais renvoyées : elles sont
    rapprochées par relecture. La fonction ne lève jamais : toute erreur est
    renvoyée sous forme de dict.

    Args:
        embeddings_json_file (str): fichier JSON des chunks (sparse, dense ou
            ``output_chunks.json`` : aucun vecteur n'est requis).
        collection_id (int, optional): collection cible (prioritaire sur le nom).
        collection_name (str, optional): nom exact de la collection cible.
        albert_api_key (str, optional): clé Bearer (jamais lue ici dans l'env).
        create_collection (bool): crée la collection privée si le nom est absent.
        ack_retention (bool): acquittement explicite de la rétention (requis).
        transport: transport httpx injecté (``httpx.MockTransport`` en test).
        sleep: fonction de sommeil des réessais (défaut ``time.sleep``).

    Returns:
        dict: ``{"status", "message", "inserted_count", "existing_count",
        "manifest_path"}`` (+ ``skipped_count`` / ``journal_path`` si la dédup a
        refusé des chunks, + ``credential_required`` pour une erreur de compte).
        ``status`` ∈ {"success", "success_partial_data", "partial_error", "error"}.
    """
    try:
        return _insert_to_albert(
            embeddings_json_file, collection_id, collection_name, albert_api_key,
            create_collection=create_collection, ack_retention=ack_retention,
            transport=transport, sleep=sleep,
        )
    except Exception as exc:  # contrat : jamais d'exception vers l'appelant
        msg = f"Erreur inattendue de l'envoi Albert : {type(exc).__name__} : {_albert_safe_text(exc, albert_api_key)}"
        print(msg)
        return _albert_result("error", msg)


def _insert_to_albert(embeddings_json_file, collection_id, collection_name, albert_api_key, *,
                      create_collection, ack_retention, transport, sleep):
    """Implémentation de ``insert_to_albert`` (peut lever ; enveloppée par l'appelant)."""
    # Interrupteur maître d'abord (« ALBERT_ENABLED=0 : ni appel ») : aucun
    # acquittement n'est demandé pour une cible indisponible.
    try:
        cfg = _albert_config_from_env()
    except ValueError as exc:
        print(str(exc))
        return _albert_result("error", str(exc))
    if not cfg.enabled:
        print(_ALBERT_DISABLED_MESSAGE)
        return _albert_result("error", _ALBERT_DISABLED_MESSAGE)
    if not ack_retention:
        print(_ALBERT_RETENTION_NOTICE)
        return _albert_result("error", _ALBERT_RETENTION_NOTICE)
    if not os.path.exists(embeddings_json_file):
        msg = f"Le fichier {embeddings_json_file} n'existe pas."
        print(msg)
        return _albert_result("error", msg)
    if not isinstance(albert_api_key, str) or not albert_api_key.strip():
        msg = "Clé API Albert (DINUM) requise. Configurez-la dans Paramètres > Mes Identifiants."
        print(msg)
        return _albert_result("error", msg, credential_required="albert_api_key")
    name = collection_name.strip() if isinstance(collection_name, str) and collection_name.strip() else None
    if collection_id is None and name is None:
        msg = "Collection Albert cible requise : id (--albert-collection-id) ou nom (--albert-collection-name)."
        print(msg)
        return _albert_result("error", msg)

    mods = _albert_modules()
    col = mods.collections
    errors = mods.errors
    try:
        fields = col.effective_metadata_fields()
    except ValueError as exc:
        print(str(exc))
        return _albert_result("error", str(exc))
    dedup_cfg = rad_dedup.DedupConfig.from_env()

    try:
        with open(embeddings_json_file, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
    except Exception as exc:
        msg = f"Erreur lors du chargement du fichier {embeddings_json_file}: {exc}"
        print(msg)
        return _albert_result("error", msg)
    if not isinstance(all_chunks, list):
        msg = f"Le fichier {embeddings_json_file} ne contient pas une liste de chunks."
        print(msg)
        return _albert_result("error", msg)
    print(f"Chargement de {len(all_chunks)} chunks depuis {embeddings_json_file}")

    prepared = []
    invalid = 0
    for raw in all_chunks:
        if not isinstance(raw, dict) or not col.chunk_text(raw).strip():
            invalid += 1
            continue
        prepared.append(col.strip_vectors(raw))
    if not prepared:
        if all_chunks:
            msg = "Aucun chunk exploitable (texte vide) : rien n'a été envoyé à Albert."
            print(msg)
            return _albert_result("error", msg)
        msg = "Aucun chunk à envoyer à Albert."
        print(msg)
        return _albert_result("success", msg)

    client = mods.client.AlbertClient(cfg, albert_api_key, transport=transport,
                                      sleep=sleep if sleep is not None else time.sleep)
    try:
        return _albert_upload(client, mods, cfg, fields, dedup_cfg, prepared, invalid,
                              embeddings_json_file, collection_id, name, create_collection, sleep=sleep)
    except errors.AlbertAuthError as exc:
        msg = f"Envoi Albert arrêté : {exc}"
        print(msg)
        return _albert_result("error", msg, credential_required=getattr(exc, "credential_required", "albert_api_key"))
    except (errors.AlbertError, col.AlbertTargetError) as exc:
        msg = f"Envoi Albert impossible : {exc}"
        print(msg)
        return _albert_result("error", msg)
    finally:
        try:
            _albert_report_usage(client, cfg, embeddings_json_file)
        except Exception as exc:  # le compte rendu d'usage ne masque jamais le résultat de l'envoi
            print(f"Avertissement : synthèse d'usage Albert indisponible ({type(exc).__name__}).")
        client.close()


def _albert_report_usage(client, cfg, embeddings_json_file):
    """Écrit ``albert_usage.jsonl`` et affiche la synthèse, seulement si Albert a été appelé.

    Décision 8 : les envois de chunks consomment le quota bge-m3 (D19) et
    sont inscrits au ledger du client (rôle ``push``) ; les appels de gestion
    (collections, documents, listes) ne le sont pas. Le journal est écrit à
    côté du fichier d'entrée (même dossier que le manifeste), sauf
    ``ALBERT_USAGE_LOG=0`` ; la synthèse s'affiche avant le bloc Result.

    Args:
        client: ``AlbertClient`` de l'envoi.
        cfg: ``AlbertConfig`` effective.
        embeddings_json_file: fichier d'entrée de l'envoi.

    Returns:
        Chemin du journal écrit, ou ``None`` (Albert non appelé, journal
        désactivé ou erreur d'écriture).
    """
    ledger = getattr(client, "ledger", None)
    if ledger is None or not ledger.called:
        return None
    path = None
    if getattr(cfg, "usage_log", True):
        try:
            path = ledger.write_jsonl(os.path.dirname(os.path.abspath(embeddings_json_file)))
        except OSError as exc:
            print(f"Avertissement : journal d'usage Albert non écrit ({exc}).")
    print(ledger.summary_line())
    if path:
        print(f"Journal d'usage Albert : {path}")
    return path


def _albert_align_dedup_fields(chunks, meta_fields, col):
    """Recopie sur les copies des chunks la valeur dérivée de chaque ``DEDUP_META_FIELDS``.

    ``rad_dedup`` corrobore un refus Tier 2 en comparant ``chunk.get(champ)``
    à la métadonnée stockée. Or plusieurs champs de la liste blanche sont
    dérivés (``item_key`` ← ``itemKey``, ``year`` ← ``date``, ``doi`` ←
    ``url``, ``filename`` ← nom de fichier seul) et n'existent pas tels quels
    dans les chunks : sans cet alignement, la corroboration échouerait
    toujours et les doublons partiraient sans être signalés. Seules les copies
    préparées (``strip_vectors``) sont modifiées, jamais le fichier d'entrée ;
    l'envoi est inchangé (``sanitize_metadata`` et ``document_name_for``
    relisent les mêmes valeurs).

    Args:
        chunks: copies préparées des chunks (modifiées en place).
        meta_fields: ``DEDUP_META_FIELDS`` effectifs.
        col: module ``rad_albert.collections``.

    Returns:
        int: nombre de valeurs recopiées.
    """
    changed = 0
    for chunk in chunks:
        for name in meta_fields:
            value = col.field_value(chunk, name)
            if value is not None and chunk.get(name) != value:
                chunk[name] = value
                changed += 1
    return changed


def _albert_upload(client, mods, cfg, fields, dedup_cfg, prepared, invalid, embeddings_json_file,
                   collection_id, collection_name, create_collection, sleep=None):
    """Collection, préchargement, dédup, append-skip puis envoi ; renvoie le dict final."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    col = mods.collections
    errors = mods.errors
    cid, cname, _created = _albert_resolve_collection(
        client, mods, collection_id, collection_name, create_collection)
    label = cname or str(cid)
    print(f"Collection Albert cible : « {label} » (id {cid}, privée).")

    run_names = list(dict.fromkeys(col.document_name_for(c) for c in prepared))
    adapter = _AlbertDedupAdapter(client, cid, meta_fields=dedup_cfg.meta_fields, collections_module=col)
    try:
        adapter.preload(document_names=None if dedup_cfg.enabled else run_names)
    except errors.AlbertAuthError:
        raise
    except Exception as exc:
        msg = (f"Échec du préchargement de la collection Albert {cid} : "
               f"{_albert_safe_text(exc)} ; envoi annulé (aucun chunk envoyé).")
        print(msg)
        return _albert_result("error", msg)
    duplicates = adapter.duplicate_names(run_names)
    if duplicates:
        detail = " ; ".join(f"« {n} » (ids : {', '.join(str(i) for i in ids)})" for n, ids in duplicates.items())
        msg = f"Documents Albert en double dans la collection {cid} : {detail} ; envoi annulé."
        print(msg)
        return _albert_result("error", msg)
    manifest_path = col.manifest_path_for(os.path.dirname(os.path.abspath(embeddings_json_file)))
    adapter.merge_manifest(col.read_manifest(manifest_path))
    print(f"Préchargement : {len(adapter.document_names)} document(s), {adapter.chunk_count} chunk(s) indexé(s).")

    if dedup_cfg.enabled:
        _albert_align_dedup_fields(prepared, dedup_cfg.meta_fields, col)
    kept, dedup_skipped, dedup_journal = _run_dedup(
        prepared, adapter, embeddings_json_file, target_desc=f"albert/{label}")

    # existing_count : content_id déjà présents côté serveur (ou au manifeste
    # récent) ; repeated : répétitions d'un content_id dans ce fichier, pour un
    # même document (jamais passées par le serveur).
    existing_count = 0
    repeated = 0
    pending = {}
    seen = {}
    for chunk in kept:
        name = col.document_name_for(chunk)
        content_id = col.content_id_for(chunk)
        run_seen = seen.setdefault(name, set())
        if content_id in run_seen:
            repeated += 1
            continue
        run_seen.add(content_id)
        if content_id in adapter.known_content_ids(name):
            existing_count += 1
            continue
        pending.setdefault(name, []).append((content_id, chunk))
    if existing_count:
        print(f"Albert : {existing_count} chunk(s) déjà présent(s) dans la collection, non renvoyé(s).")
    if repeated:
        print(f"Albert : {repeated} répétition(s) d'un chunk dans le fichier d'entrée (même "
              "content_id, même document) ignorée(s) ; jamais comptée(s) comme déjà présente(s).")

    total_pending = sum(len(v) for v in pending.values())
    if total_pending == 0:
        msg = (f"Insertion Albert terminée (collection {label}). Aucun nouveau chunk : "
               f"{existing_count} déjà présent(s).")
        if repeated:
            msg += f" {repeated} répétition(s) interne(s) au fichier ignorée(s)."
        print(msg)
        status = "success_partial_data" if invalid else "success"
        return _albert_result(status, msg, inserted_count=0, skipped_count=dedup_skipped,
                              journal_path=dedup_journal, existing_count=existing_count)

    state = _AlbertPushRun(manifest_path, cid, cname, col, sleep=sleep)
    concurrency = max(1, int(getattr(cfg, "push_concurrency", 1) or 1))
    semaphore = threading.BoundedSemaphore(concurrency)
    items = list(pending.items())

    def push(item):
        """Envoie un document (nom, entrées) ; toute erreur est versée à l'état partagé."""
        try:
            _albert_push_document(client, mods, adapter, state, item[0], item[1], fields, semaphore)
        except errors.AlbertAuthError as exc:
            state.abort(exc)
        except Exception as exc:
            state.fail(f"« {item[0]} » : {type(exc).__name__} : {_albert_safe_text(exc)}")

    try:
        if concurrency == 1 or len(items) == 1:
            for item in items:
                if state.aborted:
                    break
                push(item)
        else:
            with ThreadPoolExecutor(max_workers=min(concurrency, len(items))) as pool:
                list(pool.map(push, items))
    finally:
        state.close()

    manifest = manifest_path if state.manifest_written else None
    parts = [f"Insertion Albert terminée (collection {label}).",
             f"{state.inserted}/{total_pending} chunk(s) envoyé(s) en {state.slices} tranche(s)."]
    if existing_count:
        parts.append(f"{existing_count} déjà présent(s).")
    if repeated:
        parts.append(f"{repeated} répétition(s) interne(s) au fichier ignorée(s).")
    if invalid:
        parts.append(f"{invalid} chunk(s) sans texte ignoré(s).")
    if state.failures:
        parts.append("Échecs : " + " ; ".join(state.failures) + ".")
    if state.notes:
        parts.append("Remarques : " + " ; ".join(state.notes) + ".")
    if state.account_error is not None:
        parts.append(f"Job arrêté : {state.account_error}")
    msg = " ".join(parts)
    print(msg)
    common = dict(inserted_count=state.inserted, skipped_count=dedup_skipped, journal_path=dedup_journal,
                  existing_count=existing_count, manifest_path=manifest)
    if state.account_error is not None:
        return _albert_result("error", msg, credential_required=getattr(
            state.account_error, "credential_required", "albert_api_key"), **common)
    if state.failures:
        return _albert_result("partial_error", msg, **common)
    if invalid:
        return _albert_result("success_partial_data", msg, **common)
    return _albert_result("success", msg, **common)


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
        choices=["pinecone", "weaviate", "qdrant", "albert"],
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
    # Options Albert : toutes préfixées --albert-, sans préfixe commun avec
    # --class(-name), --tenant, --index, --namespace ou --collection (les routes
    # passent l'abréviation --class).
    parser.add_argument(
        "--albert-collection-id",
        type=int,
        default=None,
        metavar="ID",
        help="Albert collection id (albert; takes precedence over the name)"
    )
    parser.add_argument(
        "--albert-collection-name",
        default=None,
        metavar="NAME",
        help="Exact name of the private Albert collection (albert)"
    )
    parser.add_argument(
        "--albert-create-collection",
        action="store_true",
        help="Create the private Albert collection if the name is not found (albert)"
    )
    parser.add_argument(
        "--albert-ack-retention",
        action="store_true",
        help="Acknowledge that texts are stored by DINUM until deleted (required for albert)"
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

    elif args.db == "albert":
        # Contrôles locaux d'abord (aucun appel réseau) : interrupteur maître
        # (exit 2 : aucun acquittement demandé pour une cible indisponible),
        # puis acquittement, cible et clé (lue dans l'env, jamais affichée).
        try:
            albert_enabled = _albert_config_from_env().enabled
        except ValueError as exc:
            print(f"ERROR: configuration Albert invalide : {exc}")
            exit(2)
        if not albert_enabled:
            print(f"ERROR: {_ALBERT_DISABLED_MESSAGE}")
            exit(2)
        if not args.albert_ack_retention:
            print(f"ERROR: {_ALBERT_RETENTION_NOTICE}")
            exit(1)
        if args.albert_collection_id is None and not (args.albert_collection_name or "").strip():
            print("ERROR: --albert-collection-id ou --albert-collection-name est requis pour Albert")
            exit(1)
        albert_api_key = os.getenv("ALBERT_API_KEY")
        if not albert_api_key:
            print("ERROR: ALBERT_API_KEY environment variable not set")
            exit(1)

        print(f"Collection id: {args.albert_collection_id if args.albert_collection_id is not None else '(par nom)'}")
        print(f"Collection name: {args.albert_collection_name or '(par id)'}")
        print(f"Create collection: {'yes' if args.albert_create_collection else 'no'}")

        result = insert_to_albert(
            embeddings_json_file=args.input,
            collection_id=args.albert_collection_id,
            collection_name=args.albert_collection_name,
            albert_api_key=albert_api_key,
            create_collection=args.albert_create_collection,
            ack_retention=True,
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
        if args.db == "albert":
            # Lignes propres à Albert (jamais émises pour les 3 autres bases) :
            # chunks déjà présents (idempotence) et manifeste d'envoi (ligne entière).
            print(f"Skipped (existing): {result.get('existing_count', 0)}")
            if result.get('manifest_path'):
                print(f"Albert manifest: {result['manifest_path']}")
            if result.get('credential_required'):
                # Erreur de compte ou de quota Albert : arrêt du job (décision 22).
                exit(2)
        # success_partial_data = tous les chunks *valides* insérés, certains chunks
        # d'entrée n'avaient pas d'embedding → succès non-fatal. Seuls partial_error
        # (échec d'upsert réel) et error doivent faire échouer (exit 1).
        exit(0 if result.get('status') in ('success', 'success_partial_data') else 1)
    else:
        # Repli défensif : depuis le Lot 0.a, les 3 connecteurs renvoient un dict.
        print(f"Inserted: {result if isinstance(result, int) else 0}")
        exit(0 if result and result > 0 else 1)
