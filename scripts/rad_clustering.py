#!/usr/bin/env python3
"""
rad_clustering.py - Document Clustering Agent
==============================================

Clusters documents based on their embeddings using UMAP dimensionality reduction
and HDBSCAN clustering. Generates Zotero-compatible tags for automatic organization.

This script is part of the RAGpy pipeline (Step 4.b) and processes the output
from the embedding generation phase.

Usage:
    python scripts/rad_clustering.py \
        --input uploads/session/output_chunks_with_embeddings_sparse.json \
        --output uploads/session \
        --session-name MaBiblio

Output:
    clustering_results.json with cluster assignments and Zotero tags

Author: RAGpy Team
Date: 2025-12-10
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

DEFAULT_UMAP_N_COMPONENTS = 100     # Target dimensions after UMAP reduction (100D preserves more semantic nuance)
DEFAULT_UMAP_N_NEIGHBORS = 15       # UMAP neighborhood size
DEFAULT_UMAP_MIN_DIST = 0.05        # UMAP minimum distance (smaller = tighter clusters)
DEFAULT_MIN_CLUSTER_SIZE = 3        # Minimum documents per cluster (user constraint)
DEFAULT_MIN_SAMPLES = 2             # HDBSCAN minimum samples
RANDOM_SEED = 42                    # For reproducibility


# =============================================================================
# DOCUMENT EMBEDDING AGGREGATION
# =============================================================================

def aggregate_chunk_embeddings(
    chunks: List[Dict],
    method: str = "mean"
) -> Dict[str, np.ndarray]:
    """
    Aggregate chunk embeddings to document-level embeddings.

    Since documents are split into multiple chunks during processing, we need
    to combine them back into a single embedding per document for clustering.

    Args:
        chunks: List of chunk dictionaries with 'doc_id' and 'embedding' fields.
                Each chunk represents a portion of a document.
        method: Aggregation method to combine chunk embeddings.
                - 'mean': Average of all chunk embeddings (default, recommended)
                - 'max': Element-wise maximum across chunks
                - 'first': Use only the first chunk's embedding

    Returns:
        Dictionary mapping doc_id to aggregated embedding vector (numpy array).
        The embedding dimension matches the input (typically 3072 for OpenAI).

    Example:
        >>> chunks = [
        ...     {"doc_id": "doc1", "chunk_index": 0, "embedding": [0.1, 0.2, 0.3]},
        ...     {"doc_id": "doc1", "chunk_index": 1, "embedding": [0.3, 0.4, 0.5]},
        ...     {"doc_id": "doc2", "chunk_index": 0, "embedding": [0.5, 0.6, 0.7]},
        ... ]
        >>> doc_embeddings = aggregate_chunk_embeddings(chunks, method="mean")
        >>> doc_embeddings["doc1"]  # array([0.2, 0.3, 0.4])

    Note:
        Mean pooling is recommended as it preserves information from all chunks
        and produces more stable representations than max or first-chunk methods.
    """
    # Group chunks by document ID
    doc_chunks: Dict[str, List[np.ndarray]] = defaultdict(list)

    for chunk in chunks:
        doc_id = chunk.get("doc_id")
        embedding = chunk.get("embedding")

        # Skip chunks with missing doc_id or embedding
        if not doc_id or embedding is None:
            continue

        # Convert to numpy array if needed
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)

        doc_chunks[doc_id].append(embedding)

    # Aggregate embeddings per document
    doc_embeddings: Dict[str, np.ndarray] = {}

    for doc_id, embeddings in doc_chunks.items():
        if not embeddings:
            continue

        # Stack all chunk embeddings for this document
        stacked = np.vstack(embeddings)

        if method == "mean":
            # Average across chunks (most common and recommended)
            doc_embeddings[doc_id] = np.mean(stacked, axis=0)
        elif method == "max":
            # Element-wise maximum (can capture salient features)
            doc_embeddings[doc_id] = np.max(stacked, axis=0)
        elif method == "first":
            # Use first chunk only (fastest, but loses information)
            doc_embeddings[doc_id] = embeddings[0]
        else:
            raise ValueError(f"Unknown aggregation method: {method}. Use 'mean', 'max', or 'first'.")

    logger.info(f"Aggregated {len(chunks)} chunks into {len(doc_embeddings)} document embeddings")
    return doc_embeddings


# =============================================================================
# DIMENSIONALITY REDUCTION
# =============================================================================

def reduce_dimensions_umap(
    embeddings: np.ndarray,
    n_components: int = DEFAULT_UMAP_N_COMPONENTS,
    n_neighbors: int = DEFAULT_UMAP_N_NEIGHBORS,
    min_dist: float = DEFAULT_UMAP_MIN_DIST,
    random_state: int = RANDOM_SEED
) -> np.ndarray:
    """
    Reduce embedding dimensions using UMAP (Uniform Manifold Approximation and Projection).

    UMAP is used to project high-dimensional embeddings (3072D from OpenAI) into a
    lower-dimensional space (50D by default) while preserving local and global structure.
    This makes clustering more effective and computationally efficient.

    Args:
        embeddings: Array of shape (n_documents, embedding_dim), typically (N, 3072).
        n_components: Target number of dimensions after reduction. Default 50 provides
                     a good balance between information preservation and clustering performance.
        n_neighbors: Number of neighbors to consider for each point. Higher values
                    preserve more global structure, lower values preserve local structure.
        min_dist: Minimum distance between points in the low-dimensional space.
                 Lower values create tighter clusters.
        random_state: Random seed for reproducibility.

    Returns:
        Reduced embeddings array of shape (n_documents, n_components).

    Raises:
        ImportError: If umap-learn is not installed.

    Note:
        Uses 'cosine' metric which is optimal for text embeddings as it measures
        the angle between vectors rather than their magnitude.
    """
    try:
        import umap
    except ImportError:
        raise ImportError(
            "umap-learn is required for dimensionality reduction. "
            "Install with: pip install umap-learn>=0.5.4"
        )

    n_samples = embeddings.shape[0]
    original_dim = embeddings.shape[1]

    logger.info(f"Reducing dimensions: {original_dim} -> {n_components} for {n_samples} documents")

    # Adjust n_neighbors if dataset is small
    effective_n_neighbors = min(n_neighbors, n_samples - 1)
    if effective_n_neighbors < n_neighbors:
        logger.warning(
            f"Adjusted n_neighbors from {n_neighbors} to {effective_n_neighbors} "
            f"due to small dataset size ({n_samples} samples)"
        )

    # Adjust n_components if needed (must be < n_samples - 1 for spectral init)
    # We use n_samples - 2 to be safe for UMAP's spectral initialization
    max_components = max(2, n_samples - 2)
    effective_n_components = min(n_components, max_components)
    if effective_n_components < n_components:
        logger.warning(
            f"Adjusted n_components from {n_components} to {effective_n_components} "
            f"due to small dataset size ({n_samples} samples)"
        )

    # Use random initialization for very small datasets to avoid spectral issues
    init_method = "random" if n_samples <= 20 else "spectral"
    if init_method == "random":
        logger.info(f"Using random initialization for small dataset ({n_samples} samples)")

    # Initialize UMAP reducer
    reducer = umap.UMAP(
        n_components=effective_n_components,
        n_neighbors=effective_n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric="cosine",  # Optimal for text embeddings
        n_jobs=1,         # Single thread with random_state for reproducibility
        low_memory=True,  # Memory-efficient for large datasets
        init=init_method  # Random init for small datasets, spectral otherwise
    )

    # Fit and transform
    reduced = reducer.fit_transform(embeddings)

    logger.info(f"UMAP reduction complete: {reduced.shape}")
    return reduced


# =============================================================================
# CLUSTERING
# =============================================================================

def cluster_documents_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int = DEFAULT_MIN_SAMPLES
) -> np.ndarray:
    """
    Cluster documents using HDBSCAN (Hierarchical Density-Based Spatial Clustering).

    HDBSCAN automatically determines the number of clusters based on data density,
    which is ideal for academic document collections where the number of topics
    is unknown a priori.

    Args:
        embeddings: Reduced embeddings array of shape (n_documents, n_dimensions).
                   Should be output from UMAP reduction for best results.
        min_cluster_size: Minimum number of documents required to form a cluster.
                         Default is 3 as per user requirements.
        min_samples: Number of samples in a neighborhood for a point to be a core point.
                    Lower values are more conservative (fewer outliers).

    Returns:
        Array of cluster labels. Label -1 indicates noise/outlier documents
        that don't belong to any cluster.

    Raises:
        ImportError: If hdbscan is not installed.

    Note:
        The 'eom' (Excess of Mass) cluster selection method is used, which tends
        to produce more natural cluster hierarchies for document collections.
    """
    try:
        import hdbscan
    except ImportError:
        raise ImportError(
            "hdbscan is required for clustering. "
            "Install with: pip install hdbscan>=0.8.33"
        )

    n_samples = embeddings.shape[0]

    logger.info(
        f"Running HDBSCAN clustering: {n_samples} documents, "
        f"min_cluster_size={min_cluster_size}, min_samples={min_samples}"
    )

    # Initialize HDBSCAN clusterer
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",              # Standard for UMAP-reduced space
        cluster_selection_method="leaf", # Leaf method finds more granular clusters
        core_dist_n_jobs=-1,             # Parallelize core distance computation
        prediction_data=True             # Enable soft clustering predictions
    )

    # Fit and predict cluster labels
    labels = clusterer.fit_predict(embeddings)

    # Calculate statistics
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = (labels == -1).sum()
    noise_ratio = n_noise / n_samples * 100

    logger.info(
        f"HDBSCAN complete: {n_clusters} clusters found, "
        f"{n_noise} noise documents ({noise_ratio:.1f}%)"
    )

    return labels


def calculate_optimal_min_cluster_size(
    n_documents: int,
    target_ratio: float = 10.0
) -> int:
    """
    Calculate an optimal min_cluster_size to achieve approximately N/10 clusters.

    This heuristic helps balance between too many small clusters and too few
    large clusters for typical academic document collections.

    Args:
        n_documents: Total number of documents to cluster.
        target_ratio: Target ratio of documents per cluster. Default 10 means
                     approximately N/10 clusters will be created.

    Returns:
        Recommended min_cluster_size for HDBSCAN. Always at least 3 (user constraint).

    Example:
        >>> calculate_optimal_min_cluster_size(100)  # ~10 clusters
        10
        >>> calculate_optimal_min_cluster_size(50)   # ~5 clusters
        10
        >>> calculate_optimal_min_cluster_size(20)   # Small dataset
        3
    """
    # Calculate target number of clusters
    target_clusters = max(2, int(n_documents / target_ratio))

    # min_cluster_size should allow for target_clusters
    # but never be less than 3 (user constraint)
    min_cluster_size = max(3, n_documents // target_clusters)

    # Cap at reasonable maximum to avoid too few clusters
    min_cluster_size = min(min_cluster_size, 15)

    logger.debug(
        f"Calculated min_cluster_size={min_cluster_size} "
        f"for {n_documents} documents (target ~{target_clusters} clusters)"
    )

    return min_cluster_size


# =============================================================================
# TAG GENERATION
# =============================================================================

def generate_cluster_tags(
    doc_ids: List[str],
    labels: np.ndarray,
    session_name: str
) -> Dict[str, str]:
    """
    Generate Zotero-compatible cluster tags for each document.

    Tags follow the format: _<SessionName>_<ClusterNumber>
    - Prefix '_' ensures tags sort together in Zotero
    - Cluster numbers are zero-padded: 01, 02, ... 99
    - Noise/outlier documents receive the tag '_<SessionName>_noise'

    Args:
        doc_ids: List of document IDs in the same order as labels.
        labels: Array of cluster labels from HDBSCAN. -1 indicates noise.
        session_name: Name to use in tags (e.g., 'MaBiblio'). Special characters
                     and spaces are automatically removed.

    Returns:
        Dictionary mapping doc_id to cluster tag string.

    Example:
        >>> doc_ids = ["doc1", "doc2", "doc3", "doc4"]
        >>> labels = np.array([0, 0, 1, -1])
        >>> tags = generate_cluster_tags(doc_ids, labels, "Ma Biblio 2024!")
        >>> tags
        {
            "doc1": "_MaBiblio2024_01",
            "doc2": "_MaBiblio2024_01",
            "doc3": "_MaBiblio2024_02",
            "doc4": "_MaBiblio2024_noise"
        }

    Note:
        Cluster numbers start at 01 (not 00) for human readability.
        Tags are designed to be easily filterable in Zotero's tag pane.
    """
    # Clean session name: keep only alphanumeric, dash, underscore
    # Truncate to 20 chars to keep tags manageable
    clean_name = "".join(
        c for c in session_name
        if c.isalnum() or c in "-_"
    )[:20]

    if not clean_name:
        clean_name = "Cluster"

    # Generate tag for each document
    doc_tags: Dict[str, str] = {}

    for doc_id, label in zip(doc_ids, labels):
        if label == -1:
            # Noise documents get special tag
            tag = f"_{clean_name}_noise"
        else:
            # Regular clusters: 1-indexed with zero-padding
            tag = f"_{clean_name}_{label + 1:02d}"

        doc_tags[doc_id] = tag

    # Log tag distribution
    tag_counts = defaultdict(int)
    for tag in doc_tags.values():
        tag_counts[tag] += 1

    logger.info(f"Generated {len(set(doc_tags.values()))} unique tags for {len(doc_ids)} documents")

    return doc_tags


# =============================================================================
# METADATA EXTRACTION
# =============================================================================

def extract_document_metadata(
    chunks: List[Dict],
    doc_ids: List[str]
) -> Dict[str, Dict]:
    """
    Extract metadata for each document from its chunks.

    Since metadata is duplicated across chunks of the same document,
    we only need to extract it once from any chunk.

    Args:
        chunks: List of all chunk dictionaries.
        doc_ids: List of document IDs to extract metadata for.

    Returns:
        Dictionary mapping doc_id to metadata dict containing:
        - title: Document title
        - authors: Author names
        - itemKey: Zotero item key (required for tag application)
        - date: Publication date
        - And other available fields

    Note:
        The itemKey field is critical for applying tags via Zotero API.
    """
    # Index chunks by doc_id for fast lookup
    doc_to_chunk: Dict[str, Dict] = {}

    for chunk in chunks:
        doc_id = chunk.get("doc_id")
        if doc_id and doc_id not in doc_to_chunk:
            doc_to_chunk[doc_id] = chunk

    # Extract metadata for requested documents
    doc_metadata: Dict[str, Dict] = {}

    for doc_id in doc_ids:
        chunk = doc_to_chunk.get(doc_id, {})

        doc_metadata[doc_id] = {
            "title": chunk.get("title", ""),
            "authors": chunk.get("authors", ""),
            "itemKey": chunk.get("itemKey", ""),
            "date": chunk.get("date", ""),
            "abstract": chunk.get("abstract", "")[:200] if chunk.get("abstract") else "",
            "type": chunk.get("type", ""),
            "url": chunk.get("url", ""),
            "doi": chunk.get("doi", "")
        }

    return doc_metadata


# =============================================================================
# MAIN CLUSTERING PIPELINE
# =============================================================================

def run_clustering_pipeline(
    embeddings_json_path: str,
    output_dir: str,
    session_name: str,
    min_cluster_size: Optional[int] = None,
    aggregation_method: str = "mean",
    n_components: int = DEFAULT_UMAP_N_COMPONENTS
) -> Dict:
    """
    Run the complete document clustering pipeline.

    This is the main entry point that orchestrates all clustering steps:
    1. Load embeddings from JSON
    2. Aggregate chunk embeddings to document level
    3. Reduce dimensions with UMAP
    4. Cluster with HDBSCAN
    5. Generate Zotero tags
    6. Save results

    Args:
        embeddings_json_path: Path to output_chunks_with_embeddings_sparse.json
                             (or output_chunks_with_embeddings.json).
        output_dir: Directory to save clustering_results.json.
        session_name: Name for tag generation (e.g., 'MaBiblio').
                     Used in tags like '_MaBiblio_01'.
        min_cluster_size: Override for minimum cluster size. If None, automatically
                         calculated to achieve ~N/10 clusters.
        aggregation_method: How to aggregate chunk embeddings ('mean', 'max', 'first').
        n_components: UMAP target dimensions (default 50).

    Returns:
        Result dictionary containing:
        - session_name: Name used for tags
        - n_documents: Total documents clustered
        - n_clusters: Number of clusters found
        - n_noise: Number of unclustered (noise) documents
        - cluster_sizes: Dict mapping cluster_id to document count
        - documents: List of document assignments with metadata

    Raises:
        FileNotFoundError: If embeddings file doesn't exist.
        ValueError: If fewer than 3 documents (can't cluster).

    Example:
        >>> results = run_clustering_pipeline(
        ...     "uploads/session/output_chunks_with_embeddings_sparse.json",
        ...     "uploads/session",
        ...     "MaBiblio"
        ... )
        >>> print(f"Found {results['n_clusters']} clusters")
    """
    # ==========================================================================
    # Step 1: Load embeddings
    # ==========================================================================
    print(f"PROGRESS|init|0|Loading embeddings from {os.path.basename(embeddings_json_path)}")

    if not os.path.exists(embeddings_json_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_json_path}")

    with open(embeddings_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"Loaded {len(chunks)} chunks from {embeddings_json_path}")

    # ==========================================================================
    # Step 2: Aggregate to document level
    # ==========================================================================
    print(f"PROGRESS|row|1/5|Aggregating {len(chunks)} chunks to document embeddings")

    doc_embeddings = aggregate_chunk_embeddings(chunks, method=aggregation_method)

    n_docs = len(doc_embeddings)
    if n_docs < 3:
        raise ValueError(
            f"Need at least 3 documents for clustering, got {n_docs}. "
            "Add more documents or skip clustering."
        )

    # Prepare arrays for processing
    doc_ids = list(doc_embeddings.keys())
    embedding_matrix = np.vstack([doc_embeddings[d] for d in doc_ids])

    logger.info(f"Document embedding matrix: {embedding_matrix.shape}")

    # ==========================================================================
    # Step 3: UMAP dimensionality reduction
    # ==========================================================================
    effective_n_components = min(n_components, n_docs - 1)
    print(f"PROGRESS|row|2/5|Reducing dimensions with UMAP ({embedding_matrix.shape[1]}D -> {effective_n_components}D)")

    reduced_embeddings = reduce_dimensions_umap(
        embedding_matrix,
        n_components=effective_n_components
    )

    # ==========================================================================
    # Step 4: HDBSCAN clustering
    # ==========================================================================
    print(f"PROGRESS|row|3/5|Clustering {n_docs} documents with HDBSCAN")

    # Auto-calculate min_cluster_size if not specified
    if min_cluster_size is None:
        min_cluster_size = calculate_optimal_min_cluster_size(n_docs)
        logger.info(f"Auto-calculated min_cluster_size: {min_cluster_size}")

    labels = cluster_documents_hdbscan(
        reduced_embeddings,
        min_cluster_size=min_cluster_size
    )

    # ==========================================================================
    # Step 5: Generate tags
    # ==========================================================================
    print(f"PROGRESS|row|4/5|Generating Zotero cluster tags")

    doc_tags = generate_cluster_tags(doc_ids, labels, session_name)

    # ==========================================================================
    # Step 6: Build and save results
    # ==========================================================================
    print(f"PROGRESS|row|5/5|Saving clustering results")

    # Extract metadata for each document
    doc_metadata = extract_document_metadata(chunks, doc_ids)

    # Calculate cluster statistics
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = int((labels == -1).sum())

    cluster_sizes = {
        int(label): int((labels == label).sum())
        for label in unique_labels
        if label != -1
    }

    # Build document-level results
    documents = []
    for i, doc_id in enumerate(doc_ids):
        meta = doc_metadata.get(doc_id, {})
        documents.append({
            "doc_id": doc_id,
            "cluster_id": int(labels[i]),
            "cluster_tag": doc_tags[doc_id],
            "title": meta.get("title", ""),
            "authors": meta.get("authors", ""),
            "itemKey": meta.get("itemKey", ""),
            "date": meta.get("date", ""),
            "type": meta.get("type", "")
        })

    # Build final results
    results = {
        "session_name": session_name,
        "n_documents": n_docs,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": round(n_noise / n_docs * 100, 1),
        "cluster_sizes": cluster_sizes,
        "min_cluster_size_used": min_cluster_size,
        "aggregation_method": aggregation_method,
        "umap_components": effective_n_components,
        "documents": documents
    }

    # Save results to JSON
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "clustering_results.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Clustering complete: {n_clusters} clusters, results saved to {output_path}")

    return results


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """
    CLI entry point for document clustering.

    Parses command-line arguments and runs the clustering pipeline.
    Designed to be called from the RAGpy web interface or directly from terminal.
    """
    parser = argparse.ArgumentParser(
        description="Cluster documents based on embeddings and generate Zotero tags",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with auto-configured clustering
    python scripts/rad_clustering.py \\
        --input uploads/session/output_chunks_with_embeddings_sparse.json \\
        --output uploads/session \\
        --session-name MaBiblio

    # Custom min_cluster_size for fine control
    python scripts/rad_clustering.py \\
        --input data/embeddings.json \\
        --output data/ \\
        --session-name "Research2024" \\
        --min-cluster-size 5

Output:
    Creates clustering_results.json with cluster assignments and Zotero tags.
    Tags follow the format: _SessionName_01, _SessionName_02, etc.
        """
    )

    # Required arguments
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to embeddings JSON file (output_chunks_with_embeddings_sparse.json)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for clustering_results.json"
    )
    parser.add_argument(
        "--session-name", "-s",
        required=True,
        help="Session name for tag generation (e.g., 'MaBiblio')"
    )

    # Optional arguments
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=None,
        help="Minimum documents per cluster (auto-calculated if not specified)"
    )
    parser.add_argument(
        "--aggregation",
        choices=["mean", "max", "first"],
        default="mean",
        help="Chunk embedding aggregation method (default: mean)"
    )
    parser.add_argument(
        "--umap-components",
        type=int,
        default=DEFAULT_UMAP_N_COMPONENTS,
        help=f"UMAP target dimensions (default: {DEFAULT_UMAP_N_COMPONENTS})"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Run the clustering pipeline
        results = run_clustering_pipeline(
            embeddings_json_path=args.input,
            output_dir=args.output,
            session_name=args.session_name,
            min_cluster_size=args.min_cluster_size,
            aggregation_method=args.aggregation,
            n_components=args.umap_components
        )

        # Print summary
        print(f"\n{'='*60}")
        print(f"CLUSTERING COMPLETE")
        print(f"{'='*60}")
        print(f"Documents:      {results['n_documents']}")
        print(f"Clusters:       {results['n_clusters']}")
        print(f"Noise/Outliers: {results['n_noise']} ({results['noise_ratio']}%)")
        print(f"")
        print(f"Cluster sizes:")
        for cluster_id, size in sorted(results['cluster_sizes'].items()):
            print(f"  Cluster {cluster_id + 1:2d}: {size} documents")
        print(f"")
        print(f"Output: {args.output}/clustering_results.json")
        print(f"{'='*60}")

        sys.exit(0)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"ERROR: {e}")
        sys.exit(1)

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        print(f"ERROR: {e}")
        sys.exit(1)

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        print(f"ERROR: {e}")
        print("Install required dependencies: pip install umap-learn>=0.5.4 hdbscan>=0.8.33")
        sys.exit(1)

    except Exception as e:
        logger.exception(f"Clustering failed: {e}")
        print(f"ERROR: Clustering failed - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
