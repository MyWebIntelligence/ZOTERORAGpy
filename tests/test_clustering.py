#!/usr/bin/env python3
"""
Tests for the Document Clustering Feature (Step 4.b)
====================================================

This module tests the clustering functionality including:
- Chunk embedding aggregation
- Tag generation
- Clustering pipeline integration

Run with: pytest tests/test_clustering.py -v

Author: RAGpy Team
Date: 2025-12-10
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
RAGPY_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(RAGPY_ROOT))

# Import clustering functions
from scripts.rad_clustering import (
    aggregate_chunk_embeddings,
    generate_cluster_tags,
    calculate_optimal_min_cluster_size,
    extract_document_metadata
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_chunks():
    """Create sample chunk data for testing."""
    return [
        {
            "doc_id": "doc1",
            "chunk_index": 0,
            "embedding": [1.0, 2.0, 3.0],
            "title": "Document 1",
            "authors": "Author A",
            "itemKey": "ABC123"
        },
        {
            "doc_id": "doc1",
            "chunk_index": 1,
            "embedding": [3.0, 4.0, 5.0],
            "title": "Document 1",
            "authors": "Author A",
            "itemKey": "ABC123"
        },
        {
            "doc_id": "doc2",
            "chunk_index": 0,
            "embedding": [5.0, 6.0, 7.0],
            "title": "Document 2",
            "authors": "Author B",
            "itemKey": "DEF456"
        },
        {
            "doc_id": "doc3",
            "chunk_index": 0,
            "embedding": [7.0, 8.0, 9.0],
            "title": "Document 3",
            "authors": "Author C",
            "itemKey": "GHI789"
        },
    ]


@pytest.fixture
def sample_embeddings_file(tmp_path):
    """Create a temporary embeddings file for testing."""
    # Generate 15 documents with simple embeddings
    # (minimum needed for meaningful clustering tests)
    chunks = []
    np.random.seed(42)

    for i in range(15):
        # Create 2 chunks per document
        for j in range(2):
            chunks.append({
                "doc_id": f"doc_{i}",
                "chunk_index": j,
                "total_chunks": 2,
                "title": f"Test Document {i}",
                "authors": f"Author {i}",
                "itemKey": f"ZKEY{i:03d}",
                "date": f"2024-0{(i % 9) + 1}-15",
                "type": "journalArticle",
                "embedding": list(np.random.randn(64).astype(float))  # Use smaller dimension for tests
            })

    filepath = tmp_path / "test_embeddings.json"
    with open(filepath, "w") as f:
        json.dump(chunks, f)

    return filepath


# =============================================================================
# Test: Embedding Aggregation
# =============================================================================

class TestEmbeddingAggregation:
    """Tests for chunk embedding aggregation."""

    def test_aggregate_mean(self, sample_chunks):
        """Test mean aggregation of chunk embeddings."""
        result = aggregate_chunk_embeddings(sample_chunks, method="mean")

        assert len(result) == 3  # 3 unique documents
        assert "doc1" in result
        assert "doc2" in result
        assert "doc3" in result

        # doc1: mean of [1,2,3] and [3,4,5] = [2,3,4]
        np.testing.assert_array_almost_equal(result["doc1"], [2.0, 3.0, 4.0])

        # doc2: single chunk [5,6,7]
        np.testing.assert_array_almost_equal(result["doc2"], [5.0, 6.0, 7.0])

        # doc3: single chunk [7,8,9]
        np.testing.assert_array_almost_equal(result["doc3"], [7.0, 8.0, 9.0])

    def test_aggregate_max(self, sample_chunks):
        """Test max aggregation of chunk embeddings."""
        result = aggregate_chunk_embeddings(sample_chunks, method="max")

        # doc1: max of [1,2,3] and [3,4,5] = [3,4,5]
        np.testing.assert_array_almost_equal(result["doc1"], [3.0, 4.0, 5.0])

    def test_aggregate_first(self, sample_chunks):
        """Test first-chunk aggregation."""
        result = aggregate_chunk_embeddings(sample_chunks, method="first")

        # doc1: first chunk is [1,2,3]
        np.testing.assert_array_almost_equal(result["doc1"], [1.0, 2.0, 3.0])

    def test_aggregate_empty_doc_id(self):
        """Test handling of missing doc_id."""
        chunks = [
            {"doc_id": "", "embedding": [1.0, 2.0]},  # Empty doc_id
            {"doc_id": None, "embedding": [3.0, 4.0]},  # None doc_id
            {"doc_id": "doc1", "embedding": [5.0, 6.0]},  # Valid
        ]

        result = aggregate_chunk_embeddings(chunks)

        # Only doc1 should be in result
        assert len(result) == 1
        assert "doc1" in result

    def test_aggregate_missing_embedding(self):
        """Test handling of missing embedding."""
        chunks = [
            {"doc_id": "doc1", "embedding": None},
            {"doc_id": "doc1", "embedding": [1.0, 2.0]},
            {"doc_id": "doc2"},  # No embedding field at all
        ]

        result = aggregate_chunk_embeddings(chunks)

        # doc1 should have embedding from valid chunk
        assert "doc1" in result
        np.testing.assert_array_almost_equal(result["doc1"], [1.0, 2.0])

        # doc2 should not be present (no valid embeddings)
        assert "doc2" not in result

    def test_aggregate_invalid_method(self, sample_chunks):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            aggregate_chunk_embeddings(sample_chunks, method="invalid")


# =============================================================================
# Test: Tag Generation
# =============================================================================

class TestTagGeneration:
    """Tests for cluster tag generation."""

    def test_generate_tags_format(self):
        """Test basic tag format."""
        doc_ids = ["doc1", "doc2", "doc3", "doc4"]
        labels = np.array([0, 0, 1, 2])

        tags = generate_cluster_tags(doc_ids, labels, "TestBiblio")

        assert tags["doc1"] == "_TestBiblio_01"
        assert tags["doc2"] == "_TestBiblio_01"  # Same cluster as doc1
        assert tags["doc3"] == "_TestBiblio_02"
        assert tags["doc4"] == "_TestBiblio_03"

    def test_generate_tags_noise(self):
        """Test noise documents get special tag."""
        doc_ids = ["doc1", "doc2", "doc3"]
        labels = np.array([0, -1, 1])  # doc2 is noise

        tags = generate_cluster_tags(doc_ids, labels, "Session")

        assert tags["doc1"] == "_Session_01"
        assert tags["doc2"] == "_Session_noise"
        assert tags["doc3"] == "_Session_02"

    def test_generate_tags_special_chars(self):
        """Test special characters are cleaned from session name."""
        doc_ids = ["doc1"]
        labels = np.array([0])

        # Session name with spaces and special chars
        tags = generate_cluster_tags(doc_ids, labels, "Ma Biblio (2024)!")

        # Special chars should be removed, only alphanumeric kept
        assert tags["doc1"] == "_MaBiblio2024_01"

    def test_generate_tags_long_session_name(self):
        """Test long session names are truncated."""
        doc_ids = ["doc1"]
        labels = np.array([0])

        # Very long session name
        long_name = "A" * 50
        tags = generate_cluster_tags(doc_ids, labels, long_name)

        # Should be truncated to 20 chars
        assert tags["doc1"] == f"_{'A' * 20}_01"

    def test_generate_tags_empty_session_name(self):
        """Test empty session name gets default."""
        doc_ids = ["doc1"]
        labels = np.array([0])

        # Empty or special-chars-only name
        tags = generate_cluster_tags(doc_ids, labels, "!@#$%")

        # Should use default "Cluster"
        assert tags["doc1"] == "_Cluster_01"

    def test_generate_tags_zero_padding(self):
        """Test cluster numbers are zero-padded."""
        doc_ids = [f"doc{i}" for i in range(12)]
        labels = np.array(list(range(12)))  # 12 different clusters

        tags = generate_cluster_tags(doc_ids, labels, "Test")

        # First cluster should be 01
        assert tags["doc0"] == "_Test_01"
        # 10th cluster should be 10 (not 010)
        assert tags["doc9"] == "_Test_10"
        # 12th cluster should be 12
        assert tags["doc11"] == "_Test_12"


# =============================================================================
# Test: Cluster Size Calculation
# =============================================================================

class TestClusterSizeCalculation:
    """Tests for optimal min_cluster_size calculation."""

    def test_calculate_for_100_docs(self):
        """Test calculation for 100 documents."""
        result = calculate_optimal_min_cluster_size(100)

        # Should result in ~10 clusters, so min_cluster_size ~10
        assert 3 <= result <= 15

    def test_calculate_for_small_corpus(self):
        """Test minimum is always at least 3."""
        result = calculate_optimal_min_cluster_size(10)

        # Should be at least 3 (user constraint)
        assert result >= 3

    def test_calculate_for_large_corpus(self):
        """Test calculation for large corpus."""
        result = calculate_optimal_min_cluster_size(1000)

        # Should be capped at reasonable max (15)
        assert result <= 15


# =============================================================================
# Test: Metadata Extraction
# =============================================================================

class TestMetadataExtraction:
    """Tests for document metadata extraction."""

    def test_extract_metadata(self, sample_chunks):
        """Test metadata extraction from chunks."""
        doc_ids = ["doc1", "doc2"]

        metadata = extract_document_metadata(sample_chunks, doc_ids)

        assert "doc1" in metadata
        assert "doc2" in metadata

        assert metadata["doc1"]["title"] == "Document 1"
        assert metadata["doc1"]["authors"] == "Author A"
        assert metadata["doc1"]["itemKey"] == "ABC123"

        assert metadata["doc2"]["title"] == "Document 2"
        assert metadata["doc2"]["itemKey"] == "DEF456"

    def test_extract_metadata_missing_doc(self, sample_chunks):
        """Test metadata extraction for non-existent document."""
        doc_ids = ["doc1", "nonexistent"]

        metadata = extract_document_metadata(sample_chunks, doc_ids)

        # Should have entries for both, but nonexistent should have empty values
        assert "doc1" in metadata
        assert "nonexistent" in metadata
        assert metadata["nonexistent"]["title"] == ""
        assert metadata["nonexistent"]["itemKey"] == ""


# =============================================================================
# Helper Functions
# =============================================================================

def _check_clustering_deps() -> bool:
    """Check if clustering dependencies are installed."""
    try:
        import umap
        import hdbscan
        return True
    except ImportError:
        return False


# =============================================================================
# Test: Full Pipeline Integration
# =============================================================================

class TestClusteringPipelineIntegration:
    """Integration tests for the full clustering pipeline."""

    @pytest.mark.skipif(
        not _check_clustering_deps(),
        reason="UMAP/HDBSCAN not installed"
    )
    def test_full_pipeline(self, sample_embeddings_file, tmp_path):
        """Test full clustering pipeline with sample data."""
        from scripts.rad_clustering import run_clustering_pipeline

        results = run_clustering_pipeline(
            embeddings_json_path=str(sample_embeddings_file),
            output_dir=str(tmp_path),
            session_name="TestSession",
            min_cluster_size=3
        )

        # Check basic results
        assert results["n_documents"] == 15
        assert results["n_clusters"] >= 1
        assert results["n_noise"] >= 0
        assert results["n_clusters"] + results["n_noise"] <= 15

        # Check output file created
        output_path = tmp_path / "clustering_results.json"
        assert output_path.exists()

        # Check output structure
        with open(output_path) as f:
            saved = json.load(f)

        assert "documents" in saved
        assert len(saved["documents"]) == 15

        # Each document should have required fields
        for doc in saved["documents"]:
            assert "doc_id" in doc
            assert "cluster_id" in doc
            assert "cluster_tag" in doc
            assert doc["cluster_tag"].startswith("_TestSession_")

    @pytest.mark.skipif(
        not _check_clustering_deps(),
        reason="UMAP/HDBSCAN not installed"
    )
    def test_pipeline_with_small_dataset(self, tmp_path):
        """Test pipeline handles small datasets gracefully."""
        from scripts.rad_clustering import run_clustering_pipeline

        # Create minimal dataset (just above threshold)
        chunks = []
        for i in range(5):
            chunks.append({
                "doc_id": f"doc_{i}",
                "chunk_index": 0,
                "embedding": list(np.random.randn(32).astype(float)),
                "itemKey": f"KEY{i}"
            })

        filepath = tmp_path / "small_embeddings.json"
        with open(filepath, "w") as f:
            json.dump(chunks, f)

        results = run_clustering_pipeline(
            embeddings_json_path=str(filepath),
            output_dir=str(tmp_path),
            session_name="Small",
            min_cluster_size=3
        )

        assert results["n_documents"] == 5
        # With 5 docs and min_cluster_size=3, should get 1-2 clusters

    def test_pipeline_too_few_documents(self, tmp_path):
        """Test pipeline raises error with too few documents."""
        from scripts.rad_clustering import run_clustering_pipeline

        # Create dataset with only 2 documents
        chunks = [
            {"doc_id": "doc1", "embedding": [1.0, 2.0], "itemKey": "A"},
            {"doc_id": "doc2", "embedding": [3.0, 4.0], "itemKey": "B"},
        ]

        filepath = tmp_path / "tiny_embeddings.json"
        with open(filepath, "w") as f:
            json.dump(chunks, f)

        with pytest.raises(ValueError, match="at least 3 documents"):
            run_clustering_pipeline(
                embeddings_json_path=str(filepath),
                output_dir=str(tmp_path),
                session_name="Tiny"
            )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
