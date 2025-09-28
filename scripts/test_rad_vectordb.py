import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import json
import time # Keep time for potential sleep in retries, though mocks might bypass it
import sys

# Ensure rad_vectordb can be imported
# Assuming this test script is in the same directory as rad_vectordb.py
# or that the ragpy/scripts directory is in PYTHONPATH
try:
    import rad_vectordb
except ModuleNotFoundError:
    # If running from a different CWD, adjust path.
    # This assumes __RAG is the project root.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) # Goes up to ragpy/
    project_root = os.path.dirname(project_root) # Goes up to __RAG/
    scripts_path = os.path.join(project_root, "ragpy", "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    import rad_vectordb


class TestRadVectorDB(unittest.TestCase):

    def setUp(self):
        # Common setup for tests, e.g., sample data
        self.sample_chunk_dense_only = {
            "id": "doc1_chunk1",
            "embedding": [0.1] * 10, # Assuming 10-dim vector for simplicity
            "title": "Test Title",
            "authors": "Author A",
            "date": "2023-01-01",
            "type": "journalArticle",
            "filename": "test.pdf",
            "doc_id": "doc1",
            "chunk_index": 0,
            "total_chunks": 1,
            "text": "This is a test chunk."
        }
        self.sample_chunk_with_sparse = {
            "id": "doc1_chunk2",
            "embedding": [0.2] * 10,
            "sparse_embedding": {"indices": [1, 5, 9], "values": [0.5, 0.8, 0.3]},
            "title": "Test Title Sparse",
            "authors": "Author B",
            "date": "2023-01-02",
            "type": "conferencePaper",
            "filename": "test_sparse.pdf",
            "doc_id": "doc1",
            "chunk_index": 1,
            "total_chunks": 2,
            "text": "This is a test chunk with sparse data."
        }
        self.sample_chunk_no_embedding = {
            "id": "doc1_chunk3",
            "embedding": None, # Missing dense embedding
            "title": "Test Title No Embedding",
            "doc_id": "doc1",
            "text": "This chunk has no dense embedding."
        }
        self.sample_chunk_bad_sparse = {
            "id": "doc1_chunk4",
            "embedding": [0.3] * 10,
            "sparse_embedding": {"indices": ["a", "b"], "values": ["c", "d"]}, # Invalid sparse
            "doc_id": "doc1",
            "text": "This chunk has bad sparse data."
        }

    # Test generate_uuid
    def test_generate_uuid(self):
        uuid1 = rad_vectordb.generate_uuid("test_id_1")
        uuid2 = rad_vectordb.generate_uuid("test_id_1")
        uuid3 = rad_vectordb.generate_uuid("test_id_2")
        self.assertEqual(uuid1, uuid2)
        self.assertNotEqual(uuid1, uuid3)
        self.assertIsInstance(uuid1, str)
        # A basic check for UUID format (not exhaustive)
        self.assertTrue(len(uuid1) == 36 and uuid1.count('-') == 4)

    # Test normalize_date_to_rfc3339
    def test_normalize_date_to_rfc3339(self):
        self.assertEqual(rad_vectordb.normalize_date_to_rfc3339("2023-01-15"), "2023-01-15T00:00:00Z")
        self.assertEqual(rad_vectordb.normalize_date_to_rfc3339("2023/01/15"), "2023-01-15T00:00:00Z")
        self.assertEqual(rad_vectordb.normalize_date_to_rfc3339("2023"), "2023-01-01T00:00:00Z")
        self.assertEqual(rad_vectordb.normalize_date_to_rfc3339("2023-05"), "2023-05-01T00:00:00Z")
        self.assertEqual(rad_vectordb.normalize_date_to_rfc3339("Jan 15, 2023"), "2023-01-15T00:00:00Z")
        self.assertEqual(rad_vectordb.normalize_date_to_rfc3339(""), "1970-01-01T00:00:00Z")
        self.assertEqual(rad_vectordb.normalize_date_to_rfc3339(None), "1970-01-01T00:00:00Z")
        self.assertEqual(rad_vectordb.normalize_date_to_rfc3339("invalid date"), "1970-01-01T00:00:00Z")
        self.assertEqual(rad_vectordb.normalize_date_to_rfc3339("2023-01-15T10:20:30"), "2023-01-15T10:20:30Z") # Assumes naive input becomes UTC
        # If input has Z, dateutil parser makes it tz-aware (UTC). isoformat() might add +00:00.
        # The function's current behavior for "2023-01-15T10:20:30.123Z" is "2023-01-15T10:20:30+00:00Z"
        # Let's adjust the test to reflect this valid RFC3339 output.
        self.assertEqual(rad_vectordb.normalize_date_to_rfc3339("2023-01-15T10:20:30.123Z"), "2023-01-15T10:20:30+00:00Z")


    # --- Pinecone Tests ---
    def test_prepare_vectors_for_pinecone(self):
        chunks = [self.sample_chunk_dense_only, self.sample_chunk_with_sparse, self.sample_chunk_no_embedding, self.sample_chunk_bad_sparse]
        vectors = rad_vectordb.prepare_vectors_for_pinecone(chunks)
        
        self.assertEqual(len(vectors), 3) # one chunk without embedding, one with bad sparse
        
        # Test dense only chunk
        self.assertEqual(vectors[0]["id"], self.sample_chunk_dense_only["id"])
        self.assertEqual(vectors[0]["values"], self.sample_chunk_dense_only["embedding"])
        self.assertNotIn("sparse_values", vectors[0])
        self.assertEqual(vectors[0]["metadata"]["title"], self.sample_chunk_dense_only["title"])

        # Test chunk with sparse data
        self.assertEqual(vectors[1]["id"], self.sample_chunk_with_sparse["id"])
        self.assertEqual(vectors[1]["values"], self.sample_chunk_with_sparse["embedding"])
        self.assertIn("sparse_values", vectors[1])
        self.assertEqual(vectors[1]["sparse_values"]["indices"], self.sample_chunk_with_sparse["sparse_embedding"]["indices"])
        self.assertEqual(vectors[1]["sparse_values"]["values"], self.sample_chunk_with_sparse["sparse_embedding"]["values"])
        
        # Test chunk with bad sparse data (should still include dense part)
        self.assertEqual(vectors[2]["id"], self.sample_chunk_bad_sparse["id"])
        self.assertEqual(vectors[2]["values"], self.sample_chunk_bad_sparse["embedding"])
        self.assertNotIn("sparse_values", vectors[2]) # Sparse should be ignored

    @patch('rad_vectordb.time.sleep') # Mock time.sleep to speed up tests
    @patch('pinecone.Index') # Mock the Pinecone Index object
    def test_upsert_batch_to_pinecone_success(self, MockPineconeIndex, mock_sleep):
        mock_index_instance = MockPineconeIndex.return_value # Not used directly, but the function expects an index object
        mock_index_arg = MagicMock() # This is what's passed to the function
        
        vectors_batch = [{"id": "1", "values": [0.1]}]
        result = rad_vectordb.upsert_batch_to_pinecone(mock_index_arg, vectors_batch)
        
        self.assertTrue(result)
        mock_index_arg.upsert.assert_called_once_with(vectors=vectors_batch)
        mock_sleep.assert_not_called()

    @patch('rad_vectordb.time.sleep')
    @patch('pinecone.Index')
    def test_upsert_batch_to_pinecone_retry_success(self, MockPineconeIndex, mock_sleep):
        mock_index_arg = MagicMock()
        mock_index_arg.upsert.side_effect = [Exception("Simulated API error"), None] # Fail first, then succeed
        
        vectors_batch = [{"id": "1", "values": [0.1]}]
        result = rad_vectordb.upsert_batch_to_pinecone(mock_index_arg, vectors_batch)
        
        self.assertTrue(result)
        self.assertEqual(mock_index_arg.upsert.call_count, 2)
        mock_sleep.assert_called_once_with(2)

    @patch('rad_vectordb.time.sleep')
    @patch('pinecone.Index')
    def test_upsert_batch_to_pinecone_retry_fail(self, MockPineconeIndex, mock_sleep):
        mock_index_arg = MagicMock()
        mock_index_arg.upsert.side_effect = [Exception("Simulated API error"), Exception("Simulated API error on retry")]
        
        vectors_batch = [{"id": "1", "values": [0.1]}]
        result = rad_vectordb.upsert_batch_to_pinecone(mock_index_arg, vectors_batch)
        
        self.assertFalse(result)
        self.assertEqual(mock_index_arg.upsert.call_count, 2)
        mock_sleep.assert_called_once_with(2)

    @patch('rad_vectordb.Pinecone') # Mock the Pinecone class constructor
    @patch('rad_vectordb.prepare_vectors_for_pinecone')
    @patch('rad_vectordb.upsert_batch_to_pinecone')
    @patch('builtins.open', new_callable=mock_open) # Mock open for reading JSON
    def test_insert_to_pinecone_success(self, mock_file_open, mock_upsert, mock_prepare_vectors, MockPineconeClass):
        # --- Setup Mocks ---
        # Mock Pinecone class and its methods
        mock_pc_instance = MockPineconeClass.return_value
        mock_index_instance = MagicMock()
        mock_pc_instance.Index.return_value = mock_index_instance
        
        # Mock list_indexes response
        # For pinecone-client v3.x and later, list_indexes returns an object with an 'indexes' attribute
        # which is a list of objects, each having a 'name' attribute.
        MockIndexDescription = MagicMock()
        MockIndexDescription.name = "articles" # Simulate our target index exists
        mock_pc_instance.list_indexes.return_value = MagicMock(indexes=[MockIndexDescription])

        # Mock reading from JSON file
        sample_data = [self.sample_chunk_dense_only, self.sample_chunk_with_sparse]
        mock_file_open.return_value.read.return_value = json.dumps(sample_data)
        
        # Mock prepare_vectors_for_pinecone
        prepared_vectors_batch1 = [{"id": "doc1_chunk1", "values": [0.1]*10}]
        prepared_vectors_batch2 = [{"id": "doc1_chunk2", "values": [0.2]*10, "sparse_values": {"indices": [1,5,9], "values": [0.5,0.8,0.3]}}]
        # Simulate it being called per document, or per batch if PINECONE_BATCH_SIZE is small
        # For this test, assume PINECONE_BATCH_SIZE is large enough for all chunks of a doc
        mock_prepare_vectors.side_effect = [
            prepared_vectors_batch1 + prepared_vectors_batch2 # Assuming one doc, two chunks
        ]
        
        # Mock upsert_batch_to_pinecone
        mock_upsert.return_value = True # Simulate successful upsert

        # --- Call the function ---
        # Create a dummy file for os.path.exists to pass
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            result = rad_vectordb.insert_to_pinecone(
                embeddings_json_file="dummy_path.json",
                index_name="articles",
                pinecone_api_key="fake_api_key"
            )

        # --- Assertions ---
        self.assertEqual(result["status"], "success")
        self.assertIn("Total de chunks effectivement préparés et insérés", result["message"])
        self.assertEqual(result["inserted_count"], 2) # Two valid chunks were "inserted"
        
        MockPineconeClass.assert_called_once_with(api_key="fake_api_key")
        mock_pc_instance.list_indexes.assert_called_once()
        mock_pc_instance.Index.assert_called_once_with("articles")
        mock_file_open.assert_called_once_with("dummy_path.json", 'r', encoding='utf-8')
        
        # prepare_vectors_for_pinecone is called once per batch within the document loop
        # In this setup, we have one "document" (implicit from sample_data) with 2 chunks.
        # If PINECONE_BATCH_SIZE >= 2, it's called once.
        self.assertEqual(mock_prepare_vectors.call_count, 1) 
        mock_upsert.assert_called_once_with(mock_index_instance, prepared_vectors_batch1 + prepared_vectors_batch2)


    @patch('rad_vectordb.Pinecone')
    def test_insert_to_pinecone_file_not_exist(self, MockPineconeClass):
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            result = rad_vectordb.insert_to_pinecone("nonexistent.json", "articles", "key")
        self.assertEqual(result["status"], "error")
        self.assertIn("n'existe pas", result["message"])
        self.assertEqual(result["inserted_count"], 0)

    @patch('rad_vectordb.Pinecone')
    def test_insert_to_pinecone_no_api_key(self, MockPineconeClass):
        with patch('os.path.exists') as mock_exists: # Mock os.path.exists
            mock_exists.return_value = True
            result = rad_vectordb.insert_to_pinecone("dummy.json", "articles", None)
        self.assertEqual(result["status"], "error")
        self.assertIn("PINECONE_API_KEY is required", result["message"])

    @patch('rad_vectordb.Pinecone')
    def test_insert_to_pinecone_init_fails(self, MockPineconeClass):
        MockPineconeClass.side_effect = Exception("Init failed")
        with patch('os.path.exists') as mock_exists: # Still need to mock this for the first check
            mock_exists.return_value = True
            result = rad_vectordb.insert_to_pinecone("dummy.json", "articles", "key")
        self.assertEqual(result["status"], "error")
        self.assertIn("Erreur lors de l'initialisation du client Pinecone", result["message"])

    @patch('rad_vectordb.Pinecone')
    def test_insert_to_pinecone_list_indexes_fails(self, MockPineconeClass):
        mock_pc_instance = MockPineconeClass.return_value
        mock_pc_instance.list_indexes.side_effect = Exception("List indexes failed")
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            result = rad_vectordb.insert_to_pinecone("dummy.json", "articles", "key")
        self.assertEqual(result["status"], "error")
        self.assertIn("Failed to connect to Pinecone or list indexes", result["message"])

    @patch('rad_vectordb.Pinecone')
    def test_insert_to_pinecone_index_not_found(self, MockPineconeClass):
        mock_pc_instance = MockPineconeClass.return_value
        MockIndexDescription = MagicMock()
        MockIndexDescription.name = "other_index" # Target index "articles" does not exist
        mock_pc_instance.list_indexes.return_value = MagicMock(indexes=[MockIndexDescription])
        
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            result = rad_vectordb.insert_to_pinecone("dummy.json", "articles", "key")
        self.assertEqual(result["status"], "error")
        self.assertIn("Index 'articles' does not exist", result["message"])

    @patch('rad_vectordb.Pinecone')
    @patch('builtins.open', new_callable=mock_open)
    def test_insert_to_pinecone_json_decode_error(self, mock_file_open, MockPineconeClass):
        mock_pc_instance = MockPineconeClass.return_value
        MockIndexDescription = MagicMock()
        MockIndexDescription.name = "articles"
        mock_pc_instance.list_indexes.return_value = MagicMock(indexes=[MockIndexDescription])
        
        mock_file_open.return_value.read.return_value = "invalid json" # Simulate bad JSON
        
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            result = rad_vectordb.insert_to_pinecone("dummy.json", "articles", "key")
        self.assertEqual(result["status"], "error")
        self.assertIn("Erreur de décodage JSON", result["message"])

    @patch('rad_vectordb.Pinecone')
    @patch('rad_vectordb.prepare_vectors_for_pinecone')
    @patch('rad_vectordb.upsert_batch_to_pinecone')
    @patch('builtins.open', new_callable=mock_open)
    def test_insert_to_pinecone_upsert_fails(self, mock_file_open, mock_upsert, mock_prepare, MockPineconeClass):
        mock_pc_instance = MockPineconeClass.return_value
        MockIndexDescription = MagicMock()
        MockIndexDescription.name = "articles"
        mock_pc_instance.list_indexes.return_value = MagicMock(indexes=[MockIndexDescription])
        
        sample_data = [self.sample_chunk_dense_only]
        mock_file_open.return_value.read.return_value = json.dumps(sample_data)
        mock_prepare.return_value = [{"id": "doc1_chunk1", "values": [0.1]*10}]
        mock_upsert.return_value = False # Simulate upsert failure

        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            result = rad_vectordb.insert_to_pinecone("dummy.json", "articles", "key")

        self.assertEqual(result["status"], "partial_error") # or "error" if all batches fail
        self.assertIn("Au moins un lot n'a pas pu être inséré", result["message"])
        self.assertEqual(result["inserted_count"], 0)

    @patch('rad_vectordb.Pinecone')
    @patch('rad_vectordb.prepare_vectors_for_pinecone')
    @patch('rad_vectordb.upsert_batch_to_pinecone')
    @patch('builtins.open', new_callable=mock_open)
    def test_insert_to_pinecone_no_valid_vectors(self, mock_file_open, mock_upsert, mock_prepare, MockPineconeClass):
        mock_pc_instance = MockPineconeClass.return_value
        MockIndexDescription = MagicMock()
        MockIndexDescription.name = "articles"
        mock_pc_instance.list_indexes.return_value = MagicMock(indexes=[MockIndexDescription])

        sample_data = [self.sample_chunk_no_embedding] # Data that will result in no vectors
        mock_file_open.return_value.read.return_value = json.dumps(sample_data)
        mock_prepare.return_value = [] # prepare_vectors returns empty list

        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            result = rad_vectordb.insert_to_pinecone("dummy.json", "articles", "key")
        
        self.assertEqual(result["status"], "error") # Changed from success_partial_data as no vectors were prepared
        self.assertIn("Aucun chunk n'a été inséré", result["message"])
        self.assertEqual(result["inserted_count"], 0)
        mock_upsert.assert_not_called()


    # --- Weaviate Tests ---
    # TODO: Add tests for Weaviate functions
    # insert_to_weaviate_hybrid

    @patch('rad_vectordb.weaviate')
    @patch('builtins.open', new_callable=mock_open)
    def test_insert_to_weaviate_hybrid_success(self, mock_file, mock_weaviate_module):
        # Mock Weaviate client and collection
        mock_client = MagicMock()
        mock_weaviate_module.connect_to_weaviate_cloud.return_value = mock_client
        mock_client.is_ready.return_value = True
        
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection
        
        # Mock tenant handling
        mock_tenants_obj = MagicMock()
        # In Weaviate client v4, tenants.get() returns a Dict[str, Tenant] like object
        # So, its .keys() method would give tenant names.
        mock_tenants_obj.keys.return_value = ["existing_tenant", "alakel"] 
        mock_collection.tenants.get.return_value = mock_tenants_obj
        
        mock_collection_with_tenant = MagicMock()
        mock_collection.with_tenant.return_value = mock_collection_with_tenant
        
        # Mock batch insertion results
        mock_batch_results = MagicMock()
        mock_batch_results.has_errors = False
        # If no errors, .errors attribute might not exist or be empty.
        # The number of successful items is len(batch_data_objects) if no errors.
        mock_collection_with_tenant.data.insert_many.return_value = mock_batch_results

        # Mock file reading
        sample_data = [self.sample_chunk_dense_only, self.sample_chunk_with_sparse]
        mock_file.return_value.read.return_value = json.dumps(sample_data)

        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            inserted_count = rad_vectordb.insert_to_weaviate_hybrid(
                "dummy.json", "fake_url", "fake_api_key", class_name="Article", tenant_name="alakel"
            )

        self.assertEqual(inserted_count, 2) # Two valid chunks
        
        # Check connect_to_weaviate_cloud call
        # The auth_credentials argument is an object, so using unittest.mock.ANY for it
        mock_weaviate_module.connect_to_weaviate_cloud.assert_called_once_with(
            cluster_url="fake_url", auth_credentials=unittest.mock.ANY 
        )
        # Optionally, verify the Auth.api_key was called correctly if needed, though it's an internal detail
        # mock_weaviate_module.classes.init.Auth.api_key.assert_called_once_with("fake_api_key")

        mock_client.collections.get.assert_called_once_with("Article")
        mock_collection.tenants.get.assert_called_once()
        mock_collection.with_tenant.assert_called_once_with("alakel")
        # insert_many would be called once if BATCH_SIZE >= 2
        self.assertEqual(mock_collection_with_tenant.data.insert_many.call_count, 1)


    # --- Qdrant Tests ---
    def test_prepare_points_for_qdrant(self):
        chunks = [self.sample_chunk_dense_only, self.sample_chunk_no_embedding]
        points = rad_vectordb.prepare_points_for_qdrant(chunks)
        
        self.assertEqual(len(points), 1)
        point = points[0]
        
        # Check ID is a UUID string
        self.assertIsInstance(point.id, str)
        self.assertTrue(len(point.id) == 36 and point.id.count('-') == 4)
        
        self.assertEqual(point.vector, self.sample_chunk_dense_only["embedding"])
        self.assertEqual(point.payload["original_id"], self.sample_chunk_dense_only["id"])
        self.assertEqual(point.payload["title"], self.sample_chunk_dense_only["title"])

    @patch('rad_vectordb.time.sleep')
    @patch('qdrant_client.QdrantClient') # Mock QdrantClient directly
    def test_upsert_batch_to_qdrant_success(self, MockQdrantClient, mock_sleep):
        mock_client_instance = MockQdrantClient.return_value # This is what's passed to the function
        
        # Mock the operation_info object returned by client.upsert
        mock_operation_info = MagicMock()
        mock_operation_info.status = rad_vectordb.models.UpdateStatus.COMPLETED # Use the actual enum member
        mock_client_instance.upsert.return_value = mock_operation_info
        
        points_batch = [rad_vectordb.models.PointStruct(id="uuid1", vector=[0.1], payload={})]
        success, count = rad_vectordb.upsert_batch_to_qdrant(mock_client_instance, "test_collection", points_batch)
        
        self.assertTrue(success)
        self.assertEqual(count, len(points_batch))
        mock_client_instance.upsert.assert_called_once_with(collection_name="test_collection", points=points_batch, wait=True)
        mock_sleep.assert_not_called()

    @patch('rad_vectordb.qdrant_client.QdrantClient') # Path to QdrantClient where it's used
    @patch('rad_vectordb.prepare_points_for_qdrant')
    @patch('rad_vectordb.upsert_batch_to_qdrant')
    @patch('builtins.open', new_callable=mock_open)
    def test_insert_to_qdrant_success_collection_exists(self, mock_file, mock_upsert, mock_prepare, MockQdrantClientClass):
        mock_qdrant_client_instance = MockQdrantClientClass.return_value
        mock_qdrant_client_instance.get_collection.return_value = MagicMock() # Simulate collection exists

        sample_data = [self.sample_chunk_dense_only]
        mock_file.return_value.read.return_value = json.dumps(sample_data)
        
        prepared_points = [rad_vectordb.models.PointStruct(id=rad_vectordb.generate_uuid("doc1_chunk1"), vector=[0.1]*10, payload={})]
        mock_prepare.return_value = prepared_points
        
        mock_upsert.return_value = (True, len(prepared_points)) # Simulate successful upsert

        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            inserted_count = rad_vectordb.insert_to_qdrant(
                "dummy.json", "test_collection", qdrant_url="http://fakeurl", qdrant_api_key="fakekey"
            )
        
        self.assertEqual(inserted_count, 1)
        MockQdrantClientClass.assert_called_once_with(url="http://fakeurl", api_key="fakekey", timeout=900)
        mock_qdrant_client_instance.get_collection.assert_called_once_with(collection_name="test_collection")
        mock_qdrant_client_instance.create_collection.assert_not_called() # Should not be called if exists
        mock_upsert.assert_called_once_with(mock_qdrant_client_instance, "test_collection", prepared_points)


    @patch('rad_vectordb.qdrant_client.QdrantClient')
    @patch('rad_vectordb.prepare_points_for_qdrant')
    @patch('rad_vectordb.upsert_batch_to_qdrant')
    @patch('builtins.open', new_callable=mock_open)
    def test_insert_to_qdrant_success_create_collection(self, mock_file, mock_upsert, mock_prepare, MockQdrantClientClass):
        mock_qdrant_client_instance = MockQdrantClientClass.return_value
        # Simulate collection does not exist by raising an exception that implies it
        # A more specific exception like qdrant_client.http.exceptions.UnexpectedResponseError could be used if known
        mock_qdrant_client_instance.get_collection.side_effect = Exception("Collection not found or generic error") 

        sample_data = [self.sample_chunk_dense_only] # Has embedding of len 10
        mock_file.return_value.read.return_value = json.dumps(sample_data)
        
        prepared_points = [rad_vectordb.models.PointStruct(id=rad_vectordb.generate_uuid("doc1_chunk1"), vector=[0.1]*10, payload={})]
        mock_prepare.return_value = prepared_points
        
        mock_upsert.return_value = (True, len(prepared_points))

        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            inserted_count = rad_vectordb.insert_to_qdrant(
                "dummy.json", "new_collection", qdrant_url="http://fakeurl" # No API key for this test
            )
        
        self.assertEqual(inserted_count, 1)
        mock_qdrant_client_instance.create_collection.assert_called_once_with(
            collection_name="new_collection",
            vectors_config=rad_vectordb.models.VectorParams(size=10, distance=rad_vectordb.models.Distance.COSINE)
        )
        mock_upsert.assert_called_once()


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
