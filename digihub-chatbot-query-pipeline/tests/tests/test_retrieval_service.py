import pytest
from unittest.mock import MagicMock
import sys
 
@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    # Mock config
    mock_config = MagicMock()
    mock_config.SESSION_CONTAINER_NAME = "test-session-container"
    mock_config.KNOWLEDGE_BASE_CONTAINER = "test-kb-container"
    monkeypatch.setitem(sys.modules, "src.utils.config", mock_config)
 
    # Mock logger
    monkeypatch.setitem(sys.modules, "src.utils.logger", MagicMock(logger=MagicMock()))
 
    # Mock timing decorator
    monkeypatch.setitem(
        sys.modules,
        "src.utils.request_utils",
        MagicMock(timing_decorator=lambda x: x)
    )
 
    # Mock embedding service
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_embedding_service = MagicMock()
    mock_embedding_service.get_embeddings.return_value = mock_embedder
    monkeypatch.setitem(
        sys.modules,
        "src.services.embedding_service",
        MagicMock(AzureEmbeddingService=lambda: mock_embedding_service)
    )
 
    # Mock CosmosDBClientSingleton
    mock_container = MagicMock()
    # This is the default return value for all tests unless overridden
    mock_container.query_items.return_value = [
        {
            "id": "doc-default-2",
            "content": "Default content 2",
            "cosine_score": 0.5,
        },
        {
            "id": "doc-default-1",
            "content": "Default content 1",
            "cosine_score": 0.1,
        }
    ]
    mock_db = MagicMock()
    mock_db.get_container_client.return_value = mock_container
    mock_cosmos = MagicMock()
    mock_cosmos.get_database.return_value = mock_db
    monkeypatch.setitem(
        sys.modules,
        "src.services.cosmos_db_service",
        MagicMock(CosmosDBClientSingleton=lambda: mock_cosmos)
    )
 
 
def test_rag_retriever_agent(patch_dependencies):
    from src.services.retrieval_service import RetreivalService
    service = RetreivalService()
    result = service.rag_retriever_agent("test", "container", [1, 2], top_k=2)
    assert len(result) == 6
 
 
def test_retrieve_vectordb(patch_dependencies):
    from src.services.retrieval_service import RetreivalService
    service = RetreivalService()
    docs, embedding, citation = service.retrieve_vectordb("test", "container", [1], top_k=2)
    assert len(docs) == 2
    assert isinstance(embedding, list)
    assert isinstance(citation, list)
 
 
def test_retrieve_general_info_chunks(patch_dependencies):
    from src.services.retrieval_service import RetreivalService
    service = RetreivalService()
    result = service.retrieve_general_info_chunks("test", "container", [1], top_k=2, query_embedding=[0.1, 0.2, 0.3])
    assert len(result) == 5
    assert isinstance(result[0], list)
 
 
def test_retreive_neighbouring_chunks(patch_dependencies):
    from src.services.retrieval_service import RetreivalService
    service = RetreivalService()
    chunk = {
        "id": "abc-10",
        "partitionKey": "pk",
        "serviceNameid": 1,
        "serviceName": "Test",
        "content": "text",
        "metadata": {}
    }
    result = service.retreive_neighbouring_chunks([chunk], service.container)
    assert isinstance(result, list)
 

# This is the original test, now corrected.
def test_get_ranked_service_line_chunk(patch_dependencies):
    from src.services.retrieval_service import RetreivalService
    service = RetreivalService()
    # The function returns a single list, not three values.
    result = service.get_ranked_service_line_chunk("test")
    
    # Assert based on the default mock data from the fixture
    assert isinstance(result, list)
    assert len(result) == 2
    # Check the sorting from the default mock data
    assert result[0]['id'] == 'doc-default-2' # Higher score
 
 
def test_get_all_service_line(patch_dependencies):
    from src.services.retrieval_service import RetreivalService
    service = RetreivalService()
    result = service.get_all_service_line()
    assert isinstance(result, list)
 

def test_get_ranked_service_line_chunk_filters_single_line_headings():
    """
    Tests if documents containing only a single-line heading are filtered out.
    """
    from src.services.retrieval_service import RetreivalService
    service = RetreivalService()

    # Get the mock container from the service instance
    mock_container = service.database.get_container_client.return_value
    
    # Override the mock's return_value for THIS test specifically
    mock_container.query_items.return_value = [
        {"id": "doc-1", "content": "This is valid content.", "cosine_score": 0.9},
        {"id": "doc-2", "content": "## This is just a heading", "cosine_score": 0.85}, # Should be filtered
        {"id": "doc-3", "content": "Another valid\nmultiline content.", "cosine_score": 0.8}
    ]

    result = service.get_ranked_service_line_chunk("test query")
    
    # The result should only contain the valid documents
    assert len(result) == 2
    # Check that the filtered doc is not in the result
    result_ids = [doc['id'] for doc in result]
    assert "doc-2" not in result_ids


def test_get_ranked_service_line_chunk_sorting_logic():
    """
    Tests if the results are correctly sorted by cosine_score (desc) and then id (asc).
    """
    from src.services.retrieval_service import RetreivalService
    service = RetreivalService()
    
    mock_container = service.database.get_container_client.return_value
    
    # Mock data is deliberately unsorted
    mock_container.query_items.return_value = [
        {"id": "doc-c", "content": "content", "cosine_score": 0.8},
        {"id": "doc-b", "content": "content", "cosine_score": 0.9}, # Higher score
        {"id": "doc-a", "content": "content", "cosine_score": 0.9}, # Same score, lower id
    ]

    result = service.get_ranked_service_line_chunk("test query")

    # Expected order: doc-a (score 0.9), doc-b (score 0.9), doc-c (score 0.8)
    # The id is used as a tie-breaker, so 'a' comes before 'b'
    expected_order = ["doc-a", "doc-b", "doc-c"]
    actual_order = [doc['id'] for doc in result]

    assert len(result) == 3
    assert actual_order == expected_order


def test_get_ranked_service_line_chunk_returns_top_3():
    """
    Tests that the function returns at most 3 documents even if the query returns more.
    """
    from src.services.retrieval_service import RetreivalService
    service = RetreivalService()
    
    mock_container = service.database.get_container_client.return_value
    
    # Mock query to return 5 valid documents
    mock_container.query_items.return_value = [
        {"id": f"doc-{i}", "content": "valid content", "cosine_score": 0.9 - (i * 0.1)}
        for i in range(5)
    ]

    result = service.get_ranked_service_line_chunk("test query")

    # The final result should be sliced to the top 3
    assert len(result) == 3
    # Check that it returned the highest-scored items
    assert result[0]['id'] == 'doc-0'
    assert result[1]['id'] == 'doc-1'
    assert result[2]['id'] == 'doc-2'

def test_get_ranked_service_line_chunk_with_exclusion(patch_dependencies):
    """
    Tests the get_ranked_service_line_chunk function with the exclude_service_lines parameter.
    Verifies that the Cosmos DB query is constructed correctly with the WHERE...NOT IN clause
    and the appropriate parameters.
    """
    from src.services.retrieval_service import RetreivalService
    service = RetreivalService()
    
    # The service lines to exclude
    excluded_ids = [101, 202]
    
    # Call the function with the exclusion list
    service.get_ranked_service_line_chunk("test query", exclude_service_lines=excluded_ids)
    
    # Get the mock container to inspect the call to query_items
    mock_container = service.database.get_container_client()
    
    # Get the arguments passed to the mock's query_items method
    args, kwargs = mock_container.query_items.call_args
    
    # The expected query string with the WHERE clause
    expected_query_fragment = "WHERE c.serviceNameid NOT IN (@param0,@param1)"
    
    # The expected parameters that should be passed to the query
    expected_parameters = [
        {"name": "@query_embedding", "value": [0.1, 0.2, 0.3]},
        {"name": "@param0", "value": 101},
        {"name": "@param1", "value": 202}
    ]
    
    # Assert that the dynamically generated query contains the correct WHERE clause
    assert expected_query_fragment in kwargs['query']
    
    # Assert that the parameters list is correct
    assert kwargs['parameters'] == expected_parameters


def test_get_ids_from_file_paths_success(patch_dependencies, monkeypatch):
    """
    Tests the happy path where files exist in the database.
    Verifies that IDs are correctly mapped to the input file paths.
    """
    import sys
    from src.services.retrieval_service import RetreivalService
    
    # Mock the global constant used in the function
    monkeypatch.setattr("src.services.retrieval_service.COSMOSDB_SHAREPOINT_DATA_CONTAINER_NAME", "dh-chatbot-sharepoint-sync", raising=False)
    
    service = RetreivalService()
    
    # Setup mock container for this specific call
    mock_container = MagicMock()
    service.database.get_container_client.return_value = mock_container
    
    # Mock DB response
    mock_container.query_items.return_value = [
        {"id": "guid-1", "pathwithfilename": "/docs/file1.pdf"},
        {"id": "guid-2", "pathwithfilename": "/docs/file2.pdf"}
    ]
    
    input_files = ["/docs/file1.pdf", "/docs/file2.pdf"]
    result = service.get_ids_from_file_paths(input_files)
    
    assert len(result) == 2
    assert result[0]["id"] == "guid-1"
    assert result[0]["pathwithfilename"] == "/docs/file1.pdf"
    assert result[1]["id"] == "guid-2"
    
    # Verify the correct container was requested
    service.database.get_container_client.assert_called_with("dh-chatbot-sharepoint-sync")


def test_get_ids_from_file_paths_deduplication_and_missing(patch_dependencies, monkeypatch):
    """
    Tests complex scenarios:
    1. Input list contains duplicates (should be queried once, returned twice).
    2. Input list contains files not in DB (should return None id with status).
    """
    from src.services.retrieval_service import RetreivalService
    
    monkeypatch.setattr("src.services.retrieval_service.COSMOSDB_SHAREPOINT_DATA_CONTAINER_NAME", "test-container", raising=False)
    service = RetreivalService()
    mock_container = MagicMock()
    service.database.get_container_client.return_value = mock_container
    
    # Mock DB response (Only "exist.pdf" is found)
    mock_container.query_items.return_value = [
        {"id": "guid-exist", "pathwithfilename": "exist.pdf"}
    ]
    
    # Input has duplicate "exist.pdf" and one "missing.pdf"
    input_files = ["exist.pdf", "missing.pdf", "exist.pdf"]
    
    result = service.get_ids_from_file_paths(input_files)
    
    # 1. Verify result length matches original input length (3), not unique length (2)
    assert len(result) == 3
    
    # 2. Verify mapping
    assert result[0]["id"] == "guid-exist"      # First occurrence
    assert result[1]["id"] is None              # Missing file
    assert result[1]["status"] == "Not Found"
    assert result[2]["id"] == "guid-exist"      # Duplicate occurrence
    
    # 3. Verify Optimization: Query should only have requested 2 items (deduplicated), not 3
    call_args = mock_container.query_items.call_args
    _, kwargs = call_args
    parameters = kwargs['parameters']
    assert len(parameters) == 2 # "exist.pdf" and "missing.pdf"


def test_get_ids_from_file_paths_query_construction(patch_dependencies, monkeypatch):
    """
    Verifies that the SQL query and parameters are constructed correctly.
    """
    from src.services.retrieval_service import RetreivalService
    
    monkeypatch.setattr("src.services.retrieval_service.COSMOSDB_SHAREPOINT_DATA_CONTAINER_NAME", "test-container", raising=False)
    service = RetreivalService()
    mock_container = MagicMock()
    service.database.get_container_client.return_value = mock_container
    
    input_files = ["file_A.pdf", "file_B.pdf"]
    service.get_ids_from_file_paths(input_files)
    
    call_args = mock_container.query_items.call_args
    _, kwargs = call_args
    
    query = kwargs['query']
    params = kwargs['parameters']
    
    # Check SQL structure
    assert "SELECT c.id, c.pathwithfilename" in query
    assert "FROM c" in query
    assert "WHERE c.pathwithfilename IN" in query
    
    # Check Parameters
    assert len(params) == 2
    assert params[0]['value'] in input_files
    assert params[1]['value'] in input_files
    # Ensure param names in list match param names in query
    param_keys = [p['name'] for p in params]
    for key in param_keys:
        assert key in query


def test_get_ids_from_file_paths_empty_input(patch_dependencies):
    """
    Tests that an empty input list returns early without calling the database.
    """
    from src.services.retrieval_service import RetreivalService
    service = RetreivalService()
    
    # Spy on database call
    service.database.get_container_client = MagicMock()
    
    result = service.get_ids_from_file_paths([])
    
    assert result == []
    # Database should not be accessed for empty input
    service.database.get_container_client.assert_not_called()


def test_get_ids_from_file_paths_exception_handling(patch_dependencies, monkeypatch):
    """
    Tests that database exceptions are caught, logged, and an empty list is returned.
    """
    from src.services.retrieval_service import RetreivalService
    
    monkeypatch.setattr("src.services.retrieval_service.COSMOSDB_SHAREPOINT_DATA_CONTAINER_NAME", "test-container", raising=False)
    service = RetreivalService()
    
    mock_container = MagicMock()
    service.database.get_container_client.return_value = mock_container
    
    # Simulate DB Error
    mock_container.query_items.side_effect = Exception("Cosmos DB Connection Failed")

    result = service.get_ids_from_file_paths(["file1.pdf"])

    assert result == []


# ============================================================================
# Tests for Hybrid Re-ranking and ValidChunk Filter
# ============================================================================

def test_retrieve_with_question_matching_hybrid_scoring(patch_dependencies):
    """
    Tests that hybrid scores are correctly calculated with question-first weighting.
    Default weighting is 70% question similarity, 30% content similarity.
    """
    from src.services.retrieval_service import RetreivalService
    service = RetreivalService()

    mock_container = service.database.get_container_client.return_value

    # Reset any side_effect from previous tests
    mock_container.query_items.side_effect = None

    # Mock data with both question_score and content_score
    # Lower VectorDistance = higher similarity (1 - distance = similarity)
    mock_container.query_items.return_value = [
        {
            "id": "doc-1",
            "content": "This is valid content for document 1.",
            "validChunk": "yes",
            "question_score": 0.2,  # High question similarity (1 - 0.2 = 0.8)
            "content_score": 0.4,   # Medium content similarity (1 - 0.4 = 0.6)
            "heading": "Doc 1",
            "serviceName": "Test",
            "serviceNameid": 1,
            "citation": "/test/doc1.pdf"
        },
        {
            "id": "doc-2",
            "content": "This is valid content for document 2.",
            "validChunk": "yes",
            "question_score": 0.5,  # Lower question similarity (1 - 0.5 = 0.5)
            "content_score": 0.1,   # High content similarity (1 - 0.1 = 0.9)
            "heading": "Doc 2",
            "serviceName": "Test",
            "serviceNameid": 1,
            "citation": "/test/doc2.pdf"
        }
    ]

    # Mock last_response_headers
    mock_container.client_connection = MagicMock()
    mock_container.client_connection.last_response_headers = {"x-ms-documentdb-query-metrics": "test"}

    docs, _, _ = service.retrieve_with_question_matching(
        "test query",
        "container",
        [1],
        top_k=2,
        question_boost_weight=0.7
    )

    # doc-1 hybrid score = 0.7 * 0.8 + 0.3 * 0.6 = 0.56 + 0.18 = 0.74
    # doc-2 hybrid score = 0.7 * 0.5 + 0.3 * 0.9 = 0.35 + 0.27 = 0.62
    # doc-1 should rank higher because of higher question similarity (70% weight)
    assert len(docs) == 2
    assert docs[0]["id"] == "doc-1"  # Higher hybrid score
    assert docs[1]["id"] == "doc-2"
    assert docs[0].get("hybrid_score", 0) > docs[1].get("hybrid_score", 0)


def test_retrieve_with_question_matching_legacy_chunks(patch_dependencies):
    """
    Tests handling of legacy chunks that don't have questionsEmbedding.
    Legacy chunks should use content_score for both question and content similarity.
    """
    from src.services.retrieval_service import RetreivalService
    service = RetreivalService()

    mock_container = service.database.get_container_client.return_value
    mock_container.query_items.side_effect = None  # Reset any side_effect

    # Mix of new chunks (with question_score) and legacy chunks (without)
    mock_container.query_items.return_value = [
        {
            "id": "legacy-doc",
            "content": "This is legacy content without questions.",
            # No validChunk field (legacy)
            "question_score": None,  # No questionsEmbedding
            "content_score": 0.3,
            "heading": "Legacy",
            "serviceName": "Test",
            "serviceNameid": 1,
            "citation": "/test/legacy.pdf"
        },
        {
            "id": "new-doc",
            "content": "This is new content with questions.",
            "validChunk": "yes",
            "question_score": 0.2,
            "content_score": 0.5,
            "heading": "New",
            "serviceName": "Test",
            "serviceNameid": 1,
            "citation": "/test/new.pdf"
        }
    ]

    mock_container.client_connection = MagicMock()
    mock_container.client_connection.last_response_headers = {"x-ms-documentdb-query-metrics": "test"}

    docs, _, _ = service.retrieve_with_question_matching(
        "test query",
        "container",
        [1],
        top_k=2
    )

    assert len(docs) == 2

    # Find legacy doc and verify it's marked as legacy
    legacy_doc = next((d for d in docs if d["id"] == "legacy-doc"), None)
    assert legacy_doc is not None
    assert legacy_doc.get("is_legacy_chunk") is True


def test_retrieve_with_question_matching_min_threshold(patch_dependencies):
    """
    Tests the minimum question similarity threshold filtering.
    Chunks above threshold should come first, then chunks below.
    """
    from src.services.retrieval_service import RetreivalService
    service = RetreivalService()

    mock_container = service.database.get_container_client.return_value
    mock_container.query_items.side_effect = None  # Reset any side_effect

    mock_container.query_items.return_value = [
        {
            "id": "low-sim",
            "content": "Content with low question similarity.",
            "validChunk": "yes",
            "question_score": 0.8,  # Low similarity (1 - 0.8 = 0.2)
            "content_score": 0.2,   # High content similarity
            "heading": "Low Sim",
            "serviceName": "Test",
            "serviceNameid": 1,
            "citation": "/test/low.pdf"
        },
        {
            "id": "high-sim",
            "content": "Content with high question similarity.",
            "validChunk": "yes",
            "question_score": 0.3,  # High similarity (1 - 0.3 = 0.7)
            "content_score": 0.7,   # Low content similarity
            "heading": "High Sim",
            "serviceName": "Test",
            "serviceNameid": 1,
            "citation": "/test/high.pdf"
        }
    ]

    mock_container.client_connection = MagicMock()
    mock_container.client_connection.last_response_headers = {"x-ms-documentdb-query-metrics": "test"}

    # Set threshold to 0.5 - only high-sim doc should be above
    docs, _, _ = service.retrieve_with_question_matching(
        "test query",
        "container",
        [1],
        top_k=2,
        min_question_similarity=0.5
    )

    # high-sim (0.7 similarity) should come first (above threshold)
    # low-sim (0.2 similarity) should come second (below threshold)
    assert len(docs) == 2
    assert docs[0]["id"] == "high-sim"


def test_retrieve_with_question_matching_filters_heading_only(patch_dependencies):
    """
    Tests that chunks containing only a heading are filtered out.
    """
    from src.services.retrieval_service import RetreivalService
    service = RetreivalService()

    mock_container = service.database.get_container_client.return_value
    mock_container.query_items.side_effect = None  # Reset any side_effect

    mock_container.query_items.return_value = [
        {
            "id": "valid-doc",
            "content": "This is actual content\nwith multiple lines.",
            "validChunk": "yes",
            "question_score": 0.3,
            "content_score": 0.3,
            "heading": "Valid",
            "serviceName": "Test",
            "serviceNameid": 1,
            "citation": "/test/valid.pdf"
        },
        {
            "id": "heading-only",
            "content": "## Just a heading",  # Single line starting with ##
            "validChunk": "yes",
            "question_score": 0.2,  # Better score but should be filtered
            "content_score": 0.2,
            "heading": "Heading Only",
            "serviceName": "Test",
            "serviceNameid": 1,
            "citation": "/test/heading.pdf"
        }
    ]

    mock_container.client_connection = MagicMock()
    mock_container.client_connection.last_response_headers = {"x-ms-documentdb-query-metrics": "test"}

    docs, _, _ = service.retrieve_with_question_matching(
        "test query",
        "container",
        [1],
        top_k=5
    )

    # Only the valid doc should be returned
    assert len(docs) == 1
    assert docs[0]["id"] == "valid-doc"


def test_retrieve_with_question_matching_valid_chunk_filter_conditional(patch_dependencies):
    """
    Tests that the validChunk filter is conditional - includes chunks without
    the validChunk field (legacy) AND chunks with validChunk='yes'.
    """
    from src.services.retrieval_service import RetreivalService
    service = RetreivalService()

    mock_container = service.database.get_container_client.return_value
    mock_container.query_items.side_effect = None  # Reset any side_effect

    # This test verifies the query construction - the filter should allow
    # both legacy chunks (no validChunk) and validated chunks (validChunk='yes')
    mock_container.query_items.return_value = [
        {
            "id": "legacy-chunk",
            "content": "Legacy content without validChunk field.",
            # No validChunk field
            "question_score": 0.3,
            "content_score": 0.3,
            "heading": "Legacy",
            "serviceName": "Test",
            "serviceNameid": 1,
            "citation": "/test/legacy.pdf"
        },
        {
            "id": "valid-chunk",
            "content": "New content with validChunk field.",
            "validChunk": "yes",
            "question_score": 0.3,
            "content_score": 0.3,
            "heading": "Valid",
            "serviceName": "Test",
            "serviceNameid": 1,
            "citation": "/test/valid.pdf"
        }
    ]

    mock_container.client_connection = MagicMock()
    mock_container.client_connection.last_response_headers = {"x-ms-documentdb-query-metrics": "test"}

    docs, _, _ = service.retrieve_with_question_matching(
        "test query",
        "container",
        [1],
        top_k=5
    )

    # Both chunks should be returned (filter is conditional)
    assert len(docs) == 2

    # Verify the query was constructed with the conditional filter
    call_args = mock_container.query_items.call_args
    query_str = call_args[1].get('query', '') if call_args[1] else call_args[0][0]
    assert "NOT IS_DEFINED(c.validChunk) OR c.validChunk = 'yes'" in query_str


def test_cosine_similarity_calculation(patch_dependencies):
    """
    Tests the _cosine_similarity helper method.
    """
    from src.services.retrieval_service import RetreivalService
    service = RetreivalService()

    # Perfect similarity (same vector)
    vec = [1.0, 0.0, 0.0]
    assert abs(service._cosine_similarity(vec, vec) - 1.0) < 0.001

    # Orthogonal vectors (no similarity)
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    assert abs(service._cosine_similarity(vec1, vec2) - 0.0) < 0.001

    # Empty vectors
    assert service._cosine_similarity([], []) == 0.0
    assert service._cosine_similarity([1.0], []) == 0.0