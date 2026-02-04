import os
import sys
import types
import pytest
import io
import csv
from unittest.mock import MagicMock, patch
 
# --- Step 1: Mock config module ---
fake_config = types.ModuleType("config")
fake_config.cosmos_client = MagicMock(name="cosmos_client")
fake_config.COSMOSDB_NAME = "test-db"
fake_config.COSMOSDB_VECTOR_INDEX = "test-index"
fake_config.COSMOSDB_SHAREPOINT_DATA_CONTAINER_NAME = "sharepoint-container"
fake_config.COSMOSDB_SERVICE_NAME_MAPPING_CONTAINER_NAME = "mapping-container"
fake_config.COSMOSDB_DEBOUNCER_CONTAINER_NAME = "debouncer-container"
fake_config.SESSION_CONTAINER_NAME = "session-container"
fake_config.AZURE_STORAGE_BLOB_CONNECTION_STRING = "fake-connection-string"
 
# --- Step 2: Mock logger module ---
fake_logger = MagicMock()
 
# --- Step 3: Mock cosmos_initialize module ---
fake_cosmos_init = types.ModuleType("cosmos_initialize")
fake_cosmos_init.CosmosDBInitializers = MagicMock()
fake_cosmos_init.CosmosDBInitializers.initialize_cosmos = MagicMock()
 
# --- Step 4: Inject fake modules into sys.modules ---
sys.modules["src.utils.config"] = fake_config
sys.modules["src.utils.logger"] = fake_logger
sys.modules["src.utils.cosmos_initialize"] = fake_cosmos_init
 
# --- Step 5: Patch Azure SDK classes ---
@pytest.fixture
def cosmos_initializer_module():
    with patch("azure.cosmos.PartitionKey") as mock_partition_key, \
         patch("azure.storage.blob.BlobServiceClient") as mock_blob_service:
 
        # Mock Cosmos DB database
        mock_db = MagicMock()
        fake_config.cosmos_client.create_database_if_not_exists.return_value = mock_db
 
        # All containers return a generic mock
        mock_db.create_container_if_not_exists.return_value = MagicMock()
 
        # Mock Blob download to return fake CSV data
        fake_csv = io.StringIO("id,name\n1,ServiceA\n2,ServiceB")
        mock_blob_client = MagicMock()
        mock_blob_client.download_blob.return_value.content_as_text.return_value = fake_csv.getvalue()
 
        mock_blob_service.from_connection_string.return_value.get_blob_client.return_value = mock_blob_client
 
        yield
 
# --- Step 6: Test initialization flow ---
def test_initialize_cosmos_success(cosmos_initializer_module):
    from src.scripts.startup import CosmosDBInitializer
 
    initializer = CosmosDBInitializer()
    initializer.initialize_cosmos()
 
    # Assert all major steps happened
    fake_config.cosmos_client.create_database_if_not_exists.assert_called_once()
    fake_cosmos_init.CosmosDBInitializers.initialize_cosmos.assert_called_once()
    # fake_logger.info.assert_any_call("Container 'ServiceNameMapping' is ready.")
    # fake_logger.info.assert_any_call("CSV data uploaded to 'ServiceNameMapping' container successfully.")

