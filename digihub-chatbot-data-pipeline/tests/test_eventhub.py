import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import types
import pytest
from unittest.mock import MagicMock
 
# Step 1: Create fake config module
fake_config = types.ModuleType("config")
fake_config.EVENT_HUB_CONNECTION_STRING = "fake-eh-connection-string"
fake_config.EVENT_HUB_CONSUMER_GROUP = "fake-consumer-group"
fake_config.EVENT_HUB_NAME = "fake-eventhub-name"
fake_config.site_id = "fake-site-id"
fake_config.COSMOSDB_NAME = "fake-cosmosdb-name"
fake_config.cosmos_client = MagicMock()
fake_config.COSMOSDB_SHAREPOINT_DATA_CONTAINER_NAME = "fake-container-name"
fake_config.COSMOSDB_VECTOR_INDEX = "fake-vector-index"
fake_config.COSMOSDB_DEBOUNCER_CONTAINER_NAME = "fake-debouncer-container"
 
# Step 2: Create fake logger module
fake_logger = MagicMock()
 
# Step 3: Create fake token retriever module
fake_token_module = types.ModuleType("sharepoint_spn_login")
fake_token_instance = MagicMock()
fake_token_instance.get_token.return_value = "fake-token"
fake_token_module.TokenRetriever = MagicMock(return_value=fake_token_instance)
 
# Step 4: Create fake dataprocessor module
fake_dataprocessor_module = types.ModuleType("dataprocessor")
fake_processor_instance = MagicMock()
fake_dataprocessor_module.DocumentProcessor = MagicMock(return_value=fake_processor_instance)
 
# Step 5: Inject fakes into sys.modules
sys.modules["src.utils.config"] = fake_config
sys.modules["src.utils.logger"] = fake_logger
sys.modules["src.utils.sharepoint_spn_login"] = fake_token_module
sys.modules["src.service.dataprocessor"] = fake_dataprocessor_module
    
from src.scripts.eventhub import (
        on_event, process_event, start_eventhub_listener,
        download_file, list_all_files_and_folders
    )
 
# pytestmark = pytest.mark.asyncio
 
 
@pytest.mark.asyncio
async def test_on_event_success():
    partition_context = AsyncMock()
    partition_context.partition_id = "0"
    partition_context.update_checkpoint = AsyncMock()
 
    event = MagicMock()
    event.body_as_str.return_value = '{"mock": "data"}'
 
    with patch("src.scripts.eventhub.logger") as mock_logger, \
         patch("src.scripts.eventhub.process_event") as mock_process_event:
        mock_process_event.return_value = None
 
        await on_event(partition_context, event)
 
        mock_logger.info.assert_any_call("Received event from partition: 0")
        mock_logger.info.assert_any_call("Event processed successfully.")
        partition_context.update_checkpoint.assert_awaited_once_with(event)
        mock_process_event.assert_called_once_with(event)
 
 
@pytest.mark.asyncio
async def test_on_event_failure():
    partition_context = AsyncMock()
    partition_context.partition_id = "1"
    partition_context.update_checkpoint = AsyncMock()
 
    event = MagicMock()
    event.body_as_str.return_value = '{"mock": "data"}'
 
    with patch("src.scripts.eventhub.logger") as mock_logger, \
         patch("src.scripts.eventhub.process_event", side_effect=Exception("Processing error")):
        await on_event(partition_context, event)
 
        mock_logger.info.assert_any_call("Received event from partition: 1")
        mock_logger.error.assert_called_once()
        partition_context.update_checkpoint.assert_awaited_once_with(event)
 
 
@patch("src.scripts.eventhub.EventHubConsumerClient")
@patch("src.scripts.eventhub.logger")
@pytest.mark.asyncio
async def test_start_eventhub_listener(mock_logger, mock_consumer_client):
    mock_client_instance = AsyncMock()
    mock_consumer_client.from_connection_string.return_value = mock_client_instance
 
    await start_eventhub_listener()
 
    mock_client_instance.receive.assert_awaited_once()
    mock_client_instance.close.assert_awaited_once()
 
 
@patch("src.scripts.eventhub.logger")
def test_process_event_malformed_event(mock_logger):
    event = MagicMock()
    event.body_as_str.return_value = '{"value": [{}]}'
 
    process_event(event)
 
    mock_logger.info.assert_any_call(
        "KeyError: 'resource'. Event data structure might be different than expected."
    )
 
 
@patch("src.scripts.eventhub.requests.get")
@patch("src.scripts.eventhub.token_retriever.get_token")
@patch("src.scripts.eventhub.logger")
def test_drive_info_api_failure(mock_logger, mock_get_token, mock_requests_get):
    mock_get_token.return_value = "fake-token"
 
    event = MagicMock()
    event.body_as_str.return_value = '{"value": [{"resource": "/drives/drive-id/items/item-id"}]}'
 
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("404 Not Found")
    mock_requests_get.return_value = mock_response
 
    process_event(event)
 
    mock_logger.info.assert_any_call("An error occurred: 404 Not Found")
 
 
@patch("src.scripts.eventhub.download_file")
@patch("src.scripts.eventhub.cosmos_client.get_database_client")
@patch("src.scripts.eventhub.list_all_files_and_folders")
@patch("src.scripts.eventhub.requests.get")
@patch("src.scripts.eventhub.token_retriever.get_token")
@patch("src.scripts.eventhub.logger")
def test_file_update_triggers_debouncer(
    mock_logger, mock_get_token, mock_requests_get, mock_list_all_files, mock_get_db_client, mock_download_file
):
    mock_get_token.return_value = "fake-token"
 
    event = MagicMock()
    event.body_as_str.return_value = '{"value": [{"resource": "/drives/drive-id/items/item-id"}]}'
 
    mock_drive_response = MagicMock()
    mock_drive_response.json.return_value = {"name": "TestDrive"}
    mock_drive_response.raise_for_status.return_value = None
    mock_requests_get.return_value = mock_drive_response
 
    mock_list_all_files.return_value = [
        {
            "id": "1",
            "pathwithfilename": "file1.txt",
            "createdDateTime": "2023-01-01T00:00:00Z",
            "lastModifiedDateTime": "2023-01-03T00:00:00Z"
        }
    ]
 
    mock_container = MagicMock()
    mock_container.query_items.return_value = [
        {
            "id": "1",
            "pathwithfilename": "file1.txt",
            "lastModifiedDateTime": "2023-01-01T00:00:00Z"
        }
    ]
 
    mock_db = MagicMock()
    mock_db.get_container_client.return_value = mock_container
    mock_db.create_container_if_not_exists.return_value = mock_container
    mock_get_db_client.return_value = mock_db
 
    process_event(event)
 
    mock_logger.info.assert_any_call("Debouncer entry updated for: TestDrive")
 
 
@patch("src.scripts.eventhub.cosmos_client.get_database_client")
@patch("src.scripts.eventhub.list_all_files_and_folders")
@patch("src.scripts.eventhub.requests.get")
@patch("src.scripts.eventhub.token_retriever.get_token")
@patch("src.scripts.eventhub.logger")
def test_file_deletion_triggers_index_cleanup(
    mock_logger, mock_get_token, mock_requests_get, mock_list_all_files, mock_get_db_client
):
    mock_get_token.return_value = "fake-token"
 
    event = MagicMock()
    event.body_as_str.return_value = '{"value": [{"resource": "/drives/drive-id/items/item-id"}]}'
 
    mock_drive_response = MagicMock()
    mock_drive_response.json.return_value = {"name": "TestDrive"}
    mock_drive_response.raise_for_status.return_value = None
    mock_requests_get.return_value = mock_drive_response
 
    mock_list_all_files.return_value = []
 
    mock_container = MagicMock()
    mock_container.query_items.return_value = [
        {"id": "1", "pathwithfilename": "file1.txt", "foldername": "TestDrive"}
    ]
 
    mock_index_container = MagicMock()
    mock_index_container.query_items.return_value = [
        {"id": "index1", "serviceName": "svc", "metadata": {"filepath": "input_docs\\file1.txt"}}
    ]
 
    mock_db = MagicMock()
    mock_db.get_container_client.side_effect = [mock_container, mock_index_container]
    mock_get_db_client.return_value = mock_db
 
    process_event(event)
 
    mock_logger.info.assert_any_call("Deleting vector index for: file1.txt")
 
 
@patch("src.scripts.eventhub.requests.get")
@patch("src.scripts.eventhub.logger")
def test_list_all_files_and_folders_recursive(mock_logger, mock_requests_get):
    mock_requests_get.side_effect = [
        MagicMock(
            status_code=200,
            json=lambda: {"value": [{"id": "folder1", "name": "sub", "folder": {}, "createdDateTime": "2023", "lastModifiedDateTime": "2023"}]}
        ),
        MagicMock(
            status_code=200,
            json=lambda: {"value": [{"id": "file1", "name": "doc.pdf", "file": {}, "createdDateTime": "2023", "lastModifiedDateTime": "2023"}]}
        )
    ]
 
    result = list_all_files_and_folders("site", "drive", "root", "token", "rootpath")
    assert len(result) == 1
    assert result[0]["pathwithfilename"].endswith("doc.pdf")
 
 
@patch("src.scripts.eventhub.document_processor.process_file")
@patch("src.scripts.eventhub.requests.get")
@patch("src.scripts.eventhub.token_retriever.get_token")
@patch("src.scripts.eventhub.logger")
def test_download_file_success(mock_logger, mock_get_token, mock_requests_get, mock_process_file, tmp_path):
    mock_get_token.return_value = "fake-token"
 
    mock_requests_get.side_effect = [
        MagicMock(status_code=200, content=b"filecontent", raise_for_status=lambda: None),
        MagicMock(status_code=200, json=lambda: {"id": "listid"}, raise_for_status=lambda: None)
    ]
 
    from src.scripts import eventhub
    eventhub.ROOT_FOLDER = tmp_path
    mock_process_file.return_value = ("mock_result", 5)
    download_file("drive", "item", "folder/file.txt", "DriveName")
 
    mock_process_file.assert_called_once()
  