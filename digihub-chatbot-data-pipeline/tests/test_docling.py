import os
import sys
import types
from unittest.mock import patch, MagicMock, mock_open
 


def test_download_docling_with_mocked_config():
    # Step 1: Fake config and logger modules
    fake_config = types.ModuleType("config")
    fake_config.AZURE_STORAGE_BLOB_CONNECTION_STRING = "mock_connection_string"
    fake_logger = MagicMock()
 
    # Step 2: Patch sys.modules before importing the script
    with patch.dict(sys.modules, {
        "src.utils.config": fake_config,
        "src.utils.logger": fake_logger,
    }):
        # Import after mocking config/logger
        from src.scripts.doclingmodel import download_docling
 
        with patch("src.scripts.doclingmodel.BlobServiceClient") as MockBlobServiceClient, \
             patch("src.scripts.doclingmodel.open", new_callable=mock_open) as mock_open_func, \
             patch("src.scripts.doclingmodel.os.makedirs") as mock_makedirs, \
             patch("src.scripts.doclingmodel.os.path.abspath", return_value="/abs/path/docling/file.txt") as mock_abspath:
 
            # Setup mock blob structure
            mock_blob_service = MagicMock()
            mock_container_client = MagicMock()
            mock_blob_client = MagicMock()
            mock_blob = MagicMock()
            mock_blob.name = "testfile.txt"
 
            # Blob download returns bytes
            mock_blob_client.download_blob.return_value.readall.return_value = b"mocked file content"
            mock_container_client.list_blobs.return_value = [mock_blob]
            mock_container_client.get_blob_client.return_value = mock_blob_client
            mock_blob_service.get_container_client.return_value = mock_container_client
            MockBlobServiceClient.from_connection_string.return_value = mock_blob_service
 
            # Call the function
            download_docling()
 
            # ✅ Assertion 1: BlobServiceClient was called correctly
            MockBlobServiceClient.from_connection_string.assert_called_once_with("mock_connection_string")
 
            # ✅ Assertion 2: Ensure `open()` was called exactly once
            assert mock_open_func.call_count == 1
 
            # ✅ Assertion 3: Normalize and compare paths
            actual_path = mock_open_func.call_args[0][0]
            expected_path = os.path.join("docling", "testfile.txt")
            assert os.path.normpath(actual_path) == os.path.normpath(expected_path)
 
            # ✅ Assertion 4: File mode
            assert mock_open_func.call_args[0][1] == "wb"
 
            # ✅ Assertion 5: Ensure blob was downloaded
            mock_blob_client.download_blob().readall.assert_called_once()
 
            # # ✅ Assertion 6: Logger was called
            # fake_logger.info.assert_called_with("Downloaded testfile.txt to /abs/path/docling/file.txt")
 