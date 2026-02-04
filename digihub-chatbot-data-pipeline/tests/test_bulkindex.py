import sys
import types
import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path

# =============================================================================
# STEP 1: PRE-MOCK MODULES
# We must mock 'src.utils.config' and others BEFORE importing the main script
# because the main script imports variables from them at the top level.
# =============================================================================

# 1. Mock src.utils.config
mock_config = types.ModuleType("src.utils.config")
mock_config.site_id = "mock_site_id"
mock_config.COSMOSDB_SHAREPOINT_DATA_CONTAINER_NAME = "mock_metadata_container"
mock_config.COSMOSDB_VECTOR_INDEX = "mock_vector_container"  # ✅ Fixed: Added missing var
mock_config.AZURE_STORAGE_BLOB_CONNECTION_STRING = "mock_connection_string"
mock_config.COSMOSDB_SERVICE_NAME_MAPPING_CONTAINER_NAME = "mock_COSMOSDB_SERVICE_NAME_MAPPING_CONTAINER_NAME" 
# Mock Database and Containers
mock_container = MagicMock()
mock_container.upsert_item.return_value = {"id": "mock_id"}

mock_index_container = MagicMock() # ✅ Fixed: Mock for vector index

mock_database = MagicMock()
mock_database.get_container_client.side_effect = lambda name: \
    mock_index_container if name == "mock_vector_container" else mock_container

mock_config.database = mock_database
# Sometimes scripts import container clients directly from config if defined there
mock_config.container = mock_container
mock_config.index_container = mock_index_container 

sys.modules["src.utils.config"] = mock_config

# 2. Mock src.utils.logger
mock_logger = types.ModuleType("src.utils.logger")
mock_logger.logger = MagicMock()
sys.modules["src.utils.logger"] = mock_logger

# 3. Mock src.utils.sharepoint_spn_login
mock_token_module = types.ModuleType("src.utils.sharepoint_spn_login")
mock_token_instance = MagicMock()
mock_token_instance.get_token.return_value = "mock_token"
mock_token_module.TokenRetriever = MagicMock(return_value=mock_token_instance)
sys.modules["src.utils.sharepoint_spn_login"] = mock_token_module

# 4. Mock src.service.dataprocessor
mock_dp_module = types.ModuleType("src.service.dataprocessor")
mock_dp_instance = MagicMock()
mock_dp_instance.process_file.return_value = {"status": "ok"}
mock_dp_module.DocumentProcessor = MagicMock(return_value=mock_dp_instance)
sys.modules["src.service.dataprocessor"] = mock_dp_module

# =============================================================================
# STEP 2: IMPORT THE CODE TO TEST
# =============================================================================
from src.scripts import bulkindex
from src.scripts.bulkindex import (
    get_drive_info,
    download_file,
    process_folder,
    list_all_files_and_folders_flat,
    sync_deleted_files,
    main,
    site_id,
    token_retriever,
    container,
    index_container, # Ensure this is available
    document_processor,
    ROOT_FOLDER
)

# =============================================================================
# STEP 3: FIXTURES
# =============================================================================

@pytest.fixture
def mock_file_path():
    return "Documents/testfile.pdf"

@pytest.fixture
def mock_token():
    return "mocked_token"

@pytest.fixture
def mock_item():
    """
    ✅ Fixed: Added 'webUrl', 'downloadUrl', and 'size' to ensure 
    process_folder logic treats this as a valid downloadable file.
    """
    return {
        'id': 'item123',
        'name': 'testfile.pdf',
        'createdDateTime': '2023-01-01T00:00:00Z',
        'lastModifiedDateTime': '2023-01-02T00:00:00Z',
        'webUrl': 'http://sharepoint/sites/site/Documents/testfile.pdf',
        '@microsoft.graph.downloadUrl': 'http://mock.download.url',
        'size': 1024,
        'file': {
            'mimeType': 'application/pdf'
        }
    }

@pytest.fixture
def mock_drive_info():
    return {
        "value": [
            {"id": "drive123", "name": "Documents"}
        ]
    }

@pytest.fixture
def mock_list_info():
    return {"id": "list123"}

# =============================================================================
# STEP 4: TEST CASES
# =============================================================================

# --- Basic Utility Tests ---

def test_get_drive_info_success():
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "value": [{"id": "drive123", "name": "Documents"}]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
 
        result = get_drive_info("mock_site_id", "mock_token")
        assert result["value"][0]["id"] == "drive123"
 
def test_list_files_and_folders_success(mock_token):
    with patch.object(token_retriever, 'get_token', return_value=mock_token), \
         patch("requests.get") as mock_get, \
         patch("src.scripts.bulkindex.logger") as mock_logger:

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "value": [
                {"id": "file123", "name": "example.pdf"},
                {"id": "folder456", "name": "subfolder"}
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = bulkindex.list_files_and_folders("mock_site_id", "drive123", "folder123", mock_token)

        assert result["value"][0]["id"] == "file123"
        mock_logger.info.assert_called_with("Files and folders listed successfully.")


# --- Core Processing Tests ---

def test_download_file_success(mock_item, mock_token, mock_file_path):
    with patch("requests.get") as mock_get, \
         patch.object(token_retriever, 'get_token', return_value=mock_token), \
         patch.object(container, 'upsert_item') as mock_upsert, \
         patch.object(document_processor, 'process_file', return_value={"status": "ok"}), \
         patch.object(bulkindex, 'site_id', 'mock_site_id'):
 
        mock_response = MagicMock()
        mock_response.content = b"PDF content"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
 
        download_file(
            site_id,
            "drive123",
            mock_item,
            mock_token,
            mock_file_path,
            "list123",
            "Documents"
        )
        mock_upsert.assert_called()

# ... (Imports and previous fixtures remain the same) ...

@pytest.fixture
def mock_item():
    """
    Detailed mock item representing a file in SharePoint Graph API.
    Includes all keys commonly checked by processing scripts.
    """
    return {
        'id': 'item123',
        'name': 'testfile.pdf',  # Must end in .pdf or .docx
        'createdDateTime': '2023-01-01T00:00:00Z',
        'lastModifiedDateTime': '2023-01-02T00:00:00Z',
        'webUrl': 'http://sharepoint/sites/site/Documents/testfile.pdf',
        # Key often used for downloading:
        '@microsoft.graph.downloadUrl': 'http://mock.download.url', 
        'size': 1024,
        # Key used to identify as file:
        'file': {
            'mimeType': 'application/pdf'
        },
        # Ensure it's not seen as a folder
        'folder': None 
    }
# --- New Functionality Tests (Recursion & Sync) ---

def test_list_all_files_and_folders_flat_recursion():
    """
    Tests that the recursive function correctly flattens the folder structure.
    Structure: Root -> [File1, Subfolder], Subfolder -> [File2]
    """
    mock_response_root = {
        "value": [
            {"id": "file1", "name": "File1.pdf", "file": {}},
            {"id": "folder1", "name": "Subfolder", "folder": {}},
        ]
    }
    mock_response_sub = {
        "value": [
            {"id": "file2", "name": "File2.docx", "file": {}}
        ]
    }

    with patch("src.scripts.bulkindex.list_files_and_folders") as mock_list_func:
        # Side effect returns different responses for sequential calls
        mock_list_func.side_effect = [mock_response_root, mock_response_sub]
        
        results = []
        list_all_files_and_folders_flat(
            "site_id", "drive_id", "root", "token", "Documents", results
        )

        assert len(results) == 2
        assert results[0]['pathwithfilename'] == "Documents/File1.pdf"
        assert results[1]['pathwithfilename'] == "Documents/Subfolder/File2.docx"

def test_sync_deleted_files_with_deletions():
    """
    Tests that a file present in CosmosDB but missing from SharePoint 
    triggers deletion in both metadata and vector containers.
    """
    drive_name = "Documents"
    file_path_to_delete = f"{drive_name}/deleted_file.pdf"
    
    # 1. Mock SharePoint to return nothing (empty list)
    def mock_populate_sp_empty(site, drive, folder, token, path, output_list):
        pass 

    # 2. Mock DB to return the file (simulating it exists in DB)
    mock_db_items = [
        {"id": "doc_id_1", "pathwithfilename": file_path_to_delete, "foldername": drive_name}
    ]

    # 3. Mock Vector Index to return chunks
    mock_vector_chunks = [
        {"id": "chunk_1", "partitionKey": "pk1", "metadata": {"filepath": file_path_to_delete}},
        {"id": "chunk_2", "partitionKey": "pk2", "metadata": {"filepath": file_path_to_delete}}
    ]

    with patch.object(token_retriever, 'get_token', return_value="token"), \
         patch("src.scripts.bulkindex.list_all_files_and_folders_flat", side_effect=mock_populate_sp_empty), \
         patch.object(container, "query_items", return_value=mock_db_items), \
         patch.object(container, "delete_item") as mock_delete_meta, \
         patch.object(index_container, "query_items", return_value=mock_vector_chunks), \
         patch.object(index_container, "delete_item") as mock_delete_vector:
        
        sync_deleted_files("site", "drive", drive_name)

        # Assert Metadata Deletion
        mock_delete_meta.assert_called_once_with(item="doc_id_1", partition_key=drive_name)
        
        # Assert Vector Chunk Deletion (Should be called twice)
        assert mock_delete_vector.call_count == 2
        mock_delete_vector.assert_has_calls([
            call(item="chunk_1", partition_key="pk1"),
            call(item="chunk_2", partition_key="pk2")
        ])




def test_process_folder_with_file(mock_item, mock_token):
    # Setup patches
    with patch.object(token_retriever, 'get_token', return_value=mock_token), \
         patch("src.scripts.bulkindex.list_files_and_folders") as mock_list, \
         patch("src.scripts.bulkindex.get_db_record") as mock_is_indexed, \
         patch("src.scripts.bulkindex.download_file") as mock_download, \
         patch.object(bulkindex, 'site_id', 'mock_site_id'):
 
        # 1. Mock list_files_and_folders to return our file item
        mock_list.return_value = {"value": [mock_item]}
        
        # 2. Mock is_file_indexed to return False
        # This forces the logic to enter the "Download" branch
        mock_is_indexed.return_value = False 
 
        # 3. Execute process_folder
        # Arguments: site_id, drive_id, folder_id, current_path_name, list_id
        process_folder("mock_site_id", "drive123", "folder123", "Documents", "list123")
 
        # 4. Assertions
        # Verify listing happened
        mock_list.assert_called_once()
        
        # Verify check for index happened (path construction check)
        # Expected path: Documents/testfile.pdf
        mock_is_indexed.assert_called_with("Documents/testfile.pdf")
        
        # Verify download was called
        mock_download.assert_called_once()




def test_main_calls_sync_and_process(mock_token):
    # 1. Prepare mock data that matches the keys used in your main()
    mock_allowed_drives = [
        {
            'drive_id': 'drive123',
            'name': 'Documents',
            'list_id': 'list123'
        }
    ]

    # 2. Patch the ACTUAL functions called in main()
    with patch("src.scripts.bulkindex.token_retriever") as mock_tr, \
         patch("src.scripts.bulkindex.get_allowed_drives_from_db") as mock_get_db, \
         patch("src.scripts.bulkindex.process_folder") as mock_process, \
         patch("src.scripts.bulkindex.sync_deleted_files") as mock_sync, \
         patch("src.scripts.bulkindex.download_docling"), \
         patch("src.scripts.bulkindex.os.path.exists", return_value=True), \
         patch("src.scripts.bulkindex.os.getenv", return_value=None): # Ensure no env filtering

        # Setup Token
        mock_tr.get_token.return_value = mock_token
        
        # Setup the DB call (This was the missing piece!)
        mock_get_db.return_value = mock_allowed_drives
        
        # Setup Site ID (used in process_folder and sync_deleted_files)
        bulkindex.site_id = 'mock_site_id'

        # Execute
        bulkindex.main()

        # Check calls
        # 1. Check if process_folder was called with correct args
        assert mock_process.called, "process_folder was not called"
        mock_process.assert_called_with('mock_site_id', 'drive123', 'root', 'Documents', 'list123')

        # 2. Check if sync_deleted_files was called
        assert mock_sync.called, "sync_deleted_files was not called"
        mock_sync.assert_called_with('mock_site_id', 'drive123', 'Documents')

def test_is_file_indexed_true():
    mock_item_path = "Documents/testfile.pdf"
    mock_query_result = [{"pathwithfilename": mock_item_path}]

    with patch("src.scripts.bulkindex.container.query_items", return_value=mock_query_result) as mock_query:
        result = bulkindex.get_db_record(mock_item_path)
        mock_query.assert_called_once()
        # Returns the actual record dictionary
        assert result == mock_query_result[0]

def test_is_file_indexed_false():
    mock_item_path = "Documents/testfile.pdf"
    mock_query_result = [] 

    with patch("src.scripts.bulkindex.container.query_items", return_value=mock_query_result) as mock_query:
        result = bulkindex.get_db_record(mock_item_path)
        mock_query.assert_called_once()
        # Returns None when not found
        assert result is None

def test_process_folder_with_file(mock_item, mock_token):
    with patch("src.scripts.bulkindex.token_retriever.get_token", return_value=mock_token), \
         patch("src.scripts.bulkindex.list_files_and_folders") as mock_list, \
         patch("src.scripts.bulkindex.get_db_record") as mock_get_record, \
         patch("src.scripts.bulkindex.download_file") as mock_download:
 
        bulkindex.site_id = 'mock_site_id'
        mock_list.return_value = {"value": [mock_item]}
        mock_get_record.return_value = None 
 
        bulkindex.process_folder("mock_site_id", "drive123", "folder123", "Documents", "list123")
 
        mock_list.assert_called_once()
        mock_get_record.assert_called_with("Documents/testfile.pdf")
        mock_download.assert_called_once()

def test_get_allowed_drives_success():
    """Test successful retrieval of drives from DB."""
    mock_items = [
        {'drive_id': 'd1', 'name': 'Drive 1'},
        {'drive_id': 'd2', 'name': 'Drive 2'}
    ]
    
    with patch("src.scripts.bulkindex.mapping_container") as mock_container:
        # Mock the query_items to return our list
        mock_container.query_items.return_value = mock_items
        
        result = bulkindex.get_allowed_drives_from_db()
        
        assert len(result) == 2
        assert result[0]['drive_id'] == 'd1'
        mock_container.query_items.assert_called_once()

def test_get_allowed_drives_exception():
    """Test that an empty list is returned when a DB error occurs."""
    with patch("src.scripts.bulkindex.mapping_container") as mock_container:
        # Simulate a database connection error
        mock_container.query_items.side_effect = Exception("DB Connection Error")
        
        result = bulkindex.get_allowed_drives_from_db()
        
        assert result == [] # Should return empty list on exception


# --- Tests for delete_existing_index ---

def test_delete_existing_index_success():
    """Test successful deletion of metadata and vector chunks."""
    file_path = "folder/test.pdf"
    
    # Mock metadata record
    mock_meta_items = [{'id': 'meta1', 'foldername': 'folder'}]
    
    # Mock vector chunk records
    mock_index_items = [
        {'id': 'chunk1', 'partitionKey': 'pk1'},
        {'id': 'chunk2', 'serviceName': 'service-A'} # Test the fallback logic
    ]

    with patch("src.scripts.bulkindex.container") as mock_meta_cont, \
         patch("src.scripts.bulkindex.index_container") as mock_idx_cont:
        
        # Setup query returns
        mock_meta_cont.query_items.return_value = mock_meta_items
        mock_idx_cont.query_items.return_value = mock_index_items
        
        result = bulkindex.delete_existing_index(file_path)
        
        # Assertions
        assert result is True
        # Verify metadata delete call
        mock_meta_cont.delete_item.assert_called_with(item='meta1', partition_key='folder')
        # Verify vector chunk delete calls (called twice)
        assert mock_idx_cont.delete_item.call_count == 2
        
        # Verify fallback logic (the second chunk used 'serviceName' because 'partitionKey' was missing)
        mock_idx_cont.delete_item.assert_any_call(item='chunk2', partition_key='service-A')

def test_delete_existing_index_no_items_found():
    """Test when no records exist to delete (should still return True)."""
    with patch("src.scripts.bulkindex.container") as mock_meta_cont, \
         patch("src.scripts.bulkindex.index_container") as mock_idx_cont:
        
        mock_meta_cont.query_items.return_value = []
        mock_idx_cont.query_items.return_value = []
        
        result = bulkindex.delete_existing_index("non_existent.pdf")
        
        assert result is True
        mock_meta_cont.delete_item.assert_not_called()
        mock_idx_cont.delete_item.assert_not_called()

def test_delete_existing_index_exception():
    """Test that False is returned if an error occurs during deletion."""
    with patch("src.scripts.bulkindex.container") as mock_meta_cont:
        # Cause an exception during the first query
        mock_meta_cont.query_items.side_effect = Exception("Delete Failed")
        
        result = bulkindex.delete_existing_index("error_file.pdf")
        
        assert result is False