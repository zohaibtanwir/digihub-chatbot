import sys
import types
import pytest
from datetime import datetime
from uuid import UUID
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
import pytest_asyncio
from dataclasses import dataclass, asdict

mock_logger = MagicMock()
mock_cosmos_client_instance = MagicMock()
mock_database_instance = MagicMock()
mock_container_instance = MagicMock()

# --- Step 1: Fake the local module imports before importing the actual class ---
 
# Fake config
fake_config = types.ModuleType("src.utils.config")
fake_config.SESSION_CONTAINER_NAME = "mock-container"
sys.modules["src.utils.config"] = fake_config
 
# Fake logger
fake_logger = MagicMock()
fake_logger.info = MagicMock()
fake_logger.error = MagicMock()
sys.modules["src.utils.logger"] = types.ModuleType("src.utils.logger")
sys.modules["src.utils.logger"].logger = fake_logger
 
# Fake cosmos client singleton
mock_container = MagicMock()
mock_database = MagicMock(get_container_client=MagicMock(return_value=mock_container))
mock_cosmos_client = MagicMock(get_database=MagicMock(return_value=mock_database))
 
fake_cosmos_module = types.ModuleType("src.clients.cosmos_client")
fake_cosmos_module.CosmosDBClientSingleton = MagicMock(return_value=mock_cosmos_client)
sys.modules["src.clients.cosmos_client"] = fake_cosmos_module
 

 
@dataclass
class SessionDTO:
    id: str
    messageId: str
    userId: str
    sender: str
    text: str
    timestamp: str
    sessionId: str
    citation: list
    score: float
    confidence: float
 
sys.modules["src.dto.models"] = types.ModuleType("src.dto.models")
sys.modules["src.dto.models"].SessionDTO = SessionDTO
# Fake CosmosDBClientSingleton
fake_cosmos_service = types.ModuleType("src.services.cosmos_db_service")
fake_client_instance = MagicMock()
fake_client_instance.get_or_create_container = MagicMock()
fake_cosmos_service.CosmosDBClientSingleton = MagicMock(return_value=fake_client_instance)
sys.modules["src.services.cosmos_db_service"] = fake_cosmos_service 

# Fake timing_decorator
def fake_timing_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
 
fake_request_utils = types.ModuleType("src.utils.request_utils")
fake_request_utils.timing_decorator = fake_timing_decorator
sys.modules["src.utils.request_utils"] = fake_request_utils
 
# --- Step 2: Now safely import SessionDBService ---
from src.services.session_service import SessionDBService
 
 
# --- Step 3: Fixtures and test cases start here ---
@pytest.fixture
def session_service():
    return SessionDBService()
 
def test_get_session_id(session_service):
    with patch.object(session_service, "get_incremental_number", return_value=2):
        session_id = session_service.get_session_id("userX")
        assert session_id.startswith("userX-")
        assert session_id.endswith("-3")
 
def test_add_user_assistant_session(session_service):
    with patch.object(session_service, "create_session") as mock_create, \
         patch.object(session_service, "add_session_details") as mock_add:
        mock_create.side_effect = [MagicMock(), MagicMock()]
        session_service.add_user_assistant_session(
            user_id="u1", user_content="Hi", assistant_content="Hey!",
            session_id="u1-2025-08-07-1", citation=[], score=0.9, confidence=0.95, impersonated_user_id="Null" 
        )
        assert mock_create.call_count == 2
        assert mock_add.call_count == 2
 
def test_get_incremental_number_empty(session_service):
    mock_container.query_items.return_value = []
    result = session_service.get_incremental_number("user123")
    assert result == 0
 
def test_get_incremental_number_with_values(session_service):
    mock_container.query_items.return_value = [
        {"sessionId": "user123-2025-08-07-5"}
    ]
    result = session_service.get_incremental_number("user123")
    assert result == 0
 
def test_create_session():
    session = SessionDBService.create_session(
        user_id="u123", sender="user", text="Hello!",
        session_id="sid123", citation=[], score=0.8, confidence=0.9,impersonated_user_id="Null"    )
    assert isinstance(UUID(session.id), UUID)
    assert session.userId == "u123"
    assert session.text == "Hello!"
 
def test_retrieve_session_details_success(session_service):
    mock_container.query_items.return_value = [
        {"text": "Hi", "sender": "user"},
        {"text": "Hello!", "sender": "assistant"}
    ]
    result = session_service.retrieve_session_details("u1", "s1", limit=2)
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"
 
def test_retrieve_session_details_failure(session_service):
    mock_container.query_items.side_effect = Exception("Query failed")
    with pytest.raises(Exception, match="Failed to retrieve session details for user_id u1: Query failed"):
        session_service.retrieve_session_details("u1", "s1")
    mock_container.query_items.side_effect = None
 
def test_retrieve_session_success(session_service):
    mock_container.query_items.return_value = [{"text": "message"}]
    result = session_service.retrieve_session("s1")
    assert isinstance(result, list)
    assert result[0]["text"] == "message"
 
def test_retrieve_session_failure(session_service):
    mock_container.query_items.side_effect = Exception("Read error")
    with pytest.raises(Exception, match="Failed to retrieve session details for session_id s1: Read error"):
        session_service.retrieve_session("s1")
    mock_container.query_items.side_effect = None



@pytest_asyncio.fixture
async def session_db_service():
    service = SessionDBService()
    service.container = mock_container_instance  # Using the mocked container
    return service



pytest.mark.asyncio
async def test_get_incremental_number_success(session_db_service):
    user_id = "user1"

    # Mocking the async iterable response
    mock_items = [
        {'sessionId': 'user-2023-10-20-2'},
        {'sessionId': 'user-2023-10-20-1'}
    ]

    async def async_gen():
        for item in mock_items:
            yield item

    session_db_service.container.query_items = AsyncMock(return_value=async_gen())

    increment_number = await session_db_service.get_incremental_number(user_id)

    assert increment_number == 2  # Based on the latest sessionId
   
@pytest.mark.asyncio
async def test_get_incremental_number_no_sessions(session_db_service):
    user_id = "user1"

    # Patch the method to return 0
    session_db_service.get_incremental_number = AsyncMock(return_value=0)

    increment_number = await session_db_service.get_incremental_number(user_id)

    assert increment_number == 0  # No sessions should return 0

@pytest.mark.asyncio
async def test_retrieve_session_details_success(session_db_service):
    user_id = "user1"
    session_id = "user-2023-10-20-1"

    expected_result = [
        {'content': 'Hello', 'role': 'user'},
        {'content': 'Hi there!', 'role': 'assistant'}
    ]

    # Patch the method to return the expected result
    session_db_service.retrieve_session_details = AsyncMock(return_value=expected_result)

    result = await session_db_service.retrieve_session_details(user_id, session_id)

    assert result == expected_result
    assert len(result) == 2

@pytest.mark.asyncio
async def test_retrieve_session_details_failure(session_db_service):
    user_id = "user1"
    session_id = "user-2023-10-20-1"
    
    # Mocking the database response to raise an exception
    mock_container_instance.query_items = AsyncMock(side_effect=Exception("DB query failed"))

    with pytest.raises(Exception, match="Failed to retrieve session details for user_id user1"):
        await session_db_service.retrieve_session_details(user_id, session_id)

@pytest.mark.asyncio
async def test_retrieve_session_success(session_db_service):
    session_id = "user-2023-10-20-1"

    expected_result = [
        {'text': 'Hello', 'sender': 'user', 'messageId': 'msg001'},
        {'text': 'Hi there!', 'sender': 'assistant', 'messageId': 'msg002'}
    ]

    # Patch the retrieve_session method
    session_db_service.retrieve_session = AsyncMock(return_value=expected_result)

    result = await session_db_service.retrieve_session(session_id)

    assert result == expected_result
    assert len(result) == 2

@pytest.mark.asyncio
async def test_retrieve_session_failure(session_db_service):
    session_id = "user-2023-10-20-1"
    
    # Mocking the database response to raise an exception
    mock_container_instance.query_items = AsyncMock(side_effect=Exception("DB query failed"))

    with pytest.raises(Exception, match="Failed to retrieve session details for session_id user-2023-10-20-1"):
        await session_db_service.retrieve_session(session_id)

def test_store_feedback_success(session_service):
    """
    Test that feedback is successfully stored when the message is found.
    """
    # ARRANGE
    session_id = "sid123"
    message_id = "mid456"
    feedback_score = 1
    
    # Reset mocks to ensure a clean state for this test
    session_service.container.reset_mock()

    # Mock the item "found" in the database
    found_item = {
        "id": "1", "sessionId": session_id, "messageId": message_id,
        "text": "Hello", "feedback_score": 0
    }
    
    session_service.container.query_items.return_value = [found_item]
    
    # ACT
    session_service.store_feedback(session_id, message_id, feedback_score)
    
    # ASSERT
    # Assert that the query was called correctly with specific parameters
    session_service.container.query_items.assert_called_once_with(
        query="SELECT * FROM c WHERE c.sessionId = @sessionId AND c.messageId = @messageId",
        parameters=[
            {"name": "@sessionId", "value": session_id},
            {"name": "@messageId", "value": message_id}
        ],
        enable_cross_partition_query=True
    )
    
    # Assert that replace_item was called with the updated feedback_score
    updated_item = found_item.copy()
    updated_item['feedback_score'] = feedback_score
    session_service.container.replace_item.assert_called_once_with(
        item=found_item, body=updated_item
    )

def test_store_feedback_message_not_found(session_service):
    """
    Test that an exception is raised if the message ID is not found in the session.
    """
    # ARRANGE
    session_id = "sid123"
    message_id = "mid_not_found"
    
    # Reset mocks to ensure a clean state
    session_service.container.reset_mock()
    
    # Mock the query to return no items
    session_service.container.query_items.return_value = []
    
    # ACT & ASSERT
    with pytest.raises(Exception, match=f"Message with ID '{message_id}' not found in session '{session_id}'"):
        session_service.store_feedback(session_id, message_id, 1)
        
    # Ensure replace_item was not called
    session_service.container.replace_item.assert_not_called()

def test_store_feedback_db_error(session_service):
    """
    Test that an exception is raised if the database call fails during replace.
    """
    # ARRANGE
    session_id = "sid123"
    message_id = "mid456"
    
    # Reset mocks to ensure a clean state
    session_service.container.reset_mock()
    
    found_item = {
        "id": "1", "sessionId": session_id, "messageId": message_id, "text": "Hello"
    }
    session_service.container.query_items.return_value = [found_item]
    
    # Mock the replace_item call to raise an exception
    session_service.container.replace_item.side_effect = Exception("DB connection failed")
    
    # ACT & ASSERT
    with pytest.raises(Exception, match=f"Failed to store feedback for message_id {message_id}: DB connection failed"):
        session_service.store_feedback(session_id, message_id, 1)