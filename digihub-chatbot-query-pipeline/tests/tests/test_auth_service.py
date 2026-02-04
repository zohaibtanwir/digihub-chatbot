import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import sys
from fastapi import HTTPException
from requests.exceptions import RequestException
from src.dto.models import BaseHeader

# Create mock modules and attributes
mock_config = MagicMock()
mock_config.DIGIHUB_USER_MANAGEMENT_URL = "https://mock.url"

mock_logger = MagicMock()
mock_requests = MagicMock()
mock_requests.RequestException = RequestException

# Patching requests to handle HTTP calls
def mock_requests_get(*args, **kwargs):
    class MockResponse:
        def __init__(self, status_code, json_data):
            self.status_code = status_code
            self.json_data = json_data
        
        def json(self):
            return self.json_data
        
    if len(args)> 0 and "subscriptions" in args[0]:
        return MockResponse(200, {"payload": [{"id": 1, "status": "SUBSCRIBED", "name": "Test Subscription"  }]})
    elif "users" in kwargs['url']:
        return MockResponse(200, {"payload": {"isImpersonationAllowed": False}})
    
    return MockResponse(404, {})  # Default mock for other requests

mock_requests.get = mock_requests_get

# Patch sys.modules to mock all dependencies before importing the service
with patch.dict(sys.modules, {
    "src.utils.config": mock_config,
    "src.utils.logger": mock_logger,
    "requests": mock_requests,
}):
    from src.services.auth_service import AuthorizationService

@pytest.fixture
def authorization_service():
    return AuthorizationService()

@pytest.mark.asyncio
async def test_get_subscriptions_success(authorization_service):
    header = BaseHeader(
        x_digihub_emailid = "test@example.com",
        x_digihub_tenantid = "test_id",
        x_digihub_traceid = "test_trace"
    )

    # 1. Run the service
    subscriptions = authorization_service.get_subscriptions(header)

    # 2. Assert the full object structure
    expected_result = [{
        'id': 1, 
        'name': 'Test Subscription', 
        'status': 'SUBSCRIBED'
    }]
    
    assert subscriptions == expected_result

@pytest.mark.asyncio
async def test_get_subscriptions_http_error(authorization_service):
    mock_requests.get = MagicMock(side_effect=HTTPException(status_code=500))
    header = BaseHeader(
    x_digihub_emailid = "test@example.com",
    x_digihub_tenantid = "test_id",
    x_digihub_traceid = "test_trace")

    with pytest.raises(HTTPException, match="Internal Server Error"):
        authorization_service.get_subscriptions(header)

@pytest.mark.asyncio
async def test_get_subscriptions_not_found(authorization_service):
    mock_requests.get = MagicMock(return_value=MagicMock(status_code=404))
    header = BaseHeader(
    x_digihub_emailid = "test@example.com",
    x_digihub_tenantid = "test_id",
    x_digihub_traceid = "test_trace")

    with pytest.raises(HTTPException, match="Unauthorized: Unable to verify user subscriptions."):
        authorization_service.get_subscriptions(header)

@pytest.mark.asyncio
async def test_is_user_impersonation_allowed_success(authorization_service):
    email_id = "test@example.com"
    request_headers = {"x-digihub-emailid": email_id}

    allowed = authorization_service.is_user_impersonation_allowed(email_id, request_headers)

    assert allowed is False

@pytest.mark.asyncio
async def test_is_user_impersonation_allowed_http_error(authorization_service):
    mock_requests.get = MagicMock(side_effect=HTTPException(status_code=500))
    email_id = "test@example.com"
    request_headers = {"x-digihub-emailid": email_id}

    with pytest.raises(HTTPException, match="Internal Server Error"):
        authorization_service.is_user_impersonation_allowed(email_id, request_headers)
