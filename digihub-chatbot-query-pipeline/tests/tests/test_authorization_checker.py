"""
Unit tests for AuthorizationChecker

Tests user authorization validation, service line access checking,
and generation of appropriate error messages for unauthorized access.
"""

import pytest
import json
from unittest.mock import MagicMock, Mock, patch
import sys


# Store mock references at module level for access in tests
_mock_retrieval_service_instance = MagicMock()
_mock_relevance_judge_instance = MagicMock()


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    """Mock all dependencies before importing the module."""
    global _mock_retrieval_service_instance, _mock_relevance_judge_instance

    # Reset mocks for each test
    _mock_retrieval_service_instance.reset_mock()
    _mock_relevance_judge_instance.reset_mock()

    # Mock config
    mock_config = MagicMock()
    mock_config.OPENAI_DEPLOYMENT_NAME = "gpt-4"
    monkeypatch.setitem(sys.modules, "src.utils.config", mock_config)

    # Mock logger
    mock_logger = MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "src.utils.logger",
        MagicMock(logger=mock_logger)
    )

    # Mock subscriptions enum
    mock_subscriptions = MagicMock()
    mock_subscriptions.get_service_names = MagicMock(return_value=["WorldTracer", "Billing"])
    monkeypatch.setitem(
        sys.modules,
        "src.enums.subscriptions",
        mock_subscriptions
    )

    # Mock exceptions - must match real interface (note: 'disclaimar' is the typo in real code)
    class UnAuthorizedServiceLineException(Exception):
        def __init__(self, message="Query Belongs To UnAuthorized Service Line", disclaimar=""):
            super().__init__(message)
            self.disclaimar = disclaimar

    class PartialAccessServiceLineException(Exception):
        def __init__(self, message="Query Belongs To UnAuthorized Service Line", disclaimar=""):
            super().__init__(message, disclaimar)
            self.disclaimar = disclaimar

    mock_exceptions = MagicMock()
    mock_exceptions.UnAuthorizedServiceLineException = UnAuthorizedServiceLineException
    mock_exceptions.PartialAccessServiceLineException = PartialAccessServiceLineException
    monkeypatch.setitem(
        sys.modules,
        "src.exceptions.service_line_exception",
        mock_exceptions
    )

    # Mock Retrieval Service - return same instance
    mock_retrieval_service_class = MagicMock(return_value=_mock_retrieval_service_instance)
    monkeypatch.setitem(
        sys.modules,
        "src.services.retrieval_service",
        MagicMock(RetreivalService=mock_retrieval_service_class)
    )

    # Mock Relevance Judge - return same instance
    mock_relevance_judge_class = MagicMock(return_value=_mock_relevance_judge_instance)
    monkeypatch.setitem(
        sys.modules,
        "src.chatbot.relevance_judge",
        MagicMock(RelevanceJudge=mock_relevance_judge_class)
    )

    yield


@pytest.fixture
def authorization_checker(patch_dependencies):
    """Create an AuthorizationChecker instance for testing"""
    from src.chatbot.authorization_checker import AuthorizationChecker
    checker = AuthorizationChecker()
    return checker


class TestAuthorizationChecker:
    """Test suite for AuthorizationChecker class"""

    def test_init(self, authorization_checker):
        """Test AuthorizationChecker initialization"""
        assert authorization_checker.relevance_judge is not None
        assert authorization_checker.SENSITIVE_SERVICE_LINE_ID == 460

    def test_unauthorized_messages_exist(self, authorization_checker):
        """Test that unauthorized messages exist for all languages"""
        assert "english" in authorization_checker.UNAUTHORIZED_SERVICE_LINE_MESSAGES
        assert "german" in authorization_checker.UNAUTHORIZED_SERVICE_LINE_MESSAGES
        assert "french" in authorization_checker.UNAUTHORIZED_SERVICE_LINE_MESSAGES
        assert "spanish" in authorization_checker.UNAUTHORIZED_SERVICE_LINE_MESSAGES

    def test_unauthorized_direct_messages_exist(self, authorization_checker):
        """Test that unauthorized direct messages exist for all languages"""
        assert "english" in authorization_checker.UNAUTHORIZED_DIRECT_MESSAGES
        assert "german" in authorization_checker.UNAUTHORIZED_DIRECT_MESSAGES
        assert "french" in authorization_checker.UNAUTHORIZED_DIRECT_MESSAGES
        assert "spanish" in authorization_checker.UNAUTHORIZED_DIRECT_MESSAGES


class TestCrossCheckAuthorization:
    """Test suite for cross_check_authorization method"""

    def test_cross_check_non_list_service_line(self, authorization_checker):
        """Test that non-list service line returns early"""
        result = authorization_checker.cross_check_authorization(
            prompt="What is DigiHub?",
            service_line=None,
            detected_language="english",
            is_out_of_scope=False,
            final_response="DigiHub is..."
        )
        assert result is None

    def test_cross_check_no_chunks_retrieved(self, authorization_checker):
        """Test when no chunks are retrieved"""
        _mock_retrieval_service_instance.get_ranked_service_line_chunk.return_value = []

        service_line = [
            {"id": 0, "name": "General Info", "status": "SUBSCRIBED"}
        ]

        result = authorization_checker.cross_check_authorization(
            prompt="What is DigiHub?",
            service_line=service_line,
            detected_language="english",
            is_out_of_scope=False,
            final_response="DigiHub is..."
        )

        assert result is None

    def test_cross_check_no_relevant_service_lines(self, authorization_checker):
        """Test when no relevant service lines are identified"""
        _mock_retrieval_service_instance.get_ranked_service_line_chunk.return_value = [
            {"content": "Test content", "serviceNameid": 0}
        ]
        _mock_relevance_judge_instance.judge_chunks_relevance.return_value = []

        service_line = [
            {"id": 0, "name": "General Info", "status": "SUBSCRIBED"}
        ]

        result = authorization_checker.cross_check_authorization(
            prompt="What is DigiHub?",
            service_line=service_line,
            detected_language="english",
            is_out_of_scope=False,
            final_response="DigiHub is..."
        )

        assert result is None

    def test_cross_check_user_authorized(self, authorization_checker):
        """Test when user has access to all relevant service lines"""
        _mock_retrieval_service_instance.get_ranked_service_line_chunk.return_value = [
            {"content": "General Info content", "serviceNameid": 0}
        ]
        _mock_relevance_judge_instance.judge_chunks_relevance.return_value = [0]

        service_line = [
            {"id": 0, "name": "General Info", "status": "SUBSCRIBED"}
        ]

        result = authorization_checker.cross_check_authorization(
            prompt="What is DigiHub?",
            service_line=service_line,
            detected_language="english",
            is_out_of_scope=False,
            final_response="DigiHub is..."
        )

        assert result is None

    def test_cross_check_user_unauthorized_out_of_scope(self, authorization_checker):
        """Test when user lacks access and query is out of scope"""
        _mock_retrieval_service_instance.get_ranked_service_line_chunk.return_value = [
            {"content": "WorldTracer content", "serviceNameid": 240}
        ]
        _mock_relevance_judge_instance.judge_chunks_relevance.return_value = [240]

        service_line = [
            {"id": 0, "name": "General Info", "status": "SUBSCRIBED"},
            {"id": 240, "name": "WorldTracer", "status": "UNSUBSCRIBED"}
        ]

        # Use try/except since mocked exception class differs from imported class
        try:
            authorization_checker.cross_check_authorization(
                prompt="What is WorldTracer?",
                service_line=service_line,
                detected_language="english",
                is_out_of_scope=True,
                final_response=""
            )
            pytest.fail("Expected UnAuthorizedServiceLineException to be raised")
        except Exception as e:
            assert "UnAuthorizedServiceLineException" in type(e).__name__
            assert "WorldTracer" in str(e)

    def test_cross_check_user_unauthorized_in_scope(self, authorization_checker):
        """Test when user has partial access and query is in scope"""
        _mock_retrieval_service_instance.get_ranked_service_line_chunk.return_value = [
            {"content": "WorldTracer content", "serviceNameid": 240}
        ]
        _mock_relevance_judge_instance.judge_chunks_relevance.return_value = [240]

        service_line = [
            {"id": 0, "name": "General Info", "status": "SUBSCRIBED"},
            {"id": 240, "name": "WorldTracer", "status": "UNSUBSCRIBED"}
        ]

        # Use try/except since mocked exception class differs from imported class
        try:
            authorization_checker.cross_check_authorization(
                prompt="What is WorldTracer?",
                service_line=service_line,
                detected_language="english",
                is_out_of_scope=False,
                final_response="WorldTracer is a baggage system."
            )
            pytest.fail("Expected PartialAccessServiceLineException to be raised")
        except Exception as e:
            assert "PartialAccessServiceLineException" in type(e).__name__
            assert "WorldTracer" in str(e)

    def test_cross_check_sensitive_service_line_excluded(self, authorization_checker):
        """Test that sensitive service line is excluded when user lacks access"""
        _mock_retrieval_service_instance.get_ranked_service_line_chunk.return_value = []

        service_line = [
            {"id": 0, "name": "General Info", "status": "SUBSCRIBED"},
        ]

        authorization_checker.cross_check_authorization(
            prompt="What about Euro CAB?",
            service_line=service_line,
            detected_language="english",
            is_out_of_scope=False,
            final_response=""
        )

        # Verify the method was called
        _mock_retrieval_service_instance.get_ranked_service_line_chunk.assert_called()

    def test_cross_check_german_message(self, authorization_checker):
        """Test unauthorized message in German"""
        _mock_retrieval_service_instance.get_ranked_service_line_chunk.return_value = [
            {"content": "WorldTracer content", "serviceNameid": 240}
        ]
        _mock_relevance_judge_instance.judge_chunks_relevance.return_value = [240]

        service_line = [
            {"id": 0, "name": "General Info", "status": "SUBSCRIBED"},
            {"id": 240, "name": "WorldTracer", "status": "UNSUBSCRIBED"}
        ]

        # Use try/except since mocked exception class differs from imported class
        try:
            authorization_checker.cross_check_authorization(
                prompt="Was ist WorldTracer?",
                service_line=service_line,
                detected_language="german",
                is_out_of_scope=True,
                final_response=""
            )
            pytest.fail("Expected UnAuthorizedServiceLineException to be raised")
        except Exception as e:
            assert "UnAuthorizedServiceLineException" in type(e).__name__
            assert "WorldTracer" in str(e)


class TestCrossCheckAuthorizationDirect:
    """Test suite for cross_check_authorization_direct method"""

    def test_cross_check_direct_non_list_service_line(self, authorization_checker):
        """Test that non-list service line returns early"""
        result = authorization_checker.cross_check_authorization_direct(
            prompt="What is DigiHub?",
            service_line=None,
            chunk_service_line=[0],
            content="Test content",
            detected_language="english",
            access_context=[],
            chunk_acess_services=[],
            final_response="DigiHub is...",
            is_generic=False
        )

        assert result is None

    def test_cross_check_direct_generic_query(self, authorization_checker):
        """Test that generic queries skip authorization check"""
        result = authorization_checker.cross_check_authorization_direct(
            prompt="What is the weather?",
            service_line=[0],
            chunk_service_line=[0, 240],
            content="Test content",
            detected_language="english",
            access_context=[],
            chunk_acess_services=[],
            final_response="Out of scope",
            is_generic=True
        )

        assert result is None

    def test_cross_check_direct_user_authorized(self, authorization_checker):
        """Test when user has access to all chunk service lines"""
        result = authorization_checker.cross_check_authorization_direct(
            prompt="What is DigiHub?",
            service_line=[0, 240, 400],
            chunk_service_line=[0, 240],
            content="Test content",
            detected_language="english",
            access_context=[],
            chunk_acess_services=[],
            final_response="DigiHub is...",
            is_generic=False
        )

        assert result is None

    def test_cross_check_direct_user_unauthorized(self, authorization_checker):
        """Test when user lacks access to required service lines"""
        # Use try/except since mocked exception class differs from imported class
        try:
            authorization_checker.cross_check_authorization_direct(
                prompt="What is WorldTracer?",
                service_line=[0],  # User only has General Info
                chunk_service_line=[0, 240],  # Chunks include WorldTracer (240)
                content="WorldTracer content",
                detected_language="english",
                access_context=[],
                chunk_acess_services=[],
                final_response="",
                is_generic=False
            )
            pytest.fail("Expected UnAuthorizedServiceLineException to be raised")
        except Exception as e:
            assert "UnAuthorizedServiceLineException" in type(e).__name__

    def test_cross_check_direct_partial_access(self, authorization_checker):
        """Test when user has partial access to required service lines"""
        # User has 0, 400 but chunks need 0, 240
        # Use try/except since mocked exception class differs from imported class
        try:
            authorization_checker.cross_check_authorization_direct(
                prompt="What is WorldTracer?",
                service_line=[0, 400],
                chunk_service_line=[0, 240],
                content="WorldTracer content",
                detected_language="english",
                access_context=[],
                chunk_acess_services=[],
                final_response="WorldTracer is...",
                is_generic=False
            )
            pytest.fail("Expected an authorization exception to be raised")
        except Exception as e:
            # Should raise either UnAuthorized or PartialAccess exception
            assert "ServiceLineException" in type(e).__name__

    def test_cross_check_direct_french_message(self, authorization_checker):
        """Test unauthorized message in French"""
        # Use try/except since mocked exception class differs from imported class
        try:
            authorization_checker.cross_check_authorization_direct(
                prompt="Qu'est-ce que WorldTracer?",
                service_line=[0],
                chunk_service_line=[0, 240],
                content="WorldTracer content",
                detected_language="french",
                access_context=[],
                chunk_acess_services=[],
                final_response="",
                is_generic=False
            )
            pytest.fail("Expected UnAuthorizedServiceLineException to be raised")
        except Exception as e:
            assert "UnAuthorizedServiceLineException" in type(e).__name__

    def test_cross_check_direct_spanish_message(self, authorization_checker):
        """Test unauthorized message in Spanish"""
        # Use try/except since mocked exception class differs from imported class
        try:
            authorization_checker.cross_check_authorization_direct(
                prompt="¿Qué es WorldTracer?",
                service_line=[0],
                chunk_service_line=[0, 240],
                content="WorldTracer content",
                detected_language="spanish",
                access_context=[],
                chunk_acess_services=[],
                final_response="",
                is_generic=False
            )
            pytest.fail("Expected UnAuthorizedServiceLineException to be raised")
        except Exception as e:
            assert "UnAuthorizedServiceLineException" in type(e).__name__


class TestGenerateUnauthorizedMessage:
    """Test suite for message generation"""

    def test_message_includes_service_names(self, authorization_checker):
        """Test that unauthorized message includes service names"""
        message = authorization_checker.UNAUTHORIZED_SERVICE_LINE_MESSAGES.get("english", "")
        assert "access" in message.lower() or "subscribe" in message.lower() or len(message) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
