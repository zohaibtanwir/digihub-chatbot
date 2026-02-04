"""
Unit tests for ContextManager

Tests entity extraction, reference resolution, and smart query merging functionality.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from src.chatbot.context_manager import ContextManager


class TestContextManager:
    """Test suite for ContextManager class"""

    @pytest.fixture
    def context_manager(self):
        """Create a ContextManager instance for testing"""
        # Set environment variables for testing
        os.environ["ENABLE_ENTITY_TRACKING"] = "true"
        os.environ["ENABLE_REFERENCE_RESOLUTION"] = "true"
        os.environ["ENABLE_SMART_QUERY_MERGING"] = "true"

        with patch('src.chatbot.context_manager.AzureOpenAIService') as mock_service:
            mock_client = Mock()
            mock_service.return_value.get_client.return_value = mock_client
            manager = ContextManager()
            manager.client = mock_client
            return manager

    def test_has_references_detects_pronouns(self, context_manager):
        """Test that has_references correctly detects pronouns"""
        # Test cases with pronouns
        assert context_manager.has_references("How do I configure it?") is True
        assert context_manager.has_references("Tell me more about that") is True
        assert context_manager.has_references("What about those errors?") is True
        assert context_manager.has_references("Can you explain this feature?") is True
        assert context_manager.has_references("How does the service work?") is True

        # Test cases without pronouns
        assert context_manager.has_references("What is WorldTracer?") is False
        assert context_manager.has_references("Tell me about Bag Manager") is False
        assert context_manager.has_references("How to configure WorldTracer?") is False

    def test_has_references_case_insensitive(self, context_manager):
        """Test that has_references is case insensitive"""
        assert context_manager.has_references("How do I configure IT?") is True
        assert context_manager.has_references("Tell me more about THAT") is True

    def test_build_context_window_formats_correctly(self, context_manager):
        """Test that build_context_window formats messages correctly"""
        history = [
            {"role": "user", "content": "What is WorldTracer?"},
            {"role": "assistant", "content": "WorldTracer is a baggage tracing system"},
            {"role": "user", "content": "How do I configure it?"}
        ]

        context = context_manager.build_context_window(history, window_size=3)

        assert "User: What is WorldTracer?" in context
        assert "Assistant: WorldTracer is a baggage tracing system" in context
        assert "User: How do I configure it?" in context

    def test_build_context_window_respects_window_size(self, context_manager):
        """Test that build_context_window respects the window size limit"""
        history = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Message 3"},
        ]

        context = context_manager.build_context_window(history, window_size=2)

        # Should only include last 2 messages
        assert "Message 3" in context
        assert "Response 2" in context
        assert "Message 1" not in context

    def test_build_context_window_empty_history(self, context_manager):
        """Test that build_context_window handles empty history"""
        context = context_manager.build_context_window([], window_size=5)
        assert context == ""

    def test_get_session_entities_flat(self, context_manager):
        """Test flattening entities dictionary to list"""
        entities = {
            "services": ["WorldTracer", "Bag Manager"],
            "topics": ["lost baggage", "billing"],
            "technical_terms": ["Type B messages"]
        }

        flat = context_manager.get_session_entities_flat(entities)

        assert len(flat) == 5
        assert "WorldTracer" in flat
        assert "Bag Manager" in flat
        assert "lost baggage" in flat
        assert "billing" in flat
        assert "Type B messages" in flat

    def test_group_entities_from_flat_recognizes_services(self, context_manager):
        """Test grouping flat entities recognizes known services"""
        flat_entities = [
            "WorldTracer",
            "Bag Manager",
            "lost baggage",
            "Type B messages",
            "Airport Solutions"
        ]

        grouped = context_manager.group_entities_from_flat(flat_entities)

        assert "WorldTracer" in grouped["services"]
        assert "Bag Manager" in grouped["services"]
        assert "Airport Solutions" in grouped["services"]
        assert "lost baggage" in grouped["topics"]
        assert "Type B messages" in grouped["technical_terms"]

    def test_extract_entities_with_feature_flag_disabled(self, context_manager):
        """Test that extract_entities returns empty when feature flag is disabled"""
        context_manager.entity_tracking_enabled = False

        result = context_manager.extract_entities(
            query="What is WorldTracer?",
            response="WorldTracer is a baggage tracing system"
        )

        assert result == {"services": [], "topics": [], "technical_terms": []}

    @patch('src.chatbot.context_manager.ContextManager.client')
    def test_extract_entities_success(self, mock_client, context_manager):
        """Test successful entity extraction from Q&A pair"""
        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''{
            "services": ["WorldTracer"],
            "topics": ["baggage tracing"],
            "technical_terms": ["IATA codes"]
        }'''
        context_manager.client.chat.completions.create = Mock(return_value=mock_response)

        result = context_manager.extract_entities(
            query="What is WorldTracer?",
            response="WorldTracer is a baggage tracing system that uses IATA codes"
        )

        assert "WorldTracer" in result["services"]
        assert "baggage tracing" in result["topics"]
        assert "IATA codes" in result["technical_terms"]

    @patch('src.chatbot.context_manager.ContextManager.client')
    def test_extract_entities_handles_llm_error(self, mock_client, context_manager):
        """Test that extract_entities handles LLM errors gracefully"""
        # Mock LLM error
        context_manager.client.chat.completions.create = Mock(side_effect=Exception("LLM Error"))

        result = context_manager.extract_entities(
            query="What is WorldTracer?",
            response="WorldTracer is a baggage tracing system"
        )

        # Should return empty structure on error
        assert result == {"services": [], "topics": [], "technical_terms": []}

    @patch('src.chatbot.context_manager.ContextManager.client')
    def test_resolve_references_with_entities(self, mock_client, context_manager):
        """Test reference resolution with entities"""
        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''{
            "resolved_query": "How do I configure WorldTracer?",
            "entities_referenced": ["WorldTracer"]
        }'''
        context_manager.client.chat.completions.create = Mock(return_value=mock_response)

        entities = {
            "services": ["WorldTracer"],
            "topics": ["configuration"],
            "technical_terms": []
        }
        history = [
            {"role": "user", "content": "What is WorldTracer?"},
            {"role": "assistant", "content": "WorldTracer is a baggage tracing system"}
        ]

        result = context_manager.resolve_references(
            query="How do I configure it?",
            entities=entities,
            history=history
        )

        assert "WorldTracer" in result
        assert "configure" in result.lower()

    def test_resolve_references_no_references(self, context_manager):
        """Test that queries without references are returned unchanged"""
        query = "What is WorldTracer?"
        entities = {"services": [], "topics": [], "technical_terms": []}
        history = []

        result = context_manager.resolve_references(query, entities, history)

        assert result == query

    def test_resolve_references_feature_flag_disabled(self, context_manager):
        """Test that resolution is skipped when feature flag is disabled"""
        context_manager.reference_resolution_enabled = False

        query = "How do I configure it?"
        entities = {"services": ["WorldTracer"], "topics": [], "technical_terms": []}
        history = []

        result = context_manager.resolve_references(query, entities, history)

        assert result == query

    def test_build_smart_retrieval_query_independent(self, context_manager):
        """Test that independent queries are returned as-is"""
        query = "What is WorldTracer?"
        history = []
        entities = {"services": [], "topics": [], "technical_terms": []}

        result = context_manager.build_smart_retrieval_query(
            query=query,
            history=history,
            is_dependent=False,
            entities=entities
        )

        assert result == query

    def test_build_smart_retrieval_query_dependent_with_history(self, context_manager):
        """Test smart query building for dependent queries with history"""
        query = "How do I configure it?"
        history = [
            {"role": "user", "content": "What is WorldTracer?"},
            {"role": "assistant", "content": "WorldTracer is a baggage tracing system"},
            {"role": "user", "content": "What are its features?"}
        ]
        entities = {
            "services": ["WorldTracer"],
            "topics": ["configuration"],
            "technical_terms": []
        }

        result = context_manager.build_smart_retrieval_query(
            query=query,
            history=history,
            is_dependent=True,
            entities=entities
        )

        # Should include context from previous messages
        assert "Previous context:" in result or "Current query:" in result
        assert "How do I configure it?" in result

    def test_build_smart_retrieval_query_feature_flag_disabled(self, context_manager):
        """Test fallback when smart query merging is disabled"""
        context_manager.smart_query_merging_enabled = False

        query = "How do I configure it?"
        history = [
            {"role": "user", "content": "What is WorldTracer?"},
            {"role": "user", "content": "What are its features?"}
        ]
        entities = {"services": ["WorldTracer"], "topics": [], "technical_terms": []}

        result = context_manager.build_smart_retrieval_query(
            query=query,
            history=history,
            is_dependent=True,
            entities=entities
        )

        # Should fallback to simple concatenation
        assert "What is WorldTracer?" in result
        assert "How do I configure it?" in result

    def test_build_smart_retrieval_query_limits_history(self, context_manager):
        """Test that smart query building limits history to last 3 messages"""
        query = "Current query"
        history = [
            {"role": "user", "content": "Message 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "user", "content": "Message 3"},
            {"role": "user", "content": "Message 4"},
            {"role": "user", "content": "Message 5"}
        ]
        entities = {"services": [], "topics": [], "technical_terms": []}

        result = context_manager.build_smart_retrieval_query(
            query=query,
            history=history,
            is_dependent=True,
            entities=entities
        )

        # Should only include last 3 user messages (3, 4, 5) plus current
        assert "Message 5" in result or "Previous context:" in result
        assert "Current query" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
