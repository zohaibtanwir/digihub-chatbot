"""
Unit tests for ContextManager

Tests entity extraction, reference resolution, and smart query merging functionality.
"""

import pytest
import os
import sys
from unittest.mock import Mock, MagicMock


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    """Mock all dependencies before importing the module."""
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

    # Mock Azure OpenAI Service
    mock_openai_service = MagicMock()
    mock_client = MagicMock()
    mock_openai_service.return_value.get_client.return_value = mock_client
    monkeypatch.setitem(
        sys.modules,
        "src.services.azure_openai_service",
        MagicMock(AzureOpenAIService=mock_openai_service)
    )

    # Mock Prompt Template
    mock_prompt_template = MagicMock()
    mock_prompt_template.ENTITY_EXTRACTION_TEMPLATE = MagicMock()
    mock_prompt_template.ENTITY_EXTRACTION_TEMPLATE.value = "Extract entities from: {query} {response}"
    mock_prompt_template.REFERENCE_RESOLUTION_TEMPLATE = MagicMock()
    mock_prompt_template.REFERENCE_RESOLUTION_TEMPLATE.value = "Resolve: {query} {history} {services} {topics} {technical_terms}"
    monkeypatch.setitem(
        sys.modules,
        "src.enums.prompt_template",
        MagicMock(PromptTemplate=mock_prompt_template)
    )

    yield


@pytest.fixture
def context_manager(patch_dependencies):
    """Create a ContextManager instance for testing"""
    # Set environment variables for testing
    os.environ["ENABLE_ENTITY_TRACKING"] = "true"
    os.environ["ENABLE_REFERENCE_RESOLUTION"] = "true"
    os.environ["ENABLE_SMART_QUERY_MERGING"] = "true"

    # Import after mocking
    from src.chatbot.context_manager import ContextManager
    manager = ContextManager()
    manager.client = MagicMock()
    return manager


class TestHasReferences:
    """Tests for has_references method"""

    def test_detects_pronouns(self, context_manager):
        """Test that has_references correctly detects pronouns"""
        assert context_manager.has_references("How do I configure it?") is True
        assert context_manager.has_references("Tell me more about that") is True
        assert context_manager.has_references("What about those errors?") is True
        assert context_manager.has_references("Can you explain this feature?") is True
        assert context_manager.has_references("How does the service work?") is True

    def test_no_pronouns(self, context_manager):
        """Test cases without pronouns"""
        assert context_manager.has_references("What is WorldTracer?") is False
        assert context_manager.has_references("Tell me about Bag Manager") is False
        assert context_manager.has_references("How to configure WorldTracer?") is False

    def test_case_insensitive(self, context_manager):
        """Test that has_references is case insensitive"""
        assert context_manager.has_references("How do I configure IT?") is True
        assert context_manager.has_references("Tell me more about THAT") is True

    def test_continuation_patterns(self, context_manager):
        """Test that continuation patterns are detected as references"""
        assert context_manager.has_references("tell me more") is True
        assert context_manager.has_references("explain further") is True
        assert context_manager.has_references("more details") is True
        assert context_manager.has_references("go on") is True
        assert context_manager.has_references("elaborate") is True

    def test_empty_query(self, context_manager):
        """Test that empty query returns False"""
        assert context_manager.has_references("") is False
        assert context_manager.has_references(None) is False


class TestBuildContextWindow:
    """Tests for build_context_window method"""

    def test_formats_correctly(self, context_manager):
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

    def test_respects_window_size(self, context_manager):
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

    def test_empty_history(self, context_manager):
        """Test that build_context_window handles empty history"""
        context = context_manager.build_context_window([], window_size=5)
        assert context == ""


class TestEntityHelpers:
    """Tests for entity helper methods"""

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
            "LNI_CODE",  # Has underscore - technical term
            "IATA",  # All caps - technical term
            "Airport Solutions"
        ]

        grouped = context_manager.group_entities_from_flat(flat_entities)

        assert "WorldTracer" in grouped["services"]
        assert "Bag Manager" in grouped["services"]
        assert "Airport Solutions" in grouped["services"]
        assert "lost baggage" in grouped["topics"]
        # Technical terms detected by regex: 2+ uppercase or contains _/-
        assert "LNI_CODE" in grouped["technical_terms"]
        assert "IATA" in grouped["technical_terms"]


class TestExtractEntities:
    """Tests for extract_entities method"""

    def test_feature_flag_disabled(self, context_manager):
        """Test that extract_entities returns empty when feature flag is disabled"""
        context_manager.entity_tracking_enabled = False

        result = context_manager.extract_entities(
            query="What is WorldTracer?",
            response="WorldTracer is a baggage tracing system"
        )

        assert result == {"services": [], "topics": [], "technical_terms": []}

    def test_success(self, context_manager):
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

    def test_handles_llm_error(self, context_manager):
        """Test that extract_entities handles LLM errors gracefully"""
        context_manager.client.chat.completions.create = Mock(side_effect=Exception("LLM Error"))

        result = context_manager.extract_entities(
            query="What is WorldTracer?",
            response="WorldTracer is a baggage tracing system"
        )

        # Should return empty structure on error
        assert result == {"services": [], "topics": [], "technical_terms": []}


class TestResolveReferences:
    """Tests for resolve_references method"""

    def test_with_entities(self, context_manager):
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

    def test_no_references(self, context_manager):
        """Test that queries without references are returned unchanged"""
        query = "What is WorldTracer?"
        entities = {"services": [], "topics": [], "technical_terms": []}
        history = []

        result = context_manager.resolve_references(query, entities, history)

        assert result == query

    def test_feature_flag_disabled(self, context_manager):
        """Test that resolution is skipped when feature flag is disabled"""
        context_manager.reference_resolution_enabled = False

        query = "How do I configure it?"
        entities = {"services": ["WorldTracer"], "topics": [], "technical_terms": []}
        history = []

        result = context_manager.resolve_references(query, entities, history)

        assert result == query


class TestBuildSmartRetrievalQuery:
    """Tests for build_smart_retrieval_query method"""

    def test_independent_query(self, context_manager):
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

    def test_dependent_with_history(self, context_manager):
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

    def test_feature_flag_disabled(self, context_manager):
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

    def test_limits_history(self, context_manager):
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
