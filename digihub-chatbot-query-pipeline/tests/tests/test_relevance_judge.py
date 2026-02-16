"""
Unit tests for RelevanceJudge

Tests LLM-based relevance judgment of retrieved chunks to user queries.
"""

import pytest
import json
from unittest.mock import MagicMock, Mock, patch
import sys


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
    mock_prompt_template.RELEVANCE_JUDGE_TEMPLATE_BULK = MagicMock()
    mock_prompt_template.RELEVANCE_JUDGE_TEMPLATE_BULK.value = "Judge relevance: {prompt} Chunks: {chunks_json}"
    monkeypatch.setitem(
        sys.modules,
        "src.enums.prompt_template",
        MagicMock(PromptTemplate=mock_prompt_template)
    )

    yield


@pytest.fixture
def relevance_judge(patch_dependencies):
    """Create a fresh RelevanceJudge instance for each test"""
    from src.chatbot.relevance_judge import RelevanceJudge
    judge = RelevanceJudge()
    # Reset the mock for each test
    judge.client.reset_mock()
    return judge


class TestRelevanceJudge:
    """Test suite for RelevanceJudge class"""

    def test_init(self, relevance_judge):
        """Test RelevanceJudge initialization"""
        assert relevance_judge.model is not None
        assert relevance_judge.max_tokens == 16384
        assert relevance_judge.client is not None

    def test_init_with_custom_params(self, patch_dependencies):
        """Test RelevanceJudge initialization with custom parameters"""
        from src.chatbot.relevance_judge import RelevanceJudge
        judge = RelevanceJudge(model="gpt-4-turbo", max_tokens=8192)

        assert judge.model == "gpt-4-turbo"
        assert judge.max_tokens == 8192


class TestJudgeChunksRelevance:
    """Test suite for judge_chunks_relevance method"""

    def test_judge_empty_chunks(self, relevance_judge):
        """Test judging empty chunk list"""
        result = relevance_judge.judge_chunks_relevance("What is DigiHub?", [])
        assert result == []

    def test_judge_single_relevant_chunk(self, relevance_judge):
        """Test judging a single relevant chunk"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "relevant_chunks": [
                {"content": "DigiHub is SITA's portal", "serviceNameid": 0}
            ]
        })
        relevance_judge.client.chat.completions.create.return_value = mock_response

        chunks = [
            {"content": "DigiHub is SITA's customer portal", "serviceNameid": 0}
        ]

        result = relevance_judge.judge_chunks_relevance("What is DigiHub?", chunks)

        assert 0 in result
        assert len(result) == 1

    def test_judge_multiple_relevant_chunks(self, relevance_judge):
        """Test judging multiple relevant chunks"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "relevant_chunks": [
                {"content": "DigiHub is SITA's portal", "serviceNameid": 0},
                {"content": "DigiHub user guide", "serviceNameid": 440}
            ]
        })
        relevance_judge.client.chat.completions.create.return_value = mock_response

        chunks = [
            {"content": "DigiHub is SITA's customer portal", "serviceNameid": 0},
            {"content": "DigiHub user guide explains features", "serviceNameid": 440},
            {"content": "WorldTracer baggage tracing", "serviceNameid": 240}
        ]

        result = relevance_judge.judge_chunks_relevance("What is DigiHub?", chunks)

        assert 0 in result
        assert 440 in result
        assert 240 not in result

    def test_judge_no_relevant_chunks(self, relevance_judge):
        """Test judging when no chunks are relevant"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "relevant_chunks": []
        })
        relevance_judge.client.chat.completions.create.return_value = mock_response

        chunks = [
            {"content": "WorldTracer baggage tracing", "serviceNameid": 240},
            {"content": "Billing invoice management", "serviceNameid": 400}
        ]

        result = relevance_judge.judge_chunks_relevance("What is the weather?", chunks)

        assert result == []

    def test_judge_error_handling(self, relevance_judge):
        """Test error handling when LLM call fails"""
        relevance_judge.client.chat.completions.create.side_effect = Exception("API Error")

        chunks = [
            {"content": "DigiHub content", "serviceNameid": 0}
        ]

        result = relevance_judge.judge_chunks_relevance("What is DigiHub?", chunks)

        assert result == []

    def test_judge_malformed_response(self, relevance_judge):
        """Test handling of malformed LLM response"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Invalid JSON response"
        relevance_judge.client.chat.completions.create.return_value = mock_response

        chunks = [
            {"content": "DigiHub content", "serviceNameid": 0}
        ]

        result = relevance_judge.judge_chunks_relevance("What is DigiHub?", chunks)

        # Should return empty list on JSON parse error
        assert result == []

    def test_judge_missing_relevant_chunks_key(self, relevance_judge):
        """Test handling when response lacks relevant_chunks key"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "other_key": "some value"
        })
        relevance_judge.client.chat.completions.create.return_value = mock_response

        chunks = [
            {"content": "DigiHub content", "serviceNameid": 0}
        ]

        result = relevance_judge.judge_chunks_relevance("What is DigiHub?", chunks)

        # Should return empty list when relevant_chunks key is missing
        assert result == []

    def test_judge_chunks_without_serviceNameid(self, relevance_judge):
        """Test handling chunks that lack serviceNameid in LLM response"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "relevant_chunks": [
                {"content": "DigiHub content"},  # No serviceNameid
                {"content": "WorldTracer content", "serviceNameid": 240}
            ]
        })
        relevance_judge.client.chat.completions.create.return_value = mock_response

        chunks = [
            {"content": "DigiHub content", "serviceNameid": 0},
            {"content": "WorldTracer content", "serviceNameid": 240}
        ]

        result = relevance_judge.judge_chunks_relevance("Query", chunks)

        # Should only include chunks with serviceNameid in LLM response
        assert 240 in result
        assert len(result) == 1

    def test_judge_preserves_all_service_line_ids(self, relevance_judge):
        """Test that all relevant service line IDs are preserved"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "relevant_chunks": [
                {"content": "Chunk 1", "serviceNameid": 0},
                {"content": "Chunk 2", "serviceNameid": 240},
                {"content": "Chunk 3", "serviceNameid": 400},
                {"content": "Chunk 4", "serviceNameid": 440}
            ]
        })
        relevance_judge.client.chat.completions.create.return_value = mock_response

        chunks = [
            {"content": "Chunk 1", "serviceNameid": 0},
            {"content": "Chunk 2", "serviceNameid": 240},
            {"content": "Chunk 3", "serviceNameid": 400},
            {"content": "Chunk 4", "serviceNameid": 440}
        ]

        result = relevance_judge.judge_chunks_relevance("Complex query", chunks)

        assert len(result) == 4
        assert 0 in result
        assert 240 in result
        assert 400 in result
        assert 440 in result

    def test_judge_duplicate_service_lines(self, relevance_judge):
        """Test handling of duplicate service line IDs in response"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "relevant_chunks": [
                {"content": "Chunk 1", "serviceNameid": 0},
                {"content": "Chunk 2", "serviceNameid": 0},  # Duplicate
                {"content": "Chunk 3", "serviceNameid": 240}
            ]
        })
        relevance_judge.client.chat.completions.create.return_value = mock_response

        chunks = [
            {"content": "Chunk 1", "serviceNameid": 0},
            {"content": "Chunk 2", "serviceNameid": 0},
            {"content": "Chunk 3", "serviceNameid": 240}
        ]

        result = relevance_judge.judge_chunks_relevance("Query", chunks)

        # Should return all IDs (including duplicates)
        assert 0 in result
        assert 240 in result


class TestPromptFormatting:
    """Test suite for prompt formatting"""

    def test_chunks_formatted_as_json(self, relevance_judge):
        """Test that chunks are formatted as JSON for the prompt"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "relevant_chunks": []
        })
        relevance_judge.client.chat.completions.create.return_value = mock_response

        chunks = [
            {"content": "Test content", "serviceNameid": 0, "extra_field": "ignored"}
        ]

        relevance_judge.judge_chunks_relevance("Query", chunks)

        # Verify the create method was called
        relevance_judge.client.chat.completions.create.assert_called_once()


class TestEdgeCases:
    """Test edge cases"""

    def test_very_long_content(self, relevance_judge):
        """Test handling of very long content in chunks"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "relevant_chunks": [
                {"content": "Long content", "serviceNameid": 0}
            ]
        })
        relevance_judge.client.chat.completions.create.return_value = mock_response

        long_content = "A" * 10000  # Very long content
        chunks = [
            {"content": long_content, "serviceNameid": 0}
        ]

        result = relevance_judge.judge_chunks_relevance("Query", chunks)

        assert 0 in result

    def test_special_characters_in_content(self, relevance_judge):
        """Test handling of special characters in content"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "relevant_chunks": [
                {"content": "Special chars", "serviceNameid": 0}
            ]
        })
        relevance_judge.client.chat.completions.create.return_value = mock_response

        chunks = [
            {"content": "Content with 'quotes' and \"double quotes\" and\nnewlines", "serviceNameid": 0}
        ]

        result = relevance_judge.judge_chunks_relevance("Query", chunks)

        assert 0 in result

    def test_unicode_content(self, relevance_judge):
        """Test handling of unicode content"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "relevant_chunks": [
                {"content": "Unicode content", "serviceNameid": 0}
            ]
        })
        relevance_judge.client.chat.completions.create.return_value = mock_response

        chunks = [
            {"content": "Contenu français avec des accents: é à ü ñ 中文", "serviceNameid": 0}
        ]

        result = relevance_judge.judge_chunks_relevance("Query", chunks)

        assert 0 in result

    def test_null_serviceNameid(self, relevance_judge):
        """Test handling when serviceNameid is None"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "relevant_chunks": [
                {"content": "Content", "serviceNameid": None}
            ]
        })
        relevance_judge.client.chat.completions.create.return_value = mock_response

        chunks = [
            {"content": "Content", "serviceNameid": None}
        ]

        result = relevance_judge.judge_chunks_relevance("Query", chunks)

        # None should be included as it has serviceNameid key
        assert None in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
