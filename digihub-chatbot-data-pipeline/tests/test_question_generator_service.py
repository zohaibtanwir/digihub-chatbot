"""
Tests for QuestionGeneratorService and related services.

Tests the following components:
- QuestionGeneratorService (validate_chunk_quality, generate_questions, process_chunk)
- AzureOpenAIService (singleton pattern, client initialization)
- AzureEmbeddingService (singleton pattern, embedding generation)
"""

import os
import sys
import types
import pytest
from unittest.mock import MagicMock, patch

# --- Step 1: Create comprehensive mock config module ---
fake_config = types.ModuleType("config")
fake_config.AZURE_OPENAI_API_KEY = "fake-api-key"
fake_config.AZURE_OPENAI_ENDPOINT = "https://fake-endpoint.openai.azure.com"
fake_config.OPENAI_API_VERSION = "2024-02-01"
fake_config.AZURE_OPENAI_DEPLOYMENT = "text-embedding-ada-002"
fake_config.OPENAI_DEPLOYMENT_NAME = "gpt-4o-mini"
fake_config.ENABLE_QUESTION_GENERATION = True
fake_config.QUESTIONS_PER_CHUNK = 5

# --- Step 2: Create mock logger module ---
fake_logger_module = types.ModuleType("logger")
fake_logger_module.logger = MagicMock()

# --- Step 3: Pre-mock Azure SDK modules ---
mock_azure_openai = MagicMock()
mock_langchain_embeddings = MagicMock()

# Create mock AzureOpenAI class
mock_openai_client_instance = MagicMock()
mock_completion = MagicMock()
mock_completion.choices = [MagicMock()]
mock_completion.choices[0].message.content = '["What is X?", "How does Y work?", "What are the benefits?"]'
mock_openai_client_instance.chat.completions.create.return_value = mock_completion
mock_azure_openai.return_value = mock_openai_client_instance

# Create mock AzureOpenAIEmbeddings class
mock_embeddings_instance = MagicMock()
mock_embeddings_instance.embed_query.return_value = [0.1] * 1536
mock_embeddings_instance.embed_documents.return_value = [[0.1] * 1536]
mock_langchain_embeddings.return_value = mock_embeddings_instance

# --- Step 4: Inject all mocks into sys.modules BEFORE importing services ---
sys.modules["src.utils.config"] = fake_config
sys.modules["src.utils.logger"] = fake_logger_module

# Mock the openai module
mock_openai_module = types.ModuleType("openai")
mock_openai_module.AzureOpenAI = mock_azure_openai
sys.modules["openai"] = mock_openai_module

# Mock langchain_openai module
mock_langchain_openai = types.ModuleType("langchain_openai")
mock_langchain_openai.AzureOpenAIEmbeddings = mock_langchain_embeddings
sys.modules["langchain_openai"] = mock_langchain_openai

# Now we can safely import the services
# Reset any existing singleton instances
try:
    from src.services import azure_openai_service, embedding_service, question_generator_service
    azure_openai_service.AzureOpenAIService._instance = None
    embedding_service.AzureEmbeddingService._instance = None
except ImportError:
    pass

from src.services.azure_openai_service import AzureOpenAIService
from src.services.embedding_service import AzureEmbeddingService
from src.services.question_generator_service import QuestionGeneratorService


class TestValidateChunkQuality:
    """Tests for validate_chunk_quality method."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset singletons and create fresh service instance."""
        AzureOpenAIService._instance = None
        AzureEmbeddingService._instance = None
        self.service = QuestionGeneratorService()
        yield

    def test_empty_content_is_invalid(self):
        """Empty content should be invalid."""
        is_valid, reason = self.service.validate_chunk_quality("")
        assert is_valid is False
        assert "Empty content" in reason

    def test_whitespace_only_is_invalid(self):
        """Whitespace-only content should be invalid."""
        is_valid, reason = self.service.validate_chunk_quality("   \n\t   ")
        assert is_valid is False
        assert "Empty content" in reason

    def test_too_short_content_is_invalid(self):
        """Content shorter than MIN_CONTENT_LENGTH should be invalid."""
        short_content = "This is too short."
        is_valid, reason = self.service.validate_chunk_quality(short_content)
        assert is_valid is False
        assert "too short" in reason

    def test_only_headings_is_invalid(self):
        """Content with only headings should be invalid."""
        # Make content long enough to pass length check but only contains headings
        heading_only = """# Heading 1 - Introduction to the Document
## Heading 2 - Overview of Key Topics
### Heading 3 - Detailed Subtopics Here
#### Heading 4 - More Detailed Information
##### Heading 5 - Additional Context
"""
        is_valid, reason = self.service.validate_chunk_quality(heading_only)
        assert is_valid is False
        assert "only headings" in reason

    def test_placeholder_content_is_invalid(self):
        """Placeholder content like 'lorem ipsum' should be invalid."""
        # Make content long enough but contains placeholder text
        placeholder = """Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
"""
        is_valid, reason = self.service.validate_chunk_quality(placeholder)
        assert is_valid is False
        assert "placeholder" in reason

    def test_todo_content_is_invalid(self):
        """Content with TODO markers should be invalid if short."""
        todo_content = "TODO: Add documentation here\nTBD later"
        is_valid, reason = self.service.validate_chunk_quality(todo_content)
        assert is_valid is False

    def test_valid_content_passes(self):
        """Valid content with sufficient length and substance should pass."""
        valid_content = """# Document Title

This is a comprehensive document that explains the authentication process.
The authentication system uses OAuth 2.0 for secure access control.
Users must authenticate before accessing protected resources.
Multiple authentication providers are supported including Azure AD.
The system supports both single sign-on and multi-factor authentication.
"""
        is_valid, reason = self.service.validate_chunk_quality(valid_content)
        assert is_valid is True
        assert reason == "valid"

    def test_heading_parameter_optional(self):
        """Heading parameter should be optional."""
        valid_content = """This is a comprehensive document with lots of content for testing purposes.
It has multiple lines and paragraphs to meet the length requirements of validation.
The content is substantive and provides real information that would be useful.
Additional lines ensure we have enough content to pass minimum length checks here.
"""
        # Should work without heading parameter
        is_valid, reason = self.service.validate_chunk_quality(valid_content)
        assert is_valid is True


class TestGenerateQuestions:
    """Tests for generate_questions method."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset singletons and create fresh service instance."""
        AzureOpenAIService._instance = None
        AzureEmbeddingService._instance = None
        # Reset mock for each test
        mock_completion.choices[0].message.content = '["What is X?", "How does Y work?", "What are the benefits?"]'
        mock_openai_client_instance.chat.completions.create.return_value = mock_completion
        mock_openai_client_instance.chat.completions.create.side_effect = None
        self.service = QuestionGeneratorService()
        yield

    def test_returns_list_of_strings(self):
        """generate_questions should return a list of question strings."""
        content = """# Authentication Guide

This guide explains how to authenticate using OAuth 2.0 protocol.
The authentication flow involves obtaining access tokens from the server.
Users need to configure their client credentials properly for access.
"""
        questions = self.service.generate_questions(content, "Authentication Guide")

        assert isinstance(questions, list)
        assert len(questions) > 0
        assert all(isinstance(q, str) for q in questions)

    def test_respects_num_questions_limit(self):
        """Should not return more than num_questions."""
        mock_completion.choices[0].message.content = '["Q1?", "Q2?", "Q3?", "Q4?", "Q5?", "Q6?"]'

        content = "Test content " * 50  # Long enough content
        questions = self.service.generate_questions(content, "Test", num_questions=3)

        assert len(questions) <= 3

    def test_handles_json_parse_error_gracefully(self):
        """JSON parsing errors should result in fallback questions."""
        mock_completion.choices[0].message.content = "This is not valid JSON at all"

        content = """# Valid Document

This is valid content that will trigger a JSON error response.
The system should gracefully handle this and return fallback questions.
Testing error handling is critical for robust production systems.
"""
        questions = self.service.generate_questions(content, "Test Heading")

        assert isinstance(questions, list)
        assert len(questions) > 0
        # Fallback should include heading-based questions
        assert any("Test Heading" in q for q in questions)

    def test_handles_api_error_gracefully(self):
        """API errors should result in fallback questions."""
        mock_openai_client_instance.chat.completions.create.side_effect = Exception("API Error")

        content = """# Error Test Document

This content tests API error handling behavior in the question generator.
The system should handle errors gracefully and return fallback questions.
Robust error handling ensures system reliability in production.
"""
        questions = self.service.generate_questions(content, "Error Test")

        assert isinstance(questions, list)
        assert len(questions) > 0

    def test_cleans_markdown_code_blocks_json(self):
        """Should clean markdown code blocks from JSON response."""
        mock_completion.choices[0].message.content = '```json\n["Question 1?", "Question 2?"]\n```'

        content = "Test content " * 50
        questions = self.service.generate_questions(content, "Test")

        assert isinstance(questions, list)
        assert len(questions) == 2
        assert questions[0] == "Question 1?"

    def test_cleans_plain_code_blocks(self):
        """Should clean plain code blocks without json specifier."""
        mock_completion.choices[0].message.content = '```\n["Q1?", "Q2?", "Q3?"]\n```'

        content = "Test content " * 50
        questions = self.service.generate_questions(content, "Test")

        assert isinstance(questions, list)
        assert len(questions) == 3


class TestGenerateQuestionsEmbedding:
    """Tests for generate_questions_embedding method."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset singletons and create fresh service instance."""
        AzureOpenAIService._instance = None
        AzureEmbeddingService._instance = None
        mock_embeddings_instance.embed_query.return_value = [0.1] * 1536
        self.service = QuestionGeneratorService()
        yield

    def test_returns_embedding_vector(self):
        """Should return an embedding vector for questions."""
        questions = ["What is authentication?", "How does OAuth work?"]

        embedding = self.service.generate_questions_embedding(questions)

        assert isinstance(embedding, list)
        assert len(embedding) == 1536
        assert all(isinstance(v, (int, float)) for v in embedding)

    def test_empty_list_returns_empty_embedding(self):
        """Empty question list should return empty embedding."""
        embedding = self.service.generate_questions_embedding([])
        assert embedding == []

    def test_concatenates_questions_for_embedding(self):
        """Questions should be concatenated with newlines for embedding."""
        questions = ["Question 1?", "Question 2?", "Question 3?"]

        self.service.generate_questions_embedding(questions)

        # Verify embed_query was called
        mock_embeddings_instance.embed_query.assert_called()
        call_args = mock_embeddings_instance.embed_query.call_args[0][0]
        # Check all questions are in the concatenated text
        assert "Question 1?" in call_args
        assert "Question 2?" in call_args
        assert "Question 3?" in call_args


class TestProcessChunk:
    """Tests for process_chunk method (main entry point)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset singletons and mocks for each test."""
        AzureOpenAIService._instance = None
        AzureEmbeddingService._instance = None
        mock_completion.choices[0].message.content = '["What is X?", "How does Y work?", "What are the benefits?"]'
        mock_openai_client_instance.chat.completions.create.return_value = mock_completion
        mock_openai_client_instance.chat.completions.create.side_effect = None
        mock_embeddings_instance.embed_query.return_value = [0.1] * 1536
        self.service = QuestionGeneratorService()
        yield

    def test_valid_chunk_complete_result(self):
        """Valid chunk should return complete result with all fields."""
        content = """# Complete Document

This document provides comprehensive information about the topic at hand.
It contains enough detail to be useful for retrieval and search purposes.
The content quality is sufficient for question generation by the AI model.
Multiple paragraphs ensure adequate context for AI processing tasks.
"""
        result = self.service.process_chunk(content, "Complete Document")

        assert result["validChunk"] == "yes"
        assert result["validationReason"] == "valid"
        assert isinstance(result["questions"], list)
        assert len(result["questions"]) > 0
        assert isinstance(result["questionsEmbedding"], list)
        assert len(result["questionsEmbedding"]) == 1536

    def test_invalid_chunk_no_questions(self):
        """Invalid chunk should return 'no' with empty questions and embedding."""
        content = "Too short"

        result = self.service.process_chunk(content, "Short")

        assert result["validChunk"] == "no"
        assert "too short" in result["validationReason"]
        assert result["questions"] == []
        assert result["questionsEmbedding"] == []

    def test_valid_chunk_empty_questions_empty_embedding(self):
        """When questions generation returns empty, embedding should be empty."""
        mock_completion.choices[0].message.content = '[]'

        content = """# Valid But No Questions

This content is valid but the LLM returns no questions for it.
The system should handle this case and return empty embedding.
Empty questions should result in empty questionsEmbedding field.
"""
        result = self.service.process_chunk(content, "Test")

        assert result["validChunk"] == "yes"
        assert result["questions"] == []
        assert result["questionsEmbedding"] == []

    def test_result_structure(self):
        """Result should have all required keys."""
        content = """# Structure Test

This tests that the result dictionary has all required keys.
The structure should be consistent for both valid and invalid chunks.
All fields must be present for proper CosmosDB document creation.
"""
        result = self.service.process_chunk(content, "Test")

        required_keys = ["validChunk", "validationReason", "questions", "questionsEmbedding"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"


class TestFallbackQuestions:
    """Tests for fallback question generation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset singletons for each test."""
        AzureOpenAIService._instance = None
        AzureEmbeddingService._instance = None
        self.service = QuestionGeneratorService()
        yield

    def test_fallback_includes_heading_questions(self):
        """Fallback questions should include heading-based questions."""
        questions = self.service._generate_fallback_questions("Authentication", "content")

        assert any("Authentication" in q for q in questions)
        assert len(questions) >= 3

    def test_fallback_works_without_heading(self):
        """Fallback questions should work without heading."""
        questions = self.service._generate_fallback_questions("", "Some content here")

        assert isinstance(questions, list)
        assert len(questions) >= 2

    def test_fallback_ignores_untitled_heading(self):
        """Fallback should not use 'Untitled' as heading."""
        questions = self.service._generate_fallback_questions("Untitled", "Some content here")

        # Should not generate "What is Untitled?" questions
        assert not any("What is Untitled" in q for q in questions)

    def test_fallback_max_five_questions(self):
        """Fallback questions should be limited to 5."""
        questions = self.service._generate_fallback_questions(
            "Test Heading With Very Long Name",
            "First line of content\nSecond line of content"
        )

        assert len(questions) <= 5


class TestAzureOpenAIServiceSingleton:
    """Tests for AzureOpenAIService singleton pattern."""

    def test_singleton_same_instance(self):
        """Multiple instantiations should return same instance."""
        AzureOpenAIService._instance = None  # Reset

        service1 = AzureOpenAIService()
        service2 = AzureOpenAIService()

        assert service1 is service2

    def test_get_client_returns_client(self):
        """get_client should return the OpenAI client."""
        AzureOpenAIService._instance = None  # Reset

        service = AzureOpenAIService()
        client = service.get_client()

        assert client is not None


class TestAzureEmbeddingServiceSingleton:
    """Tests for AzureEmbeddingService singleton pattern."""

    def test_singleton_same_instance(self):
        """Multiple instantiations should return same instance."""
        AzureEmbeddingService._instance = None  # Reset

        service1 = AzureEmbeddingService()
        service2 = AzureEmbeddingService()

        assert service1 is service2

    def test_get_embeddings_returns_embeddings(self):
        """get_embeddings should return the embeddings instance."""
        AzureEmbeddingService._instance = None  # Reset

        service = AzureEmbeddingService()
        embeddings = service.get_embeddings()

        assert embeddings is not None

    def test_embed_text_calls_embed_query(self):
        """embed_text should call embed_query on embeddings."""
        AzureEmbeddingService._instance = None  # Reset
        mock_embeddings_instance.embed_query.return_value = [0.5] * 1536

        service = AzureEmbeddingService()
        result = service.embed_text("test text")

        mock_embeddings_instance.embed_query.assert_called_with("test text")
        assert len(result) == 1536

    def test_embed_texts_calls_embed_documents(self):
        """embed_texts should call embed_documents on embeddings."""
        AzureEmbeddingService._instance = None  # Reset
        mock_embeddings_instance.embed_documents.return_value = [[0.1] * 1536, [0.2] * 1536]

        service = AzureEmbeddingService()
        result = service.embed_texts(["text1", "text2"])

        mock_embeddings_instance.embed_documents.assert_called_with(["text1", "text2"])
        assert len(result) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
