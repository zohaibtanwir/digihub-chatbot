"""
Embedding Service for the Data Pipeline.

Provides singleton access to Azure OpenAI Embeddings for generating
vector embeddings for content and questions.
"""

from langchain_openai import AzureOpenAIEmbeddings
from src.utils.config import (
    AZURE_OPENAI_API_KEY,
    OPENAI_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT
)
from src.utils.logger import logger


class AzureEmbeddingService:
    """
    Singleton service for Azure OpenAI Embeddings.

    Provides embedding generation capabilities for both content
    and question-based embeddings used in RAG retrieval.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AzureEmbeddingService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the Azure OpenAI Embeddings client."""
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT,
            openai_api_version=OPENAI_API_VERSION,
            api_key=AZURE_OPENAI_API_KEY
        )
        logger.info("AzureEmbeddingService initialized successfully.")

    def get_embeddings(self) -> AzureOpenAIEmbeddings:
        """
        Get the Azure OpenAI Embeddings instance.

        Returns:
            AzureOpenAIEmbeddings: The initialized embeddings client.
        """
        return self.embeddings

    def embed_text(self, text: str) -> list:
        """
        Generate embedding vector for a given text.

        Args:
            text: The text to embed.

        Returns:
            list: The embedding vector.
        """
        return self.embeddings.embed_query(text)

    def embed_texts(self, texts: list) -> list:
        """
        Generate embedding vectors for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            list: List of embedding vectors.
        """
        return self.embeddings.embed_documents(texts)
