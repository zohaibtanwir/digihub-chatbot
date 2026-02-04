"""
Azure OpenAI Service for the Data Pipeline.

Provides singleton access to Azure OpenAI client for LLM operations
such as question generation and metadata extraction.
"""

from openai import AzureOpenAI
from src.utils.config import AZURE_OPENAI_API_KEY, OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT
from src.utils.logger import logger


class AzureOpenAIService:
    """
    Singleton service for Azure OpenAI client access.

    Provides a shared AzureOpenAI client instance for LLM operations
    throughout the data pipeline.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AzureOpenAIService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the Azure OpenAI client."""
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        logger.info("AzureOpenAIService initialized successfully.")

    def get_client(self) -> AzureOpenAI:
        """
        Get the Azure OpenAI client instance.

        Returns:
            AzureOpenAI: The initialized Azure OpenAI client.
        """
        return self.client
