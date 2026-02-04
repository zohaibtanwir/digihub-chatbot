
from langchain_openai import AzureOpenAIEmbeddings
from src.utils.config import AZURE_OPENAI_API_KEY, OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_BASE, \
    AZURE_OPENAI_DEPLOYMENT
from src.utils.logger import logger


class AzureEmbeddingService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AzureEmbeddingService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_OPENAI_API_BASE,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT,
            openai_api_version="2024-12-01-preview",
            api_key=AZURE_OPENAI_API_KEY
        )
        logger.info("CosmosDB client initialized successfully.")

    def get_embeddings(self):
        return self.embeddings