from openai import AzureOpenAI, BadRequestError

from src.utils.config import AZURE_OPENAI_API_KEY, OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT
from src.utils.logger import logger


class AzureOpenAIService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AzureOpenAIService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT
            )
        logger.info("CosmosDB client initialized successfully.")

    def get_client(self):
        return self.client