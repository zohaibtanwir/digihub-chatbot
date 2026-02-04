from azure.cosmos import CosmosClient
from src.utils.config import AZURE_OPENAI_API_KEY, COSMOSDB_KEY, COSMOSDB_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, \
    AZURE_OPENAI_API_BASE, DIGIHUB_DBNAME
from src.utils.logger import logger


class CosmosDBClientSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CosmosDBClientSingleton, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.client = CosmosClient(
            COSMOSDB_ENDPOINT,
            credential=COSMOSDB_KEY,
            connection_verify=False
        )
        self.database = self.client.get_database_client(DIGIHUB_DBNAME)
        logger.info("CosmosDB client initialized successfully.")

    def get_database(self):
        return self.database
