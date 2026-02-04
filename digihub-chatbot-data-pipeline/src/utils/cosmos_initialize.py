from azure.cosmos import CosmosClient, PartitionKey, exceptions
from src.utils.config import cosmos_client, COSMOSDB_NAME, COSMOSDB_VECTOR_INDEX
from src.utils.logger import logger

class CosmosDBInitializers:
    def __init__(self, client=cosmos_client, db_name=COSMOSDB_NAME, container_name=COSMOSDB_VECTOR_INDEX):
        self.client = client
        self.db_name = db_name
        self.container_name = container_name

    def initialize_cosmos(self):
        try:
            # Create database if it doesn't exist
            try:
                database = self.client.create_database(self.db_name)
                logger.info(f"Created new Cosmos DB database: {self.db_name}")
            except exceptions.CosmosResourceExistsError:
                database = self.client.get_database_client(self.db_name)
                logger.info(f"Using existing database: {self.db_name}")

            # Create container if it doesn't exist
            vector_embedding_policy = {
                "vectorEmbeddings": [
                    {
                        "path": "/embedding",
                        "dataType": "float32",
                        "distanceFunction": "cosine",
                        "dimensions": 1536
                    },
                ]
            }

            indexing_policy = {
                "indexingMode": "consistent",
                "automatic": True,
                "includedPaths": [
                    {
                        "path": "/*"
                    }
                ],
                "excludedPaths": [
                    {
                        "path": "/_etag/?"
                    },
                    {
                        "path": "/embedding/*"
                    }
                ],
                "vectorIndexes": [
                    {
                        "path": "/embedding",
                        "type": "diskANN"
                    }
                ]
            }

            try:
                container = database.create_container(
                    id=self.container_name,
                    partition_key=PartitionKey(path="/partitionKey"),
                    indexing_policy=indexing_policy,
                    vector_embedding_policy=vector_embedding_policy,
                )
                logger.info(f"Created new container: {self.container_name}")

            except exceptions.CosmosResourceExistsError:
                container = database.get_container_client(self.container_name)
                logger.info(f"Using existing container: {self.container_name}")

            return container

        except Exception as e:
            logger.error(f"Failed to initialize Cosmos DB: {e}")
            raise
    def get_cosmos(self):
        try:
            database = self.client.get_database_client(self.db_name)
            container = database.get_container_client(self.container_name)
            logger.info(f"Using existing container: {self.container_name}")

            return container

        except Exception as e:
            logger.error(f"Failed to initialize Cosmos DB: {e}")
            raise

