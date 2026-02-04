import csv
import io
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from azure.storage.blob import BlobServiceClient
from src.utils.config import (
    cosmos_client,
    COSMOSDB_NAME,
    COSMOSDB_VECTOR_INDEX,
    COSMOSDB_SHAREPOINT_DATA_CONTAINER_NAME,
    COSMOSDB_SERVICE_NAME_MAPPING_CONTAINER_NAME,
    AZURE_STORAGE_BLOB_CONNECTION_STRING,
    COSMOSDB_SERVICE_NAME_MAPPING_CONTAINER_NAME,
    COSMOSDB_DEBOUNCER_CONTAINER_NAME,
    SESSION_CONTAINER_NAME
)
from src.utils.logger import logger
from src.utils.cosmos_initialize import CosmosDBInitializers

class CosmosDBInitializer:
    def __init__(self, client=cosmos_client, db_name=COSMOSDB_NAME, container_name=COSMOSDB_VECTOR_INDEX):
        self.client = client
        self.db_name = db_name
        self.container_name = container_name

    def initialize_cosmos(self):
        try:
            # Create or get database
            database = self.client.create_database_if_not_exists(id=self.db_name)
            logger.info(f"Using or created database: {self.db_name}")

            CosmosDBInitializers.initialize_cosmos(self)
            # Create SharePoint data container
            database.create_container_if_not_exists(
                    id=COSMOSDB_SHAREPOINT_DATA_CONTAINER_NAME,
                    partition_key=PartitionKey(path='/foldername')
                )
            logger.info(f"Container '{COSMOSDB_SHAREPOINT_DATA_CONTAINER_NAME}' is ready.")


            # Create ServiceNameMapping container

            service_mapping_container = database.create_container_if_not_exists(
                    id=COSMOSDB_SERVICE_NAME_MAPPING_CONTAINER_NAME,
                    partition_key=PartitionKey(path='/name')
                )
            logger.info("Container 'ServiceNameMapping' is ready.")


            # Create debouncer container
            debouncer_container = database.create_container_if_not_exists(
                    id=COSMOSDB_DEBOUNCER_CONTAINER_NAME,
                    partition_key=PartitionKey(path='/drive_id')
                )
            logger.info("Container 'ServiceNameMapping' is ready.")
               
            # Create Session container

            session_container = database.create_container_if_not_exists(
                    id=SESSION_CONTAINER_NAME,
                    partition_key=PartitionKey(path='/sessionName')
                )
            logger.info("Container 'ServiceNameMapping' is ready.")

            # Read CSV from Blob and insert into ServiceNameMapping container
            try:
                blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_BLOB_CONNECTION_STRING)
                blob_client = blob_service_client.get_blob_client(container="servicemapping", blob="drive_service_mapping.csv")

                stream = io.StringIO(blob_client.download_blob().content_as_text())
                reader = csv.DictReader(stream)  # Uses first row as headers

                for row in reader:
                    if "id" in row and "name" in row:
                        item = dict(row)  # Include all columns
                        service_mapping_container.upsert_item(item)
                    else:
                        logger.warning(f"Skipping row due to missing 'id' or 'name': {row}")

                logger.info("CSV data uploaded to 'ServiceNameMapping' container successfully.")
            except Exception as blob_ex:
                logger.error(f"Failed to read CSV from Blob or insert into Cosmos DB: {blob_ex}")
                raise

        except Exception as e:
            logger.error(f"Failed to initialize Cosmos DB: {e}")
            raise


if __name__ == "__main__":
    initializer = CosmosDBInitializer()
    initializer.initialize_cosmos()
