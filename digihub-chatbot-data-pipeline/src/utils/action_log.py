from azure.cosmos import CosmosClient, PartitionKey, exceptions
import uuid
import time
import os
from src.utils.logger import logger
from azure.core.exceptions import ServiceResponseError
import traceback
class CosmosDBSetupError(Exception):
    """Custom exception for handling Cosmos DB setup errors."""
    pass
 
def logging_container():
    from src.utils.config import COSMOS_LOGGING_CONTAINER_NAME,cosmos_client,COSMOSDB_NAME
    """
    Initializes a logging container in Azure Cosmos DB.
 
    - Checks if the container already exists.
    - Creates a new container if it does not exist.
 
    Args:
        cosmos_endpoint (str): The Cosmos DB endpoint.
        cosmos_key (str): The authentication key.
        db_name (str): The database name.
 
    Returns:
        container_client: The Cosmos DB container client.
    Raises:
        CosmosDBSetupError: If the container creation fails.
    """
    logger.info("Initializing logging container setup...")
    try:
        database = cosmos_client.get_database_client(COSMOSDB_NAME)
        logger.debug(f"Connected to database: {COSMOSDB_NAME}")
        container_client = database.get_container_client(COSMOS_LOGGING_CONTAINER_NAME)
        logger.debug(f"Attempting to read container: {COSMOS_LOGGING_CONTAINER_NAME}")

        try:
            container_client.read()
            logger.info(f"Logging container '{COSMOS_LOGGING_CONTAINER_NAME}' already exists.")
        except exceptions.CosmosResourceNotFoundError:
            logger.warning(f"Container '{COSMOS_LOGGING_CONTAINER_NAME}' not found. Creating new container...")
            container_client = database.create_container(
                id=COSMOS_LOGGING_CONTAINER_NAME,
                partition_key=PartitionKey(path="/serviceName")
            )
            logger.info(f"Container '{COSMOS_LOGGING_CONTAINER_NAME}' created successfully.")
 
        return container_client
 
    except Exception as e:
        logger.error(f"Failed to setup logging container: {e}", exc_info=True)
        raise CosmosDBSetupError(f"Failed to setup logging container: {e}")    
class AuditLogException(Exception):
    """Custom exception for audit logging errors."""
    pass
def cosmos_index_logger(log_container, action: str, folder_name: str, file_name: str, source_path: str, duration):
    """
    Logs an action to Cosmos DB about document index processing 
    into the DigiHub-Chatbot-KnowledgeBaseLogs container.
 
    Args:
        log_container: The Cosmos DB container client.
        action (str): The action performed (e.g., 'index', 'delete').
        folder_name (str): The folder/service name.
        file_name (str): The name of the file being logged.
        knowledge_base_id (str): Identifier for the knowledge base.
        source_path (str): Path of the source document.
 
    Raises:
        AuditLogException: If logging the action fails.
    """

    try:
        log_item = {
            "id": str(uuid.uuid4()),
            "serviceName": folder_name,
            "fileName": file_name,
            "action": action,
            "timestamp": int(time.time()),
            "sourcePath": source_path,
            "duration": duration
        }
        try:
            log_container.upsert_item(log_item)
            logger.info(f"Successfully logged action '{action}'")
        except ServiceResponseError as e:
            logger.error(f"Data base connection error:{e} {traceback.format_exc()}")


    except Exception as e:
        logger.error(f"Could not store audit log for file '{file_name}': {e}", exc_info=True)
        raise AuditLogException(f"Could not store audit log for file '{file_name}': {e}")