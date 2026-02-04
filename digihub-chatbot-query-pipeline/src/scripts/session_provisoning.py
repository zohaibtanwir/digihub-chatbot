

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from azure.cosmos import CosmosClient, PartitionKey

from src.utils.config import SESSION_CONTAINER_NAME, COSMOSDB_KEY, COSMOSDB_ENDPOINT


def session_container(db_name: str):
    try:

        # Initialize the Cosmos client
        client = CosmosClient(COSMOSDB_ENDPOINT, COSMOSDB_KEY)

        # Create a database
        database = client.create_database_if_not_exists(id=db_name)

        # Create a container (collection)
        container_name = SESSION_CONTAINER_NAME
        container = database.create_container_if_not_exists(
            id = container_name,
            partition_key = PartitionKey(path="/sessionName"),
        )

    except Exception as e:
        raise Exception(f"[SessionCreationFailure] Failed To Setup Session Container")

session_container("DigiHubChatBot")