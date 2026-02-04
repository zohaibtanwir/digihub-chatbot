"""
Script to update validChunk field to 'hide' for documents where metadata.filepath
starts with 'Billing/Billing Updates' in Azure Cosmos DB.

Usage:
    python scripts/update_billing_chunks.py
"""

import sys
import os

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from azure.cosmos import CosmosClient, exceptions
from src.utils.config import ConfigManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


def update_billing_chunks():
    """
    Update validChunk field to 'hide' for all documents where
    metadata.filepath starts with 'Billing/Billing Updates'
    """
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = ConfigManager.load_config()

        # Initialize Cosmos DB client
        logger.info("Connecting to Cosmos DB...")
        client = CosmosClient(
            config['COSMOSDB_ENDPOINT'],
            credential=config['COSMOSDB_KEY']
        )

        # Get container reference
        database = client.get_database_client(config['COSMOSDB_DATABASE_NAME'])
        container = database.get_container_client(config['KNOWLEDGE_BASE_CONTAINER'])

        # Query for documents to update
        query = """
        SELECT * FROM c
        WHERE STARTSWITH(c.metadata.filepath, 'Billing/Billing Updates')
        """

        logger.info("Fetching documents to update...")
        logger.info(f"Query: {query}")

        # Fetch all matching documents
        items = list(container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))

        total_count = len(items)
        logger.info(f"Found {total_count} documents to update")

        if total_count == 0:
            logger.info("No documents found matching the criteria.")
            return

        # Ask for confirmation
        print(f"\nFound {total_count} documents to update.")
        print("Sample documents:")
        for i, item in enumerate(items[:5]):
            print(f"  {i+1}. ID: {item['id']}, Filepath: {item['metadata']['filepath']}")

        if total_count > 5:
            print(f"  ... and {total_count - 5} more")

        confirmation = input("\nDo you want to proceed with the update? (yes/no): ")

        if confirmation.lower() not in ['yes', 'y']:
            logger.info("Update cancelled by user.")
            return

        # Update each document
        updated_count = 0
        failed_count = 0

        logger.info("Starting bulk update...")

        for idx, item in enumerate(items, 1):
            try:
                # Update the validChunk field
                item['validChunk'] = 'hide'

                # Upsert the document
                container.upsert_item(item)
                updated_count += 1

                if idx % 10 == 0:  # Log progress every 10 documents
                    logger.info(f"Progress: {idx}/{total_count} documents processed")

            except exceptions.CosmosHttpResponseError as e:
                failed_count += 1
                logger.error(f"Error updating document {item['id']}: {str(e)}")
            except Exception as e:
                failed_count += 1
                logger.error(f"Unexpected error updating document {item['id']}: {str(e)}")

        # Summary
        logger.info("=" * 60)
        logger.info("Update completed!")
        logger.info(f"Total documents found: {total_count}")
        logger.info(f"Successfully updated: {updated_count}")
        logger.info(f"Failed: {failed_count}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error in bulk update: {str(e)}")
        raise


def preview_documents():
    """
    Preview documents that would be updated without making changes
    """
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = ConfigManager.load_config()

        # Initialize Cosmos DB client
        logger.info("Connecting to Cosmos DB...")
        client = CosmosClient(
            config['COSMOSDB_ENDPOINT'],
            credential=config['COSMOSDB_KEY']
        )

        # Get container reference
        database = client.get_database_client(config['COSMOSDB_DATABASE_NAME'])
        container = database.get_container_client(config['KNOWLEDGE_BASE_CONTAINER'])

        # Query for documents to update
        query = """
        SELECT c.id, c.partitionKey, c.serviceName, c.metadata.filepath, c.validChunk
        FROM c
        WHERE STARTSWITH(c.metadata.filepath, 'Billing/Billing Updates')
        """

        logger.info("Fetching documents (preview mode)...")

        # Fetch all matching documents
        items = list(container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))

        total_count = len(items)
        logger.info(f"Found {total_count} documents that would be updated")

        print("\n" + "=" * 80)
        print("PREVIEW: Documents that would be updated")
        print("=" * 80)

        for idx, item in enumerate(items, 1):
            print(f"\n{idx}. ID: {item['id']}")
            print(f"   Partition Key: {item['partitionKey']}")
            print(f"   Service Name: {item['serviceName']}")
            print(f"   Filepath: {item['metadata']['filepath']}")
            print(f"   Current validChunk: {item.get('validChunk', 'N/A')}")
            print(f"   Will be set to: hide")

        print("\n" + "=" * 80)
        print(f"Total: {total_count} documents")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Error in preview: {str(e)}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Update validChunk field for Billing Updates documents"
    )
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Preview documents without making changes'
    )

    args = parser.parse_args()

    if args.preview:
        preview_documents()
    else:
        update_billing_chunks()
