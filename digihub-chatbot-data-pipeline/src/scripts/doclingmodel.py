from azure.storage.blob import BlobServiceClient
import os
from src.utils.config import AZURE_STORAGE_BLOB_CONNECTION_STRING
from src.utils.logger import logger
# Define the connection string and container name
def download_docling():
    container_name = "docling-model"
    local_path = "./docling"

    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_BLOB_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(container_name)

    os.makedirs(local_path, exist_ok=True)

    for blob in container_client.list_blobs():
        try:
            blob_client = container_client.get_blob_client(blob.name)
            download_file_path = os.path.join(local_path, blob.name)
            os.makedirs(os.path.dirname(download_file_path), exist_ok=True)

            with open(download_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())

            full_path = os.path.abspath(download_file_path)
            logger.info(f"Downloaded {blob.name} to {full_path}")
        except Exception as e:
            logger.error(f"Failed to download {blob.name}: {e}")
