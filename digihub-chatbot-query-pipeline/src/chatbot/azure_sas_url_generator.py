from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions, generate_container_sas, ContainerSasPermissions
from datetime import datetime, timedelta
from src.utils.logger import logger
from src.utils.config import AZURE_STORAGE_ACCOUNT_NAME, AZURE_STORAGE_ACCOUNT_URL, AZURE_STORAGE_CONTAINER_NAME, AZURE_STORAGE_CONTAINER_SAS_URL_VALIDITY
from azure.core.pipeline.transport import RequestsTransport 
 
class SASGenerationError(Exception):
    """Custom exception for SAS URL generation errors."""
    def __init__(self, message="Failed to generate SAS URL", original_exception=None):
        super().__init__(message)
        self.original_exception = original_exception

custom_transport = RequestsTransport(connection_timeout=20, read_timeout=60) 
 
credential = DefaultAzureCredential()
blob_service_client = BlobServiceClient(account_url=AZURE_STORAGE_ACCOUNT_URL, credential=credential, transport=custom_transport)
 
 
def generate_container_sas_url():
    try:
        logger.debug("Getting container client...")
        container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)
        VALIDITY_HOURS = float(AZURE_STORAGE_CONTAINER_SAS_URL_VALIDITY)
        logger.debug("Setting SAS start and expiry times...")
        start_time = datetime.utcnow()
        expiry_time = start_time + timedelta(hours=VALIDITY_HOURS)
        epoch_expiry_time = int(expiry_time.timestamp())
        logger.debug("Fetching user delegation key...")
        user_delegation_key = blob_service_client.get_user_delegation_key(start_time, expiry_time)
        logger.debug("User delegation key is fetched")
        logger.debug("Generating container SAS token...")
        sas_token = generate_container_sas(
            account_name=AZURE_STORAGE_ACCOUNT_NAME,
            container_name=AZURE_STORAGE_CONTAINER_NAME,
            user_delegation_key=user_delegation_key,
            permission=ContainerSasPermissions(read=True, list=True),
            expiry=expiry_time
        )
 
        sas_url = f"https://{AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{AZURE_STORAGE_CONTAINER_NAME}?{sas_token}"
 
        logger.info("Generated container-level SAS URL successfully.")
        return {
            "container_sas_url": sas_url,
            "expires_on": epoch_expiry_time,
            "container": AZURE_STORAGE_CONTAINER_NAME
        }
 
    except Exception as e:
        logger.error(f"Error generating container SAS URL: {e}")
        raise SASGenerationError("SAS URL generation failed due to an internal error.", e)